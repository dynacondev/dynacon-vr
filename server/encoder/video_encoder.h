/*
 * WiVRn VR streaming
 * Copyright (C) 2022  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2022  Patrick Nicolas <patricknicolas@laposte.net>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#pragma once

#include "driver/clock_offset.h"
#include "idr_handler.h"
#include "wivrn_packets.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <vulkan/vulkan_raii.hpp>

namespace wivrn
{

struct encoder_settings;
struct vk_bundle;
class wivrn_session;

inline const char * encoder_nvenc = "nvenc";
inline const char * encoder_vaapi = "vaapi";
inline const char * encoder_x264 = "x264";
inline const char * encoder_vulkan = "vulkan";
inline const char * encoder_raw = "raw";

class video_encoder
{
protected:
	struct data
	{
		video_encoder * encoder;
		std::span<uint8_t> span;
		std::shared_ptr<void> mem;
		// true if data should be sent over reliable (TCP) socket
		bool prefer_control = false;
	};

private:
	class sender
	{
		std::mutex mutex;
		std::condition_variable cv;
		std::deque<data> pending;
		std::jthread thread;
		void run();
		sender();

	public:
		void push(data &&);
		static std::shared_ptr<sender> get();
		void wait_idle(video_encoder *);
	};

public:
	const uint8_t stream_idx;
	const uint32_t target_queue;
	const bool need_transfer;
	// Layout the encoder needs y_cbcr to arrive in. Set during construction
	// or init() and read by the compositor on every layer_commit — must not
	// change after the first frame.
	vk::ImageLayout target_layout = vk::ImageLayout::eGeneral;
	static const uint8_t num_slots = 2;
	const double bitrate_multiplier;

	// Shared wedge counter: every backend increments this on a fence
	// timeout in the worker path (commit-thread present polls don't count
	// — those are routine backpressure). Read by the compositor triage.
	static inline std::atomic<uint64_t> wedge_timeouts{0};

private:
	using state_t = std::atomic_unsigned_lock_free::value_type;
	static const state_t idle = 0;
	static const state_t busy = 1;
	static const state_t skip = 2;
	std::array<std::atomic_unsigned_lock_free, num_slots> state = {idle, idle};
	uint8_t present_slot = 0;
	uint8_t encode_slot = 0;

	std::mutex mutex;

	// temporary data
	wivrn_session * cnx;

	// shard to send
	to_headset::video_stream_data_shard shard;

	// Encoded-size triage (logging only): frame_idx -> warped flag for
	// in-flight frames, plus per-window byte/frame totals split by
	// kind. Fed by note_frame_kind on the worker thread, consumed in
	// SendData (any thread); all under mutex.
	std::map<uint64_t, bool> size_kind_by_frame;
	uint64_t size_win_start_ns = 0;
	uint64_t size_w_frames = 0;
	uint64_t size_w_bytes = 0;
	uint64_t size_s_frames = 0;
	uint64_t size_s_bytes = 0;

	void accumulate_size_locked(uint64_t frame_index, size_t bytes, bool end_of_frame);
	void maybe_report_size_locked();

	to_headset::video_stream_data_shard::timing_info_t timing_info;
	clock_offset clock;

	std::ofstream video_dump;

	std::shared_ptr<sender> shared_sender;

protected:
	std::atomic_uint32_t pending_bitrate;
	std::atomic<float> pending_framerate;
	std::unique_ptr<idr_handler> idr;
	const vk::Extent2D extent;

public:
	static std::unique_ptr<video_encoder> create(
	        wivrn::vk_bundle &,
	        const encoder_settings & settings,
	        uint8_t stream_idx);

	video_encoder(vk_bundle &, uint8_t stream_idx, uint32_t target_queue, const encoder_settings & settings, std::unique_ptr<idr_handler>, bool async_send);
	virtual ~video_encoder();

	// Outcome of the commit-thread handoff. Both skip variants advance
	// present_slot/encode_slot in lockstep (the worker consumes skips).
	// skip (encoder state) == drop (triage): the frame is shed, nothing
	// is lost. Only skipped_busy counts as backpressure in triage.
	enum class present_outcome
	{
		presented,
		skipped_idr,
		skipped_busy,
	};
	present_outcome present_image(vk::Image y_cbcr,
	                            vk::SemaphoreSubmitInfo sem_info,
	                            uint64_t frame_index);

	void on_feedback(const from_headset::feedback &);
	void reset();

	// bitrate_bps is the bitrate for the whole stream
	// the encoder bitrate will be scaled accordingly
	void set_bitrate(uint32_t bitrate_bps);
	void set_framerate(float framerate);

	void encode(wivrn_session & cnx,
	            const to_headset::video_stream_data_shard::view_info_t & view_info,
	            uint64_t frame_index);

	// Tags a frame for the encoded-size triage below. Called on the
	// worker thread before encode() for the same frame.
	void note_frame_kind(uint64_t frame_index, bool warped);

protected:
	// Called on present to submit command buffers for the image.
	// Returns false when the slot's previous work hasn't completed (fence
	// or sender-busy poll miss): the caller marks the frame skipped
	// without recording or mutating any backend state. Must not block.
	virtual bool try_present_image(vk::Image y_cbcr,
	                           vk::SemaphoreSubmitInfo sem_info,
	                           uint8_t slot,
	                           uint64_t frame_index) = 0;

	// called when command buffer finished executing
	virtual std::optional<data> encode(uint8_t slot, uint64_t frame_index) = 0;

	void SendData(std::span<uint8_t> data, bool end_of_frame, bool control = false);
};

} // namespace wivrn
