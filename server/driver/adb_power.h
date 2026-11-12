/*
 * WiVRn VR streaming
 * Copyright (C) 2026  Guillaume Meunier <guillaume.meunier@centraliens.net>
 * Copyright (C) 2026  Patrick Nicolas <patricknicolas@laposte.net>
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

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace wivrn
{

// Settings for the server-side ADB clock override, mapped from
// configuration().adb by the session (kept free of server includes so
// this module stays dependency-light and unit-testable).
struct adb_power_config
{
	bool enabled = true;
	// adb binary; under flatpak this runs as `flatpak-spawn --host <binary>`
	std::string binary = "adb";
	// explicit `adb -s` selector, empty = derived from the peer IP (see note_reconnected)
	std::string target;
	// per-command timeout, seconds
	int timeout_s = 5;
};

// Best-effort `adb shell setprop debug.oculus.{cpu,gpu}Level` driver.
//
// All work happens on an internal worker thread; every failure is logged
// and never propagates: callers request levels and carry on streaming.
// Levels: -1 = Auto (no override, the key is cleared), 0..6 = forced.
class adb_power_control
{
public:
	static constexpr int level_auto = -1;
	static constexpr int level_min = 0;
	static constexpr int level_max = 6;

	static constexpr const char * cpu_property = "debug.oculus.cpuLevel";
	static constexpr const char * gpu_property = "debug.oculus.gpuLevel";

	// Build "<host>:5555" from a numeric peer IP (brackets IPv6).
	static std::string target_for_peer_host(const std::string & host);

	explicit adb_power_control(adb_power_config config = {});
	~adb_power_control();

	adb_power_control(const adb_power_control &) = delete;
	adb_power_control & operator=(const adb_power_control &) = delete;

	// (Re)bind to a headset address. The transport may have been replaced
	// (reconnect, often after a headset reboot which wipes debug props):
	// pushed state is forgotten so the current wants are re-pushed.
	// Empty target disables override attempts (logged).
	void note_reconnected(std::string target, bool auto_connect);

	// Request levels; out-of-range values become Auto with a warning.
	// Only keys that differ from the last-applied state are sent; keys
	// back to Auto are cleared (empty setprop value).
	void request(int cpu_level, int gpu_level);

	// Clear any levels this object applied (no-op if none). Runs
	// synchronously, bounded by the command timeout.
	void clear();

private:
	void worker_main(std::stop_token stop);
	void apply_now(const std::string & target, int cpu, int gpu, bool auto_connect, const adb_power_config & config);

	// Run one adb command, capturing output. Returns true on exit 0;
	// all command failures are logged here.
	bool run_adb(const std::vector<std::string> & args, std::string & output);

	adb_power_config config_;
	std::mutex mutex;
	std::condition_variable_any cv;
	std::jthread worker;
	bool dirty = false;

	std::string target_;
	bool auto_connect_ = false;
	int want_cpu_ = level_auto;
	int want_gpu_ = level_auto;

	// Last state pushed to the device (per current target).
	std::string applied_target_;
	int applied_cpu_ = level_auto;
	int applied_gpu_ = level_auto;
	bool connected_ = false;
};

} // namespace wivrn
