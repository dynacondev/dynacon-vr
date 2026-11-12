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

#include "adb_power.h"

#include "utils/flatpak.h"
#include "util/u_logging.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <cstring>
#include <poll.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace std::chrono_literals;

namespace wivrn
{

adb_power_control::adb_power_control(adb_power_config config) :
        config_(std::move(config)),
        worker([this](std::stop_token stop) { worker_main(stop); })
{
	if (not config_.enabled)
		U_LOG_I("ADB clock override disabled by configuration, headset CPU/GPU levels follow the app's own hints");
}

adb_power_control::~adb_power_control()
{
	// Best-effort final clear of anything we pinned, then join. clear()
	// short-circuits when nothing was applied, so the common teardown
	// path costs no subprocess at all.
	clear();
	worker.request_stop();
	cv.notify_all();
	// ~jthread joins here, after the queue drained
}

std::string adb_power_control::target_for_peer_host(const std::string & host)
{
	if (host.empty())
		return {};
	// IPv6 literal: bracket for `adb connect [addr]:port`
	if (host.find(':') != std::string::npos and host.front() != '[')
		return "[" + host + "]:5555";
	return host + ":5555";
}

static int clamp_level(int level, const char * name)
{
	if (level < adb_power_control::level_auto or level > adb_power_control::level_max)
	{
		U_LOG_W("Ignoring out-of-range %s level override %d (valid: -1 Auto .. 6)", name, level);
		return adb_power_control::level_auto;
	}
	return level;
}

void adb_power_control::note_reconnected(std::string target, bool auto_connect)
{
	std::lock_guard lock(mutex);
	bool target_changed = target != target_ or auto_connect != auto_connect_;
	target_ = std::move(target);
	auto_connect_ = auto_connect;
	connected_ = false;
	// A reconnect often follows a headset reboot, which wipes debug props:
	// forget pushed state so the current wants are re-pushed. All-Auto
	// with an unchanged target stays silent (nothing to push).
	if (target_changed or want_cpu_ != level_auto or want_gpu_ != level_auto or
	    applied_cpu_ != level_auto or applied_gpu_ != level_auto)
	{
		applied_cpu_ = applied_gpu_ = level_auto;
		applied_target_.clear();
		dirty = true;
		cv.notify_all();
	}
}

void adb_power_control::request(int cpu_level, int gpu_level)
{
	std::lock_guard lock(mutex);
	cpu_level = clamp_level(cpu_level, "CPU");
	gpu_level = clamp_level(gpu_level, "GPU");
	if (cpu_level == want_cpu_ and gpu_level == want_gpu_ and not dirty)
		return;
	want_cpu_ = cpu_level;
	want_gpu_ = gpu_level;
	dirty = true;
	cv.notify_all();
}

void adb_power_control::clear()
{
	{
		std::lock_guard lock(mutex);
		want_cpu_ = want_gpu_ = level_auto;
		if (applied_cpu_ == level_auto and applied_gpu_ == level_auto)
			return; // nothing pinned anywhere: no subprocess, no wait
		dirty = true;
	}
	cv.notify_all();
	// Drain the queue synchronously so teardown never leaves a stale pin:
	// bounded by two command timeouts in the worst case.
	while (true)
	{
		std::unique_lock lock(mutex);
		if (not dirty)
			return;
		lock.unlock();
		std::this_thread::sleep_for(50ms);
	}
}

void adb_power_control::worker_main(std::stop_token stop)
{
	for (;;)
	{
		std::unique_lock lock(mutex);
		cv.wait(lock, stop, [this] { return dirty; });
		if (not dirty)
			return; // stop requested, queue drained
		int cpu = want_cpu_;
		int gpu = want_gpu_;
		std::string target = target_;
		bool auto_connect = auto_connect_;
		adb_power_config config = config_;
		lock.unlock();

		apply_now(target, cpu, gpu, auto_connect, config);

		lock.lock();
		// A newer request may have landed mid-apply: only go clean when
		// the queue still matches what was just pushed.
		if (want_cpu_ == cpu and want_gpu_ == gpu and target_ == target and auto_connect_ == auto_connect)
			dirty = false;
	}
}

void adb_power_control::apply_now(const std::string & target, int cpu, int gpu, bool auto_connect, const adb_power_config & config)
{
	if (not config.enabled)
		return;

	if (target.empty())
	{
		U_LOG_W("ADB clock override requested but no headset target is known (set adb.target or use wireless ADB); levels follow the app's own hints");
		std::lock_guard lock(mutex);
		applied_target_ = target;
		applied_cpu_ = cpu;
		applied_gpu_ = gpu;
		return;
	}

	{
		std::lock_guard lock(mutex);
		if (applied_target_ != target)
		{
			// New target (connect or session change): re-push everything.
			applied_cpu_ = applied_gpu_ = level_auto;
			applied_target_ = target;
			connected_ = false;
		}
	}

	if (auto_connect and not connected_)
	{
		std::string output;
		if (run_adb({"connect", target}, output))
		{
			U_LOG_I("ADB connected to %s", target.c_str());
			connected_ = true;
		}
		else
		{
			U_LOG_W("ADB connect to %s failed; is wireless debugging enabled and paired on the headset? "
			        "Levels follow the app's own hints until then. Output: %s",
			        target.c_str(),
			        output.c_str());
			return;
		}
	}

	// Push each changed key; commit under lock only if the target did not
	// move meanwhile (locks are never held across subprocess calls).
	int cur_cpu, cur_gpu;
	{
		std::lock_guard lock(mutex);
		cur_cpu = applied_cpu_;
		cur_gpu = applied_gpu_;
	}

	auto push = [&](const char * property, int level, int current) {
		if (level == current)
			return current;
		// Empty value clears the override back to the app's own hint.
		std::string value = level == level_auto ? "" : std::to_string(level);
		std::string output;
		if (run_adb({"-s", target, "shell", "setprop", property, value}, output))
		{
			if (level == level_auto)
				U_LOG_I("ADB clock override cleared: %s on %s (headset decides again)", property, target.c_str());
			else
				U_LOG_I("ADB clock override applied: %s = %d on %s", property, level, target.c_str());
			return level;
		}
		U_LOG_W("ADB clock override failed: %s = %s on %s. Streaming continues with the app's own hints; "
		        "pair wireless ADB on the PC (adb pair / adb connect %s) and retry by changing the setting. Output: %s",
		        property,
		        value.empty() ? "<clear>" : value.c_str(),
		        target.c_str(),
		        target.c_str(),
		        output.c_str());
		return current;
	};

	int new_cpu = push(cpu_property, cpu, cur_cpu);
	int new_gpu = push(gpu_property, gpu, cur_gpu);
	{
		std::lock_guard lock(mutex);
		if (applied_target_ == target)
		{
			applied_cpu_ = new_cpu;
			applied_gpu_ = new_gpu;
		}
	}
}

bool adb_power_control::run_adb(const std::vector<std::string> & args, std::string & output)
{
	output.clear();

	// Build argv in the parent: after fork the child must not allocate
	// (fork in a multithreaded process), so it only fills a stack array.
	std::vector<std::string> storage;
	if (wivrn::flatpak_key(wivrn::flatpak::section::session_bus_policy, "org.freedesktop.Flatpak") == "talk")
	{
		// Under flatpak the binary lives on the host (permission
		// --talk-name=org.freedesktop.Flatpak is in the manifest).
		storage.emplace_back("flatpak-spawn");
		storage.emplace_back("--host");
	}
	storage.push_back(config_.binary);
	storage.insert(storage.end(), args.begin(), args.end());

	int pipefd[2] = {-1, -1};
	if (pipe(pipefd) != 0)
	{
		U_LOG_W("ADB clock override failed: pipe: %s", strerror(errno));
		return false;
	}

	pid_t pid = fork();
	if (pid < 0)
	{
		U_LOG_W("ADB clock override failed: fork: %s", strerror(errno));
		close(pipefd[0]);
		close(pipefd[1]);
		return false;
	}

	if (pid == 0)
	{
		// Child: redirect stdout+stderr to the pipe, new process group
		// so a timeout kill takes the whole tree.
		dup2(pipefd[1], STDOUT_FILENO);
		dup2(pipefd[1], STDERR_FILENO);
		close(pipefd[0]);
		close(pipefd[1]);
		setpgrp();

		// Stack array only: at most flatpak-spawn(2) + binary(1) + 6 args.
		char * argv[12] = {};
		if (storage.size() + 1 > std::size(argv))
			_exit(127);
		for (size_t i = 0; i < storage.size(); i++)
			argv[i] = storage[i].data();

		execvp(argv[0], argv);
		// execvp only returns on failure (e.g. adb not installed).
		dprintf(STDOUT_FILENO, "exec failed: %s\n", strerror(errno));
		_exit(127);
	}

	// Parent: drain the pipe until EOF or timeout, then reap.
	close(pipefd[1]);

	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(std::max(1, config_.timeout_s));
	bool timed_out = false;
	{
		char buf[1024];
		// Short poll slices: no timeout arithmetic can overflow, and a
		// stuck child is reaped promptly after the deadline.
		while (std::chrono::steady_clock::now() < deadline)
		{
			pollfd pfd{.fd = pipefd[0], .events = POLLIN, .revents = 0};
			int r = poll(&pfd, 1, 100);
			if (r < 0)
			{
				if (errno == EINTR)
					continue;
				break;
			}
			if (r == 0)
				continue;
			if (pfd.revents & (POLLIN | POLLHUP))
			{
				ssize_t n = read(pipefd[0], buf, sizeof(buf));
				if (n <= 0)
					break;
				output.append(buf, size_t(n));
				if (output.size() > 4096)
					output.resize(4096);
			}
			else
				break; // POLLERR/POLLNVAL
		}
		timed_out = std::chrono::steady_clock::now() >= deadline;
	}
	close(pipefd[0]);

	if (timed_out)
	{
		kill(-pid, SIGKILL); // whole process group (see setpgrp above)
		int status = 0;
		while (waitpid(pid, &status, 0) < 0 and errno == EINTR)
			;
		U_LOG_W("ADB clock override failed: adb command timed out after %ds and was killed", std::max(1, config_.timeout_s));
		return false;
	}

	int status = 0;
	while (waitpid(pid, &status, 0) < 0 and errno == EINTR)
		;
	if (WIFEXITED(status) and WEXITSTATUS(status) == 0)
		return true;

	U_LOG_W("ADB clock override failed: adb exited with status %d", WIFEXITED(status) ? WEXITSTATUS(status) : -1);
	return false;
}

} // namespace wivrn
