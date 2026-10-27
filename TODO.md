# TODO — commit/GPU decoupling (tick-train punctuality under saturation)

## Why

`layer_commit` records, submits, then parks the commit thread in
`waitSemaphores` (up to 1 s) until our GPU work completes
(`server/compositor/compositor.cpp:787-795`). Every millisecond of GPU
variance (X-Plane saturating the SMs, driver bubbles, our 0.5–3.5 ms
warp) lands directly in the *next* tick's start time. Past ~11 ms of
tail, a tick slips (17–21 ms inter-commit gaps in the logs) → visible
stutter. `VK_EXT_global_priority=HIGH` shrinks the variance; this plan
removes its *transmission* to the tick train.

Mental model (start vs finish deadlines): each tick must *start* every
11.1 ms, but a frame may *finish* late without shifting the train. The
2-deep image pool gives ~22 ms of finish room for ~11 ms of start
cadence: transients absorb as single-frame lateness, exhaustion drops
instead of slipping. No additional buffering: pool depths stay 2,
commit-start rule (latest layer at start) unchanged.

## Design

Slot = compositor image index (0/1). Compositor-side state for
`images[i]` uses slot `i`; encoder internals keep their own
present/encode lockstep (unchanged, deliberately NOT re-keyed by image
index — backend slots, fences, output buffers, DPB and timestamp pools
are self-consistent via the base state machine + per-slot fences).
Safety argument (no timing arguments): slot `i` is touched only while
image `i` is held (busy), and image `i` is freed only after encode
consumed it. Per-backend completion link: vulkan/x264/raw wait our
timeline semaphore explicitly; nvenc/va ignore the semaphore parameter
(`video_encoder_nvenc.cpp:425`, `video_encoder_va.cpp:439`) and are
ordered by same-queue FIFO behind our submit plus the compositor-side
release barrier (`compositor.cpp:702-733`). Recycled image ⇒ completed
submit ⇒ free slot, except under wedge (see policy below). Dropped
commits touch nothing.

Wedge policy (explicit gap in the proof): every backend `encode()`
frees its image after a 1 s fence timeout, so slot-reuse-after-timeout
is formally unproven. Keep existing free-and-log behavior (recovery
over leak: leaking would halve the pool forever on a transient wedge),
plus a loud wedge counter in triage. The commit-tail reset hazard
disappears with the tail wait; the worker-timeout hazard is bounded by
the 1 s timeouts and converts wedge to drops + log, never to a parked
tick train.

1. **Per-slot frame context (compositor side).**
   - `compositor`: 2× cmd pool+buffer (no `eResetCommandBuffer` on the
     pool, so per-slot pools — reset only the acquired slot), 2×
     timestamp query pool of 4 (or 1×8 with `slot*4` offsets and
     slot-scoped reset only). Timeline `sem` stays single (monotonic
     `sem_value`, commit-thread-only; per-submit values waited by each
     encoder support arbitrary in-flight depth; per-encoder
     `stageMask` mutation is on by-value copies). `images[i]` barriers
     keep `oldLayout Undefined` (discards; no per-slot layout tracking
     needed). `view_info`/`frame_index`/`warped_frame` already
     image-slotted (`compositor.h:56`); drop paths touch only atomics +
     frame state.
   - `depth_reprojector`: `target[slot][eye]`, `splat[slot][eye]`
     (2→4 images+views), 4+4 UBOs, 4+4 sets (pool counts 4/4/6→8/8/12),
     `warp_ready[slot][eye]`; `reproject()`/`get_views()` take slot.
     Degenerate-view early returns leave `warp_ready` untouched.
   - `foveation`: 2× `gpu_buffer`, 2× `descriptor_set` (pool
     2/2/1→4/4/2), `last[2]`; `foveate()` takes slot. Member `params`
     is rewritten-before-use per call — no slotting. The
     `vmaCopy`+`memcpy` double-write stays (harmless).
   - `layer_squasher`: 4× UBO + 4× sets are MUST (CPU rewrite races);
     2nd `render_target` is OPTIONAL — sharing stays safe via
     same-queue FIFO + `Undefined`-discard ordering on the path where
     the warp stands down. Default: keep one target (halves the diff);
     record the decision at implementation. `do_layers()` and
     accessors take slot either way.
2. **Non-blocking encode handoff (ordering is load-bearing).**
   - Backend poll order, before `cmd.begin`, before `dpb.get_ref`,
     before rate-control exchanges: fence-poll (zero timeout) AND
     Vulkan `slot_item.busy` poll (`video_encoder_vulkan.cpp:653` —
     a second, indefinite commit-thread park the naive fix misses).
     On either not-ready: mark skip and return without recording or
     mutating DPB/rate state.
   - Base order (`video_encoder.cpp:237-250`): busy-poll precedes
     `idr->should_skip()` — skipping after that call poisons the IDR
     state machine (~100 frames of skips). Skip preserves
     `present_slot`/`encode_slot` lockstep (both sides take the alpha
     skip identically); partial per-stream skip is acceptable, with
     per-stream telemetry.
   - `encoder_work` (`compositor.cpp:915-929`): move try/catch inside
     the per-encoder loop — one encoder's `encode()` throwing must not
     stall the others' `encode_slot` while their `present_slot`
     already advanced (use-after-submit). `image.busy=false` still
     runs.
   - Raw backend: add the missing sender-lifetime guard (mirror the
     Vulkan `busy` flag + poll, or make raw synchronous). Raw returns
     a span to its per-slot buffer with no `mem` guard while the
     sender thread may still transmit it — pre-existing, widened by
     running ~1 frame further ahead.
   - `encode()` on the worker thread keeps its blocking waits
     (fences, timestamp `eWait` in `gpu_timestamp_pool.cpp:87`); only
     the commit-thread waits become polls. `sender`/`shard` lifetime
     via `wait_idle` stays — do not simplify it away.
3. **Non-blocking query reads.** Poll timestamp availability per slot;
   read slot `i` lazily on next acquire of `i` (reuse implies
   completion, so reads are almost always ready). Skip + count on
   miss (counter doubles as wedge detector). Never park for telemetry.
4. **Delete the tail `waitSemaphores` + `eWait` reads.** Per-tick
   commit-thread work becomes record + submit + handoff with no
   GPU-progress waits (remaining parks are CPU/driver-lock bounded —
   see run criteria). A slow GPU delays individual frame *completion*,
   never the next tick. Drop counting (`drop_busy`/`drop_acquire` +
   new `drop_present` counter) stays as the backpressure valve.

Deliberately *not* merged: warp+encode stay on separate queues/engines
(SMs vs NVENC) chained by semaphore/fifo, not one submission.
Separating engines lets N's (uncontended) encode overlap N+1's
(contended) warp; for NVENC/VA the pixel-copy already serializes with
the warp on the same queue, so the overlap benefit applies to the
HW-encode portion. Merging would serialize stages for no gain and
still not exclude X-Plane's cross-device interleaving (only priority
arbitrates that; HIGH is the ceiling). Pools stay depth-matched at 2,
not unified — unifying 2+2 gives shared-2, not 4; depth 3 would buy
absorption as staleness, worse than drops for VR.

## Check items (status)

1. `comp_swapchain_shared_garbage_collect` — CLOSED, safe: only pops
   the destroy stack and calls `real_destroy`
   (`comp_swapchain.c:677-684`), no fence/semaphore wait. Callable
   pre-completion, no repositioning.
2. `present_image` waits — CONFIRMED present (base `state.wait` + all
   backends' `waitForFences` + Vulkan `slot_item.busy.wait`). Covered
   by design step 2 above with the stated poll order.
3. Commit threading — design does not depend on it: single-threaded
   still overlaps GPU-side (N+1 records while N executes);
   multi-threaded is already hazardous on the single `cmd` today.
   Verify only to size review, not to gate.

## Slices (in order)

1. Compositor per-slot cmd pool+buffer and query pool; slot plumbing
   through `layer_commit`. No behavior change (wait still present).
2. Warp per-slot targets/splat/UBOs/sets/`warp_ready`; slot param
   through `reproject()`/`get_views()`. No math changes.
3. Foveation per-slot buffer/set/`last`; slot param through
   `foveate()`. Careful, review-heavy, no behavior change.
4. Squasher per-slot UBOs/sets (+target decision recorded); slot param
   through `do_layers()` and accessors.
5. Non-blocking queries + availability-miss counter; non-blocking
   `try_present_image` (base + all backends, incl. `slot_item.busy`
   poll, DPB/IDR poll-ordering, per-encoder try/catch, raw lifetime
   guard) with `drop_present` counter, plus a per-encoder try/catch
   around the commit-thread present loop so a throw can't strand an
   image. Still with the tail wait (proves the new
   paths under the old safety net). Flag day is gated on all of the
   above, not just the tail wait.
6. Remove tail wait + `eWait`. Nothing outstanding (items closed
   above); this is the flag day.
7. Run: `gap_over` 0 under saturation that previously slipped;
   `warp_gpu` distribution unchanged; no validation errors (including
   maintenance9-non-optimal multi-family QFOT reuse, which is
   spec-suspect but almost certainly never hit — verify, don't
   redesign); drops (busy/acquire/present) only under true
   overload; steady-state motion-to-photon latency unchanged
   (transient +≤1 frame under spikes by design). Residual `gap_over`
   suspects if non-zero: queue-mutex contention (`vk.queue.mutex`
   shared with Monado IPC, backend queue mutexes), blocking tracking
   TCP sends under the tracking mutex (twice per commit), foveation
   `compute_params` CPU per commit — measure first, all explicitly out
   of this plan.

## Non-goals / follow-ups (out of this plan)

Warp math/barriers/submit order (beyond slotting), `view_info`, UBO
contents, pool depth changes (stays 2), warp+encode pool merging,
async compute, client-side warp, motion gating, realtime priority.
Follow-ups if residual gaps survive: move tracking sends off the
commit thread / off the tracking mutex; quantize the gaze cache if
`compute_params` is hot; queue-mutex contention is measured-first,
`try_lock`/drop later. Priority (HIGH) stays as the variance reducer;
this plan is the transmission remover.
