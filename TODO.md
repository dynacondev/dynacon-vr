# TODO — commit/GPU decoupling (tick-train punctuality under saturation)

## Why

`layer_commit` records, submits, then parks the commit thread in
`waitSemaphores` (up to 1 s) until our GPU work completes. If commits
serialize on that thread, every millisecond of GPU variance (X-Plane
saturating the SMs, driver bubbles, our 0.5–3.5 ms warp) lands directly
in the *next* tick's start time. Past ~11 ms of tail, a tick slips
(17–21 ms inter-commit gaps in the logs, ~every 2 s) → visible stutter.
`VK_EXT_global_priority=HIGH` shrinks the variance; this plan removes
its *transmission* to the tick train. No additional buffering: the
commit-start rule (latest layer at start) and queue depths are
unchanged — worst-case frame latency strictly improves, and the parked
thread (a hidden variable-length buffer) goes away.

## Design

1. **Slot shared upload state by image index.** Double warp UBOs/sets
   and foveation UBO/set; the commit for `images[i]` uses slot `i`.
   Safety proof (no timing arguments): slot `i` is touched only while
   image `i` is held (busy), and image `i` is freed only after encode
   consumed it — which strictly postdates our cmd's completion (encode
   waits our timeline semaphore). Recycled image ⇒ completed submit ⇒
   free slot. Dropped commits touch nothing. Holds even for a wedged
   GPU (today's timeout-path hazard class disappears with the timeout).
2. **Non-blocking query reads.** Poll timestamp availability; skip the
   timing update when not ready. Never park for telemetry.
3. **Delete the tail `waitSemaphores`.** With (1)+(2), nothing remains
   that needs completion. Per-tick thread work becomes pure CPU
   (record + submit + encode handoff). A slow GPU delays individual
   frame *completion*, never the next tick. Drop counting
   (`drop_busy`/`drop_acquire`) stays as the backpressure valve.

## Check items (verify at implementation, in this order)

1. `comp_swapchain_shared_garbage_collect`: does it self-gate on
   per-object fences (safe to call pre-completion, as upstream's
   wait-free design implies), or must it be repositioned? Do not delete
   the protection blindly.
2. `present_image`: does it contain bounded encoder-slot waits that
   park the thread the same way the tail did? If yes, they join the
   removal (or get the slot treatment). Grep the encoder backends, not
   just the base.
3. Confirm commit serialization assumption (single recommit thread):
   if commits can genuinely overlap today, the gap mechanism needs
   re-derivation before touching anything.

## Slices

1. Double warp UBOs/sets (`depth_reprojector`), indexed by view+slot;
   rewrite per warp into the commit's slot. No math/recording changes.
2. Double foveation UBO/set; thread the slot (== image index) through
   `foveate()`. Careful, review-heavy, no behavior change.
3. Non-blocking query reads; extend `warp_gpu` triage with an
   availability-miss counter so skipped reads are visible.
4. Remove the tail wait; resolve check items (1)+(2) first.
5. Run: `gap_over` must stay 0 under saturation that previously
   slipped; `warp_gpu` unchanged; no validation errors; drops only
   under true overload.

## Non-goals

Warp math/barriers/submit order, encode path, `view_info`, UBO
contents, query indices, async compute, client-side warp,
motion gating, realtime priority. Priority (HIGH) stays as the
variance reducer; this plan is the transmission remover.
