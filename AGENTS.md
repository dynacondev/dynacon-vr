# AGENTS.md — WiVRn agent notes

Lab notebook for coding agents working on this repo. Append lessons that cost real device round-trips.

## Fullscreen black + per-frame "Invalid pose submitted": check GUI quads first (Quest 3)

- Symptom: persistent full-frame black (video *and* overlay GUI gone), log shows `Invalid pose submitted` every frame, recovery as soon as GUI hides. Head tracking fine, no swapchain/LUT errors. First found via a combo dropdown whose popup overlapped the tooltip-strip viewport.
- Cause: `imgui_impl::end_frame()` (`client/render/imgui_impl.cpp`) emits one `XrCompositionLayerQuad` per viewport overlapped by any active window. A viewport whose `position`/`orientation` was never assigned (default `glm::quat` is *not* zero-initialized — reads as `(0,0,0,0)` in practice) produces a zero-quaternion quad. One bad layer fails the whole `xrEndFrame` with `XR_ERROR_POSE_INVALID`, so a single bad GUI quad blacks out everything, every frame the overlap persists.
- Concrete instance: stream GUI viewports (`client/scenes/stream.cpp:556`) — main `[0]` posed every frame, popup `[2]` placed via `place_layer_relative`, but tooltip strip `[1]` relied solely on the per-tooltip hook. Any non-tooltip window crossing its image rect (e.g. a low combo popup) emitted the poison quad. High combos worked (popup stayed inside viewport `[0]`), which misdirected toward the LUT code.
- Rules:
  1. Every viewport in `layers_` must hold a valid pose on *every* frame it could overlap a window — place fallback poses up front (see `place_layer_relative(1, 0, ...)` in `stream_gui.cpp`), refine per-window after (tooltip hook pattern).
  2. `end_frame()` now skips non-normalized viewport poses with a one-shot warn instead of submitting — never let one bad quad fail the frame.
  3. Prefer non-popup selectors (‹ › steppers, radios) for controls near the bottom of scrollable pages: combo popups open downward past the panel into the next viewport's image rect and split across quads (chopped, detached look even when poses are valid).
  4. The `end_frame` quad guard (`imgui_impl.cpp`: skip non-normalized viewport quads with a one-shot warn) is the backstop: with it, the worst case for an unplaced viewport is a missing quad, never a failed frame.
- Related trap: the same "everything black" signature previously also came from `xrReleaseSwapchainImage ... CALL_ORDER_INVALID` after a throw between `begin_frame`/`end_frame` skipped the submit (valueless `passthrough` variant poisoned every later frame). `CALL_ORDER_INVALID` (not `POSE_INVALID`) points at frame-loop poisoning (`session::enable_passthrough` emplace guard), not poses.

## Adreno: never use `mat3` constructors/ternaries in fragment shaders (Quest 3, Adreno 740, driver 512.837.9)

- `mat3(vec4, vec4, vec4)` (and the `(cond) ? mat3 : mat3` ternary) **miscompiles** in a multiview fragment shader: the resulting matrix reads back as garbage/zero on device with no error, no validation complaint, no crash.
- Symptom: everything downstream of the matrix goes dark (e.g. `p.z <= 0` guards fire on every pixel). CPU-side logging looks perfect, GPU timestamps show the pass executing, merge/seed/layers all proven — only the `mat3` math is wrong.
- Diagnostic trap: any echo/viz gated behind `mat3`-derived values (even a constant echo placed *after* the guards) also reads black, which falsely implicates the data transport (UBO/push/texel buffer all looked "dead" here while likely fine). **Always validate a new data path with an unconditional constant output placed *before* any computed guards.**
- Rule: spell small matrix math as **plain scalar dot products** (`float px = c0.x * uv.x + ...`), select per-eye columns with integer arithmetic (`int e = gl_ViewIndex * 3`), no ternaries on matrices.
- Proven-good read units in this driver path, in order of evidence: `sampler2DArray` texture taps, per-eye UBO (see `depth_overlay`), scalar push constants in the vertex stage. Unproven here (all masked by the above): fragment-stage UBO/push/texel-buffer reads — treat as guilty until proven innocent with a guard-free echo.

## Client display refresh rate: request late, then restore on reset (Quest)

- Symptom: lobby always runs at the runtime default (72Hz on Quest 3) even with 90/120 saved; toggling the setting fixes it until next launch.
- Cause chain: nothing applied `preferred_refresh_rate` at startup (only the GUI toggle and stream setup called `set_refresh_rate`). Worse, the runtime resets the rate back to default around session start (READY→SYNCHRONIZED→VISIBLE→FOCUSED), announced via `XR_TYPE_EVENT_DATA_DISPLAY_REFRESH_RATE_CHANGED_FB` *after* any early request — so proactive requests lose the race and skip-guards (`get_current_refresh_rate` checks) correctly no-op right before the reset lands.
- Fix (`application::apply_preferred_refresh_rate`, called from lobby `on_focused` and at FOCUSED, plus restore in `poll_events` on rate-changed): request late, then restore when the reset lands. Restores capped (3 per focused session, counter reset on FOCUSED) so an enforcing runtime or a deliberate system-side change is not fought forever; each honored restore ends the cycle since `to == preferred` stops further restores. While the stream scene is active the server owns the rate (`scene::manages_refresh_rate` guard, stream returns true) — lobby re-applies on return.

## Quest: colorScaleBias alpha scale is ignored, RGB + alpha bias are honored

- Symptom: hidden-object color pulse showed as a *steady* tint (then, reframed as a blink, a harsh strobe). CPU-side heartbeat proven oscillating in-device via throttled log while the display never varied — the runtime drops the alpha channel of `colorScale` on this path, so any alpha-graded overlay is impossible; mask-image alpha does pulse but ghosts game content over real objects.
- Design that threads it (`stream.cpp` hidden record/submit): opaque color matte twin below flat + dipping reveal above. One extra tier-0 matte raster (own swapchain, steady opacity 1, white silhouette × RGB scale = config color) sits above video below flat; the B-mask image alpha breathes `1−0.5·strength·s²` at ~1.2Hz, so the region mixes camera↔color with game structurally unable to show. `pulse-strength: 0` skips the matte (steady cutout).
- Rules:
  1. Never route warning visibility through `colorScale.a` (ignored) or mask-image alpha alone (game bleed). Opaque matte + dipping reveal composes only honored channels.
  2. Mask-layer RGB is invisible by design (`mask.frag.glsl`: ZERO color blend factors, flat covers via DST_ALPHA) — any color warning must live above flat or manifest as alpha.
  3. When display disagrees with proven-oscillating logs, suspect the compositor channel, not the math: gate one variable per device round-trip with a throttled forensic log.

## Hidden passthrough idle discipline: CPU pre-gate, activity-pinned flat, one sphere field
> HISTORICAL (dots overlay removed Nov 2026): rules 2-4 name the deleted dots layer / borrowed views. Rule 1 (CPU pre-gate) still applies.

- Proximity used to be evaluated only on the GPU (mask depth test, dots frag discards), so idle frames (object placed but out of range) paid full acquire/record/submit: 2 swapchain acquires, ~7 half-res passes incl. the sphere field rasterized TWICE (mask RP1 + dots Pass1b), 3 submitted layers (hidden mask, dots, flat pinned by placement).
- Rules:
  1. Gate on the CPU first (`stream.cpp` P0): world-space AABB per silhouette (baked in `build_hidden_silhouettes` from the model matrix) vs head+palm spheres, +2cm test-only hysteresis. Gate-miss erases the silhouette, so mask draws, dots draws and sphere MVPs all see survivors — idle costs ~zero GPU and no layers. Unknown bounds pass conservatively.
  2. Pin `need_flat` on submit activity (`hidden_group.active or hidden_dots_active`), never placement — a gated object must not hold the flat compositor layer (P1).
  3. One sphere raster per frame (P4): the dots-owned D16 field is THE shared image — mask RP1 renders into borrowed views, explicit barrier hands it to shader-readable for the dots pass later in the same command buffer, dots records only when the mask recorded (fresh-field rule). Next frame re-begins with initialLayout UNDEFINED + CLEAR (transient-target idiom, already precedented) — no transition back needed.
  4. Borrowed-framebuffer discipline: mask FBs bake dots-owned views — `reset_targets()` before the dots renderer dies (map-drop) or rebuilds (`ensure()` now returns rebuilt). Destroying views first is a validation error / driver hazard.
  5. Hands: every `locate()` is a full 26-joint IPC (no peek API) — cost is per call, not per handle. Proximity borrows the GUI pair (no dedicated trackers); `hidden_wants_hands` shields the pair from the GUI interactability reset while a hidden map is live.

## Hidden multiview + submit-layout notes (Quest 3 validation pending)
> HISTORICAL (dots overlay removed Nov 2026): dots passes/shaders/push notes below describe the deleted system. Mask MV rules still apply.

- Hidden path runs 4 multiview passes (viewMask=both, one draw broadcasts to both layers) instead of 8 per-eye: mask RP1+RP2 (`feather_mask.cpp` `record_hidden`), dots prepass+color (`hidden_dots.cpp` `record`). New shader twins (`mask_mv.vert/frag`, `hidden_dotfill_mv.vert`, `hidden_dots_mv.vert/frag`): plain `gl_ViewIndex` indexing only. Scissors are both-eyes unions (skip iff both degenerate); submitted color passes keep full renderArea/CLEAR; dots prepass renderArea is the content union (sampled only at dot centers).
- Push-size trap (caught by audit pre-commit, would have shipped silent garbage): with a shared push range, the VERT-declared block and the FRAG-declared block must agree byte-for-byte — pairing `mask_mv.vert` (opacity @128) with legacy `mask.frag` (opacity @64) passes pipeline creation and validation silently while the fragment reads `mvp[1][0][0]` as alpha (≈1 often, orientation-dependent reveal). Hence `mask_mv.frag` exists as a separate twin; never mix MV vert with legacy frag across a shared layout.
- 160B pushes (mask 132B, dots 176B) exceed the 128B Vulkan minimum: Adreno 740 provides 256B; both renderers throw loudly at init when the device reports less (no silent fallback — no 128B stereo-MVP encoding exists; fallback would be a dynamic-UBO redesign). In-tree precedent routes MV matrices via UBOs (`lit.vert.glsl`); push was chosen here for per-draw simplicity.
- Submitted color finals are COLOR_ATTACHMENT_OPTIMAL on the hidden paths only (mask + dots; nothing app-side re-reads those images). Feather GENERAL lifecycle deliberately untouched (cutout re-begin, seed_layout, make_readable olds, submit barrier — ~25 entangled sites; its own follow-up, biggest prize is the plain full-res mask). If the Quest runtime objects to OPTIMAL at submit, revert is 2 lines (`hidden_color_rp`, dots color RP) — failure would be loud (missing layers), not silent.

## Hidden dots overlay removed (trial, Nov 2026)

- The dots overlay system is stripped client-side: hidden reveal is the
  hard-edged proximity cutout only (mask RP1 spheres + RP2 silhouettes,
  merged, flat consumes). Removed: hidden_dots renderer + shaders,
  dot-lattice baking (feather set_mesh is soup-only), dots record/submit
  wiring in stream.cpp. The sphere field image moved from the deleted
  dots renderer into feather (ensure_hidden_field, waitIdled prewarm).
- Fold-into-merge was tried first and reverted: single-layer additive
  (ONE,ONE color) washes dots out over bright video (no true over
  without a video tap); binary hole-punch was discussed but not shipped.
- Config + packet fields removed too (server dot-pitch/dot-color,
  packet members, session fill, docs): dev branch, no compat shims.

## Soft-occlusion depth path replaces volume/VR compare (Nov 2026)

- Single `3d-depth-passthrough` only: `stream.cpp` counts placed depth
  volumes, `>1` latches `soft_multi_volume_error` + error log and sits
  out (exact-only) so no merge semantics are needed. Feather default is
  12px (packets + server config + docs), tier-1 half-res group.
- Compare is `soft_occlude_mv.frag.glsl`: single center tap, binary
  `sensorLin < mergedLin - mergedLin*bias` in linear meters (infinite-far
  branch like `depth_viz`), merged = `min(volume far, virtual, exact)`.
  144B push (`volume_homography` + `bias_near_far`) exceeds the 128B
  Vulkan minimum — ctor throws loudly under 144B like the silhouette
  (132B) and volume (136B) paths; Adreno 740 has 256B.
- Exact depth = passthrough (same feather, non-reversed) + hidden
  nearest raster (`exact_raster_pipeline_mv`, the renamed VR MIN pipe)
  into new `exact_depth` 320px target, binding 3; empty set rebinds the
  volume canvas. Reversed never rastered to depth (tweak). Hands: sensor
  hand removal on (fallback: depth hands), joint spheres
  (`soft_hand_sphere` soup, radii from `hand_tracker::mesh()` or
  35/16mm) appended as silhouette positives so they share the blur.
  Behind-virtual hands over-reveal (accepted, documented).
- Deleted outright: `volume_compare_mv` / `vr_occlude_compare_mv`
  shaders, VR eye-space compare + MIN-merge pipes, presmooth
  (binary feeds the shared blur directly), time-bias + eye-velocity
  history + slider (bias slider replaces it), all `vr_*` stats/plots/
  heartbeat, `ensure_vr_targets`, `volume_smooth`. `vr_draws` field and
  `use_vr` gating removed; `virt_eye_extent` field kept but unread.
