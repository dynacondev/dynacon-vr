# AGENTS.md — WiVRn agent notes

Lab notebook for coding agents working on this repo. Append lessons that cost real device round-trips.

## Decoder YCbCr generations: views must pin the conversion they were baked with (Quest 3, Nov 2026)

- Symptom: SIGSEGV (`SEGV_MAPERR`, tiny fault addr) inside Adreno `vkUpdateDescriptorSets`, on the first depth-paired frame after a mid-run AHB format renegotiation. Tombstone backtrace pointed at `union_merge::record_finish`'s apply update — but the writes were byte-identical to the working build and validation-silent, which misdirected toward the mask path for a full audit round.
- Cause: pre-existing decoder lifecycle, not the mask code. `android_decoder::map_hardware_buffer` (decoder thread) destroyed `ycbcr_conversion`/`ycbcr_sampler` and cleared `hardware_buffer_map` on format change, while in-flight blit handles still referenced views baked with the dead conversion (`blit_handle::image_view` is a non-owning handle; `mapped_hardware_buffer` ownership via `vk_data` kept views alive but nothing kept the conversion). The render thread additionally read `decoder->sampler()` unlocked — both a data race and a dangling handle. A stale pre-renegotiation frame paired via the ±1 fallback then segfaulted the driver resolving the dead conversion (a mere mismatch would be VUID-01948, not a SEGV — the SEGV proves use-after-free).
- Fix (`client/decoder/android_decoder.*`, `decoder.h`): per-format-generation shared ownership — `ycbcr_generation { conversion, sampler }` held by each mapped buffer and transitively by each blit handle; `blit_handle::sampler` carries the frame's own generation; the render thread never touches decoder-owned sampler state (`decoder::sampler()` virtual deleted). Stale frames now sample as merely-old depth. Same pattern applied to the defoveator (`stream_defoveator`: track baked samplers per pipeline, rebuild on generation change — it previously baked the first sampler forever) and to the call sites (`dh->sampler` everywhere, incl. video color/alpha + viz).
- Rules:
  1. Never store a decoder-owned Vulkan object in a longer-lived structure and never read one cross-thread without a generation pin: AHB renegotiation destroys conversion + sampler + views at any format change.
  2. When a tombstone blames `updateDescriptorSets` with identical-to-working writes, suspect the objects behind the handles (destroyed conversion / dangling view), not the write structs — and check whether a renegotiation just happened (`decoded image size` + `verdict layout rebuilt` within ~100ms of the crash is the fingerprint).
  3. Validation-silent + release-only-SEGV on this path means use-after-free, not a usage error: VVL would have named a mismatch (01948); it named nothing.
- Residual (known, out of scope): `v4l2_decoder`'s destructor bypasses its own `held_by_vulkan` guard (source-change path throws correctly); Quest uses the android path. `stream_defoveator`'s pool has zero headroom (`maxSets = 4` = max live) — any future alloc-before-free reordering throws `OutOfPoolMemory`.
- CORRECTION (device-verified after this entry was written): the generation fix above did NOT resolve the crash — it recurred identically before and after (same fault addr, register-identical tombstones across builds). The actual crasher was the apply descriptor update's write pattern (see next section); the generation pinning stands as latent-hazard hardening, but rule 2's fingerprint (`decoded image size` + renegotiation) was a red herring for this incident — the crash reproduces with no renegotiation at all, on the first paired frame of the session.

## Adreno: always write the full descriptor footprint — partial writes across frames fault updateDescriptorSets (Quest 3, Nov 2026)

- Symptom: deterministic SIGSEGV (`SEGV_MAPERR`, fault addr `0x2d`) inside Adreno `vkUpdateDescriptorSets`, on the first depth-paired frame of the session — placed object + depth on. Crashed identically across builds (register-identical tombstones); validation layer silent (usage is spec-legal), debug build clean. A 5-mode bisect isolated it: crash iff the apply update contained the binding-1 write (verdict R8 2DArray view + plain nearest sampler + `SHADER_READ_ONLY`); binding-0-only clean; skipping both clean; binding-1-only crashed with a different fault addr (`0x277`). Inline debug info pinned the fault to the apply `updateDescriptorSets` line itself.
- Cause: `record_finish` wrote binding 0 alone on mask-only frames, then bindings 0+1 on the first paired frame — the first-ever binding-1 write to a binding-0-dirtied set faulted Adreno's (release-only) descriptor path. Suspected first-touch bookkeeping for the newly-written binding (unconfirmed driver internals); NOT a bad handle (all handles valid), NOT the conversion lifetime (crashed with the verdict update compiled out), NOT rendered-vs-unrendered (crashed with the verdict draw both skipped and executed).
- Fix (`client/render/union_merge.cpp`): always write both bindings on every update, paired or mask-only — binding 1 carries the verdict view even when the mask pipeline (declares binding 0 only) never samples it, which is legal since the layout declares both. Device-verified: no crash since, paired frames render with live `umgpu` splits.
- Rules:
  1. Never grow a descriptor set's written footprint across frames on Adreno: first write must cover every binding the set layout declares (unused bindings get a valid-but-unsampled descriptor). The crash signature is unmistakable — first-use of a new binding in an update call, release-only SEGV inside the driver, VVL silent.
  2. A "validation ran clean" result only exonerates code paths the validation run actually executed: the paired verdict path runs zero times per session until an object is placed with depth paired (here: zero times in ~54s). Drive the paired path under validation before claiming coverage.
  3. When bisecting a driver crash, prefer empty-pass skips (keep renderpass shells, barriers, layouts; drop update+draw together) — every executed call stays identical to normal, so disappearance/appearance is conclusive. And symbolize with the exact-build `.so` (match tombstone BuildId): `llvm-addr2line -i` names the precise update call through the inline chain.

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

## Lobby hands: arbitrate on joints + input-source + buttons, never joints alone (Quest)

- Symptom x2: (a) rays stuck on after putting controllers down, (b) one hand poke-dead (no ray, no click) while holding controllers. Both traced to the poke/aim arbiter in `imgui_impl::read_controllers_state()` keying on joint presence alone: (a) was the runtime still routing to awake desk controllers (`is_input_source()` false, CONTROLLER_GHOST data source), (b) was spuriously-valid joints on a gripped hand pinning that side to poke.
- Rules:
  1. Hands drive a side only when joints locate AND `is_input_source()` AND no controller-exclusive buttons (squeeze > 0.05, thumbstick > 0.2 deadzone — drift must not veto). The shared trigger action is deliberately excluded: under the hand profile it carries aim-activate, and counting it kills poke on every pinch-click.
  2. `locate()` only guarantees wrist + palm + index chain (`hand_tracker.cpp`); other fingers may be curled/occluded. The old all-26 bail never fired in debug builds (`#ifndef NDEBUG` pre-sets flags valid) — reproduces on device only.
  3. Debug via a transition log (`located input_source data_source buttons -> poke/aim` per side, fires on change only): `located=0` = joint bail, `input_source=0` = runtime routing, `buttons=1` = veto. Removed after validation; re-add if regressions return.
  4. Poke needs no runtime cooperation (joint-driven touch), so forcing poke when the runtime routes elsewhere is safe; the reverse (forcing aim) strands clicks. When in doubt, bias toward poke.

## Validation-layer workflow (Quest 3, Nov 2026)

- `assembleDebugWithValidationLayer` downloads VVL 1.4.321.0 android
  binaries, bundles the `.so`, and the app auto-enables
  `VK_LAYER_KHRONOS_validation` in non-NDEBUG builds
  (`application.cpp`). The debug callback aborts on error-level
  messages — each validation abort masks every later bug, so expect
  fix-and-rerun iterations (this branch needed four: pool flags,
  framebuffer layers, then two YCbCr rounds). Release builds run the
  same code silently: validation hits are UB that Adreno tolerates
  until it doesn't (GPU page fault → CP opcode errors → DeviceLost
  surfacing as `vk::DeviceLostError` in a later `waitIdle` — the abort
  message names the symptom, never the cause).
- `vk_layer_settings.txt` (`DEBUG_ACTION_LOG_MSG,BREAK`, thread-safety
  validation) is written but only takes effect if packaged; a bare
  SIGTRAP with no message means BREAK fired without the log path.
- Startup `VUID-vkCreateInstance-ppEnabledExtensionNames-01388`
  (runtime enables `surface_protected_capabilities` without
  `get_surface_capabilities2`): satisfied by requesting the latter
  ourselves when enumerated — harmless if the runtime stops needing it.

## YCbCr immutable samplers (Quest 3 AHB imports, Nov 2026)

- Any fragment sampling of an AHB-imported YCbCr view through a
  combined image sampler MUST use immutable samplers:
  mutable layout + converted view = VUID-02738; immutable layout +
  mismatched conversion = VUID-01948. Both abort debug builds and are
  silent GPU-fault/undefined reads in release.
- Proven pattern (defoveator + verdict): bake the DECODER's live
  sampler as the immutable sampler, rebuild the full verdict state
  (set layout + sets + pipeline layout + pipeline — a pipeline bakes
  its set layout, never one without the other) when the conversion
  definition OR the decoder sampler handle changes, waitIdle'd (rare).
- Failed pattern (do NOT retry without new evidence): building a
  separate conversion from copied AHB format params (format/model/range/
  components/offsets/filter/external all matched on paper, chain built
  per headers) still tripped 01948 against the decoder's view
  conversion. Cause undetermined — params vs chain-construction quirk
  vs VVL external-format strictness.
- Every backend's converted views must feed the definition plumbing:
  Android AHB (per-handle, always converted), raw 420 streams
  (explicit format, no external node — the external node must NOT ride
  along when format != Undefined), v4l2 capture (capture format).
  ffmpeg/plain views stay mutable. The definition rides the blit handle
  (immutable after map), never a shared decoder object (recreated on
  format change — a shared reference would strand).
- Pre-existing quirk (out of scope, noted not fixed): the defoveator
  bakes the decoder sampler once and never rebuilds on mid-session
  decoder recreation — same staleness class the verdict now handles.

## Multiview framebuffers: layers=1 (spec, Nov 2026)

- `VUID-VkFramebufferCreateInfo-renderPass-02531`: a render pass with
  non-zero view masks requires framebuffer `layers == 1`. The eye
  routing comes from the view mask into the 2-layer image views, not
  from the framebuffer — `.layers = 2` (13 sites: merge + feather +
  hidden + submit targets) was spec-invalid but Adreno-tolerated, which
  is why stereo worked for years with zero signal until validation ran.
- Fix is behavior-identical on a conformant driver (same pixels, same
  layers) but touches every stereo pass: verify both-eyes-distinct on
  device after any change here. `arrayLayers = 2` / `layerCount = 2` /
  view/correlation masks / render areas are the actual 2-layer
  machinery and must NOT change. Non-multiview passes (defoveator,
  gizmo, imgui, depth overlay: per-eye loops, no MV pNext) correctly
  keep their own values.

## Descriptor pools: FREE flag or whole-pool reset (Nov 2026)

- `vk::raii::DescriptorSet` destruction calls `vkFreeDescriptorSets` —
  any pool whose sets die individually needs
  `eFreeDescriptorSet`, or validation aborts (silent UB in release).
  Fixed: feather pool, depth-overlay pool (plus clear-before-recreate:
  destroying the pool first strands the live set). Pattern precedent:
  union_merge pools already had it.
- Alternative discipline (defoveator `ds_pool`): sets are raw handles
  never freed individually — legal without the flag ONLY with
  whole-pool destruction, plus an explicit `pool.reset()` in
  `reset_pipelines` (abandoned sets otherwise leak the 4-set pool into
  `OutOfPoolMemory` on re-ensure). Declared member order must destroy
  sets before their pool. Comment the discipline where it applies or
  the next reader will regress it.

## GPU crash forensics (Adreno, Nov 2026)

- Signature of a bad GPU read: `kgsl-3d0: GPU PAGE FAULT: addr = ...`
  (deterministic low bits across runs = real overrun of a real
  allocation, not a wild pointer), then `adreno-gen7-gmu: CP opcode
  error` repeats, `gpu fault threshold exceeded`, `ErrorDeviceLost`,
  abort in the next `waitIdle`. Triage order: faults first (they name
  the victim allocation), CP errors second (wedged context), the abort
  last (never the cause).
- Victim identification: log `requirements.size` (and extents/format)
  at every owned-image creation (merge targets, decoder AHB maps);
  match against the kernel's fault-adjacent `vk_devicememory` ranges in
  the same run (cross-run sizes vary ~24KB — same-run pairing only).
- Validation aborts (debug) pre-empt DeviceLost runs: a crash that moves
  earlier under validation is usually the validator catching the bug
  upstream, not a new bug — compare write sites (01948 names the
  binding), not just timestamps.

## Verdict opt #7 revert: cached SSBO loads beat per-pixel run walks (Quest 3, Nov 2026)

- Compact-run verdict (shader walks foveation runs, no CPU expand, ~1KB
  SSBO) was valid on device but regressed `umgpu` verdict ~2.7ms to
  ~4.5-6.4ms across many paired samples. Reverted (`464148d9` +
  `48c207a8`), tree back to byte-identical pre-#7.
- Why: the old 2 random SSBO loads/px were already cheap (34KB map
  written once, streams through cache once per frame). The walk added
  per-pixel trip-count divergence (warps spanning run boundaries
  serialize) plus an integer divide per axis — Adreno emulates it.
- Rule: `umcpu` decides whether CPU work is worth killing — it was
  0.02-0.05ms before AND after, so the expand was never the cost.
  Never trade 2 cached loads for per-pixel search on this GPU without
  device numbers; the "uniform-ish ALU" prediction ignored divergence.
- The verdict's remaining cost is the full-res pass itself (layer tap +
  YCbCr conversion tap + R8 write, x2 eyes) — bandwidth territory (#8
  fold), not ALU.

## Depth has no plain Y-plane view on the client (Quest 3, Nov 2026)

- Opt #9 (raw R16 tap of the depth Y plane) is blocked as framed — do
  not retry without new evidence. The staging image's plain R16 views
  (`compositor.cpp` `out_y`) never cross the encode; the client sees
  only the H.265 Main10 decode via MediaCodec (opaque PRIVATE
  AImageReader). Device log proves the import: Vulkan format
  `Undefined`, external format `0x28e`, so every view of the image must
  carry a YCbCr conversion — a conversion-free R16 plane view of a
  UNDEFINED-format image is spec-illegal (no compatible plain format).
- Switching the AImageReader off PRIVATE to get a Vulkan-mappable RGB
  format would add a full-res decoder-side conversion, lose the
  zero-copy import, and risk the 10-bit codes — expected net loss.
- Even hypothetically, the prize is small: the verdict taps virt once
  per surviving pixel already; the conversion-hardware delta on one tap
  is minor next to the pass's read+write bandwidth.
