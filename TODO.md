# TODO — Vulkan validation hits (Quest/PC device runs, Nov 2026)

Captured with `VK_LAYER_KHRONOS_validation` + `VK_LOADER_DEBUG=layer` in the
`tech.dynacon.vr` flatpak (`--command=wivrn-server`). Validation only runs
when explicitly enabled — release builds silently swallow all of the below.

## Depth-blockers (cause the +335-code VR offset — chroma reads 0)

FIXED in `34dcbcdf`, pending device verification (expect: zero VUIDs,
`srv chroma` 32768/512, client G converging to server values):
- [x] `VUID-vkCmdClearColorImage-aspectMask-02498` — neutral-chroma
  `clearColorImage` on the depth SBS staging image uses
  `aspectMask = PLANE_1`; only `COLOR_BIT` is legal for clears.
  (`server/compositor/compositor.cpp`, depth pack block)
- [x] `VUID-vkCmdClearColorImage-image-01545` — same call targets
  `VK_FORMAT_G10X6_B10X6R10X6_2PLANE_420_UNORM_3PACK16`, a format
  requiring YCbCr conversion, which `clearColorImage` forbids outright.
  The driver drops the clear → Plane 1 stays zero-filled → every
  decoded pixel gets `G = Y + 0.328` (+335 codes) under the forced
  709/Full conversion. Fix direction: write neutral chroma from the
  pack compute shader through the existing `out_cbcr` storage view
  (or a legal transfer fill), delete the `clearColorImage` block.

## Pre-existing, unrelated (session churn, not depth)

Root cause (3/3 independent audits agree): `compositor.cpp:717` is the
only `vkResetCommandPool` in `server/` — all three VUIDs are the same
per-slot compositor command buffer. `layer_commit` submits with NO
fence (timeline-sem only); recycle relies on `image.busy`, which the
worker clears unconditionally even on all-skip frames (post-reconnect
IDR wait: every encoder `skipped_idr`, worker returns early with zero
GPU sync, nobody ever waits on the submit). Next acquire reuses the
slot while the submit is still pending.
Fix: per-slot compositor fence (created signalled), `resetFences` +
pass to `submit2`; poll it at acquire (timeout 0) and take the
existing `drop_acquire` path on miss. Follow-ups: harden
`present_image` against throws (stuck-busy); nvenc/va submit without
waiting the compositor sem (torn-frame race, separate — CONFIRMED:
both take it unnamed, `video_encoder_nvenc.cpp:425`,
`video_encoder_va.cpp:439`); give `dpb_state::should_skip` a stall
timeout like the default IDR handler.
FIXED in `26cc0876`, pending device verification under validation
(expect: zero 00040/00049/03875, bounded `drop_acquire` instead).

- [x] `VUID-vkResetCommandPool-commandPool-00040` — reset with a
  command buffer still pending (client disconnect/reconnect path).
- [x] `VUID-vkBeginCommandBuffer-commandBuffer-00049` — begin on an
  active (recording/pending) command buffer. Same root cause/fix.
- [x] `VUID-vkQueueSubmit2-commandBuffer-03875` — submit of a pending
  command buffer not recorded with `SIMULTANEOUS_USE_BIT`. Same fix.
- [ ] Benign loader noise (not app bugs, ignore): `terminator_CreateDevice`
  / `vkCreateDevice: Failed to create device chain` (NVIDIA ICD quirk,
  falls back), `HIGH queue priority rejected` (falls back).

## Forensics TEMP commits to revert before merge (feat/occlusion-tweaks2)

- `7faaf1c4` TEMP: packed-vs-unpacked depth range logging
- `cfadad59` TEMP: barrier-order fixup
- `2ea3adc7` TEMP: tri-channel range + full YCbCr definition logging
- `b0671739` TEMP: staging chroma-plane histogram
