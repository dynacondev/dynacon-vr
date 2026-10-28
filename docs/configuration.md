# Configurable items

Configuration is split between headset and server. Headset contains most configuration, while server has system specific items such as encoders and paths.
Files are read from
- `/usr/share/wivrn/config.json` (where `/usr` is selected at configure time with `CMAKE_INSTALL_PREFIX`)
- `/etc/wivrn/config.json`
- `$XDG_CONFIG_HOME/wivrn/config.json` or if `$XDG_CONFIG_HOME` is not set, `$HOME/.config/wivrn/config.json`.

Files later in the list replace top-level values from previous ones.

If you installed WiVRn from a flatpack, host files are also searched: `/run/host/usr/share/wivrn/config.json`, `/run/host/etc/wivrn/config.json` and the host `$HOST_XDG_CONFIG_HOME/wivrn/config.json` (usually `~/.config/wivrn/config.json`, read from outside the sandbox, logged with `(via host)`), all before the sandbox files. When the host does not export `XDG_CONFIG_HOME` (the usual default setup), `$HOME/.config` is used instead, since `$HOME` names the same location inside and outside the sandbox. The sandbox per-app config in `$HOME/.var/app/tech.dynacon.vr/config/wivrn/config.json` is read last, so dashboard edits win. The dashboard home page lists every searched location with its status, and opening any entry shows it in the file browser.

All elements are optional and have default values.

## `bit-depth`
Default value: 10-bit if supported by the selected codec, encoder and decoder. 8-bit otherwise.

Bit depth of the video. 8-bit is supported by all encoders. 10-bit is supported by `vulkan`, `vaapi` and `nvenc` with `h265` or `av1` codecs.

## `encoder`
The encoder to use, either a single string or object applied to all streams, or a list of string or objects with values for left, right, alpha and virtual depth (in that order).
When a string it is used, it is equivalent to the `encoder` item of the object.

WiVRn encodes each eye separately, and the alpha channel as one for both eyes. Each stream is processed independently, this may use resources more effectively and reduce latency. A fourth stream carries the warped virtual depth (640x320 side-by-side, 8-bit 1cm/LSB) for depth-tested passthrough; it prefers `h265` and is pinned to ~5% of the total bitrate.

### `encoder`
Default value, in order: `vulkan` if supported, `nvenc` on Nvidia GPU, `vaapi` on all other GPU, else `x264`.

Identifier of the encoder, one of
* `x264`: software encoding
* `nvenc`: Nvidia hardware encoding
* `vaapi`: AMD/Intel hardware encoding
* `vulkan`: Vulkan Video encode, cross-vendor hardware encoding

### `codec`
Default value: best supported by both headset and encoder of `av1`, `h264`, `h265`.

One of `h264`, `h265`, `av1`, `raw`.

Not all encoders support every codec:
- `x264` encoder only supports `h264` codec
- `vulkan` encoder supports `h264` and `h265` codecs
- `raw` encoder only supports `raw` codec
- `nvenc` and `vaapi` support all codecs, except `raw`

If `nvenc` encoder is in use, you can refer to [nvidia website](https://developer.nvidia.com/video-encode-decode-support-matrix) to make sure that your GPU supports encoding with the desired codec.

### Examples
1. Simple configuration
```json
{
	"encoder": {
		"encoder": "vaapi",
		"codec": "h265"
	}
}
```
Use vaapi hardware encoding, h265 video codec (HEVC).

2. Hardware + software encoder
```json
{
	"encoder": [
		{
			"encoder": "vaapi",
			"codec": "h265",
		},
		{
			"encoder": "x264",
			"codec": "h264",
		},
		{
			"encoder": "vaapi",
			"codec": "h265",
		},
	]
}
```
Creates a hardware encoder for left eye and transparency, and a software encoder for right eye.

A list may also have 3 entries (left, right, alpha): the virtual depth
stream then uses default selection (same backend as configured, H.265
preferred when the headset supports it).

### `device`, only for vaapi
Default value: unset

Manually specify the device for encoding, can be used to offload encode to an iGPU. Device shall be in the form "/dev/dri/renderD128".


### `options` (very advanced), only for vaapi
Default value: unset

Json object of additional options to pass directly to ffmpeg `avcodec_open2`'s `option` parameter.

## `application`
Default value: unset

An application to start when connection with the headset is established, can be a string or an array of strings if parameters need to be provided.

### Example
```json
{
	"application": ["steam", "steam://launch/275850/VR"]
}
```
Launch No Man's Sky in VR mode on Steam when connection with headset is established.

## `tcp-only`
Default value: `false`

Only use TCP for communications with the client, this may have increased latency.
If `false` or unset, WiVRn will use both TCP and UDP.

### Example
```json
{
	"tcp-only": true
}
```

## `reprojection`
Server-side depth reprojection: every frame carrying depth is
reprojected from its known generation pose to the latest predicted
pose before encode.

### `enabled`
Default value: `true`

Master switch for the warp. `false` serves frames straight (no
reprojection).

### `source-reverse-z`
Default value: `true`

Depth encoding of the submitting app: `true` = reverse-Z (X-Plane 12
Vulkan). Normal-Z apps set `false` explicitly.

### `gpu-priority`
Default value: `high`

Device-global queue priority for the compositor, transfer and encode
queues: `high` prefers our submits when the app saturates the GPU (the
only lever that arbitrates across our Vulkan device and the app's).
`medium` requests nothing (driver default). `realtime` additionally
needs OS privilege (CAP_SYS_NICE) and falls back to high, then
default, when denied — the won tier is logged at startup. Silently
default when the driver lacks support.

### Example
```json
{
	"reprojection": {
		"enabled": true,
		"source-reverse-z": true,
		"gpu-priority": "high"
	}
}
```

## `publish-service`
Default value: `avahi`

How to publish the service over the network, `avahi` or null.

If set to null, service will not be published and address has to be entered manually on the headset.

## `openvr-compat-path`
Default value: unset

Provides the path to the directory of an OpenVR compatibility tool (such as OpenComposite).

If unset, WiVRn will autodetect the path of such a tool as usual (see [the SteamVR guide](./steamvr.md)).

If set to an null, WiVRn will not manage the OpenVR configuration.

## `hid-forwarding`
Default value: `false`

Only available when the `uinput` kernel module is loaded and the user has write access.

Mirrors input devices forwarded from the headset (keyboard, mouse, gamepad) to `uinput`
devices, for applications that do not read them through OpenXR. Which devices are forwarded is
chosen on the headset.

A forwarded gamepad is also exposed as a native OpenXR gamepad at `/user/gamepad`, which needs
no permission and is always available. Only the digested controller state is forwarded
(buttons, two sticks, analog triggers and a d-pad), so device specific features such as gyro,
adaptive triggers or the touchpad are not available.

## `debug-gui`
Default value: `false`

Only available when built with `WIVRN_FEATURE_DEBUG_GUI`.

Enables the Monado debug gui.

## `use-steamvr-lh`
Default value: `false`

Only available when built with `WIVRN_FEATURE_STEAMVR_LIGHTHOUSE`.

Enables the driver to load SteamVR Lighthouse devices.

## `lh-max-extrapolation`
Default value: unset

Only available when built with `WIVRN_FEATURE_STEAMVR_LIGHTHOUSE`.

Maximum time in milliseconds that poses may be extrapolated ahead for SteamVR Lighthouse devices.

## `lh-stick-deadzone`
Default value: `0`

Only available when built with `WIVRN_FEATURE_STEAMVR_LIGHTHOUSE`

Applies a deadzone to joysticks on SteamVR controllers (e.g. Index).

## `port`
Default value: `9757`

Change the TCP/UDP port used for the connection.

## `hostname`
Default value: unset

If set, overrides the name displayed in the server list.

## `fiducials`
Default value: unset (feature disabled)

Quest fiducial tracking. Each entry resolves its markers to one 6DoF
frame, smoothed with its own tuning; frames are reported in the
headset Passthrough tab.

The Quest runtime tracks **QR codes** (its AprilTag capability
is not exposed to third-party apps), so each marker is identified by
the exact decoded QR payload string. The server sends the map to the
headset on connect.

Each `fiducials` entry has:

- `id`: stable key linking config, protocol and headset UI (required,
  must be unique)
- `tag`: display-only label shown in the headset status UI (optional;
  renaming never affects tracking or linkage)
- `static`: `true` (default) or `false`. Maps to the runtime's
  `optimizeForStaticMarker`: keep `true` for a stationary rig, set `false`
  for a moving reference marker. Toggling recreates the client spatial
  context (brief tracking hitch).
- Resolver/smoothing tuning (all optional, filter auto-adapts):
  `sigma-k` (default `3`, outlier tolerance), `euro-min-cutoff`
  (default `0.4`), `euro-beta` (default `0.07`),
  `hold-inner-mm`/`hold-inner-deg` (defaults `1`/`0.1`, sub-threshold
  residuals hold; outer knees are 5x inner internally).
- `markers`: list of markers resolving to the one fiducial frame, each with:
  - `marker-data`: exact QR payload string to match, byte-for-byte (required)
  - `marker-size-m`: physical marker size in meters (required for pose
    scale; measure the printed QR's outer edge)
  - `position`: `[x, y, z]` tag center in fiducial meters, i.e.
    where the tag sits on the board (default `[0,0,0]`). Position
    edits move along world axes regardless of angles.
  - `orientation`: `[rx, ry, rz]` tag orientation relative to fiducial
    axes, in degrees (default `[0,0,0]`). Orientation tweaks pivot
    about the tag (the tag stays glued). Fixed-frame rotations about
    X, then Y, then Z, so single-axis values do the obvious thing
    (e.g. `[0,90,0]` yaws 90°)

Marker offsets are authored tag-in-board and inverted once at load —
never hand-invert in config. (Identity case: negate position.)
Poses compose as
`fiducialPose = observedMarkerPose * invert(markerOffset)`.
Print the QR encoding exactly the `marker-data` string, e.g.
`qrencode -o marker11.png -s 10 "wivrn:11"`. Matching is exact and
case-sensitive; any other QR code in view is reported in the headset
Passthrough tab (red) to help you copy it verbatim into the config.

## `suppress-meta-boundary`
Default value: `true`

Suppress the Meta (OS) boundary while WiVRn passthrough is shown.
Pushed to the headset on connect (there is no in-headset toggle for
this). This is a request: the runtime may refuse, and the boundary is
always shown in opaque VR regardless. The headset logs which verdict
the server sent on every map receipt.

## `passthrough-until-lock`
Default value: unset (disabled)

Full flat passthrough until the named fiducial locks. Set to a fiducial
`id` from this config; any other value (including unset) disables it.
The first live solve for that fiducial latches and never resets on loss
(map change clears it). While waiting, the flat feed saturates to
fullscreen through the same veil floor as boundary zones (no geometry);
combines with zones as OR: either source can demand passthrough.
Headset removal (user-presence loss) leaves the lock untouched, so
remount goes straight back into VR; placements hold their last pose
while fiducial filters reseat smoothly on the next sighting (next solve
taken as truth, glided to without popping).

## `passthrough-fade-time`
Default value: `2` (seconds)

Smoothstep fade from full passthrough to VR once the
`passthrough-until-lock` target latches. Applies to that transition
only — boundary zones and everything else are untouched (zones still
hold passthrough independently via min-combine: either source can
demand it). `0` releases instantly.

A fiducial with several markers fuses them into one 6DoF frame: every
sighted marker votes its board hypothesis (observed pose composed with
its offset), and a robust estimator (Cauchy IRLS with leave-one-out
outlier selection at 3+ votes) collapses the votes into one pose, which
is then temporally smoothed. A lone marker passes through directly; a
single jumping vote is held at the last good pose instead of displayed
(the K1 jump guard). The Passthrough tab shows the per-tag weights,
residuals, range/incidence/settle inputs and the filter state for each
fiducial, and the server log carries the same detail (`board solve:`,
`board tags:`, `board pop:`, `board flip:` lines).

The Passthrough tab also hosts the QR debug overlays (off by default):
green boxes on configured codes, red on anything else sighted, RGB axes
tripods, orange fused-corrected boxes, and board origin/smoothed
tripods, all with a 3s stale fade. `Marker window cutout` punches a live
passthrough window at every sighted code through the flat mask (works
with no map: enable overlays and every code in view reports red, ready
to copy verbatim into the config).

> [!NOTE]
> A QR payload is a single identity: printing the *same* payload twice
> does not create two trackables. The runtime reports one jumping track
> for identical prints. For multi-location coverage, print distinct
> payloads and configure one fiducial per location.

## `passthrough`
Default value: unset (no objects)

Behavior objects placed from solved fiducial frames. Each entry
references fiducials by id and places one instance per visible
pairing (never fused); objects persist once placed and follow every
frame, live or held.

Each `passthrough` entry has:

- `type`: behavior type (required). `"3d-passthrough"` renders the
  model as a surface-projected passthrough cutout; unknown types are
  skipped with a warning. `"3d-passthrough-reversed"` reverse-cuts
  the model from every mask layer instead (game video shows through
  the silhouette, depth-tested against the windows covering each
  pixel). `"3d-depth-passthrough"` reveals the flat feed where the
  runtime depth map resolves *inside* the model volume (depth-tested
  volume, feathered like a normal window): the volume rasterizes its
  far side at 320px, strict `sensor < render` flattens to a binary
  mask, and it upscales+MAXes over every layer — including reversed
  cutthroughs — before the shared erode/blur. Per-eye display MVPs
  keep it in sync with VR content; pose-delay correction is future
  work, so expect a texel of swim during fast motion. The headset
  Passthrough tab offers a Rotation reprojection toggle (default on):
  the sensor tap is remapped through the capture <-> display relative
  rotation plus a depth-dependent lateral shift (sideways parallax;
  forward/radial and depth-value terms excluded), so the comparison
  tests the same world ray, falling back to the direct tap when off.
  A Depth time bias slider (+/-50ms) shifts the assumed content age
  to hunt a constant sensor lag (+ = older = larger correction). Meshes should be
  **closed volumes** (far side is well-defined). `"3d-boundary"` shows
  game video only *inside* the model volume and passthrough outside
  it — a Quest-guardian-style zone with adjustable size, shape and
  fade. `"3d-boundary-reversed"` inverts it (passthrough inside, game
  outside: exclusion zones around e.g. a desk). Zones combine as a
  union: game video shows iff the headset is
  inside ANY normal zone (or no normal zone is placed) AND outside ALL
  reversed zones; unplaced zones are ignored (fail-open to game, so
  tracking loss never flashes passthrough). Boundary meshes must be
  **closed volumes** (inside/outside comes from ray parity, undefined
  for open meshes); keep them simple (room-size boxes are ideal, ~5k
  triangles budget). Both eyes fade together from a single per-frame
  headset distance.
- `id`: stable identifier (required, must be unique); `tag`:
  display-only label (optional)
- `fiducial`: referenced fiducial id, or list of ids (required;
  dangling references are dropped with a warning, and an object left
  with none is skipped)
- `model-path`: path to a `.glb`/`.gltf` file on the server (optional,
  models larger than 64MB are skipped). May be a host absolute path:
  under flatpak the server reads it from outside the sandbox. Served
  to the headset on demand and cached by content hash.
- `position`: `[x, y, z]` fiducial-to-object offset in meters
  (default `[0,0,0]`)
- `orientation`: `[rx, ry, rz]` fiducial-to-object rotation in degrees
  (default `[0,0,0]`, same fixed-frame convention as markers)
- `scale`: uniform object scale, number or single-element array
  (default `1`, applied last)
- `feather-px`: passthrough window feather width in screen pixels
  (default `24`; `0` disables feathering). Mask stacks group by this
  value so each object feathers independently; applied live. Depth
  volumes reuse their feather value and feather through the same
  erode/blur chain. Not accepted for `3d-passthrough-reversed` or
  `3d-boundary(-reversed)` (warned + ignored): negatives cut every
  layer and inherit each layer's feathering, while zones fade by
  distance instead (see `transition-distance`).
- `transition-distance`: boundary fade width in meters
  (`3d-boundary(-reversed)` only, default `0.3`, applied live). The
  scene cross-fades over ± this distance around the zone surface
  (symmetric smoothstep, 0.5 at the surface); `0` is a hard step.
  Composited by lifting the flat union's mask floor, so no extra layers
  or swapchains: exiting saturates the existing flat feed to fullscreen.
- `surface-projected`: the headset projects camera imagery onto this
  object's mesh instead of the flat reconstruction feed (default
  `true`; opt out per object with `false`, plus a live in-headset
  override in the Passthrough tab). Ignored for
  `3d-passthrough-reversed`, `3d-boundary(-reversed)` or
  `3d-depth-passthrough` (warned + ignored): zones never map camera
  onto geometry, and depth volumes always reveal the flat feed.

Poses compose as
`objectPose = fiducialPose * fiducialToObjectOffset` (scale last).
