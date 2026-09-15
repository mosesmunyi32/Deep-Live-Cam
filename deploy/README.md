# Deep-Live-Cam — real-time streaming on a Runpod GPU Pod

Serves the upstream `face_swapper` frame processor as a live, networked service.
Nothing under `modules/` is modified; everything here is additive.

## Why a WebSocket and not WebRTC

The obvious design for low-latency video is WebRTC. It is not available on this
platform: **Runpod routes TCP/HTTP only and does not support UDP**, which is the
transport WebRTC media normally rides on. Working around that would mean relaying
every frame through an external TURN server over TCP/443 — a third-party
dependency, an extra network hop, and `aioice`'s weakest code path.

So frames go over a plain WebSocket instead: JPEG up, JPEG down, one frame in
flight at a time. The client paces itself to whatever the GPU sustains rather
than queueing frames into the socket.

| | WebRTC over UDP | This (WebSocket/TCP) |
|---|---|---|
| Runpod support | not available | works |
| Round-trip | ~50-120 ms | ~120-250 ms |
| Congestion behaviour | adaptive bitrate | frame dropping, head-of-line blocking |
| Moving parts | TURN, STUN, ICE, SDP | one socket |

If you later need true sub-100 ms, the answer is a host that routes UDP, not a
change to this code.

## Architecture

```
browser ──getUserMedia──▶ canvas ──JPEG──▶ WebSocket ──▶ aiohttp
                                                            │
                                                    ThreadPoolExecutor(1)
                                                            │
                                              face_swapper.process_frame
                                                            │
   <img> ◀────────── JPEG ◀────── WebSocket ◀───────────────┘
```

A single worker thread serializes GPU access; a one-slot mailbox (`LatestSlot`)
drops stale frames so latency stays bounded when inference falls behind.

## Using a phone as the camera

The control page uses `getUserMedia`, so any device that can open the page can
be the camera — including a phone. Browsers only expose `getUserMedia` in a
**secure context**, which is the whole difficulty:

| Where | Secure context? | What is needed |
|---|---|---|
| `localhost` on this machine | yes (localhost is exempt) | nothing |
| **Runpod proxy URL** | yes, HTTPS already | **nothing — just open it on the phone** |
| `http://<lan-ip>:8080` from a phone | no | TLS, below |

On Runpod this is free: open the pod URL in Safari or Chrome on the phone, pick
the camera, and it streams. No companion app, no virtual-camera bridge.

For local testing, generate a self-signed certificate and enable TLS:

```bash
./deploy/make-cert.sh 192.168.1.146       # your LAN IP

docker run ... \
  -e DLC_TLS_CERT=/app/deploy/certs/cert.pem \
  -e DLC_TLS_KEY=/app/deploy/certs/key.pem \
  -v "$PWD/deploy:/app/deploy:ro" ...
```

Then browse to `https://192.168.1.146:8080/?token=…` and accept the warning.

Two things the script handles that hand-rolled certs usually miss: iOS ignores
Common Name entirely and requires the address in **subjectAltName**, and it
rejects certificates valid for more than **825 days**. If Safari still refuses
the camera after accepting the warning, install the cert properly — mail
`cert.pem` to yourself, open it, Settings → *Profile Downloaded* → Install, then
Settings → General → About → **Certificate Trust Settings** and enable full
trust for it.

`deploy/certs/` is gitignored. Do not commit private keys.

An alternative that avoids certificate warnings altogether is a tunnel with a
real certificate — `cloudflared tunnel --url http://localhost:8080` or a
Tailscale funnel — at the cost of routing your video through a third party.

Bridging apps (Iriun, DroidCam) also work: they present the phone as a
`/dev/video*` device that appears in the camera selector. That adds an encode
and decode hop for no benefit over just opening the page on the phone.

## Choosing the camera

The control page lists every `videoinput` device and passes the chosen one to
`getUserMedia` as an exact `deviceId`. Changing the selector while streaming
swaps the local capture only — the session, the socket and the loaded source
face all survive.

Two browser behaviours the selector has to work around:

- **Device labels are hidden until permission is granted.** Before the first
  successful `getUserMedia` the list shows `Camera 1`, `Camera 2`; the page
  re-enumerates immediately after Start to replace those with real names.
- **An exact `deviceId` fails hard** if that camera is busy or unplugged, so a
  failure falls back to the default device with a visible notice rather than
  leaving a dead preview.

`⟳` re-scans on demand, and the page also listens for `devicechange`, so
plugging a webcam in mid-session updates the list by itself.

## Output view and OBS

The control page at `/` is for driving the swap. The swapped frames are also
published separately, so the output can live in its own window or go straight
into OBS without the surrounding UI:

| Path | What it is |
|---|---|
| `/output?token=…&s=<session>` | Output-only page — black background, frame scaled to fit |
| `/output?token=…&s=<session>&bare=1` | Same, with the error strip suppressed. Use this for OBS |
| `/stream.mjpg?token=…&s=<session>` | Raw `multipart/x-mixed-replace` MJPEG |

**OBS:** Sources → **+** → **Browser** → paste the `bare=1` URL, set 960×540,
and uncheck *Shutdown source when not visible* so it keeps streaming while you
are on another scene. The control page's **Copy OBS browser-source URL** button
produces exactly this URL with the token and session filled in.

### OBS and TLS

OBS embeds CEF, which **rejects a self-signed certificate and offers no way to
accept one**. So once TLS is enabled for a phone, OBS can no longer read the
HTTPS origin. The server therefore listens twice:

| Port | Scheme | For |
|---|---|---|
| `8080` | HTTPS | the phone — `getUserMedia` needs a secure context |
| `8081` | HTTP | OBS on this machine — CEF cannot be given a cert to trust |

`DLC_PLAIN_PORT` only opens when TLS is on. Publish it on loopback so it never
reaches the network:

```bash
-p 8080:8080 -p 127.0.0.1:8081:8081
```

Note the bind inside the container is `0.0.0.0`, not `127.0.0.1`: Docker's port
forwarder arrives over the bridge interface, so a container-loopback bind is
unreachable and the port simply refuses connections. The restriction comes from
the **publish address**, `-p 127.0.0.1:…`. Running outside a container, set
`DLC_PLAIN_HOST=127.0.0.1` instead.

The copy button accounts for this and hands out the `http://127.0.0.1:8081`
origin, while **Open output window** stays on the HTTPS origin so it reuses the
certificate you already accepted.

### Which session the output follows

Without `s`, the output follows whichever session **most recently produced a
frame** — not the most recently opened one. With a control page idling on the
laptop and a phone actually streaming, "newest session" picks the wrong one.
`GET /sessions` lists the live sessions with frame counts and idle times if you
want to pin one explicitly.

`s` selects which session to mirror. Omit it and the output follows the most
recent session, which is what you want with a single streamer; it matters only
when `DLC_MAX_SESSIONS` is above 1.

A separate `<img>` consuming MJPEG is used rather than reusing the session
WebSocket, because the output window is a different client from the one sending
camera frames. The page reconnects with backoff if the stream drops, so OBS
recovers on its own across a server restart.

## Startup order

The model is loaded during `on_startup`, which completes **before the listener
opens** — the log shows `warm-up complete` immediately followed by `listening
on`. A client therefore cannot connect to a server whose model is not ready;
`/ws` refuses with 503 and `/healthz` reports `initializing` until it is.

The control page connects its WebSocket on **page load**, not on Start. The
model list, current settings and face library all arrive over that socket, so
opening it late left every control showing "loading…" until the camera was
already running, and made it impossible to choose a model or stage faces
beforehand. Start now only opens the camera; Stop only closes it, keeping the
session, its face library and its settings intact.

A dropped socket reconnects with backoff, and distinguishes "server still
loading" from "server gone" by checking `/healthz`, since a page opened during
startup would otherwise report a connection failure.

## Hair, caps and head shape

`inswapper` preserves them already — it is not a setting. The swap is confined
to an ellipse in aligned-face space:

```python
axes = (int(w * 0.44), int(h * 0.44))     # _create_elliptical_mask
```

Only that region is replaced. Hair, headwear, ears, jawline edges and
background all come from the target footage. This is exactly why it is *not* a
head swap: models like BFS replace the entire head including hair, which is the
opposite behaviour.

Blend controls that genuinely affect the result:

| Control | What it does | Cost |
|---|---|---|
| `opacity` | Blend strength against the original face | free |
| `mouth_mask` | Keeps the real mouth from your footage | small |
| `sharpness` | Sharpens the swapped region | small |
| `poisson_blend` | Seamless-clone the edges — best join | **~10-30 ms/frame** |
| `enable_interpolation` | Blends with the previous frame, reducing flicker | small |
| `skin_tone` | Pulls the swapped face's colour back towards your own | ~1 ms |

### Keeping your own complexion

`inswapper` brings the source's identity **and** their skin tone, which is what
makes a swap read as pasted-on when the two were lit differently. **Keep my skin
tone** (`skin_tone`, 0-1) pulls the colour back towards the face being replaced
while leaving the geometry - the identity - alone. Measured against the real
face's mean LAB, with a source of a different complexion:

| strength | distance from the real face |
|---|---|
| 0.0 | 2.0 |
| 0.5 | 1.0 |
| 1.0 | 0.6 |

Three decisions worth knowing, because the obvious version of this looks wrong:

- **It happens in aligned space, on the 128x128 `bgr_fake`, before paste-back.**
  That is the only point where the swapped pixels and the ones they replace are
  in exact correspondence, so the two sets of statistics describe the same
  features rather than two different framings.
- **Statistics are sampled through an eroded ellipse**, not the whole crop. The
  aligned square contains background at its corners and a feathered rim that is
  half background; averaging those in measures the room, not a skin tone.
- **The per-channel scale is clamped to 0.6-1.6.** Matching the standard
  deviation as well as the mean is what makes tone match under different
  lighting, but an unclamped scale posterises skin when the target face is in
  harsh light.

**`apply_color_transfer` is still unused, and still should be.** It is defined in
`face_swapper.py` and matches over an entire image, which is precisely the
whole-crop average described above. Exposing *it* would be the control that
silently does the wrong thing; `deploy/skin_tone.py` is the one wired in.

Nothing under `modules/` is modified for this. The correction is inserted by
wrapping the swapper model's own `get()`, which is where upstream hands back the
swapped face and its affine.

### Making the swap belong to the footage

A swapped face reads as fake for a specific, measurable reason: it does not
match the frame it is sitting in. `inswapper` returns a 128x128 patch that is
smoother, sharper, cleaner and more evenly lit than the camera's own pixels
around it. Everything else in frame - the hands, the background, the jewellery,
the sensor noise, the lens distortion, the handheld jitter - is already real,
because it came out of a real camera. It needs nothing done to it, and adding
film grain or blur over the top only degrades genuine footage.

So the **Realism** controls do not invent texture. Each pass measures the face
being replaced and hands what it finds to the face replacing it, in the
swapper's aligned space where the two correspond pixel for pixel:

| Pass | What it takes from the real face |
|---|---|
| `detail` | High frequencies above ~1.6 px: pores, stubble, hair, and the camera's own noise |
| `light` | Low-frequency luminance, so highlights and shadows fall where the room put them |
| `softness` | The lens's focus, blurring the swap down to it — never sharpening up, which would invent detail |
| `motion` | Frame-to-frame displacement by phase correlation, applied as a line kernel: the shutter's own smear |
| `grain` | The gap between the plate's high-frequency energy and the swap's, as monochrome noise |
| `fringe` | A sub-pixel radial channel offset — the colour fringing every lens has and no swap does |

Measured on a swap against its own plate: skin texture 7.18 → 7.87 where the
real face reads 8.18, and **2 ms per frame** for all six passes. Order matters
and follows the light: texture, then lighting, then lens, then shutter, then
sensor — so blur lands on detail rather than detail landing on blur.

Two deliberate refusals. **Softness never sharpens**, because raising detail the
camera did not record is the exact failure this exists to avoid. And **grain
defaults to zero**, because `detail` already carries the plate's real noise
across; adding more on top reads as dirt.

*Match the camera* sets the preset. It applies to face swap only — portrait mode
replaces the whole head, so there is no surrounding face to match against.

### Keeping the original face instead

Two controls already do this, and they answer different questions:

| Control | Effect |
|---|---|
| `opacity` | Blends the swap with your real face. At 0.5 you get half of each; at 0 the swap is off |
| `mouth_mask` | Keeps your real mouth and its movement, swapping everything else |

Together with `skin_tone` these cover "make it look like me": your complexion,
your mouth, and as much or as little of the swapped identity as you want.

### The real gap: occlusion

Nothing here handles something crossing *in front of* the face — a cap brim
low on the forehead, a hand, a hair strand over the cheek. The elliptical mask
is geometric, not content-aware, so the swapped face is painted over the
occluder. Fixing that needs a face-parsing/segmentation model (BiSeNet-class)
baked in and applied per frame, which is real work and real latency. The eye and
eyebrow masks in `face_masking.py` are not wired into this pipeline either.

## Portrait mode

Portrait mode is the answer to "swap the hair too". It does not replace a region
in your frame at all — it animates the *chosen picture* with your expression and
head motion, so the output is that person's whole head: hair, ears, jaw, skin,
and their background too. Your camera supplies the performance, not the pixels.
That framing is the trade: you cannot have their hair **and** your background.

### The controls, and which ones are free

Every parameter is read inside the pipeline's per-frame `_run`, so a change
lands on the next frame without rebuilding the ~1.5 GB of ONNX sessions:

| Control | What it does |
|---|---|
| `animation_region` | `all`, or restrict to `exp` / `pose` / `lip` / `eyes` — hold the head still and animate only the face, or only the mouth |
| `driving_multiplier` | Scales motion away from the source pose. Above 1 exaggerates, below 1 damps |
| `flag_relative_motion` | Off drives the absolute pose instead of the change since the reference frame |
| `flag_eye_retargeting` / `flag_lip_retargeting` | Match the picture's eye and mouth opening to yours |

Three others — `flag_pasteback`, `flag_stitching`, `flag_normalize_lip` — are
**not** free. `prepare_source` bakes their consequences into each face's
`src_info`: the paste-back mask is built there and lip normalisation is applied
there. Changing one makes every prepared face stale, so the server drops and
re-prepares them, about a second each. The control page says so rather than
appearing to stall.

The controls stay usable in swap mode instead of being hidden with the mode they
belong to: staging the look and then switching is the normal order, and a
control that only exists after the switch cannot be set up beforehand.

**Re-zero pose** exists because LivePortrait measures motion against the pose it
saw first. Whatever you were doing at that instant became "neutral" — a head
turned away, or a mouth left open, stays baked in as the rest position. The
button re-takes it from your next frame.

### Restaging the picture: background removal

Portrait mode animates the source picture and pastes the head back into it, so
everything around that head goes out on the stream — a screenshot's window
chrome, a watermark, the room it was taken in. **Background** replaces it:

| Backdrop | What you get |
|---|---|
| *keep the picture as it is* | default, no matting, no cost |
| *solid colour* | the people, on a colour of your choice — green for a chroma key downstream |
| *my own image* | the people, over an uploaded picture, scaled to fill rather than letterboxed |
| **Head only** | crops to the head — hair included — before any of the above |

This is cheap because it is a **still-image** problem. The source is staged once
when you add a face, so `u2net_human_seg` (the person-segmentation half of
U^2-Net) runs about **1.2 s per picture on CPU** and never appears on the frame
path. It is CPU by default, and not merely as a fallback: it would otherwise sit
beside the LivePortrait pipeline on a 4 GB card that has no room for it.
`DLC_MATTING_DEVICE=cuda` moves it on a card with headroom.

The mask is cached per face and survives a change of backdrop, because the
expensive half does not depend on what goes behind it. Changing colour or
picture is instant; the first cut-out is not.

Three things the naive version got wrong, all visible in the output before they
were fixed:

- **A halo of the old background.** Edge pixels are part subject, part
  backdrop; composited onto green they fringe green. `refine()` raises the
  mask's contrast and erodes it a pixel.
- **"No usable face" on a head crop.** A head cropped out of a screenshot can
  be 130 px across, and LivePortrait's detector finds nothing in that — while
  the message blames the photo. Crops are enlarged to 512 px first.
- **A strip of UI that survived the matte.** Segmentation keeps anything
  person-shaped, and the strip touched a shoulder, so it was one connected blob
  with the person — component isolation could not separate them. Head-only
  staging bounds the mask geometrically, with a soft ellipse centred slightly
  *above* the face box, since what has to be kept above the brow is hair.

Background is **per session**, unlike the mode and the model: it belongs to the
picture you staged, the way the face library does. And it applies to portrait
mode only — face swap takes nothing from the source but the face embedding, so
restaging there would buy a matte for nothing.

Note that `models/u2net_human_seg.onnx` is excluded from `GET /models`. It lives
there because that is where weights go, but it is not a swapper, and offering it
in the selector would only invite an error.

### GridSample3D: why stock onnxruntime cannot load the weights

`warping_spade` is exported with **GridSample3D**, a custom operator that only
the patched onnxruntime FasterLivePortrait ships knows about. Stock ORT refuses
the graph outright:

```
INVALID_GRAPH : No Op registered for GridSample3D with domain_version of 16
```

The fix does not need a patched runtime. ONNX **opset 20** added 5D support to
the standard `GridSample`, so the Dockerfile retargets the two nodes onto it
(`mode: bilinear` becomes opset 20's `linear`) and converts the graph. Same
semantics, and it runs on the unmodified `onnxruntime-gpu` already installed.

This is also why the build now opens an ORT session against *every* LivePortrait
weight. Importing the pipeline and checking the files exist both passed while
this was broken — the failure only appeared on the first animated frame, which
is an expensive place to discover it.

### VRAM: the two model sets do not coexist

Measured on the 4 GB T1000, with the swapper resident, LivePortrait loads and
then dies partway through preparing a face — 50 MB free, trying to allocate 20.
So **switching mode unloads the other side**: entering portrait mode releases
the swapper, leaving it reloads the swapper and drops the pipeline along with
every prepared source, since those hold device tensors of their own. The cost is
a reload of a few seconds on the way back.

ORT's CUDA defaults were the other half. An EXHAUSTIVE convolution search with
max cuDNN workspace and a doubling arena cost about 1.1 GB of pure headroom;
`deploy/lp_engine.py` injects `HEURISTIC`, `cudnn_conv_use_max_workspace=0` and
`kSameAsRequested` around pipeline construction, which brings the load down to
1.7 GB and the peak to ~2.8 GB. That fits, with the desktop's ~760 MB alongside.

Unloading also has to reach past the vendored code. `clean_models()` empties the
pipeline's own dict, but the predictor is a **singleton keyed by model path** — a
class-level cache built to stop the same weights loading twice, which outlives
the pipeline that created it. With it in place the unload freed 10 MB of 3 GB
and the switch back to swap died on a 36 MB allocation; clearing
`OnnxRuntimePredictorSingleton._instance` releases the sessions for real, and the
next load rebuilds it. Measured: **2187 MB free after the unload, against 154 MB
before the fix.**

`torch.cuda.empty_cache()` is needed for the same reason on the torch side —
freed blocks go back to torch's cache, not to the driver, so the swapper's
reload would otherwise still find nothing.

An allocation failure that happens inside the vendored `prepare_source` is
caught there and returned as `False`, which reaches the page as "no usable face
in this image" — sending you to find a better photo when the problem is memory.
`_raise_if_oom` checks free VRAM before reporting that, and says so instead.

### Two traps in the mode switch

**The mode is process-wide, and a test leaves it that way.** Switching to
portrait to try something and walking away leaves every client — including a
phone that connects an hour later — animating a picture instead of swapping.
"Face swap stopped working" is what that looks like from the page. `GET /models`
reports the current mode; the control page shows it on connect.

**The swapper must be reloaded explicitly, never lazily.** Upstream's
`get_face_swapper()` reloads on demand when `FACE_SWAPPER` is None, but it picks
the **fp16** model whenever `torch.cuda` is importable — and torch is now
installed, because LivePortrait needs it. fp16 measured six times slower on this
card. Portrait mode sets `FACE_SWAPPER` to None, so the swap path would have
walked straight into that on the next frame; it reloads the selected model
instead.

### A driving frame with no face

`run()` reports "nothing face-shaped in this frame" by returning a **tuple of
Nones**, not by returning None — so it is only visible after unpacking. It
happens constantly: a phone camera still warming up, a head turned away, a room
too dark for the detector. Portrait mode answers with the staged picture,
holding still, and says so over the socket at most once every five seconds,
because a portrait that cannot see you looks exactly like a broken one from the
other end.

Note *when* that can happen. The pipeline detects a face only on the first frame
after a reset; from then on it tracks the landmarks it already has, which is
what keeps it affordable. So a face that leaves the frame mid-stream does not
report a miss — the head simply keeps moving on stale landmarks until you press
**Re-zero pose**, which is also the fix if it ever locks onto nothing.

Getting this wrong cost a session per miss. A colour-space fix put `cvtColor`
in front of the None check, and `cv2.cvtColor(None, …)` raises the *same*
`!_src.empty()` assertion an empty array does — so the first frame the detector
missed killed the stream, with an error that pointed at colour conversion rather
than at face detection.

### Eye retargeting crashed on every frame

`calc_combined_eye_ratio` reshapes the driving ratio to `(1, 1)`, which is what
the video path passes. The realtime path passes **both eyes** — a `(1, 2)` array
— so switching eye retargeting on raised `cannot reshape array of size 2 into
shape (1,1)` and killed the session. The vendored pipeline now averages the two,
since the retargeting network takes a single driving value. The lip path already
passed a `(1, 1)`, but it got the same coercion so the two cannot diverge later.

### Speed, and why no amount of tuning fixes it here

**~680 ms per frame on the T1000**, against 184 ms for a swap. Profiled per
model, it is not spread around:

| stage | p50 |
|---|---|
| **warping_spade** | **632 ms** |
| motion_extractor | 24.9 ms |
| face_analysis (first frame only) | 22.2 ms |
| landmark | 12.3 ms |
| stitching | 0.2 ms |

One model is 94% of the frame, and it is compute-bound. Counting the graph's
convolutions gives **617 G MACs — 1.23 TFLOP per frame**. A T1000 Max-Q peaks at
about 2.6 TFLOPS fp32, so the floor at *100% of theoretical peak* is **474 ms**.
Measured 632 ms is roughly 75% of peak, which is close to what a real kernel
achieves. There is no factor of ten hiding anywhere.

Confirmed from the other end too: under load the card sits at 1440-1515 MHz
against a 1530 MHz maximum, 100% utilisation, 39 W, 62 °C. Nothing is throttled
and nothing is idle — it is simply a small GPU running a large generator.

What that means in practice: **portrait mode is a big-GPU feature.** Same
1.23 TFLOP against an A40's ~37 TFLOPS is ~33 ms at peak, so expect **20-25 fps**
there; a 4090 more. Measure rather than trust the ratio.

Things tried, measured, and rejected — recorded so nobody spends the afternoon
again:

| Idea | Result |
|---|---|
| Softmax over axis 1 rewritten as transpose → softmax → transpose | Bit-identical output, **slower** (711 vs 670 ms) |
| Same softmax decomposed into ReduceMax/Sub/Exp/ReduceSum/Div | 708 ms, and 3.5e-06 of drift for the privilege |
| Four ORT memory/algo configurations, EXHAUSTIVE included | 676-686 ms. All within noise of each other |
| Checking for CPU fallback | None. All 828 nodes on CUDA |
| Detecting the driving face every frame instead of tracking | Tracked and detected landmarks differ by **0.2 px**; costs 20 ms for nothing |
| TensorRT EP | Not installed — `libnvinfer` absent, ~2 GB to add. Worth it on a card where the result would be real-time; not on this one |

fp16 is the one lever left untested, and deliberately so: TU117 has no tensor
cores, and fp16 measured **6x slower** for the swapper here. On an Ampere card it
is the obvious next thing to try.

## Changing face during a stream

Faces are a per-session **library**, not a single slot. Add several images and
switch between them by clicking a thumbnail or pressing number keys `1`-`9`.

Switching costs **0.3 ms** measured, because the expensive part — detecting and
embedding the face — happens once at upload. Changing the active face after
that is a dictionary lookup, so it does not interrupt the stream or drop a
frame. `DLC_MAX_FACES` caps the library, default 12.

Every uploaded image goes through the NSFW screen and face detection
independently, so an image with no detectable face is rejected at upload rather
than silently producing unswapped frames later.

## Which model for live streaming

`inswapper` — and in practice nothing else. It is one forward pass of a small
128x128 network, which is what makes ~30 fps achievable on a mid-range GPU.

Diffusion-based swappers (BFS, and anything built on FLUX, Qwen-Image-Edit or
Krea) produce visibly better single images and **cannot stream at all**: they
run 20-50 denoising steps through a multi-billion-parameter transformer, which
is seconds per frame against a ~33 ms budget. That gap is architectural, not a
tuning problem. They are worth using for stills and offline video; they are not
an option for live.

## Adding models

`GET /models` lists **every `.onnx` in `models/`**, so adding one is a file
copy — no code change:

```bash
cp my_swapper.onnx models/
# models/ is bind-mounted, so it appears in the selector on reconnect
docker run ... -v "$PWD/models:/app/models:ro" ...
```

To ship it in the deployed image instead, add a download line beside the
existing ones in `deploy/Dockerfile` so cold pods still fetch nothing.

Only **inswapper-architecture** models actually load. Selection validates that
what came back is a swapper — insightface will happily return a detector or
recogniser for the wrong file, and without the check that surfaces much later
as a confusing error inside `swap_face`. Incompatible files are listed but
rejected with the reason, rather than hidden, so a model that does not work
says why. Filenames are basename-checked; path traversal is refused.

Face enhancers (`face_enhancer`, GPEN 256/512) are a different processor
upstream and are not offered here: their weights are not in the image, and at
their per-frame cost they are not viable for a live stream. They are worth
baking in for still or offline video work.

## Lighting and camera controls

Two independent layers, because device support is unreliable:

- **Lighting** (brightness / contrast / saturation) is applied **on the server,
  before the swap**. Before rather than after on purpose: poor lighting costs
  detections, and a frame the detector misses cannot be swapped at all. Each
  step is skipped when it would be a no-op, so the default path costs nothing.
  This works on every device.
- **Camera controls** are the device's own, built at runtime from
  `track.getCapabilities()` — zoom, exposure, white balance, torch, focus mode
  and so on. Nothing is hardcoded, because support is wildly uneven: Chrome on
  Android exposes a lot, **iOS Safari usually exposes none at all**. When a
  device reports nothing adjustable, the row says so and points at the
  server-side lighting controls.

So on an iPhone, expect the native row to be empty and use Lighting instead.

## Choosing the model

The control page's **Model** selector switches models at runtime.

Switching is **process-wide, not per-session** — these are module-level globals
in the upstream processor, so a change affects every connected client. The same
applies to the *All faces*, *Mouth mask*, *Colour correct* and *Opacity*
controls.

The switch invalidates the recorded CUDA graph as well as the cached session.
That matters: the graph is recorded against one model's input and output
buffers, so reusing it after a model change would silently keep replaying the
previous model.

Measured through the running server on the T1000:

| model | end-to-end p50 |
|---|---|
| fp32 `inswapper_128.onnx` | 181 ms |
| fp16 `inswapper_128_fp16.onnx` | 881 ms |

fp16 loses badly here because TU117 has no tensor cores — see the fp16 note
under GPU sizing. Try the selector on the deployment GPU before assuming either
way. The face enhancers upstream offers (`face_enhancer`, GPEN 256/512) are not
selectable because their weights are not baked into the image, and at their cost
per frame they are not realistic for a live stream anyway.

## Testing without deploying

`deploy/` is bind-mounted into the container, so editing the server, the engine
or the page needs a restart, not a rebuild:

```bash
docker restart dlc-local          # ~30 s, same URLs
```

An image rebuild is only needed when a dependency or a baked weight changes.
Pushing rebuilds the ghcr image, which matters for a pod and for nothing else.

Two end-to-end checks run against a live server over its real protocol:

```bash
python deploy/smoketest.py --url http://127.0.0.1:8081 --token <token>
python deploy/portrait_smoketest.py --token <token>
```

The first covers the swap path and reports round-trip latency. The second covers
everything portrait mode added — both mode switches, the retargeting toggles,
restaging onto a colour, an image and a head crop, a rejected bad colour, and a
driving frame with no face in it. Every one of those is there because it broke
in a way the build's import check could not see.

**A test leaves the mode where it put it.** The mode is process-wide, so a
portrait test that ends in portrait mode leaves the next person to connect
animating a picture and wondering why nothing swaps. Both scripts end in swap
mode deliberately.

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `DLC_AUTH_TOKEN` | — | **Required.** Server refuses to start without it. |
| `PORT` | `8080` | Listen port; must match the exposed HTTP port. |
| `DLC_MAX_SESSIONS` | `2` | Concurrent streams before returning 429. |
| `DLC_JPEG_QUALITY` | `80` | Return-path JPEG quality. |
| `DLC_NSFW_FILTER` | `1` | Screen uploaded source faces. |
| `DLC_MATTING_DEVICE` | `cpu` | Where background removal runs. `cuda` needs VRAM the pipeline may want. |
| `DLC_MANY_FACES` | `0` | Swap every detected face, not just one. |
| `DLC_MOUTH_MASK` | `0` | Preserve the target's mouth region. |

`DLC_AUTH_TOKEN` is mandatory by design. The Runpod HTTP proxy is **public and
unauthenticated** — anyone with the pod URL reaches the service, and the pod ID
is obscurity, not access control.

## Build and push

The image is ~13 GB, so CI builds it. `.github/workflows/build-image.yml` fires
on any push touching `deploy/` or `modules/` and pushes to:

```
ghcr.io/<owner>/deep-live-cam-stream:latest
ghcr.io/<owner>/deep-live-cam-stream:sha-<commit>
```

Nothing that large has to cross your uplink. To build locally anyway:

```bash
docker build -f deploy/Dockerfile -t deep-live-cam-stream:latest .
```

The image bakes in `inswapper_128.onnx`, `inswapper_128_fp16.onnx`, `buffalo_l`
and the opennsfw2 weights, so a cold pod downloads no models.

### Registry visibility

**A public repository does not make its packages public.** They are separate
settings, and a package pushed by CI starts private regardless of the repo. Only
the package's visibility decides whether Runpod can pull. Check it directly
rather than inferring it from the repo:

```bash
TOK=$(curl -s "https://ghcr.io/token?scope=repository:<owner>/deep-live-cam-stream:pull&service=ghcr.io" \
      | python3 -c "import json,sys; print(json.load(sys.stdin)['token'])")
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $TOK" \
  https://ghcr.io/v2/<owner>/deep-live-cam-stream/manifests/latest
```

`200` means anonymous pulls work. `403` means it is private.

Two ways to make the pod able to pull:

- **Make the package public** — <https://github.com/users/OWNER/packages/container/deep-live-cam-stream/settings>
  → Change visibility → Public. No credential needed, and `--registry-auth-id`
  can be dropped from the deploy command. This publishes a ready-to-run
  face-swap image under your name.
- **Keep it private** and register a credential: a GitHub PAT with
  `read:packages` via `runpodctl registry create`, passed as
  `--registry-auth-id`. See Deploy below.

Note on precision: the runtime uses fp32, because upstream gates the fp16 model
behind `torch.cuda` and torch is deliberately not installed. Do not "fix" that
by adding torch — see the fp16 measurement under GPU sizing first.

## Deploy

The package is public, so no registry credential is required:

```bash
runpodctl pod create \
  --name deep-live-cam \
  --image ghcr.io/<owner>/deep-live-cam-stream:latest \
  --gpu-id "NVIDIA A40" \
  --gpu-count 1 \
  --container-disk-in-gb 40 \
  --ports "8080/http" \
  --env '{"DLC_AUTH_TOKEN":"<token>"}' \
  --wait
```

If you make the package private later, register a credential with
`runpodctl registry create --name ghcr-dlc --username <user> --password <PAT
with read:packages>` and pass the resulting id as `--registry-auth-id`. Do that
in your own shell — a PAT on a command line lands in history.

Two things that are easy to get wrong:

- `--container-disk-in-gb` defaults to **20**. The image is 4.71 GB compressed
  but ~13 GB unpacked on disk, plus the writable layer, so 20 is not enough.
  40 leaves headroom.
- `runpodctl create pod` (the old argument order, `--imageName`/`--gpuType`) is
  deprecated. The current form is `runpodctl pod create` with `--image` and
  `--gpu-id`, and `--env` takes a JSON object rather than `KEY=value`.

Then open:

```
https://<pod-id>-8080.proxy.runpod.net/?token=<token>
```

Readiness is `/healthz` — it returns 503 until the models finish loading, so
poll it rather than trusting the console's green "Running":

```bash
until curl -sf https://<pod-id>-8080.proxy.runpod.net/healthz; do sleep 5; done
```

### GPU sizing

Measured with `deploy/bench.py` on a Quadro T1000 Max-Q, fp32, 640 px:

| stage | p50 |
|---|---|
| detect | 48.6 ms |
| swap (inference) | 135.3 ms |
| post-processing | 0.8 ms |
| **total** | **185.7 ms — 5.4 fps** |

End-to-end through the WebSocket measured 183 ms, so JPEG encode/decode and the
socket cost nothing measurable. Throughput is bound entirely by `inswapper`
inference, which is GPU-bound and so scales with the card. A T1000 Max-Q is
about 2.6 TFLOPS fp32. An A40 is roughly fourteen times that, which should put
a single stream comfortably past 30 fps. Verify on the pod rather than trusting
the extrapolation:

```bash
python deploy/bench.py --frames 100 --width 640
```

Note what this rules out: detection is not the bottleneck, post-processing is
free, and the transport is free. If a stream is slow, the swap model is the
only thing worth attacking.

#### Which GPU

`inswapper_128` needs about 2 GB of VRAM, so memory is never the constraint —
compute is. Live prices and stock from `runpodctl gpu list`:

| GPU | VRAM | $/hr | stock | vs T1000 |
|---|---|---|---|---|
| **A40** | 48G | **0.44** | High | ~14x |
| RTX A6000 | 48G | 0.53 | Low | ~15x |
| RTX 3090 | 24G | 0.50 | Low | ~14x |
| RTX 4090 | 24G | 0.74 | High | ~32x |
| RTX 4000 Ada | 20G | 0.28 | Low | ~10x |

The A40 is the default here: cheapest card with **High** stock, and already far
more than one stream needs. Reach for a 4090 only for several concurrent
sessions or `DLC_MANY_FACES=1`. Anything marked Low stock may simply fail to
provision.

Do not size on VRAM — a 48 GB A40 and a 24 GB 3090 perform about the same here,
because the model is tiny and the work is compute-bound.

### fp16 — measure, do not assume

`bench.py --fp16` forces the fp16 model without installing torch. On the T1000
it was **6x slower** (841 ms vs 135 ms): that die is TU117, the one Turing part
with no tensor cores, so fp16 runs as emulation plus cast overhead. On an
Ampere card with real tensor cores it may well win. Run the flag on the
deployment GPU before enabling it in the server.

### Local GPU testing

If `--gpus all` fails with `nvidia-cuda-mps-control: no such file or directory`,
that is a host toolkit quirk, not the image. Use the runtime directly:

```bash
docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$PWD/media:/app/media:ro" deep-live-cam-stream:latest \
  python deploy/bench.py
```

### Cost

A pod bills continuously while it exists — there is no scale-to-zero. At the
A40's $0.44/hr that is about **$10.50 a day, or $315 a month**, whether or not
anyone is streaming. Stop it when idle:

```bash
runpodctl stop pod <pod-id>
```

## Licensing

Upstream is **AGPL-3.0**. Section 13 means that if you let other people use this
over a network, you must offer them the corresponding source, including your
changes. This repository is public, which satisfies that directly.

Deep-Live-Cam's own terms require consent from anyone whose likeness you use.
`DLC_NSFW_FILTER` defaults to on here — upstream's CLI defaults it off — and the
auth token is mandatory rather than optional, both because a hosted service has a
wider blast radius than a desktop app.
