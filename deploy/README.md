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

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `DLC_AUTH_TOKEN` | — | **Required.** Server refuses to start without it. |
| `PORT` | `8080` | Listen port; must match the exposed HTTP port. |
| `DLC_MAX_SESSIONS` | `2` | Concurrent streams before returning 429. |
| `DLC_JPEG_QUALITY` | `80` | Return-path JPEG quality. |
| `DLC_NSFW_FILTER` | `1` | Screen uploaded source faces. |
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
