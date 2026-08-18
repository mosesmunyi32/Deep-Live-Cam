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

A package built from a private repo is private, and Runpod cannot pull it
anonymously. Two options:

- **Make the package public** — GitHub → Packages → the package → Package
  settings → Change visibility. Runpod then pulls with no credentials. Note this
  publishes a ready-to-run face-swap image under your name.
- **Keep it private** and give Runpod a registry credential: a GitHub personal
  access token with `read:packages`, added under Runpod's container registry
  auth and referenced when creating the pod.

Note on precision: the fp16 model is only selected when `torch.cuda` is
available, and torch is deliberately not installed (every torch import in
`modules/` is guarded). The runtime therefore uses fp32 — fine for a 128×128
model. Add torch to `requirements-server.txt` if you want the fp16 path.

## Deploy

```bash
runpodctl create pod \
  --name deep-live-cam \
  --imageName <dockerhub-user>/deep-live-cam-stream:latest \
  --gpuType "NVIDIA RTX A5000" \
  --gpuCount 1 \
  --containerDiskSize 25 \
  --ports "8080/http" \
  --env DLC_AUTH_TOKEN=<token>
```

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

`inswapper_128` is small; detection dominates. An RTX A4000/A5000 comfortably
holds 25-30 fps at 640×480 for one face. Scale up for `DLC_MANY_FACES=1` or
several concurrent sessions.

### Cost

A pod bills continuously while it exists — there is no scale-to-zero. Stop it
when idle:

```bash
runpodctl stop pod <pod-id>
```

## Licensing

Upstream is **AGPL-3.0**. Section 13 means that if you let other people use this
over a network, you must offer them the corresponding source, including your
changes. Keeping this repository public is the simplest way to satisfy that.

Deep-Live-Cam's own terms require consent from anyone whose likeness you use.
`DLC_NSFW_FILTER` defaults to on here — upstream's CLI defaults it off — and the
auth token is mandatory rather than optional, both because a hosted service has a
wider blast radius than a desktop app.
