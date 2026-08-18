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
