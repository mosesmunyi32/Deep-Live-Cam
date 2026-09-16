"""Real-time face-swap streaming server for Deep-Live-Cam on Runpod.

Transport is a WebSocket carrying JPEG frames in both directions, not WebRTC.
Runpod does not route UDP (TCP/HTTP only), so the usual WebRTC media path is
unavailable on this platform; a TCP frame stream is the transport that actually
works here. See deploy/README.md for the latency trade-off.

Endpoints
---------
    GET  /                    control UI
    GET  /output              output-only view (pop-out window / OBS source)
    GET  /stream.mjpg         MJPEG of the swapped output
    GET  /models              available swapper models and current settings
    GET  /healthz             503 until models are loaded, then 200
    WS   /ws                  the streaming session

Session protocol
----------------
    client -> server   TEXT    {"type": "source", "data": "<base64 jpeg>"}
    client -> server   TEXT    {"type": "config", ...}
    client -> server   BINARY  a JPEG frame from the webcam
    server -> client   BINARY  the swapped JPEG frame
    server -> client   TEXT    {"type": "status"|"error"|"hello", ...}

The server keeps only the newest inbound frame per session: if inference falls
behind the camera, stale frames are dropped rather than queued, which is what
keeps latency bounded under load.
"""

import asyncio
import base64
import collections
import json
import logging
import os
import secrets
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional

import cv2
import numpy as np
from aiohttp import WSMsgType, web

# --- Keep Qt out of the image -------------------------------------------------
# modules/core.py imports modules.ui (PySide6) at module scope, and
# face_swapper imports update_status from modules.core. Nothing in the live path
# needs the GUI, so a stand-in module is installed before the first import of
# the processors. This avoids patching upstream files, which keeps the diff
# against hacksider/Deep-Live-Cam small and easy to rebase.
_LOG = logging.getLogger("dlc.server")

_ui_stub = types.ModuleType("modules.ui")
_ui_stub.update_status = lambda message, scope="DLC": _LOG.info("%s: %s", scope, message)
_ui_stub.check_and_ignore_nsfw = lambda target, destroy=None: False
_ui_stub.init = lambda *args, **kwargs: None
sys.modules["modules.ui"] = _ui_stub

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import modules.globals  # noqa: E402
import modules.processors.frame.face_swapper as face_swapper  # noqa: E402
from modules.face_analyser import get_one_face  # noqa: E402
import hyperswap  # noqa: E402
import matting  # noqa: E402
import realism  # noqa: E402
import restore  # noqa: E402
import skin_tone  # noqa: E402
from lp_engine import ENGINE as LP  # noqa: E402


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


PORT = int(os.environ.get("PORT", "8080"))
AUTH_TOKEN = os.environ.get("DLC_AUTH_TOKEN", "").strip()
JPEG_QUALITY = int(os.environ.get("DLC_JPEG_QUALITY", "80"))
# Clients now hold a session from page load, not from the first frame, so this
# counts open tabs rather than active streams. Idle sessions cost no GPU - the
# single inference thread is what actually serialises work - so the cap is
# mainly a guard against unbounded face libraries.
MAX_SESSIONS = int(os.environ.get("DLC_MAX_SESSIONS", "4"))
MAX_FACES = int(os.environ.get("DLC_MAX_FACES", "12"))
NSFW_FILTER = _env_flag("DLC_NSFW_FILTER", True)

TLS_CERT = os.environ.get("DLC_TLS_CERT", "").strip()
TLS_KEY = os.environ.get("DLC_TLS_KEY", "").strip()
# When TLS is on, also serve plain HTTP on loopback. OBS's browser source is
# CEF, which rejects a self-signed certificate outright and offers no way to
# accept one, so a local consumer needs a non-TLS door. Loopback only: the
# token would otherwise cross the LAN in cleartext.
PLAIN_PORT = int(os.environ.get("DLC_PLAIN_PORT", "8081"))
# Bind address for that port. Inside a container this must be 0.0.0.0: binding
# the container's own 127.0.0.1 is unreachable, because Docker's forwarder
# arrives over the bridge interface, not loopback. Restrict exposure on the
# host side instead, with `-p 127.0.0.1:8081:8081`. Outside a container, set
# this to 127.0.0.1.
PLAIN_HOST = os.environ.get("DLC_PLAIN_HOST", "0.0.0.0")

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# One worker: a single GPU serializes inference anyway, and a single thread
# keeps frame latency predictable instead of letting requests interleave.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swap")

_ready = False
_sessions: Dict[str, "Session"] = {}
_sessions_lock = asyncio.Lock()
_last_session_id: Optional[str] = None


# --- Model selection ----------------------------------------------------------
# Both inswapper variants are baked into the image. Upstream picks between them
# with a torch.cuda probe; torch is not installed, so the choice is driven here.

# Friendly names for the two variants the image ships with. Anything else
# dropped into models/ is offered under its filename.
# Label with the model's actual name, not just its precision: showing only
# "fp32"/"fp16" hides which model is loaded, which is confusing as soon as a
# second model exists in the directory.
KNOWN_LABELS = {
    "inswapper_128.onnx": "inswapper 128 · fp32 (default)",
    "inswapper_128_fp16.onnx": "inswapper 128 · fp16 (slow without tensor cores)",
    # Twice inswapper's resolution. Heavier, so better on a pod than a laptop.
    "hyperswap_1a_256.onnx": "HyperSwap 1a · 256 px",
    "hyperswap_1b_256.onnx": "HyperSwap 1b · 256 px",
    "hyperswap_1c_256.onnx": "HyperSwap 1c · 256 px",
}
DEFAULT_MODEL = "inswapper_128.onnx"

_model_lock = threading.Lock()
_current_model = DEFAULT_MODEL

# "swap"     - inswapper: keeps the driver's head and hair, replaces the face.
# "portrait" - LivePortrait: animates the chosen portrait, so the output has
#              *that image's* hair, head and background, driven by the camera.
_mode = "swap"

# Frames a client may keep in flight. One locally, where a round trip is
# nothing; more when the GPU is a continent away. See LatestSlot.
IN_FLIGHT = max(1, min(8, int(os.environ.get("DLC_IN_FLIGHT", "1"))))

# How hard each realism pass matches the swapped face to the frame around it.
# Process-wide, like the model and the mode, because the swapper is.
_realism = dict(realism.DEFAULTS)

# Face restoration after the swap. "none" by default: it costs a second
# detection plus a 256-512 px network per face, and taste varies.
_restore = {"model": "none", "strength": 0.6}


def available_models():
    """Every .onnx in models/ — drop a file in and it shows up.

    Only inswapper-architecture models actually load; the rest are rejected at
    selection time with the reason, rather than being hidden here, so a model
    that does not work says why.
    """
    out = []
    try:
        names = sorted(os.listdir(MODELS_DIR))
    except OSError:
        return out
    for fname in names:
        if not fname.endswith(".onnx"):
            continue
        # Not a swapper. It sits in models/ because that is where weights live,
        # but offering it in the model selector would only invite an error.
        if os.path.basename(matting.MODEL_PATH) == fname:
            continue
        path = os.path.join(MODELS_DIR, fname)
        out.append({
            "id": fname,
            "file": fname,
            "label": KNOWN_LABELS.get(fname, fname[:-5]),
            "size_mb": round(os.path.getsize(path) / 1e6),
            "builtin": fname in KNOWN_LABELS,
        })
    return out


def set_model(fname: str) -> str:
    """Load a swapper model by filename. Global — it affects every session."""
    global _current_model

    # Reject traversal: this name reaches the filesystem.
    if fname != os.path.basename(fname) or not fname.endswith(".onnx"):
        raise ValueError(f"invalid model name {fname!r}")
    path = os.path.join(MODELS_DIR, fname)
    if not os.path.exists(path):
        raise ValueError(f"{fname} is not in {MODELS_DIR}")

    with _model_lock:
        if fname == _current_model and face_swapper.FACE_SWAPPER is not None:
            return _current_model

        import insightface

        previous = face_swapper.FACE_SWAPPER
        try:
            if hyperswap.is_hyperswap(fname):
                # Not an inswapper, so insightface cannot load it - but the
                # adapter answers get() the same way, so everything after this
                # line treats it identically.
                model = hyperswap.HyperSwap(path, providers=modules.globals.execution_providers)
            else:
                model = insightface.model_zoo.get_model(
                    path, providers=modules.globals.execution_providers)
        except Exception as exc:
            raise ValueError(f"{fname} failed to load: {exc}") from exc

        # insightface happily returns a detector or recogniser for the wrong
        # file. Without this check the failure would surface much later as a
        # confusing error inside swap_face.
        if not hasattr(model, "get") or not hasattr(model, "input_size"):
            raise ValueError(
                f"{fname} loaded but is not a face-swapper model "
                "(no get()/input_size) - inswapper-architecture models only")

        # The CUDA graph is recorded against one model's input/output buffers,
        # so it must not survive a model change. It is only ever recorded when
        # _HAS_TORCH_CUDA is set, but reset it regardless.
        face_swapper._cuda_graph_session.update(
            session=None, io_binding=None, ort_input=None,
            ort_latent=None, recorded=False,
        )
        face_swapper.FACE_DETECTION_CACHE.clear()
        face_swapper.FRAME_CACHE.clear()
        # Additive, not a patch to modules/: the correction sits between the
        # swapper and upstream's caller, where the swapped face and the face it
        # replaces are still in the same aligned space.
        skin_tone.wrap(model,
                       lambda: float(getattr(modules.globals, "skin_tone", 0.0)),
                       lambda: dict(_realism))
        face_swapper.FACE_SWAPPER = model
        del previous

        _current_model = fname
        _LOG.info("swapper model -> %s (input %s)", fname, model.input_size)
        return _current_model


def unload_swapper() -> None:
    """Release the swapper's GPU memory. Blocking; call on the inference thread.

    Portrait mode never touches the swapper, and on a small card the two model
    sets do not fit at once: with inswapper resident, LivePortrait runs a 4 GB
    card out of memory partway through preparing a face.
    """
    import gc

    with _model_lock:
        if face_swapper.FACE_SWAPPER is None:
            return
        # Same invalidation as a model change: the graph is recorded against
        # buffers that are about to be freed.
        face_swapper._cuda_graph_session.update(
            session=None, io_binding=None, ort_input=None,
            ort_latent=None, recorded=False,
        )
        face_swapper.FACE_DETECTION_CACHE.clear()
        face_swapper.FRAME_CACHE.clear()
        face_swapper.FACE_SWAPPER = None
        gc.collect()
    _LOG.info("swapper unloaded to free VRAM for portrait mode")


def obs_base() -> Optional[str]:
    """Origin an OBS browser source should use.

    Only meaningful when TLS is on: OBS embeds CEF, which refuses a self-signed
    certificate and gives no way to accept one, so the copy-URL button in the
    control page has to point at the plain loopback port instead of the HTTPS
    origin the page itself was loaded from.
    """
    if TLS_CERT and TLS_KEY and PLAIN_PORT:
        return f"http://127.0.0.1:{PLAIN_PORT}"
    return None


_adjust = {"brightness": 0.0, "contrast": 1.0, "saturation": 1.0}


def adjust_frame(frame: np.ndarray) -> np.ndarray:
    """Cheap exposure/colour correction, applied before the swap.

    Before rather than after on purpose: poor lighting costs detections, and a
    frame the detector misses cannot be swapped at all. Each step is skipped
    when it would be a no-op, so the default path costs nothing.
    """
    b, c, sat = _adjust["brightness"], _adjust["contrast"], _adjust["saturation"]
    if c != 1.0 or b != 0.0:
        frame = cv2.convertScaleAbs(frame, alpha=c, beta=b)
    if sat != 1.0:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= sat
        frame = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    return frame


def current_settings():
    return {
        "mode": _mode,
        "liveportrait": LP.status(),
        "portrait": LP.params(),
        "in_flight": IN_FLIGHT,
        "matting": {"available": matting.ENGINE.available()},
        "model": _current_model,
        "brightness": _adjust["brightness"],
        "contrast": _adjust["contrast"],
        "saturation": _adjust["saturation"],
        "many_faces": bool(modules.globals.many_faces),
        "mouth_mask": bool(getattr(modules.globals, "mouth_mask", False)),
        "mouth_mask_size": float(getattr(modules.globals, "mouth_mask_size", 0.0)),
        "opacity": float(getattr(modules.globals, "opacity", 1.0)),
        "sharpness": float(getattr(modules.globals, "sharpness", 0.0)),
        "skin_tone": float(getattr(modules.globals, "skin_tone", 0.0)),
        "realism": dict(_realism),
        "restore": dict(_restore,
                        available=[{"id": k, "label": restore.MODELS[k]["label"]}
                                   for k in restore.available()]),
        "poisson_blend": bool(getattr(modules.globals, "poisson_blend", False)),
        "enable_interpolation": bool(getattr(modules.globals, "enable_interpolation", False)),
        "interpolation_weight": float(getattr(modules.globals, "interpolation_weight", 0.2)),
        "nsfw_filter": NSFW_FILTER,
    }


# --- Inference ----------------------------------------------------------------

def configure_globals() -> None:
    """Set the module-level globals the processors read at call time."""
    modules.globals.frame_processors = ["face_swapper"]
    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    modules.globals.headless = True
    modules.globals.many_faces = _env_flag("DLC_MANY_FACES", False)
    modules.globals.mouth_mask = _env_flag("DLC_MOUTH_MASK", False)
    modules.globals.nsfw_filter = NSFW_FILTER
    modules.globals.map_faces = False
    modules.globals.color_correction = True   # read by other processors, not this one
    modules.globals.opacity = 1.0
    modules.globals.sharpness = 0.0
    modules.globals.poisson_blend = False
    modules.globals.enable_interpolation = False
    modules.globals.interpolation_weight = 0.2
    modules.globals.mouth_mask_size = 0.0

    # torch is installed for LivePortrait, and face_swapper gates its fp16 model
    # on torch.cuda. Left alone, adding torch silently switches the swapper to
    # fp16 - measured 5x slower on a GPU without tensor cores. Pin the gate off
    # so the model is chosen here, not by a side effect of another feature.
    face_swapper._HAS_TORCH_CUDA = False


def placeholder_jpeg(text: str) -> bytes:
    """A frame for output views to show before any real frame exists."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(img, text, (40, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (110, 110, 110), 2, cv2.LINE_AA)
    return cv2.imencode(".jpg", img)[1].tobytes()


def warm_up() -> None:
    """Load the swapper and analyser, and run one throwaway inference.

    Without this the first real frame pays model load plus CUDA kernel
    autotuning, which reads as a multi-second freeze on the client.
    """
    if not face_swapper.pre_start():
        raise RuntimeError(
            f"face_swapper.pre_start() failed - no inswapper model in {MODELS_DIR}. "
            "The image should have baked it in; check the Dockerfile download step."
        )
    face_swapper.process_frame(None, np.zeros((480, 640, 3), dtype=np.uint8))
    _LOG.info("warm-up complete (model %s)", _current_model)


def decode_jpeg(payload: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


def encode(frame: np.ndarray) -> Optional[bytes]:
    ok, enc = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return enc.tobytes() if ok else None


def process_jpeg(entry: dict, payload: bytes, first: bool = False) -> Optional[bytes]:
    """Decode -> swap or animate -> re-encode. Runs on EXECUTOR, not the loop."""
    frame = decode_jpeg(payload)
    if frame is None:
        return None
    frame = adjust_frame(frame)

    if _mode == "portrait":
        source = entry.get("lp")
        if source is None:
            return None
        out = LP.animate(frame, source, first_frame=first)
        return encode(out) if out is not None else None

    # Cheap attribute check. The model reaches this point from three different
    # loaders - selection, warm-up, and upstream's lazy path - and only the
    # first goes through set_model.
    if face_swapper.FACE_SWAPPER is not None:
        skin_tone.wrap(face_swapper.FACE_SWAPPER,
                       lambda: float(getattr(modules.globals, "skin_tone", 0.0)),
                       lambda: dict(_realism))

    if face_swapper.FACE_SWAPPER is None:
        # Portrait mode unloads it. Upstream's lazy loader would happily reload
        # on the next call, but it picks fp16 whenever torch.cuda is importable -
        # which it now is, because LivePortrait needs torch - and fp16 measured
        # six times slower on this card. Reload what was actually selected.
        set_model(_current_model)

    restoring = _restore["model"] != "none" and _restore["strength"] > 0
    # The swap can write into the frame it is given, and restoration needs the
    # untouched original as its source of real texture - copy only when it will
    # actually be used.
    plate = frame.copy() if restoring else None
    out = face_swapper.process_frame(entry["face"], frame)

    if restoring and out is not None:
        from modules.face_analyser import get_many_faces, get_one_face
        faces = (get_many_faces(plate) if modules.globals.many_faces
                 else [f for f in [get_one_face(plate)] if f is not None])
        out = restore.RESTORER.apply(
            out, plate, faces or [], _restore["model"], _restore["strength"],
            detail=float(_realism.get("detail", 0.0)),
            providers=modules.globals.execution_providers)
    return encode(out)


def is_nsfw(frame: np.ndarray) -> bool:
    """Screen a still image. Imported lazily so TensorFlow stays off the hot path."""
    if not NSFW_FILTER:
        return False
    try:
        import opennsfw2
        from PIL import Image

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return bool(opennsfw2.predict_image(Image.fromarray(rgb)) > 0.85)
    except Exception as exc:  # never let the screen fail open silently
        _LOG.warning("NSFW screen errored, rejecting the upload: %s", exc)
        return True


# --- Session plumbing ---------------------------------------------------------

class LatestSlot:
    """A short queue of the newest frames, oldest dropped when it overflows.

    Depth 1 is the original behaviour and the right one locally: a single frame
    in flight, the sender paced by whatever the GPU sustains, no queueing.

    It stops being right the moment the GPU is far away. One frame in flight
    means one round trip per frame, so throughput is capped at 1/RTT no matter
    how fast the card is - measured from Nairobi to a European pod, 333 ms of
    round trip caps a 20 ms swap at three frames a second. Depth is what buys
    that back: the client keeps `depth` frames in flight, and throughput becomes
    depth/RTT while latency stays one round trip.

    Still bounded, and still newest-first on overflow, because the alternative
    is a queue that grows until the stream is minutes behind the camera.
    """

    def __init__(self, depth: int = 1) -> None:
        self._items: Deque[bytes] = collections.deque(maxlen=max(1, depth))
        self._event = asyncio.Event()

    def set_depth(self, depth: int) -> None:
        """Resize in place, keeping whatever is already queued."""
        self._items = collections.deque(self._items, maxlen=max(1, depth))

    def put(self, item: bytes) -> None:
        self._items.append(item)  # a full deque drops from the left
        self._event.set()

    async def get(self) -> bytes:
        while True:
            if self._items:
                return self._items.popleft()
            self._event.clear()
            await self._event.wait()


class Broadcast:
    """Fans the newest swapped frame out to output views and OBS.

    Separate from the session socket because an output window is a different
    client from the one sending camera frames.
    """

    def __init__(self, initial: bytes) -> None:
        self._frame = initial
        self._seq = 0
        self._cond = asyncio.Condition()

    async def publish(self, jpeg: bytes) -> None:
        async with self._cond:
            self._frame = jpeg
            self._seq += 1
            self._cond.notify_all()

    async def get_since(self, seq: int):
        async with self._cond:
            await self._cond.wait_for(lambda: self._seq != seq)
            return self._frame, self._seq

    def snapshot(self):
        return self._frame, self._seq


class Session:
    def __init__(self, sid: str) -> None:
        self.id = sid
        # A library, not one face. Analysis is the expensive part, so faces are
        # embedded once on upload and switching afterwards is a dict lookup -
        # which is what makes changing face mid-stream instant.
        self.faces: Dict[str, dict] = {}
        self.active_face_id: Optional[str] = None
        self.slot = LatestSlot(IN_FLIGHT)
        self.broadcast = Broadcast(placeholder_jpeg("waiting for stream"))
        # Counters exist to answer "is the client actually sending frames?".
        # Without them a silent client and a broken swap look identical here.
        self.frames_in = 0
        self.frames_out = 0
        self.bytes_in = 0
        self.last_frame_at = 0.0

        self.first_frame = True

        # Restaging belongs to the picture you staged, so it is per session
        # rather than process-wide like the mode and the model.
        self.background = {"mode": "keep", "colour": "#000000", "head_crop": False}
        self.background_image = None

    def active_entry(self):
        return self.faces.get(self.active_face_id or "")

    @property
    def source_face(self):
        entry = self.active_entry()
        return entry["face"] if entry else None

    def face_list(self):
        return [{"id": fid, "label": e["label"], "thumb": e["thumb"],
                 "active": fid == self.active_face_id}
                for fid, e in self.faces.items()]


def authorized(request: web.Request) -> bool:
    if not AUTH_TOKEN:
        return True
    supplied = request.query.get("token") or ""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        supplied = header[len("Bearer "):]
    return secrets.compare_digest(supplied, AUTH_TOKEN)


def resolve_session(request: web.Request) -> Optional["Session"]:
    """Pick the session an output view should follow.

    An explicit ?s= wins. Otherwise follow whichever session most recently
    produced a frame, not the most recently opened one: with a control page
    idling on the laptop and a phone actually streaming, "newest session" picks
    the wrong one.
    """
    sid = request.query.get("s")
    if sid:
        return _sessions.get(sid)
    active = [s for s in _sessions.values() if s.frames_out]
    if active:
        return max(active, key=lambda s: s.last_frame_at)
    if _last_session_id:
        return _sessions.get(_last_session_id)
    return None


async def handle_sessions(request: web.Request) -> web.Response:
    """Which sessions exist and which are live — for picking an output source."""
    if not authorized(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    now = time.monotonic()
    return web.json_response({
        "sessions": [
            {
                "id": s.id,
                "frames_in": s.frames_in,
                "frames_out": s.frames_out,
                "idle_s": round(now - s.last_frame_at, 1) if s.last_frame_at else None,
                "has_source": s.source_face is not None,
            }
            for s in _sessions.values()
        ]
    })


# --- Handlers -----------------------------------------------------------------

async def handle_index(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def handle_output(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(os.path.join(STATIC_DIR, "output.html"))


async def handle_health(request: web.Request) -> web.Response:
    """Readiness, not liveness: 503 until the models are actually loaded."""
    if not _ready:
        return web.json_response({"status": "initializing"}, status=503)
    return web.json_response({"status": "ready", "sessions": len(_sessions)})


async def handle_models(request: web.Request) -> web.Response:
    if not authorized(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    return web.json_response({
        "models": available_models(),
        "settings": current_settings(),
        "obs_base": obs_base(),
    })


async def handle_mjpeg(request: web.Request) -> web.StreamResponse:
    """multipart/x-mixed-replace — an <img> or OBS browser source consumes this."""
    if not authorized(request):
        return web.json_response({"error": "unauthorized"}, status=401)

    resp = web.StreamResponse(status=200, headers={
        "Content-Type": "multipart/x-mixed-replace; boundary=dlcframe",
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Connection": "close",
    })
    await resp.prepare(request)

    session = resolve_session(request)
    idle = Broadcast(placeholder_jpeg("no active session"))
    source = session.broadcast if session else idle
    frame, seq = source.snapshot()

    try:
        while True:
            await resp.write(
                b"--dlcframe\r\nContent-Type: image/jpeg\r\nContent-Length: "
                + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n"
            )
            # Re-resolve each iteration so an output window opened before the
            # session started latches on once frames begin.
            session = resolve_session(request) or session
            source = session.broadcast if session else idle
            try:
                frame, seq = await asyncio.wait_for(source.get_since(seq), timeout=2.0)
            except asyncio.TimeoutError:
                frame, seq = source.snapshot()   # keep-alive re-send
    except (asyncio.CancelledError, ConnectionResetError, ConnectionError):
        pass
    return resp


async def process_loop(ws: web.WebSocketResponse, session: Session) -> None:
    loop = asyncio.get_running_loop()
    last_notice = 0.0
    while True:
        payload = await session.slot.get()
        if session.source_face is None:
            await ws.send_json({"type": "error", "message": "no source face set"})
            continue
        if _mode == "portrait" and session.active_entry().get("lp") is None:
            await ws.send_json({"type": "error",
                                "message": "this face has no portrait source - re-select it"})
            continue
        try:
            entry = session.active_entry()
            if entry is None:
                continue
            first, session.first_frame = session.first_frame, False
            missed_before = LP.missed
            out = await loop.run_in_executor(EXECUTOR, process_jpeg, entry, payload, first)
            # Say so, at most every few seconds: a portrait holding still
            # because it cannot see you looks identical, from the other end, to
            # one that has broken.
            if LP.missed > missed_before and time.time() - last_notice > 5:
                last_notice = time.time()
                await ws.send_json({
                    "type": "status",
                    "message": "No face found in the camera - the portrait is "
                               "holding still."})
        except Exception as exc:
            _LOG.exception("swap failed")
            await ws.send_json({"type": "error", "message": f"swap failed: {exc}"})
            continue
        if out is not None:
            session.frames_out += 1
            session.last_frame_at = time.monotonic()
            await session.broadcast.publish(out)
            if not ws.closed:
                await ws.send_bytes(out)


async def handle_ws(request: web.Request) -> web.StreamResponse:
    global _last_session_id

    if not authorized(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    if not _ready:
        return web.json_response({"error": "server still initializing"}, status=503)

    sid = secrets.token_hex(4)
    async with _sessions_lock:
        if len(_sessions) >= MAX_SESSIONS:
            return web.json_response(
                {"error": f"at capacity ({MAX_SESSIONS} sessions)"}, status=429)
        session = Session(sid)
        _sessions[sid] = session
        _last_session_id = sid

    # compress=False: the payload is JPEG, which does not compress further, so
    # permessage-deflate only burns CPU. It also removes the extension
    # negotiation that produced "Received frame with non-zero reserved bits".
    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=20,
                               compress=False)
    await ws.prepare(request)
    consumer = asyncio.create_task(process_loop(ws, session))
    _LOG.info("session %s opened (%d/%d)", sid, len(_sessions), MAX_SESSIONS)

    try:
        await ws.send_json({
            "type": "hello",
            "session": sid,
            "models": available_models(),
            "settings": current_settings(),
            # Per-session, unlike everything in settings, so it rides along
            # separately rather than pretending to be global state.
            "background": session.background,
            "obs_base": obs_base(),
        })
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                session.frames_in += 1
                session.bytes_in += len(msg.data)
                if session.frames_in == 1:
                    _LOG.info("session %s: first frame in (%d bytes)",
                              sid, len(msg.data))
                elif session.frames_in % 100 == 0:
                    _LOG.info("session %s: %d in / %d out, avg %.0f KB/frame",
                              sid, session.frames_in, session.frames_out,
                              session.bytes_in / session.frames_in / 1024)
                session.slot.put(msg.data)
            elif msg.type == WSMsgType.TEXT:
                try:
                    body = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "malformed json"})
                    continue
                kind = body.get("type")
                if kind == "source":
                    await set_source(ws, session, body.get("data", ""),
                                     body.get("label", ""))
                elif kind == "use_face":
                    await use_face(ws, session, body.get("id", ""))
                elif kind == "drop_face":
                    await drop_face(ws, session, body.get("id", ""))
                elif kind == "background":
                    try:
                        await set_background(ws, session, body)
                    except Exception as exc:
                        await ws.send_json({"type": "error",
                                            "message": f"background rejected: {exc}"})
                elif kind == "config":
                    await apply_config(ws, body)
            elif msg.type == WSMsgType.ERROR:
                _LOG.warning("ws error: %s", ws.exception())
    finally:
        consumer.cancel()
        async with _sessions_lock:
            _sessions.pop(sid, None)
        _LOG.info("session %s closed after %d frames in / %d out%s",
                  sid, session.frames_in, session.frames_out,
                  "" if session.frames_in else
                  " - CLIENT SENT NO FRAMES (camera never started, or the "
                  "page was backgrounded)")
    return ws


async def apply_config(ws: web.WebSocketResponse, body: dict) -> None:
    """Apply settings. These are process-wide, so they affect every session."""
    loop = asyncio.get_running_loop()
    try:
        if "mode" in body:
            await set_mode(ws, body["mode"])
        if "portrait" in body:
            await set_portrait_params(ws, body["portrait"])
        if body.get("recenter"):
            recenter()
        if "model" in body:
            # Model reload is blocking and touches the GPU: keep it on the
            # inference thread so it cannot overlap a swap in flight.
            await loop.run_in_executor(EXECUTOR, set_model, body["model"])
        for key, lo, hi in (("brightness", -100.0, 100.0),
                            ("contrast", 0.2, 3.0),
                            ("saturation", 0.0, 3.0)):
            if key in body:
                _adjust[key] = max(lo, min(hi, float(body[key])))
        if "in_flight" in body:
            global IN_FLIGHT
            IN_FLIGHT = max(1, min(8, int(body["in_flight"])))
            for sess in _sessions.values():
                sess.slot.set_depth(IN_FLIGHT)
            _LOG.info("frames in flight -> %d", IN_FLIGHT)
        if "restore" in body:
            upd = body["restore"]
            if not isinstance(upd, dict):
                raise ValueError("restore settings must be an object")
            if "model" in upd:
                if upd["model"] != "none" and upd["model"] not in restore.available():
                    raise ValueError(f"restore model {upd['model']!r} is not installed")
                _restore["model"] = upd["model"]
            if "strength" in upd:
                _restore["strength"] = max(0.0, min(1.0, float(upd["strength"])))
            _LOG.info("restore -> %s", _restore)
        if "realism" in body:
            updates = body["realism"]
            if not isinstance(updates, dict):
                raise ValueError("realism settings must be an object")
            if updates.get("preset") == "camera":
                _realism.update(realism.PRESET)
            elif updates.get("preset") == "off":
                _realism.update(realism.DEFAULTS)
            for key, value in updates.items():
                if key == "preset":
                    continue
                if key not in realism.DEFAULTS:
                    raise ValueError(f"unknown realism pass {key!r}")
                _realism[key] = max(0.0, min(1.0, float(value)))
            _LOG.info("realism -> %s", {k: round(v, 2) for k, v in _realism.items()})
        if "many_faces" in body:
            modules.globals.many_faces = bool(body["many_faces"])
        # color_correction is deliberately absent: apply_color_transfer is
        # defined in face_swapper.py but never called from the swap path, so
        # exposing it would be a control that silently does nothing.
        for key in ("mouth_mask", "poisson_blend", "enable_interpolation"):
            if key in body:
                setattr(modules.globals, key, bool(body[key]))
        for key, lo, hi in (("opacity", 0.0, 1.0),
                            ("sharpness", 0.0, 1.0),
                            ("skin_tone", 0.0, 1.0),
                            ("interpolation_weight", 0.0, 1.0),
                            ("mouth_mask_size", 0.0, 100.0)):
            if key in body:
                setattr(modules.globals, key, max(lo, min(hi, float(body[key]))))
    except Exception as exc:
        await ws.send_json({"type": "error", "message": f"config rejected: {exc}"})
        return
    await ws.send_json({"type": "settings", "settings": current_settings()})


def make_thumb(frame: np.ndarray, size: int = 96) -> str:
    """Small data-URI preview so the client can render the library."""
    h, w = frame.shape[:2]
    scale = size / max(h, w)
    small = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    ok, enc = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        return ""
    return "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode()


async def set_mode(ws: web.WebSocketResponse, mode: str) -> None:
    global _mode
    if mode not in ("swap", "portrait"):
        raise ValueError(f"unknown mode {mode!r}")
    if mode == "portrait" and not LP.available():
        raise ValueError(f"LivePortrait weights missing: {missing_or_none()}")
    if mode == _mode:
        return
    _mode = mode
    LP.reset()
    _LOG.info("mode -> %s", mode)

    # Only one of the two model sets is ever in use, and they do not both fit on
    # a small card, so the idle one is released on every switch. The cost is a
    # reload on the way back - seconds, once, against portrait mode not running
    # at all on 4 GB.
    loop = asyncio.get_running_loop()
    if mode == "portrait":
        await loop.run_in_executor(EXECUTOR, unload_swapper)
        await loop.run_in_executor(EXECUTOR, restore.RESTORER.unload)
    else:
        # Prepared sources hold device tensors of their own, so they are dropped
        # *before* the unload rather than after: a live reference at that moment
        # pins the very memory the unload exists to release, and the swapper's
        # reload then fails for want of 36 MB.
        for sess in _sessions.values():
            for entry in sess.faces.values():
                entry["lp"] = None
        LP.unload()
        await loop.run_in_executor(EXECUTOR, set_model, _current_model)

    # Prepare the active face for every live session, so switching mode does not
    # produce a stall on the next frame instead of a picture.
    for sess in list(_sessions.values()):
        if sess.active_face_id:
            if mode == "portrait":
                await ensure_portrait_source(ws, sess, sess.active_face_id)
            sess.first_frame = True


async def set_portrait_params(ws: web.WebSocketResponse, updates) -> None:
    """Apply LivePortrait parameters, re-preparing faces when one invalidates them.

    Process-wide like the model and the mode, for the same reason: the pipeline
    and its config are module-level state shared by every session.
    """
    if not isinstance(updates, dict):
        raise ValueError("portrait settings must be an object")
    loop = asyncio.get_running_loop()
    # The writes are trivial, but they land on the config the inference thread
    # reads mid-frame, so they go through the executor like a model change does.
    applied, stale = await loop.run_in_executor(EXECUTOR, LP.set_params, updates)
    if not applied:
        return

    for sess in list(_sessions.values()):
        if stale:
            # prepare_source baked the old values in, so every prepared face is
            # now wrong. Dropping them is not enough on its own - the active one
            # is re-prepared here, or the next frame would stall on it instead.
            for entry in sess.faces.values():
                entry["lp"] = None
            if _mode == "portrait" and sess.active_face_id:
                await ensure_portrait_source(ws, sess, sess.active_face_id)
    recenter()


def recenter() -> None:
    """Re-take the neutral reference pose from each session's next frame.

    LivePortrait measures motion against the pose it saw first. Whatever you
    were doing at that instant became "neutral", so a head that was turned or a
    mouth that was open is baked in as the rest position until it is re-taken.
    """
    LP.reset()
    for sess in list(_sessions.values()):
        sess.first_frame = True


def missing_or_none():
    from lp_engine import missing_weights
    return missing_weights()


async def set_source(ws: web.WebSocketResponse, session: Session,
                     data_b64: str, label: str = "") -> None:
    """Screen and analyse a face, add it to the session library, make it active."""
    loop = asyncio.get_running_loop()
    try:
        raw = base64.b64decode(data_b64.split(",")[-1])
    except Exception:
        await ws.send_json({"type": "error", "message": "source is not valid base64"})
        return

    frame = decode_jpeg(raw)
    if frame is None:
        await ws.send_json({"type": "error", "message": "source is not a decodable image"})
        return

    if await loop.run_in_executor(EXECUTOR, is_nsfw, frame):
        await ws.send_json({"type": "error", "message": "source image rejected by the NSFW filter"})
        return

    face = await loop.run_in_executor(EXECUTOR, get_one_face, frame)
    if face is None:
        await ws.send_json({"type": "error", "message": "no face found in the source image"})
        return

    if len(session.faces) >= MAX_FACES:
        await ws.send_json({"type": "error",
                            "message": f"face library is full ({MAX_FACES})"})
        return

    fid = secrets.token_hex(3)
    session.faces[fid] = {
        "face": face,
        "image": frame,          # kept for portrait mode, which needs the picture
        "lp": None,              # prepared lazily, only if portrait mode is used
        "label": label or f"face {len(session.faces) + 1}",
        "thumb": make_thumb(frame),
    }
    session.active_face_id = fid
    session.first_frame = True
    if _mode == "portrait":
        await ensure_portrait_source(ws, session, fid)
    await ws.send_json({"type": "faces", "faces": session.face_list(),
                        "message": "source face set"})
    _LOG.info("session %s: face %s added (%d in library)",
              session.id, fid, len(session.faces))


async def use_face(ws: web.WebSocketResponse, session: Session, fid: str) -> None:
    """Switch the active face. No GPU work - the embedding already exists."""
    if fid not in session.faces:
        await ws.send_json({"type": "error", "message": f"no such face {fid}"})
        return
    session.active_face_id = fid
    session.first_frame = True
    if _mode == "portrait":
        await ensure_portrait_source(ws, session, fid)
    await ws.send_json({"type": "faces", "faces": session.face_list(),
                        "message": f"switched to {session.faces[fid]['label']}"})


def hex_to_bgr(value: str):
    value = value.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"colour must be #rrggbb, not {value!r}")
    r, g, b = (int(value[i:i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def portrait_image(session: "Session", entry: dict) -> np.ndarray:
    """The picture LivePortrait animates, after any restaging.

    Only portrait mode uses this. In swap mode the background of the source is
    irrelevant - only the face embedding is taken from it - so restaging there
    would cost a matte for nothing.
    """
    bg = session.background
    if bg["mode"] == "keep" and not bg["head_crop"]:
        return entry["image"]
    image, alpha = matting.restage(
        entry["image"], bg["mode"],
        colour_bgr=hex_to_bgr(bg["colour"]),
        background=session.background_image,
        bbox=entry["face"].bbox if bg["head_crop"] else None,
        alpha=entry.get("alpha"))
    entry["alpha"] = alpha  # the mask survives a change of backdrop
    return image


async def set_background(ws: web.WebSocketResponse, session: "Session",
                         body: dict) -> None:
    """Restage every staged picture in this session."""
    bg = dict(session.background)
    if "mode" in body:
        if body["mode"] not in ("keep", "colour", "image"):
            raise ValueError(f"unknown background mode {body['mode']!r}")
        bg["mode"] = body["mode"]
    if "colour" in body:
        hex_to_bgr(body["colour"])  # validate before storing
        bg["colour"] = body["colour"]
    if "head_crop" in body:
        bg["head_crop"] = bool(body["head_crop"])
    if body.get("image"):
        raw = base64.b64decode(body["image"].split(",")[-1])
        image = decode_jpeg(raw)
        if image is None:
            await ws.send_json({"type": "error",
                                "message": "background image is not decodable"})
            return
        session.background_image = image
    if bg["mode"] == "image" and session.background_image is None:
        await ws.send_json({"type": "error",
                            "message": "choose a background image first"})
        return
    if bg["mode"] != "keep" and not matting.ENGINE.available():
        await ws.send_json({"type": "error",
                            "message": "matting model missing - background removal unavailable"})
        return

    session.background = bg
    # Restaging changes the picture itself, so everything prepared from the old
    # one is stale. The alpha is kept: it is the expensive half and does not
    # depend on what goes behind it.
    for entry in session.faces.values():
        entry["lp"] = None
    _LOG.info("session %s: background -> %s", session.id, bg)

    if _mode == "portrait" and session.active_face_id:
        await ensure_portrait_source(ws, session, session.active_face_id)
    session.first_frame = True
    await ws.send_json({"type": "settings", "settings": current_settings(),
                        "background": bg})


async def ensure_portrait_source(ws: web.WebSocketResponse, session: Session, fid: str) -> bool:
    """Prepare a face for portrait mode. Expensive, so done once and cached."""
    entry = session.faces.get(fid)
    if entry is None or entry.get("lp") is not None:
        return entry is not None
    loop = asyncio.get_running_loop()
    await ws.send_json({"type": "status", "message": "preparing portrait…"})
    try:
        image = await loop.run_in_executor(EXECUTOR, portrait_image, session, entry)
        src = await loop.run_in_executor(EXECUTOR, LP.prepare_source, image)
    except Exception as exc:
        _LOG.exception("portrait prepare failed")
        await ws.send_json({"type": "error", "message": f"portrait prepare failed: {exc}"})
        return False
    if src is None:
        await ws.send_json({"type": "error",
                            "message": f"no usable face in {entry['label']} for portrait mode"})
        return False
    entry["lp"] = src
    session.first_frame = True
    return True


async def drop_face(ws: web.WebSocketResponse, session: Session, fid: str) -> None:
    session.faces.pop(fid, None)
    if session.active_face_id == fid:
        session.active_face_id = next(iter(session.faces), None)
    await ws.send_json({"type": "faces", "faces": session.face_list(),
                        "message": "face removed"})


# --- Entrypoint ---------------------------------------------------------------

async def on_startup(app: web.Application) -> None:
    global _ready
    await asyncio.get_running_loop().run_in_executor(EXECUTOR, warm_up)
    _ready = True


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/output", handle_output)
    app.router.add_get("/stream.mjpg", handle_mjpeg)
    app.router.add_get("/models", handle_models)
    app.router.add_get("/sessions", handle_sessions)
    app.router.add_get("/healthz", handle_health)
    app.router.add_get("/ws", handle_ws)
    app.router.add_static("/static/", STATIC_DIR)
    app.on_startup.append(on_startup)
    return app


def build_ssl_context():
    """Optional TLS.

    Browsers only expose getUserMedia in a secure context. localhost counts, so
    same-machine use needs nothing; a phone on the LAN reaches this by IP and
    does not, hence the option. On Runpod the proxy already terminates TLS, so
    leave this unset there.
    """
    if not (TLS_CERT and TLS_KEY):
        return None
    import ssl

    for path in (TLS_CERT, TLS_KEY):
        if not os.path.exists(path):
            raise SystemExit(f"TLS enabled but {path} does not exist")
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(TLS_CERT, TLS_KEY)
    return ctx


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("DLC_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    if not AUTH_TOKEN:
        # The Runpod HTTP proxy is public and adds no auth of its own, so an
        # unauthenticated bind here would expose a face swapper to the internet.
        _LOG.error(
            "DLC_AUTH_TOKEN is unset. Refusing to start an unauthenticated service on a "
            "public URL. Set it to a random string, e.g. `openssl rand -hex 24`."
        )
        raise SystemExit(1)

    configure_globals()
    ssl_ctx = build_ssl_context()
    _LOG.info("starting (nsfw_filter=%s max_sessions=%d models=%s)",
              NSFW_FILTER, MAX_SESSIONS,
              ",".join(m["id"] for m in available_models()))
    asyncio.run(serve(ssl_ctx))


async def serve(ssl_ctx) -> None:
    runner = web.AppRunner(build_app(), access_log=None)
    await runner.setup()

    await web.TCPSite(runner, "0.0.0.0", PORT, ssl_context=ssl_ctx).start()
    _LOG.info("listening on %s://0.0.0.0:%d",
              "https" if ssl_ctx else "http", PORT)

    if ssl_ctx and PLAIN_PORT:
        # Exists so a local CEF client (OBS) can read /output without a
        # certificate it cannot be made to trust. Keep it off the LAN by
        # publishing it as -p 127.0.0.1:8081:8081, not by binding loopback here.
        await web.TCPSite(runner, PLAIN_HOST, PLAIN_PORT).start()
        _LOG.info("listening on http://%s:%d (plain, for OBS; publish on "
                  "127.0.0.1 to keep it off the LAN)", PLAIN_HOST, PLAIN_PORT)

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    main()
