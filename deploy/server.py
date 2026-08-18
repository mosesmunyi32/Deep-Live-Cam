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

import modules.globals  # noqa: E402
import modules.processors.frame.face_swapper as face_swapper  # noqa: E402
from modules.face_analyser import get_one_face  # noqa: E402


def _env_flag(name: str, default: bool) -> bool:
    return os.environ.get(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


PORT = int(os.environ.get("PORT", "8080"))
AUTH_TOKEN = os.environ.get("DLC_AUTH_TOKEN", "").strip()
JPEG_QUALITY = int(os.environ.get("DLC_JPEG_QUALITY", "80"))
MAX_SESSIONS = int(os.environ.get("DLC_MAX_SESSIONS", "2"))
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

PRECISIONS = {
    "fp32": "inswapper_128.onnx",
    "fp16": "inswapper_128_fp16.onnx",
}
_model_lock = threading.Lock()
_current_precision = "fp32"


def available_models():
    out = []
    for pid, fname in PRECISIONS.items():
        path = os.path.join(MODELS_DIR, fname)
        if os.path.exists(path):
            out.append({
                "id": pid,
                "file": fname,
                "size_mb": round(os.path.getsize(path) / 1e6),
            })
    return out


def set_precision(precision: str) -> str:
    """Switch the inswapper variant. Global — it affects every session."""
    global _current_precision
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    if not os.path.exists(os.path.join(MODELS_DIR, PRECISIONS[precision])):
        raise ValueError(f"{PRECISIONS[precision]} is not present in the image")

    with _model_lock:
        if precision == _current_precision:
            return _current_precision

        # Upstream gates fp16 on torch.cuda purely as a "is there a usable GPU"
        # probe. onnxruntime already answered that, so drive the gate directly
        # rather than installing ~2.5 GB of torch to satisfy it.
        face_swapper._HAS_TORCH_CUDA = (precision == "fp16")
        face_swapper.FACE_SWAPPER = None

        # The CUDA graph is recorded against one model's input/output buffers.
        # Carrying it across a model change would replay the previous model.
        face_swapper._cuda_graph_session.update(
            session=None, io_binding=None, ort_input=None,
            ort_latent=None, recorded=False,
        )
        face_swapper.FACE_DETECTION_CACHE.clear()
        face_swapper.FRAME_CACHE.clear()

        face_swapper.get_face_swapper()   # load now, not on the next frame
        _current_precision = precision
        _LOG.info("swapper precision -> %s (%s)", precision, PRECISIONS[precision])
        return _current_precision


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


def current_settings():
    return {
        "precision": _current_precision,
        "many_faces": bool(modules.globals.many_faces),
        "mouth_mask": bool(getattr(modules.globals, "mouth_mask", False)),
        "opacity": float(getattr(modules.globals, "opacity", 1.0)),
        "color_correction": bool(modules.globals.color_correction),
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
    modules.globals.color_correction = True
    modules.globals.opacity = 1.0


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
    _LOG.info("warm-up complete")


def decode_jpeg(payload: bytes) -> Optional[np.ndarray]:
    return cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)


def swap_jpeg(source_face, payload: bytes) -> Optional[bytes]:
    """Decode -> swap -> re-encode. Runs on EXECUTOR, never the event loop."""
    frame = decode_jpeg(payload)
    if frame is None:
        return None
    swapped = face_swapper.process_frame(source_face, frame)
    ok, encoded = cv2.imencode(".jpg", swapped, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return encoded.tobytes() if ok else None


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
    """A one-slot mailbox. Writing overwrites, so consumers always get the newest."""

    def __init__(self) -> None:
        self._item: Optional[bytes] = None
        self._event = asyncio.Event()

    def put(self, item: bytes) -> None:
        self._item = item
        self._event.set()

    async def get(self) -> bytes:
        await self._event.wait()
        self._event.clear()
        item, self._item = self._item, None
        return item


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
        self.source_face = None
        self.slot = LatestSlot()
        self.broadcast = Broadcast(placeholder_jpeg("waiting for stream"))
        # Counters exist to answer "is the client actually sending frames?".
        # Without them a silent client and a broken swap look identical here.
        self.frames_in = 0
        self.frames_out = 0
        self.bytes_in = 0
        self.last_frame_at = 0.0


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
    while True:
        payload = await session.slot.get()
        if session.source_face is None:
            await ws.send_json({"type": "error", "message": "no source face set"})
            continue
        try:
            out = await loop.run_in_executor(EXECUTOR, swap_jpeg, session.source_face, payload)
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
                    await set_source(ws, session, body.get("data", ""))
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
        if "precision" in body:
            # Model reload is blocking and touches the GPU: keep it on the
            # inference thread so it cannot overlap a swap in flight.
            await loop.run_in_executor(EXECUTOR, set_precision, body["precision"])
        if "many_faces" in body:
            modules.globals.many_faces = bool(body["many_faces"])
        if "mouth_mask" in body:
            modules.globals.mouth_mask = bool(body["mouth_mask"])
        if "color_correction" in body:
            modules.globals.color_correction = bool(body["color_correction"])
        if "opacity" in body:
            modules.globals.opacity = max(0.0, min(1.0, float(body["opacity"])))
    except Exception as exc:
        await ws.send_json({"type": "error", "message": f"config rejected: {exc}"})
        return
    await ws.send_json({"type": "settings", "settings": current_settings()})


async def set_source(ws: web.WebSocketResponse, session: Session, data_b64: str) -> None:
    """Accept, screen, and analyse the source face for this session."""
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

    session.source_face = face
    await ws.send_json({"type": "status", "message": "source face set"})
    _LOG.info("source face set for session %s", session.id)


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
