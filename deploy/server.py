"""Real-time face-swap streaming server for Deep-Live-Cam on Runpod.

Transport is a WebSocket carrying JPEG frames in both directions, not WebRTC.
Runpod does not route UDP (TCP/HTTP only), so the usual WebRTC media path is
unavailable on this platform; a TCP frame stream is the transport that actually
works here. See deploy/README.md for the latency trade-off.

Protocol
--------
    client -> server   TEXT    {"type": "source", "data": "<base64 jpeg>"}
    client -> server   BINARY  a JPEG frame from the webcam
    server -> client   BINARY  the swapped JPEG frame
    server -> client   TEXT    {"type": "status"|"error", ...}

The server keeps only the newest inbound frame per session: if inference falls
behind the camera, stale frames are dropped rather than queued, which is what
keeps latency bounded under load.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
MANY_FACES = _env_flag("DLC_MANY_FACES", False)
MOUTH_MASK = _env_flag("DLC_MOUTH_MASK", False)

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# One worker: a single GPU serializes inference anyway, and a single thread
# keeps frame latency predictable instead of letting requests interleave.
EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="swap")

_ready = False
_active_sessions = 0
_sessions_lock = asyncio.Lock()


# --- Inference ----------------------------------------------------------------

def configure_globals() -> None:
    """Set the module-level globals the processors read at call time."""
    modules.globals.frame_processors = ["face_swapper"]
    modules.globals.execution_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    modules.globals.headless = True
    modules.globals.many_faces = MANY_FACES
    modules.globals.mouth_mask = MOUTH_MASK
    modules.globals.nsfw_filter = NSFW_FILTER
    modules.globals.map_faces = False
    modules.globals.color_correction = True


def warm_up() -> None:
    """Load the swapper and analyser, and run one throwaway inference.

    Without this the first real frame pays model load plus CUDA kernel
    autotuning, which reads as a multi-second freeze on the client.
    """
    if not face_swapper.pre_start():
        raise RuntimeError(
            "face_swapper.pre_start() failed - the inswapper model is missing from "
            f"{modules.globals.__dict__.get('models_dir', 'models/')}. "
            "The image should have baked it in; check the Dockerfile download step."
        )
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    face_swapper.process_frame(None, dummy)
    _LOG.info("warm-up complete")


def decode_jpeg(payload: bytes) -> Optional[np.ndarray]:
    frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    return frame


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
        score = opennsfw2.predict_image(Image.fromarray(rgb))
        return bool(score > 0.85)
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


class Session:
    def __init__(self) -> None:
        self.source_face = None
        self.slot = LatestSlot()
        self.dropped = 0


def authorized(request: web.Request) -> bool:
    if not AUTH_TOKEN:
        return True
    supplied = request.query.get("token") or ""
    header = request.headers.get("Authorization", "")
    if header.startswith("Bearer "):
        supplied = header[len("Bearer "):]
    return supplied == AUTH_TOKEN


# --- Handlers -----------------------------------------------------------------

async def handle_index(request: web.Request) -> web.StreamResponse:
    return web.FileResponse(os.path.join(STATIC_DIR, "index.html"))


async def handle_health(request: web.Request) -> web.Response:
    """Readiness, not liveness: 503 until the models are actually loaded."""
    if not _ready:
        return web.json_response({"status": "initializing"}, status=503)
    return web.json_response({"status": "ready", "sessions": _active_sessions})


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
        if out is not None and not ws.closed:
            await ws.send_bytes(out)


async def handle_ws(request: web.Request) -> web.StreamResponse:
    global _active_sessions

    if not authorized(request):
        return web.json_response({"error": "unauthorized"}, status=401)
    if not _ready:
        return web.json_response({"error": "server still initializing"}, status=503)

    async with _sessions_lock:
        if _active_sessions >= MAX_SESSIONS:
            return web.json_response(
                {"error": f"at capacity ({MAX_SESSIONS} sessions)"}, status=429
            )
        _active_sessions += 1

    ws = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024, heartbeat=20)
    await ws.prepare(request)
    session = Session()
    consumer = asyncio.create_task(process_loop(ws, session))
    _LOG.info("session opened (%d/%d)", _active_sessions, MAX_SESSIONS)

    try:
        await ws.send_json({"type": "status", "message": "connected"})
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                session.slot.put(msg.data)
            elif msg.type == WSMsgType.TEXT:
                try:
                    body = json.loads(msg.data)
                except json.JSONDecodeError:
                    await ws.send_json({"type": "error", "message": "malformed json"})
                    continue
                if body.get("type") == "source":
                    await set_source(ws, session, body.get("data", ""))
            elif msg.type == WSMsgType.ERROR:
                _LOG.warning("ws error: %s", ws.exception())
    finally:
        consumer.cancel()
        async with _sessions_lock:
            _active_sessions -= 1
        _LOG.info("session closed")
    return ws


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
    _LOG.info("source face set for session")


# --- Entrypoint ---------------------------------------------------------------

async def on_startup(app: web.Application) -> None:
    global _ready
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(EXECUTOR, warm_up)
    _ready = True


def build_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/healthz", handle_health)
    app.router.add_get("/ws", handle_ws)
    app.router.add_static("/static/", STATIC_DIR)
    app.on_startup.append(on_startup)
    return app


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
    _LOG.info(
        "starting on :%d (nsfw_filter=%s many_faces=%s max_sessions=%d)",
        PORT, NSFW_FILTER, MANY_FACES, MAX_SESSIONS,
    )
    web.run_app(build_app(), host="0.0.0.0", port=PORT, access_log=None)


if __name__ == "__main__":
    main()
