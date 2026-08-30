"""End-to-end check of portrait mode, over the real WebSocket protocol.

`smoketest.py` covers the swap path. This covers everything portrait mode
added, and it exists because each of these failed in a way the build's import
check could not see:

  * switching modes both ways - the unload has to free enough VRAM for the
    other side to load, which it silently did not
  * the retargeting toggles - eye retargeting raised on every frame
  * restaging the source picture onto a colour, an image, or a head crop
  * a driving frame with no face in it - which killed the session

    python deploy/portrait_smoketest.py --url http://127.0.0.1:8081 --token <token>
"""

import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "deploy"))

from smoketest import gif_faces  # noqa: E402

FAILURES = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{' - ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(label)
    return ok


async def settle(ws, kinds=("settings", "error"), timeout=240):
    """Wait for the server's acknowledgement of a config or background change."""
    import aiohttp

    end = time.time() + timeout
    while time.time() < end:
        msg = await asyncio.wait_for(ws.receive(), timeout=end - time.time())
        if msg.type == aiohttp.WSMsgType.TEXT:
            body = json.loads(msg.data)
            if body.get("type") in kinds:
                return body
    raise TimeoutError(f"no {kinds} within {timeout}s")


async def push(ws, frames, label):
    """Send frames, return (last frame back, latencies). None means an error."""
    import aiohttp

    last, lat, notices = None, [], []
    for i, payload in enumerate(frames):
        started = time.time()
        await ws.send_bytes(payload)
        while True:
            msg = await asyncio.wait_for(ws.receive(), timeout=240)
            if msg.type == aiohttp.WSMsgType.BINARY:
                last = msg.data
                break
            if msg.type == aiohttp.WSMsgType.TEXT:
                body = json.loads(msg.data)
                if body.get("type") == "error":
                    check(False, label, body["message"])
                    return None, [], notices
                if body.get("type") == "status":
                    notices.append(body["message"])
        if i:
            lat.append((time.time() - started) * 1000)
    return last, lat, notices


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--token", required=True)
    args = ap.parse_args()

    import aiohttp

    frames = gif_faces(os.path.join(ROOT, "media", "demo.gif"), 6)
    source, drive = frames[0], frames[1:5]
    # A frame with nothing face-shaped in it, to exercise the miss path.
    blank = cv2.imencode(".jpg", np.full((360, 640, 3), 40, np.uint8))[1].tobytes()
    backdrop = base64.b64encode(
        open(os.path.join(ROOT, "media", "instruction.png"), "rb").read()).decode()

    ws_url = f"{args.url.replace('http', 'ws')}/ws?token={args.token}"
    async with aiohttp.ClientSession() as sess:
        async with sess.ws_connect(ws_url, max_msg_size=16 * 1024 * 1024) as ws:
            hello = await settle(ws, ("hello",))
            settings = hello["settings"]
            print(f"connected: session={hello.get('session')} "
                  f"portrait={settings['liveportrait']['available']} "
                  f"matting={settings['matting']['available']}")
            if not settings["liveportrait"]["available"]:
                print("FAIL: portrait mode unavailable -", settings["liveportrait"]["missing"])
                return 1

            await ws.send_json({"type": "source", "label": "smoketest",
                                "data": "data:image/jpeg;base64," +
                                        base64.b64encode(source).decode()})
            await ws.receive()

            print("modes")
            for mode in ("portrait", "swap", "portrait"):
                await ws.send_json({"type": "config", "mode": mode})
                reply = await settle(ws)
                if reply.get("type") == "error":
                    check(False, f"-> {mode}", reply["message"])
                    return 1
                out, lat, _ = await push(ws, drive, f"-> {mode}")
                check(out is not None, f"-> {mode}",
                      f"p50 {statistics.median(lat):.0f} ms" if lat else "")

            print("portrait parameters")
            for patch, label in (
                ({"animation_region": "lip"}, "animate lips only"),
                ({"animation_region": "all", "driving_multiplier": 1.8}, "motion x1.8"),
                ({"flag_eye_retargeting": True}, "eye retargeting"),
                ({"flag_lip_retargeting": True}, "lip retargeting"),
                ({"flag_eye_retargeting": False, "flag_lip_retargeting": False,
                  "driving_multiplier": 1.0}, "back to defaults"),
                ({"flag_pasteback": False}, "paste-back off"),
                ({"flag_pasteback": True}, "paste-back on"),
            ):
                await ws.send_json({"type": "config", "portrait": patch})
                reply = await settle(ws)
                if reply.get("type") == "error":
                    check(False, label, reply["message"])
                    continue
                out, _, _ = await push(ws, drive[:2], label)
                if out:
                    check(True, label, f"{len(out) / 1024:.0f} KB")

            print("background")
            for patch, label in (
                ({"mode": "colour", "colour": "#00b140"}, "solid colour"),
                ({"mode": "image", "image": "data:image/png;base64," + backdrop}, "custom image"),
                ({"head_crop": True}, "head only"),
                ({"mode": "keep", "head_crop": False}, "picture as it is"),
            ):
                await ws.send_json(dict({"type": "background"}, **patch))
                reply = await settle(ws)
                if reply.get("type") == "error":
                    check(False, label, reply["message"])
                    continue
                out, _, _ = await push(ws, drive[:2], label)
                if out:
                    check(True, label, f"{len(out) / 1024:.0f} KB")

            await ws.send_json({"type": "background", "colour": "not-a-colour"})
            reply = await settle(ws, ("error", "settings"))
            check(reply.get("type") == "error", "a bad colour is refused")

            print("driver with no face")
            # Re-zero first. After one good frame the pipeline tracks the
            # landmarks it already has instead of detecting again, so a blank
            # frame sails through on stale ones; detection only happens on the
            # first frame after a reset, which is where the miss is visible.
            await ws.send_json({"type": "config", "recenter": True})
            await settle(ws)
            out, _, notices = await push(ws, [blank, blank], "session survives")
            check(out is not None, "session survives")
            check(any("No face" in n for n in notices), "and says so",
                  notices[0] if notices else "no notice sent")
            out, _, _ = await push(ws, drive[:2], "recovers on the next real frame")
            check(out is not None, "recovers on the next real frame")

            await ws.send_json({"type": "config", "mode": "swap"})
            await settle(ws)
            out, lat, _ = await push(ws, drive, "swap still works afterwards")
            check(out is not None, "swap still works afterwards",
                  f"p50 {statistics.median(lat):.0f} ms" if lat else "")

    print()
    if FAILURES:
        print(f"FAIL ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("PASS")
    return 0


sys.exit(asyncio.run(main()))
