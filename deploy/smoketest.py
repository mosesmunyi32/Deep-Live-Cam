"""End-to-end check of the streaming server over its real WebSocket protocol.

Exercises what a browser does: upload a source face, push frames, read swapped
frames back. Measures round-trip latency including JPEG encode/decode, which
the raw benchmark deliberately excludes.

    python deploy/smoketest.py --url http://127.0.0.1:8080 --token <token>
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


def gif_faces(path, want, width=640):
    """Frames from a GIF, resized, as JPEG bytes."""
    from PIL import Image, ImageSequence

    out = []
    with Image.open(path) as im:
        for frame in ImageSequence.Iterator(im):
            bgr = cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]
            bgr = cv2.resize(bgr, (width, int(h * width / w)))
            ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                out.append(enc.tobytes())
            if len(out) >= want:
                break
    return out


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--token", required=True)
    ap.add_argument("--frames", type=int, default=20)
    args = ap.parse_args()

    import aiohttp

    frames = gif_faces(os.path.join(ROOT, "media", "demo.gif"), args.frames + 1)
    if len(frames) < 2:
        print("FAIL: could not build test frames")
        return 1
    source, targets = frames[0], frames[1:]
    print(f"prepared 1 source + {len(targets)} target frames")

    async with aiohttp.ClientSession() as sess:
        async with sess.get(f"{args.url}/healthz") as r:
            print(f"GET /healthz -> {r.status} {await r.text()}")
            if r.status != 200:
                print("FAIL: server not ready")
                return 1

        # Auth must actually be enforced, not just configured.
        async with sess.get(f"{args.url}/ws") as r:
            if r.status != 401:
                print(f"FAIL: /ws without a token returned {r.status}, expected 401")
                return 1
            print("GET /ws without token -> 401 (auth enforced)")

        ws_url = f"{args.url.replace('http', 'ws')}/ws?token={args.token}"
        async with sess.ws_connect(ws_url, max_msg_size=16 * 1024 * 1024) as ws:
            msg = await ws.receive_json()
            print(f"connected: {msg}")

            await ws.send_json({"type": "source",
                                "data": base64.b64encode(source).decode()})
            msg = await ws.receive_json()
            print(f"source upload -> {msg}")
            if msg.get("type") != "status":
                print("FAIL: source face was rejected")
                return 1

            lat = []
            got = 0
            for payload in targets:
                t = time.perf_counter()
                await ws.send_bytes(payload)
                reply = await asyncio.wait_for(ws.receive(), timeout=30)
                if reply.type is aiohttp.WSMsgType.BINARY:
                    lat.append((time.perf_counter() - t) * 1000)
                    got += 1
                    dec = cv2.imdecode(np.frombuffer(reply.data, np.uint8),
                                       cv2.IMREAD_COLOR)
                    if dec is None:
                        print("FAIL: returned frame did not decode")
                        return 1
                else:
                    print(f"unexpected reply: {reply.type} {reply.data}")

            print(f"\nround-tripped {got}/{len(targets)} frames")
            if not lat:
                print("FAIL: no frames came back")
                return 1
            print(f"  p50 {statistics.median(lat):.1f} ms   "
                  f"min {min(lat):.1f}   max {max(lat):.1f}")
            print(f"  sustained {1000 / statistics.median(lat):.1f} fps end-to-end")

    print("\nPASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
