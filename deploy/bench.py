"""Headless benchmark for the face-swap hot path.

Measures what the streaming server actually does per frame, with no camera and
no network in the way. Source faces and target frames come from the repo's own
demo GIFs, so it needs nothing downloaded.

    python deploy/bench.py [--frames 100] [--width 640] [--provider cuda|cpu]

Reports detection and swap costs separately, because they scale differently:
detection dominates at high resolution, the swap is fixed-cost at 128x128.
"""

import argparse
import os
import statistics
import sys
import time
import types

import cv2
import numpy as np

# Same stub the server installs - modules.core imports PySide6 at module scope.
_ui = types.ModuleType("modules.ui")
_ui.update_status = lambda message, scope="DLC": None
_ui.check_and_ignore_nsfw = lambda target, destroy=None: False
_ui.init = lambda *a, **k: None
sys.modules["modules.ui"] = _ui

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import modules.globals  # noqa: E402
import modules.processors.frame.face_swapper as face_swapper  # noqa: E402
from modules.face_analyser import get_one_face  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def gif_frames(path, limit=60):
    """Yield BGR frames from a GIF via Pillow (cv2 does not read GIFs reliably)."""
    from PIL import Image, ImageSequence

    with Image.open(path) as im:
        for i, frame in enumerate(ImageSequence.Iterator(im)):
            if i >= limit:
                break
            yield cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR)


def find_face_frame(paths, width):
    """First frame across the given GIFs that yields a detectable face."""
    for path in paths:
        if not os.path.exists(path):
            continue
        for frame in gif_frames(path):
            frame = resize(frame, width)
            face = get_one_face(frame)
            if face is not None:
                return frame, face, path
    return None, None, None


def resize(frame, width):
    h, w = frame.shape[:2]
    if w == width:
        return frame
    return cv2.resize(frame, (width, int(h * width / w)))


def pct(values, p):
    return statistics.quantiles(values, n=100)[p - 1] if len(values) > 2 else max(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--provider", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument("--no-color-correction", action="store_true")
    ap.add_argument("--fp16", action="store_true",
                    help="force the fp16 swapper without installing torch")
    args = ap.parse_args()

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    modules.globals.frame_processors = ["face_swapper"]
    modules.globals.execution_providers = providers
    modules.globals.headless = True
    modules.globals.many_faces = False
    modules.globals.map_faces = False
    modules.globals.color_correction = not args.no_color_correction

    import onnxruntime as ort

    print(f"onnxruntime {ort.__version__}")
    print(f"available providers: {ort.get_available_providers()}")
    print(f"requested providers: {providers}")

    media = os.path.join(ROOT, "media")
    gifs = [os.path.join(media, n) for n in
            ("demo.gif", "live_show.gif", "streamers.gif", "ludwig.gif", "movie.gif")]

    if args.fp16:
        # Upstream gates fp16 on torch.cuda purely as a "is there a usable GPU"
        # probe. onnxruntime already answered that, so satisfy the gate directly
        # rather than adding ~2.5 GB of torch to the image.
        face_swapper._HAS_TORCH_CUDA = True
        print("fp16: forced on")

    print("\nloading models…")
    t0 = time.perf_counter()
    if not face_swapper.pre_start():
        print("FAIL: face_swapper.pre_start() returned False (model missing?)")
        return 1
    print(f"models loaded in {time.perf_counter() - t0:.1f}s")

    src_frame, src_face, src_path = find_face_frame(gifs, args.width)
    if src_face is None:
        print("FAIL: no detectable face in any demo GIF")
        return 1
    print(f"source face from {os.path.basename(src_path)}")

    # Build a pool of target frames that actually contain faces.
    targets = []
    for path in gifs:
        if not os.path.exists(path):
            continue
        for frame in gif_frames(path, limit=40):
            frame = resize(frame, args.width)
            if get_one_face(frame) is not None:
                targets.append(frame)
            if len(targets) >= 30:
                break
        if len(targets) >= 30:
            break
    if not targets:
        print("FAIL: no target frames with faces")
        return 1
    print(f"{len(targets)} target frames at {targets[0].shape[1]}x{targets[0].shape[0]}")

    print("\nwarming up…")
    for _ in range(5):
        face_swapper.process_frame(src_face, targets[0])

    detect_ms, raw_ms, swap_ms, total_ms = [], [], [], []
    print(f"benchmarking {args.frames} frames…")
    for i in range(args.frames):
        frame = targets[i % len(targets)]

        t = time.perf_counter()
        target_face = get_one_face(frame)
        d = (time.perf_counter() - t) * 1000

        # Pure model inference, no post-processing.
        t = time.perf_counter()
        face_swapper.swap_face(src_face, target_face, frame.copy())
        r = (time.perf_counter() - t) * 1000

        # Same swap plus masking / colour transfer / blending.
        t = time.perf_counter()
        face_swapper.process_frame(src_face, frame, target_face=target_face)
        s = (time.perf_counter() - t) * 1000

        detect_ms.append(d)
        raw_ms.append(r)
        swap_ms.append(s)
        total_ms.append(d + s)

    def row(name, xs):
        print(f"  {name:<10} p50 {statistics.median(xs):7.1f}   "
              f"p95 {pct(xs, 95):7.1f}   max {max(xs):7.1f}")

    print(f"\n--- {args.width}px, {args.provider} ---")
    print("  stage        p50 (ms)      p95 (ms)      max (ms)")
    row("detect", detect_ms)
    row("swap:raw", raw_ms)
    row("swap:full", swap_ms)
    post = [f - r for f, r in zip(swap_ms, raw_ms)]
    row("post-proc", post)
    row("total", total_ms)
    med = statistics.median(total_ms)
    print(f"\n  sustained: {1000 / med:.1f} fps  ({med:.1f} ms/frame)")
    print("  note: excludes JPEG encode/decode and network round-trip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
