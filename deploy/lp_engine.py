"""Adapter around the vendored LivePortrait pipeline.

LivePortrait is not a face swapper and does not share the swapper's contract.
It *animates a source portrait* using a driving face: the output is the chosen
person's image — their hair, their head, their background — moving with the
driver's expression and pose. That is the opposite of `inswapper`, which keeps
the driver's head and replaces only the face region.

Kept behind a lazy loader so a deployment that never selects portrait mode pays
neither the ~1.5 GB of ONNX sessions nor the VRAM.
"""

import logging
import os
import sys
import tempfile
import threading
from typing import Optional, Tuple

import cv2
import numpy as np

_LOG = logging.getLogger("dlc.liveportrait")

LP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "liveportrait")
CONFIG = os.path.join(LP_DIR, "configs", "onnx_infer.yaml")
WEIGHTS_DIR = "/app/models/liveportrait"
REQUIRED = (
    "appearance_feature_extractor.onnx", "motion_extractor.onnx",
    "warping_spade-fix.onnx", "stitching.onnx", "landmark.onnx",
    "retinaface_det_static.onnx", "face_2dpose_106_static.onnx",
)


def missing_weights():
    return [f for f in REQUIRED if not os.path.exists(os.path.join(WEIGHTS_DIR, f))]


class LivePortraitEngine:
    def __init__(self) -> None:
        self._pipe = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None

    def available(self) -> bool:
        return os.path.exists(CONFIG) and not missing_weights()

    def status(self) -> dict:
        return {
            "available": self.available(),
            "loaded": self._pipe is not None,
            "missing": missing_weights(),
            "error": self._error,
        }

    def load(self):
        """Build the pipeline on first use. Blocking; call on the inference thread."""
        with self._lock:
            if self._pipe is not None:
                return self._pipe
            if not self.available():
                raise RuntimeError(
                    f"LivePortrait weights missing from {WEIGHTS_DIR}: {missing_weights()}")

            # The vendored code imports as `src.…`, so its root must be importable.
            if LP_DIR not in sys.path:
                sys.path.insert(0, LP_DIR)

            from omegaconf import OmegaConf
            from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline

            _LOG.info("loading LivePortrait pipeline…")
            cfg = OmegaConf.load(CONFIG)
            self._pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)
            _LOG.info("LivePortrait pipeline ready")
            return self._pipe

    def prepare_source(self, bgr: np.ndarray) -> Optional[Tuple]:
        """Analyse a source portrait. Returns (src_img, src_info), or None if no face.

        prepare_source() upstream takes a path rather than an array, so the image
        goes through a temp file. It also writes into pipeline-level lists, hence
        copying the result out immediately — the values are passed back to run()
        explicitly, which is what lets one pipeline serve several sources.
        """
        pipe = self.load()
        fd, path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        try:
            cv2.imwrite(path, bgr)
            if not pipe.prepare_source(path, realtime=True):
                return None
            if not pipe.src_imgs or not pipe.src_infos:
                return None
            return pipe.src_imgs[0], pipe.src_infos[0]
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def animate(self, driving_bgr: np.ndarray, source: Tuple,
                first_frame: bool = False) -> Optional[np.ndarray]:
        """Drive the prepared source with one frame. Returns the full-size result."""
        pipe = self.load()
        src_img, src_info = source
        out = pipe.run(driving_bgr, src_img, src_info, first_frame=first_frame)
        if out is None:
            return None
        _, out_crop, out_org, _ = out
        # out_org is the full source image with the animated head pasted back;
        # out_crop is only the head region and is the fallback when pasteback is off.
        return out_org if out_org is not None else out_crop

    def reset(self) -> None:
        """Clear per-run state so a new source does not inherit the last one's."""
        if self._pipe is not None:
            self._pipe.src_lmk_pre = None


ENGINE = LivePortraitEngine()
