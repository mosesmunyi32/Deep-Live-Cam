"""Face restoration after the swap: GPEN-BFR and GFPGAN, chosen from a list.

A swap at 128 px is soft - measured, raw inswapper output carries about two
thirds of a real face's fine texture. A restoration model rebuilds that detail.
Which one is a matter of taste, and the difference is visible:

    GPEN-BFR 256   fast; modest sharpening
    GPEN-BFR 512   natural - the one that still reads as a photograph
    GFPGAN 1.4     strongest, and it looks it: harder lines, heavier contrast,
                   and a tendency to "restore" anything near the face -
                   on-screen text, overlays - into coloured artefacts

All three are ONNX, from FaceFusion's HuggingFace mirror, so they run on the
onnxruntime already here. Upstream's GFPGAN path instead needs the gfpgan,
basicsr and facexlib packages, which pin old torch and fight modern stacks;
upstream's own GPEN download URL now returns 404.

Strength blends the restored face with the unrestored one. And because
restoration is exactly what smooths skin, the Realism texture pass runs again
afterwards on the restored crop - otherwise choosing a restorer would quietly
undo "Skin texture", which put real pores back a step earlier.
"""

import logging
import os
import threading
from typing import Iterable, Optional

import cv2
import numpy as np
import onnxruntime as ort

_LOG = logging.getLogger("dlc.restore")

MODELS = {
    "gpen_bfr_256": {"file": "gpen_bfr_256.onnx", "size": 256, "label": "GPEN-BFR 256 · fast"},
    "gpen_bfr_512": {"file": "gpen_bfr_512.onnx", "size": 512, "label": "GPEN-BFR 512 · natural"},
    "gfpgan_1.4": {"file": "gfpgan_1.4.onnx", "size": 512, "label": "GFPGAN 1.4 · strongest"},
}

RESTORE_DIR = os.environ.get(
    "DLC_RESTORE_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "restore"))

# FFHQ-style 5-point template these restorers are trained on, at 512.
# Different from the arcface template the swappers use, which is why the
# restore step aligns the face again rather than reusing the swap's affine.
FFHQ_512 = np.array([
    [192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935],
    [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)


def available():
    return [k for k, v in MODELS.items() if os.path.exists(os.path.join(RESTORE_DIR, v["file"]))]


class Restorer:
    def __init__(self):
        self._sessions = {}
        self._lock = threading.Lock()

    def _session(self, key: str, providers):
        with self._lock:
            if key not in self._sessions:
                path = os.path.join(RESTORE_DIR, MODELS[key]["file"])
                if not os.path.exists(path):
                    raise RuntimeError(f"restore model missing: {path}")
                self._sessions[key] = ort.InferenceSession(path, providers=providers)
                _LOG.info("restore model loaded: %s", key)
            return self._sessions[key]

    def unload(self):
        with self._lock:
            self._sessions.clear()

    def apply(self, frame: np.ndarray, plate: np.ndarray, faces: Iterable, key: str,
              strength: float, detail: float = 0.0, providers=None) -> np.ndarray:
        """Restore every face in `frame`, using `plate` (the unswapped frame)
        as the source of real texture for the detail pass."""
        if key not in MODELS or strength <= 0:
            return frame
        size = MODELS[key]["size"]
        sess = self._session(key, providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        template = FFHQ_512 * (size / 512.0)
        h, w = frame.shape[:2]
        out = frame.astype(np.float32)

        for face in faces:
            kps = getattr(face, "kps", None)
            if kps is None:
                continue
            M, _ = cv2.estimateAffinePartial2D(np.asarray(kps, np.float32), template, method=cv2.LMEDS)
            if M is None:
                continue
            crop = cv2.warpAffine(frame, M, (size, size), borderMode=cv2.BORDER_REPLICATE)

            blob = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            blob = (blob * 2.0 - 1.0).transpose(2, 0, 1)[None]
            restored = sess.run(None, {name: blob})[0][0].transpose(1, 2, 0)
            restored = cv2.cvtColor(((np.clip(restored, -1, 1) + 1) * 127.5).astype(np.uint8),
                                    cv2.COLOR_RGB2BGR).astype(np.float32)

            a = float(np.clip(strength, 0.0, 1.0))
            restored = crop.astype(np.float32) * (1 - a) + restored * a

            if detail > 0 and plate is not None:
                import realism
                plate_crop = cv2.warpAffine(plate, M, (size, size), borderMode=cv2.BORDER_REPLICATE)
                restored = realism.detail_transfer(
                    restored, plate_crop.astype(np.float32), detail,
                    realism.region_mask((size, size)), sigma=1.6 * size / 128.0)

            inv = cv2.invertAffineTransform(M)
            back = cv2.warpAffine(np.clip(restored, 0, 255).astype(np.uint8), inv, (w, h),
                                  borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
            mask = np.zeros((size, size), np.float32)
            cv2.ellipse(mask, (size // 2, int(size * 0.53)), (int(size * 0.40), int(size * 0.46)),
                        0, 0, 360, 1.0, -1)
            mask = cv2.GaussianBlur(mask, (0, 0), size * 0.04)
            m = cv2.warpAffine(mask, inv, (w, h))[..., None]
            out = out * (1 - m) + back * m

        return np.clip(out, 0, 255).astype(np.uint8)


RESTORER = Restorer()
