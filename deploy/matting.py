"""Cut a person out of a still picture, so portrait mode can restage them.

Portrait mode animates the source picture and pastes the head back into it, so
whatever surrounds that head - a screenshot's window chrome, a watermark, a
room - ends up in the stream. Removing it is a still-image problem: the source
is prepared once when the face is staged, not per frame, so a proper matting
model is affordable here in a way it never would be on the hot path.

u2net_human_seg is the segmentation half of U^2-Net, trained on people. It is
run through the onnxruntime already installed; nothing else is added.
"""

import logging
import os
import threading
from typing import Optional

import cv2
import numpy as np

_LOG = logging.getLogger("dlc.matting")

MODEL_PATH = os.environ.get(
    "DLC_MATTING_MODEL",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "models", "u2net_human_seg.onnx"))

# The published preprocessing for this network: 320x320, scaled by the image's
# own maximum rather than by 255, then ImageNet normalisation.
SIZE = 320
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class Matter:
    def __init__(self) -> None:
        self._sess = None
        self._lock = threading.Lock()

    def available(self) -> bool:
        return os.path.exists(MODEL_PATH)

    def load(self):
        with self._lock:
            if self._sess is not None:
                return self._sess
            if not self.available():
                raise RuntimeError(f"matting model missing: {MODEL_PATH}")
            import onnxruntime as ort

            # CPU by default, and not as a fallback: this runs while the
            # LivePortrait pipeline is resident, and on a 4 GB card there is no
            # room for another CUDA session. It costs ~1.2 s once per staged
            # picture, which is a one-off at upload, not per frame. Set
            # DLC_MATTING_DEVICE=cuda on a card with headroom.
            if os.environ.get("DLC_MATTING_DEVICE", "cpu").lower() == "cuda":
                providers = [("CUDAExecutionProvider",
                              {"cudnn_conv_algo_search": "HEURISTIC",
                               "cudnn_conv_use_max_workspace": "0",
                               "arena_extend_strategy": "kSameAsRequested"}),
                             "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]
            self._sess = ort.InferenceSession(MODEL_PATH, providers=providers)
            _LOG.info("matting model loaded (%s on %s)",
                      os.path.basename(MODEL_PATH), providers[0][0]
                      if isinstance(providers[0], tuple) else providers[0])
            return self._sess

    def unload(self) -> None:
        with self._lock:
            self._sess = None

    def alpha(self, bgr: np.ndarray) -> np.ndarray:
        """Soft mask of the person, float32 in [0, 1], at the input's size."""
        sess = self.load()
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        small = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        peak = small.max()
        small = small / (peak if peak > 0 else 1.0)
        small = (small - MEAN) / STD
        blob = small.transpose(2, 0, 1)[None].astype(np.float32)

        pred = sess.run(None, {sess.get_inputs()[0].name: blob})[0][0, 0]
        lo, hi = float(pred.min()), float(pred.max())
        pred = (pred - lo) / (hi - lo) if hi > lo else np.zeros_like(pred)
        return cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)


ENGINE = Matter()


def refine(alpha: np.ndarray, sharpness: float = 3.0, erode_px: int = 1) -> np.ndarray:
    """Tighten a soft matte so the old background does not bleed into the new one.

    Partly-transparent edge pixels are a blend of subject and whatever was
    behind them. Composited onto a different backdrop they show as a halo of the
    old one - the fringe visible around hair. Raising the contrast of the mask
    and eroding it by a pixel drops those; the small blur afterwards keeps the
    edge from turning into a cutout.
    """
    a = np.clip((alpha - 0.5) * sharpness + 0.5, 0.0, 1.0)
    if erode_px > 0:
        k = np.ones((erode_px * 2 + 1, erode_px * 2 + 1), np.uint8)
        a = cv2.erode(a, k)
        a = cv2.GaussianBlur(a, (0, 0), 0.8)
    return a


def head_box(bbox, shape, up=1.15, side=0.55, down=0.75):
    """Expand a face box to a head box - hair above, jaw and neck below.

    The detector's box is the face, so cropping to it decapitates the hair the
    whole feature exists to keep. The margins are asymmetric for the same
    reason: there is more to include above the brow than below the chin.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    bw, bh = x2 - x1, y2 - y1
    return (max(0, int(x1 - bw * side)), max(0, int(y1 - bh * up)),
            min(w, int(x2 + bw * side)), min(h, int(y2 + bh * down)))


# LivePortrait works at 512 internally and runs its own detector over whatever
# it is handed, so a crop smaller than this fails as "no usable face" - which
# reads as a complaint about the photo rather than about the crop taken from it.
MIN_CROP = 512


def upscale_min(frame: np.ndarray, minimum: int = MIN_CROP) -> np.ndarray:
    """Enlarge a small crop before anything downstream tries to detect in it."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest >= minimum or longest == 0:
        return frame
    scale = minimum / longest
    return cv2.resize(frame, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_CUBIC)


def solid(shape, colour_bgr) -> np.ndarray:
    out = np.empty((shape[0], shape[1], 3), dtype=np.uint8)
    out[:, :] = colour_bgr
    return out


def cover(background: np.ndarray, shape) -> np.ndarray:
    """Scale a background to fill the frame, cropping the overflow.

    Fill rather than fit: letterboxing a chosen backdrop reintroduces exactly
    the borders this feature is for.
    """
    th, tw = shape[0], shape[1]
    bh, bw = background.shape[:2]
    scale = max(tw / bw, th / bh)
    resized = cv2.resize(background, (max(1, int(bw * scale + 0.5)),
                                      max(1, int(bh * scale + 0.5))),
                         interpolation=cv2.INTER_AREA)
    y = (resized.shape[0] - th) // 2
    x = (resized.shape[1] - tw) // 2
    return resized[y:y + th, x:x + tw]


def composite(foreground: np.ndarray, background: np.ndarray,
              alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float32)
    return (foreground.astype(np.float32) * a
            + background.astype(np.float32) * (1.0 - a)).astype(np.uint8)


def isolate(alpha: np.ndarray, seed, threshold: float = 0.5) -> np.ndarray:
    """Keep only the connected region the face sits in.

    Person segmentation keeps anything person-shaped, and sometimes things that
    merely sit next to a person: a strip of the app's own UI survived a matte
    taken from a screenshot. Restricting the mask to the component containing
    the detected face drops those without touching the subject. Only used for
    head-only staging, where "just this face" is the whole point - a full
    picture with two people in it should keep both.
    """
    mask = (alpha > threshold).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 2:  # background plus at most one blob: nothing to choose between
        return alpha
    h, w = alpha.shape[:2]
    x = int(np.clip(seed[0], 0, w - 1))
    y = int(np.clip(seed[1], 0, h - 1))
    label = labels[y, x]
    if label == 0:
        # The seed landed on background - a detector box slightly off the matte.
        # Fall back to the biggest blob rather than returning an empty mask.
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        label = int(sizes.argmax())
    return alpha * (labels == label)


def head_limit(shape, centre, half_w: float, half_h: float,
               feather: float = 0.18) -> np.ndarray:
    """A soft ellipse around the head, for cutting away everything that is not it.

    Component isolation cannot help here: a strip of UI that touches a shoulder
    is one blob with the person, and so is the body itself. Head-only staging
    means exactly that, so the mask is bounded geometrically - centred a little
    above the face box, because what has to be included above the brow is hair.
    """
    h, w = shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xs - centre[0]) / max(half_w, 1.0)) ** 2
                + ((ys - centre[1]) / max(half_h, 1.0)) ** 2)
    return np.clip((1.0 + feather - r) / (2 * feather), 0.0, 1.0)


def restage(bgr: np.ndarray, mode: str, colour_bgr=(0, 0, 0),
            background: Optional[np.ndarray] = None,
            bbox=None, alpha: Optional[np.ndarray] = None):
    """Re-stage a source picture. Returns (image, alpha) so the mask is reusable.

    The mask is the expensive half and does not depend on what goes behind it,
    so changing colour or backdrop reuses it.
    """
    frame, seed = bgr, None
    if bbox is not None:
        x1, y1, x2, y2 = head_box(bbox, bgr.shape)
        if x2 > x1 and y2 > y1:
            crop = bgr[y1:y2, x1:x2]
            frame = upscale_min(crop)
            scale = frame.shape[0] / crop.shape[0]
            seed = (((float(bbox[0]) + float(bbox[2])) / 2 - x1) * scale,
                    ((float(bbox[1]) + float(bbox[3])) / 2 - y1) * scale)
            face_w = (float(bbox[2]) - float(bbox[0])) * scale
            face_h = (float(bbox[3]) - float(bbox[1])) * scale
            alpha = None  # a mask for the uncropped picture no longer lines up

    if mode == "keep":
        return frame, alpha

    if alpha is None or alpha.shape[:2] != frame.shape[:2]:
        alpha = refine(ENGINE.alpha(frame))
        if seed is not None:
            alpha = isolate(alpha, seed)
            # Up rather than centred: the face box stops at the brow, and the
            # hair it has to keep is above that.
            alpha = alpha * head_limit(frame.shape,
                                       (seed[0], seed[1] - face_h * 0.18),
                                       face_w * 1.05, face_h * 1.15)

    if mode == "image" and background is not None:
        back = cover(background, frame.shape)
    else:
        back = solid(frame.shape, colour_bgr)
    return composite(frame, back, alpha), alpha
