"""Keep the target's complexion when swapping in someone else's face.

`inswapper` brings the source's identity *and* their skin tone with it, which is
what makes a swap read as pasted-on under lighting that does not match. This
pulls the colour statistics of the swapped face back towards the face it is
replacing, leaving the geometry - the identity - untouched.

Everything happens in the swapper's own aligned-face space, on the 128x128
`bgr_fake` before it is pasted back. That is deliberate: it is the only place
where the swapped pixels and the pixels they replace are in exact
correspondence, so the two sets of statistics describe the same features.

Upstream's `apply_color_transfer` is not used. It matches over an entire image,
and here that would include the black corners of the aligned crop and whatever
background sits inside it - the mean of a square that is only partly a face is
not a skin tone. This samples through the same elliptical mask the swap itself
blends with, so only pixels that actually get replaced are measured.
"""

import cv2
import numpy as np

_MASK_CACHE = {}


def _sample_mask(size):
    """Where to measure skin: the region the swap actually replaces, eroded.

    Eroded rather than exact, because the mask's feathered rim is a blend of
    face and background, and averaging that in would drag the measurement
    towards whatever is behind the head.
    """
    if size in _MASK_CACHE:
        return _MASK_CACHE[size]
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.34), int(h * 0.34)), 0, 0, 360, 255, -1)
    _MASK_CACHE[size] = mask.astype(bool)
    return _MASK_CACHE[size]


def match(bgr_fake: np.ndarray, frame: np.ndarray, M: np.ndarray,
          strength: float) -> np.ndarray:
    """Move `bgr_fake`'s colour towards the face it replaces in `frame`.

    `M` is the swap's own affine into aligned space, so warping the frame with
    it lands the original face in the same 128x128 frame as the swapped one.
    """
    if strength <= 0 or bgr_fake is None or bgr_fake.size == 0:
        return bgr_fake
    h, w = bgr_fake.shape[:2]
    aligned = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    mask = _sample_mask((h, w))
    fake_lab = cv2.cvtColor(bgr_fake, cv2.COLOR_BGR2LAB).astype(np.float32)
    real_lab = cv2.cvtColor(aligned, cv2.COLOR_BGR2LAB).astype(np.float32)

    fake_px, real_px = fake_lab[mask], real_lab[mask]
    if fake_px.size == 0 or real_px.size == 0:
        return bgr_fake

    fake_mean, fake_std = fake_px.mean(0), fake_px.std(0)
    real_mean, real_std = real_px.mean(0), real_px.std(0)
    # A flat region gives a standard deviation near zero, and dividing by it
    # turns sensor noise into banding. Hold the scale at 1 there instead.
    scale = np.where(fake_std > 1.0, real_std / np.maximum(fake_std, 1e-3), 1.0)
    # Contrast is where this gets ugly fastest: a target face in harsh light has
    # a wide L spread, and stretching the swap to match it posterises the skin.
    scale = np.clip(scale, 0.6, 1.6)

    corrected = (fake_lab - fake_mean) * scale + real_mean
    corrected = cv2.cvtColor(np.clip(corrected, 0, 255).astype(np.uint8),
                             cv2.COLOR_LAB2BGR)

    strength = float(np.clip(strength, 0.0, 1.0))
    if strength >= 1.0:
        return corrected
    return cv2.addWeighted(bgr_fake, 1.0 - strength, corrected, strength, 0)


def wrap(model, strength_of):
    """Put `match` between the swapper and its caller.

    The swapper model is upstream's, and so is the function that calls it;
    wrapping the instance's `get` is what lets this be additive rather than a
    patch to `modules/`. `strength_of` is read per frame so the slider takes
    effect without reloading anything.
    """
    # The model can arrive already wrapped: it is loaded on selection, by the
    # warm-up, and by upstream's lazy path, and wrapping twice would apply the
    # correction twice.
    if getattr(model, "_dlc_skin_tone_wrapped", False):
        return model
    original = model.get

    def get(frame, target_face, source_face, paste_back=False, **kwargs):
        out = original(frame, target_face, source_face, paste_back=paste_back, **kwargs)
        # With paste_back=True the model returns a finished frame, not the pair,
        # and the aligned-space correspondence this relies on is gone. The live
        # path always passes False; anything else is left alone.
        if paste_back or not isinstance(out, tuple) or len(out) != 2:
            return out
        bgr_fake, M = out
        strength = strength_of()
        if strength <= 0:
            return out
        return match(bgr_fake, frame, M, strength), M

    model.get = get
    model._dlc_skin_tone_wrapped = True
    return model
