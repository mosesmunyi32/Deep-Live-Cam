"""Make the swapped face belong to the footage it is sitting in.

Everything outside the swap is already real: the camera's own noise, its lens,
its motion blur, the light in the room. `inswapper` returns a 128x128 patch that
has none of that - it is smoother, sharper, cleaner and more evenly lit than the
frame around it, and that mismatch is what reads as fake. Not the absence of
film grain.

So none of this invents texture. Every pass takes something measurable from the
face being replaced - its high frequencies, its lighting, its blur, its noise -
and gives it to the face replacing it. The reference and the result are in the
swapper's aligned space, where they correspond pixel for pixel, which is the
only place this comparison is meaningful.

Passes, and what each answers from the request:

    detail    pores, skin texture, hair detail, and the plate's own grain
    light     highlights and shadows that behave like the real light did
    softness  the lens's focus falloff, instead of a uniformly sharp face
    motion    the shutter's blur when the head moves, instead of a frozen face
    grain     sensor noise, matched to what the camera is actually producing
    fringe    the lens's colour fringing at the edge of the region
"""

import logging
import math
from typing import Optional

import cv2
import numpy as np

_LOG = logging.getLogger("dlc.realism")

# Strengths, 0-1, tuned so that 1.0 is "as measured" rather than "as much as
# possible". Above ~0.6 detail starts ghosting the original face's features.
DEFAULTS = {
    "detail": 0.0,
    "light": 0.0,
    "softness": 0.0,
    "motion": 0.0,
    "grain": 0.0,
    "fringe": 0.0,
}

# What "Match the camera" sets. Detail and light do most of the work; motion is
# deliberately gentle because over-blurring a moving face is worse than a sharp
# one; grain sits at zero because detail transfer already carries the plate's
# noise with it, and doubling that looks like dirt.
PRESET = {
    "detail": 0.45,
    "light": 0.55,
    "softness": 0.5,
    "motion": 0.3,
    "grain": 0.0,
    "fringe": 0.15,
}

_mask_cache = {}
_previous_plate = {"gray": None}


def region_mask(size, feather=0.18):
    """Where the swap actually lands, as a soft 0-1 mask.

    Matching is only meaningful inside the replaced region: the rest of the
    frame is already the camera's own work and must not be touched.
    """
    key = (size, feather)
    if key in _mask_cache:
        return _mask_cache[key]
    h, w = size
    m = np.zeros((h, w), np.float32)
    cv2.ellipse(m, (w // 2, h // 2), (int(w * 0.44), int(h * 0.44)), 0, 0, 360, 1.0, -1)
    m = cv2.GaussianBlur(m, (0, 0), max(1.0, w * feather * 0.25))
    _mask_cache[key] = m
    return m


def _blend(base, altered, strength, mask):
    """Apply an altered version only inside the region, at `strength`."""
    a = (mask * float(np.clip(strength, 0.0, 1.0)))[..., None]
    return base * (1.0 - a) + altered * a


def detail_transfer(fake, plate, strength, mask, sigma=1.6):
    """Give the swap the plate's high frequencies - pores, stubble, hair, noise.

    Frequency separation: the plate's detail layer is everything above `sigma`,
    which at this scale is skin texture rather than facial structure. Added onto
    the swap, it restores the pores inswapper smoothed away *and* carries the
    camera's real noise with it, which is why the grain pass can stay at zero.

    Keep sigma small. Widen it and the plate's features - the edge of the
    original mouth, the original eyebrows - start ghosting through.
    """
    detail = plate - cv2.GaussianBlur(plate, (0, 0), sigma)
    return _blend(fake, fake + detail, strength, mask)


def light_transfer(fake, plate, strength, mask, sigma=12.0):
    """Relight the swap with the plate's own lighting.

    Swaps come out evenly lit because the model has no idea where the lamp is.
    Swapping the low-frequency luminance for the plate's puts the highlight back
    on the same cheek the room actually lit, and drops the shadow back under the
    same jaw - without touching the identity, which lives in the frequencies
    above this.
    """
    fake_lab = cv2.cvtColor(np.clip(fake, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    plate_lab = cv2.cvtColor(np.clip(plate, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
    fake_low = cv2.GaussianBlur(fake_lab[..., 0], (0, 0), sigma)
    plate_low = cv2.GaussianBlur(plate_lab[..., 0], (0, 0), sigma)
    relit = fake_lab.copy()
    relit[..., 0] = np.clip(fake_lab[..., 0] - fake_low + plate_low, 0, 255)
    relit = cv2.cvtColor(relit.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    return _blend(fake, relit, strength, mask)


def _sharpness(gray):
    """Variance of the Laplacian - higher is sharper. Only ratios are used."""
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def match_softness(fake, plate, strength, mask):
    """Blur the swap down to the plate's focus, never the other way.

    A face at the edge of the depth of field is soft; a swap pasted into it is
    perfectly sharp, and the eye reads that instantly. Sharpening the swap *up*
    to a sharper plate is deliberately not done - that invents detail, which is
    the failure mode this whole module exists to avoid.
    """
    fg = cv2.cvtColor(np.clip(fake, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    pg = cv2.cvtColor(np.clip(plate, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    fs, ps = _sharpness(fg), _sharpness(pg)
    if ps <= 0 or fs <= ps * 1.15:
        return fake
    # Sharpness falls roughly with the square of the blur radius, so the radius
    # that closes the gap is the fourth root of the ratio.
    sigma = min(2.0, (fs / max(ps, 1e-3)) ** 0.25 - 1.0)
    if sigma <= 0.05:
        return fake
    return _blend(fake, cv2.GaussianBlur(fake, (0, 0), sigma), strength, mask)


def match_motion(fake, plate, strength, mask):
    """Smear the swap the way the shutter smeared everything else.

    The plate's own motion blur is baked in by the camera; the swap is rendered
    from a still. Frame-to-frame displacement of the aligned crop gives both the
    direction and the distance, and a line kernel along it is what a shutter
    open for part of a frame actually does.
    """
    gray = cv2.cvtColor(np.clip(plate, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev = _previous_plate["gray"]
    _previous_plate["gray"] = gray
    if prev is None or prev.shape != gray.shape:
        return fake
    try:
        (dx, dy), _ = cv2.phaseCorrelate(prev, gray)
    except cv2.error:
        return fake
    dist = math.hypot(dx, dy)
    if dist < 0.6:                      # still enough that blur would be wrong
        return fake
    length = int(min(9, max(3, round(dist))))
    kernel = np.zeros((length, length), np.float32)
    angle = math.atan2(dy, dx)
    for i in range(length):
        t = i / (length - 1) - 0.5
        x = int(round((length - 1) / 2 + math.cos(angle) * t * (length - 1)))
        y = int(round((length - 1) / 2 + math.sin(angle) * t * (length - 1)))
        kernel[np.clip(y, 0, length - 1), np.clip(x, 0, length - 1)] = 1.0
    kernel /= kernel.sum()
    return _blend(fake, cv2.filter2D(fake, -1, kernel), strength, mask)


def match_grain(fake, plate, strength, mask):
    """Top up sensor noise to the level the camera is actually producing.

    Measured, not chosen: the gap between the plate's high-frequency energy and
    the swap's is how much noise is missing. Monochrome, because a sensor's
    luma noise dominates and per-channel noise reads as colour speckle.
    """
    pg = cv2.cvtColor(np.clip(plate, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    fg = cv2.cvtColor(np.clip(fake, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    plate_noise = float((pg - cv2.GaussianBlur(pg, (0, 0), 1.0)).std())
    fake_noise = float((fg - cv2.GaussianBlur(fg, (0, 0), 1.0)).std())
    missing = plate_noise - fake_noise
    if missing <= 0.4:
        return fake
    noise = np.random.normal(0.0, min(missing, 6.0), fake.shape[:2]).astype(np.float32)
    return _blend(fake, fake + noise[..., None], strength, mask)


def add_fringe(fake, strength, mask, pixels=0.6):
    """A trace of the lens's colour fringing, which grows away from centre.

    Every real lens does this and no swap has it. Radial by construction: red
    scaled out, blue scaled in, by well under a pixel at the edge.
    """
    h, w = fake.shape[:2]
    shift = pixels * float(np.clip(strength, 0.0, 1.0))
    if shift < 0.02:
        return fake
    out = fake.copy()
    for channel, direction in ((2, 1.0), (0, -1.0)):
        scale = 1.0 + direction * shift / max(w, h)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), 0, scale)
        out[..., channel] = cv2.warpAffine(fake[..., channel], m, (w, h),
                                           flags=cv2.INTER_LINEAR,
                                           borderMode=cv2.BORDER_REPLICATE)
    return _blend(fake, out, 1.0, mask)


def apply(bgr_fake: np.ndarray, plate: np.ndarray, settings: dict) -> np.ndarray:
    """Run the enabled passes, in the order a camera would have produced them.

    Texture first, because everything after it should act on textured skin.
    Then lighting, then the lens, then the shutter, then the sensor - which is
    the path light actually takes, and means blur lands on detail rather than
    detail landing on blur.
    """
    if bgr_fake is None or bgr_fake.size == 0 or plate is None or plate.size == 0:
        return bgr_fake
    if not any(float(settings.get(k, 0.0)) > 0 for k in DEFAULTS):
        return bgr_fake

    mask = region_mask(bgr_fake.shape[:2])
    fake = bgr_fake.astype(np.float32)
    ref = plate.astype(np.float32)

    if settings.get("detail", 0):
        fake = detail_transfer(fake, ref, settings["detail"], mask)
    if settings.get("light", 0):
        fake = light_transfer(fake, ref, settings["light"], mask)
    if settings.get("softness", 0):
        fake = match_softness(fake, ref, settings["softness"], mask)
    if settings.get("motion", 0):
        fake = match_motion(fake, ref, settings["motion"], mask)
    if settings.get("grain", 0):
        fake = match_grain(fake, ref, settings["grain"], mask)
    if settings.get("fringe", 0):
        fake = add_fringe(fake, settings["fringe"], mask)

    return np.clip(fake, 0, 255).astype(np.uint8)
