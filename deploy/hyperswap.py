"""HyperSwap 256 behind inswapper's interface.

HyperSwap (FaceFusion Labs) swaps at 256x256 where inswapper swaps at 128 -
twice the resolution, four times the pixels on the face. Everything downstream
of the swapper in this server was built against inswapper, though: upstream's
paste-back, the mouth mask, Poisson blending, skin-tone matching and the
Realism passes all consume `get(frame, target, source, paste_back=False)`
returning the aligned swapped face and its affine.

So rather than teach each of those a second model, this presents HyperSwap
through the same call. Everything that already works keeps working, at twice
the resolution, and choosing it is a dropdown entry rather than a code path.

The contract, taken from FaceFusion's own face_swapper:

    template   arcface_128 - the same 5-point alignment inswapper uses
    size       256x256
    target     RGB, (x / 255 - 0.5) / 0.5, NCHW, input name "target"
    source     the *normalised* ArcFace embedding, 1x512, input name "source"
               (inswapper instead multiplies the raw embedding by an emap
               matrix stored in its own weights - HyperSwap needs no such step)
    output     NCHW in [-1, 1], undone the same way

insightface's norm_crop2 already produces arcface_128 at any size that is not a
multiple of 112: it scales the 112 template by size/128 and shifts x by
8 * size/128, which is exactly FaceFusion's arcface_128 scaled to 256.
"""

import cv2
import numpy as np
import onnxruntime as ort
from insightface.utils import face_align


def is_hyperswap(filename: str) -> bool:
    return filename.lower().startswith("hyperswap")


class HyperSwap:
    """A HyperSwap model that answers the way insightface's INSwapper does."""

    def __init__(self, model_path: str, providers=None):
        self.model_file = model_path
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        inputs = {i.name: i for i in self.session.get_inputs()}
        # Validate the contract up front. A mismatched file would otherwise
        # load quietly and fail on the first frame with a shape error that
        # says nothing about which assumption was wrong.
        if set(inputs) != {"source", "target"}:
            raise ValueError(
                f"not a HyperSwap model: expected inputs source/target, got {sorted(inputs)}")
        size = inputs["target"].shape[-1]
        if not isinstance(size, int) or size <= 0:
            raise ValueError(f"HyperSwap target input has no fixed size: {inputs['target'].shape}")
        self.input_size = (size, size)
        self.output_name = self.session.get_outputs()[0].name

    def get(self, img, target_face, source_face, paste_back=True):
        size = self.input_size[0]
        crop, M = face_align.norm_crop2(img, target_face.kps, size)

        target = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        target = ((target - 0.5) / 0.5).transpose(2, 0, 1)[None]
        source = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)

        out = self.session.run([self.output_name], {"source": source, "target": target})[0][0]
        out = np.clip(out.transpose(1, 2, 0) * 0.5 + 0.5, 0.0, 1.0)
        bgr_fake = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)

        if not paste_back:
            return bgr_fake, M
        return self._paste_back(img, bgr_fake, M)

    @staticmethod
    def _paste_back(img, bgr_fake, M):
        """Plain paste for the paste_back=True callers. The live path never
        uses it - upstream's _fast_paste_back does - but a model that only
        half-honours the interface is a trap for the next caller."""
        h, w = img.shape[:2]
        size = bgr_fake.shape[0]
        inv = cv2.invertAffineTransform(M)
        warped = cv2.warpAffine(bgr_fake, inv, (w, h), borderValue=0.0)
        mask = np.zeros((size, size), np.float32)
        cv2.ellipse(mask, (size // 2, size // 2), (int(size * 0.44), int(size * 0.44)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), size * 0.05)
        mask = cv2.warpAffine(mask, inv, (w, h), borderValue=0.0)[..., None]
        return (img.astype(np.float32) * (1 - mask) + warped.astype(np.float32) * mask).astype(np.uint8)
