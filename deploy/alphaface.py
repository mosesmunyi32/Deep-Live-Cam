"""AlphaFace behind inswapper's interface.

AlphaFace (Yu et al., 2026) is a 256 px swapper trained on FFHQ plus LPFF -
Large-Pose Flickr Faces - which is where its robustness to turned heads comes
from, the case inswapper and HyperSwap handle worst. The paper reports 24 ms a
face on an RTX 4090.

Like hyperswap.py, this exists so everything downstream keeps working: paste-
back, the mouth mask, seamless edges, skin tone, restore and Realism all consume
`get(frame, target, source, paste_back=False)` returning the aligned swapped
face and its affine.

The contract differs from HyperSwap's in two ways that matter:

    alignment   FFHQ, not arcface_128. The training crops are FFHQ-aligned,
                which frames more of the head; feeding an arcface crop would
                put the face at the wrong scale and position for the network.
    target      RGB in [0, 1], not [-1, 1].
    source      the normalised w600k_r50 embedding insightface already
                computes. The export folds in AlphaFace's identity projection,
                which is inswapper's emap - see alphaface_export.py.
"""

import os

import cv2
import numpy as np
import onnxruntime as ort

# FFHQ 5-point template at 512 (eyes, nose tip, mouth corners), as used by
# FaceFusion's ffhq_512 warp. Halved for the 256 px network.
FFHQ_512 = np.array([
    [192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935],
    [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)


def is_alphaface(filename: str) -> bool:
    return filename.lower().startswith("alphaface")


class AlphaFace:
    def __init__(self, model_path: str, providers=None):
        self.model_file = model_path
        self.session = ort.InferenceSession(
            model_path, providers=providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        inputs = {i.name: i for i in self.session.get_inputs()}
        if set(inputs) != {"source", "target"}:
            raise ValueError(
                f"not an AlphaFace export: expected inputs source/target, got {sorted(inputs)}")
        size = inputs["target"].shape[-1]
        self.input_size = (int(size), int(size))
        self.template = FFHQ_512 * (size / 512.0)
        self.output_name = self.session.get_outputs()[0].name
        self.coverage = os.environ.get("DLC_ALPHAFACE_MASK", "medium").lower()
        self.face_mask = self._face_mask(size, self.coverage)

    # How much of AlphaFace's output survives into the paste, by name.
    #
    # Measured on one frame, and the two ends pull against each other:
    #
    #   full    likeness to source 0.893, none of the original face left (-0.011),
    #           but a red band along the hairline and a smeared microphone
    #   tight   no artefacts, but likeness drops to 0.668 and the original face
    #           leaks back (0.203) - face shape carries identity, and a tight
    #           mask keeps the original's
    #
    # Medium is the default by reasoning rather than measurement: it keeps the
    # forehead-to-chin width that carries face shape while stopping short of
    # the hairline and the chin, the two places the artefacts appeared. The
    # host that ran the sweep ran out of memory before it finished, so tune
    # this on a GPU pod with DLC_ALPHAFACE_MASK - and with turned heads, which
    # is the case AlphaFace exists for.
    COVERAGE = {
        "tight": (0.55, 0.75, 1.05),
        "medium": (0.95, 0.60, 1.30),
        "wide": (1.30, 0.55, 1.50),
        "full": None,
    }

    def _face_mask(self, size, coverage):
        """Where AlphaFace's output is kept, inside its own FFHQ crop.

        FFHQ framing takes in far more than the face, but upstream's paste-back
        pastes a fixed ellipse sized for the tight arcface crops the other
        swappers use. Pasted whole, that carried the network's weakest region
        back too. So the output is confined to a region derived from the
        template's own landmarks, and the rest of the crop stays original.
        """
        spec = self.COVERAGE.get(coverage, self.COVERAGE["medium"])
        if spec is None:
            return np.ones((size, size, 1), np.float32)
        top_k, bottom_k, width_k = spec
        eyes_y = float(self.template[:2, 1].mean())
        mouth_y = float(self.template[3:, 1].mean())
        gap = float(self.template[1, 0] - self.template[0, 0])
        span = mouth_y - eyes_y
        top, bottom = eyes_y - top_k * span, mouth_y + bottom_k * span
        mask = np.zeros((size, size), np.float32)
        cv2.ellipse(mask, (size // 2, int((top + bottom) / 2)),
                    (int(gap * width_k), int((bottom - top) / 2)), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(mask, (0, 0), size * 0.035)[..., None]

    def get(self, img, target_face, source_face, paste_back=True):
        size = self.input_size[0]
        M, _ = cv2.estimateAffinePartial2D(
            np.asarray(target_face.kps, np.float32), self.template, method=cv2.LMEDS)
        if M is None:
            # Landmarks too degenerate to align - hand the frame back untouched
            # rather than swap a face that is not where the network expects it.
            return (None, None) if not paste_back else img
        crop = cv2.warpAffine(img, M, (size, size), borderMode=cv2.BORDER_REPLICATE)

        target = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        target = target.transpose(2, 0, 1)[None]
        source = np.asarray(source_face.normed_embedding, dtype=np.float32).reshape(1, -1)

        out = self.session.run([self.output_name], {"target": target, "source": source})[0][0]
        swapped = cv2.cvtColor((out.transpose(1, 2, 0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR)
        bgr_fake = (crop.astype(np.float32) * (1.0 - self.face_mask)
                    + swapped.astype(np.float32) * self.face_mask).astype(np.uint8)

        if not paste_back:
            return bgr_fake, M
        import hyperswap
        return hyperswap.HyperSwap._paste_back(img, bgr_fake, M)
