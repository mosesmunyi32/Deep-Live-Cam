"""Adapter around the vendored LivePortrait pipeline.

LivePortrait is not a face swapper and does not share the swapper's contract.
It *animates a source portrait* using a driving face: the output is the chosen
person's image — their hair, their head, their background — moving with the
driver's expression and pose. That is the opposite of `inswapper`, which keeps
the driver's head and replaces only the face region.

Kept behind a lazy loader so a deployment that never selects portrait mode pays
neither the ~1.5 GB of ONNX sessions nor the VRAM.
"""

import contextlib
import gc
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

# Parameters the control page can change while a stream is running.
#
# Every one of these is read inside the pipeline's per-frame `_run`, off
# `self.cfg.infer_params`, so writing to the config takes effect on the next
# frame - without reloading the ~1.5 GB of ONNX sessions a reconstruction
# would cost.
#
# The three marked `source` are the exception. `prepare_source` bakes their
# consequences into each face's `src_info`: the paste-back mask is built there,
# and lip normalisation is applied there. Changing one leaves every already
# prepared face stale, so `set_params` reports it and the caller re-prepares.
PARAMS = {
    "animation_region":     {"choices": ("all", "exp", "pose", "lip", "eyes")},
    "driving_multiplier":   {"range": (0.0, 2.5)},
    "flag_relative_motion": {"bool": True},
    "flag_eye_retargeting": {"bool": True},
    "flag_lip_retargeting": {"bool": True},
    "flag_stitching":       {"bool": True, "source": True},
    "flag_pasteback":       {"bool": True, "source": True},
    "flag_normalize_lip":   {"bool": True, "source": True},
}


def _coerce(key: str, value):
    """Validate one parameter, clamping numbers and refusing unknown choices."""
    spec = PARAMS[key]
    if spec.get("bool"):
        return bool(value)
    if "choices" in spec:
        if value not in spec["choices"]:
            raise ValueError(
                f"{key} must be one of {', '.join(spec['choices'])}, not {value!r}")
        return value
    lo, hi = spec["range"]
    return max(lo, min(hi, float(value)))


def missing_weights():
    return [f for f in REQUIRED if not os.path.exists(os.path.join(WEIGHTS_DIR, f))]


# ORT's CUDA knobs, chosen for the card that is actually present.
#
# The frugal set exists for a 4 GB T1000: ORT's defaults - an EXHAUSTIVE
# convolution search with max cuDNN workspace and an arena that doubles on each
# extension - cost about 1.1 GB of headroom that card does not have, and the
# pipeline ran out of memory partway through its first frame without them.
#
# They are the wrong answer on a big card, and quietly so. kSameAsRequested
# makes the arena ask for exactly what each allocation needs rather than growing
# in power-of-two blocks, so a few dozen frames of differently-sized transients
# fragment the pool. On a 24 GB 4090 portrait mode died after thirty frames:
#
#   Non-zero status code returned while running Conv node
#   Name:'/dense_motion_network/occlusion/Conv'
#   Failed to allocate memory for requested buffer of size 605342720
#
# 577 MB, unplaceable, on a card with gigabytes free - fragmentation, not
# exhaustion. So measure the card and only economise where economy is needed.
# DLC_ORT_TUNING=frugal|default forces it either way.
FRUGAL_OPTIONS = {
    "cudnn_conv_algo_search": "HEURISTIC",
    "cudnn_conv_use_max_workspace": "0",
    "arena_extend_strategy": "kSameAsRequested",
}
SMALL_CARD_BYTES = 8 * 1024 ** 3


def cuda_provider_options() -> dict:
    """Provider options for this machine's GPU."""
    tuning = os.environ.get("DLC_ORT_TUNING", "auto").lower()
    if tuning == "auto":
        try:
            import torch

            total = torch.cuda.get_device_properties(0).total_memory
            tuning = "frugal" if total < SMALL_CARD_BYTES else "default"
            _LOG.info("ORT tuning %s for a %.0f GB card", tuning, total / 1024 ** 3)
        except Exception as exc:
            # No torch, no CUDA, no answer - assume the small card, because
            # being slow on a big one beats not running on a small one.
            _LOG.debug("could not size the GPU (%s); assuming a small card", exc)
            tuning = "frugal"
    opts = {
        # TF32 is on by default in ORT, and on Ampere and later it routes fp32
        # convolutions through tensor cores at ~10 bits of mantissa. First thing
        # to turn off when a model behaves differently on a bigger GPU.
        "use_tf32": os.environ.get("DLC_TF32", "1"),
    }
    if tuning == "frugal":
        opts.update(FRUGAL_OPTIONS)
    return opts


@contextlib.contextmanager
def _frugal_sessions():
    """Apply cuda_provider_options() to every session the pipeline builds.

    The vendored predictor hardcodes bare provider names, so the options are
    injected around construction instead of by editing it - keeping the vendored
    tree a straight copy of upstream.
    """
    import onnxruntime as ort

    # insightface subclasses InferenceSession at import time, and a subclass of a
    # plain function is a TypeError. Importing it first means that has already
    # happened by the time the name is swapped.
    import insightface  # noqa: F401

    original = ort.InferenceSession

    def build(path, *args, **kwargs):
        kwargs.pop("provider_options", None)
        kwargs["providers"] = [("CUDAExecutionProvider", cuda_provider_options()),
                               "CPUExecutionProvider"]
        return original(path, *args, **kwargs)

    ort.InferenceSession = build
    try:
        yield
    finally:
        ort.InferenceSession = original


def _vram() -> str:
    """Free/total VRAM as text. Unloading is only useful if this moves."""
    try:
        import torch

        free, total = torch.cuda.mem_get_info()
        return f"{free / 2**20:.0f} MB free of {total / 2**20:.0f} MB"
    except Exception:
        return "vram unknown"


class LivePortraitEngine:
    def __init__(self) -> None:
        self._pipe = None
        self._lock = threading.Lock()
        self._error: Optional[str] = None
        # Parameter changes are kept here as well as written to the live config,
        # so they survive being set before the pipeline is ever built - the page
        # can be adjusted while still in swap mode, which is the common case.
        self._overrides: dict = {}
        # Frames whose driver had no detectable face. Counted rather than
        # logged per frame: at 1.4 fps a miss is common and a log line each time
        # would bury everything else.
        self.missed = 0

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
            for key, value in self._overrides.items():
                cfg.infer_params[key] = value
            with _frugal_sessions():
                self._pipe = FasterLivePortraitPipeline(cfg=cfg, is_animal=False)

            # The pipeline loads its paste-back mask with cv2.imread, which
            # returns None for a missing file and leaves an empty array behind.
            # Nothing checks it until warpAffine asserts deep inside
            # prepare_source, where the exception is swallowed and reported as
            # "no usable face in this image". Fail here instead, where the
            # message can say what is actually wrong.
            mask = getattr(self._pipe, "mask_crop", None)
            if mask is None or getattr(mask, "size", 0) == 0:
                self._pipe = None
                raise RuntimeError(
                    f"LivePortrait mask template missing or unreadable: "
                    f"{cfg.infer_params.mask_crop_path}")
            _LOG.info("LivePortrait pipeline ready")
            return self._pipe

    def unload(self) -> None:
        """Drop the pipeline and its ONNX sessions, releasing their VRAM.

        Needed because the swapper and this pipeline together do not fit on a
        small card. Prepared sources hold device tensors of their own, so the
        caller has to drop those too - see `set_mode` in the server.
        """
        with self._lock:
            pipe, self._pipe = self._pipe, None
            if pipe is None:
                return
            try:
                pipe.clean_models()
            except Exception as exc:  # a half-built pipeline may have no models
                _LOG.debug("clean_models failed: %s", exc)
            del pipe

            # clean_models() empties the pipeline's own dict, but the vendored
            # predictor is a *singleton* keyed by model path - a class-level
            # cache built to stop the same weights loading twice. It outlives
            # the pipeline, so every ONNX session stayed alive and the unload
            # freed 10 MB of 3 GB. Dropping the cache is what actually releases
            # them; the next load rebuilds it.
            try:
                from src.models.predictor import OnnxRuntimePredictorSingleton

                OnnxRuntimePredictorSingleton._instance.clear()
            except Exception as exc:
                _LOG.debug("could not clear the predictor singleton: %s", exc)
        gc.collect()
        # torch hands freed blocks back to its own cache, not to the driver, so
        # without this the swapper's reload still finds no memory - which is
        # exactly how a switch back to swap mode failed with a 36 MB allocation
        # error while nvidia-smi showed the card full.
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception as exc:
            _LOG.debug("empty_cache failed: %s", exc)
        _LOG.info("LivePortrait pipeline unloaded (%s)", _vram())

    @staticmethod
    def _raise_if_oom() -> None:
        """Tell an allocation failure apart from a photo with no face in it.

        The vendored `prepare_source` catches everything and returns False, so
        running out of memory arrives here looking exactly like a bad picture.
        Free VRAM is the tell, and reporting the innocent cause sends you off
        hunting for a better photo when the real fix is memory.
        """
        try:
            import torch

            free, total = torch.cuda.mem_get_info()
        except Exception:
            return
        if free < 128 * 1024 * 1024:
            raise RuntimeError(
                f"GPU out of memory - {free / 1e6:.0f} MB free of {total / 1e6:.0f} MB. "
                "Portrait mode needs room that the swapper is holding.")

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
                # The vendored code catches everything and returns False, so
                # this is the only place the real cause can be recovered.
                self._error = getattr(pipe, "last_error", None)
                if self._error:
                    _LOG.error("portrait prepare failed:\n%s", self._error)
                self._raise_if_oom()
                return None
            self._error = None
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

        # "No face in this driving frame" is reported as a tuple of Nones rather
        # than as None, so it can only be seen after unpacking. It happens
        # constantly in practice - a camera still warming up, a head turned
        # away, a dark room - and the honest answer is the staged picture,
        # holding still, rather than a dead stream. (Letting it reach cvtColor
        # raised the same "empty source" assertion an empty array does, which
        # killed the session on the first frame the detector missed.)
        if out_crop is None and out_org is None:
            self.missed += 1
            return cv2.cvtColor(src_img, cv2.COLOR_RGB2BGR)

        # out_org is the full source picture with the animated head pasted back;
        # out_crop is the head region alone. Which one is right is decided by the
        # paste-back flags, not by out_org being None - it never is. The pipeline
        # seeds it with the *untouched* source and only paints into it when
        # pasting back, so keying on None returned an unanimated picture and made
        # the toggle look like it did nothing.
        params = pipe.cfg.infer_params
        pasted = bool(params.flag_pasteback and params.flag_do_crop
                      and params.flag_stitching)
        frame = out_org if (pasted and out_org is not None) else out_crop

        # LivePortrait is RGB throughout - the source is converted on load and the
        # generator's output stays that way - while everything on this side is
        # BGR: the JPEG decode, the encoder, the swapper. Without this the stream
        # comes back with red and blue exchanged, which reads as a cold cast on
        # skin rather than as an obvious fault. Measured against the source frame,
        # mean absolute error 13.35 as-is versus 1.04 corrected.
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def _file_params(self) -> dict:
        """Defaults straight from the YAML, for reporting before the first load."""
        try:
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(CONFIG)
            return {k: cfg.infer_params[k] for k in PARAMS}
        except Exception as exc:  # a missing config already shows as unavailable
            _LOG.debug("could not read LivePortrait defaults: %s", exc)
            return {}

    def params(self) -> dict:
        """Current values. Reported whether or not the pipeline exists yet."""
        if self._pipe is not None:
            live = {k: self._pipe.cfg.infer_params[k] for k in PARAMS}
        else:
            live = self._file_params()
            live.update(self._overrides)
        return {k: (bool(v) if PARAMS[k].get("bool")
                    else v if "choices" in PARAMS[k] else float(v))
                for k, v in live.items()}

    def set_params(self, updates: dict) -> Tuple[dict, bool]:
        """Apply parameter changes. Returns (what changed, sources now stale).

        Unknown keys raise rather than being dropped: a control that silently
        does nothing is the failure mode the colour-correction toggle was.
        """
        current = self.params()
        applied, stale = {}, False
        for key, raw in updates.items():
            if key not in PARAMS:
                raise ValueError(f"unknown portrait parameter {key!r}")
            value = _coerce(key, raw)
            if current.get(key) == value:
                continue
            applied[key] = value
            stale = stale or bool(PARAMS[key].get("source"))
        if not applied:
            return {}, False
        with self._lock:
            self._overrides.update(applied)
            if self._pipe is not None:
                for key, value in applied.items():
                    self._pipe.cfg.infer_params[key] = value
        _LOG.info("portrait params -> %s", applied)
        return applied, stale

    def reset(self) -> None:
        """Clear per-run state so a new source does not inherit the last one's."""
        if self._pipe is not None:
            self._pipe.src_lmk_pre = None


ENGINE = LivePortraitEngine()
