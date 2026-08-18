# Deep-Live-Cam real-time streaming server, built for a Runpod GPU Pod.
#
# Build from the repository root:
#   docker build -f deploy/Dockerfile -t <user>/deep-live-cam-stream:latest .
#
# onnxruntime-gpu 1.26 targets CUDA 12.x + cuDNN 9; this base supplies both. If
# ORT ever reports "no CUDAExecutionProvider", the CUDA/cuDNN pairing here is
# the first thing to bump.
#
# Ubuntu 24.04, not 22.04, is deliberate: onnxruntime-gpu 1.26.0 publishes
# wheels for cp311-cp314 only, so 22.04's stock Python 3.10 cannot satisfy the
# pin upstream's requirements.txt uses. 24.04 ships Python 3.12.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_FORCE_GPU_ALLOW_GROWTH=true \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# The venv is not cosmetic: 24.04 marks the system Python as externally managed
# (PEP 668), so a system-wide pip install refuses to run.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-dev python3.12-venv \
      build-essential cmake \
      ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

WORKDIR /app

# insightface 0.7.3 compiles native extensions at install time and expects numpy
# and cython to already be importable, so they go in first.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "numpy>=2.0,<3" cython

COPY deploy/requirements-server.txt /app/deploy/requirements-server.txt
RUN python -m pip install -r /app/deploy/requirements-server.txt

# Prove the Keras backend resolved and bake the NSFW weights. opennsfw2 fetches
# them on first predict otherwise, which would be a cold-start download on the
# pod. Failing here is far cheaper than failing on the first upload, where
# is_nsfw() fails closed and would reject every source face.
RUN python -c "\
import keras, opennsfw2; \
opennsfw2.make_open_nsfw_model(); \
print('nsfw stack OK, keras', keras.__version__)"

# --- Bake the weights so a cold pod does not download ~1 GB on first request ---
# fp32 is the variant actually selected at runtime: the fp16 path is gated on
# torch.cuda, and torch is deliberately not installed. fp16 is fetched anyway so
# the image still works if torch is added later.
RUN mkdir -p /app/models \
    && curl -fsSL --retry 3 -o /app/models/inswapper_128.onnx \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx?download=true" \
    && curl -fsSL --retry 3 -o /app/models/inswapper_128_fp16.onnx \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true"

# LivePortrait ONNX weights for portrait mode. Same reasoning as above: baked in
# so a cold pod downloads nothing. warping_spade-fix is the corrected export and
# is what deploy/liveportrait/configs/onnx_infer.yaml points at.
RUN mkdir -p /app/models/liveportrait \
    && for f in appearance_feature_extractor.onnx motion_extractor.onnx \
                warping_spade-fix.onnx stitching.onnx stitching_eye.onnx \
                stitching_lip.onnx landmark.onnx retinaface_det_static.onnx \
                face_2dpose_106_static.onnx; do \
         curl -fsSL --retry 3 -o "/app/models/liveportrait/$f" \
           "https://huggingface.co/warmshao/FasterLivePortrait/resolve/main/liveportrait_onnx/$f" \
           || exit 1; \
       done

# buffalo_l (detection + recognition + landmarks). Downloaded on CPU providers
# because the build host has no GPU; it lands in ~/.insightface, which is where
# modules/face_analyser.py looks for it at runtime.
RUN python -c "\
import insightface; \
insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'], \
allowed_modules=['detection','recognition','landmark_2d_106'])"

COPY . /app

# Fail the build rather than the pod if the GPU provider went missing.
RUN python -c "\
import onnxruntime as ort; \
ps = ort.get_available_providers(); \
print('providers:', ps); \
assert 'CUDAExecutionProvider' in ps, 'onnxruntime-gpu has no CUDA provider'"

# Fail the build, not the pod, if the vendored pipeline cannot even import or a
# weight is missing. No GPU here, so this checks wiring only, not inference.
RUN python -c "\
import sys; sys.path.insert(0, '/app/deploy'); \
from lp_engine import ENGINE, missing_weights; \
m = missing_weights(); \
assert not m, f'LivePortrait weights missing: {m}'; \
sys.path.insert(0, '/app/deploy/liveportrait'); \
from omegaconf import OmegaConf; \
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline; \
print('liveportrait import OK, weights present')"

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-u", "deploy/server.py"]
