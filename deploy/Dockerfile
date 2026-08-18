# Deep-Live-Cam real-time streaming server, built for a Runpod GPU Pod.
#
# Build from the repository root:
#   docker build -f deploy/Dockerfile -t <user>/deep-live-cam-stream:latest .
#
# onnxruntime-gpu 1.26 targets CUDA 12.x + cuDNN 9; this base supplies both. If
# ORT ever reports "no CUDAExecutionProvider", the CUDA/cuDNN pairing here is
# the first thing to bump.
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_FORCE_GPU_ALLOW_GROWTH=true

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip \
      build-essential cmake \
      ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
      curl ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python

WORKDIR /app

# insightface 0.7.3 compiles native extensions at install time and expects numpy
# and cython to already be importable, so they go in first.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install "numpy>=2.0,<3" cython

COPY deploy/requirements-server.txt /app/deploy/requirements-server.txt
RUN python -m pip install -r /app/deploy/requirements-server.txt

# --- Bake the weights so a cold pod does not download ~1 GB on first request ---
# fp32 is the variant actually selected at runtime: the fp16 path is gated on
# torch.cuda, and torch is deliberately not installed. fp16 is fetched anyway so
# the image still works if torch is added later.
RUN mkdir -p /app/models \
    && curl -fL --retry 3 -o /app/models/inswapper_128.onnx \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx?download=true" \
    && curl -fL --retry 3 -o /app/models/inswapper_128_fp16.onnx \
        "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true"

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

ENV PORT=8080
EXPOSE 8080

CMD ["python", "-u", "deploy/server.py"]
