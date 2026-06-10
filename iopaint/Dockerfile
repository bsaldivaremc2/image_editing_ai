FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# PyTorch 2.7.0 with cu128 wheels (SM 12.0 Blackwell support).
# Pinned to 2.7.0 to avoid nightly builds that break TorchScript compat with LaMa.
# Note: LaMa big-lama.pt TorchScript still fails on GPU (SM 12.0 not in its compiled kernels).
# Use IOPAINT_DEVICE=cpu for LaMa, or switch to sd1.5 for GPU inference.
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# IOPaint — Apache-2.0 inpainting server with FastAPI backend + React web UI
RUN pip install --no-cache-dir iopaint

WORKDIR /workspace

RUN mkdir -p /workspace/output /workspace/input

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=15s --start-period=90s --retries=5 \
  CMD wget -q --spider http://localhost:8080/ || exit 1

ENV IOPAINT_MODEL=lama
ENV IOPAINT_DEVICE=cuda

CMD ["sh", "-c", "iopaint start --model=${IOPAINT_MODEL} --device=${IOPAINT_DEVICE} --port=8080 --host=0.0.0.0"]
