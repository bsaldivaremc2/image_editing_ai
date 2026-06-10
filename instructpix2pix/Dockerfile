FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

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
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# PyTorch 2.7.0 with CUDA 12.8 wheels — first release with SM 12.0 (Blackwell) kernel support
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Core ML dependencies — no version pin to avoid huggingface_hub compatibility cascade
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    safetensors

# Gradio 5+ removed the huggingface_hub.HfFolder import that broke Gradio 4.x
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    "gradio>=5.0.0,<6.0" \
    pillow \
    pydantic

WORKDIR /app
COPY app.py /app/app.py

RUN mkdir -p /workspace/output /workspace/input

EXPOSE 7860 8000

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
  CMD wget -q --spider http://localhost:8000/health || exit 1

CMD ["python3", "/app/app.py"]
