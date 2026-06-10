FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 wget \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# PyTorch 2.7.0 cu128 — required for Blackwell SM 12.0 support
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

RUN pip install --no-cache-dir \
    diffusers transformers accelerate safetensors

RUN pip install --no-cache-dir \
    fastapi "uvicorn[standard]" "gradio>=5.0.0,<6.0" pillow pydantic numpy

WORKDIR /app
COPY app.py /app/app.py
RUN mkdir -p /workspace/output /workspace/input

EXPOSE 7861 8001

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=5 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')" || exit 1

CMD ["python3", "/app/app.py"]
