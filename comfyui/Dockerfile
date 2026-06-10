FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3.10-dev \
    git wget libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# PyTorch 2.7.0 cu128 — Blackwell SM 12.0 support
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Clone ComfyUI
RUN git clone https://github.com/comfyanonymous/ComfyUI /comfyui

# Install ComfyUI's own requirements (it pulls einops, kornia, aiohttp, etc.)
RUN pip install --no-cache-dir -r /comfyui/requirements.txt

# Model downloader
RUN pip install --no-cache-dir huggingface_hub

RUN mkdir -p /comfyui/models/checkpoints \
             /comfyui/models/vae \
             /comfyui/models/loras \
             /workspace/output \
             /workspace/input

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8188

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=5 \
  CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8188/system_stats')" || exit 1

CMD ["/entrypoint.sh"]
