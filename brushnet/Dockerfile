FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev git wget curl ca-certificates \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

# Blackwell (SM 12.0) requires cu128 wheels
RUN pip install --no-cache-dir \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Install BrushNet as a fork of diffusers (0.27.0 + BrushNet pipeline/model)
# The TencentARC/BrushNet repo IS a fork of diffusers — installs as 'diffusers'
RUN git clone --depth 1 https://github.com/TencentARC/BrushNet.git /app/BrushNet && \
    pip install --no-cache-dir -e /app/BrushNet

# Patch: huggingface_hub 0.25+ removed cached_download (community-pipeline download only).
RUN sed -i 's/from huggingface_hub import cached_download, hf_hub_download, model_info/from huggingface_hub import hf_hub_download, model_info; cached_download = None/' \
    /app/BrushNet/src/diffusers/utils/dynamic_modules_utils.py

# Modern transformers/accelerate are compatible (fork only needs >=4.25.1 / >=0.11.0)
RUN pip install --no-cache-dir \
    transformers==4.51.3 \
    accelerate==1.6.0 \
    safetensors==0.5.3 \
    huggingface_hub==0.30.2

RUN pip install --no-cache-dir \
    fastapi==0.115.12 \
    uvicorn==0.34.2 \
    pydantic==2.11.3 \
    Pillow==11.2.1 \
    numpy==1.26.4 \
    gradio==5.29.1

WORKDIR /app
COPY app.py /app/app.py

ENV HF_HOME=/root/.cache/huggingface
ENV PYTHONUNBUFFERED=1

EXPOSE 7863 8003

CMD ["python3", "app.py"]
