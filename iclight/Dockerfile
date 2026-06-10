FROM nvidia/cuda:12.8.1-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=UTC PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libgomp1 wget git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# PyTorch cu128 — required for Blackwell SM 12.0 (RTX 50xx)
RUN pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Diffusers stack
RUN pip install diffusers transformers accelerate safetensors

# Service stack
RUN pip install fastapi "uvicorn[standard]" "gradio>=5.0.0,<6.0" pillow pydantic huggingface_hub

# Background removal (rembg + onnxruntime)
RUN pip install rembg onnxruntime

# Pre-download rembg U2Net model into the image (~170 MB, avoids runtime download)
RUN python3 -c "import rembg; rembg.new_session('u2net')" 2>&1 | tail -5

COPY app.py /app.py

EXPOSE 7862 8002

CMD ["python3", "/app.py"]
