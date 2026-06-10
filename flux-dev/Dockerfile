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

RUN pip install --no-cache-dir \
    diffusers==0.33.1 \
    transformers==4.51.3 \
    accelerate==1.6.0 \
    safetensors==0.5.3 \
    sentencepiece==0.2.0 \
    huggingface_hub==0.30.2 \
    optimum-quanto==0.2.6

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

EXPOSE 7864 8004

CMD ["python3", "app.py"]
