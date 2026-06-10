# InstructPix2Pix — Local Docker Deployment

## What It Does

InstructPix2Pix edits images using natural-language instructions. You provide an image and a text command such as _"change the shirt to blue"_ or _"add sunglasses"_, and the model returns an edited image.

Unlike inpainting tools that require painting a mask, InstructPix2Pix interprets the instruction directly and applies the edit across the entire image in a semantically consistent way.

## How It Works

The model is built on Stable Diffusion 1.5 and fine-tuned on 454,445 (image, instruction, edited-image) triplets generated using GPT-3 and Stable Diffusion img2img. At inference time it uses **classifier-free guidance on two streams simultaneously**:

1. **Text guidance** (`guidance_scale`) — how strongly to follow the edit instruction
2. **Image guidance** (`image_guidance_scale`) — how closely to preserve the original image

The combination lets you dial in the trade-off: more image guidance → subtle edit, less → aggressive transformation.

**Scheduler:** EulerAncestralDiscreteScheduler (recommended by the paper for best quality).

## GPU Compatibility

Uses HuggingFace `diffusers` — kernels compile at runtime via `torch.compile`. **Fully compatible with Blackwell (SM 12.0)**. No precompiled TorchScript issues.

## Ports

| Port | Service | URL |
|---|---|---|
| 7860 | Gradio Web UI | http://localhost:7860 |
| 8000 | FastAPI REST API | http://localhost:8000/api/v1/edit |
| 8000 | Swagger docs | http://localhost:8000/docs |
| 8000 | Health check | http://localhost:8000/health |

## REST API

### Edit an image
```
POST http://localhost:8000/api/v1/edit
Content-Type: application/json
```

**Request body:**
```json
{
  "image": "<base64 encoded PNG or JPEG>",
  "instruction": "change the green shirt to a blue shirt",
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "image_guidance_scale": 1.5,
  "seed": 42
}
```

**Response:** `200 OK` — binary PNG of the edited image.

### Parameter guide

| Parameter | Default | Effect |
|---|---|---|
| `num_inference_steps` | 50 | Higher = better quality, slower |
| `guidance_scale` | 7.5 | Higher = follows instruction more aggressively |
| `image_guidance_scale` | 1.5 | Higher = preserves original image more |
| `seed` | -1 (random) | Fix for reproducible results |

**Tuning tip:** If the edit is too subtle, lower `image_guidance_scale` (try 1.0). If it changes too much, raise it (try 2.0–3.0).

### cURL example
```bash
# Encode image
IMAGE_B64=$(base64 -w 0 bryan_green.jpg)

curl -X POST http://localhost:8000/api/v1/edit \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$IMAGE_B64\", \"instruction\":\"change the green shirt to blue\", \"seed\":42}" \
  --output result.png
```

## Running

```bash
cd instructpix2pix
docker compose up -d
# First run downloads ~5 GB model from HuggingFace (cached after first run)
```

## Running Tests

```bash
docker cp /mnt/c/Users/bsald/Downloads/bryan_green.jpg instructpix2pix:/workspace/input/
docker exec instructpix2pix python3 /workspace/test_api.py
```

## Literature

**InstructPix2Pix: Learning to Follow Image Editing Instructions** (Brooks et al., 2022)  
- arXiv: https://arxiv.org/abs/2211.09800  
- GitHub: https://github.com/timothybrooks/instruct-pix2pix  
- HuggingFace: https://huggingface.co/timbrooks/instruct-pix2pix

**Key contributions:**
1. Generates training data entirely synthetically — GPT-3 writes edit captions, SD img2img applies them. No human annotation required.
2. Dual classifier-free guidance allows independent control of text and image fidelity.
3. Achieves high-quality edits with 50 inference steps at 512×512 in ~4s on modern GPUs.

## Resource Consumption

Measured on RTX 5070 Ti Laptop (12 GB VRAM, Blackwell SM 12.0), PyTorch 2.7.0+cu128, float16.

| Metric | Value |
|---|---|
| VRAM used (model loaded, idle) | ~5620 MB |
| VRAM free | ~6237 MB |
| RAM usage (idle) | ~3.2 GB |
| GPU utilization at rest | 1% |
| Inference time (460×460, 50 steps) | ~6–7s |
| Docker image size | 12.9 GB |

### Blackwell (RTX 50 series) Fix

PyTorch cu126 wheels only ship kernels up to SM 9.0 (Hopper). Blackwell (SM 12.0) requires **cu128 wheels**:

```dockerfile
# In Dockerfile — use this instead of cu126
RUN pip install torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

Also use `nvidia/cuda:12.8.1-base-ubuntu22.04` as the base image.
