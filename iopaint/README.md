# IOPaint — Local Docker Deployment

## What It Does

IOPaint (formerly Lama Cleaner) is an inpainting and object removal tool. You paint a mask over the area to erase or replace; the model fills it in seamlessly. It bundles multiple backends selectable at runtime.

## How It Works

1. The FastAPI backend receives an image + mask pair.
2. The selected backend model generates a plausible fill for the masked region.
3. The React web UI provides a browser-based painting canvas; the REST API exposes the same functionality programmatically.

## Models / Backends

| Backend | VRAM | Prompt | Notes |
|---|---|---|---|
| `lama` | ~2 GB | No | Fast, structural fills, works great for object removal |
| `mat` | ~4 GB | No | Multi-scale adaptive transformer |
| `zits` | ~4 GB | No | Connects edges across mask boundary |
| `sd1.5` | ~8 GB | Yes | Stable Diffusion 1.5 inpainting |
| `sd2` | ~8 GB | Yes | Stable Diffusion 2 inpainting |
| `cv2` | CPU | No | OpenCV classical inpainting, no GPU needed |

## Ports

| Port | Service |
|---|---|
| 8080 | Web UI + REST API |
| 8080/docs | FastAPI Swagger UI |

## REST API

### Inpaint
```
POST /api/v1/inpaint
Content-Type: multipart/form-data

Files:
  image      — PNG/JPEG input image
  mask       — PNG mask (white = fill, black = keep)

Data:
  req        — JSON string with inference parameters (see below)
```

#### Minimal req JSON (LaMa)
```json
{
  "ldmSteps": 25,
  "hdStrategy": "Original",
  "prompt": "",
  "sdMaskBlur": 5
}
```

#### Response
`200 OK` — binary PNG of the inpainted result.

### Other Endpoints
```
GET  /               → Web UI
GET  /docs           → Swagger API docs
GET  /api/v1/model   → current model name
POST /api/v1/switch_model  → change model at runtime
```

## Running

### Quick start (default LaMa model)
```bash
docker compose up -d
```

### Switch to a different model
```bash
IOPAINT_MODEL=mat docker compose up -d
# or at runtime via API:
curl -X POST http://localhost:8080/api/v1/switch_model -H "Content-Type: application/json" -d '{"name":"mat"}'
```

### Run tests
```bash
pip install requests pillow numpy
python test/test_api.py
```

## Literature

- **LaMa** — Resolution-robust Large Mask inpainting with Fourier Convolutions (Suvorov et al., 2022).  
  Key advantage: Fourier convolutions give a theoretically infinite receptive field, enabling coherent fills across large masked areas without tiling artifacts.  
  Paper: https://arxiv.org/abs/2109.07161

- **IOPaint GitHub:** https://github.com/Sanster/IOPaint  
- **IOPaint Docs:** https://www.iopaint.com/

## Resource Consumption

Measured on RTX 5070 Ti Laptop (12 GB VRAM, Blackwell SM 12.0), IOPaint 1.6.0, PyTorch 2.7.0+cu126.

| Metric | LaMa (CPU) | Notes |
|---|---|---|
| RAM idle | ~744 MB | IOPaint server running |
| RAM peak (inference) | ~857 MB | 460×460 image |
| VRAM | 0 MB | CPU-only mode |
| Inference time (460×460) | 0.64s (avg) | 3-run benchmark |
| Docker image size | 11.1 GB | Base + PyTorch + IOPaint |

### Blackwell (RTX 50 series) GPU Limitation

`big-lama.pt` is a precompiled TorchScript model. Its CUDA kernels target older architectures (SM ≤ 8.x, i.e., Ampere and below). Blackwell (SM 12.0) is not included, resulting in:

```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

**Workaround options:**
1. **Run LaMa on CPU** (`--device=cpu`) — works, 0.64s on 460×460
2. **Use `--model=cv2`** — pure OpenCV inpainting, no GPU needed, instant
3. **Use `--model=sd1.5`** — Stable Diffusion 1.5 via diffusers (compiles for SM 12.0 at runtime), requires ~4 GB model download and ~8 GB VRAM
