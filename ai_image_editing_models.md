# AI Image Editing Models — Local Deployment Research

**Date:** 2026-06-10  
**Target GPU:** NVIDIA GeForce RTX 5070 Ti Laptop (12 GB VRAM, Compute Cap 12.0 — Blackwell)  
**Driver:** 577.03 ✓ (560+ required for Blackwell)

---

## GPU Compatibility Notes

- Compute Capability 12.0 = Blackwell architecture (RTX 50 series)
- Requires **NVIDIA Driver 560+** — ✓ already satisfied
- Requires **CUDA 12.6+** for full Blackwell support
- 12 GB VRAM: can run most models with FP8/NF4 quantization; 24 GB+ recommended for full-precision FLUX
- Docker: need [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed

---

## Models & Tools

### 1. FLUX.1 Kontext Dev — Black Forest Labs
| Property | Value |
|---|---|
| **Task** | Image editing via text instructions (style, object, composition) |
| **License** | FLUX.1-dev Non-Commercial |
| **VRAM** | 12 GB (FP8), 24 GB (full FP16) |
| **Status** | ✅ Fits in 12 GB with FP8 |

**Install via ComfyUI (recommended):**
```bash
# Docker (Blackwell-ready)
git clone https://github.com/ChiefNakor/comfyui-blackwell-docker
docker compose up
# Then install ComfyUI-FLUX-Kontext custom node from the manager
```

**Install via diffusers (Python/pip):**
```bash
pip install diffusers transformers accelerate torch torchvision
# Model: black-forest-labs/FLUX.1-Kontext-dev (HuggingFace)
```

**Docker Hub:** `frefrik/comfyui-flux` (Docker Hub — FLUX ComfyUI setup)  
**References:** [HuggingFace model card](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) · [ComfyUI Wiki guide](https://comfyui-wiki.com/en/tutorial/advanced/image/flux/flux-1-kontext) · [Windows tutorial](https://github.com/FurkanGozukara/Stable-Diffusion/wiki/FLUX-Kontext-Dev-Detailed-Local-Windows-How-To-Tutorial-Better-Than-ChatGPT-and-Gemini-Image-Editing)

---

### 2. FLUX.1 Dev / Schnell — Black Forest Labs
| Property | Value |
|---|---|
| **Task** | Text-to-image generation (strong baseline for editing pipelines) |
| **License** | Apache 2.0 (Schnell) / Non-Commercial (Dev) |
| **VRAM** | 10 GB+ (Schnell FP8), 12 GB+ (Dev FP8) |
| **Status** | ✅ Fits in 12 GB with FP8 |

**Docker:**
```bash
docker pull frefrik/comfyui-flux
# or
docker pull ghcr.io/Condiolov/comfyui-flux-8GB-vram
```

**References:** [LocalAI Master guide](https://localaimaster.com/blog/flux-local-image-generation) · [Docker image](https://hub.docker.com/r/frefrik/comfyui-flux)

---

### 3. IOPaint — Sanster
| Property | Value |
|---|---|
| **Task** | Inpainting, outpainting, object removal/replacement |
| **License** | **Apache-2.0** (fully open) |
| **VRAM** | Varies by backend model (LaMa: ~2 GB, SD-based: 8 GB+) |
| **Status** | ✅ Works on 12 GB; supports LaMa for CPU-only too |

Bundles multiple backends: LaMa, MAT, Big-LaMa, Stable Diffusion Inpainting, BrushNet, and any HuggingFace SD inpainting model.

**Docker:**
```bash
docker run -d --gpus all -p 8080:8080 uiewy/iopaint
# or
docker run -d --gpus all -p 8080:8080 thr3a/iopaint:20250404
```

**Python/pip:**
```bash
pip install iopaint
iopaint start --model=lama --device=cuda --port=8080
```

**References:** [GitHub](https://github.com/Sanster/IOPaint) · [Official docs](https://www.iopaint.com/install) · [Docker Hub (uiewy)](https://hub.docker.com/r/uiewy/iopaint)

---

### 4. ComfyUI — comfyanonymous
| Property | Value |
|---|---|
| **Task** | Node-based workflow runner for SD, SDXL, FLUX, and any diffusion model |
| **License** | GPL-3.0 |
| **VRAM** | 8 GB+ (SDXL), 12 GB (FLUX FP8), 24 GB (FLUX full) |
| **Status** | ✅ Best all-around local UI; Blackwell Docker available |

ComfyUI is the recommended front-end for running any of the models below. It supports custom nodes for FLUX Kontext, IP-Adapters, ControlNet, BrushNet, etc.

**Docker (Blackwell / RTX 50 series):**
```bash
git clone https://github.com/ChiefNakor/comfyui-blackwell-docker
cd comfyui-blackwell-docker
docker compose up -d
```

**Python (manual):**
```bash
git clone https://github.com/comfyanonymous/ComfyUI
cd ComfyUI
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python main.py --cuda-device 0
```

**References:** [ComfyUI 2026 guide](https://localaimaster.com/blog/comfyui-complete-guide) · [Blackwell Docker](https://github.com/ChiefNakor/comfyui-blackwell-docker)

---

### 5. Stable Diffusion / SDXL Inpainting — Stability AI
| Property | Value |
|---|---|
| **Task** | Inpainting, outpainting, img2img editing |
| **License** | CreativeML OpenRAIL-M (SD 1.x/2.x), various (SDXL, SD3.5) |
| **VRAM** | 6 GB (SD 1.5), 8 GB (SDXL) |
| **Status** | ✅ Very comfortable on 12 GB |

Use via ComfyUI, A1111 Forge, or IOPaint. SDXL has dedicated inpainting checkpoints.

**Docker (A1111 Forge):**
```bash
docker pull ghcr.io/ai-dock/stable-diffusion-webui-forge:latest
docker run --gpus all -p 7860:7860 ghcr.io/ai-dock/stable-diffusion-webui-forge:latest
```

**References:** [Setup guide 2026](https://aumiqx.com/ai-tools/stable-diffusion-webui-setup-guide-2026/)

---

### 6. BrushNet / BrushEdit — TencentARC
| Property | Value |
|---|---|
| **Task** | Plug-and-play inpainting with fine pixel-level mask control |
| **License** | Apache-2.0 |
| **VRAM** | 12 GB+ (SD-based) |
| **Status** | ✅ Built into IOPaint; also has ComfyUI custom node |

BrushEdit surpasses BrushNet on both mask-based and random-mask benchmarks. Injects at every layer (encoder + mid + decoder) vs. ControlNet's decoder-only approach.

**Install via IOPaint (easiest):**
```bash
pip install iopaint
iopaint start --model=brushnet --device=cuda
```

**References:** [TencentARC project page](https://tencentarc.github.io/BrushNet/) · [IOPaint BrushNet docs](https://www.iopaint.com/models/diffusion/brushnet) · [BrushEdit paper](https://arxiv.org/html/2412.10316v2)

---

### 7. IC-Light (V2) — lllyasviel
| Property | Value |
|---|---|
| **Task** | Image relighting — change lighting conditions / backgrounds |
| **License** | Apache-2.0 |
| **VRAM** | ~8 GB |
| **Status** | ✅ Fits in 12 GB |

Two model variants: text-conditioned relighting and background-conditioned relighting.

**Python/conda:**
```bash
git clone https://github.com/lllyasviel/IC-Light
cd IC-Light
conda create -n iclight python=3.10
conda activate iclight
pip install torch torchvision
pip install -r requirements.txt
python gradio_demo.py   # models download automatically
```

**References:** [GitHub](https://github.com/lllyasviel/IC-Light) · [Clore.ai guide](https://docs.clore.ai/guides/image-processing/iclight)

---

### 8. InstructPix2Pix — timbrooks
| Property | Value |
|---|---|
| **Task** | Text-instruction guided pixel-level editing ("make it sunset", "add a hat") |
| **License** | MIT |
| **VRAM** | ~8 GB (SD 1.5 base) |
| **Status** | ✅ Fits in 12 GB |

Classic instruction-following editing model. Works via the diffusers pipeline out of the box.

**Python/pip:**
```bash
pip install diffusers accelerate safetensors transformers
```
```python
from diffusers import StableDiffusionInstructPix2PixPipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16
).to("cuda")
```

**References:** [HuggingFace model](https://huggingface.co/timbrooks/instruct-pix2pix) · [diffusers API docs](https://huggingface.co/docs/diffusers/en/api/pipelines/pix2pix) · [GitHub](https://github.com/timothybrooks/instruct-pix2pix)

---

### 9. FLUX.1-dev IP-Adapter — InstantX
| Property | Value |
|---|---|
| **Task** | Image-reference guided generation/editing (style/content transfer) |
| **License** | FLUX.1-dev Non-Commercial |
| **VRAM** | 12 GB+ (FP8 FLUX base required) |
| **Status** | ⚠️ Tight on 12 GB — may need CPU offload |

Adds image reference capability to FLUX.1-dev. Injects into 38 single + 19 double transformer blocks. Not yet in diffusers — use local files.

**Python:**
```bash
git clone https://huggingface.co/spaces/InstantX/flux-IP-adapter
cd flux-IP-adapter
pip install -r requirements.txt
```

**References:** [HuggingFace model](https://huggingface.co/InstantX/FLUX.1-dev-IP-Adapter) · [ComfyUI Wiki](https://comfyui-wiki.com/en/news/2024-11-22-instantx-flux-ipadapter-release)

---

### 10. OmniGen2 — VectorSpaceLab (June 2025)
| Property | Value |
|---|---|
| **Task** | Unified multimodal generation + instruction-based editing |
| **License** | To be confirmed (check HuggingFace) |
| **VRAM** | 17 GB native; **CPU offload** required for 12 GB |
| **Status** | ⚠️ Requires CPU offload on 12 GB GPU — may be slow |

Very new (released June 2025 at CVPR). Supports complex instruction-based edits with high precision.

**Python:**
```bash
git clone https://github.com/VectorSpaceLab/OmniGen2
cd OmniGen2
pip install -r requirements.txt
# Enable CPU offload in config for <17GB VRAM
```

**References:** [GitHub](https://github.com/VectorSpaceLab/OmniGen2) · [CVPR 2025 paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Xiao_OmniGen_Unified_Image_Generation_CVPR_2025_paper.pdf)

---

## Summary Table

| Model | Task | License | VRAM Fit (12 GB) | Docker | pip/conda |
|---|---|---|---|---|---|
| FLUX.1 Kontext Dev | Editing (text) | Non-commercial | ✅ FP8 | ✅ | ✅ diffusers |
| FLUX.1 Dev/Schnell | Generation base | Apache 2.0 / NC | ✅ FP8 | ✅ | ✅ diffusers |
| IOPaint | Inpaint/outpaint | **Apache-2.0** | ✅ | ✅ | ✅ pip |
| ComfyUI | Workflow runner | GPL-3.0 | ✅ | ✅ Blackwell | ✅ pip |
| SD / SDXL Inpaint | Inpaint/img2img | OpenRAIL-M | ✅ | ✅ | ✅ diffusers |
| BrushNet/BrushEdit | Inpainting | **Apache-2.0** | ✅ | via IOPaint | ✅ pip |
| IC-Light V2 | Relighting | **Apache-2.0** | ✅ | — | ✅ conda |
| InstructPix2Pix | Text editing | **MIT** | ✅ | — | ✅ pip |
| FLUX IP-Adapter | Style/ref transfer | Non-commercial | ⚠️ tight | — | ✅ pip |
| OmniGen2 | Unified editing | TBC | ⚠️ offload | — | ✅ pip |

---

## Recommended Starting Points

**Easiest all-in-one (inpainting/removal):**
```bash
docker run -d --gpus all -p 8080:8080 uiewy/iopaint
```

**Best workflow flexibility (FLUX + any model):**
```bash
git clone https://github.com/ChiefNakor/comfyui-blackwell-docker && docker compose up
```

**Quickest Python-only setup for text-based editing:**
```bash
pip install diffusers accelerate transformers torch torchvision
# Then use InstructPix2Pix or FLUX.1 Kontext via diffusers
```

---

## Blackwell (RTX 50 series) Notes

- Docker setup from [comfyui-blackwell-docker](https://github.com/ChiefNakor/comfyui-blackwell-docker) uses **4-bit NVFP4 quantization** specifically for RTX 50 series — significantly faster inference
- PyTorch must be 2.7+ with CUDA 12.8 for full Blackwell support
- Older Docker images that bake in CUDA 11.x will NOT work — verify the base image uses `nvidia/cuda:12.x`
- Install check: `docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi`

---

*Last updated: 2026-06-10*
