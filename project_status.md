# AI Image Editing — Local Docker Deployment

**Goal:** Deploy each AI image editing model as a self-contained Docker container exposing both a REST API and a Web UI. Each container must be tested against `bryan_green.jpg` before proceeding to the next model.

**Hardware:** NVIDIA RTX 5070 Ti Laptop · 12 GB VRAM · CUDA 12.9 · Driver 577.03 · Blackwell CC 12.0  
**OS:** WSL2 Ubuntu on Windows  
**Test image:** `/mnt/c/Users/bsald/Downloads/bryan_green.jpg` (460×460 JPEG, headshot)

---

## Model Deployment Order (Easiest → Hardest)

| # | Model | Task | VRAM Fit | API+UI | Status |
|---|---|---|---|---|---|
| 1 | **IOPaint** | Inpaint / object removal | ✅ LaMa CPU ~857 MB RAM | Native REST + React UI | ✅ COMPLETE |
| 2 | **InstructPix2Pix** | Text-instruction editing | ✅ ~5.6 GB VRAM GPU | FastAPI (8000) + Gradio (7860) | ✅ COMPLETE |
| 3 | **Stable Diffusion / SDXL** | Inpaint / img2img | ✅ ~8.3 GB VRAM GPU | Gradio (7861) + FastAPI (8001) | ✅ COMPLETE |
| 4 | **ComfyUI** | Node workflow (FLUX, SD, etc.) | ✅ 12 GB FP8 | Built-in ComfyUI UI + API | ✅ COMPLETE |
| 5 | **IC-Light** | Relighting | ✅ ~8 GB | Gradio (7862) + FastAPI (8002) | ✅ COMPLETE |
| 6 | **BrushNet** | Pixel-level inpainting | ✅ ~10 GB | Gradio (7863) + FastAPI (8003) | ✅ COMPLETE |
| 7 | **FLUX.1 Dev/Schnell** | Text-to-image base | ✅ INT8 ~12 GB | Gradio (7864) + FastAPI (8004) | 🔨 Ready to Build |
| 8 | **FLUX.1 Kontext Dev** | Text-guided editing | ✅ FP8 12 GB | ComfyUI or diffusers API | ⏳ Pending |
| 9 | **FLUX IP-Adapter** | Style/reference transfer | ⚠️ tight | Custom FastAPI + Gradio | ⏳ Pending |
| 10 | **OmniGen2** | Unified multimodal editing | ⚠️ CPU offload | Custom FastAPI + Gradio | ⏳ Pending |

---

## Model 1 — IOPaint

### Overview
IOPaint (formerly Lama Cleaner) is an open-source inpainting tool by Sanster. It ships a FastAPI backend + React web UI in a single process. The user paints a mask over the area to erase/replace and the model fills it in.

**Directory:** `./iopaint/`  
**Port:** 8080  
**License:** Apache-2.0  
**Default model:** LaMa (fast, low VRAM, no prompts needed)

### Goals
- [x] Create Dockerfile with CUDA 12.6 base + PyTorch + IOPaint
- [x] Create docker-compose.yml with GPU passthrough + volume mounts
- [x] Create test/test_api.py — REST API smoke test with bryan_green.jpg
- [x] `docker compose build` succeeds
- [x] `docker compose up` → container starts, GPU visible
- [x] Web UI reachable at http://localhost:8080
- [x] API test passes (inpainting returns a modified image)
- [x] Resource consumption documented

### Errors & Solutions

**Error 1 — Multipart form rejection (500 UnicodeDecodeError)**
- Cause: Initial test sent files as `multipart/form-data`; IOPaint 1.6 API expects `application/json` with base64-encoded image + mask.
- Fix: Rewrote test script to send JSON body with `base64.b64encode()`.

**Error 2 — PyTorch nightly build installed (2.12.0)**
- Cause: `pip install torch torchvision --index-url https://...cu126` without pinning resolves to a nightly/pre-release (2.12.0-dev) incompatible with LaMa TorchScript.
- Fix: Pinned `torch==2.7.0 torchvision==0.22.0` in Dockerfile. Rebuild confirmed 2.7.0+cu126.

**Error 3 — LaMa TorchScript incompatible with Blackwell GPU (SM 12.0) — KNOWN LIMITATION**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
- Cause: `big-lama.pt` is a precompiled TorchScript model. CUDA kernels were compiled for SM ≤ 8.x (Ampere and older). Blackwell (SM 12.0) is not included in the binary.
- Fix applied: Use `--device=cpu`. LaMa infers in 0.64s on CPU for 460×460 images — acceptable.
- GPU alternative: Switch to `--model=sd1.5` (diffusers-based, compiles kernels at runtime for SM 12.0).

### Resource Consumption (LaMa, CPU mode)

| Metric | Value |
|---|---|
| RAM idle | ~744 MB |
| RAM peak (460×460 inference) | ~857 MB |
| VRAM | 0 MB (CPU mode) |
| Inference time (avg, 460×460) | 0.64s |
| Docker image size | 11.1 GB |

### Test Results
- Web UI: ✅ http://localhost:8080 returns React HTML
- API inpaint: ✅ 200 OK, 149 KB PNG returned, saved to `iopaint/output/bryan_inpainted.png`
- Model info API: ✅ `/api/v1/model` and `/api/v1/server-config` return JSON
- GPU passthrough: ✅ `nvidia-smi` visible inside container
- Swagger docs: ✅ http://localhost:8080/docs

> **Global Blackwell (SM 12.0) fix for all GPU models:** Use `nvidia/cuda:12.8.1-base-ubuntu22.04` base image and `pip install torch --index-url https://download.pytorch.org/whl/cu128`. cu126 wheels only support up to SM 9.0 (Hopper).

---

## Model 2 — InstructPix2Pix

### Overview
timbrooks/instruct-pix2pix is an SD 1.5-based model trained to follow natural language editing instructions ("make it sunset", "remove the green shirt"). It uses classifier-free guidance on both text and image. We wrap it with FastAPI (REST API) + Gradio (Web UI).

**Directory:** `./instructpix2pix/`  
**Port:** 7860 (Gradio Web UI) + 8000 (FastAPI REST API + Swagger)  
**License:** MIT  
**Status:** ✅ Complete

### Goals
- [x] Build FastAPI REST API (`POST /api/v1/edit`)
- [x] Build Gradio 5 Web UI (port 7860)
- [x] Both served from single container/process via threads
- [x] `docker compose build` succeeds
- [x] GPU inference working on Blackwell
- [x] All 4 test edits pass (blue shirt, red shirt, black bg, sunglasses)
- [x] Resource consumption documented

### Errors & Solutions

**Error 1 — Gradio 4.21 + huggingface_hub 0.25 incompatibility**
- Cause: Gradio 4.21 imports `HfFolder` from `huggingface_hub`; removed in hub 0.25.
- Fix: Upgraded to `gradio>=5.0.0`.

**Error 2 — diffusers 0.27.2 + huggingface_hub 0.25 incompatibility**
- Cause: diffusers 0.27.2 imports `cached_download`; removed in hub 0.25.
- Fix: Upgraded to `diffusers` (latest, 0.38.0).

**Error 3 — PyTorch 2.7/cu126 has no SM 12.0 (Blackwell) kernels**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```
- Cause: cu126 wheels compile for SM 50/60/70/75/80/86/90 only. SM 12.0 not included.
- Fix: Switched to `nvidia/cuda:12.8.1-base-ubuntu22.04` + cu128 PyTorch wheels.
- **This fix applies to ALL remaining GPU models.**

### Resource Consumption

| Metric | Value |
|---|---|
| VRAM (model loaded) | ~5620 MB |
| VRAM free | ~6237 MB |
| RAM usage | ~3.2 GB |
| Inference time (50 steps, 460×460) | 6–7s |
| Docker image size | 12.9 GB |

### Test Results
- Web UI: ✅ http://localhost:7860 — Gradio 5 interface
- REST API: ✅ http://localhost:8000/api/v1/edit
- Swagger: ✅ http://localhost:8000/docs
- GPU: ✅ RTX 5070 Ti, float16, cu128
- Edits tested: blue shirt ✅, red shirt ✅, black background ✅, sunglasses ✅
- Output: `instructpix2pix/output/bryan_*.png`

---

## Model 3 — Stable Diffusion / SDXL Inpainting

### Overview
Mask-guided inpainting using `runwayml/stable-diffusion-inpainting` (SD 1.5). Custom container (same cu128 pattern) with Gradio 5 paint-to-mask canvas + FastAPI. SDXL switchable via `MODEL_ID` env var.

**Directory:** `./stable-diffusion/`  
**Ports:** 7861 (Gradio) + 8001 (FastAPI)  
**License:** OpenRAIL-M  
**Status:** ✅ Complete

### Goals
- [x] Gradio 5 UI with `gr.ImageEditor` brush/mask canvas (port 7861)
- [x] FastAPI REST API `POST /api/v1/inpaint` (port 8001)
- [x] GPU inference on Blackwell (cu128)
- [x] All 4 prompt tests pass
- [x] Resources documented

### Errors & Solutions
None — cu128 + Gradio 5 pattern carried over cleanly from Model 2.

### Resource Consumption

| Metric | SD 1.5 (GPU, float16) |
|---|---|
| VRAM used | ~8340 MB |
| VRAM free | ~3517 MB |
| RAM | ~4.1 GB |
| Inference time (30 steps) | 3–5s |
| Docker image size | 12.9 GB |

### Test Results
- Web UI: ✅ http://172.24.170.196:7861 — paint-to-mask Gradio canvas
- REST API: ✅ http://172.24.170.196:8001/api/v1/inpaint
- Swagger: ✅ http://172.24.170.196:8001/docs
- GPU: ✅ RTX 5070 Ti float16 cu128
- Tests: blue shirt ✅, white t-shirt ✅, dark hoodie ✅, no shirt ✅
- Output: `stable-diffusion/output/bryan_*.png`

---

## Model 4 — ComfyUI

### Overview
ComfyUI is a node-based visual workflow engine for Stable Diffusion and FLUX models. Each workflow is a JSON DAG of nodes (loaders, samplers, encoders, decoders, savers) submitted to a queue via REST API. The same port (8188) serves both the browser UI and the programmatic API — there is no separate service to run. Models can be swapped without rebuilding by dropping checkpoints into the `models/checkpoints` volume.

**Directory:** `./comfyui/`  
**Port:** 8188 (UI + API)  
**License:** GPL-3.0  
**Status:** ✅ Complete

### Goals
- [x] Dockerfile: cu128 + ComfyUI from source + SD 1.5 checkpoint (entrypoint.sh with fallback download chain)
- [x] docker-compose.yml with GPU passthrough + persistent model volume
- [x] `workflows/txt2img.json` — 20-step KSampler portrait generation
- [x] `workflows/inpaint.json` — VAEEncodeForInpaint + KSampler mask inpaint
- [x] `test/test_api.py` — upload image/mask → queue → poll → download
- [x] txt2img workflow passes on GPU
- [x] Inpaint workflow with bryan_green.jpg + shirt mask passes on GPU
- [x] Resource consumption documented

### How It Works
ComfyUI's REST API flow:
1. **Upload** images to `/upload/image` (multipart) → returns filename used by server
2. **Queue** a workflow to `POST /prompt` with `{"prompt": <dag>, "client_id": <uuid>}` → returns `prompt_id`
3. **Poll** `GET /history/{prompt_id}` until the key appears in the response
4. **Download** from `GET /view?filename=X&subfolder=&type=output`

Workflow nodes are JSON objects keyed by node ID. Nodes reference each other's outputs via `["node_id", output_index]` tuples.

### Errors & Solutions
None — carried over the cu128 + SD 1.5 pattern. SD 1.5 loads as a safetensors checkpoint (not TorchScript), so Blackwell GPU works without any workaround.

**Checkpoint download chain in entrypoint.sh:**
1. `Comfy-Org/stable-diffusion-v1-5-archive` (public, no auth) → `v1-5-pruned-emaonly-fp16.safetensors` ✅
2. `runwayml/stable-diffusion-v1-5` (fallback)
3. `stable-diffusion-v1-5/stable-diffusion-v1-5` (fallback)

### Resource Consumption

| Metric | Value |
|---|---|
| VRAM (SD 1.5 loaded) | ~3.5 GB |
| VRAM free | ~8.5 GB |
| RAM | ~2.5 GB |
| Txt2img (20 steps, 512×512) | 6.0s |
| Inpaint (25 steps, 460×460) | 3.0s |
| Docker image size | ~15 GB |
| SD 1.5 checkpoint | 2 GB (downloaded at first start) |

### API Reference

| Endpoint | Method | Purpose |
|---|---|---|
| `/system_stats` | GET | Health check, GPU info, PyTorch version |
| `/upload/image` | POST | Upload input image or mask (multipart) |
| `/prompt` | POST | Queue a workflow DAG |
| `/history/{prompt_id}` | GET | Poll for completion, get output filenames |
| `/view` | GET | Download output image by filename |
| `/object_info` | GET | List all available node types |

### Test Results
- Web UI: ✅ http://172.24.170.196:8188 — full ComfyUI node canvas
- Txt2img: ✅ 20-step portrait, 512×512, 6.0s → 335 KB PNG
- Inpaint: ✅ bryan_green.jpg + shirt mask, 25-step, 3.0s → 224 KB PNG
- GPU: ✅ RTX 5070 Ti, float16, cu128, no TorchScript issues
- Output: `comfyui/output/comfy_txt2img.png`, `comfyui/output/comfy_inpaint.png`

### Literature
- ComfyUI GitHub: https://github.com/comfyanonymous/ComfyUI
- The queue-based DAG API allows automated pipeline integration — same workflows usable from scripts as from the browser UI.
- SD 1.5 (RunwayML, 2022): latent diffusion via UNet + VAE. CVPR 2022 paper: "High-Resolution Image Synthesis with Latent Diffusion Models" (Rombach et al.).

---

## Model 5 — IC-Light

### Overview
lllyasviel's IC-Light relight model (foreground-conditioned variant `iclight_sd15_fc.safetensors`). The model modifies SD 1.5's UNet from 4 → 8 input channels, concatenating the noise latent with a foreground VAE latent at every denoising step. Text prompt steers the target lighting environment. Subject is extracted from background via rembg (U2Net) before conditioning.

**Directory:** `./iclight/`  
**Ports:** 7862 (Gradio Web UI) + 8002 (FastAPI REST API)  
**License:** Apache-2.0  
**Status:** ✅ COMPLETE

### Goals
- [x] Dockerfile: cu128 + SD 1.5 base + IC-Light safetensors + rembg
- [x] docker-compose.yml with GPU passthrough + hf_cache volume
- [x] FastAPI REST API `POST /api/v1/relight` (port 8002)
- [x] Gradio Web UI with lighting prompt + remove-BG toggle (port 7862)
- [x] Inference producing clean relighting on `bryan_green.jpg`
- [x] All 5 API tests pass (health + 4 lighting prompts)
- [x] Resources documented

### Architecture

**UNet patching (4 → 8 channels):**
The standard SD 1.5 `conv_in` (4→320) is zero-padded to (8→320). IC-Light's `conv_in.weight` (shape 8-channel) is loaded absolutely, replacing the full 8-channel weight.

**Delta-add weight loading:**
IC-Light stores weight *deltas* from SD 1.5, not absolute weights. The correct loading strategy is `new_weight = SD15_weight + IC-Light_delta` for all keys **except** `conv_in.*` (absolute). Loading IC-Light weights directly (without delta-add) produces near-zero UNet output (std ≈ 0.0003 instead of 0.13).

**x0 prediction with DDIMScheduler:**
The IC-Light UNet outputs x0 predictions (std ≈ 0.13 across all timesteps), not epsilon predictions (std ≈ 1.0). Using EulerAncestral or epsilon-mode DDIM causes latent divergence (latents exploding to ±100). Fix: `DDIMScheduler(prediction_type='sample')` which treats UNet output as x0 directly.

**Raw VAE foreground conditioning:**
Foreground conditioning uses `vae.encode(fg).latent_dist.mean` **without** `* scaling_factor`. Using the scaled version (std ≈ 0.81) causes color artifacts; raw latents (std ≈ 4.5) preserve subject structure correctly. Subject is composited on mid-gray (0.5) before encoding.

**CFG on x0:**
`x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)` where uncond embedding is index 0. Guidance scale must be low (2.0) — the x0-space correction is much larger per unit CFG than epsilon-space, so values like 7.0 cause over-correction artifacts.

**Contrast boost:**
IC-Light's DDIM inference converges to latents with std ≈ 0.10–0.17 (low contrast). Post-decoding boost: `factor = min(0.25 / std, 3.0); output = clip(0.5 + (raw - mean) * factor, 0, 1)`.

### Key Debugging Discoveries

| Issue | Symptom | Root Cause | Fix |
|---|---|---|---|
| Near-zero UNet output | std=0.0003, blank output | IC-Light stores deltas, not absolute weights | Delta-add: `SD15 + IC-Light_delta` |
| Latent divergence | ±100 explosion with EulerAncestral | UNet is x0-predictor, not epsilon | `DDIMScheduler(prediction_type='sample')` |
| Color mosaic artifacts | Colorful noise/patches on face | fg conditioning used `* scaling_factor` | Raw VAE latent, no `* scaling_factor` |
| Washed-out output | std ≈ 0.07, very pale | x0 latents converge to low contrast | Post-decode contrast boost to std=0.25 |
| Color artifacts at CFG=7 | Severe colored blotches on face | CFG=7.0 is too high for x0 prediction | Use CFG=2.0 for x0-space guidance |

### API Reference

**`GET /health`** → `{"status": "ok", "model": "iclight_sd15_fc.safetensors", "device": "cuda"}`

**`POST /api/v1/relight`**
```json
{
  "image": "<base64-encoded PNG or JPEG>",
  "prompt": "soft studio lighting from the left, professional portrait",
  "remove_background": true,
  "num_steps": 30,
  "guidance_scale": 2.0,
  "seed": 42
}
```
Returns: binary PNG (image/png)

### Resource Consumption

| Metric | Value |
|---|---|
| VRAM (model loaded) | ~8 GB |
| Inference time (30 steps, 512×512) | 1.4–1.7s |
| Docker image size | ~15 GB |

### Test Results
- Web UI: ✅ http://172.24.170.196:7862 — Gradio lighting editor
- REST API: ✅ http://172.24.170.196:8002/api/v1/relight
- Swagger: ✅ http://172.24.170.196:8002/docs
- GPU: ✅ RTX 5070 Ti float16 cu128
- Tests: studio ✅ (1.7s, 467 KB), sunset ✅ (1.7s, 459 KB), cinematic ✅ (1.6s, 474 KB), natural ✅ (1.4s, 499 KB)
- Output: `iclight/output/iclight_*.png`

### Literature
- IC-Light paper/repo: https://github.com/lllyasviel/IC-Light — foreground relighting via concatenated latent conditioning
- Inference mode: custom diffusers x0-prediction path; the original implementation uses k-diffusion ODE; both should produce equivalent results given correct scheduler setup.

---

## Model 6 — BrushNet

### Overview
TencentARC's BrushNet injects features at every encoder + mid + decoder layer of SD 1.5's UNet (unlike ControlNet which only injects at the decoder). Conditions on the original image + binary mask; the model learns to preserve masked-in regions and generate new content in masked-out regions. Best used for adding content to empty regions or background replacement — not for direct object replacement (which requires pre-erasing the original object).

**Directory:** `./brushnet/`  
**Ports:** 7863 (Gradio Web UI) + 8003 (FastAPI REST API)  
**License:** Apache-2.0  
**Status:** ✅ COMPLETE

### Goals
- [x] Dockerfile: cu128 + TencentARC/BrushNet fork of diffusers + SD 1.5
- [x] docker-compose.yml with GPU passthrough + shared iclight_hf_cache (avoids re-downloading SD 1.5)
- [x] FastAPI REST API `POST /api/v1/inpaint` (port 8003)
- [x] Gradio Web UI with image editor + mask painter (port 7863)
- [x] All 5 API tests pass (health + 4 inpainting prompts)
- [x] Resources documented

### Architecture

**BrushNet model**: Loaded from `Sanster/brushnet_random_mask` (public mirror of TencentARC/BrushNet). Accepts 5-channel conditioning: 4 VAE latent channels (original image) + 1 mask channel.

**Mask convention**: White (255) = inpaint region, Black (0) = preserve region. Internally: `mask = (sum_channels < 0)` after [-1,1] normalization, giving 1 for black (keep) and 0 for white (inpaint).

**Conditioning**: `conditioning_latents = concat([vae_encode(image), mask], dim=1)` — shape [B, 5, H/8, W/8]. BrushNet injects these features at every UNet layer.

**Key parameter**: `brushnet_conditioning_scale` controls preservation vs generation:
- 1.0 → strong preservation of original image (minimal visible change)
- 0.5 → balanced: visible texture/pattern changes while maintaining rough structure
- 0.3 → strong generation: visible content change but original identity may drift

### Dependency Challenges

**TencentARC/BrushNet is gated on HuggingFace**: 401 Unauthorized. Solution: use `Sanster/brushnet_random_mask` (public mirror, same weights).

**TencentARC/BrushNet repo IS a diffusers fork (0.27.0)**: Not a separate package — must install from their repo instead of official diffusers. `StableDiffusionBrushNetPipeline` is not in official diffusers 0.33.1.

**`cached_download` removed from huggingface_hub 0.25+**: The BrushNet fork imports it for community pipelines (unused in our app). Patched with `sed` in Dockerfile.

**Shared hf_cache**: Updated `docker-compose.yml` to use `iclight_hf_cache` (external volume) — SD 1.5 was already downloaded by Model 5, avoiding a 4 GB re-download.

### API Reference

**`GET /health`** → `{"status": "ok", "model": "Sanster/brushnet_random_mask", "device": "cuda"}`

**`POST /api/v1/inpaint`**
```json
{
  "image": "<base64 PNG/JPEG>",
  "mask": "<base64 PNG — white=inpaint, black=keep>",
  "prompt": "a crisp white dress shirt, professional portrait",
  "negative_prompt": "worst quality, low quality, bad anatomy",
  "num_steps": 50,
  "guidance_scale": 7.5,
  "brushnet_scale": 0.5,
  "seed": 42
}
```
Returns: binary PNG (image/png)

### Resource Consumption

| Metric | Value |
|---|---|
| VRAM (models loaded) | ~10 GB (SD 1.5 + BrushNet) |
| Inference time (50 steps, 460×460) | 6–8s |
| Docker image size | ~15 GB + SD 1.5 from shared cache |

### Test Results
- Web UI: ✅ http://172.24.170.196:7863 — Gradio mask painter
- REST API: ✅ http://172.24.170.196:8003/api/v1/inpaint
- Swagger: ✅ http://172.24.170.196:8003/docs
- GPU: ✅ RTX 5070 Ti float16 cu128
- Tests: shirt_white ✅ (6.7s), shirt_navy ✅ (6.3s), shirt_hoodie ✅ (6.7s), bg_studio ✅ (8.0s)
- Output: `brushnet/output/brushnet_*.png`
- Note: `brushnet_conditioning_scale=0.5` gives visible inpainting changes. Scale=1.0 preserves original too strongly; scale=0.3 loses subject identity.

---

## Model 7 — FLUX.1 Dev/Schnell

### Overview
Black Forest Labs FLUX.1 family — state-of-the-art text-to-image generation (12B parameter transformer, flow-matching). Schnell (Apache-2.0, 4-step CFG-distilled, fast) is the default; Dev (non-commercial, 28-step) can be selected via `FLUX_MODEL_ID` env var. Deployed with INT8-quantized transformer via `optimum-quanto` to fit 12 GB VRAM.

**Directory:** `./flux-dev/`  
**Ports:** 7864 (Gradio Web UI) + 8004 (FastAPI REST API)  
**License:** Apache-2.0 (Schnell) / Non-Commercial (Dev)  
**Status:** 🔨 Ready to Build — all files written, not yet tested

### Goals
- [x] Dockerfile: cu128 + PyTorch 2.7.0 + diffusers 0.33.1 + optimum-quanto + sentencepiece
- [x] docker-compose.yml with GPU passthrough + dedicated hf_cache volume
- [x] app.py: FluxPipeline + INT8 transformer quantization + model_cpu_offload + VAE tiling
- [x] FastAPI REST API `POST /api/v1/generate` (port 8004)
- [x] Gradio Web UI (port 7864)
- [x] test/test_api.py — 4 portrait generation prompts
- [ ] `docker compose build` succeeds
- [ ] `docker compose up` → FLUX.1-schnell downloaded (first run ~34 GB)
- [ ] All 5 API tests pass (health + 4 portrait prompts)
- [ ] Resources documented

### Architecture

**FLUX.1 transformer**: 12B parameter MM-DiT (Multi-Modal Diffusion Transformer). Uses T5-XXL (~9.5 GB FP16) for text encoding and CLIP-L for image pooled embeddings.

**VRAM strategy**:
- `optimum-quanto` INT8-quantizes transformer weights: 24 GB FP16 → ~12 GB INT8
- `pipe.enable_model_cpu_offload()`: each sub-model (CLIP, T5, transformer, VAE) moves to GPU only for its forward pass, then back to CPU
- Peak GPU VRAM = transformer (12 GB INT8) + activations for that timestep
- `enable_vae_tiling()` + `enable_vae_slicing()` prevent VAE OOM at 1024×1024

**Schnell vs Dev**:
- Schnell: CFG-distilled, `guidance_scale` not used, 4 steps sufficient, Apache-2.0
- Dev: classifier-free guidance (`guidance_scale=3.5`), 28 steps, non-commercial only

**T5 text encoder**: Handles long, complex prompts up to 512 tokens. FLUX generates significantly better results with detailed descriptive prompts than SD 1.5/SDXL.

### API Reference

**`GET /health`** → `{"status": "ok", "model": "...", "variant": "schnell|dev", "quant": "int8"}`

**`POST /api/v1/generate`**
```json
{
  "prompt": "a professional headshot of a man in a white shirt, studio lighting, 8k",
  "height": 1024,
  "width":  1024,
  "num_steps": 4,
  "guidance_scale": 0.0,
  "seed": 42,
  "max_seq_length": 512
}
```
Returns: binary PNG (image/png)

### Model Download Sizes (one-time)
| Component | Size |
|---|---|
| FLUX.1-schnell transformer | ~23.8 GB (FP16 safetensors) |
| T5-XXL text encoder | ~9.5 GB |
| CLIP-L | ~250 MB |
| VAE | ~340 MB |
| **Total** | **~34 GB** |

---

## Model 8 — FLUX.1 Kontext Dev

### Overview
FLUX.1 Kontext is Black Forest Labs' editing variant — takes an input image + text instruction and edits the image while preserving identity and context. FP8 fits in 12 GB.

**Directory:** `./flux-kontext/`  
**Port:** 7860  
**License:** Non-Commercial  
**Status:** Pending

---

## Model 9 — FLUX IP-Adapter

### Overview
InstantX FLUX IP-Adapter injects into 38 single + 19 double transformer blocks for style and reference-guided generation. Tight on 12 GB — needs CPU offload flags.

**Directory:** `./flux-ipadapter/`  
**Port:** 7860  
**License:** Non-Commercial  
**Status:** Pending

---

## Model 10 — OmniGen2

### Overview
VectorSpaceLab OmniGen2 (CVPR 2025). Unified multimodal generation + instruction-based editing. 17 GB native VRAM; requires `--offload_model=True` on 12 GB GPUs. Slowest on list.

**Directory:** `./omnigen2/`  
**Port:** 7860  
**License:** TBC  
**Status:** Pending

---

## Change Log

| Date | Model | Event |
|---|---|---|
| 2026-06-10 | All | Initial plan created |
| 2026-06-10 | IOPaint | Dockerfile, compose, tests created |
| 2026-06-10 | IOPaint | Built and started — PyTorch 2.7.0+cu126 |
| 2026-06-10 | IOPaint | LaMa GPU blocked by Blackwell TorchScript compat; switched to CPU mode |
| 2026-06-10 | IOPaint | All API + UI tests pass — COMPLETE |
| 2026-06-10 | InstructPix2Pix | Dockerfile + app.py + tests created |
| 2026-06-10 | InstructPix2Pix | Fixed Gradio/diffusers/huggingface_hub version cascade |
| 2026-06-10 | InstructPix2Pix | **Blackwell fix**: switched to cu128 PyTorch + CUDA 12.8.1 base |
| 2026-06-10 | InstructPix2Pix | All 4 GPU inference tests pass — COMPLETE |
| 2026-06-10 | StableDiffusion | Custom Dockerfile + app.py + tests — no new errors |
| 2026-06-10 | StableDiffusion | All 4 GPU inpainting tests pass — COMPLETE |
| 2026-06-10 | ComfyUI | Dockerfile + entrypoint.sh + workflows + test — built |
| 2026-06-10 | ComfyUI | SD 1.5 checkpoint downloaded (2 GB); GPU inference confirmed |
| 2026-06-10 | ComfyUI | txt2img (6.0s) + inpaint (3.0s) both pass — COMPLETE |
| 2026-06-10 | IC-Light | Dockerfile + app.py + tests created |
| 2026-06-10 | IC-Light | Fixed near-zero UNet output: delta-add weight loading (SD15 + IC-Light_delta) |
| 2026-06-10 | IC-Light | Fixed latent divergence: switched to DDIMScheduler prediction_type='sample' (x0-predictor) |
| 2026-06-10 | IC-Light | Fixed color artifacts: raw VAE fg latent without scaling_factor |
| 2026-06-10 | IC-Light | Fixed CFG artifacts: guidance_scale=2.0 (not 7.0) for x0-space guidance |
| 2026-06-10 | IC-Light | All 5 tests pass, 1.4–1.7s inference — COMPLETE |
| 2026-06-10 | BrushNet | Dockerfile + app.py + tests created |
| 2026-06-10 | BrushNet | Fixed: TencentARC/BrushNet repo is gated → use Sanster/brushnet_random_mask (public mirror) |
| 2026-06-10 | BrushNet | Fixed: diffusers fork install (not in official diffusers), cached_download patch |
| 2026-06-10 | BrushNet | Fixed: SD 1.5 re-download → shared iclight_hf_cache volume |
| 2026-06-10 | BrushNet | brushnet_conditioning_scale=0.5 gives visible inpainting; all 5 tests pass — COMPLETE |
| 2026-06-10 | FLUX.1 | Dockerfile + app.py + docker-compose.yml + test/test_api.py written — ready to build |
