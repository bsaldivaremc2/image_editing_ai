# Stable Diffusion Inpainting — Local Docker Deployment

## What It Does

Mask-guided inpainting: you paint a mask over the region to change, provide a text prompt describing what should appear there, and the model generates a photorealistic fill that blends seamlessly with the surrounding image. Unlike InstructPix2Pix (which edits globally), SD inpainting gives you **precise spatial control** over what gets changed.

## How It Works

Stable Diffusion Inpainting is a fine-tuned variant of SD 1.5 trained specifically on masked image/completion pairs. At inference time:

1. The original image is encoded into latent space by the VAE.
2. The masked region is **noised** according to the `strength` parameter (1.0 = fully noised = unconstrained fill).
3. The UNet denoises the masked latents conditioned on the text prompt, while the unmasked region is preserved by blending at each step.
4. The VAE decoder reconstructs the final image.

The key advantage over simple diffusion: the model sees **both** the masked region (to fill) and the unmasked context (to match), so it produces fills that are spatially coherent with the surrounding pixels.

## Supported Models

| MODEL_ID env var | Architecture | VRAM | Resolution | Quality |
|---|---|---|---|---|
| `runwayml/stable-diffusion-inpainting` (default) | SD 1.5 | ~4 GB | 512×512 | Good, fast |
| `stabilityai/stable-diffusion-2-inpainting` | SD 2.0 | ~4 GB | 512×512 | Similar to 1.5 |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | SDXL | ~8 GB | 1024×1024 | Best quality |

Switch model by editing `docker-compose.yml` → `MODEL_ID=...` and restarting.

## Ports

| Port | Service | URL |
|---|---|---|
| 7861 | Gradio Web UI (paint-to-mask canvas) | http://172.24.170.196:7861 |
| 8001 | FastAPI REST API | http://172.24.170.196:8001/api/v1/inpaint |
| 8001 | Swagger docs | http://172.24.170.196:8001/docs |

## Web UI

Open http://172.24.170.196:7861 in your browser:
1. Upload your image using the canvas component
2. Select the **brush tool** and paint white over the area to change
3. Enter a prompt (e.g. `"a blue dress shirt, photorealistic"`)
4. Click **Inpaint**

## REST API

```
POST http://172.24.170.196:8001/api/v1/inpaint
Content-Type: application/json
```

```json
{
  "image": "<base64 PNG/JPEG>",
  "mask":  "<base64 PNG — white = fill, black = keep>",
  "prompt": "a blue dress shirt, high quality, photorealistic",
  "negative_prompt": "ugly, blurry, low quality",
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "strength": 1.0,
  "seed": 42
}
```

**Response:** `200 OK` — binary PNG at the model's native resolution (512×512 for SD 1.5, 1024×1024 for SDXL).

### cURL example
```bash
IMG=$(base64 -w0 image.jpg)
MASK=$(base64 -w0 mask.png)
curl -X POST http://172.24.170.196:8001/api/v1/inpaint \
  -H "Content-Type: application/json" \
  -d "{\"image\":\"$IMG\",\"mask\":\"$MASK\",\"prompt\":\"a red hoodie\"}" \
  --output result.png
```

## Running

```bash
cd stable-diffusion
docker compose up -d
# First run downloads ~4 GB model (SD 1.5) or ~7 GB (SDXL)
```

## Resource Consumption

Measured on RTX 5070 Ti Laptop (12 GB VRAM), PyTorch 2.7.0+cu128, float16.

| Metric | SD 1.5 Inpainting |
|---|---|
| VRAM used (model loaded) | ~8340 MB |
| VRAM free | ~3517 MB |
| RAM | ~4.1 GB |
| Inference time (30 steps, 512×512) | ~3–5s |
| Docker image size | 12.9 GB |

## Literature

**Stable Diffusion (Rombach et al., 2022)** — Latent Diffusion Models for High-Resolution Image Synthesis  
- arXiv: https://arxiv.org/abs/2112.10752

**Inpainting fine-tune** — RunwayML fine-tuned SD 1.5 on paired (image, mask, caption) triplets using a modified UNet that receives 9-channel input (4 latent + 1 mask + 4 masked-latent channels).

Key benefits over classical inpainting (e.g. LaMa):
- **Text-conditioned**: you control *what* fills the masked area
- **Photorealistic textures**: diffusion generates detailed fabric, skin, materials
- **Context awareness**: surrounding pixels guide style and lighting
