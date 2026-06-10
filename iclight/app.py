"""
IC-Light — relighting via text prompt.

Architecture:
  - Base: SD 1.5 UNet patched from 4 → 8 input channels
  - IC-Light weights: delta-add over SD 1.5 (SD15 + ICL deltas)
  - Foreground conditioning: subject composited on gray → raw VAE latent (no scaling_factor)
  - Background removal: rembg (U2Net) extracts subject RGBA before conditioning
  - Inference: DDIMScheduler prediction_type='sample' (UNet output treated as x0)

Key inference discoveries:
  - IC-Light UNet outputs x0 predictions (std ~0.13), not epsilon (std ~1.0)
  - Use DDIMScheduler(prediction_type='sample') to match this behavior
  - fg conditioning uses raw VAE latent WITHOUT * scaling_factor
  - Weights are loaded as delta-add: new_weight = SD15_weight + IC-Light_delta
  - Output contrast boost to target std=0.25 for natural-looking results

Ports:
  - 8002  FastAPI REST API  (POST /api/v1/relight)
  - 7862  Gradio Web UI
"""

import io
import os
import base64
import threading

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from diffusers import StableDiffusionPipeline, DDIMScheduler
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import gradio as gr
import rembg

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda"
DTYPE  = torch.float16
API_PORT = int(os.environ.get("API_PORT", 8002))
UI_PORT  = int(os.environ.get("UI_PORT",  7862))

SD_MODEL_IDS = [
    os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5"),
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
]
IC_REPO     = "lllyasviel/ic-light"
IC_FILENAME = "iclight_sd15_fc.safetensors"   # foreground-conditioned variant

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print("Downloading IC-Light weights …", flush=True)
ic_model_path = hf_hub_download(repo_id=IC_REPO, filename=IC_FILENAME)
print(f"  Model cached at {ic_model_path}", flush=True)

print("Loading SD 1.5 base pipeline …", flush=True)
pipe = None
for model_id in SD_MODEL_IDS:
    try:
        print(f"  Trying {model_id} …", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        )
        print(f"  Loaded from {model_id}", flush=True)
        break
    except Exception as exc:
        print(f"  Failed: {exc}", flush=True)

if pipe is None:
    raise RuntimeError("Could not load any SD 1.5 base model.")

# DDIMScheduler with prediction_type='sample':
# IC-Light UNet outputs x0 predictions (std~0.13), not epsilon.
# This scheduler treats model output as the denoised image directly.
pipe.scheduler = DDIMScheduler.from_config(
    pipe.scheduler.config, prediction_type="sample"
)

# Patch UNet conv_in: 4 → 8 input channels (noise latent || fg latent).
# IC-Light's conv_in is the full 8-channel weight (shape changed, loaded absolute).
# All other layers are loaded as DELTAS added on top of SD 1.5.
print("Patching UNet conv_in to 8 channels …", flush=True)
with torch.no_grad():
    orig = pipe.unet.conv_in          # Conv2d(4, 320, 3, padding=1)
    new_conv = torch.nn.Conv2d(8, 320, 3, padding=1, dtype=DTYPE)
    new_conv.weight.zero_()
    new_conv.weight[:, :4].copy_(orig.weight)
    new_conv.bias.copy_(orig.bias)
    pipe.unet.conv_in = new_conv
    pipe.unet.config["in_channels"] = 8

# Load IC-Light weights as delta-add over SD 1.5:
#   conv_in: loaded absolutely (8-ch shape changed)
#   all others: SD15_weight + IC-Light_delta
print("Loading IC-Light UNet weights (delta-add over SD 1.5) …", flush=True)
ic_sd = load_file(ic_model_path)
current_sd = pipe.unet.state_dict()
for k, v in ic_sd.items():
    if k in ("conv_in.weight", "conv_in.bias"):
        current_sd[k] = v.to(DTYPE)
    elif k in current_sd:
        current_sd[k] = current_sd[k].to(DTYPE) + v.to(DTYPE)
    else:
        current_sd[k] = v.to(DTYPE)
missing, unexpected = pipe.unet.load_state_dict(current_sd, strict=False)
if missing:
    print(f"  Missing keys ({len(missing)}): {missing[:5]} …")
if unexpected:
    print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

pipe.to(DEVICE)
print("IC-Light ready on GPU.", flush=True)

# Background removal session (U2Net, CPU inference — fast enough)
rembg_session = rembg.new_session("u2net")

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def resize_for_model(img: Image.Image, max_px: int = 512) -> Image.Image:
    """Resize so longest side ≤ max_px and both dims are multiples of 8."""
    w, h = img.size
    scale = min(max_px / max(w, h), 1.0)
    nw = max(8, (int(w * scale) // 8) * 8)
    nh = max(8, (int(h * scale) // 8) * 8)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


def encode_foreground(rgba: Image.Image) -> torch.Tensor:
    """
    Composite RGBA on mid-gray (0.5) → normalize → raw VAE encode (no scaling_factor).

    IC-Light expects raw VAE latents for the fg conditioning channel.
    Using scaled latents (× scaling_factor) causes washed-out artifacts.
    Returns [1, 4, H//8, W//8] fg latent tensor on GPU.
    """
    arr = np.array(rgba.convert("RGBA")).astype(np.float32) / 255.0
    rgb   = arr[:, :, :3]
    alpha = arr[:, :, 3:4]
    gray  = np.full_like(rgb, 0.5)
    comp  = rgb * alpha + gray * (1.0 - alpha)          # composite on gray

    t = torch.from_numpy(comp).permute(2, 0, 1).unsqueeze(0).to(DEVICE, DTYPE)
    t = t * 2.0 - 1.0                                   # [−1, 1]
    with torch.no_grad():
        # Raw VAE latent — deliberately NOT multiplied by scaling_factor
        fg_lat = pipe.vae.encode(t).latent_dist.mean
    return fg_lat                                        # [1, 4, H//8, W//8]


def tensor_to_pil(t: torch.Tensor, contrast_boost: bool = True) -> Image.Image:
    """Decode VAE output tensor to PIL, optionally boosting contrast to std=0.25."""
    arr = (t / 2 + 0.5).clamp(0, 1)[0].permute(1, 2, 0).cpu().float().numpy()
    if contrast_boost and arr.std() > 0.01:
        factor = min(0.25 / arr.std(), 3.0)
        arr = np.clip(0.5 + (arr - arr.mean()) * factor, 0, 1)
    return Image.fromarray((arr * 255).round().astype(np.uint8))

# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def relight(
    image: Image.Image,
    prompt: str,
    remove_bg: bool = True,
    num_steps: int = 30,
    guidance_scale: float = 2.0,
    seed: int = -1,
) -> Image.Image:
    img = resize_for_model(image.convert("RGB"))
    W, H = img.size

    # Background removal
    if remove_bg:
        fg_rgba = rembg.remove(img.convert("RGBA"), session=rembg_session)
    else:
        fg_rgba = img.convert("RGBA")

    # Encode foreground: raw VAE latent, no scaling_factor
    fg_latents = encode_foreground(fg_rgba)              # [1,4,H//8,W//8]

    # Text embeddings — uncond FIRST (standard CFG convention)
    tok = pipe.tokenizer(
        ["", prompt],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        text_emb = pipe.text_encoder(**tok).last_hidden_state  # [2, 77, 768]

    # Initial noise latents
    if seed >= 0:
        torch.manual_seed(seed)
    latents = torch.randn((1, 4, H // 8, W // 8), device=DEVICE, dtype=DTYPE)
    pipe.scheduler.set_timesteps(num_steps, device=DEVICE)
    latents = latents * pipe.scheduler.init_noise_sigma

    for t in pipe.scheduler.timesteps:
        lat_input = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        fg_rep    = fg_latents.repeat(2, 1, 1, 1)
        model_in  = torch.cat([lat_input, fg_rep], dim=1)  # [2,8,H//8,W//8]

        with torch.no_grad():
            # UNet outputs x0 predictions (DDIM 'sample' mode)
            x0_pred = pipe.unet(
                model_in, t, encoder_hidden_states=text_emb
            ).sample                                           # [2,4,H//8,W//8]

        # CFG on x0: uncond=chunk[0], cond=chunk[1]
        x0_uncond, x0_cond = x0_pred.chunk(2)
        x0_guided = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

        latents = pipe.scheduler.step(x0_guided, t, latents).prev_sample

    with torch.no_grad():
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample

    return tensor_to_pil(decoded, contrast_boost=True)

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
api = FastAPI(title="IC-Light REST API", version="1.0")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class RelightRequest(BaseModel):
    image: str               # base64-encoded PNG or JPEG
    prompt: str
    remove_background: bool = True
    num_steps: int = 30
    guidance_scale: float = 2.0
    seed: int = -1


@api.get("/health")
def health():
    return {"status": "ok", "model": IC_FILENAME, "device": DEVICE}


@api.post("/api/v1/relight")
def api_relight(req: RelightRequest):
    raw = base64.b64decode(req.image)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    out = relight(
        img, req.prompt,
        remove_bg=req.remove_background,
        num_steps=req.num_steps,
        guidance_scale=req.guidance_scale,
        seed=req.seed,
    )
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def gradio_relight(image, prompt, remove_bg, steps, cfg, seed):
    if image is None:
        return None
    return relight(image, prompt, remove_bg=remove_bg,
                   num_steps=int(steps), guidance_scale=cfg, seed=int(seed))


EXAMPLE_PROMPTS = [
    "soft studio lighting from the left, professional portrait",
    "golden hour sunset lighting, warm orange tones",
    "dramatic cinematic lighting, dark moody background",
    "bright natural window light, soft diffused shadows",
    "neon city lights at night, blue and purple tones",
]

with gr.Blocks(title="IC-Light — AI Relighting") as ui:
    gr.Markdown("## IC-Light — Relight any subject with a text prompt")
    gr.Markdown(
        "Upload a photo, describe the lighting you want. "
        "**Remove Background** isolates the subject for clean relighting."
    )
    with gr.Row():
        inp = gr.Image(type="pil", label="Input Image")
        out = gr.Image(type="pil", label="Relit Output")

    prompt_box = gr.Textbox(
        value=EXAMPLE_PROMPTS[0],
        label="Lighting Prompt",
        placeholder="Describe the desired lighting …",
    )
    remove_bg_cb = gr.Checkbox(value=True, label="Remove Background (recommended)")

    with gr.Row():
        steps_sl = gr.Slider(10, 50, value=30, step=1, label="Steps")
        cfg_sl   = gr.Slider(1.0, 10.0, value=2.0, step=0.5, label="Guidance Scale")
        seed_nb  = gr.Number(value=42, label="Seed (-1 = random)")

    gr.Examples(
        examples=[[None, p, True, 30, 2.0, 42] for p in EXAMPLE_PROMPTS],
        inputs=[inp, prompt_box, remove_bg_cb, steps_sl, cfg_sl, seed_nb],
    )

    btn = gr.Button("Relight", variant="primary")
    btn.click(
        gradio_relight,
        inputs=[inp, prompt_box, remove_bg_cb, steps_sl, cfg_sl, seed_nb],
        outputs=[out],
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def _start_api():
    uvicorn.run(api, host="0.0.0.0", port=API_PORT, log_level="warning")


if __name__ == "__main__":
    threading.Thread(target=_start_api, daemon=True).start()
    ui.launch(server_name="0.0.0.0", server_port=UI_PORT)
