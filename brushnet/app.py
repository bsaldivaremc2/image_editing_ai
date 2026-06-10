"""
BrushNet — pixel-level mask-guided inpainting.

Architecture:
  - Base: SD 1.5 UNet
  - BrushNet: injects features at every encoder + mid + decoder layer (unlike ControlNet
    which only injects at decoder). Uses the masked image as conditioning.
  - Variant: random_mask — suitable for arbitrary user-drawn masks

Ports:
  - 8003  FastAPI REST API  (POST /api/v1/inpaint)
  - 7863  Gradio Web UI
"""

import io
import os
import base64
import threading

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler
from huggingface_hub import hf_hub_download
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import gradio as gr

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE   = "cuda"
DTYPE    = torch.float16
API_PORT = int(os.environ.get("API_PORT", 8003))
UI_PORT  = int(os.environ.get("UI_PORT",  7863))

SD_MODEL_IDS = [
    os.environ.get("SD_MODEL_ID", "runwayml/stable-diffusion-v1-5"),
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
]
BRUSHNET_REPO      = "Sanster/brushnet_random_mask"  # public mirror of TencentARC/BrushNet random_mask_brushnet_ckpt_v1_0
BRUSHNET_SUBFOLDER = None                              # weights at repo root

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
print("Loading BrushNet model …", flush=True)
load_kwargs = dict(torch_dtype=DTYPE)
if BRUSHNET_SUBFOLDER:
    load_kwargs["subfolder"] = BRUSHNET_SUBFOLDER
brushnet = BrushNetModel.from_pretrained(BRUSHNET_REPO, **load_kwargs)
print("  BrushNet weights loaded.", flush=True)

print("Loading SD 1.5 base pipeline …", flush=True)
pipe = None
for model_id in SD_MODEL_IDS:
    try:
        print(f"  Trying {model_id} …", flush=True)
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            model_id,
            brushnet=brushnet,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
            low_cpu_mem_usage=False,
        )
        print(f"  Loaded from {model_id}", flush=True)
        break
    except Exception as exc:
        print(f"  Failed: {exc}", flush=True)

if pipe is None:
    raise RuntimeError("Could not load SD 1.5 base model.")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to(DEVICE)
print("BrushNet ready on GPU.", flush=True)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def resize_for_model(img: Image.Image, max_px: int = 512) -> Image.Image:
    w, h = img.size
    scale = min(max_px / max(w, h), 1.0)
    nw = max(8, (int(w * scale) // 8) * 8)
    nh = max(8, (int(h * scale) // 8) * 8)
    if (nw, nh) != (w, h):
        img = img.resize((nw, nh), Image.LANCZOS)
    return img


def prepare_mask(mask: Image.Image, size: tuple) -> Image.Image:
    """Convert any mask input to L-mode at the target size. White=inpaint, black=keep."""
    mask = mask.convert("L").resize(size, Image.NEAREST)
    return mask


# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------
def inpaint(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "worst quality, low quality, bad anatomy",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    brushnet_scale: float = 1.0,
    seed: int = -1,
) -> Image.Image:
    image = resize_for_model(image.convert("RGB"))
    mask  = prepare_mask(mask, image.size)

    if seed >= 0:
        generator = torch.Generator(DEVICE).manual_seed(seed)
    else:
        generator = None

    with torch.no_grad():
        result = pipe(
            prompt,
            image,
            mask,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            brushnet_conditioning_scale=brushnet_scale,
            generator=generator,
        )
    return result.images[0]


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
api = FastAPI(title="BrushNet Inpainting API", version="1.0")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class InpaintRequest(BaseModel):
    image: str               # base64 PNG/JPEG — the original image
    mask: str                # base64 PNG/JPEG — white=inpaint, black=keep
    prompt: str
    negative_prompt: str = "worst quality, low quality, bad anatomy"
    num_steps: int = 50
    guidance_scale: float = 7.5
    brushnet_scale: float = 1.0
    seed: int = -1


@api.get("/health")
def health():
    return {"status": "ok", "model": BRUSHNET_REPO, "device": DEVICE}


@api.post("/api/v1/inpaint")
def api_inpaint(req: InpaintRequest):
    img  = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
    mask = Image.open(io.BytesIO(base64.b64decode(req.mask)))
    out  = inpaint(
        img, mask, req.prompt,
        negative_prompt=req.negative_prompt,
        num_steps=req.num_steps,
        guidance_scale=req.guidance_scale,
        brushnet_scale=req.brushnet_scale,
        seed=req.seed,
    )
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
def gradio_inpaint(image_dict, prompt, negative_prompt, steps, cfg, brushnet_scale, seed):
    """Gradio ImageEditor returns dict with 'background' and 'layers'."""
    if image_dict is None:
        return None
    bg    = image_dict.get("background")
    layer = image_dict.get("layers", [None])[0]
    if bg is None:
        return None
    bg_img = Image.fromarray(bg).convert("RGB") if not isinstance(bg, Image.Image) else bg
    if layer is not None:
        layer_img = Image.fromarray(layer) if not isinstance(layer, Image.Image) else layer
        # Extract alpha as mask (painted area = mask)
        if layer_img.mode == "RGBA":
            mask = layer_img.split()[3]
        else:
            mask = layer_img.convert("L")
    else:
        mask = Image.new("L", bg_img.size, 0)
    return inpaint(
        bg_img, mask, prompt,
        negative_prompt=negative_prompt,
        num_steps=int(steps),
        guidance_scale=cfg,
        brushnet_scale=brushnet_scale,
        seed=int(seed),
    )


EXAMPLE_PROMPTS = [
    "a crisp white dress shirt, professional portrait, studio lighting",
    "a navy blue suit jacket, business professional",
    "a casual black hoodie, soft natural lighting",
    "a bright red flannel shirt, warm color grading",
]

with gr.Blocks(title="BrushNet — Mask Inpainting") as ui:
    gr.Markdown("## BrushNet — Pixel-precise mask inpainting")
    gr.Markdown(
        "Paint a mask over the area to replace, then describe what you want there."
    )
    with gr.Row():
        inp = gr.ImageEditor(
            type="numpy",
            label="Paint mask on image (white brush = inpaint area)",
        )
        out = gr.Image(type="pil", label="Inpainted Output")

    prompt_box = gr.Textbox(
        value=EXAMPLE_PROMPTS[0],
        label="Inpainting Prompt",
        placeholder="Describe what should fill the masked area …",
    )
    neg_box = gr.Textbox(
        value="worst quality, low quality, bad anatomy",
        label="Negative Prompt",
    )

    with gr.Row():
        steps_sl   = gr.Slider(10, 100, value=50, step=1,   label="Steps")
        cfg_sl     = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="CFG Scale")
        bn_sl      = gr.Slider(0.1, 2.0, value=1.0, step=0.1,  label="BrushNet Scale")
        seed_nb    = gr.Number(value=42, label="Seed (-1 = random)")

    btn = gr.Button("Inpaint", variant="primary")
    btn.click(
        gradio_inpaint,
        inputs=[inp, prompt_box, neg_box, steps_sl, cfg_sl, bn_sl, seed_nb],
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
