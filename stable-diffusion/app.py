"""
Stable Diffusion Inpainting — mask-guided image editing with text prompts.

Services (single process, two threads):
  Gradio Web UI  → port 7861  (paint-to-mask canvas + prompt)
  FastAPI REST   → port 8001  (/api/v1/inpaint, /health, /docs)

Supported models (set MODEL_ID env var):
  runwayml/stable-diffusion-inpainting           (default, SD 1.5, ~4 GB VRAM)
  stabilityai/stable-diffusion-2-inpainting      (SD 2.0, ~4 GB VRAM)
  diffusers/stable-diffusion-xl-1.0-inpainting-0.1 (SDXL, ~8 GB VRAM)
"""

import base64
import io
import os
import threading

import gradio as gr
import numpy as np
import torch
import uvicorn
from diffusers import StableDiffusionInpaintPipeline, StableDiffusionXLInpaintPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image, ImageChops
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "runwayml/stable-diffusion-inpainting")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
IS_XL = "xl" in MODEL_ID.lower()

print(f"Loading {MODEL_ID} ({'SDXL' if IS_XL else 'SD'}) on {DEVICE} ({DTYPE}) ...", flush=True)

if IS_XL:
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    )
else:
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, safety_checker=None
    )

pipe.to(DEVICE)
pipe.enable_attention_slicing()
print(f"Model ready — {MODEL_ID} on {DEVICE}", flush=True)


def prepare_inputs(image: Image.Image, mask: Image.Image):
    """Resize image and mask to the model's expected resolution."""
    target = 1024 if IS_XL else 512
    image = image.convert("RGB").resize((target, target), Image.LANCZOS)
    mask = mask.convert("L").resize((target, target), Image.NEAREST)
    return image, mask


def run_inference(
    image: Image.Image,
    mask: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    strength: float = 1.0,
    seed: int = -1,
) -> Image.Image:
    image, mask = prepare_inputs(image, mask)
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    kwargs = dict(
        prompt=prompt,
        image=image,
        mask_image=mask,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    if not IS_XL:
        kwargs["strength"] = strength

    result = pipe(**kwargs)
    return result.images[0]


# ---------------------------------------------------------------------------
# FastAPI REST API
# ---------------------------------------------------------------------------
api = FastAPI(
    title="Stable Diffusion Inpainting API",
    description=(
        f"Mask-guided inpainting with `{MODEL_ID}`.\n\n"
        "Send an image, a mask (white = region to fill), and a text prompt. "
        "Receive the inpainted image as PNG."
    ),
    version="1.0.0",
)


class InpaintRequest(BaseModel):
    image: str                      # base64 PNG/JPEG — original image
    mask: str                       # base64 PNG — white = fill, black = keep
    prompt: str = ""
    negative_prompt: str = "ugly, blurry, low quality, watermark"
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    strength: float = 1.0
    seed: int = -1


@api.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "dtype": str(DTYPE)}


@api.post(
    "/api/v1/inpaint",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    summary="Inpaint a masked region using a text prompt",
)
def inpaint(req: InpaintRequest):
    try:
        image = Image.open(io.BytesIO(base64.b64decode(req.image))).convert("RGB")
        mask = Image.open(io.BytesIO(base64.b64decode(req.mask))).convert("L")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image/mask: {exc}")

    result = run_inference(
        image, mask,
        req.prompt, req.negative_prompt,
        req.num_inference_steps, req.guidance_scale,
        req.strength, req.seed,
    )

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def start_api():
    uvicorn.run(api, host="0.0.0.0", port=8001, log_level="info")


# ---------------------------------------------------------------------------
# Gradio Web UI — paint-to-mask canvas
# ---------------------------------------------------------------------------
def extract_mask(editor_value) -> Image.Image | None:
    """Convert Gradio ImageEditor layers to a binary mask (white = inpaint)."""
    if editor_value is None:
        return None
    background = editor_value.get("background")
    if background is None:
        return None
    layers = editor_value.get("layers", [])
    mask = Image.new("L", background.size, 0)
    for layer in layers:
        if layer is not None:
            layer_rgba = layer.convert("RGBA")
            _, _, _, alpha = layer_rgba.split()
            mask = ImageChops.lighter(mask, alpha)
    return mask


def gradio_inpaint(editor_value, prompt, neg_prompt, steps, guidance, strength, seed):
    if editor_value is None:
        return None, "Upload an image and paint the area to edit."
    background = editor_value.get("background")
    if background is None:
        return None, "No image loaded."
    mask = extract_mask(editor_value)
    if mask is None or np.array(mask).max() == 0:
        return None, "No mask drawn — paint over the area you want to change."
    image = background.convert("RGB")
    result = run_inference(
        image, mask, prompt, neg_prompt,
        int(steps), guidance, strength, int(seed),
    )
    return result, "Done."


EXAMPLE_PROMPTS = [
    "a blue dress shirt, high quality, photorealistic",
    "a white t-shirt, casual, photorealistic",
    "a dark grey hoodie, photorealistic",
    "solid grey background, studio lighting",
]

with gr.Blocks(title="SD Inpainting") as ui:
    gr.Markdown(
        f"# Stable Diffusion Inpainting\n"
        f"**Model:** `{MODEL_ID}` · "
        f"**API:** http://localhost:8001/api/v1/inpaint · "
        f"[Swagger](http://localhost:8001/docs)\n\n"
        f"**How to use:** Upload an image, paint over the region to change with the brush, "
        f"enter a prompt for what should appear there, and click **Inpaint**."
    )
    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                type="pil",
                label="Upload image & paint mask (brush over area to inpaint)",
                brush=gr.Brush(default_size=24, colors=["#ffffff"], color_mode="fixed"),
            )
            prompt = gr.Textbox(
                label="Prompt — what to fill the masked area with",
                placeholder='e.g. "a blue dress shirt, photorealistic"',
                value="a blue dress shirt, high quality, photorealistic",
                lines=2,
            )
            neg_prompt = gr.Textbox(
                label="Negative prompt",
                value="ugly, blurry, low quality, watermark, deformed",
                lines=1,
            )
            with gr.Accordion("Advanced", open=False):
                steps = gr.Slider(10, 100, value=50, step=1, label="Steps")
                guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance scale")
                strength = gr.Slider(0.1, 1.0, value=1.0, step=0.05, label="Inpaint strength")
                seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
            btn = gr.Button("Inpaint", variant="primary")

        with gr.Column(scale=1):
            output = gr.Image(type="pil", label="Result")
            status = gr.Textbox(label="Status", interactive=False)

    btn.click(
        gradio_inpaint,
        inputs=[editor, prompt, neg_prompt, steps, guidance, strength, seed],
        outputs=[output, status],
    )

    gr.Markdown("### Example prompts")
    gr.Markdown("\n".join(f"- `{p}`" for p in EXAMPLE_PROMPTS))


if __name__ == "__main__":
    threading.Thread(target=start_api, daemon=True).start()
    ui.launch(server_name="0.0.0.0", server_port=7861)
