"""
InstructPix2Pix — text-instruction guided image editing.

Runs two services in one process:
  - FastAPI REST API  →  port 8000  (/api/v1/edit, /health, /docs)
  - Gradio Web UI     →  port 7860  (/)
"""

import base64
import io
import os
import threading

import gradio as gr
import torch
import uvicorn
from diffusers import EulerAncestralDiscreteScheduler, StableDiffusionInstructPix2PixPipeline
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "timbrooks/instruct-pix2pix")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

print(f"Loading {MODEL_ID} on {DEVICE} ({DTYPE}) ...", flush=True)

pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    safety_checker=None,
)
pipe.to(DEVICE)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_attention_slicing()

print(f"Model ready — device={DEVICE}, dtype={DTYPE}", flush=True)


def run_inference(
    image: Image.Image,
    instruction: str,
    num_steps: int = 50,
    guidance_scale: float = 7.5,
    image_guidance_scale: float = 1.5,
    seed: int = -1,
) -> Image.Image:
    generator = None
    if seed >= 0:
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

    result = pipe(
        instruction,
        image=image,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        generator=generator,
    )
    return result.images[0]


# ---------------------------------------------------------------------------
# FastAPI REST API
# ---------------------------------------------------------------------------
api = FastAPI(
    title="InstructPix2Pix API",
    description=(
        "Text-instruction guided image editing using timbrooks/instruct-pix2pix.\n\n"
        "Send an image as base64 + a natural-language instruction. "
        "Receive the edited image as a PNG binary response."
    ),
    version="1.0.0",
)


class EditRequest(BaseModel):
    image: str
    instruction: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    seed: int = -1


class EditResponse(BaseModel):
    pass  # response is binary PNG


@api.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": DEVICE, "dtype": str(DTYPE)}


@api.post(
    "/api/v1/edit",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
    summary="Edit an image with a text instruction",
)
def edit_image(req: EditRequest):
    try:
        raw = base64.b64decode(req.image)
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")

    result = run_inference(
        image,
        req.instruction,
        req.num_inference_steps,
        req.guidance_scale,
        req.image_guidance_scale,
        req.seed,
    )

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def start_api():
    uvicorn.run(api, host="0.0.0.0", port=8000, log_level="info")


# ---------------------------------------------------------------------------
# Gradio Web UI
# ---------------------------------------------------------------------------
def gradio_edit(image, instruction, num_steps, guidance_scale, image_guidance_scale, seed):
    if image is None:
        return None, "No image provided."
    result = run_inference(
        image, instruction, int(num_steps), guidance_scale, image_guidance_scale, int(seed)
    )
    return result, "Done."


with gr.Blocks(title="InstructPix2Pix — Image Editing") as ui:
    gr.Markdown("# InstructPix2Pix")
    gr.Markdown(
        "Upload an image and type a natural-language instruction to edit it.\n\n"
        "> _API available at_ **http://localhost:8000/api/v1/edit** · "
        "[Swagger docs](http://localhost:8000/docs)"
    )

    with gr.Row():
        with gr.Column(scale=1):
            inp_image = gr.Image(type="pil", label="Input image")
            instruction = gr.Textbox(
                label="Edit instruction",
                placeholder='e.g. "change the shirt to red" or "add sunglasses"',
                value="change the green shirt to a blue shirt",
                lines=2,
            )
            with gr.Accordion("Advanced", open=False):
                num_steps = gr.Slider(10, 150, value=50, step=1, label="Inference steps")
                guidance = gr.Slider(1.0, 20.0, value=7.5, step=0.5, label="Guidance scale")
                img_guidance = gr.Slider(1.0, 5.0, value=1.5, step=0.1, label="Image guidance scale")
                seed = gr.Number(value=42, precision=0, label="Seed (-1 = random)")
            btn = gr.Button("Edit", variant="primary")

        with gr.Column(scale=1):
            out_image = gr.Image(type="pil", label="Edited image")
            status = gr.Textbox(label="Status", interactive=False)

    btn.click(
        gradio_edit,
        inputs=[inp_image, instruction, num_steps, guidance, img_guidance, seed],
        outputs=[out_image, status],
    )


if __name__ == "__main__":
    # Start REST API in a background thread
    api_thread = threading.Thread(target=start_api, daemon=True)
    api_thread.start()

    # Start Gradio in the main thread
    ui.launch(server_name="0.0.0.0", server_port=7860)
