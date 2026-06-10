"""FLUX.1 Dev/Schnell text-to-image — FastAPI REST API + Gradio Web UI.

Model selection via FLUX_MODEL_ID env var:
  black-forest-labs/FLUX.1-schnell  (default, Apache-2.0, 4-step CFG-distilled)
  black-forest-labs/FLUX.1-dev      (non-commercial, 28-step, higher quality)

VRAM strategy: INT8 transformer quantization + model_cpu_offload → ~12 GB peak.
"""

import base64
import gc
import io
import os
import threading

import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import gradio as gr

from diffusers import FluxPipeline
from optimum.quanto import quantize, freeze, qint8

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID   = os.environ.get("FLUX_MODEL_ID", "black-forest-labs/FLUX.1-schnell")
DTYPE      = torch.bfloat16   # FLUX was trained with bfloat16
API_PORT   = int(os.environ.get("API_PORT", 8004))
UI_PORT    = int(os.environ.get("UI_PORT",  7864))
IS_SCHNELL = "schnell" in MODEL_ID.lower()

DEFAULT_STEPS   = 4   if IS_SCHNELL else 28
DEFAULT_CFG     = 0.0 if IS_SCHNELL else 3.5
MAX_STEPS_UI    = 8   if IS_SCHNELL else 50
DEFAULT_W       = 1024
DEFAULT_H       = 1024

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------

print(f"Loading {MODEL_ID} …", flush=True)
print("  Step 1/3: from_pretrained (CPU) …", flush=True)

pipe = FluxPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
)

print("  Step 2/3: INT8-quantize transformer …", flush=True)
quantize(pipe.transformer, weights=qint8)
freeze(pipe.transformer)

print("  Step 3/3: enable_model_cpu_offload + VAE tiling …", flush=True)
# model_cpu_offload moves each sub-model (text enc, transformer, VAE) to GPU
# only during its forward pass → peak VRAM = transformer only (~12 GB INT8).
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()    # handles large resolution images safely
pipe.enable_vae_slicing()   # further reduces VAE memory peak

print("Model ready.", flush=True)

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

_lock = threading.Lock()

def generate(
    prompt:         str,
    height:         int   = DEFAULT_H,
    width:          int   = DEFAULT_W,
    num_steps:      int   = DEFAULT_STEPS,
    guidance_scale: float = DEFAULT_CFG,
    seed:           int   = -1,
    max_seq_length: int   = 512,
) -> Image.Image:
    # Round dimensions to multiples of 64 (FLUX VAE requirement)
    height = max(256, (height // 64) * 64)
    width  = max(256, (width  // 64) * 64)

    generator = (
        torch.Generator("cpu").manual_seed(seed) if seed >= 0 else None
    )

    kwargs = dict(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        max_sequence_length=max_seq_length,
        generator=generator,
    )
    # Schnell is CFG-distilled; guidance_scale is not used for it.
    # Dev uses guidance_scale (default 3.5).
    if not IS_SCHNELL:
        kwargs["guidance_scale"] = guidance_scale

    with _lock:
        result = pipe(**kwargs)

    gc.collect()
    torch.cuda.empty_cache()
    return result.images[0]


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

api = FastAPI(title="FLUX.1 Image Generation API", version="1.0.0")
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt:         str
    height:         int   = DEFAULT_H
    width:          int   = DEFAULT_W
    num_steps:      int   = DEFAULT_STEPS
    guidance_scale: float = DEFAULT_CFG
    seed:           int   = -1
    max_seq_length: int   = 512


@api.get("/health")
def health():
    return {
        "status":  "ok",
        "model":   MODEL_ID,
        "variant": "schnell" if IS_SCHNELL else "dev",
        "device":  "cuda",
        "dtype":   "bfloat16",
        "quant":   "int8",
    }


@api.post("/api/v1/generate")
def api_generate(req: GenerateRequest) -> Response:
    img = generate(
        req.prompt, req.height, req.width,
        req.num_steps, req.guidance_scale,
        req.seed, req.max_seq_length,
    )
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

variant_label = "Schnell (4-step, fast)" if IS_SCHNELL else "Dev (28-step, quality)"

with gr.Blocks(title="FLUX.1 Text-to-Image") as demo:
    gr.Markdown(f"## FLUX.1 Text-to-Image — {variant_label}\n`{MODEL_ID}`")

    with gr.Row():
        with gr.Column(scale=1):
            prompt_box = gr.Textbox(
                label="Prompt",
                lines=4,
                placeholder="a professional headshot of a man in a white dress shirt, studio lighting, 8k, photorealistic",
            )
            with gr.Row():
                width_sl  = gr.Slider(256, 1536, value=DEFAULT_W, step=64, label="Width")
                height_sl = gr.Slider(256, 1536, value=DEFAULT_H, step=64, label="Height")
            with gr.Row():
                steps_sl = gr.Slider(1, MAX_STEPS_UI, value=DEFAULT_STEPS, step=1, label="Steps")
                cfg_sl   = gr.Slider(0.0, 10.0, value=DEFAULT_CFG, step=0.1,
                                     label="CFG Scale (dev only)")
            seed_box = gr.Number(label="Seed (-1 = random)", value=42, precision=0)
            gen_btn  = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Generated Image", type="pil")

    gen_btn.click(
        fn=lambda p, w, h, s, cfg, seed: generate(
            p, int(h), int(w), int(s), float(cfg), int(seed)
        ),
        inputs=[prompt_box, width_sl, height_sl, steps_sl, cfg_sl, seed_box],
        outputs=[output_img],
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _run_api():
    uvicorn.run(api, host="0.0.0.0", port=API_PORT, log_level="warning")


if __name__ == "__main__":
    api_thread = threading.Thread(target=_run_api, daemon=True)
    api_thread.start()
    print(f"FastAPI listening on :{API_PORT}", flush=True)
    demo.launch(server_name="0.0.0.0", server_port=UI_PORT, share=False)
