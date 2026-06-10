#!/usr/bin/env python3
"""
BrushNet API test — inpaints regions of bryan_green.jpg with 4 different prompts.

API:
  GET  /health                 → server status
  POST /api/v1/inpaint         → JSON body → binary PNG
    { image: base64, mask: base64, prompt: str,
      negative_prompt: str, num_steps: int, guidance_scale: float,
      brushnet_scale: float, seed: int }
"""

import argparse
import base64
import io
import os
import sys
import time

import numpy as np
import requests
from PIL import Image, ImageDraw

HOST       = "http://localhost:8003"
IMAGE_PATH = "/workspace/input/bryan_green.jpg"
OUT_DIR    = "/output"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def make_shirt_mask(w: int, h: int) -> Image.Image:
    """White rectangle covering ~lower 45% of image (shirt area)."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    top  = int(h * 0.55)
    draw.rectangle([0, top, w, h], fill=255)
    return mask


def make_center_mask(w: int, h: int) -> Image.Image:
    """White rectangle covering the center third of the image."""
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([w // 3, h // 3, 2 * w // 3, 2 * h // 3], fill=255)
    return mask


def make_full_bg_mask(w: int, h: int) -> Image.Image:
    """White everywhere except the face oval (~upper center)."""
    mask = Image.new("L", (w, h), 255)
    draw = ImageDraw.Draw(mask)
    cx, cy = w // 2, int(h * 0.35)
    rx, ry = int(w * 0.28), int(h * 0.35)
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=0)
    return mask


def wait_for_server(host: str, timeout: int = 300) -> bool:
    print(f"Waiting for BrushNet at {host}/health …", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/health", timeout=5)
            if r.status_code == 200:
                info = r.json()
                print(f"  Server ready — {info}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_health(host: str) -> None:
    print("\n[1/5] Health check — GET /health")
    r = requests.get(f"{host}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    print(f"  model={data['model']}  device={data['device']}")
    print("  PASS")


def inpaint_test(
    host: str,
    img: Image.Image,
    mask: Image.Image,
    test_id: str,
    prompt: str,
    num_steps: int = 50,
) -> None:
    print(f"\n[{test_id}] Prompt: {prompt!r}")
    t0 = time.time()
    r = requests.post(
        f"{host}/api/v1/inpaint",
        json={
            "image":           to_b64(img),
            "mask":            to_b64(mask),
            "prompt":          prompt,
            "negative_prompt": "worst quality, low quality, bad anatomy, distorted face",
            "num_steps":       num_steps,
            "guidance_scale":  7.5,
            "brushnet_scale":  0.5,
            "seed":            42,
        },
        timeout=300,
    )
    elapsed = time.time() - t0
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
    assert r.headers["content-type"] == "image/png", "Expected PNG response"

    out_path = os.path.join(OUT_DIR, f"brushnet_{test_id}.png")
    with open(out_path, "wb") as f:
        f.write(r.content)
    size_kb = len(r.content) // 1024
    print(f"  {elapsed:.1f}s → {out_path} ({size_kb} KB)  PASS")


def test_inpainting(host: str, image_path: str) -> None:
    print(f"\n[2/5 – 5/5] Inpainting tests (source: {image_path})")
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    print(f"  Source image: {w}×{h} px")

    cases = [
        ("2/5", "shirt_white",   make_shirt_mask(w, h),    "a crisp white dress shirt, professional portrait, studio lighting, 8k"),
        ("3/5", "shirt_navy",    make_shirt_mask(w, h),    "a navy blue suit jacket, business casual, soft lighting"),
        ("4/5", "shirt_hoodie",  make_shirt_mask(w, h),    "a casual black hoodie, soft natural lighting, realistic"),
        ("5/5", "bg_studio",     make_full_bg_mask(w, h),  "clean studio backdrop, gradient gray background, professional portrait"),
    ]
    for tid, name, mask, prompt in cases:
        inpaint_test(host, img, mask, name, prompt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",  default=HOST)
    parser.add_argument("--image", default=IMAGE_PATH)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if not wait_for_server(args.host):
        print("ERROR: BrushNet server not ready after timeout.")
        sys.exit(1)

    test_health(args.host)
    test_inpainting(args.host, args.image)

    print("\nAll BrushNet tests passed.")


if __name__ == "__main__":
    main()
