#!/usr/bin/env python3
"""
SD Inpainting API test — uses bryan_green.jpg with a shirt-area mask and several prompts.

Usage (inside container):
    python3 /workspace/test_api.py
"""

import argparse
import base64
import io
import sys
import time

import requests
from PIL import Image, ImageDraw


API_HOST = "http://localhost:8001"
IMAGE_PATH = "/workspace/input/bryan_green.jpg"

TESTS = [
    ("blue_shirt",  "a blue dress shirt, high quality, photorealistic",   ""),
    ("white_tshirt","a clean white t-shirt, casual, photorealistic",      ""),
    ("hoodie",      "a dark grey hoodie, photorealistic, detailed fabric",""),
    ("no_shirt",    "bare skin, natural, photorealistic",                 ""),
]


def wait_for_server(host: str, timeout: int = 180) -> bool:
    print(f"Waiting for server at {host}/health ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/health", timeout=5)
            if r.status_code == 200:
                info = r.json()
                print(f"  Up — model={info['model']}  device={info['device']}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    return False


def make_shirt_mask(img: Image.Image) -> Image.Image:
    """White mask over lower 40% (shirt area)."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([0, int(h * 0.60), w, h], fill=255)
    return mask


def b64(img: Image.Image, fmt="PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def test_health(host):
    print("\n[1/2] Health check")
    r = requests.get(f"{host}/health", timeout=10)
    assert r.status_code == 200
    info = r.json()
    print(f"  model : {info['model']}")
    print(f"  device: {info['device']}")
    print("  PASS")


def test_inpaint(host, image_path):
    print("\n[2/2] Inpaint tests")
    img = Image.open(image_path).convert("RGB")
    mask = make_shirt_mask(img)
    img_b64 = b64(img)
    mask_b64 = b64(mask)

    # Save mask for reference
    mask.save("/workspace/output/test_mask.png")

    for tag, prompt, neg in TESTS:
        payload = {
            "image": img_b64,
            "mask": mask_b64,
            "prompt": prompt,
            "negative_prompt": neg or "ugly, blurry, low quality, deformed",
            "num_inference_steps": 30,
            "guidance_scale": 7.5,
            "strength": 1.0,
            "seed": 42,
        }
        t0 = time.time()
        r = requests.post(f"{host}/api/v1/inpaint", json=payload, timeout=300)
        elapsed = time.time() - t0

        print(f"  [{tag:12s}] status={r.status_code}  time={elapsed:.1f}s  size={len(r.content)//1024}KB")
        if r.status_code != 200:
            print(f"    ERROR: {r.text[:300]}")
            sys.exit(1)

        out = f"/workspace/output/bryan_{tag}.png"
        with open(out, "wb") as f:
            f.write(r.content)
        print(f"    → {out}")

    print("\n  All inpaint tests PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=API_HOST)
    parser.add_argument("--image", default=IMAGE_PATH)
    args = parser.parse_args()

    if not wait_for_server(args.host):
        print("ERROR: server not ready"); sys.exit(1)

    test_health(args.host)
    test_inpaint(args.host, args.image)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
