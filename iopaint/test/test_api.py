#!/usr/bin/env python3
"""
IOPaint API test — uses bryan_green.jpg, masks the shirt area, and calls the inpaint endpoint.

Usage (inside container):
    python3 /workspace/test_api.py --host http://localhost:8080 --image /workspace/input/bryan_green.jpg
"""

import argparse
import base64
import io
import json
import sys
import time

import requests
from PIL import Image, ImageDraw


BASE_URL = "http://localhost:8080"
IMAGE_PATH = "/workspace/input/bryan_green.jpg"


def wait_for_server(host: str, timeout: int = 120) -> bool:
    print(f"Waiting for IOPaint server at {host} ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/", timeout=5)
            if r.status_code == 200:
                print("  Server is up.")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(3)
    return False


def make_shirt_mask(img: Image.Image) -> Image.Image:
    """White mask over the lower 35% (shirt area), black everywhere else."""
    w, h = img.size
    mask = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(mask)
    y_start = int(h * 0.65)
    draw.rectangle([0, y_start, w, h], fill=255)
    return mask


def img_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def test_health(host: str) -> None:
    print("\n[1/3] Health check — GET /")
    r = requests.get(f"{host}/", timeout=10)
    print(f"  Status: {r.status_code}")
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    print("  PASS")


def test_inpaint(host: str, image_path: str) -> None:
    print("\n[2/3] Inpaint test — POST /api/v1/inpaint")

    img = Image.open(image_path).convert("RGB")
    mask = make_shirt_mask(img)
    print(f"  Image size: {img.size}")
    print(f"  Mask: lower 35% white (shirt area)")

    payload = {
        "image": img_to_b64(img),
        "mask": img_to_b64(mask),
        "ldm_steps": 20,
        "hd_strategy": "Original",
        "hd_strategy_crop_trigger_size": 800,
        "hd_strategy_crop_margin": 128,
        "hd_strategy_resize_limit": 1280,
        "prompt": "",
        "negative_prompt": "",
    }

    t0 = time.time()
    response = requests.post(
        f"{host}/api/v1/inpaint",
        json=payload,
        timeout=120,
    )
    elapsed = time.time() - t0

    print(f"  Response status: {response.status_code}")
    print(f"  Inference time : {elapsed:.2f}s")
    print(f"  Content-Type   : {response.headers.get('content-type', 'unknown')}")

    if response.status_code != 200:
        print(f"  ERROR: {response.text[:500]}")
        sys.exit(1)

    # Response is binary image (PNG/JPEG)
    content = response.content
    print(f"  Response size  : {len(content) / 1024:.1f} KB")

    try:
        result_img = Image.open(io.BytesIO(content))
        out_path = "/workspace/output/bryan_inpainted.png"
        result_img.save(out_path)
        print(f"  Result saved   → {out_path}")
    except Exception:
        # Some versions return base64 JSON
        data = response.json()
        if isinstance(data, str):
            raw = base64.b64decode(data)
        else:
            raw = base64.b64decode(data.get("image", data))
        result_img = Image.open(io.BytesIO(raw))
        out_path = "/workspace/output/bryan_inpainted.png"
        result_img.save(out_path)
        print(f"  Result saved (b64) → {out_path}")

    mask_rgb = mask.convert("RGB")
    mask_out = "/workspace/output/bryan_mask.png"
    mask_rgb.save(mask_out)
    print(f"  Mask saved     → {mask_out}")
    print("  PASS")


def test_model_info(host: str) -> None:
    print("\n[3/3] Model info / server config")
    for path in ["/api/v1/model", "/api/v1/server-config"]:
        try:
            r = requests.get(f"{host}{path}", timeout=5)
            print(f"  GET {path} → {r.status_code}")
            if r.status_code == 200:
                try:
                    print(f"    {json.dumps(r.json(), indent=2)[:400]}")
                except Exception:
                    pass
        except Exception as e:
            print(f"  GET {path} → skipped ({e})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=BASE_URL)
    parser.add_argument("--image", default=IMAGE_PATH)
    args = parser.parse_args()

    if not wait_for_server(args.host):
        print("ERROR: server did not come up in time.")
        sys.exit(1)

    test_health(args.host)
    test_inpaint(args.host, args.image)
    test_model_info(args.host)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
