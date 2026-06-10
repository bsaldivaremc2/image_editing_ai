#!/usr/bin/env python3
"""
IC-Light API test — relights bryan_green.jpg with 4 different lighting prompts.

API:
  GET  /health                      → server status
  POST /api/v1/relight              → JSON body → binary PNG
    { image: base64, prompt: str, remove_background: bool,
      num_steps: int, guidance_scale: float, seed: int }
"""

import argparse
import base64
import io
import os
import sys
import time

import requests
from PIL import Image

HOST       = "http://localhost:8002"
IMAGE_PATH = "/workspace/input/bryan_green.jpg"
OUT_DIR    = "/output"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def wait_for_server(host: str, timeout: int = 300) -> bool:
    print(f"Waiting for IC-Light at {host}/health …", flush=True)
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


def relight_test(
    host: str,
    img: Image.Image,
    test_id: str,
    prompt: str,
    remove_bg: bool = True,
    num_steps: int = 20,
) -> None:
    print(f"\n  Prompt: {prompt!r}  remove_bg={remove_bg}")
    t0 = time.time()
    r = requests.post(
        f"{host}/api/v1/relight",
        json={
            "image": to_b64(img),
            "prompt": prompt,
            "remove_background": remove_bg,
            "num_steps": num_steps,
            "guidance_scale": 2.0,
            "seed": 42,
        },
        timeout=180,
    )
    elapsed = time.time() - t0
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:200]}"
    assert r.headers["content-type"] == "image/png", "Expected PNG response"

    out_path = os.path.join(OUT_DIR, f"iclight_{test_id}.png")
    with open(out_path, "wb") as f:
        f.write(r.content)
    size_kb = len(r.content) // 1024
    print(f"  {elapsed:.1f}s → {out_path} ({size_kb} KB)  PASS")


def test_relighting(host: str, image_path: str) -> None:
    print(f"\n[2/5 – 5/5] Relighting tests (source: {image_path})")
    img = Image.open(image_path).convert("RGB")
    print(f"  Source image: {img.size[0]}×{img.size[1]} px")

    cases = [
        ("studio",   "soft studio lighting from the left, professional portrait",    True),
        ("sunset",   "golden hour sunset lighting, warm orange tones",               True),
        ("cinematic","dramatic cinematic lighting, dark moody background",           True),
        ("natural",  "bright natural window light, soft diffused shadows",           False),
    ]
    for test_id, prompt, remove_bg in cases:
        relight_test(host, img, test_id, prompt, remove_bg=remove_bg)


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
        print("ERROR: IC-Light server not ready after timeout.")
        sys.exit(1)

    test_health(args.host)
    test_relighting(args.host, args.image)

    print("\nAll IC-Light tests passed.")


if __name__ == "__main__":
    main()
