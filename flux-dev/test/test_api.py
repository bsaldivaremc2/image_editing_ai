#!/usr/bin/env python3
"""
FLUX.1 API test — generates 4 portrait images to verify the container is working.

API:
  GET  /health              → server status
  POST /api/v1/generate     → JSON body → binary PNG
    { prompt: str, height: int, width: int,
      num_steps: int, guidance_scale: float, seed: int,
      max_seq_length: int }

Schnell default: 4 steps, guidance_scale=0.0 (CFG-distilled).
"""

import argparse
import io
import os
import sys
import time

import requests
from PIL import Image

HOST    = "http://localhost:8004"
OUT_DIR = "/output"


def wait_for_server(host: str, timeout: int = 600) -> bool:
    print(f"Waiting for FLUX.1 at {host}/health …", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/health", timeout=5)
            if r.status_code == 200:
                info = r.json()
                print(f"  Server ready — model={info['model']}  variant={info['variant']}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(10)
    return False


def test_health(host: str) -> None:
    print("\n[1/5] Health check — GET /health")
    r = requests.get(f"{host}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    data = r.json()
    assert data["status"] == "ok"
    print(f"  variant={data['variant']}  quant={data['quant']}  device={data['device']}")
    print("  PASS")


def generate_test(
    host: str,
    test_id: str,
    name: str,
    prompt: str,
    num_steps: int = 4,
    guidance_scale: float = 0.0,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
) -> None:
    print(f"\n[{test_id}] {name}")
    print(f"  Prompt: {prompt[:80]}…" if len(prompt) > 80 else f"  Prompt: {prompt}")
    t0 = time.time()
    r = requests.post(
        f"{host}/api/v1/generate",
        json={
            "prompt":          prompt,
            "height":          height,
            "width":           width,
            "num_steps":       num_steps,
            "guidance_scale":  guidance_scale,
            "seed":            seed,
            "max_seq_length":  512,
        },
        timeout=600,
    )
    elapsed = time.time() - t0
    assert r.status_code == 200, f"Status {r.status_code}: {r.text[:300]}"
    assert r.headers.get("content-type") == "image/png", \
        f"Expected image/png, got {r.headers.get('content-type')}"

    # Verify output is a valid non-black image
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    import numpy as np
    arr = np.array(img)
    assert arr.std() > 5.0, "Output image is nearly uniform (black/white) — inference failed"

    out_path = os.path.join(OUT_DIR, f"flux_{name}.png")
    with open(out_path, "wb") as f:
        f.write(r.content)
    size_kb = len(r.content) // 1024
    print(f"  {elapsed:.1f}s → {out_path} ({img.width}×{img.height}, {size_kb} KB)  PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",    default=HOST)
    parser.add_argument("--steps",   type=int,   default=4,   help="inference steps")
    parser.add_argument("--cfg",     type=float, default=0.0, help="guidance scale")
    parser.add_argument("--width",   type=int,   default=1024)
    parser.add_argument("--height",  type=int,   default=1024)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    if not wait_for_server(args.host):
        print("ERROR: FLUX.1 server not ready after timeout.")
        sys.exit(1)

    test_health(args.host)

    cases = [
        (
            "2/5", "portrait_studio",
            "a professional headshot of a man with short dark hair, white dress shirt, "
            "clean studio backdrop, soft box lighting, sharp focus, 8k resolution, photorealistic",
        ),
        (
            "3/5", "portrait_outdoor",
            "a candid portrait of a man in a green t-shirt standing outdoors in natural sunlight, "
            "bokeh background, authentic expression, high quality DSLR photography",
        ),
        (
            "4/5", "portrait_suit",
            "professional business portrait of a man in a navy blue suit, "
            "confident expression, shallow depth of field, corporate headshot, "
            "warm lighting, ultra detailed",
        ),
        (
            "5/5", "portrait_dramatic",
            "dramatic cinematic portrait of a man with side lighting, moody atmosphere, "
            "deep shadows, rim light, film grain, anamorphic lens, high contrast, "
            "editorial photography style",
        ),
    ]

    for tid, name, prompt in cases:
        generate_test(
            args.host, tid, name, prompt,
            num_steps=args.steps,
            guidance_scale=args.cfg,
            width=args.width,
            height=args.height,
        )

    print(f"\nAll FLUX.1 tests passed.  Outputs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
