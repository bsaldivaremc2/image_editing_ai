#!/usr/bin/env python3
"""
InstructPix2Pix API test.

Usage (inside container or host with requests/pillow installed):
    python3 test_api.py [--host http://localhost:8000] [--image /path/to/image.jpg]
"""

import argparse
import base64
import io
import json
import sys
import time

import requests
from PIL import Image


API_HOST = "http://localhost:8000"
IMAGE_PATH = "/workspace/input/bryan_green.jpg"

INSTRUCTIONS = [
    ("blue_shirt",    "change the green shirt to a blue shirt"),
    ("red_shirt",     "change the green shirt to a red shirt"),
    ("black_bg",      "make the background dark grey"),
    ("sunglasses",    "add sunglasses to the person"),
]


def wait_for_server(host: str, timeout: int = 180) -> bool:
    print(f"Waiting for server at {host}/health ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/health", timeout=5)
            if r.status_code == 200:
                print(f"  Server up: {r.json()}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    return False


def encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def test_health(host: str) -> None:
    print("\n[1/2] Health check")
    r = requests.get(f"{host}/health", timeout=10)
    assert r.status_code == 200, f"Expected 200, got {r.status_code}"
    info = r.json()
    print(f"  model  : {info['model']}")
    print(f"  device : {info['device']}")
    print(f"  dtype  : {info['dtype']}")
    print("  PASS")


def test_edit(host: str, image_path: str) -> None:
    print("\n[2/2] Edit tests")
    img_b64 = encode_image(image_path)

    for tag, instruction in INSTRUCTIONS:
        payload = {
            "image": img_b64,
            "instruction": instruction,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "image_guidance_scale": 1.5,
            "seed": 42,
        }
        t0 = time.time()
        r = requests.post(f"{host}/api/v1/edit", json=payload, timeout=300)
        elapsed = time.time() - t0

        print(f"  [{tag}] status={r.status_code}  time={elapsed:.1f}s  size={len(r.content)/1024:.0f}KB")
        if r.status_code != 200:
            print(f"    ERROR: {r.text[:300]}")
            sys.exit(1)

        out_path = f"/workspace/output/bryan_{tag}.png"
        with open(out_path, "wb") as f:
            f.write(r.content)
        print(f"    saved → {out_path}")

    print("  All edits PASS")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=API_HOST)
    parser.add_argument("--image", default=IMAGE_PATH)
    args = parser.parse_args()

    if not wait_for_server(args.host):
        print("ERROR: server did not start in time")
        sys.exit(1)

    test_health(args.host)
    test_edit(args.host, args.image)
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()
