#!/usr/bin/env python3
"""
ComfyUI API test — uploads images and submits txt2img + inpaint workflows.

ComfyUI API endpoints:
  GET  /system_stats              → health / GPU info
  POST /upload/image              → upload input image/mask
  POST /prompt                    → queue a workflow
  GET  /history/{prompt_id}       → poll for completion
  GET  /view?filename=X&type=output → download result image
"""

import argparse
import base64
import io
import json
import sys
import time
import uuid

import requests
from PIL import Image, ImageDraw


HOST = "http://localhost:8188"
IMAGE_PATH = "/workspace/input/bryan_green.jpg"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_server(host: str, timeout: int = 240) -> bool:
    print(f"Waiting for ComfyUI at {host}/system_stats ...", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{host}/system_stats", timeout=5)
            if r.status_code == 200:
                info = r.json()
                devices = info.get("system", {}).get("cuda", {})
                print(f"  Server up — CUDA devices: {devices}")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)
    return False


def upload_image(host: str, image: Image.Image, filename: str) -> str:
    """Upload a PIL image to ComfyUI /upload/image. Returns filename used."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    r = requests.post(
        f"{host}/upload/image",
        files={"image": (filename, buf, "image/png")},
        data={"type": "input", "overwrite": "true"},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["name"]


def queue_workflow(host: str, workflow: dict) -> str:
    client_id = str(uuid.uuid4())
    r = requests.post(
        f"{host}/prompt",
        json={"prompt": workflow, "client_id": client_id},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()["prompt_id"]


def wait_for_result(host: str, prompt_id: str, timeout: int = 300) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = requests.get(f"{host}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            history = r.json()
            if prompt_id in history:
                return history[prompt_id]
        time.sleep(3)
    raise TimeoutError(f"Workflow {prompt_id} did not complete in {timeout}s")


def download_output(host: str, result: dict, save_path: str) -> bool:
    """Find the first output image in the result and save it."""
    for node_id, node_output in result.get("outputs", {}).items():
        if "images" in node_output:
            img_info = node_output["images"][0]
            filename = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            img_type = img_info.get("type", "output")
            params = f"filename={filename}&subfolder={subfolder}&type={img_type}"
            r = requests.get(f"{host}/view?{params}", timeout=30)
            r.raise_for_status()
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
    return False


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_system_stats(host: str) -> None:
    print("\n[1/3] System stats — GET /system_stats")
    r = requests.get(f"{host}/system_stats", timeout=10)
    assert r.status_code == 200
    stats = r.json()
    sys_info = stats.get("system", {})
    print(f"  Python  : {sys_info.get('python_version', '?')}")
    print(f"  PyTorch : {sys_info.get('torch_version', '?')}")
    cuda = sys_info.get("cuda", {})
    if cuda:
        for dev_id, dev_info in cuda.items():
            print(f"  GPU [{dev_id}]: {dev_info.get('name', '?')}  "
                  f"VRAM: {dev_info.get('vram_total', 0)//1024//1024} MB")
    print("  PASS")


def test_txt2img(host: str) -> None:
    print("\n[2/3] Txt2img workflow")
    with open("/comfyui_workflows/txt2img.json") as f:
        workflow = json.load(f)

    t0 = time.time()
    prompt_id = queue_workflow(host, workflow)
    print(f"  Queued prompt_id={prompt_id}")

    result = wait_for_result(host, prompt_id)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    out_path = "/workspace/output/comfy_txt2img.png"
    if download_output(host, result, out_path):
        size = len(open(out_path, "rb").read()) // 1024
        print(f"  Saved → {out_path}  ({size} KB)")
        print("  PASS")
    else:
        print("  ERROR: no output image in result")
        sys.exit(1)


def test_inpaint(host: str, image_path: str) -> None:
    print("\n[3/3] Inpaint workflow")

    # Prepare images
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([0, int(h * 0.60), w, h], fill=255)
    mask_rgb = mask.convert("RGB")

    # Upload to ComfyUI
    img_name  = upload_image(host, img,      "bryan_green.jpg")
    mask_name = upload_image(host, mask_rgb, "shirt_mask.png")
    print(f"  Uploaded image={img_name}  mask={mask_name}")

    # Patch workflow with uploaded filenames
    with open("/comfyui_workflows/inpaint.json") as f:
        workflow = json.load(f)
    workflow["4"]["inputs"]["image"] = img_name
    workflow["5"]["inputs"]["image"] = mask_name

    t0 = time.time()
    prompt_id = queue_workflow(host, workflow)
    print(f"  Queued prompt_id={prompt_id}")

    result = wait_for_result(host, prompt_id)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s")

    out_path = "/workspace/output/comfy_inpaint.png"
    if download_output(host, result, out_path):
        size = len(open(out_path, "rb").read()) // 1024
        print(f"  Saved → {out_path}  ({size} KB)")
        print("  PASS")
    else:
        print("  ERROR: no output image in result")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--image", default=IMAGE_PATH)
    args = parser.parse_args()

    if not wait_for_server(args.host):
        print("ERROR: server not ready"); sys.exit(1)

    test_system_stats(args.host)
    test_txt2img(args.host)
    test_inpaint(args.host, args.image)
    print("\nAll ComfyUI tests passed.")


if __name__ == "__main__":
    main()
