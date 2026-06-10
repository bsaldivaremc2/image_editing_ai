#!/bin/bash
set -e

MODEL_DIR="/comfyui/models/checkpoints"
MODEL_FILE="$MODEL_DIR/v1-5-pruned-emaonly-fp16.safetensors"

if [ ! -f "$MODEL_FILE" ]; then
    echo "Downloading SD 1.5 checkpoint..."
    python3 - <<'PYEOF'
import sys
from huggingface_hub import hf_hub_download
import shutil, os

target = "/comfyui/models/checkpoints/v1-5-pruned-emaonly-fp16.safetensors"

# Try primary source (Comfy-Org archive — public, no auth required)
sources = [
    ("Comfy-Org/stable-diffusion-v1-5-archive", "v1-5-pruned-emaonly-fp16.safetensors"),
    ("runwayml/stable-diffusion-v1-5",           "v1-5-pruned-emaonly.safetensors"),
    ("stable-diffusion-v1-5/stable-diffusion-v1-5", "v1-5-pruned-emaonly.safetensors"),
]

for repo_id, filename in sources:
    try:
        print(f"  Trying {repo_id}/{filename} ...", flush=True)
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir="/root/.cache/huggingface",
        )
        shutil.copy(path, target)
        print(f"  Model saved to {target}")
        sys.exit(0)
    except Exception as e:
        print(f"  Failed: {e}", flush=True)

print("ERROR: Could not download model from any source.")
print("Manually place a .safetensors checkpoint at:")
print(f"  {target}")
print("Then restart the container.")
# Don't exit — start ComfyUI anyway so UI is accessible
PYEOF
fi

echo "Starting ComfyUI on :8188 ..."
cd /comfyui
exec python3 main.py \
    --cuda-device 0 \
    --listen 0.0.0.0 \
    --port 8188 \
    --enable-cors-header \
    --output-directory /workspace/output \
    --input-directory /workspace/input
