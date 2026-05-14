#!/usr/bin/env bash
# start-live.sh -- start the live Dreamer-MC inference container.
#
# Usage:
#   start-live.sh                    # cold-start, steps_size=4 (recommended baseline)
#   start-live.sh 1                  # steps_size=1 (fast, hallucinates)
#   start-live.sh 2                  # steps_size=2 (faster, still unstable)
#
# Environment variables:
#   DREAMERV4_REPO         path to the repo on the host (default: $PWD)
#   DREAMERV4_CHECKPOINTS  path to checkpoints/ on the host (default: $DREAMERV4_REPO/checkpoints)
#   DREAMERV4_PORT         host port to expose (default: 8765)
#   DREAMERV4_CONTAINER    container name (default: dreamerv4-ngc)
#   DREAMERV4_IMAGE        container image (default: nvcr.io/nvidia/pytorch:26.03-py3)
#
# Boot takes ~30s for pip install + model load + CUDA graph capture.
# Tail logs:    docker logs -f $DREAMERV4_CONTAINER 2>&1 | grep -E 'TIMING|ACTION'
# Open:         http://localhost:$DREAMERV4_PORT (tunnel from your dev machine if remote)
set -euo pipefail

STEPS=${1:-4}

REPO=${DREAMERV4_REPO:-$PWD}
CHECKPOINTS=${DREAMERV4_CHECKPOINTS:-$REPO/checkpoints}
PORT=${DREAMERV4_PORT:-8765}
NAME=${DREAMERV4_CONTAINER:-dreamerv4-ngc}
IMAGE=${DREAMERV4_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}

if [[ ! -d "$REPO/src" || ! -d "$CHECKPOINTS" ]]; then
    echo "ERROR: expected repo at $REPO and checkpoints at $CHECKPOINTS"
    echo "       set DREAMERV4_REPO and DREAMERV4_CHECKPOINTS or run from the repo root"
    exit 2
fi

echo "[start-live] stopping any prior container..."
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[start-live] starting: steps_size=$STEPS"
docker run -d --name "$NAME" \
    --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p "$PORT:8765" \
    -v "$REPO:/workspace/dreamerv4-mc" \
    -v "$CHECKPOINTS:/workspace/dreamerv4-mc/checkpoints" \
    -w /workspace/dreamerv4-mc \
    "$IMAGE" \
    bash -lc "
        grep -v -E '^(torch|torchvision)==|^torch\$|^torchvision\$' requirements.txt > /tmp/requirements-no-torch.txt
        pip install -q -r /tmp/requirements-no-torch.txt
        pip install -q -e . --no-deps
        python ui/inference_ui.py \
            --dynamic_path=checkpoints/dynamic \
            --tokenizer_path=checkpoints/tokenizer \
            --port 8765 \
            --steps_size ${STEPS}
    " >/dev/null

echo "[start-live] container started; ~30s for pip + model load."
echo "[start-live] tail logs:  docker logs -f $NAME 2>&1 | grep -E 'TIMING|ACTION'"
echo "[start-live] browser:    http://localhost:$PORT"
