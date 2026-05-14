#!/usr/bin/env bash
# start-capture.sh -- start the live container in CAPTURE MODE.
#
# Records every (frame, action_id) pair generated during play into a .pt file
# plus a sidecar mp4 for visual inspection. Used to seed offline renders with
# real human-played KV cache state instead of a synthetic prefill.
#
# Usage:
#   start-capture.sh                    # steps_size=4, max 200 frames (~10s @ 20fps)
#   start-capture.sh 4 400              # steps_size=4, max 400 frames (~20s)
#
# Environment variables:
#   DREAMERV4_REPO         path to the repo on the host (default: $PWD)
#   DREAMERV4_CHECKPOINTS  path to checkpoints/ on the host (default: $DREAMERV4_REPO/checkpoints)
#   DREAMERV4_PORT         host port to expose (default: 8765)
#   DREAMERV4_CONTAINER    container name (default: dreamerv4-ngc)
#   DREAMERV4_IMAGE        container image (default: nvcr.io/nvidia/pytorch:26.03-py3)
#
# Captures saved to $DREAMERV4_REPO/.captures/ as:
#   capture-<timestamp>.pt   frames + actions
#   capture-<timestamp>.mp4  sidecar for inspection
#   latest.pt                always points at the most recent
#
# After capture completes, replay it offline with:
#   render-replay.sh latest.pt <record_steps> <record_frames>
set -euo pipefail

STEPS=${1:-4}
MAX=${2:-200}

REPO=${DREAMERV4_REPO:-$PWD}
CHECKPOINTS=${DREAMERV4_CHECKPOINTS:-$REPO/checkpoints}
PORT=${DREAMERV4_PORT:-8765}
NAME=${DREAMERV4_CONTAINER:-dreamerv4-ngc}
IMAGE=${DREAMERV4_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}

CAP_HOST="$REPO/.captures"
CAP_GUEST="/workspace/dreamerv4-mc/.captures"

if [[ ! -d "$REPO/src" || ! -d "$CHECKPOINTS" ]]; then
    echo "ERROR: expected repo at $REPO and checkpoints at $CHECKPOINTS"
    exit 2
fi
mkdir -p "$CAP_HOST"
chmod a+rwX "$CAP_HOST"

echo "[start-capture] stopping any prior container..."
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "[start-capture] starting: steps_size=$STEPS capture_max=$MAX dir=$CAP_HOST"
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
            --steps_size ${STEPS} \
            --capture_dir ${CAP_GUEST} \
            --capture_max_frames ${MAX}
    " >/dev/null

echo "[start-capture] container started; ~50s for pip + model load + cuda graph capture."
echo "[start-capture] tail capture progress:  docker logs -f $NAME 2>&1 | grep capture"
echo "[start-capture] browser:                http://localhost:$PORT"
echo "[start-capture] captures appear in:     $CAP_HOST"
echo
echo "After capture completes, replay offline with:"
echo "  scripts/ngc/render-replay.sh latest.pt <record_steps> <record_frames>"
