#!/usr/bin/env bash
# render-replay.sh -- render an offline demo from a captured live-play session.
#
# Loads a .pt capture file (frames + action_ids) produced by start-capture.sh,
# prefills the model's KV cache from those real human-played frames, then
# renders the record phase against it.
#
# RECORD action source:
#   - prefill_frames=0 (default): record uses synthetic walk-forward + sway
#     (legacy; compatible with old captures)
#   - prefill_frames>0: first N captured frames -> prefill; next n_frames
#     captured action_ids -> record (closes the synthetic-vs-real action seam)
#
# Usage:
#   render-replay.sh latest.pt 16 200 200             # 200f prefill + 200 captured-action record at steps=16
#   render-replay.sh latest.pt 32 200 200             # same, steps=32 (only works if cache is good enough)
#   render-replay.sh capture-<ts>.pt 16 200 200       # a specific capture
#
# Environment variables:
#   DREAMERV4_REPO         path to the repo on the host (default: $PWD)
#   DREAMERV4_CHECKPOINTS  path to checkpoints/ on the host (default: $DREAMERV4_REPO/checkpoints)
#   DREAMERV4_OUT          path for output mp4s (default: $DREAMERV4_REPO/.demos)
#   DREAMERV4_LIVE_NAME    name of the live container (default: dreamerv4-ngc)
#   DREAMERV4_DEMO_NAME    name of the ephemeral render container (default: dreamerv4-demo)
#   DREAMERV4_IMAGE        container image (default: nvcr.io/nvidia/pytorch:26.03-py3)
set -euo pipefail

CAP_REL=${1:-latest.pt}
STEPS=${2:-16}
N_FRAMES=${3:-200}
PREFILL_FRAMES=${4:-0}

REPO=${DREAMERV4_REPO:-$PWD}
CHECKPOINTS=${DREAMERV4_CHECKPOINTS:-$REPO/checkpoints}
OUT_HOST=${DREAMERV4_OUT:-$REPO/.demos}
LIVE_NAME=${DREAMERV4_LIVE_NAME:-dreamerv4-ngc}
DEMO_NAME=${DREAMERV4_DEMO_NAME:-dreamerv4-demo}
IMAGE=${DREAMERV4_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}

CAP_HOST="$REPO/.captures/$CAP_REL"
CAP_GUEST="/workspace/dreamerv4-mc/.captures/$CAP_REL"
OUT_FILE="$OUT_HOST/bucket-water-latest.mp4"

if [[ ! -f "$CAP_HOST" ]]; then
    echo "[replay] ERROR: capture file not found: $CAP_HOST"
    echo "[replay] available captures:"
    find "$REPO/.captures" -name '*.pt' 2>/dev/null | sed 's|^|  |'
    exit 1
fi

if [[ "$PREFILL_FRAMES" -gt 0 ]]; then
    echo "[replay] config: capture=$CAP_REL steps=$STEPS n_frames=$N_FRAMES prefill_frames=$PREFILL_FRAMES (captured-action continuation)"
else
    echo "[replay] config: capture=$CAP_REL steps=$STEPS n_frames=$N_FRAMES (legacy synthetic record actions)"
fi

mkdir -p "$OUT_HOST"

LIVE_RUNNING=0
if docker ps --filter "name=$LIVE_NAME" --format '{{.Names}}' | grep -q "^$LIVE_NAME$"; then
    LIVE_RUNNING=1
    echo "[replay] stopping live container to free GPU memory..."
    docker stop "$LIVE_NAME" >/dev/null
fi

EXTRA_ARGS=""
[[ "$PREFILL_FRAMES" -gt 0 ]] && EXTRA_ARGS="--capture-prefill-frames $PREFILL_FRAMES"

docker rm -f "$DEMO_NAME" >/dev/null 2>&1 || true
DEMO_EXIT=0
docker run --rm --name "$DEMO_NAME" \
    --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v "$REPO:/workspace/dreamerv4-mc" \
    -v "$CHECKPOINTS:/workspace/dreamerv4-mc/checkpoints" \
    -w /workspace/dreamerv4-mc \
    "$IMAGE" \
    bash -lc "
        grep -v -E '^(torch|torchvision)==|^torch\$|^torchvision\$' requirements.txt > /tmp/requirements-no-torch.txt
        pip install -q -r /tmp/requirements-no-torch.txt
        pip install -q -e . --no-deps
        python tools/render_demo.py \
            --capture-replay ${CAP_GUEST} \
            --steps-size ${STEPS} \
            --n-frames ${N_FRAMES} \
            ${EXTRA_ARGS}
    " || DEMO_EXIT=$?

if [[ "$LIVE_RUNNING" -eq 1 ]]; then
    echo "[replay] restarting live container..."
    docker start "$LIVE_NAME" >/dev/null
    for i in 1 2 3 4 5 6; do
        if curl -fs "http://localhost:${DREAMERV4_PORT:-8765}/api/start-frames" >/dev/null 2>&1; then
            echo "[replay] live container ready after ${i}0s"
            break
        fi
        sleep 10
    done
fi

[[ "$DEMO_EXIT" -ne 0 ]] && { echo "[replay] FAILED with exit $DEMO_EXIT"; exit "$DEMO_EXIT"; }

chmod a+rwX "$OUT_HOST" "$OUT_FILE" 2>/dev/null || true
echo
echo "=========================================="
echo "[replay] DONE -> $OUT_FILE"
[[ -f "$OUT_FILE" ]] && echo "[replay]   size: $(du -h "$OUT_FILE" | cut -f1)"
echo "=========================================="
