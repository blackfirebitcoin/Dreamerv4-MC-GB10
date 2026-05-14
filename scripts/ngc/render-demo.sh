#!/usr/bin/env bash
# render-demo.sh -- one-off offline demo render.
#
# Stops the live container (if running), renders an MP4, restarts the live
# container. Output overwrites $DREAMERV4_OUT/bucket-water-latest.mp4 every
# run by design -- copy it off with a descriptive name before the next render.
#
# Usage:
#   render-demo.sh                                  # 100 frames, steps=16, imgs_0.png
#   render-demo.sh imgs_4.png                       # different start frame
#   render-demo.sh imgs_4.png 16 80                 # start, steps, n_frames
#   render-demo.sh imgs_0.png 32 200 200 4          # warmup-then-record:
#                                                   #   200 warmup frames at steps=4 (cache fill, discarded)
#                                                   #   then 200 record frames at steps=32 (saved)
#   IDLE=1 render-demo.sh crickle.png 8 1200        # 60s motionless static-camera render
#
# Environment variables:
#   DREAMERV4_REPO         path to the repo on the host (default: $PWD)
#   DREAMERV4_CHECKPOINTS  path to checkpoints/ on the host (default: $DREAMERV4_REPO/checkpoints)
#   DREAMERV4_OUT          path for output mp4s (default: $DREAMERV4_REPO/.demos)
#   DREAMERV4_LIVE_NAME    name of the live container (default: dreamerv4-ngc)
#   DREAMERV4_DEMO_NAME    name of the ephemeral render container (default: dreamerv4-demo)
#   DREAMERV4_IMAGE        container image (default: nvcr.io/nvidia/pytorch:26.03-py3)
#   IDLE                   if =1, pass --idle to the renderer (motionless action sequence)
set -euo pipefail

START_FRAME=${1:-imgs_0.png}
STEPS=${2:-16}
N_FRAMES=${3:-100}
WARMUP_FRAMES=${4:-0}
WARMUP_STEPS=${5:-4}
IDLE=${IDLE:-0}

REPO=${DREAMERV4_REPO:-$PWD}
CHECKPOINTS=${DREAMERV4_CHECKPOINTS:-$REPO/checkpoints}
OUT_HOST=${DREAMERV4_OUT:-$REPO/.demos}
LIVE_NAME=${DREAMERV4_LIVE_NAME:-dreamerv4-ngc}
DEMO_NAME=${DREAMERV4_DEMO_NAME:-dreamerv4-demo}
IMAGE=${DREAMERV4_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3}

OUT_FILE=${OUT_HOST}/bucket-water-latest.mp4

if [[ ! -d "$REPO/src" || ! -d "$CHECKPOINTS" ]]; then
    echo "ERROR: expected repo at $REPO and checkpoints at $CHECKPOINTS"
    exit 2
fi

echo "[demo] config: start=${START_FRAME} steps=${STEPS} n_frames=${N_FRAMES} warmup=${WARMUP_FRAMES}f@${WARMUP_STEPS} idle=${IDLE}"
RECORD_MIN=$(awk "BEGIN{printf \"%.1f\", ${N_FRAMES} * ${STEPS} * 0.075 / 60}")
WARMUP_MIN=$(awk "BEGIN{printf \"%.1f\", ${WARMUP_FRAMES} * ${WARMUP_STEPS} * 0.075 / 60}")
echo "[demo] expected runtime: ~${RECORD_MIN} min record + ~${WARMUP_MIN} min warmup"

mkdir -p "$OUT_HOST"

LIVE_RUNNING=0
if docker ps --filter "name=$LIVE_NAME" --format '{{.Names}}' | grep -q "^$LIVE_NAME$"; then
    LIVE_RUNNING=1
    echo "[demo] stopping live container to free GPU memory..."
    docker stop "$LIVE_NAME" >/dev/null
fi

echo "[demo] starting render container ($DEMO_NAME)..."
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
            --start-frame ${START_FRAME} \
            --steps-size ${STEPS} \
            --n-frames ${N_FRAMES} \
            --warmup-frames ${WARMUP_FRAMES} \
            --warmup-steps ${WARMUP_STEPS} \
            \$( [ \"${IDLE}\" = \"1\" ] && echo --idle )
    " || DEMO_EXIT=$?

if [[ "$LIVE_RUNNING" -eq 1 ]]; then
    echo "[demo] restarting live container..."
    docker start "$LIVE_NAME" >/dev/null
    for i in 1 2 3 4 5 6; do
        if curl -fs "http://localhost:${DREAMERV4_PORT:-8765}/api/start-frames" >/dev/null 2>&1; then
            echo "[demo] live container ready after ${i}0s"
            break
        fi
        sleep 10
    done
fi

if [[ "$DEMO_EXIT" -ne 0 ]]; then
    echo "[demo] FAILED with exit $DEMO_EXIT"
    exit "$DEMO_EXIT"
fi

chmod a+rwX "$OUT_HOST" "$OUT_FILE" 2>/dev/null || true

echo
echo "=========================================="
echo "[demo] DONE -> $OUT_FILE"
[[ -f "$OUT_FILE" ]] && echo "[demo]   size: $(du -h "$OUT_FILE" | cut -f1)"
echo
echo "Copy with a descriptive name BEFORE the next render (the file overwrites):"
echo "  cp $OUT_FILE $OUT_HOST/<name>.mp4"
echo "=========================================="
