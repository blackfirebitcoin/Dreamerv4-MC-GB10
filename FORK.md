# Dreamer4-MC — DGX Spark / GB10 fork

Inference port and stability fixes for **NVIDIA DGX Spark / GB10 (sm_121, aarch64)**
of the [IamCreateAI/Dreamerv4-MC](https://github.com/IamCreateAI/Dreamerv4-MC)
real-time autoregressive Minecraft world model.

## What this fork is

Targeted at running the upstream 1.7B-param dynamic model and 430M-param MAE
tokenizer on a single GB10 inside an NGC PyTorch container, with the bugs
and operating-envelope surprises that came up in practice fixed at the
inference layer. No training-side or model-architecture changes.

## Hardware target

- **GPU:** NVIDIA GB10 / DGX Spark, compute capability `sm_121`
- **Arch:** aarch64 Linux
- **Memory:** unified memory (128 GB tested)
- **Container:** `nvcr.io/nvidia/pytorch:26.03-py3`
  - PyTorch 2.11, flash_attn 2.7.4, Triton 3.6.0
  - Both the dynamic model and tokenizer decoder capture cleanly as CUDA graphs

## What changed vs upstream

### Inference-layer fixes (all upstream-PR candidates)
| Commit subject | What it fixes |
|---|---|
| Make `steps_size` config-driven and add per-frame timing to live path | The interactive render path had `steps_size=4` hardcoded, so the config knob was a no-op. |
| Prevent duplicate websocket clients from halving Dreamerv4 FPS | Each `/ws` connection started its own render loop against the single global engine; two open tabs serialized on `engine.lock` and split FPS in half. |
| Initialize capture mode after CLI overrides are applied | Engine `__init__` ran before argparse mutated the shared config, so `--capture_dir` was silently dropped. |
| Log capture requests before server startup | Capture banner now appears before model load so wrapper failures are distinguishable from later load failures. |

### GB10-specific
| Commit subject | What it does |
|---|---|
| RoPE Triton kernel: FP64 -> FP32 internal math for GB10 throughput | GB10 FP64 throughput is ~1/64 of FP32 (vs ~1/2 on H100). Switching the in-kernel rotation math to FP32 buys ~6% on the dynamic-model hot path with relative L2 error ~9.95e-06 (single bf16 quantum). Output dtype unchanged. |

### Stability / operating envelope
| Commit subject | What it does |
|---|---|
| Stop live chaotic-mouse decoherence at the action-tensor boundary | Per-frame `|dx|`, `|dy|` are clipped (default 25/frame) and optionally low-pass filtered (one-pole IIR, alpha 0.5). Filter state resets on V-key. |
| Smooth offline demo actions to cut jitter and stop late-time freeze | Default offline action sequence smoothed; broken bucket-place sequence replaced with a robust walk-forward + sway. |
| Keep high-step offline renders out of cold-start collapse | High step counts (16, 32) collapse to absorbing states from a cold prefill. New warmup phase fills the KV cache at low steps before recording at high steps. |
| Allow static-camera demo renders without synthetic walking | `--idle` flag in `tools/render_demo.py` produces a motionless record action sequence. |
| Make capture replay genuinely multi-frame safe | `prefilling()` shape bugs that only worked accidentally for single-frame prefill. |
| Replay live-played KV cache state as multi-frame offline prefill | Capture-then-replay workflow: live UI saves `(frames, action_ids)` to `.pt`; offline renderer loads it as KV-cache prefill. |
| Replay captured action_ids during record to close the synthetic-vs-real action seam at high steps_size | Splits captures into prefill + record halves so high-step renders can run on real human play. |

## Operating envelope (verified on GB10)

| Mode | Settings | Throughput | Notes |
|---|---|---:|---|
| **Live browser** | `steps_size=4` cold + RoPE FP32 + single-client guard | ~2.9 server FPS | Stable, recommended default |
| **Cold offline** | `steps_size in {1,2,4,8}` from a single start frame | varies | `steps_size=8` is the cold ceiling. `steps_size=2` is unstable; `steps_size=1` hallucinates. |
| **High-step offline** | `steps_size in {16, 32}` from cold start | does not work | Collapses to absorbing states (black, yellow, void) within ~4 seconds. **Use the warmup-then-record protocol instead.** |
| **High-step offline (with warmup)** | `steps_size=4` warmup of 200 frames, then record at `steps_size in {16, 32}` | tractable | The KV cache built by the well-trained `steps=4` mode anchors the high-step record phase. Mean luma stable over 30+ seconds. |
| **Static camera** | any steps + `--idle` | works but crystallizes | Motionless input causes the model's conditional prior to converge to "frame_t+1 ≈ frame_t". This is correct learned behavior, not a bug. |

## Why high `steps_size` cold-collapses

The dynamic model uses **frozen one-hot embeddings** over `K_samples_step=64` discrete
timesteps and `log2(K_samples_step)+1 = 7` discrete strides
(`src/modules/dynamic_model.py:645`). Each `steps_size in {1,2,4,8,16,32,64}` is its
own conditional mode sharing weights, with **no interpolation between modes** — the
fingerprint of a few-step distilled diffusion model. The high-step modes appear to
have received less / different training supervision than the low-step modes. They are
not unusable; they cannot bootstrap from cold noise.

## Quickstart

Inside the NGC container, the inference and CLI surface is unchanged from upstream
plus the new flags above. Wrapper scripts that match this fork's operating envelope
are in `scripts/ngc/`:

```bash
# Live browser (~2.9 FPS at steps_size=4)
scripts/ngc/start-live.sh

# Capture a play session for offline replay
scripts/ngc/start-capture.sh 4 200       # steps_size, max_frames

# Render an offline demo at any steps_size
scripts/ngc/render-demo.sh imgs_0.png 8 200                 # cold steps=8
scripts/ngc/render-demo.sh imgs_0.png 32 200 200 4          # warmup-then-record at steps=32
IDLE=1 scripts/ngc/render-demo.sh crickle.png 8 1200        # 60s of motionless render

# Replay a captured session at any steps_size with the captured actions
scripts/ngc/render-replay.sh latest.pt 16 200 200
```

The wrappers expect paths via `DREAMERV4_REPO`, `DREAMERV4_CHECKPOINTS`, and
`DREAMERV4_OUT` environment variables (see each script for details). They are not
host-specific.

## Security note

The FastAPI/WebSocket server is **unauthenticated** and was designed for trusted
local use. Do not expose port 8765 directly to the internet. Run it behind an SSH
tunnel or a reverse proxy with auth.

## Upstream

This fork tracks https://github.com/IamCreateAI/Dreamerv4-MC. PRs back to upstream
for the inference-layer fixes are welcome and intended.
