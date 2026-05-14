# GB10 benchmarks and render findings

These are exploratory inference notes from the DGX Spark / GB10 port. They are
included so the fork is falsifiable: a reader can see the observed step-size
tradeoffs, the failure cases, and the exact rendered clips used to reach the
current operating envelope.

These are **not** formal ML benchmark-suite results. They are single-machine
measurements taken during interactive development on GB10 / `sm_121` under the
NGC PyTorch container described in [FORK.md](FORK.md). Treat them as practical
operator guidance, not a claim about the upstream checkpoint in every setting.

## Test environment

| Item | Value |
|---|---|
| GPU | NVIDIA GB10 / DGX Spark (`sm_121`) |
| Host arch | aarch64 Linux |
| Container | `nvcr.io/nvidia/pytorch:26.03-py3` |
| Runtime stack | PyTorch 2.11, Triton 3.6.0, flash_attn 2.7.4 |
| Output | 640x384 MP4s for offline renders; browser display crops/serves the live app |
| CUDA graphs | Dynamic model and tokenizer decoder both captured |
| Live guard | one active websocket client, mouse clip=25/frame, mouse EMA alpha=0.5 |

## Step-size envelope

The dynamic model supports power-of-two `steps_size` values because the
scheduler uses discrete stride conditioning. Non-powers of two are invalid.

| `steps_size` | Approx server FPS / speed | Observed behavior | Current recommendation |
|---:|---:|---|---|
| 1 | ~7.8 FPS | Fast but hallucinates; one-shot reconstruction is not stable. | Do not use for live play without retraining/distillation. |
| 2 | ~5.0 FPS | Faster than baseline, but world drifts and decoheres. | Not recommended. |
| 4 | ~2.8-3.1 FPS | Stable live operating point with RoPE FP32 and single-client guard. | **Recommended live/browser mode.** |
| 8 | ~1.5-1.6 FPS | Slower but cold offline renders are generally coherent. | Cold offline ceiling. Useful for clips, not live play. |
| 16 | ~0.87 FPS live; ~1.15 s/frame observed | Cold/live runs eventually decohere, especially with mouse movement. | Diagnostic only; use warmup-then-record if needed. |
| 32 | ~0.41 FPS offline; ~2.42 s/frame observed | Cold renders collapse into absorbing states (black/yellow/void). | Use only with a strong warm KV cache. |

Representative timing logs after the GB10 changes:

| Case | Observed log / run output |
|---|---|
| Live `steps_size=4` | `[TIMING] generate≈316-320ms decode≈44ms total≈360-364ms steps=4` |
| Live `steps_size=16` | `[TIMING] generate≈1107-1135ms decode≈44-45ms total≈1151-1181ms steps=16` |
| Offline `steps_size=8`, 1200 frames | `1200 frames in 805.5s` (`0.67s/frame`, ~1.49 generated FPS) |
| Offline `steps_size=32`, 100 frames | `100 frames in 242.2s` (`2.42s/frame`, ~0.41 generated FPS) |

The rough cost model on this hardware is dominated by the denoising loop:
`~70-80ms * steps_size + ~44ms tokenizer decode + small websocket/UI overhead`.
The exact number varies by run and whether it is live or offline.

## Major findings

### 1. `steps_size=4` is the live sweet spot

The app becomes playable only when the model stays inside the trained action
and stride envelope. `steps_size=4` with the RoPE FP32 patch, single-client
websocket guard, and mouse-input guard is the best observed live configuration.

### 2. High step counts do not cold-start reliably

Cold `steps_size=16` and `steps_size=32` are not simply "more compute = better
frames." In this checkpoint, higher stride modes appear to be different
conditional modes rather than a continuous solver refinement. Cold high-step
runs can enter absorbing visual states after a few seconds.

The architecture supports this interpretation: the dynamic model uses frozen
one-hot timestep and stride embeddings. Each `steps_size in {1,2,4,8,16,32,64}`
is a separate discrete mode sharing weights, not an interpolated continuous
ODE/flow solver setting.

### 3. Warm KV cache can make high-step renders viable

High-step cold-start failures improve when the rolling KV cache is populated by
low-step frames first. The practical protocol is:

```bash
# 200 warmup frames at steps=4, discarded; then 200 record frames at steps=32
scripts/ngc/render-demo.sh imgs_0.png 32 200 200 4
```

This does not prove high-step modes are universally stable. It means the
stronger low-step context can anchor the high-step record phase away from the
cold-start attractors.

### 4. Mouse movement is the hardest live action

Standing still regresses toward the cached frame. WASD largely translates known
context. Camera movement forces new geometry to be invented at the edge of the
current view. At low live FPS, mouse deltas also accumulate over longer wall-clock
intervals before being quantized. This is why live `steps_size=16` decoheres
most reliably under mouse movement.

### 5. Static-camera idle crystallizes

The idle action sequence is not supposed to create a living screensaver. With
no movement keys, no mouse delta, and no mouse buttons, the learned conditional
mode is "next frame ~= current frame." Static-camera renders therefore converge
toward a fixed point. This is expected behavior, not necessarily a bug.

## Render gallery

The MP4s below are tracked intentionally even though the repository ignores
most generated videos. They are small, model-generated test artifacts used to
anchor the observations above. The offline renderer writes MP4s at 20fps by
default unless otherwise noted.

| File | What it demonstrates | Settings | Size |
|---|---|---|---:|
| [`benchmarks/renders/crickle-step8-60s.mp4`](benchmarks/renders/crickle-step8-60s.mp4) | The original crickle render with the default synthetic walking + yaw-sway action; useful as a comparison for motion-driven drift/decoherence. | `crickle.png`, `steps_size=8`, 1200 frames, 60s @ 20fps, default action sequence | 6.75 MB |
| [`benchmarks/renders/crickle-step8-60s-idle.mp4`](benchmarks/renders/crickle-step8-60s-idle.mp4) | Same prompt but motionless; shows the static-camera crystallization/fixed-point behavior. | `crickle.png`, `steps_size=8`, 1200 frames, 60s @ 20fps, `--idle` | 3.34 MB |
| [`benchmarks/renders/crickle-step4-30s-idle.mp4`](benchmarks/renders/crickle-step4-30s-idle.mp4) | Browser inference settings but offline 20fps encoding; 600 generated frames compressed into 30s of viewing time. | `crickle.png`, `steps_size=4`, 600 frames, 30s @ 20fps, `--idle` | 1.67 MB |
| [`benchmarks/renders/crickle-step4-90f-idle-20fps.mp4`](benchmarks/renders/crickle-step4-90f-idle-20fps.mp4) | The raw 90 generated frames used for browser-wall-clock comparison, encoded at the offline default 20fps. | `crickle.png`, `steps_size=4`, 90 frames, 4.5s @ 20fps, `--idle` | 0.49 MB |
| [`benchmarks/renders/crickle-step4-30s-idle-browserfps.mp4`](benchmarks/renders/crickle-step4-30s-idle-browserfps.mp4) | Approximate what a user sees staring at the browser for 30 wall-clock seconds at ~3 FPS. | same 90 frames as above, re-encoded to 30s @ 3fps | 0.66 MB |
| [`benchmarks/renders/move3-200f-steps16.mp4`](benchmarks/renders/move3-200f-steps16.mp4) | Capture/replay preview with 100-frame prefill + 100-frame record at high steps. It is a limited preview because the available capture was effectively stand-still. | capture replay, 100 prefill + 100 record, `steps_size=16`, 5s @ 20fps | 0.62 MB |
| [`benchmarks/renders/move3-200f-steps32.mp4`](benchmarks/renders/move3-200f-steps32.mp4) | Same limited capture/replay preview at `steps_size=32`. Useful as an artifact, not a final proof that Move 3 fixes high-step cold collapse. | capture replay, 100 prefill + 100 record, `steps_size=32`, 5s @ 20fps | 0.59 MB |

### Render checksums

```text
f34849a472e12b9f49d12d09065f85ead400428df93d11488d53da2128c54ab1  benchmarks/renders/crickle-step4-30s-idle-browserfps.mp4
d0a7d92aa6b56622b9abf357508a3843be0d54a35564d57c7bd650630a4dd3cf  benchmarks/renders/crickle-step4-30s-idle.mp4
d362a87d38e2f717be6db029d27144f0d81afbdfe8b8385f3f5e512e5b919f5f  benchmarks/renders/crickle-step4-90f-idle-20fps.mp4
996beebdd078fc1d183faf7d00f0ceb16c00b3f6c7aaa2a2553e1bdc06814d4e  benchmarks/renders/crickle-step8-60s-idle.mp4
8239b87284057d88ebf2f223bf283d2261103064aaabdf308e26d4762776f43d  benchmarks/renders/crickle-step8-60s.mp4
c3229f22d2cf90a883b84a6c4eed6d71b4af0eaaa15d665ecac92fe1c794aa64  benchmarks/renders/move3-200f-steps16.mp4
525b7080beae7897fb18aba5a32453aa98bf4b7f7eb8bade176bb84b02115cbb  benchmarks/renders/move3-200f-steps32.mp4
```


### Live capture + steps=4 exploration clips

These clips were recorded during interactive testing on GB10. They sit alongside the
crickle/Move 3 clips above and exist to show what the model produces on real
in-distribution Minecraft scenes (not just synthetic geometric prompts) at the
recommended live `steps_size=4` operating point.

| File | What it demonstrates | Settings | Size |
|---|---|---|---:|
| [`benchmarks/renders/capture-preview.mp4`](benchmarks/renders/capture-preview.mp4) | Sidecar MP4 emitted by the live capture-mode server while a human played in the browser. Demonstrates the live `--capture_dir` -> `.captures/<ts>.pt` + sidecar pipeline that feeds `render-replay.sh`. | live capture, `steps_size=4`, 200 frames, 10s @ 20fps, 640x360 native | 1.9 MB |
| [`benchmarks/renders/portal-steps4-explore.mp4`](benchmarks/renders/portal-steps4-explore.mp4) | Offline render against a portal start frame at the recommended live setting; useful comparison point for the portal-transition issue documented in FORK.md. | offline render, `steps_size=4`, 200 frames, 10s @ 20fps, 640x384 | 0.7 MB |
| [`benchmarks/renders/boat-steps4-explore.mp4`](benchmarks/renders/boat-steps4-explore.mp4) | Offline render against a boat / water start frame at the recommended live setting. Water is a useful stress test for ambient-motion learned priors. | offline render, `steps_size=4`, 200 frames, 10s @ 20fps, 640x384 | 1.2 MB |
| [`benchmarks/renders/normal-steps4-warm200-walking.mp4`](benchmarks/renders/normal-steps4-warm200-walking.mp4) | Offline render with the warmup-then-record protocol: a 200-frame walking warmup phase fills the KV cache, then 200 frames are recorded at `steps_size=4`. This is the exact protocol that unlocks high-step renders without cold collapse. | offline render, warmup 200 frames @ steps=4 then record 200 frames @ steps=4 with walking action sequence, 10s @ 20fps, 640x384 | 1.4 MB |

### Operating reminder

`steps_size` of 16 and higher produces rapid scene decoherence on this checkpoint
(usually within ~4-8 seconds of cold start). The live and cold-offline operating
envelope should stay at `steps_size in {4, 8}`. Higher step counts only become
viable when paired with the warmup-then-record protocol shown above; even then,
they are demo-grade rather than interactive-grade.

### Additional render checksums

```text
c44b44814317bb7208fc64a644e603085cf986678ae363034089073f60b1fdbb  benchmarks/renders/capture-preview.mp4
f17d5e5c193f9b6a160c8293be58bd3466f21b5144c086ee4c64e8d0c3669a7a  benchmarks/renders/portal-steps4-explore.mp4
ca63365c1ca36f263289750ee836f1edb2ade5b90dca0a3d0e79b008caceb4ed  benchmarks/renders/boat-steps4-explore.mp4
01cb5152463133832d1de8a627e6b7aa52f4963082ef4247db08532783c29693  benchmarks/renders/normal-steps4-warm200-walking.mp4
```

## Inline video gallery

GitHub markdown strips raw HTML5 `<video>` tags from README/blob views. The embedded gallery is therefore checked in as `docs/video-gallery.html` and served through a renderer that preserves native video controls.

- **[Open the embedded video gallery](https://raw.githack.com/blackfirebitcoin/Dreamerv4-MC-GB10/dgx-spark-gb10/docs/video-gallery.html)**
- [Gallery source in this repo](docs/video-gallery.html)
- [GitHub Pages URL](https://blackfirebitcoin.github.io/Dreamerv4-MC-GB10/) — the `gh-pages` branch has been pushed; enable Pages in repo settings if this URL returns 404.

Direct MP4 links:

- [`normal-steps4-warm200-walking.mp4`](benchmarks/renders/normal-steps4-warm200-walking.mp4) — Recommended live operating point.
- [`capture-preview.mp4`](benchmarks/renders/capture-preview.mp4) — Live browser capture preview.
- [`crickle-step8-60s.mp4`](benchmarks/renders/crickle-step8-60s.mp4) — Crickle, synthetic walking + yaw-sway.
- [`crickle-step8-60s-idle.mp4`](benchmarks/renders/crickle-step8-60s-idle.mp4) — Crickle, static-camera idle.
- [`crickle-step4-30s-idle.mp4`](benchmarks/renders/crickle-step4-30s-idle.mp4) — Crickle, idle at steps_size=4.
- [`crickle-step4-90f-idle-20fps.mp4`](benchmarks/renders/crickle-step4-90f-idle-20fps.mp4) — Crickle, 90 generated idle frames.
- [`crickle-step4-30s-idle-browserfps.mp4`](benchmarks/renders/crickle-step4-30s-idle-browserfps.mp4) — Crickle, browser-FPS approximation.
- [`move3-200f-steps16.mp4`](benchmarks/renders/move3-200f-steps16.mp4) — Move 3 replay, steps_size=16.
- [`move3-200f-steps32.mp4`](benchmarks/renders/move3-200f-steps32.mp4) — Move 3 replay, steps_size=32.
- [`portal-steps4-explore.mp4`](benchmarks/renders/portal-steps4-explore.mp4) — Portal scene exploration.
- [`boat-steps4-explore.mp4`](benchmarks/renders/boat-steps4-explore.mp4) — Boat / water scene exploration.

## Reproduction commands

Live baseline:

```bash
scripts/ngc/start-live.sh 4
# open http://localhost:8765 or tunnel it from a remote workstation
```

Static-camera crickle, 60s at steps=8:

```bash
IDLE=1 scripts/ngc/render-demo.sh crickle.png 8 1200
```

Browser-wall-clock static-camera crickle approximation:

```bash
# 90 generated frames is roughly 30 seconds of live wall-clock time at ~3 FPS.
IDLE=1 scripts/ngc/render-demo.sh crickle.png 4 90
# The checked-in browserfps video is the same 90 frames re-encoded at 3fps.
```

High-step warmup protocol:

```bash
scripts/ngc/render-demo.sh imgs_0.png 32 200 200 4
```

Capture/replay workflow:

```bash
scripts/ngc/start-capture.sh 4 400
# play in the browser until .captures/latest.pt is written
scripts/ngc/render-replay.sh latest.pt 16 200 200
scripts/ngc/render-replay.sh latest.pt 32 200 200
```

## Caveats

- The rendered clips are qualitative validation artifacts, not a substitute for
  a repeatable automated benchmark harness.
- The available Move 3 preview capture was effectively stand-still, so the
  `move3-*` videos do not prove captured-action continuation solves high-step
  drift under real played movement.
- The crickle/starburst start frames are geometric prompts used for stress tests,
  not part of the upstream model weights or training data.
- Browser FPS depends on network transport, browser decode/repaint, and whether
  duplicate websocket clients are connected. Server timing logs are the more
  stable metric for model performance.
