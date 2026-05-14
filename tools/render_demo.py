#!/usr/bin/env python3
"""Render a one-off Dreamerv4-MC demo video.

Two-phase design (added 2026-05-14):
- Optional WARMUP phase: generate N frames at low steps_size using cheap
  "look around" actions; output is DISCARDED but the KV cache retains
  coherent recent context. This pulls the model out of the cold-start
  fixed-point basin (e.g. all-black absorbing state) before the recording
  phase commits to a high-step trajectory.
- RECORD phase: generate the bucket-place action sequence at the user-
  selected steps_size against the now-warm cache; this is the output mp4.

If --warmup-frames=0, behaves identically to the old single-phase renderer.

Usage (inside the NGC container):
    python tools/render_demo.py [--start-frame imgs_0.png] [--steps-size 32] \
        [--n-frames 200] [--warmup-frames 200] [--warmup-steps 4]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torchvision.io as tvio
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, "/workspace/dreamerv4-mc")

from src.inference.mc_vw_infer import MCWorldModelInfer
from src.modules.actokenizer import MineCraftActionTokenizer


def build_action_sequence(action_tok: MineCraftActionTokenizer, n_frames: int) -> torch.Tensor:
    """Walk forward with a gentle yaw sway (RECORD phase).

    Empirically the most robust offline action sequence at every steps_size
    we have tested: produces coherent multi-second gameplay regardless of
    biome (verified on grass, water, and Nether start frames). Earlier
    candidates layered pitch-down + right-click on top to dramatize a
    bucket-placement narrative, but that combination reads as 'digging
    into ground' in training and biased the model toward going underground
    (luma drifted from ~80 to ~17 over 10s).
    """
    actions: list[list[int]] = []
    for i in range(n_frames):
        action_dict = {
            "mouse": {
                "dx": 6.0 * math.sin(2 * math.pi * i / 80.0),
                "dy": 0.0,
                "buttons": [],
            },
            "keyboard": {"keys": ["key.keyboard.w"]},
            "hotbar": 0,
        }
        env_action, _ = action_tok.json_action_to_env_action(action_dict, hotbar=True)
        actions.append(action_tok.get_action_index_from_actiondict(env_action, include_gui=True))
    return torch.tensor(actions, dtype=torch.long)


def build_warmup_action_sequence(action_tok: MineCraftActionTokenizer, n_frames: int) -> torch.Tensor:
    """Cheap, varied gameplay actions used only to fill the KV cache.

    Pattern: walk forward holding W, with slow camera sway. The goal is
    to produce coherent 'first-person walking' frames so the cache holds
    plausible gameplay context, not raw prefill noise.
    """
    actions: list[list[int]] = []
    for i in range(n_frames):
        keys = ["key.keyboard.w"]
        # gentle sinusoidal yaw sway, ~6 unit/frame amplitude, period 60 frames
        dx = 6.0 * math.sin(2 * math.pi * i / 60.0)
        dy = 0.0
        buttons: list[int] = []
        hotbar = 0
        action_dict = {
            "mouse": {"dx": dx, "dy": dy, "buttons": buttons},
            "keyboard": {"keys": keys},
            "hotbar": hotbar,
        }
        env_action, _ = action_tok.json_action_to_env_action(action_dict, hotbar=True)
        actions.append(action_tok.get_action_index_from_actiondict(env_action, include_gui=True))
    return torch.tensor(actions, dtype=torch.long)


def load_start_frame(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    pil = Image.open(path).convert("RGB")
    tfm = T.Compose([T.ToTensor(), T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    img = tfm(pil)
    if img.shape[1] != 384:
        diff = 384 - img.shape[1]
        pad_top = diff // 2
        pad_bot = diff - pad_top
        img = torch.nn.functional.pad(img, (0, 0, pad_top, pad_bot), mode="constant", value=0)
    return img[None, None].to(device).to(dtype)


def save_video(frames: torch.Tensor, out_path: Path, fps: int = 20) -> None:
    frames = (frames.clamp(-1, 1) + 1) / 2
    frames = (frames * 255).to(torch.uint8).permute(0, 2, 3, 1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    tvio.write_video(str(out_path), frames.cpu(), fps=fps)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-frame", default="imgs_0.png")
    ap.add_argument("--steps-size", type=int, default=16, help="RECORD phase steps_size")
    ap.add_argument("--n-frames", type=int, default=100, help="RECORD phase frame count")
    ap.add_argument("--warmup-frames", type=int, default=0,
                    help="If >0, run a warmup phase to populate the KV cache before recording")
    ap.add_argument("--warmup-steps", type=int, default=4,
                    help="steps_size used during the warmup phase (default: 4 = live baseline)")
    ap.add_argument("--capture-replay", default=None,
                    help="If set, ignore --start-frame and --warmup-frames; prefill from a live-capture .pt file (frames + actions) and run record phase against that context.")
    ap.add_argument("--output", default="/workspace/dreamerv4-mc/.demos/bucket-water-latest.mp4")
    ap.add_argument("--dynamic-path", default="checkpoints/dynamic")
    ap.add_argument("--tokenizer-path", default="checkpoints/tokenizer")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16

    print(f"[demo] config: start={args.start_frame} record_steps={args.steps_size} "
          f"record_frames={args.n_frames} warmup_frames={args.warmup_frames} "
          f"warmup_steps={args.warmup_steps}", flush=True)
    print(f"[demo] loading model: dynamic={args.dynamic_path} tokenizer={args.tokenizer_path}",
          flush=True)
    t_load = time.perf_counter()
    model = MCWorldModelInfer(
        dynamic_model_path=args.dynamic_path,
        tokenizer_path=args.tokenizer_path,
        record_video_output_path="",
        steps_size=(args.warmup_steps if (args.warmup_frames > 0 and not args.capture_replay) else args.steps_size),
        device=device,
        dtype=dtype,
        random_generator=torch.Generator(device=device).manual_seed(args.seed),
        use_cuda_graph=True,
        refresh_kvcache=False,
    )
    print(f"[demo] model loaded in {time.perf_counter() - t_load:.1f}s", flush=True)

    if args.capture_replay:
        init_frames = None
        print("[demo] capture-replay mode: skipping single-image start frame load",
              flush=True)
    else:
        start_path = Path("/workspace/dreamerv4-mc/ui/static/start_frames") / args.start_frame
        print(f"[demo] loading start frame: {start_path}", flush=True)
        init_frames = load_start_frame(start_path, device, dtype)

    action_tok = MineCraftActionTokenizer()
    action_tok.camera_scaler = 360.0 / 2400.0 * 2

    # ---- PREFILL PATH SELECTION ----
    if args.capture_replay:
        # Multi-frame prefill from a real human-played trajectory.
        # Bypasses both single-image prefill and synthetic warmup.
        cap_path = Path(args.capture_replay)
        print(f"[demo] CAPTURE-REPLAY: loading {cap_path}", flush=True)
        payload = torch.load(cap_path, map_location="cpu", weights_only=False)
        cap_frames = payload["frames"]            # (N, 3, H, W) fp16 in [-1, 1]
        cap_actions = payload["action_ids"]       # (N, 12) long
        n_cap = cap_frames.shape[0]
        print(f"[demo] CAPTURE-REPLAY: {n_cap} frames, action_ids shape={tuple(cap_actions.shape)}",
              flush=True)
        # Reshape to (1, N, 3, H, W) and push to device/dtype.
        cap_frames_dev = cap_frames.unsqueeze(0).to(device).to(dtype)
        cap_actions_dev = cap_actions.to(device)
        # Direct prefill - no generation in this phase.
        t_pf = time.perf_counter()
        model.prefilling_kvcache(init_frames=cap_frames_dev, action_ids=cap_actions_dev)
        print(f"[demo] CAPTURE-REPLAY: prefilled in {time.perf_counter() - t_pf:.1f}s; "
              f"frame_idx now at {model.frame_idx}", flush=True)
        torch.cuda.empty_cache()
        model.steps_size = args.steps_size
        record_init_frames = None
    elif args.warmup_frames > 0:
        # ---- WARMUP PHASE (synthetic) ----
        warmup_actions = build_warmup_action_sequence(action_tok, args.warmup_frames).to(device)
        per_frame_ms = args.warmup_steps * 75 + 50
        eta_min = args.warmup_frames * per_frame_ms / 60000
        print(f"[demo] WARMUP: {args.warmup_frames}f at steps={args.warmup_steps} "
              f"(~{eta_min:.1f} min) — output discarded, KV cache retained",
              flush=True)
        t_warm = time.perf_counter()
        _ = model.infer_video(action_ids=warmup_actions, init_frames=init_frames)
        del _
        torch.cuda.empty_cache()
        warm_elapsed = time.perf_counter() - t_warm
        print(f"[demo] WARMUP done in {warm_elapsed:.1f}s "
              f"({warm_elapsed/args.warmup_frames:.2f}s/frame); "
              f"frame_idx now at {model.frame_idx}", flush=True)
        model.steps_size = args.steps_size
        record_init_frames = None
    else:
        record_init_frames = init_frames

    # ---- RECORD PHASE ----
    record_actions = build_action_sequence(action_tok, args.n_frames).to(device)
    per_frame_ms = args.steps_size * 75 + 50
    eta_min = args.n_frames * per_frame_ms / 60000
    print(f"[demo] RECORD: {args.n_frames}f at steps={args.steps_size} "
          f"(~{eta_min:.1f} min)", flush=True)
    t_render = time.perf_counter()
    output_video = model.infer_video(action_ids=record_actions, init_frames=record_init_frames)
    elapsed = time.perf_counter() - t_render
    n_out = output_video.shape[0]
    print(f"[demo] RECORD done: {n_out} frames in {elapsed:.1f}s "
          f"({elapsed / n_out:.2f}s/frame)", flush=True)

    out_path = Path(args.output)
    print(f"[demo] saving to {out_path}", flush=True)
    save_video(output_video, out_path, fps=20)
    print(f"[demo] DONE → {out_path} ({out_path.stat().st_size:,} bytes, {n_out} frames @ 20fps "
          f"= {n_out/20:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
