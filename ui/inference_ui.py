import asyncio
import json
import time
import io
import threading
import os
import argparse
from pathlib import Path
import sys
from dataclasses import dataclass, field
from contextlib import asynccontextmanager
from typing import Dict, Set, Optional, Tuple, Callable, Any, Union

import torch
import numpy as np
import av
from PIL import Image
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from torch.nn import functional as F
import torchvision.transforms as T
from src.inference.mc_vw_infer import MCWorldModelInfer
from src.modules.actokenizer import MineCraftActionTokenizer


@dataclass
class ServerConfig:
    dynamic_model_path: str = os.getenv("DYNAMIC_PATH", "")
    tokenizer_path: str = os.getenv("TOKENIZER_PATH", "")
    record_video_output_path: str = os.getenv("RECORD_VIDEO_OUTPUT_PATH", "")
    
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    steps_size: int = 4
    seed: int = 42
    target_fps: float = 20.0
    camera_scaler: float = 360.0 / 2400.0 * 2
    
    jpeg_quality: int = 60
    resize_width: int = 640
    resize_height: int = 360

    # Mouse OOD-spike guards (live-input decoherence fix).
    # Per-frame |dx|, |dy| are clipped to mouse_clip before being passed
    # to the action tokenizer; then optionally low-pass filtered with a
    # one-pole IIR (filtered = alpha*new + (1-alpha)*prev). Set
    # mouse_ema_alpha=0.0 to disable smoothing, 1.0 for passthrough.
    mouse_clip: float = 25.0
    mouse_ema_alpha: float = 0.5

    # Capture mode (multi-frame prefill seed source).
    # If capture_dir is non-empty, the engine records every generated frame
    # and the action that produced it; after capture_max_frames it writes a
    # single .pt file plus a sidecar mp4 to capture_dir. V-key resets the
    # buffer so multiple takes can be recorded in one session.
    capture_dir: str = ""
    capture_max_frames: int = 200

# 初始化全局配置
config = ServerConfig()

# ... [保留 KEYWORD_BUTTON_MAPPING, HOTBAR_KEY_MAPPING 代码不变] ...
KEYWORD_BUTTON_MAPPING = {
    "F1": "key.keyboard.escape", "KeyS": "key.keyboard.s", "KeyW": "key.keyboard.w",
    "KeyA": "key.keyboard.a", "KeyD": "key.keyboard.d", "KeyE": "key.keyboard.e",
    "KeyQ": "key.keyboard.q", "Digit1": "key.keyboard.1", "Digit2": "key.keyboard.2",
    "Digit3": "key.keyboard.3", "Digit4": "key.keyboard.4", "Digit5": "key.keyboard.5",
    "Digit6": "key.keyboard.6", "Digit7": "key.keyboard.7", "Digit8": "key.keyboard.8",
    "Digit9": "key.keyboard.9", "Space": "key.keyboard.space", "ShiftLeft": "key.keyboard.left.shift",
    "ControlLeft": "key.keyboard.left.control", "KeyF": "key.keyboard.f"
}

HOTBAR_KEY_MAPPING = {
    "key.keyboard.1": 0, "key.keyboard.2": 1, "key.keyboard.3": 2,
    "key.keyboard.4": 3, "key.keyboard.5": 4, "key.keyboard.6": 5,
    "key.keyboard.7": 6, "key.keyboard.8": 7, "key.keyboard.9": 8
}

mouse_button_mapping = {
    0: 0,
    1: 2,
    2: 1
}

# ... [保留 InputState, InputStateManager, LatestFrameContainer 代码不变] ...
@dataclass
class InputState:
    keys_down: Set[str] = field(default_factory=set)
    mouse_buttons: Set[str] = field(default_factory=set)
    frame_dx: float = 0.0
    frame_dy: float = 0.0
    seq: int = 0
    frame_id: int = 0
    current_hotbar_idx: int = 0
    wheel: float = 0.0

class InputStateManager:
    def __init__(self):
        self.state = InputState()
        self.lock = threading.Lock()

    def process_event(self, data: dict):
        ev_type = data.get("event")
        if ev_type == "keydown":
            #self.state.keys_down.add(data.get("key", ""))
            raw_key = data.get("key", "")
            self.state.keys_down.add(raw_key)
            mc_key_name = KEYWORD_BUTTON_MAPPING.get(raw_key)
            if mc_key_name and mc_key_name in HOTBAR_KEY_MAPPING:
                self.state.current_hotbar_idx = HOTBAR_KEY_MAPPING[mc_key_name]

            return raw_key
        elif ev_type == "keyup":
            self.state.keys_down.discard(data.get("key", ""))
        elif ev_type == "mousemove":
            self.state.frame_dx += float(data.get("dx", 0.0))
            self.state.frame_dy += float(data.get("dy", 0.0))
        elif ev_type == "mousedown":
            self.state.mouse_buttons.add(str(data.get("button", "")))
        elif ev_type == "mouseup":
            self.state.mouse_buttons.discard(str(data.get("button", "")))
        return None

    def get_snapshot_and_reset(self):
        with self.lock:
            dx = self.state.frame_dx
            dy = self.state.frame_dy
            self.state.frame_dx = 0
            self.state.frame_dy = 0
            self.state.frame_id += 1
            
            state_snapshot = InputState(
                keys_down=self.state.keys_down.copy(),
                mouse_buttons=self.state.mouse_buttons.copy(),
                frame_dx=dx,
                frame_dy=dy,
                seq=self.state.seq,
                frame_id=self.state.frame_id,
                current_hotbar_idx=self.state.current_hotbar_idx
            )
            return state_snapshot, dx, dy

class LatestFrameContainer:
    def __init__(self):
        self.frame_bytes: Optional[bytes] = None
        self.meta: Optional[dict] = None
        self.condition = asyncio.Condition()
        self.updated = False

    async def put(self, frame_bytes, meta):
        async with self.condition:
            self.frame_bytes = frame_bytes
            self.meta = meta
            self.updated = True
            self.condition.notify()

    async def get(self):
        async with self.condition:
            await self.condition.wait_for(lambda: self.updated)
            self.updated = False
            return self.frame_bytes, self.meta

# ==========================================
# 3. 推理引擎 (使用全局 config)
# ==========================================
class InferenceEngine:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = MineCraftActionTokenizer()
        self.tokenizer.camera_scaler = config.camera_scaler
        self.lock = threading.Lock()
        # Mouse low-pass filter state; reset on V-key / scene refresh.
        self._mouse_ema_dx = 0.0
        self._mouse_ema_dy = 0.0
        # Capture mode state. Do not read config.capture_dir here: this
        # engine is constructed at import time, before argparse mutates the
        # shared config object. load_model() initializes capture after CLI
        # overrides have been applied.
        self._capture_dir = None
        self._capture_max = int(config.capture_max_frames)
        self._captured_frames: list = []
        self._captured_actions: list = []
        self._capture_done: bool = False

    def _init_capture_state(self):
        """Initialize capture after CLI config overrides are visible."""
        self._capture_dir = Path(self.config.capture_dir) if self.config.capture_dir else None
        self._capture_max = int(self.config.capture_max_frames)
        self._captured_frames = []
        self._captured_actions = []
        self._capture_done = False
        if self._capture_dir is not None:
            self._capture_dir.mkdir(parents=True, exist_ok=True)
            print(f"[capture] active: dir={self._capture_dir} max_frames={self._capture_max}")

    def load_model(self):
        # 这里的 config.dynamic_model_path 已经被 argparse 或 env 更新
        print(f"Loading Model from: {self.config.dynamic_model_path}")
        print(f"Loading Tokenizer from: {self.config.tokenizer_path}")
        self._init_capture_state()
        
        self.model = MCWorldModelInfer(
            dynamic_model_path=self.config.dynamic_model_path,
            tokenizer_path=self.config.tokenizer_path,
            record_video_output_path=self.config.record_video_output_path,
            steps_size=self.config.steps_size,
            device=torch.device(self.config.device),
            dtype=self.config.dtype,
            random_generator=torch.Generator(device=self.config.device).manual_seed(self.config.seed),
            use_cuda_graph=True,
            refresh_kvcache=False,
        )
        print("Model Loaded.")

    def _save_capture(self):
        """Serialize captured frames+actions to disk for offline replay."""
        import time as _time
        ts = _time.strftime("%Y%m%d-%H%M%S")
        out_pt = self._capture_dir / f"capture-{ts}.pt"
        latest = self._capture_dir / "latest.pt"
        # Stack to (N, 3, H, W) and (N, 12).
        frames = torch.stack(self._captured_frames, dim=0)
        actions = torch.stack(self._captured_actions, dim=0)
        payload = {
            "frames": frames,             # fp16, [-1, 1], (N, 3, H, W)
            "action_ids": actions,        # long, (N, 12)
            "camera_scaler": float(self.config.camera_scaler),
            "steps_size": int(self.config.steps_size),
            "timestamp": ts,
        }
        torch.save(payload, out_pt)
        torch.save(payload, latest)
        print(f"[capture] saved {frames.shape[0]} frames to {out_pt}")
        # Sidecar mp4 for human inspection of what was captured.
        try:
            import torchvision.io as _tvio
            vid = ((frames.float().clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
            vid = vid.permute(0, 2, 3, 1)  # (N, H, W, 3)
            mp4_path = out_pt.with_suffix(".mp4")
            if mp4_path.exists():
                mp4_path.unlink()
            _tvio.write_video(str(mp4_path), vid, fps=20)
            print(f"[capture] sidecar mp4: {mp4_path}")
        except Exception as e:
            print(f"[capture] sidecar mp4 failed (non-fatal): {e}")

    # ... [保留 reset_kv_cache 和 render 代码不变] ...
    def reset_kv_cache(self):
        if self.model:
            with self.lock:
                self.model.clean_kvcache()
        # Stale mouse filter state would corrupt the first frame after
        # a scene reset; clear it together with the KV cache.
        self._mouse_ema_dx = 0.0
        self._mouse_ema_dy = 0.0
        # V-key starts a fresh capture take. If a previous capture already
        # completed, this re-arms capture so multiple takes can be recorded
        # in one browser session.
        if self._capture_dir is not None:
            if self._captured_frames and not self._capture_done:
                print(f"[capture] V-key reset; discarding partial buffer ({len(self._captured_frames)} frames)")
            elif self._capture_done:
                print("[capture] V-key reset; starting a fresh capture take")
            self._captured_frames = []
            self._captured_actions = []
            self._capture_done = False

    def render(self, state: InputState, dx: float, dy: float) -> bytes:
        if not self.model: return b''

        # ---- Mouse OOD-spike guard ----
        # 1) hard-clip the per-frame magnitude so chaotic input cannot push
        #    the action tensor into bins the model never saw in training.
        clip = float(self.config.mouse_clip)
        if clip > 0:
            if dx > clip: dx = clip
            elif dx < -clip: dx = -clip
            if dy > clip: dy = clip
            elif dy < -clip: dy = -clip
        # 2) optional one-pole IIR low-pass on top, to soften the remaining
        #    frame-to-frame jumps that drive autoregressive amplification.
        alpha = float(self.config.mouse_ema_alpha)
        if 0.0 < alpha < 1.0:
            self._mouse_ema_dx = alpha * dx + (1.0 - alpha) * self._mouse_ema_dx
            self._mouse_ema_dy = alpha * dy + (1.0 - alpha) * self._mouse_ema_dy
            dx = self._mouse_ema_dx
            dy = self._mouse_ema_dy
        else:
            # Keep the filter state in sync even when disabled, so toggling
            # alpha at runtime does not introduce a step.
            self._mouse_ema_dx = dx
            self._mouse_ema_dy = dy

        pressed_keys = [KEYWORD_BUTTON_MAPPING[k] for k in state.keys_down if k in KEYWORD_BUTTON_MAPPING]
        pressed_mouse_btns = [mouse_button_mapping[int(k)] for k in state.mouse_buttons]
        
        #pressed_hotbar = [HOTBAR_KEY_MAPPING[k] for k in pressed_keys if k in HOTBAR_KEY_MAPPING]
        #if pressed_hotbar: state.current_hotbar_idx = pressed_hotbar[0]

        action_dict = {
            "mouse": {"dx": dx, "dy": dy, "buttons": pressed_mouse_btns},
            "keyboard": {"keys": pressed_keys},
            "hotbar": state.current_hotbar_idx,
        }
        
        env_action, _ = self.tokenizer.json_action_to_env_action(action_dict, hotbar=True)
        action_idx_seq = self.tokenizer.get_action_index_from_actiondict(env_action, include_gui=True)
        action_tensor = torch.tensor(action_idx_seq).to(self.model.device)

        with self.lock:
            frame = self.model.render_next_frame(action_id=action_tensor)

        # ---- Capture hook ----
        if self._capture_dir is not None and not self._capture_done:
            self._captured_frames.append(frame.detach().cpu().to(torch.float16))
            self._captured_actions.append(action_tensor.detach().cpu())
            n = len(self._captured_frames)
            if n % 20 == 0:
                print(f"[capture] {n}/{self._capture_max} frames")
            if n >= self._capture_max:
                self._save_capture()
                self._capture_done = True

        img_tensor = ((frame.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)
        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(img_array)
        
        if self.config.resize_width != 640:
            image = image.resize((self.config.resize_width, self.config.resize_height))

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=self.config.jpeg_quality, optimize=True)
        return buf.getvalue()

    def prefilling_from_image(self, image_name: str):
        base_dir = Path(__file__).resolve().parent
        image_path = base_dir / "static" / "start_frames" / image_name
        print(f"[Engine] Prefilling with: {image_name}...")
        
        pil_image = Image.open(image_path).convert("RGB")
        width, height = pil_image.size
        img_tensor = self._apply_image_transform_wo_reshape(pil_image)
        if height != 384:
            #img_tensor = _apply_image_transform_wo_reshape(img)
            img_tensor = self._do_symmetric_pad(
                img_tensor, 1, 384, mode="constant", value=0
            )
        img_tensor = img_tensor[None, None].to(self.model.device).to(self.config.dtype)
        
       
        if self.model:
            with self.lock:
                self.model.clean_kvcache()
                self.model.prefilling_kvcache(init_frames=img_tensor)
  
        
    def record_video(self):
        if self.model:
            with self.lock:
                self.model.record_video()
                
                
    def _do_symmetric_pad(
        self,
        arr: torch.Tensor,
        axis: int,
        target_length: int,
        mode: str = "constant",              # "constant" | "reflect" | "replicate" | "circular" | "edge"
        value: Union[int, float] = 0,
    ) -> torch.Tensor:
        if target_length < 0:
            raise ValueError("target_length must be >= 0")

        ndim = arr.dim()
        if ndim == 0:
            raise ValueError("Scalar tensor not supported.")
        axis = axis % ndim

        cur = arr.shape[axis]
        if target_length <= cur:
            return arr

        total = target_length - cur
        before = total // 2
        after  = total - before

        # numpy "edge" -> torch "replicate"
        torch_mode = "replicate" if mode == "edge" else mode
        if torch_mode not in ("constant", "reflect", "replicate", "circular"):
            raise ValueError(f"Unsupported mode: {mode}")

        # reflect 模式的限制：pad 大小不能超过该维度 size-1
        if torch_mode == "reflect":
            if cur <= 1:
                raise ValueError("Cannot reflect-pad when the dimension size <= 1.")
            if before > cur - 1 or after > cur - 1:
                raise ValueError(
                    f"Reflect padding too large for dim size {cur}: before={before}, after={after} (must be <= {cur-1})."
                )

        # 将目标轴移到最后一维，便于只在最后一维 pad
        perm = [i for i in range(ndim) if i != axis] + [axis]
        inv_perm = [0] * ndim
        for i, p in enumerate(perm):
            inv_perm[p] = i

        x = arr.permute(*perm)
        # 只在最后一维 padding：(before, after)
        pad_tuple = (before, after)

        if torch_mode == "constant":
            out = F.pad(x, pad_tuple, mode=torch_mode, value=float(value))
        else:
            out = F.pad(x, pad_tuple, mode=torch_mode)

        return out.permute(*inv_perm)
    
    def _apply_image_transform_wo_reshape(
        self,
        img_pil: Image.Image,
    ):
        tfm = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5]),
        ])
        return tfm(img_pil)

# ==========================================
# 4. 初始化
# ==========================================

# 1. 创建引擎实例
engine = InferenceEngine(config)

# 2. 定义生命周期 (启动时加载模型)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 这里会读取最新的 config (包括 argparse 修改后的)
    engine.load_model()
    yield
from fastapi.staticfiles import StaticFiles
BASE_DIR = Path(__file__).resolve().parent
static_dir_path = BASE_DIR / "static"
app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=static_dir_path), name="static")
#app.mount("static", StaticFiles(directory="static"), name="static")

# Only one active websocket may drive the global world model at a time.
# Each connection owns an expensive render_loop; multiple tabs/profilers would
# contend on InferenceEngine.lock and roughly divide the FPS between them.
active_ws_lock = asyncio.Lock()
active_ws: Optional[WebSocket] = None
@app.get("/api/start-frames")
def get_start_frames():
    # 指向 static/start_frames 目录
    target_dir = Path(f"{static_dir_path}/start_frames")
    
    # 如果目录不存在，返回空列表
    if not target_dir.exists():
        print(f"Warning: Directory not found: {target_dir}")
        return []
    
    # 扫描 png, jpg, jpeg
    files = []
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG"]:
        files.extend(target_dir.glob(ext))
        
    # 只取文件名，并排序
    file_names = sorted([f.name for f in files])
    print(f"Found frames: {file_names}") # 打印一下方便调试
    return file_names

@app.get("/")
def index():
    return FileResponse(static_dir_path / "index.html")

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    global active_ws
    await ws.accept()
    async with active_ws_lock:
        if active_ws is not None:
            await ws.send_text(json.dumps({
                "type": "error",
                "message": "Another Dreamerv4-MC browser/client is already connected. Close the other tab before opening a new one.",
            }))
            await ws.close(code=1013)
            return
        active_ws = ws
    
    state_manager = InputStateManager()
    latest_frame = LatestFrameContainer()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def recv_loop():
        try:
            while not stop_event.is_set():
                msg = await ws.receive_text()
                data = json.loads(msg)
                if data.get("type") == "event":
                    with state_manager.lock:
                        key = state_manager.process_event(data)
                    if key == "KeyV":
                        engine.reset_kv_cache()
                        state_manager.state.frame_id = 0
                        
                    if key == "KeyR":
                        engine.record_video()
                    
                    event_name = data.get("event")
                    if event_name == "set_start_frame":
                        image_name = data.get("name")
                        if image_name:
                            print(f"Received prefill request: {image_name}")
                            
                            # 重点：这是一个耗时操作(读图+推理)，必须扔到线程池里
                            # 否则会卡住 WebSocket 心跳导致断连
                            await loop.run_in_executor(
                                None, 
                                engine.prefilling_from_image, 
                                image_name
                            )
                            
                            # 重置前端显示的帧计数
                            state_manager.state.frame_id = 0
                            
                            # 可选：发送一个 meta 告诉前端重置成功
                            await ws.send_text(json.dumps({
                                "type": "meta", 
                                "frame_id": 0, 
                                "info": f"Prefilled {image_name}"
                            }))
        except:
            stop_event.set()

    async def render_loop():
        target_interval = 1.0 / config.target_fps
        next_ts = time.perf_counter()
        
        try:
            while not stop_event.is_set():
                now = time.perf_counter()
                if now < next_ts:
                    await asyncio.sleep(next_ts - now)
                next_ts += target_interval

                state_snapshot, dx, dy = state_manager.get_snapshot_and_reset()
                
                # 在线程池中执行
                jpeg_bytes = await loop.run_in_executor(
                    None, 
                    engine.render, 
                    state_snapshot, 
                    dx, 
                    dy
                )

                meta = {
                    "type": "meta",
                    "dx": dx, "dy": dy, 
                    "seq": state_snapshot.seq, 
                    "frame_id": state_snapshot.frame_id
                }
                await latest_frame.put(jpeg_bytes, meta)
        except Exception as e:
            print(f"Render Error: {e}")
            stop_event.set()

    async def send_loop():
        try:
            while not stop_event.is_set():
                frame_bytes, meta = await latest_frame.get()
                await ws.send_text(json.dumps(meta))
                await ws.send_bytes(frame_bytes)
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"Send Error: {e}")
            stop_event.set()

    t1 = asyncio.create_task(recv_loop())
    t2 = asyncio.create_task(render_loop())
    t3 = asyncio.create_task(send_loop())
    
    try:
        await stop_event.wait()
    finally:
        async with active_ws_lock:
            if active_ws is ws:
                active_ws = None
        t1.cancel()
        t2.cancel()
        t3.cancel()

# ==========================================
# 5. 命令行启动入口 (新增)
# ==========================================
if __name__ == "__main__":
    import uvicorn
    
    parser = argparse.ArgumentParser(description="MC World Model Inference Server")
    parser.add_argument("--dynamic_path", type=str, help="Path to dynamic model checkpoint")
    parser.add_argument("--tokenizer_path", type=str, help="Path to tokenizer checkpoint")
    parser.add_argument("--record_video_output_path", type=str, help="Path to save recorded videos")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--steps_size", type=int, default=None, help="Override flow-matching denoise steps; valid powers of 2 only (1,2,4,8,...)")
    parser.add_argument("--mouse_clip", type=float, default=None, help="Per-frame |dx|,|dy| cap before action quantization (default 25.0; 0 disables)")
    parser.add_argument("--mouse_ema_alpha", type=float, default=None, help="One-pole IIR low-pass on per-frame mouse delta; 0.0 disables, 1.0 passthrough (default 0.5)")
    parser.add_argument("--capture_dir", type=str, default=None, help="If set, record every generated (frame, action) pair to .pt files in this dir for multi-frame prefill replay")
    parser.add_argument("--capture_max_frames", type=int, default=None, help="Number of frames to capture before auto-saving (default 200; ~10s @ 20fps)")
    
    args = parser.parse_args()
    if args.steps_size is not None and (
        args.steps_size < 1 or args.steps_size & (args.steps_size - 1)
    ):
        parser.error("--steps_size must be a positive power of two (1, 2, 4, 8, ...)")
    
    # 覆盖默认配置
    if args.dynamic_path:
        config.dynamic_model_path = args.dynamic_path
    if args.tokenizer_path:
        config.tokenizer_path = args.tokenizer_path
        
    if args.record_video_output_path:
        config.record_video_output_path = args.record_video_output_path
    if args.steps_size is not None:
        config.steps_size = args.steps_size
    if args.mouse_clip is not None:
        if args.mouse_clip < 0:
            parser.error("--mouse_clip must be >= 0")
        config.mouse_clip = args.mouse_clip
    if args.mouse_ema_alpha is not None:
        if not (0.0 <= args.mouse_ema_alpha <= 1.0):
            parser.error("--mouse_ema_alpha must be in [0.0, 1.0]")
        config.mouse_ema_alpha = args.mouse_ema_alpha
    print(f"Mouse guard: clip=±{config.mouse_clip}/frame, ema_alpha={config.mouse_ema_alpha}")
    if args.capture_dir is not None:
        config.capture_dir = args.capture_dir
    if args.capture_max_frames is not None:
        if args.capture_max_frames < 1:
            parser.error("--capture_max_frames must be >= 1")
        config.capture_max_frames = args.capture_max_frames
        
    print(f"Starting server on {args.host}:{args.port}")
    
    # 直接运行 uvicorn
    uvicorn.run(app, host=args.host, port=args.port)