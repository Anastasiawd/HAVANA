#!/usr/bin/env python3
"""Havana Policy Server — async slow + fast policy for online VLN closed-loop eval.

Pipeline (one server, two threads)
    Thread A — Fast Policy Loop (main thread of the simulator step loop):
        observation_t           -> ring buffer
        observation_t + z_g_v   -> Stage-2 (CFM) -> trajectory
        trajectory              -> TrajectoryActionAdapter -> Habitat action
    Thread B — Slow Replan Worker (background daemon, queue depth = 1):
        recent obs frames + instruction -> Slow World Model (Cosmos LoRA)
                                       -> future video
                                       -> Stage-1 -> z_g (atomic update)

Top-level usage from `eval_fast.py`:

    from fast_system.scripts.run_pipeline import (
        HavanaPolicyServer,
        PipelineConfig,
        load_pipeline_config,
        PolicyActionProvider,
    )

    server_cfg = load_pipeline_config(Path("/path/to/policy.yaml"))
    server = HavanaPolicyServer(server_cfg)
    server.start()  # blocks until both fast & slow are loaded
    provider = PolicyActionProvider(server, action_names=cfg.action_names)
    run_test_loop(sim, cfg, provider, display=...)
    server.stop()

The class-based public API:
    server.submit_observation(obs, step_id)      # write to ring buffer + maybe trigger replan
    action = server.get_action(obs, step_id)     # observation -> ring buffer -> Stage-2 -> action
    server.set_instruction(text)                 # change the language prompt mid-run
    server.start() / server.stop()               # explicit lifecycle

Slow system can be disabled (`slow_system.enabled: false` in YAML) for fast-only
debugging. In that mode the server runs Stage-2 with `input_goal_mask=1` so the
trained "no-goal" path is exercised, and z_g defaults to a zero tensor.
"""
from __future__ import annotations

# stdlib --------------------------------------------------------------------- #
import argparse
import collections
import logging
import math
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# third-party ---------------------------------------------------------------- #
import numpy as np
import torch
import yaml
from PIL import Image


# Path setup (mirrors demo.py / eval_pipeline.py) ---------------------------- #
FAST_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
HAVANA_ROOT = FAST_SYSTEM_ROOT.parent

if str(HAVANA_ROOT) not in sys.path:
    sys.path.insert(0, str(HAVANA_ROOT))

# fast_system bundles a few research packages as nested source trees rather
# than installing them. Make them importable as top-level modules so
# `from flownav.models...` / `from diffusion_policy.model...` resolve without
# requiring a full `pip install -e`.
_FLOWNAV_SRC = FAST_SYSTEM_ROOT / "model" / "flownav"
_DIFFUSION_POLICY_SRC = FAST_SYSTEM_ROOT / "diffusion_policy"  # has inner pkg
if _FLOWNAV_SRC.is_dir() and str(_FLOWNAV_SRC.parent) not in sys.path:
    sys.path.insert(0, str(_FLOWNAV_SRC.parent))
if _DIFFUSION_POLICY_SRC.is_dir() and str(_DIFFUSION_POLICY_SRC) not in sys.path:
    sys.path.insert(0, str(_DIFFUSION_POLICY_SRC))


# fast_system imports -------------------------------------------------------- #
from fast_system.scripts.eval_pipeline import (  # noqa: E402
    action_stats_from_cfg,
    load_stage1,
    load_stage2,
    make_autocast,
    sample_traj,
)
from fast_system.trainer.train_stage_2 import (  # noqa: E402
    load_config as load_stage2_config,
    resolve_device,
    seed_everything,
)


# =========================================================================== #
# Logging
# =========================================================================== #
LOG = logging.getLogger("havana.policy")


def setup_logger(level: int = logging.INFO) -> None:
    if LOG.handlers:
        LOG.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
                          datefmt="%H:%M:%S")
    )
    LOG.addHandler(handler)
    LOG.setLevel(level)
    LOG.propagate = False


# =========================================================================== #
# Configuration
# =========================================================================== #
@dataclass
class FastSystemConfig:
    """Runtime knobs for the fast (Stage-1 + Stage-2) pipeline."""

    config_path: str = str(FAST_SYSTEM_ROOT / "configs" / "stage2.yaml")
    stage1_ckpt: str = str(FAST_SYSTEM_ROOT / "model_weights" / "stage1.pt")
    stage2_ckpt: str = str(FAST_SYSTEM_ROOT / "model_weights" / "stage2.pt")
    cfm_steps: int = 10
    num_trajectory_samples: int = 1
    use_amp: bool = True


@dataclass
class SlowSystemConfig:
    """How the fast process talks to the slow RPC server.

    The actual cosmos pipeline lives in `slow_system/scripts/slow_server.py`,
    running in its own venv. We only hold the connection knobs here.
    """

    enabled: bool = True
    remote_address: str = "/tmp/havana-slow.sock"   # unix socket path or host:port
    remote_authkey: str = "havana-slow"
    connect_timeout: float = 60.0                   # wait for the slow server to come up
    num_output_frames: int = 77                     # informational fallback if info() fails
    seed: int = 0


@dataclass
class ActionAdapterConfig:
    """Trajectory-to-Habitat-action mapping thresholds."""

    turn_threshold_rad: float = math.radians(10.0)   # > this -> turn
    forward_threshold_m: float = 0.50                 # > this -> forward
    stop_threshold_m: float = 0.05                    # below this & no heading error -> stop
    lookahead_index: int = 8                          # waypoint to evaluate (in trajectory horizon)


@dataclass
class DebugConfig:
    """Debug visualisation: side-by-side mp4 of obs + BEV + slow future-frame."""

    enabled: bool = False
    output_dir: str = "/tmp/havana_debug"
    fps: float = 4.0
    queue_size: int = 32
    save_slow_video: bool = True   # also dump each replan's predicted future as its own mp4


@dataclass
class PipelineConfig:
    """Top-level runtime config for the Havana policy server."""

    instruction: str = "Walk forward and explore the room."
    obs_history_len: int = 32
    slow_replan_interval: int = 8       # trigger slow worker every N fast steps
    slow_min_obs: int = 2               # need ≥ this many frames before first slow replan
    slow_video_frames: int = 8          # how many recent obs frames to send to slow system
    target_fps: float = 4.0             # frames-per-second cadence used for stage1 / slow input
    route_window: int = 16              # Stage-1 input window (matches training)
    fast_sleep_time: float = 0.0        # sleep after each fast tick (lets slow catch up)
    device: str = "cuda"
    seed: int = 0
    log_level: str = "INFO"

    fast: FastSystemConfig = field(default_factory=FastSystemConfig)
    slow: SlowSystemConfig = field(default_factory=SlowSystemConfig)
    adapter: ActionAdapterConfig = field(default_factory=ActionAdapterConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)


def _deep_update(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _instantiate_dataclass(cls, data: dict) -> Any:
    """Construct a (possibly nested) dataclass from a plain dict.

    `from __future__ import annotations` turns field annotations into strings,
    so we can't rely on `field.type` being an actual class. Instead, build a
    default instance and inspect each attribute's runtime type to decide which
    nested fields are themselves dataclasses.
    """
    if not isinstance(data, dict):
        return data
    fields_map = {f.name: f for f in cls.__dataclass_fields__.values()}
    default_obj = cls()
    kwargs: Dict[str, Any] = {}
    for k, v in data.items():
        if k not in fields_map:
            LOG.warning("config: unknown field '%s' for %s — ignoring", k, cls.__name__)
            continue
        default_val = getattr(default_obj, k, None)
        nested_cls = type(default_val) if hasattr(default_val, "__dataclass_fields__") else None
        if nested_cls is not None and isinstance(v, dict):
            kwargs[k] = _instantiate_dataclass(nested_cls, v)
        else:
            kwargs[k] = v
    return cls(**kwargs)


def load_pipeline_config(yaml_path: Optional[Path]) -> PipelineConfig:
    """Load a `PipelineConfig` from an optional YAML file (deep-merged with defaults)."""
    base = PipelineConfig()
    if yaml_path is None:
        return base

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        LOG.warning("pipeline config not found at %s — using defaults", yaml_path)
        return base

    with yaml_path.open() as f:
        loaded = yaml.safe_load(f) or {}

    # Build a default-as-dict view so we can merge.
    def _to_dict(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {f.name: _to_dict(getattr(obj, f.name)) for f in obj.__dataclass_fields__.values()}
        return obj

    merged = _to_dict(base)
    _deep_update(merged, loaded)
    return _instantiate_dataclass(PipelineConfig, merged)


# =========================================================================== #
# Observation ring buffer
# =========================================================================== #
@dataclass
class StoredFrame:
    """A single observation snapshot kept in the ring buffer."""

    step_id: int
    rgb: np.ndarray                         # (H, W, 3) uint8
    depth: Optional[np.ndarray] = None      # (H, W) float32 — may be unused
    timestamp: float = 0.0


class ObservationRingBuffer:
    """Thread-safe ring buffer of recent RGB (and optional depth) frames."""

    def __init__(self, capacity: int):
        self._capacity = max(1, int(capacity))
        self._buf: collections.deque[StoredFrame] = collections.deque(maxlen=self._capacity)
        self._lock = threading.Lock()

    def push(self, frame: StoredFrame) -> None:
        with self._lock:
            self._buf.append(frame)

    def latest_n(self, n: int) -> List[StoredFrame]:
        """Return the up-to-n most recent frames in chronological order."""
        with self._lock:
            return list(self._buf)[-int(n):] if n > 0 else []

    def latest(self) -> Optional[StoredFrame]:
        with self._lock:
            return self._buf[-1] if self._buf else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# =========================================================================== #
# Shared latent state
# =========================================================================== #
@dataclass(frozen=True)
class LatentInfo:
    z_g: torch.Tensor          # [z_dim] on the fast system's device
    version: int               # monotonic, 0 = none
    source_step: int           # fast-loop step that triggered the slow replan
    timestamp: float
    # Stage-1 decode artefacts (CPU numpy) — kept here so the debug visualizer
    # can render the goal/anchors that *go with* this z_g without a second
    # forward pass. None when the slow path didn't run (e.g. stub backend).
    goal_xy: Optional[np.ndarray] = None        # (2,) coarse goal in meters
    anchors_xy: Optional[np.ndarray] = None     # (K, 2) anchor waypoints in meters
    turn_probs: Optional[np.ndarray] = None     # (3,) softmax over [left, straight, right]
    slow_video_u8: Optional[np.ndarray] = None  # (T, H, W, 3) uint8 future video


class SharedLatentState:
    """Atomic single-writer / many-readers exchange for the latest z_g."""

    def __init__(self):
        self._info: Optional[LatentInfo] = None
        self._lock = threading.Lock()
        self._counter = 0

    def update(
        self,
        z_g: torch.Tensor,
        source_step: int,
        *,
        goal_xy: Optional[np.ndarray] = None,
        anchors_xy: Optional[np.ndarray] = None,
        turn_probs: Optional[np.ndarray] = None,
        slow_video_u8: Optional[np.ndarray] = None,
    ) -> int:
        z_g = z_g.detach().clone()
        with self._lock:
            self._counter += 1
            self._info = LatentInfo(
                z_g=z_g,
                version=self._counter,
                source_step=source_step,
                timestamp=time.time(),
                goal_xy=goal_xy,
                anchors_xy=anchors_xy,
                turn_probs=turn_probs,
                slow_video_u8=slow_video_u8,
            )
            return self._counter

    def get(self) -> Optional[LatentInfo]:
        with self._lock:
            return self._info


# =========================================================================== #
# Trajectory-to-action adapter
# =========================================================================== #
class TrajectoryActionAdapter:
    """Convert a Stage-2 trajectory (body-frame meters) to a Habitat discrete action.

    Convention (matches `render_bev` / dataset `rotate_to_local`):
        x is forward, y is left.

    Logic (intentionally minimal — extend later):
        1. Pick a near-future waypoint (`lookahead_index`).
        2. heading_error = atan2(y, x).
        3. |heading_error| > turn_threshold_rad ->
              y > 0 -> turn_left, else -> turn_right
        4. else x > forward_threshold_m -> move_forward
        5. else (waypoint very close & aligned) -> stop
    """

    def __init__(self, cfg: ActionAdapterConfig, action_names: Sequence[str]):
        self.cfg = cfg
        self.available = set(action_names)
        # Resolve canonical action ids — fall back gracefully if the action
        # space is missing one of them.
        self.left = "turn_left" if "turn_left" in self.available else None
        self.right = "turn_right" if "turn_right" in self.available else None
        self.forward = "move_forward" if "move_forward" in self.available else None
        self.stop = "stop" if "stop" in self.available else None
        missing = [n for n in ("move_forward", "turn_left", "turn_right", "stop")
                   if n not in self.available]
        if missing:
            LOG.warning("[Adapter] habitat action space is missing %s — adapter may degrade",
                        missing)

    def convert(self, trajectory_m: np.ndarray, *, step_id: int) -> Tuple[Optional[str], str]:
        """trajectory_m: [H, 2] in meters (x forward, y left). Returns (action_name, reason)."""
        if trajectory_m is None or trajectory_m.size == 0:
            return self.stop, "empty trajectory -> stop"

        H = trajectory_m.shape[0]
        idx = max(0, min(int(self.cfg.lookahead_index), H - 1))
        wp = trajectory_m[idx]
        x_fwd = float(wp[0])
        y_left = float(wp[1])
        heading = math.atan2(y_left, x_fwd) if (x_fwd != 0.0 or y_left != 0.0) else 0.0
        dist = math.hypot(x_fwd, y_left)

        # Decide
        if abs(heading) > self.cfg.turn_threshold_rad:
            if y_left > 0 and self.left is not None:
                action = self.left
                reason = "heading_error %.2frad > %.2f -> turn_left" % (heading, self.cfg.turn_threshold_rad)
            elif y_left <= 0 and self.right is not None:
                action = self.right
                reason = "heading_error %.2frad < -%.2f -> turn_right" % (heading, self.cfg.turn_threshold_rad)
            else:
                action = self.forward or self.stop
                reason = "fallback when turn unavailable"
        elif x_fwd > self.cfg.forward_threshold_m and self.forward is not None:
            action = self.forward
            reason = "forward_dist %.2fm > %.2f -> move_forward" % (x_fwd, self.cfg.forward_threshold_m)
        elif dist < self.cfg.stop_threshold_m and self.stop is not None:
            action = self.stop
            reason = "near origin (dist=%.2fm) -> stop" % dist
        else:
            action = self.forward or self.stop
            reason = "default near-zero motion -> %s" % action

        LOG.info("[Adapter] step=%d wp=[%.2f,%.2f]m heading=%.2frad dist=%.2fm -> %s (%s)",
                 step_id, x_fwd, y_left, heading, dist, action, reason)
        return action, reason


# =========================================================================== #
# Image preprocessing helpers
# =========================================================================== #
def _strip_alpha_to_rgb_u8(rgb: np.ndarray) -> np.ndarray:
    """Coerce arbitrary RGB array to (H, W, 3) uint8."""
    arr = np.asarray(rgb)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected (H,W,3) or (H,W); got shape {arr.shape}")
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            arr = (arr * 255.0).clip(0, 255).astype(np.uint8) if arr.max() <= 1.0 + 1e-6 else arr.clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
    return np.ascontiguousarray(arr)


def to_chw_float(frame_u8: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    """Resize and normalize an RGB frame to (3, H, W) float32 in [0, 1]."""
    img = Image.fromarray(_strip_alpha_to_rgb_u8(frame_u8))
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


# =========================================================================== #
# Slow system backends
# =========================================================================== #
class SlowBackend:
    """Abstract: produces a future-video tensor (T, 3, H, W) in [0, 1] floats."""

    def is_real(self) -> bool:
        return False

    def generate(
        self,
        recent_frames_u8: List[np.ndarray],
        instruction: str,
        seed: int,
    ) -> torch.Tensor:
        raise NotImplementedError


class StubSlowBackend(SlowBackend):
    """Echo the most recent observation as a 'future' video.

    For development / debugging — does not actually run the slow world model.
    """

    def __init__(self, num_frames: int, image_size: int):
        self.num_frames = int(num_frames)
        self.image_size = int(image_size)

    def generate(self, recent_frames_u8, instruction, seed) -> torch.Tensor:
        size = (self.image_size, self.image_size)
        if not recent_frames_u8:
            blank = torch.zeros(3, *size)
            return blank.unsqueeze(0).expand(self.num_frames, -1, -1, -1).clone()
        frames = [to_chw_float(f, size) for f in recent_frames_u8[-self.num_frames:]]
        # Repeat the last frame to fill the future window.
        while len(frames) < self.num_frames:
            frames.append(frames[-1].clone())
        return torch.stack(frames, dim=0)


class RemoteSlowBackend(SlowBackend):
    """Talks to `slow_system/scripts/slow_server.py` over multiprocessing.connection.

    The fast process never imports cosmos. It sends the recent RGB frames + the
    instruction, and gets back a future-video numpy array. We convert to a
    (T, 3, H, W) float tensor on this side so Stage-1 can consume it directly.
    """

    def __init__(self, cfg: SlowSystemConfig):
        from multiprocessing.connection import Client  # stdlib
        self.cfg = cfg
        self._Client = Client
        self._conn = None
        self._connected = False
        self._info: dict = {}
        self._lock = threading.Lock()  # connection is not thread-safe
        self._connect()

    # --------------------------------------------------------------- public #
    def is_real(self) -> bool:
        return self._connected

    @torch.no_grad()
    def generate(self, recent_frames_u8, instruction, seed) -> torch.Tensor:
        if not self._connected:
            raise RuntimeError("slow RPC not connected")
        # `multiprocessing.connection` is not safe under concurrent send/recv,
        # but in practice only the SlowReplanWorker calls us. The lock is here
        # purely to make any future call site safe.
        with self._lock:
            try:
                self._conn.send(("generate", list(recent_frames_u8), str(instruction), int(seed)))
                resp = self._conn.recv()
            except (EOFError, ConnectionResetError, BrokenPipeError) as exc:
                LOG.error("[Slow][RPC] connection lost (%s); reconnecting", exc)
                self._reconnect()
                self._conn.send(("generate", list(recent_frames_u8), str(instruction), int(seed)))
                resp = self._conn.recv()

        if not isinstance(resp, dict) or not resp.get("ok"):
            err = resp.get("error") if isinstance(resp, dict) else repr(resp)
            raise RuntimeError(f"slow RPC error: {err}")
        result = resp["result"]
        video_np = result["video"]                           # (T, H, W, 3) uint8
        elapsed = float(result.get("elapsed", 0.0))
        if video_np.ndim != 4 or video_np.shape[-1] != 3:
            raise ValueError(f"unexpected slow video shape: {video_np.shape}")
        # -> (T, 3, H, W) float32 in [0, 1]
        video = (torch.from_numpy(video_np).float() / 255.0).permute(0, 3, 1, 2).contiguous()
        LOG.info("[Slow][RPC] received future-video shape=%s elapsed=%.2fs",
                 tuple(video.shape), elapsed)
        return video

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
            self._connected = False

    # ------------------------------------------------------------- internal #
    @staticmethod
    def _parse_address(addr: str):
        if isinstance(addr, str) and ":" in addr and not addr.startswith("/"):
            host, port = addr.rsplit(":", 1)
            return (host, int(port))
        return addr

    def _connect(self) -> None:
        addr = self._parse_address(self.cfg.remote_address)
        authkey = self.cfg.remote_authkey.encode()
        deadline = time.time() + max(0.0, float(self.cfg.connect_timeout))
        last_err: Optional[Exception] = None
        LOG.info("[Slow][RPC] connecting to %s (timeout=%.0fs)", addr, self.cfg.connect_timeout)
        while time.time() < deadline:
            try:
                self._conn = self._Client(addr, authkey=authkey)
                self._connected = True
                LOG.info("[Slow][RPC] connected")
                # Probe info() so downstream code knows what shape to expect.
                try:
                    self._conn.send(("info",))
                    info_resp = self._conn.recv()
                    if isinstance(info_resp, dict) and info_resp.get("ok"):
                        self._info = info_resp.get("result", {}) or {}
                        LOG.info("[Slow][RPC] server info: %s", self._info)
                except Exception:  # noqa: BLE001
                    pass
                return
            except (ConnectionRefusedError, FileNotFoundError) as exc:
                last_err = exc
                time.sleep(0.5)
        raise ConnectionError(
            f"could not reach slow RPC server at {addr} within "
            f"{self.cfg.connect_timeout:.0f}s: {last_err}"
        )

    def _reconnect(self) -> None:
        self.close()
        self._connect()


# =========================================================================== #
# Slow replan worker
# =========================================================================== #
class SlowReplanWorker(threading.Thread):
    """Background worker: turns recent obs + instruction into a fresh z_g.

    Only ever holds a single pending request — older requests are dropped when
    a newer one arrives (so the slow system never falls behind).
    """

    def __init__(
        self,
        *,
        slow: SlowBackend,
        stage1: torch.nn.Module,
        shared_latent: SharedLatentState,
        device: torch.device,
        use_amp: bool,
        goal_image_size: int,
        route_window: int,
        instruction_holder: "_InstructionHolder",
        seed: int,
    ):
        super().__init__(daemon=True, name="SlowReplanWorker")
        self._slow = slow
        self._stage1 = stage1
        self._shared = shared_latent
        self._device = device
        self._use_amp = use_amp
        self._goal_size = int(goal_image_size)
        self._route_window = int(route_window)
        self._instruction_holder = instruction_holder
        self._seed = int(seed)

        self._lock = threading.Lock()
        self._pending: Optional[Tuple[int, List[np.ndarray]]] = None
        self._has_request = threading.Event()
        self._stop_event = threading.Event()
        self._is_busy = threading.Event()

    @property
    def busy(self) -> bool:
        return self._is_busy.is_set()

    def request_replan(self, step_id: int, frames: List[np.ndarray]) -> None:
        """Drop any older request and queue the fresh one."""
        with self._lock:
            if self._pending is not None:
                old_step = self._pending[0]
                LOG.info("[SlowWorker] Drop stale slow replan request at step=%d "
                         "(replaced by step=%d)", old_step, step_id)
            self._pending = (step_id, frames)
            self._has_request.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._has_request.set()

    def run(self) -> None:
        LOG.info("[SlowWorker] thread started (real=%s)", self._slow.is_real())
        while not self._stop_event.is_set():
            self._has_request.wait(timeout=1.0)
            if self._stop_event.is_set():
                break
            with self._lock:
                req = self._pending
                self._pending = None
                self._has_request.clear()
            if req is None:
                continue
            try:
                self._is_busy.set()
                self._process(*req)
            except Exception as exc:  # noqa: BLE001
                LOG.error("[SlowWorker] replan failed: %s\n%s", exc, traceback.format_exc())
            finally:
                self._is_busy.clear()
        LOG.info("[SlowWorker] thread stopped")

    @torch.no_grad()
    def _process(self, step_id: int, frames: List[np.ndarray]) -> None:
        instruction = self._instruction_holder.get()
        LOG.info("[SlowWorker] start replan step=%d frames=%d instruction='%s'",
                 step_id, len(frames), instruction[:120])
        t0 = time.time()

        future_video = self._slow.generate(frames, instruction, seed=self._seed)
        slow_secs = time.time() - t0
        LOG.info("[SlowWorker] slow_inference done step=%d shape=%s elapsed=%.2fs",
                 step_id, tuple(future_video.shape), slow_secs)

        # Keep a CPU uint8 copy for the debug visualiser before we move to GPU.
        try:
            slow_video_u8 = (future_video.detach().clamp(0, 1) * 255.0
                             ).to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
        except Exception:  # noqa: BLE001
            slow_video_u8 = None

        # Stage-1 wants [B, route_window, 3, H, W] at goal_image_size.
        future_video = future_video.to(self._device).float()
        if future_video.ndim != 4:
            raise ValueError(f"future_video must be (T, 3, H, W); got {tuple(future_video.shape)}")
        T = future_video.shape[0]
        # Sample `route_window` frames evenly across the predicted future video.
        idx = torch.linspace(0, T - 1, steps=self._route_window).round().long().clamp(0, T - 1)
        sampled = future_video.index_select(0, idx.to(future_video.device))
        # Resize each frame to (goal_size, goal_size).
        sampled = torch.nn.functional.interpolate(
            sampled, size=(self._goal_size, self._goal_size),
            mode="bilinear", align_corners=False,
        )
        goal_batch = sampled.unsqueeze(0)  # (1, route_window, 3, H, W)
        LOG.info("[SlowWorker] stage1 input shape=%s", tuple(goal_batch.shape))

        t1 = time.time()
        with make_autocast(self._device, self._use_amp):
            stage1_out = self._stage1(goal_batch, sample=False)
        z_g = stage1_out["z_g"].detach().float().squeeze(0)  # (z_dim,)
        # Decode artefacts for visualisation (single batch element).
        try:
            goal_xy = stage1_out["goal_pred"].detach().float().cpu().numpy()[0]
        except Exception:  # noqa: BLE001
            goal_xy = None
        try:
            anchors_xy = stage1_out["anchor_pred"].detach().float().cpu().numpy()[0]
        except Exception:  # noqa: BLE001
            anchors_xy = None
        try:
            turn_probs = torch.softmax(
                stage1_out["turn_logits"].detach().float(), dim=-1
            ).cpu().numpy()[0]
        except Exception:  # noqa: BLE001
            turn_probs = None

        stage1_secs = time.time() - t1
        version = self._shared.update(
            z_g, source_step=step_id,
            goal_xy=goal_xy, anchors_xy=anchors_xy,
            turn_probs=turn_probs, slow_video_u8=slow_video_u8,
        )
        LOG.info("[SlowWorker] stage1 done step=%d z_g.shape=%s version=%d stage1_time=%.2fs total=%.2fs",
                 step_id, tuple(z_g.shape), version, stage1_secs, time.time() - t0)


class _InstructionHolder:
    """Thread-safe single-string container."""

    def __init__(self, text: str):
        self._text = text
        self._lock = threading.Lock()

    def get(self) -> str:
        with self._lock:
            return self._text

    def set(self, text: str) -> None:
        with self._lock:
            self._text = text


# =========================================================================== #
# Debug visualiser
# =========================================================================== #
@dataclass
class DebugSnapshot:
    """One frame's worth of debug info; rendered async on a background thread."""

    step_id: int
    obs_rgb_u8: np.ndarray                       # (H, W, 3) raw observation
    trajectory_m: np.ndarray                     # (H, 2) Stage-2 prediction
    action: Optional[str]
    z_version: int
    goal_xy: Optional[np.ndarray] = None         # (2,) coarse goal in meters
    anchors_xy: Optional[np.ndarray] = None      # (K, 2) anchor waypoints
    turn_probs: Optional[np.ndarray] = None      # (3,) turn-prob for the latest z_g
    slow_video_u8: Optional[np.ndarray] = None   # (T, H, W, 3) — only set on a fresh replan


class DebugVisualizer(threading.Thread):
    """Async writer that turns DebugSnapshots into a side-by-side debug mp4.

    Layout per frame (rendered with matplotlib via demo.py's helpers):
        [ obs RGB | BEV with pred + anchors + goal | latest slow future-frame ]

    Runs on its own daemon thread so heavy matplotlib renders never block the
    closed loop. Drops snapshots silently if the queue overflows.
    """

    def __init__(self, cfg: DebugConfig):
        super().__init__(daemon=True, name="DebugVisualizer")
        self.cfg = cfg
        self._queue: "collections.deque[DebugSnapshot]" = collections.deque(
            maxlen=max(1, int(cfg.queue_size))
        )
        self._lock = threading.Lock()
        self._has_item = threading.Event()
        self._stop_event = threading.Event()

        out_dir = Path(cfg.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._main_path = out_dir / f"debug_{ts}.mp4"
        self._slow_dir = out_dir / f"slow_videos_{ts}"
        if cfg.save_slow_video:
            self._slow_dir.mkdir(parents=True, exist_ok=True)

        self._writer = None
        self._side_px = 256
        self._frame_h = 0
        self._frame_w = 0
        self._extent = 4.0
        self._dropped = 0
        # Most recent slow-video panel (resized to obs height). Cached so every
        # rendered frame has a fixed 3-column layout — without this the writer
        # would be opened on a 2-panel frame (no slow video yet) and crash the
        # first time a slow frame finally lands.
        self._last_slow_panel: Optional[np.ndarray] = None

    @property
    def output_path(self) -> Path:
        return self._main_path

    def submit(self, snap: DebugSnapshot) -> None:
        with self._lock:
            if len(self._queue) == self._queue.maxlen:
                self._dropped += 1
                # Drop the oldest pending so the most recent frame survives.
                self._queue.popleft()
            self._queue.append(snap)
            self._has_item.set()

    def stop(self) -> None:
        self._stop_event.set()
        self._has_item.set()

    # ----------------------------------------------------------------- run #
    def run(self) -> None:
        LOG.info("[DebugViz] started -> %s", self._main_path)
        last_slow_version = 0
        try:
            while not self._stop_event.is_set() or self._queue:
                self._has_item.wait(timeout=0.5)
                while True:
                    with self._lock:
                        if not self._queue:
                            self._has_item.clear()
                            break
                        snap = self._queue.popleft()
                    self._render_one(snap)
                    if (self.cfg.save_slow_video and snap.slow_video_u8 is not None
                            and snap.z_version != last_slow_version):
                        self._dump_slow_video(snap)
                        last_slow_version = snap.z_version
        except Exception as exc:  # noqa: BLE001
            LOG.error("[DebugViz] crashed: %s\n%s", exc, traceback.format_exc())
        finally:
            if self._writer is not None:
                try:
                    self._writer.close()
                except Exception:  # noqa: BLE001
                    pass
            if self._dropped:
                LOG.warning("[DebugViz] dropped %d snapshots due to queue overflow", self._dropped)
            LOG.info("[DebugViz] done -> %s", self._main_path)

    # ---------------------------------------------------------- internals #
    def _ensure_writer(self, frame_h: int, frame_w: int) -> None:
        if self._writer is not None:
            return
        import imageio.v2 as imageio  # lazy: imageio is optional in cosmos venv
        self._writer = imageio.get_writer(
            str(self._main_path), fps=float(self.cfg.fps), macro_block_size=1
        )
        self._side_px = int(frame_h)

    def _render_bev(self, snap: DebugSnapshot, side_px: int) -> np.ndarray:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        traj = snap.trajectory_m
        # Rolling extent — keep the camera tight on whatever we're plotting now.
        candidates = [np.abs(traj).max() if traj.size else 0.0]
        if snap.anchors_xy is not None and snap.anchors_xy.size:
            candidates.append(float(np.abs(snap.anchors_xy).max()))
        if snap.goal_xy is not None:
            candidates.append(float(np.abs(snap.goal_xy).max()))
        extent = max(1.0, max(candidates) * 1.2)

        fig = plt.figure(figsize=(4, 4), dpi=max(64, side_px // 4))
        ax = fig.add_subplot(111)
        ax.plot([0.0, *traj[:, 1].tolist()], [0.0, *traj[:, 0].tolist()],
                "-o", color="#d62728", markersize=3, linewidth=2, label="Pred")
        if snap.anchors_xy is not None and snap.anchors_xy.size:
            ax.plot([0.0, *snap.anchors_xy[:, 1].tolist()],
                    [0.0, *snap.anchors_xy[:, 0].tolist()],
                    "--", color="#ff7f0e", linewidth=1.5, alpha=0.9, label="anchors (z_g)")
            ax.scatter(snap.anchors_xy[:, 1], snap.anchors_xy[:, 0],
                       c="#ff7f0e", s=15, zorder=4)
        if snap.goal_xy is not None:
            ax.scatter([snap.goal_xy[1]], [snap.goal_xy[0]], marker="*",
                       c="#ffbf00", s=140, edgecolors="black",
                       linewidths=0.8, zorder=6, label="goal (z_g)")
        ax.scatter([0.0], [0.0], c="k", marker="s", s=50, zorder=5, label="origin")
        ax.set_xlim(extent, -extent)
        ax.set_ylim(-0.5, extent * 2)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("y (left, m)")
        ax.set_ylabel("x (forward, m)")
        ax.legend(loc="upper left", fontsize=7)
        title = f"step {snap.step_id}  z_v{snap.z_version}  -> {snap.action or '·'}"
        if snap.turn_probs is not None:
            title += f"   turn=[{snap.turn_probs[0]:.2f},{snap.turn_probs[1]:.2f},{snap.turn_probs[2]:.2f}]"
        ax.set_title(title, fontsize=9)
        fig.tight_layout(pad=0.4)

        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
        plt.close(fig)
        return np.asarray(Image.fromarray(img).resize((side_px, side_px), Image.BILINEAR))

    @staticmethod
    def _resize_to_h(img: np.ndarray, target_h: int) -> np.ndarray:
        if img.shape[0] == target_h:
            return img
        new_w = max(1, int(img.shape[1] * target_h / img.shape[0]))
        return np.asarray(Image.fromarray(img).resize((new_w, target_h), Image.BILINEAR))

    def _render_one(self, snap: DebugSnapshot) -> None:
        obs = _strip_alpha_to_rgb_u8(snap.obs_rgb_u8)
        # Lock the row height to whatever the very first obs frame had so that
        # a downstream camera resize never breaks the writer.
        if self._writer is None:
            self._side_px = int(obs.shape[0])
        side_px = int(self._side_px)
        bev = self._render_bev(snap, side_px=side_px)

        # Refresh the cached slow panel whenever a new replan landed; otherwise
        # reuse the last one so frame width stays constant for the mp4 writer.
        if snap.slow_video_u8 is not None and snap.slow_video_u8.size:
            mid = snap.slow_video_u8.shape[0] // 2
            self._last_slow_panel = self._resize_to_h(snap.slow_video_u8[mid], side_px)
        if self._last_slow_panel is None:
            slow_panel = np.zeros((side_px, side_px, 3), dtype=np.uint8)
        else:
            slow_panel = self._resize_to_h(self._last_slow_panel, side_px)

        panels = [self._resize_to_h(obs, side_px),
                  self._resize_to_h(bev, side_px),
                  slow_panel]
        frame = np.concatenate(panels, axis=1)
        if frame.shape[0] % 2 or frame.shape[1] % 2:
            frame = np.pad(frame, ((0, frame.shape[0] % 2),
                                   (0, frame.shape[1] % 2),
                                   (0, 0)), mode="edge")
        # Pin the writer's expected frame shape on first use; force-resize every
        # subsequent frame to the same shape so we never raise on mismatch.
        if self._writer is None:
            self._frame_h, self._frame_w = frame.shape[:2]
        if frame.shape[0] != self._frame_h or frame.shape[1] != self._frame_w:
            frame = np.asarray(Image.fromarray(frame).resize(
                (self._frame_w, self._frame_h), Image.BILINEAR))
        self._ensure_writer(self._frame_h, self._frame_w)
        self._writer.append_data(frame)

    def _dump_slow_video(self, snap: DebugSnapshot) -> None:
        try:
            import imageio.v2 as imageio
            path = self._slow_dir / f"slow_v{snap.z_version:04d}_step{snap.step_id:06d}.mp4"
            with imageio.get_writer(str(path), fps=float(self.cfg.fps), macro_block_size=1) as w:
                for fr in snap.slow_video_u8:
                    f = fr
                    if f.shape[0] % 2 or f.shape[1] % 2:
                        f = np.pad(f, ((0, f.shape[0] % 2),
                                       (0, f.shape[1] % 2),
                                       (0, 0)), mode="edge")
                    w.append_data(f)
            LOG.info("[DebugViz] dumped slow video -> %s (T=%d)",
                     path, snap.slow_video_u8.shape[0])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[DebugViz] failed to dump slow video: %s", exc)


# =========================================================================== #
# Fast policy runner
# =========================================================================== #
class FastPolicyRunner:
    """Stage-2 policy: builds an obs context from the ring buffer and predicts a trajectory.

    Pre-conditions:
        - `obs_buffer` has at least one frame.
        - `shared_latent.get()` is not None (otherwise call `predict_fallback`).
    """

    def __init__(
        self,
        *,
        stage2: torch.nn.Module,
        cfg_fast: FastSystemConfig,
        stage2_yaml_cfg: Dict[str, Any],
        device: torch.device,
        use_amp: bool,
    ):
        self.stage2 = stage2
        self.cfg = cfg_fast
        self.device = device
        self.use_amp = use_amp

        data_cfg = stage2_yaml_cfg.get("scand_data", {})
        self.obs_image_size: int = int(data_cfg.get("obs_image_size", 256))
        self.context_size: int = int(
            data_cfg.get("context_size", stage2_yaml_cfg.get("model", {}).get("context_size", 3))
        )
        self.context_dt_frames: int = int(data_cfg.get("context_dt_frames", 1))
        self.action_horizon: int = int(data_cfg.get("action_horizon", 16))
        self.action_dim: int = int(stage2_yaml_cfg.get("model", {}).get("action_dim", 2))
        self.waypoint_spacing: float = float(data_cfg.get("metric_waypoint_spacing", 0.38))
        self.normalize_actions: bool = bool(data_cfg.get("normalize_actions", True))
        self.pred_to_meter: float = self.waypoint_spacing if self.normalize_actions else 1.0
        self.action_stats = action_stats_from_cfg(stage2_yaml_cfg)

        # Inferred from Stage-2 ckpt (z_dim).
        self.z_dim: int = int(stage2_yaml_cfg.get("model", {}).get("z_dim", 128))
        self.zero_z_g = torch.zeros(self.z_dim, device=self.device)

    @property
    def goal_image_size(self) -> int:
        # Stage-1 takes goal_image_size frames; mirror demo.py default.
        # We deliberately keep this the same as obs to avoid a second resize path.
        return self.obs_image_size

    def _build_obs_context_chw(self, recent: List[StoredFrame]) -> torch.Tensor:
        """Build a (1, context+1, 3, H, W) Stage-2 obs tensor from recent frames."""
        size = (self.obs_image_size, self.obs_image_size)
        if not recent:
            blank = to_chw_float(np.zeros((size[1], size[0], 3), dtype=np.uint8), size)
            stack = blank.unsqueeze(0).expand(self.context_size + 1, -1, -1, -1).clone()
            return stack.unsqueeze(0).to(self.device)

        # Walk back at training temporal spacing; clamp to oldest frame.
        chw_cache: Dict[int, torch.Tensor] = {}
        n = len(recent)
        cur = n - 1
        idx_seq: List[int] = []
        for off in range(self.context_size, -1, -1):
            j = cur - off * self.context_dt_frames
            j = max(0, min(j, n - 1))
            idx_seq.append(j)
        for j in idx_seq:
            if j not in chw_cache:
                chw_cache[j] = to_chw_float(recent[j].rgb, size)
        stack = torch.stack([chw_cache[j] for j in idx_seq], dim=0)
        return stack.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_trajectory(
        self,
        recent: List[StoredFrame],
        z_g: Optional[torch.Tensor],
        *,
        step_id: int,
        latent_version: int,
    ) -> np.ndarray:
        """Returns trajectory of shape (H, 2) in meters (x forward, y left)."""
        obs_batch = self._build_obs_context_chw(recent)
        if z_g is None:
            z_used = self.zero_z_g.unsqueeze(0)
            mask = torch.ones(1, device=self.device, dtype=torch.long)
            mode = "no-zg (input_goal_mask=1)"
        else:
            z_used = z_g.detach().to(self.device).unsqueeze(0)
            mask = None
            mode = f"z_g v{latent_version}"

        t0 = time.time()
        traj = sample_traj(
            self.stage2, obs_batch, z_used,
            horizon=self.action_horizon,
            action_dim=self.action_dim,
            steps=self.cfg.cfm_steps,
            action_stats=self.action_stats,
            input_goal_mask=mask,
            num_samples=max(1, int(self.cfg.num_trajectory_samples)),
            device=self.device,
            amp=self.use_amp,
        )
        traj_m = (traj * self.pred_to_meter).detach().cpu().numpy()[0]  # (H, 2)
        elapsed = time.time() - t0
        preview = traj_m[: min(3, traj_m.shape[0])]
        LOG.info("[FastLoop] step=%d mode=%s obs=%s traj=%s preview=%s fast_time=%.3fs",
                 step_id, mode, tuple(obs_batch.shape), tuple(traj_m.shape),
                 np.array2string(preview, precision=2, suppress_small=True), elapsed)
        return traj_m


# =========================================================================== #
# Havana policy server
# =========================================================================== #
class HavanaPolicyServer:
    """Top-level coordinator. One instance per simulator process."""

    def __init__(self, cfg: PipelineConfig):
        setup_logger(getattr(logging, str(cfg.log_level).upper(), logging.INFO))
        self.cfg = cfg
        self._device = resolve_device(cfg.device)
        self._use_amp = bool(cfg.fast.use_amp) and self._device.type == "cuda"
        seed_everything(int(cfg.seed))

        LOG.info("=" * 72)
        LOG.info("Havana Policy Server starting up")
        LOG.info("  device         : %s (amp=%s)", self._device, self._use_amp)
        LOG.info("  fast.config    : %s", cfg.fast.config_path)
        LOG.info("  fast.stage1    : %s", cfg.fast.stage1_ckpt)
        LOG.info("  fast.stage2    : %s", cfg.fast.stage2_ckpt)
        LOG.info("  slow.enabled   : %s", cfg.slow.enabled)
        if cfg.slow.enabled:
            LOG.info("  slow.remote    : %s", cfg.slow.remote_address)
            LOG.info("  slow.timeout   : %.0fs", cfg.slow.connect_timeout)
        LOG.info("  instruction    : %s", cfg.instruction)
        LOG.info("=" * 72)

        # ----- Fast system load --------------------------------------------- #
        t_fast = time.time()
        stage2_cfg = load_stage2_config(Path(cfg.fast.config_path))
        self._stage1 = load_stage1(Path(cfg.fast.stage1_ckpt), self._device)
        self._stage2 = load_stage2(Path(cfg.fast.stage2_ckpt), stage2_cfg, self._device).to(self._device)
        self._stage2_cfg = stage2_cfg
        LOG.info("[Init] fast system loaded in %.2fs", time.time() - t_fast)

        # ----- Fast runner -------------------------------------------------- #
        self._fast_runner = FastPolicyRunner(
            stage2=self._stage2,
            cfg_fast=cfg.fast,
            stage2_yaml_cfg=stage2_cfg,
            device=self._device,
            use_amp=self._use_amp,
        )

        # ----- Observation buffer / shared state --------------------------- #
        self._obs_buf = ObservationRingBuffer(cfg.obs_history_len)
        self._shared_zg = SharedLatentState()
        self._instruction = _InstructionHolder(cfg.instruction)

        # ----- Slow backend ------------------------------------------------- #
        if cfg.slow.enabled:
            t_slow = time.time()
            try:
                slow_backend: SlowBackend = RemoteSlowBackend(cfg.slow)
                LOG.info("[Init] slow RPC backend ready in %.1fs", time.time() - t_slow)
            except Exception as exc:  # noqa: BLE001
                LOG.error("[Init] failed to reach slow RPC server (%s); "
                          "falling back to stub. Set slow_system.enabled=false to silence.",
                          exc)
                LOG.debug(traceback.format_exc())
                slow_backend = StubSlowBackend(
                    num_frames=cfg.slow.num_output_frames,
                    image_size=self._fast_runner.goal_image_size,
                )
        else:
            LOG.info("[Init] slow backend disabled — using stub (zero z_g via masked path)")
            slow_backend = StubSlowBackend(
                num_frames=cfg.slow.num_output_frames,
                image_size=self._fast_runner.goal_image_size,
            )
        self._slow_backend = slow_backend

        # ----- Slow worker (started in start()) ----------------------------- #
        self._slow_worker = SlowReplanWorker(
            slow=self._slow_backend,
            stage1=self._stage1,
            shared_latent=self._shared_zg,
            device=self._device,
            use_amp=self._use_amp,
            goal_image_size=self._fast_runner.goal_image_size,
            route_window=cfg.route_window,
            instruction_holder=self._instruction,
            seed=cfg.seed,
        )

        # ----- Debug visualiser (optional) --------------------------------- #
        self._visualizer: Optional[DebugVisualizer] = None
        if cfg.debug.enabled:
            self._visualizer = DebugVisualizer(cfg.debug)
            LOG.info("[Init] debug visualiser enabled -> %s", self._visualizer.output_path)

        self._adapter: Optional[TrajectoryActionAdapter] = None
        self._step_count = 0
        self._started = False
        # Tracks when the *first* z_g has been published. Until then we always
        # try to trigger a replan; after that we only replan on the configured
        # interval (so we know the scene has had time to actually change).
        self._first_zg_seen = False
        self._last_visualised_version = 0

        LOG.info("Policy server is ready (slow_real=%s, fast_amp=%s)",
                 self._slow_backend.is_real(), self._use_amp)

    # ------------------------------------------------------------------ API #
    def attach_action_space(self, action_names: Sequence[str]) -> None:
        """Tell the server which Habitat actions are legal — required before get_action."""
        self._adapter = TrajectoryActionAdapter(self.cfg.adapter, action_names)
        LOG.info("[Init] adapter ready for actions=%s", list(action_names))

    def set_instruction(self, text: str) -> None:
        LOG.info("[Server] instruction updated: '%s'", text)
        self._instruction.set(text)

    def start(self) -> None:
        if self._started:
            return
        self._slow_worker.start()
        if self._visualizer is not None:
            self._visualizer.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        LOG.info("[Server] shutting down")
        self._slow_worker.stop()
        self._slow_worker.join(timeout=5.0)
        if self._visualizer is not None:
            self._visualizer.stop()
            self._visualizer.join(timeout=10.0)
        # If the slow backend holds an RPC connection, close it cleanly.
        close_fn = getattr(self._slow_backend, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:  # noqa: BLE001
                pass
        self._started = False

    def __enter__(self) -> "HavanaPolicyServer":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ----------------------------------------------------------- ingestion #
    def submit_observation(self, obs: Dict[str, Any], step_id: int) -> None:
        """Push obs into the ring buffer and possibly trigger the slow worker."""
        rgb = obs.get("rgb_sensor", obs.get("rgb"))
        if rgb is None:
            LOG.warning("[Server] obs has no rgb_sensor/rgb key (keys=%s)", list(obs.keys()))
            return
        rgb_u8 = _strip_alpha_to_rgb_u8(rgb)
        depth = obs.get("depth_sensor", obs.get("depth"))
        depth_arr = None
        if isinstance(depth, np.ndarray):
            d = depth
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            depth_arr = d.astype(np.float32)

        frame = StoredFrame(step_id=step_id, rgb=rgb_u8, depth=depth_arr, timestamp=time.time())
        self._obs_buf.push(frame)

        if step_id % 50 == 0:
            LOG.info("[Server] step=%d obs_keys=%s rgb_shape=%s depth_shape=%s buffer_len=%d",
                     step_id, list(obs.keys()), rgb_u8.shape,
                     depth_arr.shape if depth_arr is not None else None,
                     len(self._obs_buf))

        # Decide whether to wake the slow worker:
        #   - Until the very first z_g lands, hammer it as soon as we have
        #     enough history (no use waiting `slow_replan_interval` ticks if
        #     we don't even have a z_g yet — Stage-2 has nothing real to use).
        #   - After the first z_g, only replan on the configured cadence so
        #     that consecutive replans see meaningfully different scenes.
        have_history = len(self._obs_buf) >= max(1, int(self.cfg.slow_min_obs))
        interval = max(1, int(self.cfg.slow_replan_interval))
        if not self._first_zg_seen:
            self._first_zg_seen = self._shared_zg.get() is not None
        if have_history and (
            not self._first_zg_seen or step_id % interval == 0
        ):
            recent_frames_u8 = [f.rgb for f in self._obs_buf.latest_n(self.cfg.slow_video_frames)]
            self._slow_worker.request_replan(step_id, recent_frames_u8)

    # ------------------------------------------------------------- action #
    def get_action(self, obs: Optional[Dict[str, Any]] = None,
                   step_id: Optional[int] = None) -> Optional[str]:
        """One simulator tick: ingest obs (if provided) and produce a Habitat action.

        Returns None when the system is not yet ready (e.g. z_g not available
        and the real slow system is still warming up). Callers should treat
        None as "skip / no-op this tick" and try again next step.
        """
        if step_id is None:
            step_id = self._step_count
        if obs is not None:
            self.submit_observation(obs, step_id)
        self._step_count = max(self._step_count, step_id + 1)

        if self._adapter is None:
            LOG.error("[Server] adapter not attached — call attach_action_space() first")
            return None

        recent = self._obs_buf.latest_n(self._fast_runner.context_size + 1)
        if not recent:
            LOG.warning("[FastLoop] step=%d obs buffer empty -> safe fallback (no-op)", step_id)
            return None

        latent = self._shared_zg.get()
        slow_real = self._slow_backend.is_real()

        if latent is None and slow_real:
            # Real slow system is still warming up — emit a no-op (None) so the
            # caller can simply repeat next tick without ending the episode.
            busy = "busy" if self._slow_worker.busy else "idle"
            LOG.info("[FastLoop] step=%d z_g is not ready yet (slow worker %s) "
                     "-> safe fallback (no-op)", step_id, busy)
            return None

        # Pick the conditioning path:
        #   - real slow + valid z_g -> use it
        #   - stub slow OR latent is None -> always run masked (no-zg) trajectory
        #     so we can still drive the closed loop without trusting a stubbed
        #     latent.
        if slow_real and latent is not None:
            z_arg = latent.z_g
            version_arg = latent.version
        else:
            z_arg = None
            version_arg = 0

        try:
            traj_m = self._fast_runner.predict_trajectory(
                recent, z_g=z_arg, step_id=step_id, latent_version=version_arg,
            )
        except Exception as exc:
            LOG.error("[FastLoop] inference failed: %s\n%s", exc, traceback.format_exc())
            return None

        try:
            action, _ = self._adapter.convert(traj_m, step_id=step_id)
        except Exception as exc:
            LOG.error("[Adapter] conversion failed: %s\n%s", exc, traceback.format_exc())
            return None

        self._maybe_publish_debug(recent[-1], traj_m, action, latent, step_id)

        sleep_s = float(getattr(self.cfg, "fast_sleep_time", 0.0))
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        return action

    def _maybe_publish_debug(
        self,
        last_frame: StoredFrame,
        traj_m: np.ndarray,
        action: Optional[str],
        latent: Optional[LatentInfo],
        step_id: int,
    ) -> None:
        if self._visualizer is None:
            return
        # Only attach the slow video to the snapshot the *first* time we see a
        # given z_g version — otherwise the debug mp4 / per-replan dump would
        # repeat the same future video on every fast tick.
        slow_video = None
        if latent is not None and latent.slow_video_u8 is not None:
            if latent.version != self._last_visualised_version:
                slow_video = latent.slow_video_u8
                self._last_visualised_version = latent.version
        snap = DebugSnapshot(
            step_id=step_id,
            obs_rgb_u8=last_frame.rgb,
            trajectory_m=np.asarray(traj_m, dtype=np.float32),
            action=action,
            z_version=latent.version if latent is not None else 0,
            goal_xy=latent.goal_xy if latent is not None else None,
            anchors_xy=latent.anchors_xy if latent is not None else None,
            turn_probs=latent.turn_probs if latent is not None else None,
            slow_video_u8=slow_video,
        )
        self._visualizer.submit(snap)


# =========================================================================== #
# Habitat ActionProvider adapter (used by eval_fast.py)
# =========================================================================== #
class PolicyActionProvider:
    """A drop-in replacement for `BaseActionProvider` from eval_fast.py.

    eval_fast.py never imports this directly; it uses duck-typing — only
    `get_action(obs) -> Optional[str]` is required. We also accept an optional
    callable for `step_id` so callers can drive the counter externally.
    """

    def __init__(self, server: HavanaPolicyServer, action_names: Sequence[str]):
        self._server = server
        if server._adapter is None:
            server.attach_action_space(action_names)
        self._step_id = 0

    def get_action(self, obs: Dict[str, Any]) -> Optional[str]:
        sid = self._step_id
        self._step_id += 1
        return self._server.get_action(obs, sid)


# =========================================================================== #
# RPC server (cross-process / cross-venv)
# =========================================================================== #
class PolicyRpcServer:
    """Wraps a `HavanaPolicyServer` behind a `multiprocessing.connection` listener.

    Lets the Habitat venv (which has no torch / cosmos) talk to the policy
    venv over a unix socket or TCP. Single-client, blocking; that matches the
    one-simulator-per-server contract.

    Wire format: each request is a tuple `(op, *args)` and each response is
    `{"ok": bool, "result": ...}` or `{"ok": False, "error": "..."}`.
    """

    _OPS = {"ping", "attach_action_space", "set_instruction", "get_action"}

    def __init__(self, server: "HavanaPolicyServer", address: str,
                 authkey: bytes = b"havana"):
        from multiprocessing.connection import Listener
        self.server = server
        self.address = self._parse_address(address)
        self.authkey = authkey
        # Ensure stale unix sockets don't block restart.
        if isinstance(self.address, str):
            try:
                Path(self.address).unlink()
            except FileNotFoundError:
                pass
        self._listener = Listener(self.address, authkey=authkey)
        LOG.info("[RPC] listening on %s", self.address)

    @staticmethod
    def _parse_address(addr: str):
        if isinstance(addr, str) and ":" in addr and not addr.startswith("/"):
            host, port = addr.rsplit(":", 1)
            return (host, int(port))
        return addr

    def serve_forever(self) -> None:
        try:
            while True:
                LOG.info("[RPC] waiting for client...")
                conn = self._listener.accept()
                LOG.info("[RPC] client connected from %s", self._listener.last_accepted)
                try:
                    self._serve_one(conn)
                except (EOFError, ConnectionResetError):
                    LOG.info("[RPC] client disconnected")
                finally:
                    try:
                        conn.close()
                    except Exception:  # noqa: BLE001
                        pass
        except KeyboardInterrupt:
            LOG.info("[RPC] interrupted by user")
        finally:
            try:
                self._listener.close()
            except Exception:  # noqa: BLE001
                pass

    def _serve_one(self, conn) -> None:
        while True:
            try:
                req = conn.recv()
            except EOFError:
                return
            if not isinstance(req, tuple) or not req:
                conn.send({"ok": False, "error": f"bad request: {req!r}"})
                continue
            op, *args = req
            if op not in self._OPS:
                conn.send({"ok": False, "error": f"unknown op {op!r}"})
                continue
            try:
                result = self._dispatch(op, args)
                conn.send({"ok": True, "result": result})
            except Exception as exc:  # noqa: BLE001
                LOG.error("[RPC] op=%s failed: %s\n%s", op, exc, traceback.format_exc())
                conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})

    def _dispatch(self, op: str, args: list):
        if op == "ping":
            return "pong"
        if op == "attach_action_space":
            self.server.attach_action_space(args[0])
            return None
        if op == "set_instruction":
            self.server.set_instruction(args[0])
            return None
        if op == "get_action":
            obs, step_id = args[0], args[1]
            return self.server.get_action(obs, step_id)
        raise AssertionError(f"unhandled op {op!r}")


# =========================================================================== #
# CLI smoke test
# =========================================================================== #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Boot the Havana policy server.")
    p.add_argument("--config", type=Path, default=None,
                   help="Path to a YAML PipelineConfig (optional).")
    p.add_argument("--instruction", type=str, default=None,
                   help="Override instruction text.")
    p.add_argument("--slow-disabled", action="store_true",
                   help="Force slow_system.enabled=false for quick fast-only smoke tests.")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="Run a synthetic in-process smoke loop for this many steps.")
    p.add_argument("--serve", type=str, default=None,
                   help="Run as an RPC server on this address (UNIX socket path or host:port). "
                        "Used to bridge Habitat venv -> policy venv.")
    p.add_argument("--authkey", type=str, default="havana",
                   help="HMAC key for the multiprocessing.connection listener.")
    return p.parse_args()


def _smoke_loop(server: HavanaPolicyServer, num_steps: int) -> None:
    """Drive the server with random RGB observations to verify wiring."""
    LOG.info("[SmokeTest] running %d synthetic steps", num_steps)
    server.attach_action_space(["stop", "move_forward", "turn_left", "turn_right",
                                "look_up", "look_down"])
    rng = np.random.default_rng(0)
    for sid in range(num_steps):
        obs = {
            "rgb_sensor": rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8),
            "depth_sensor": rng.random((480, 640), dtype=np.float32) * 5.0,
        }
        action = server.get_action(obs, sid)
        LOG.info("[SmokeTest] step=%d -> action=%s", sid, action)
        time.sleep(0.05)


def main() -> int:
    args = _parse_args()
    cfg = load_pipeline_config(args.config)
    if args.instruction:
        cfg.instruction = args.instruction
    if args.slow_disabled:
        cfg.slow.enabled = False

    server = HavanaPolicyServer(cfg)
    server.start()
    try:
        if args.serve:
            rpc = PolicyRpcServer(server, args.serve, authkey=args.authkey.encode())
            rpc.serve_forever()
        elif args.smoke_steps > 0:
            _smoke_loop(server, args.smoke_steps)
        else:
            LOG.info("server is up — pass --serve <addr> to expose it over RPC, "
                     "or --smoke-steps N for a synthetic test")
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
