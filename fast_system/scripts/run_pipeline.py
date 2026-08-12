#!/usr/bin/env python3
"""Havana Policy Server — async slow + fast policy for online VLN closed-loop eval.

Pipeline (one server, two threads)
    Thread A — Fast Policy Loop (main thread of the simulator step loop):
        observation_t           -> ring buffer
        observation_t + plan_v  -> PlanEncoder + Stage-2 (CFM) -> trajectory
        trajectory              -> TrajectoryActionAdapter -> Habitat action
    Thread B — Slow Replan Worker (background daemon, queue depth = 1):
        recent RGB frames + instruction -> Cosmos Reason1 plan (atomic update)

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

The slow system can be disabled for RGB-D-only debugging. In that mode Stage-2
uses a zero plan with ``input_goal_mask=1``.
"""
from __future__ import annotations

# stdlib --------------------------------------------------------------------- #
import argparse
import collections
import logging
import math
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
    make_autocast,
    sample_traj,
)
from fast_system.model.nav_model import build_fast_nav_model  # noqa: E402
from fast_system.model.plan_encoder import (  # noqa: E402
    PLAN_SCHEMA_VERSION,
    build_plan_encoder,
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
    """Runtime knobs for the jointly-trained PlanEncoder + Stage-2 policy."""

    config_path: str = str(FAST_SYSTEM_ROOT / "configs" / "plan_stage2.yaml")
    checkpoint: str = str(
        FAST_SYSTEM_ROOT / "checkpoints" / "plan_stage2_single_rgbd_time_strat3" / "last.pt"
    )
    cfm_steps: int = 8
    num_trajectory_samples: int = 1
    use_amp: bool = True
    # One step_id increment represents one incoming simulator observation.
    # This converts plan age from observation ticks to seconds.
    observation_fps: float = 30.0


def plan_time_seconds_from_steps(
    step_id: int,
    source_step: int,
    observation_fps: float,
    plan_horizon_seconds: float,
) -> float:
    """Convert simulator observation ticks to bounded plan-relative seconds."""

    observation_fps = float(observation_fps)
    plan_horizon_seconds = float(plan_horizon_seconds)
    if not math.isfinite(observation_fps) or observation_fps <= 0:
        raise ValueError("observation_fps must be positive.")
    if not math.isfinite(plan_horizon_seconds) or plan_horizon_seconds <= 0:
        raise ValueError("plan_horizon_seconds must be positive.")
    elapsed_steps = max(0, int(step_id) - int(source_step))
    return min(plan_horizon_seconds, float(elapsed_steps) / observation_fps)


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
    """Debug visualisation for RGB and fast trajectories."""

    enabled: bool = False
    output_dir: str = "/tmp/havana_debug"
    fps: float = 4.0
    queue_size: int = 32
    save_slow_video: bool = False  # deprecated compatibility option


@dataclass
class PipelineConfig:
    """Top-level runtime config for the Havana policy server."""

    instruction: str = "Walk forward and explore the room."
    obs_history_len: int = 32
    slow_replan_interval: int = 12      # trigger Reason1 every N fast steps
    slow_min_obs: int = 5               # five fixed conditional RGB frames
    slow_input_frames: int = 5
    target_fps: float = 10.0
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
    depth_m: np.ndarray                     # (H, W) float32 meters
    timestamp: float = 0.0


class ObservationRingBuffer:
    """Thread-safe ring buffer of recent RGB frames."""

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

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


# =========================================================================== #
# Shared latent state
# =========================================================================== #
def validate_plan_bundle(plan: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Validate the cross-process plan contract and canonicalize it on CPU."""
    if str(plan.get("schema_version", "")) != PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported plan schema {plan.get('schema_version')!r}; expected {PLAN_SCHEMA_VERSION!r}."
        )
    expected = {
        "reason_context": (None, 1024),
        "context_mask": (None,),
        "traj_latents": (3, 3584),
    }
    out: Dict[str, torch.Tensor] = {}
    for name, shape in expected.items():
        value = plan.get(name)
        if not torch.is_tensor(value):
            raise TypeError(f"Plan field {name!r} must be a tensor, got {type(value).__name__}.")
        actual = tuple(value.shape)
        valid = (
            len(actual) == len(shape)
            and all(want is None or got == want for got, want in zip(actual, shape))
        )
        if not valid:
            raise ValueError(f"Plan field {name!r} has shape {actual}; expected {shape}.")
        out[name] = value.detach().cpu().contiguous()
    if out["reason_context"].shape[0] != out["context_mask"].shape[0]:
        raise ValueError("Plan context and context_mask lengths differ.")
    out["context_mask"] = out["context_mask"].bool()
    if not out["context_mask"].any():
        raise ValueError("Plan has no valid Reason1 context tokens.")
    out["reason_context"] = out["reason_context"].half()
    out["traj_latents"] = out["traj_latents"].half()
    if not all(torch.isfinite(value).all() for name, value in out.items() if name != "context_mask"):
        raise ValueError("Plan contains non-finite tensor values.")
    return out


@dataclass(frozen=True)
class PlanBundleInfo:
    reason_context: torch.Tensor  # [L,1024] fp16 CPU
    context_mask: torch.Tensor    # [L] bool CPU
    traj_latents: torch.Tensor    # [3,3584] fp16 CPU
    version: int               # monotonic, 0 = none
    source_step: int           # fast-loop step that triggered the slow replan
    timestamp: float
    elapsed: float = 0.0
    schema_version: str = PLAN_SCHEMA_VERSION
    instruction: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlanBundleState:
    """Atomic single-writer / many-reader exchange for a complete plan."""

    def __init__(self):
        self._info: Optional[PlanBundleInfo] = None
        self._lock = threading.Lock()
        self._counter = 0

    def update(
        self,
        plan: Dict[str, Any],
        source_step: int,
    ) -> int:
        tensors = validate_plan_bundle(plan)
        with self._lock:
            self._counter += 1
            self._info = PlanBundleInfo(
                reason_context=tensors["reason_context"].clone(),
                context_mask=tensors["context_mask"].clone(),
                traj_latents=tensors["traj_latents"].clone(),
                version=self._counter,
                source_step=source_step,
                timestamp=time.time(),
                elapsed=float(plan.get("elapsed", 0.0)),
                schema_version=str(plan.get("schema_version", PLAN_SCHEMA_VERSION)),
                instruction=str(plan.get("instruction", "")),
                metadata=dict(plan.get("metadata", {})),
            )
            return self._counter

    def get(self) -> Optional[PlanBundleInfo]:
        with self._lock:
            return self._info

    def clear(self) -> None:
        """Drop the current latent while keeping version ids monotonic."""
        with self._lock:
            self._info = None


# =========================================================================== #
# Trajectory-to-action adapter
# =========================================================================== #
class TrajectoryActionAdapter:
    """Convert a Stage-2 trajectory (body-frame meters) to a Habitat discrete action.

    Convention (matches `render_bev` / dataset `rotate_to_local`):
        x is forward, y is left.

    Logic (intentionally minimal — extend later):
        1. Pick a near-future waypoint (`lookahead_index`).
        2. heading_error comes from the predicted yaw channel.
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
        """trajectory_m: [H,3] as (x forward, y left, yaw)."""
        if trajectory_m is None or trajectory_m.size == 0:
            return self.stop, "empty trajectory -> stop"

        H = trajectory_m.shape[0]
        idx = max(0, min(int(self.cfg.lookahead_index), H - 1))
        wp = trajectory_m[idx]
        x_fwd = float(wp[0])
        y_left = float(wp[1])
        heading = (
            float((float(wp[2]) + math.pi) % (2.0 * math.pi) - math.pi)
            if wp.shape[0] >= 3
            else (math.atan2(y_left, x_fwd) if (x_fwd != 0.0 or y_left != 0.0) else 0.0)
        )
        dist = math.hypot(x_fwd, y_left)

        # Decide
        if abs(heading) > self.cfg.turn_threshold_rad:
            if heading > 0 and self.left is not None:
                action = self.left
                reason = "heading_error %.2frad > %.2f -> turn_left" % (heading, self.cfg.turn_threshold_rad)
            elif heading <= 0 and self.right is not None:
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


def _coerce_depth_m(depth: np.ndarray, min_depth_m: float = 0.1, max_depth_m: float = 5.0) -> np.ndarray:
    arr = np.asarray(depth)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected depth with shape (H,W) or (H,W,1), got {arr.shape}.")
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32) / 10000.0
    else:
        arr = arr.astype(np.float32, copy=False)
    valid = np.isfinite(arr) & (arr >= float(min_depth_m)) & (arr <= float(max_depth_m))
    return np.ascontiguousarray(np.where(valid, arr, 0.0), dtype=np.float32)


def to_depth_float(depth_m: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    image = Image.fromarray(np.asarray(depth_m, dtype=np.float32), mode="F")
    if image.size != size:
        image = image.resize(size, getattr(Image, "Resampling", Image).NEAREST)
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).contiguous()


# =========================================================================== #
# Slow system backends
# =========================================================================== #
class SlowBackend:
    """Abstract Reason1 plan backend."""

    def is_real(self) -> bool:
        return False

    def generate_plan(
        self,
        recent_frames_u8: List[np.ndarray],
        instruction: str,
        seed: int,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class StubSlowBackend(SlowBackend):
    """Disabled backend; the fast runner will use its masked RGB-D-only path."""

    def generate_plan(self, recent_frames_u8, instruction, seed) -> Dict[str, Any]:
        raise RuntimeError("Reason1 plan backend is disabled")


class RemoteSlowBackend(SlowBackend):
    """Request CPU plan bundles without importing Cosmos in the fast process."""

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
    def generate_plan(self, recent_frames_u8, instruction, seed) -> Dict[str, Any]:
        if not self._connected:
            raise RuntimeError("slow RPC not connected")
        # `multiprocessing.connection` is not safe under concurrent send/recv,
        # but in practice only the SlowReplanWorker calls us. The lock is here
        # purely to make any future call site safe.
        with self._lock:
            try:
                self._conn.send(("generate_plan", list(recent_frames_u8), str(instruction), int(seed)))
                resp = self._conn.recv()
            except (EOFError, ConnectionResetError, BrokenPipeError) as exc:
                LOG.error("[Slow][RPC] connection lost (%s); reconnecting", exc)
                self._reconnect()
                self._conn.send(("generate_plan", list(recent_frames_u8), str(instruction), int(seed)))
                resp = self._conn.recv()

        if not isinstance(resp, dict) or not resp.get("ok"):
            err = resp.get("error") if isinstance(resp, dict) else repr(resp)
            raise RuntimeError(f"slow RPC error: {err}")
        result = dict(resp["result"])
        tensors = validate_plan_bundle(result)
        result.update(tensors)
        LOG.info(
            "[Slow][RPC] received plan context=%d trajectory_tokens=%d elapsed=%.2fs",
            int(tensors["context_mask"].sum()),
            int(tensors["traj_latents"].shape[0]),
            float(result.get("elapsed", 0.0)),
        )
        return result

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
    """Background worker: turns recent RGB + instruction into a plan bundle.

    Only ever holds a single pending request — older requests are dropped when
    a newer one arrives (so the slow system never falls behind).
    """

    def __init__(
        self,
        *,
        slow: SlowBackend,
        shared_plan: PlanBundleState,
        instruction_holder: "_InstructionHolder",
        seed: int,
    ):
        super().__init__(daemon=True, name="SlowReplanWorker")
        self._slow = slow
        self._shared = shared_plan
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

    def discard_pending_and_wait(self, timeout: float) -> None:
        """Drain work from the previous episode before policy state is reset."""
        with self._lock:
            self._pending = None
            self._has_request.clear()

        deadline = time.time() + max(0.0, float(timeout))
        while self._is_busy.is_set():
            if time.time() >= deadline:
                raise TimeoutError(
                    "timed out waiting for the slow replan worker to finish "
                    "the previous episode"
                )
            time.sleep(0.05)

        # No get_action call runs concurrently with benchmark reset, but clear
        # once more in case a request was queued immediately before reset.
        with self._lock:
            self._pending = None
            self._has_request.clear()

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
                if req is not None:
                    # Set this while holding _lock so reset cannot observe the
                    # request as neither pending nor busy.
                    self._is_busy.set()
            if req is None:
                continue
            try:
                self._process(*req)
            except Exception as exc:  # noqa: BLE001
                LOG.error("[SlowWorker] replan failed: %s\n%s", exc, traceback.format_exc())
                # A failed or invalid RPC result must not leave a stale plan
                # looking current; the fast loop immediately takes its
                # zero-plan, input_goal_mask=1 RGB-D-only path.
                self._shared.clear()
            finally:
                self._is_busy.clear()
        LOG.info("[SlowWorker] thread stopped")

    @torch.no_grad()
    def _process(self, step_id: int, frames: List[np.ndarray]) -> None:
        instruction = self._instruction_holder.get()
        LOG.info("[SlowWorker] start replan step=%d frames=%d instruction='%s'",
                 step_id, len(frames), instruction[:120])
        t0 = time.time()

        plan = self._slow.generate_plan(frames, instruction, seed=self._seed)
        slow_secs = time.time() - t0
        # A depth-one queue can receive a fresher request while RPC is running.
        # Never publish the now-stale result in that case.
        with self._lock:
            newer_step = self._pending[0] if self._pending is not None else None
        if newer_step is not None and newer_step > step_id:
            LOG.info("[SlowWorker] discard stale result step=%d; pending step=%d", step_id, newer_step)
            return
        plan["instruction"] = instruction
        plan["metadata"] = {
            **dict(plan.get("metadata", {})),
            "source_step": int(step_id),
            "source_timestamp": float(t0),
        }
        version = self._shared.update(plan, source_step=step_id)
        LOG.info(
            "[SlowWorker] published step=%d version=%d context=%d trajectory_tokens=%d slow_time=%.2fs",
            step_id,
            version,
            int(plan["context_mask"].sum()),
            int(plan["traj_latents"].shape[0]),
            slow_secs,
        )


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
    trajectory_m: np.ndarray                     # (H, 2) Stage-2 xy prediction
    action: Optional[str]
    z_version: int
    context_length: int = 0
    plan_age: float = 0.0
    slow_latency: float = 0.0
    fast_latency: float = 0.0
    goal_xy: Optional[np.ndarray] = None         # final Reason1 waypoint
    anchors_xy: Optional[np.ndarray] = None      # all Reason1 waypoints
    turn_probs: Optional[np.ndarray] = None      # deprecated
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

        traj = snap.trajectory_m[:, :2]
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
        title = (
            f"step {snap.step_id} plan_v{snap.z_version} L={snap.context_length} "
            f"age={snap.plan_age:.1f}s slow={snap.slow_latency:.2f}s "
            f"fast={snap.fast_latency:.3f}s -> {snap.action or '·'}"
        )
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

        panels = [self._resize_to_h(obs, side_px), self._resize_to_h(bev, side_px)]
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
def load_plan_policy(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """Load a v2 token-plan checkpoint; legacy policy checkpoints are rejected."""
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise TypeError("Plan policy checkpoint must be a mapping.")
    schema = checkpoint.get("plan_schema_version")
    if schema != PLAN_SCHEMA_VERSION:
        raise ValueError(
            f"Checkpoint schema {schema!r} is incompatible; expected {PLAN_SCHEMA_VERSION!r}. "
            "Legacy Stage-1/Stage-2 checkpoints cannot be loaded."
        )
    if "model" not in checkpoint or "plan_encoder" not in checkpoint:
        raise KeyError("Plan policy checkpoint requires both 'model' and 'plan_encoder' state dictionaries.")
    stage2 = build_fast_nav_model(cfg.get("model", {}))
    plan_encoder = build_plan_encoder(cfg.get("plan_encoder", {}))
    stage2.load_state_dict(checkpoint["model"], strict=True)
    plan_encoder.load_state_dict(checkpoint["plan_encoder"], strict=True)
    stage2.to(device).eval()
    plan_encoder.to(device).eval()
    for module in (stage2, plan_encoder):
        for parameter in module.parameters():
            parameter.requires_grad = False
    return stage2, plan_encoder


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
        plan_encoder: torch.nn.Module,
        cfg_fast: FastSystemConfig,
        stage2_yaml_cfg: Dict[str, Any],
        device: torch.device,
        use_amp: bool,
    ):
        self.stage2 = stage2
        self.plan_encoder = plan_encoder
        self.cfg = cfg_fast
        self.device = device
        self.use_amp = use_amp

        data_cfg = stage2_yaml_cfg.get("intern_data", stage2_yaml_cfg.get("scand_data", {}))
        self.obs_image_size: int = int(data_cfg.get("obs_image_size", 256))
        self.context_size: int = int(
            data_cfg.get("context_size", stage2_yaml_cfg.get("model", {}).get("context_size", 0))
        )
        self.context_dt_frames: int = int(data_cfg.get("obs_context_dt_frames", 1))
        self.action_horizon: int = int(data_cfg.get("action_horizon", 16))
        self.action_dim: int = int(stage2_yaml_cfg.get("model", {}).get("action_dim", 3))
        self.waypoint_spacing: float = float(data_cfg.get("metric_waypoint_spacing", 0.38))
        self.normalize_actions: bool = bool(data_cfg.get("normalize_actions", True))
        self.pred_to_meter: float = self.waypoint_spacing if self.normalize_actions else 1.0
        self.action_stats = action_stats_from_cfg(stage2_yaml_cfg)
        self.observation_fps = float(cfg_fast.observation_fps)
        if not math.isfinite(self.observation_fps) or self.observation_fps <= 0:
            raise ValueError("fast.observation_fps must be positive.")
        plan_cfg = stage2_yaml_cfg.get("plan_encoder", {})
        anchor_times = tuple(float(value) for value in plan_cfg.get("trajectory_anchor_times", (0.4, 0.8, 1.2)))
        if not anchor_times or any(value <= 0 for value in anchor_times):
            raise ValueError("plan_encoder.trajectory_anchor_times must contain positive seconds.")
        self.plan_horizon_seconds = max(anchor_times)

        self.plan_dim: int = int(stage2_yaml_cfg.get("model", {}).get("plan_dim", 384))
        self.num_plan_tokens: int = int(getattr(plan_encoder, "num_plan_tokens", 19))
        self.zero_plan = torch.zeros(self.num_plan_tokens, self.plan_dim, device=self.device)
        self.last_latency = 0.0

    def _build_obs_context_chw(self, recent: List[StoredFrame]) -> tuple[torch.Tensor, torch.Tensor]:
        """Build RGB context and the latest aligned metric-depth tensor."""
        size = (self.obs_image_size, self.obs_image_size)
        if not recent:
            raise ValueError("FastNav requires at least one RGB-D observation.")

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
        depth = to_depth_float(recent[-1].depth_m, size)
        return stack.unsqueeze(0).to(self.device), depth.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict_trajectory(
        self,
        recent: List[StoredFrame],
        plan: Optional[PlanBundleInfo],
        *,
        step_id: int,
        latent_version: int,
    ) -> np.ndarray:
        """Return ``(x meters, y meters, yaw radians)`` with shape ``[H,3]``."""
        t0 = time.time()
        obs_batch, depth_batch = self._build_obs_context_chw(recent)
        if plan is None:
            z_used = self.zero_plan.unsqueeze(0)
            mask = torch.ones(1, device=self.device, dtype=torch.long)
            mode = "obs-only (input_goal_mask=1)"
        else:
            plan_time_seconds = plan_time_seconds_from_steps(
                step_id,
                plan.source_step,
                self.observation_fps,
                self.plan_horizon_seconds,
            )
            with make_autocast(self.device, self.use_amp):
                z_used = self.plan_encoder(
                    plan.reason_context.unsqueeze(0),
                    plan.context_mask.unsqueeze(0),
                    plan.traj_latents.unsqueeze(0),
                    torch.tensor([plan_time_seconds], device=self.device),
                )
            mask = None
            mode = (
                f"plan v{latent_version} plan_time={plan_time_seconds:.3f}s "
                f"age={max(0.0, time.time() - plan.timestamp):.2f}s"
            )

        traj = sample_traj(
            self.stage2, obs_batch, z_used,
            depth=depth_batch,
            horizon=self.action_horizon,
            action_dim=self.action_dim,
            steps=self.cfg.cfm_steps,
            action_stats=self.action_stats,
            input_goal_mask=mask,
            num_samples=max(1, int(self.cfg.num_trajectory_samples)),
            device=self.device,
            amp=self.use_amp,
        )
        traj_m = traj.detach().cpu().numpy()[0]
        traj_m[:, :2] *= self.pred_to_meter
        elapsed = time.time() - t0
        self.last_latency = elapsed
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
        LOG.info("  fast.checkpoint: %s", cfg.fast.checkpoint)
        LOG.info("  slow.enabled   : %s", cfg.slow.enabled)
        if cfg.slow.enabled:
            LOG.info("  slow.remote    : %s", cfg.slow.remote_address)
            LOG.info("  slow.timeout   : %.0fs", cfg.slow.connect_timeout)
        LOG.info("  instruction    : %s", cfg.instruction)
        LOG.info("=" * 72)

        # ----- Fast system load --------------------------------------------- #
        t_fast = time.time()
        stage2_cfg = load_stage2_config(Path(cfg.fast.config_path))
        self._stage2, self._plan_encoder = load_plan_policy(
            Path(cfg.fast.checkpoint), stage2_cfg, self._device
        )
        self._stage2_cfg = stage2_cfg
        LOG.info("[Init] fast system loaded in %.2fs", time.time() - t_fast)

        # ----- Fast runner -------------------------------------------------- #
        self._fast_runner = FastPolicyRunner(
            stage2=self._stage2,
            plan_encoder=self._plan_encoder,
            cfg_fast=cfg.fast,
            stage2_yaml_cfg=stage2_cfg,
            device=self._device,
            use_amp=self._use_amp,
        )

        # ----- Observation buffer / shared state --------------------------- #
        self._obs_buf = ObservationRingBuffer(cfg.obs_history_len)
        self._shared_plan = PlanBundleState()
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
                slow_backend = StubSlowBackend()
        else:
            LOG.info("[Init] slow backend disabled — using stub (zero z_g via masked path)")
            slow_backend = StubSlowBackend()
        self._slow_backend = slow_backend

        # ----- Slow worker (started in start()) ----------------------------- #
        self._slow_worker = SlowReplanWorker(
            slow=self._slow_backend,
            shared_plan=self._shared_plan,
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
        # Tracks when the first plan has been published. Until then we always
        # try to trigger a replan; after that we only replan on the configured
        # interval (so we know the scene has had time to actually change).
        self._first_plan_seen = False
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

    def reset_episode(self, instruction: str = "") -> None:
        """Clear recurrent policy state before starting a new eval episode."""
        if self._started:
            timeout = max(30.0, float(self.cfg.slow.connect_timeout) * 2.0)
            self._slow_worker.discard_pending_and_wait(timeout=timeout)
        self._obs_buf.clear()
        self._shared_plan.clear()
        self._instruction.set(str(instruction))
        self._step_count = 0
        self._first_plan_seen = False
        self._last_visualised_version = 0
        LOG.info("[Server] episode reset; instruction='%s'", instruction)

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
            raise KeyError(f"Observation has no rgb_sensor/rgb key; keys={list(obs.keys())}.")
        depth = obs.get("depth_sensor", obs.get("depth"))
        if depth is None:
            raise KeyError(
                f"FastNav RGB-D policy requires depth_sensor/depth; keys={list(obs.keys())}."
            )
        rgb_u8 = _strip_alpha_to_rgb_u8(rgb)
        depth_m = _coerce_depth_m(depth)
        frame = StoredFrame(step_id=step_id, rgb=rgb_u8, depth_m=depth_m, timestamp=time.time())
        self._obs_buf.push(frame)

        if step_id % 50 == 0:
            LOG.info("[Server] step=%d obs_keys=%s rgb_shape=%s depth_shape=%s buffer_len=%d",
                     step_id, list(obs.keys()), rgb_u8.shape, depth_m.shape, len(self._obs_buf))

        # Decide whether to wake the slow worker:
        #   - Until the first plan lands, submit as soon as enough RGB history
        #     is available and the worker is idle.
        #   - After the first plan, only replan on the configured cadence so
        #     that consecutive replans see meaningfully different scenes.
        have_history = len(self._obs_buf) >= max(1, int(self.cfg.slow_min_obs))
        interval = max(1, int(self.cfg.slow_replan_interval))
        self._first_plan_seen = self._shared_plan.get() is not None
        should_request = (
            (not self._first_plan_seen and not self._slow_worker.busy)
            or (self._first_plan_seen and step_id % interval == 0)
        )
        if self._slow_backend.is_real() and have_history and should_request:
            recent_frames_u8 = [f.rgb for f in self._obs_buf.latest_n(self.cfg.slow_input_frames)]
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

        plan = self._shared_plan.get()
        if plan is None:
            busy = "busy" if self._slow_worker.busy else "idle"
            LOG.info("[FastLoop] step=%d plan unavailable (slow worker %s); using RGB-D-only fallback", step_id, busy)
        version_arg = plan.version if plan is not None else 0

        try:
            traj_m = self._fast_runner.predict_trajectory(
                recent, plan=plan, step_id=step_id, latent_version=version_arg,
            )
        except Exception as exc:
            LOG.error("[FastLoop] inference failed: %s\n%s", exc, traceback.format_exc())
            return None

        try:
            action, _ = self._adapter.convert(traj_m, step_id=step_id)
        except Exception as exc:
            LOG.error("[Adapter] conversion failed: %s\n%s", exc, traceback.format_exc())
            return None

        self._maybe_publish_debug(recent[-1], traj_m, action, plan, step_id)

        sleep_s = float(getattr(self.cfg, "fast_sleep_time", 0.0))
        if sleep_s > 0.0:
            time.sleep(sleep_s)
        return action

    def _maybe_publish_debug(
        self,
        last_frame: StoredFrame,
        traj_m: np.ndarray,
        action: Optional[str],
        latent: Optional[PlanBundleInfo],
        step_id: int,
    ) -> None:
        if self._visualizer is None:
            return
        snap = DebugSnapshot(
            step_id=step_id,
            obs_rgb_u8=last_frame.rgb,
            trajectory_m=np.asarray(traj_m[:, :2], dtype=np.float32),
            action=action,
            z_version=latent.version if latent is not None else 0,
            context_length=int(latent.context_mask.sum()) if latent is not None else 0,
            plan_age=max(0.0, time.time() - latent.timestamp) if latent is not None else 0.0,
            slow_latency=latent.elapsed if latent is not None else 0.0,
            fast_latency=self._fast_runner.last_latency,
            goal_xy=None,
            anchors_xy=None,
            slow_video_u8=None,
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

    _OPS = {
        "ping",
        "attach_action_space",
        "set_instruction",
        "reset_episode",
        "get_action",
    }

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
        if op == "reset_episode":
            self.server.reset_episode(args[0] if args else "")
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
            "depth_sensor": rng.uniform(0.1, 5.0, size=(480, 640)).astype(np.float32),
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
