#!/usr/bin/env python3
"""Lightweight Habitat test and R2R benchmark loops.

Goals:
    1. Load Habitat-Sim with a Matterport3D scene.
    2. Build RGB / Depth sensors and the agent action space from
       InternNav's `vln_r2r.yaml` (so sensors / actions stay in one place).
    3. Step the simulator in a loop, render observations, and let the user
       drive the agent from the keyboard.
    4. Keep the action source behind an `ActionProvider` interface so a
       policy can be plugged in later without touching the loop.
    5. Evaluate R2R episodes directly from their JSON/JSON.GZ dataset without
       importing InternNav's Evaluator framework.

Config priority (matches the request):
    * Sensor & action definitions       -> InternNav vln_r2r.yaml
    * Scene + debug-only parameters     -> havana/scripts/eval/config/test.yaml

Run (test mode, GUI + keyboard):

    python /home/wangdzh/ffnav/havana/scripts/eval/eval_fast.py \\
        --mode test --display --keyboard

Run (R2R val_unseen, remote Havana policy):

    python /home/wangdzh/ffnav/havana/scripts/eval/eval_fast.py \\
        --mode benchmark --policy-remote /tmp/havana.sock \\
        --split val_unseen --max-steps 500
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("eval_fast")


# --------------------------------------------------------------------------- #
# Config loading
# --------------------------------------------------------------------------- #

@dataclass
class SensorSpec:
    name: str
    sensor_type: str        # "color" | "depth"
    width: int
    height: int
    hfov: float
    position: List[float]
    min_depth: float = 0.0
    max_depth: float = 10.0


@dataclass
class ResolvedConfig:
    # Sim
    scene_path: Path
    forward_step_size: float
    turn_angle_deg: float
    tilt_angle_deg: float
    gpu_device_id: int
    agent_height: float
    agent_radius: float

    # Sensors / actions sourced from vln_r2r.yaml.
    sensors: List[SensorSpec]
    action_names: List[str]                 # ordered, e.g. ["stop", "move_forward", ...]

    # Optional starting pose (test debugging).
    start_position: Optional[List[float]] = None
    start_rotation: Optional[List[float]] = None  # [x, y, z, w]

    # Test-only debug bits.
    keyboard_map: Dict[str, str] = field(default_factory=dict)
    scene_id: str = ""
    episode_id: Optional[int] = None


@dataclass
class BenchmarkSpec:
    """Dataset-level settings resolved from InternNav's vln_r2r.yaml."""

    data_path: Path
    scenes_dir: Path
    split: str
    success_distance: float
    episodes: List[Dict[str, Any]]


def _read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _resolve_vln_config_path(
    test_config_path: Path, test_cfg: Dict[str, Any]
) -> Path:
    raw_path = test_cfg.get("vln_r2r_config")
    if not raw_path:
        raise ValueError(
            f"Missing vln_r2r_config in test config: {test_config_path}"
        )
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = (test_config_path.parent / path).resolve()
    return path


def _resolve_relative_dataset_path(path: Path, vln_config_path: Path) -> Path:
    """Resolve a Habitat dataset path without assuming the caller's cwd."""
    path = path.expanduser()
    if path.is_absolute():
        return path

    candidates = [(Path.cwd() / path).resolve()]
    if len(vln_config_path.parents) > 3:
        # scripts/eval/configs/vln_r2r.yaml -> InternNav repository root.
        candidates.append((vln_config_path.parents[3] / path).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _read_r2r_dataset(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"R2R dataset not found: {path}")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("episodes"), list):
        raise ValueError(f"R2R dataset has no episodes list: {path}")
    return data


def load_benchmark_spec(
    test_config_path: Path,
    split_override: Optional[str] = None,
    data_path_override: Optional[str] = None,
) -> BenchmarkSpec:
    """Load R2R episodes and benchmark settings directly from vln_r2r.yaml."""
    test_cfg = _read_yaml(test_config_path)
    vln_path = _resolve_vln_config_path(test_config_path, test_cfg)
    vln_cfg = _read_yaml(vln_path)
    habitat_cfg = vln_cfg.get("habitat", {}) or {}
    dataset_cfg = habitat_cfg.get("dataset", {}) or {}

    split = str(split_override or dataset_cfg.get("split") or "val_unseen")
    data_template = data_path_override or dataset_cfg.get("data_path")
    if not data_template:
        raise ValueError(
            f"Missing habitat.dataset.data_path in {vln_path}; "
            "pass --data-path to override it"
        )
    try:
        formatted_data_path = str(data_template).format(split=split)
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Invalid R2R data_path template {data_template!r}: {exc}"
        ) from exc
    data_path = _resolve_relative_dataset_path(Path(formatted_data_path), vln_path)

    scenes_raw = dataset_cfg.get("scenes_dir") or test_cfg.get("scenes_dir")
    if not scenes_raw:
        raise ValueError(
            f"No scenes_dir in {vln_path} or {test_config_path}"
        )
    scenes_dir = _resolve_relative_dataset_path(Path(scenes_raw), vln_path)
    if not scenes_dir.exists():
        raise FileNotFoundError(f"MP3D scenes_dir not found: {scenes_dir}")

    measurements = ((habitat_cfg.get("task", {}) or {}).get(
        "measurements", {}
    ) or {})
    success_cfg = measurements.get("success", {}) or {}
    success_distance = float(success_cfg.get("success_distance", 3.0))

    data = _read_r2r_dataset(data_path)
    return BenchmarkSpec(
        data_path=data_path,
        scenes_dir=scenes_dir,
        split=split,
        success_distance=success_distance,
        episodes=list(data["episodes"]),
    )


def _resolve_scene(scenes_dir: Path, scene_id: str) -> Path:
    candidate = scenes_dir / scene_id / f"{scene_id}.glb"
    if not candidate.exists():
        raise FileNotFoundError(
            f"Scene not found: expected {candidate}. "
            f"Check scenes_dir={scenes_dir} and scene_id={scene_id}."
        )
    return candidate


def _lookup_episode(
    episodes_path: Optional[Path], episode_id: Optional[int], scene_id: str
):
    """Best-effort episode lookup; returns (start_position, start_rotation, found_id)."""
    if episodes_path is None or episode_id is None:
        return None, None, None
    if not episodes_path.exists():
        log.warning("episodes_path %s missing — ignoring --episode-id", episodes_path)
        return None, None, None
    try:
        with episodes_path.open("r") as f:
            data = json.load(f)
        for ep in data.get("episodes", []):
            if ep.get("episode_id") == episode_id:
                if scene_id and scene_id not in ep.get("scene_id", ""):
                    log.warning(
                        "episode_id=%s belongs to scene %s, not %s — using its pose anyway",
                        episode_id, ep.get("scene_id"), scene_id,
                    )
                return ep.get("start_position"), ep.get("start_rotation"), episode_id
        log.warning("episode_id=%s not found in %s", episode_id, episodes_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("failed to read episodes %s: %s", episodes_path, exc)
    return None, None, None


def load_config(
    mode: str,
    test_config_path: Path,
    cli_scene_id: Optional[str],
    cli_episode_id: Optional[int],
) -> ResolvedConfig:
    """Build a `ResolvedConfig` from vln_r2r.yaml + test.yaml + CLI overrides."""
    if mode not in {"test", "benchmark"}:
        raise ValueError(f"Unknown mode: {mode}")

    test_cfg = _read_yaml(test_config_path)

    vln_path = _resolve_vln_config_path(test_config_path, test_cfg)
    vln_cfg = _read_yaml(vln_path)

    sim_cfg = (vln_cfg.get("habitat", {}) or {}).get("simulator", {}) or {}
    agents_cfg = sim_cfg.get("agents", {}) or {}
    main_agent = agents_cfg.get("main_agent", {}) or {}
    sim_sensors_cfg = main_agent.get("sim_sensors", {}) or {}
    if not sim_sensors_cfg:
        raise ValueError(
            f"No sim_sensors found under habitat.simulator.agents.main_agent in {vln_path}"
        )

    # --- Sensors --------------------------------------------------------- #
    # Defaults match habitat-lab's HabitatSim*SensorConfig (see
    # default_structured_configs.py) so we only need to pull the overrides
    # the user wrote into vln_r2r.yaml.
    sensor_defaults = {
        "rgb_sensor": dict(
            width=640, height=480, hfov=90,
            position=[0.0, 1.25, 0.0], sensor_type="color",
            min_depth=0.0, max_depth=10.0,
        ),
        "depth_sensor": dict(
            width=640, height=480, hfov=90,
            position=[0.0, 1.25, 0.0], sensor_type="depth",
            min_depth=0.0, max_depth=10.0,
        ),
    }
    sensors: List[SensorSpec] = []
    for s_name, s_overrides in sim_sensors_cfg.items():
        if s_name not in sensor_defaults:
            log.warning("sensor %s not recognised — skipping", s_name)
            continue
        merged = dict(sensor_defaults[s_name])
        merged.update(s_overrides or {})
        sensors.append(
            SensorSpec(
                name=s_name,
                sensor_type=merged["sensor_type"],
                width=int(merged["width"]),
                height=int(merged["height"]),
                hfov=float(merged["hfov"]),
                position=list(merged.get("position", [0.0, 1.25, 0.0])),
                min_depth=float(merged.get("min_depth", 0.0)),
                max_depth=float(merged.get("max_depth", 10.0)),
            )
        )
    if not sensors:
        raise ValueError("No usable sensors after parsing vln_r2r.yaml")

    # --- Actions --------------------------------------------------------- #
    actions_cfg = (vln_cfg.get("habitat", {}) or {}).get("task", {}) or {}
    actions_cfg = actions_cfg.get("actions", {}) or {}
    if not actions_cfg:
        raise ValueError(f"No task.actions found in {vln_path}")
    action_names = list(actions_cfg.keys())

    # --- Scene + pose ---------------------------------------------------- #
    scenes_dir = Path(test_cfg.get("scenes_dir", "")).expanduser()
    if not scenes_dir.exists():
        raise FileNotFoundError(f"scenes_dir not found: {scenes_dir}")
    scene_id = cli_scene_id or test_cfg.get("default_scene_id")
    if not scene_id:
        raise ValueError(
            "No scene specified: pass --scene-id or set default_scene_id in test.yaml"
        )
    scene_path = _resolve_scene(scenes_dir, scene_id)

    start_position = test_cfg.get("start_position")
    start_rotation = test_cfg.get("start_rotation")
    episodes_path = test_cfg.get("episodes_path")
    episodes_path = Path(episodes_path).expanduser() if episodes_path else None
    ep_pos, ep_rot, ep_found = _lookup_episode(episodes_path, cli_episode_id, scene_id)
    if ep_pos is not None and start_position is None:
        start_position = ep_pos
    if ep_rot is not None and start_rotation is None:
        start_rotation = ep_rot
    if cli_episode_id is not None and ep_found is None and start_position is None:
        log.info(
            "--episode-id=%s could not be resolved; falling back to scene's default pose",
            cli_episode_id,
        )

    keyboard_map = test_cfg.get("keyboard") or {}

    return ResolvedConfig(
        scene_path=scene_path,
        forward_step_size=float(sim_cfg.get("forward_step_size", 0.01)),
        turn_angle_deg=float(sim_cfg.get("turn_angle", 15)),
        tilt_angle_deg=float(sim_cfg.get("tilt_angle", 15)),
        gpu_device_id=int(
            (sim_cfg.get("habitat_sim_v0") or {}).get("gpu_device_id", 0)
        ),
        agent_height=1.5,
        agent_radius=0.1,
        sensors=sensors,
        action_names=action_names,
        start_position=start_position,
        start_rotation=start_rotation,
        keyboard_map={k: str(v) for k, v in keyboard_map.items()},
        scene_id=scene_id,
        episode_id=ep_found,
    )


# --------------------------------------------------------------------------- #
# Habitat env build / reset
# --------------------------------------------------------------------------- #

def _build_sensor_specs(cfg: ResolvedConfig):
    import habitat_sim

    specs = []
    for s in cfg.sensors:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = s.name
        spec.resolution = [s.height, s.width]
        spec.position = s.position
        spec.hfov = s.hfov
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        if s.sensor_type == "color":
            spec.sensor_type = habitat_sim.SensorType.COLOR
        elif s.sensor_type == "depth":
            spec.sensor_type = habitat_sim.SensorType.DEPTH
        else:
            raise ValueError(f"Unsupported sensor type: {s.sensor_type}")
        specs.append(spec)
    return specs


def build_env(cfg: ResolvedConfig):
    """Construct the habitat-sim Simulator with sensors + action space."""
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(cfg.scene_path)
    sim_cfg.gpu_device_id = cfg.gpu_device_id
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = cfg.agent_height
    agent_cfg.radius = cfg.agent_radius
    agent_cfg.sensor_specifications = _build_sensor_specs(cfg)

    # Map vln_r2r.yaml action names -> habitat-sim ActionSpecs. `stop`
    # has no built-in actuation; we register a no-op move_forward(0.0)
    # so it's a valid (though motionless) action call.
    name_to_actuation = {
        "move_forward": ("move_forward", cfg.forward_step_size),
        "turn_left": ("turn_left", cfg.turn_angle_deg),
        "turn_right": ("turn_right", cfg.turn_angle_deg),
        "look_up": ("look_up", cfg.tilt_angle_deg),
        "look_down": ("look_down", cfg.tilt_angle_deg),
        "stop": ("move_forward", 0.0),
    }
    action_space = {}
    unknown = []
    for name in cfg.action_names:
        if name not in name_to_actuation:
            unknown.append(name)
            continue
        sim_name, amount = name_to_actuation[name]
        action_space[name] = habitat_sim.agent.ActionSpec(
            sim_name, habitat_sim.agent.ActuationSpec(amount=amount)
        )
    if unknown:
        log.warning("ignoring unsupported action names: %s", unknown)
    if not action_space:
        raise ValueError("Resolved action space is empty")
    agent_cfg.action_space = action_space

    backend = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(backend)
    return sim


def reset_env(sim, cfg: ResolvedConfig):
    """Reset and place the agent. Returns initial observation dict."""
    import habitat_sim

    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()

    if cfg.start_position is not None:
        state.position = np.array(cfg.start_position, dtype=np.float32)
    else:
        state.position = sim.pathfinder.get_random_navigable_point()

    if cfg.start_rotation is not None:
        # vln_ce stores [x, y, z, w]. quaternion.quaternion expects (w, x, y, z).
        import quaternion  # noqa: F401  (registers numpy dtype)
        x, y, z, w = cfg.start_rotation
        state.rotation = np.quaternion(w, x, y, z)

    agent.set_state(state)
    return sim.get_sensor_observations()


# --------------------------------------------------------------------------- #
# Observation handling / display
# --------------------------------------------------------------------------- #

def render_observation(obs: Dict[str, np.ndarray], window_name: str = "habitat") -> bool:
    """Show RGB + Depth side-by-side. Returns False if user pressed q/ESC."""
    import cv2

    panels = []
    if "rgb_sensor" in obs:
        rgb = obs["rgb_sensor"]
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        panels.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if "depth_sensor" in obs:
        depth = obs["depth_sensor"]
        if depth.ndim == 3:
            depth = depth[..., 0]
        d = np.clip(depth, 0.0, np.nanmax(depth) if np.isfinite(depth).any() else 1.0)
        d_max = d.max() if d.max() > 0 else 1.0
        d_vis = (d / d_max * 255.0).astype(np.uint8)
        d_vis = cv2.applyColorMap(d_vis, cv2.COLORMAP_VIRIDIS)
        panels.append(d_vis)

    if not panels:
        return True

    # Match heights for hstack.
    h = max(p.shape[0] for p in panels)
    panels = [
        cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h)) if p.shape[0] != h else p
        for p in panels
    ]
    canvas = np.concatenate(panels, axis=1)
    cv2.imshow(window_name, canvas)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def handle_observation(obs: Dict[str, np.ndarray], step_idx: int) -> None:
    """Single hook for downstream processing (policy inference, logging, ...).

    Today this only logs shapes; future hooks (policy preprocessing,
    feature extraction, ROS publishing) plug in here.
    """
    if step_idx % 50 == 0:
        shapes = {
            k: tuple(v.shape)
            for k, v in obs.items()
            if isinstance(v, np.ndarray)
        }
        log.debug("step %d obs shapes: %s", step_idx, shapes)


# --------------------------------------------------------------------------- #
# Action providers
# --------------------------------------------------------------------------- #

class BaseActionProvider:
    def begin_episode(self, instruction: str) -> None:
        """Reset policy-side episode state and install the new instruction."""

    def get_action(self, obs: Dict[str, np.ndarray]) -> Optional[str]:
        raise NotImplementedError


class NullActionProvider(BaseActionProvider):
    """Placeholder used when no driver is connected. Returns None forever."""

    def get_action(self, obs):
        return None


class RemotePolicyActionProvider(BaseActionProvider):
    """Adapt the lightweight PolicyClient to the local ActionProvider API."""

    def __init__(self, client):
        self._client = client
        self._step_id = 0

    def begin_episode(self, instruction: str) -> None:
        self._step_id = 0
        reset = getattr(self._client, "reset_episode", None)
        if callable(reset):
            try:
                reset(instruction)
                return
            except RuntimeError as exc:
                if "unknown op" not in str(exc):
                    raise
                log.warning(
                    "connected policy server predates reset_episode(); "
                    "restart it to guarantee benchmark episode isolation"
                )
        else:
            # Compatibility with an older policy server. This updates the
            # language prompt but cannot clear its observation/latent state.
            log.warning(
                "policy server client has no reset_episode(); "
                "episode history may leak across benchmark episodes"
            )
        self._client.set_instruction(instruction)

    def get_action(self, obs):
        step_id = self._step_id
        self._step_id += 1
        return self._client.get_action(obs, step_id)


class InProcessPolicyActionProvider(BaseActionProvider):
    """Small adapter around HavanaPolicyServer with explicit episode reset."""

    def __init__(self, server):
        self._server = server
        self._step_id = 0

    def begin_episode(self, instruction: str) -> None:
        self._step_id = 0
        reset = getattr(self._server, "reset_episode", None)
        if callable(reset):
            reset(instruction)
        else:
            log.warning(
                "policy server has no reset_episode(); "
                "episode history may leak across benchmark episodes"
            )
            self._server.set_instruction(instruction)

    def get_action(self, obs):
        step_id = self._step_id
        self._step_id += 1
        return self._server.get_action(obs, step_id)


class KeyboardActionProvider(BaseActionProvider):
    """Reads a single key (via OpenCV's window) and maps it to an action name.

    OpenCV's `waitKey` is the simplest cross-platform option that works while
    we're already showing a window for visualization; it avoids pulling in
    `pynput` / curses / a separate event loop.
    """

    _SPECIAL_KEYS = {"space": ord(" "), "esc": 27, "enter": 13, "tab": 9}

    def __init__(self, key_to_action: Dict[str, str], action_space: List[str]):
        self.action_space = set(action_space)
        self.key_to_action: Dict[int, str] = {}
        for action_name, key in key_to_action.items():
            if action_name in ("quit",):
                # Reserved control key; don't try to validate against actions.
                code = self._key_code(key)
                if code is not None:
                    self.key_to_action[code] = "__quit__"
                continue
            if action_name not in self.action_space:
                log.warning(
                    "keyboard config maps key %r to action %r which is not in the "
                    "resolved action space %s — skipping",
                    key, action_name, sorted(self.action_space),
                )
                continue
            code = self._key_code(key)
            if code is None:
                log.warning("could not parse key spec %r — skipping", key)
                continue
            self.key_to_action[code] = action_name

    @classmethod
    def _key_code(cls, key: str) -> Optional[int]:
        if key is None:
            return None
        key = str(key)
        if key.lower() in cls._SPECIAL_KEYS:
            return cls._SPECIAL_KEYS[key.lower()]
        if len(key) == 1:
            return ord(key)
        return None

    def announce(self) -> None:
        log.info("Keyboard controls:")
        rev = {}
        for code, action in self.key_to_action.items():
            ch = chr(code) if 32 <= code <= 126 else f"<{code}>"
            rev.setdefault(action, []).append(ch)
        for action, keys in rev.items():
            log.info("  %s -> %s", " / ".join(keys), action)

    def get_action(self, obs):
        import cv2
        key = cv2.waitKey(0) & 0xFF  # block until a key is pressed
        if key == 255:
            return None
        return self.key_to_action.get(key)


# --------------------------------------------------------------------------- #
# Main test loop
# --------------------------------------------------------------------------- #

def run_test_loop(
    sim,
    cfg: ResolvedConfig,
    action_provider: BaseActionProvider,
    display: bool,
    max_steps: int = 100000,
) -> None:
    obs = reset_env(sim, cfg)
    handle_observation(obs, 0)

    if display:
        keep = render_observation(obs)
        if not keep:
            log.info("display window closed before first action")
            return

    if isinstance(action_provider, KeyboardActionProvider):
        action_provider.announce()

    step_idx = 0
    while step_idx < max_steps:
        action = action_provider.get_action(obs)

        if action == "__quit__":
            log.info("quit requested by user")
            break
        if action == "stop":
            log.info("stop action received — ending loop")
            break
        if action is None:
            # No action this tick (NullActionProvider, or no key bound).
            if not display:
                # Without GUI and without an action source, sleeping forever is
                # not useful — exit so the script doesn't hang silently.
                log.info("no action source connected; exiting")
                break
            if not render_observation(obs):
                break
            continue

        if action not in cfg.action_names:
            log.warning("ignoring action %r not in action space", action)
            continue

        obs = sim.step(action)
        step_idx += 1
        handle_observation(obs, step_idx)

        if display and not render_observation(obs):
            break

    log.info("test loop finished after %d steps", step_idx)


def run_policy_loop(
    sim,
    cfg: ResolvedConfig,
    action_provider: "BaseActionProvider",
    display: bool,
    max_steps: int = 5000,
) -> None:
    """Closed-loop driver for the Havana policy server.

    Differences vs `run_test_loop`:
        - `None` is treated as "policy not ready yet, advance one obs tick"
          (we resample the same scene without stepping the simulator) instead
          of exiting. This lets the slow system warm up.
        - The episode terminates only on an explicit `stop`, the user hitting
          q/ESC in display mode, or hitting `max_steps`.
    """
    obs = reset_env(sim, cfg)
    handle_observation(obs, 0)
    if display and not render_observation(obs):
        log.info("display window closed before first action")
        return

    step_idx = 0
    no_op_streak = 0
    while step_idx < max_steps:
        action = action_provider.get_action(obs)

        if action == "__quit__":
            log.info("quit requested by user")
            break
        if action == "stop":
            log.info("policy emitted stop — ending episode at step %d", step_idx)
            break
        if action is None:
            no_op_streak += 1
            if no_op_streak % 10 == 1:
                log.info("policy not ready (no-op streak=%d) — waiting for slow system",
                         no_op_streak)
            # Re-sample observation so we still feed the buffer / display.
            obs = sim.get_sensor_observations()
            if display and not render_observation(obs):
                break
            continue
        no_op_streak = 0

        if action not in cfg.action_names:
            log.warning("ignoring action %r not in action space", action)
            continue

        obs = sim.step(action)
        step_idx += 1
        handle_observation(obs, step_idx)

        if display and not render_observation(obs):
            break

    log.info("policy loop finished after %d steps", step_idx)


# --------------------------------------------------------------------------- #
# Lightweight R2R benchmark
# --------------------------------------------------------------------------- #

def _episode_scene_id(episode: Dict[str, Any]) -> str:
    raw_scene_id = str(episode.get("scene_id") or "")
    if not raw_scene_id:
        raise ValueError(f"episode {episode.get('episode_id')} has no scene_id")
    scene_id = Path(raw_scene_id).stem
    if not scene_id:
        raise ValueError(
            f"could not resolve scene folder from scene_id={raw_scene_id!r}"
        )
    return scene_id


def _episode_instruction(episode: Dict[str, Any]) -> str:
    instruction = episode.get("instruction", "")
    if isinstance(instruction, dict):
        instruction = (
            instruction.get("instruction_text")
            or instruction.get("text")
            or ""
        )
    return str(instruction).strip()


def _episode_config(
    base_cfg: ResolvedConfig,
    benchmark: BenchmarkSpec,
    episode: Dict[str, Any],
) -> ResolvedConfig:
    scene_id = _episode_scene_id(episode)
    return replace(
        base_cfg,
        scene_path=_resolve_scene(benchmark.scenes_dir, scene_id),
        scene_id=scene_id,
        episode_id=episode.get("episode_id"),
        start_position=episode.get("start_position"),
        start_rotation=episode.get("start_rotation"),
        keyboard_map={},
    )


def _goal_positions(episode: Dict[str, Any]) -> List[np.ndarray]:
    goals = []
    for goal in episode.get("goals", []) or []:
        position = goal.get("position") if isinstance(goal, dict) else None
        if position is not None:
            goals.append(np.asarray(position, dtype=np.float32))
    if not goals:
        raise ValueError(f"episode {episode.get('episode_id')} has no goal positions")
    return goals


def _geodesic_distance(sim, position: np.ndarray, goals: List[np.ndarray]) -> float:
    """Return the shortest navigable distance from position to any goal."""
    import habitat_sim

    best = float("inf")
    for goal in goals:
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(position, dtype=np.float32)
        path.requested_end = np.asarray(goal, dtype=np.float32)
        if sim.pathfinder.find_path(path):
            best = min(best, float(path.geodesic_distance))
    return best


def _finite_or_none(value: float) -> Optional[float]:
    value = float(value)
    return value if math.isfinite(value) else None


def run_benchmark_episode(
    sim,
    cfg: ResolvedConfig,
    episode: Dict[str, Any],
    action_provider: BaseActionProvider,
    *,
    instruction: str,
    success_distance: float,
    display: bool,
    max_steps: int,
    max_no_op_calls: int,
) -> Dict[str, Any]:
    """Run one R2R episode and compute Habitat-style navigation metrics."""
    goals = _goal_positions(episode)
    action_provider.begin_episode(instruction)

    obs = reset_env(sim, cfg)
    handle_observation(obs, 0)
    if display and not render_observation(obs):
        return {
            "episode_id": episode.get("episode_id"),
            "scene_id": cfg.scene_id,
            "instruction": instruction,
            "status": "skipped",
            "termination_reason": "display_closed",
        }

    initial_position = np.asarray(
        sim.get_agent(0).get_state().position, dtype=np.float32
    ).copy()
    initial_distance = _geodesic_distance(sim, initial_position, goals)
    dataset_distance = (episode.get("info") or {}).get("geodesic_distance")
    if dataset_distance is not None and math.isfinite(float(dataset_distance)):
        shortest_path_length = float(dataset_distance)
    else:
        shortest_path_length = initial_distance

    previous_position = initial_position
    final_distance = initial_distance
    oracle_navigation_error = initial_distance
    path_length = 0.0
    oracle_success_path_length = (
        0.0 if initial_distance < success_distance else None
    )
    simulator_steps = 0
    policy_calls = 0
    no_op_calls = 0
    invalid_action_calls = 0
    collision_count = 0
    stop_called = False
    termination_reason = "max_steps"
    started_at = time.perf_counter()

    # Invalid policy outputs are also bounded by policy_call_limit, so a broken
    # adapter cannot make an episode loop forever.
    policy_call_limit = max_steps + max(0, max_no_op_calls)
    while simulator_steps < max_steps and policy_calls < policy_call_limit:
        action = action_provider.get_action(obs)
        policy_calls += 1

        if action == "__quit__":
            termination_reason = "policy_quit"
            break
        if action == "stop":
            stop_called = True
            termination_reason = "stop"
            break
        if action is None:
            no_op_calls += 1
            if no_op_calls >= max_no_op_calls:
                termination_reason = "max_no_op_calls"
                break
            # Re-submit the current frame while the asynchronous slow system
            # warms up. This mirrors run_policy_loop without consuming a sim
            # action or path length.
            obs = sim.get_sensor_observations()
            if display and not render_observation(obs):
                termination_reason = "display_closed"
                break
            continue
        if action not in cfg.action_names:
            invalid_action_calls += 1
            log.warning(
                "episode=%s ignoring action %r not in action space",
                episode.get("episode_id"),
                action,
            )
            continue

        obs = sim.step(action)
        simulator_steps += 1
        handle_observation(obs, simulator_steps)

        current_position = np.asarray(
            sim.get_agent(0).get_state().position, dtype=np.float32
        ).copy()
        displacement = float(np.linalg.norm(current_position - previous_position))
        path_length += displacement
        if action == "move_forward" and displacement < 1e-5:
            collision_count += 1
        previous_position = current_position

        if displacement > 1e-8:
            final_distance = _geodesic_distance(sim, current_position, goals)
        oracle_navigation_error = min(
            oracle_navigation_error, final_distance
        )
        if (
            oracle_success_path_length is None
            and final_distance < success_distance
        ):
            oracle_success_path_length = path_length

        if display and not render_observation(obs):
            termination_reason = "display_closed"
            break

    if (
        simulator_steps < max_steps
        and policy_calls >= policy_call_limit
        and termination_reason == "max_steps"
    ):
        termination_reason = "max_policy_calls"

    success = float(stop_called and final_distance < success_distance)
    oracle_success = float(oracle_navigation_error < success_distance)
    if (
        success
        and math.isfinite(shortest_path_length)
        and shortest_path_length >= 0.0
    ):
        spl = shortest_path_length / max(shortest_path_length, path_length, 1e-8)
    else:
        spl = 0.0
    if (
        oracle_success
        and oracle_success_path_length is not None
        and math.isfinite(shortest_path_length)
        and shortest_path_length >= 0.0
    ):
        oracle_spl = shortest_path_length / max(
            shortest_path_length, oracle_success_path_length, 1e-8
        )
    else:
        oracle_spl = 0.0

    return {
        "episode_id": episode.get("episode_id"),
        "trajectory_id": episode.get("trajectory_id"),
        "scene_id": cfg.scene_id,
        "instruction": instruction,
        "status": "ok",
        "termination_reason": termination_reason,
        "stop_called": stop_called,
        "simulator_steps": simulator_steps,
        # Matches InternNav's StepsTaken convention by counting STOP.
        "steps_taken": simulator_steps + int(stop_called),
        "policy_calls": policy_calls,
        "no_op_calls": no_op_calls,
        "invalid_action_calls": invalid_action_calls,
        "collision_count": collision_count,
        "path_length": path_length,
        "shortest_path_length": _finite_or_none(shortest_path_length),
        "distance_to_goal": _finite_or_none(final_distance),
        "navigation_error": _finite_or_none(final_distance),
        "oracle_navigation_error": _finite_or_none(
            oracle_navigation_error
        ),
        "success": success,
        "oracle_success": oracle_success,
        "spl": float(spl),
        "oracle_spl": float(oracle_spl),
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def _aggregate_benchmark_metrics(
    episode_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    successful_results = [
        result for result in episode_results if result.get("status") == "ok"
    ]
    metric_names = [
        "distance_to_goal",
        "navigation_error",
        "oracle_navigation_error",
        "success",
        "oracle_success",
        "spl",
        "oracle_spl",
        "path_length",
        "steps_taken",
        "policy_calls",
        "collision_count",
        "elapsed_seconds",
    ]
    metrics: Dict[str, Any] = {
        "episodes_attempted": len(episode_results),
        "episodes_completed": len(successful_results),
        "episodes_failed": sum(
            result.get("status") == "error" for result in episode_results
        ),
        "episodes_skipped": sum(
            result.get("status") == "skipped" for result in episode_results
        ),
    }
    for name in metric_names:
        values = [
            float(result[name])
            for result in successful_results
            if result.get(name) is not None
        ]
        metrics[name] = sum(values) / len(values) if values else None
    return metrics


def _write_benchmark_results(
    output_path: Path,
    *,
    benchmark: BenchmarkSpec,
    selected_episode_count: int,
    episode_results: List[Dict[str, Any]],
    interrupted: bool,
) -> None:
    payload = {
        "schema_version": 1,
        "mode": "benchmark",
        "split": benchmark.split,
        "data_path": str(benchmark.data_path),
        "scenes_dir": str(benchmark.scenes_dir),
        "success_distance": benchmark.success_distance,
        "selected_episode_count": selected_episode_count,
        "interrupted": interrupted,
        "metrics": _aggregate_benchmark_metrics(episode_results),
        "episodes": episode_results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    os.replace(temp_path, output_path)


def run_benchmark(args: argparse.Namespace, test_config_path: Path) -> int:
    benchmark = load_benchmark_spec(
        test_config_path,
        split_override=args.split,
        data_path_override=args.data_path,
    )
    episodes = benchmark.episodes
    if args.scene_id:
        episodes = [
            episode
            for episode in episodes
            if _episode_scene_id(episode) == args.scene_id
        ]
    if args.episode_id is not None:
        episodes = [
            episode
            for episode in episodes
            if str(episode.get("episode_id")) == str(args.episode_id)
        ]
    if args.episode_limit is not None:
        if args.episode_limit < 1:
            raise ValueError("--episode-limit must be >= 1")
        episodes = episodes[:args.episode_limit]
    if not episodes:
        raise ValueError("No R2R episodes matched the benchmark filters")

    first_scene_id = _episode_scene_id(episodes[0])
    base_cfg = load_config(
        "test", test_config_path, first_scene_id, cli_episode_id=None
    )
    first_cfg = _episode_config(base_cfg, benchmark, episodes[0])

    output_template = args.benchmark_output or (
        "./logs/habitat/havana_r2r/{split}_metrics.json"
    )
    output_path = Path(
        str(output_template).format(split=benchmark.split)
    ).expanduser().resolve()

    if args.keyboard:
        raise ValueError("--keyboard is only supported in --mode test")
    if not args.policy and not args.policy_remote:
        args.policy_remote = "/tmp/havana.sock"
        log.info(
            "benchmark mode defaulting to Havana RPC server at %s",
            args.policy_remote,
        )
    if args.policy and args.policy_remote:
        log.warning("--policy-remote takes precedence over in-process --policy")
        args.policy = False

    log.info("===== eval_fast R2R benchmark =====")
    log.info("split            : %s", benchmark.split)
    log.info("data_path        : %s", benchmark.data_path)
    log.info("scenes_dir       : %s", benchmark.scenes_dir)
    log.info("episodes         : %d", len(episodes))
    log.info("success_distance : %.2f m", benchmark.success_distance)
    log.info("max_steps        : %d", args.max_steps)
    log.info("output           : %s", output_path)
    log.info("===================================")

    policy_server = None
    policy_client = None
    provider: BaseActionProvider
    havana_root = Path(__file__).resolve().parents[2]
    if str(havana_root) not in sys.path:
        sys.path.insert(0, str(havana_root))

    if args.policy_remote:
        from fast_system.scripts.policy_client import PolicyClient  # noqa: WPS433

        policy_client = PolicyClient(
            args.policy_remote,
            authkey=args.policy_authkey.encode(),
            action_names=first_cfg.action_names,
        )
        provider = RemotePolicyActionProvider(policy_client)
    else:
        from fast_system.scripts.run_pipeline import (  # noqa: WPS433
            HavanaPolicyServer,
            load_pipeline_config,
        )

        pipe_cfg_path = (
            Path(args.policy_config).expanduser().resolve()
            if args.policy_config
            else None
        )
        pipe_cfg = load_pipeline_config(pipe_cfg_path)
        policy_server = HavanaPolicyServer(pipe_cfg)
        policy_server.attach_action_space(first_cfg.action_names)
        policy_server.start()
        provider = InProcessPolicyActionProvider(policy_server)

    results: List[Dict[str, Any]] = []
    sim = None
    current_scene_id = None
    interrupted = False
    exit_code = 0
    try:
        for index, episode in enumerate(episodes, start=1):
            cfg = _episode_config(base_cfg, benchmark, episode)
            instruction = args.instruction or _episode_instruction(episode)
            log.info(
                "[%d/%d] episode=%s scene=%s instruction=%r",
                index,
                len(episodes),
                episode.get("episode_id"),
                cfg.scene_id,
                instruction[:100],
            )
            try:
                if sim is None or current_scene_id != cfg.scene_id:
                    if sim is not None:
                        sim.close()
                    sim = build_env(cfg)
                    current_scene_id = cfg.scene_id
                result = run_benchmark_episode(
                    sim,
                    cfg,
                    episode,
                    provider,
                    instruction=instruction,
                    success_distance=benchmark.success_distance,
                    display=args.display,
                    max_steps=int(args.max_steps),
                    max_no_op_calls=int(args.max_no_op_calls),
                )
                results.append(result)
                log.info(
                    "[%d/%d] episode=%s done: success=%.0f spl=%.3f "
                    "nav_error=%s steps=%d reason=%s",
                    index,
                    len(episodes),
                    episode.get("episode_id"),
                    result.get("success", 0.0),
                    result.get("spl", 0.0),
                    result.get("navigation_error"),
                    result.get("steps_taken", 0),
                    result.get("termination_reason"),
                )
            except Exception as exc:  # noqa: BLE001
                log.exception(
                    "[%d/%d] episode=%s failed",
                    index,
                    len(episodes),
                    episode.get("episode_id"),
                )
                results.append({
                    "episode_id": episode.get("episode_id"),
                    "scene_id": cfg.scene_id,
                    "instruction": instruction,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                })
                exit_code = 1
                if args.fail_fast:
                    raise

            if index % args.save_every == 0:
                _write_benchmark_results(
                    output_path,
                    benchmark=benchmark,
                    selected_episode_count=len(episodes),
                    episode_results=results,
                    interrupted=False,
                )
    except KeyboardInterrupt:
        interrupted = True
        exit_code = 130
        log.warning("benchmark interrupted; writing partial results")
    finally:
        _write_benchmark_results(
            output_path,
            benchmark=benchmark,
            selected_episode_count=len(episodes),
            episode_results=results,
            interrupted=interrupted,
        )
        if sim is not None:
            sim.close()
        if policy_client is not None:
            policy_client.close()
        if policy_server is not None:
            policy_server.stop()
        if args.display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass

    metrics = _aggregate_benchmark_metrics(results)
    log.info("benchmark metrics: %s", metrics)
    log.info("benchmark results written to %s", output_path)
    return exit_code


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Minimal Habitat test/eval loop.")
    p.add_argument("--mode", choices=["test", "benchmark"], default="test")
    p.add_argument("--display", action="store_true", help="show RGB + Depth window")
    p.add_argument("--keyboard", action="store_true", help="drive agent from keyboard")
    p.add_argument("--policy", action="store_true",
                   help="drive agent from the Havana policy server in-process "
                        "(requires torch / cosmos available in the current venv)")
    p.add_argument("--policy-remote", default=None,
                   help="connect to a remote Havana policy server over "
                        "multiprocessing.connection. Pass a unix socket path "
                        "(e.g. /tmp/havana.sock) or host:port. Use this when "
                        "Habitat and policy live in different venvs.")
    p.add_argument("--policy-authkey", default="havana",
                   help="HMAC key shared with `run_pipeline.py --serve`.")
    p.add_argument("--policy-config", default=None,
                   help="optional YAML PipelineConfig path for the policy server "
                        "(in-process mode only)")
    p.add_argument("--instruction", default=None,
                   help="override the episode language instruction")
    p.add_argument("--max-steps", type=int, default=500,
                   help="max simulator steps per episode")
    p.add_argument("--max-no-op-calls", type=int, default=1000,
                   help="max policy None/no-op responses per benchmark episode")
    p.add_argument("--scene-id", default=None, help="MP3D scene folder name")
    p.add_argument("--episode-id", type=int, default=None,
                   help="optional episode id (also filters benchmark mode)")
    p.add_argument("--split", default=None,
                   help="R2R split; defaults to habitat.dataset.split in vln_r2r.yaml")
    p.add_argument("--data-path", default=None,
                   help="optional R2R JSON/JSON.GZ path or {split} path template")
    p.add_argument("--episode-limit", type=int, default=None,
                   help="evaluate only the first N matching benchmark episodes")
    p.add_argument("--benchmark-output", default=None,
                   help="benchmark JSON output path; may contain {split}")
    p.add_argument("--save-every", type=int, default=10,
                   help="checkpoint benchmark JSON every N episodes")
    p.add_argument("--fail-fast", action="store_true",
                   help="abort benchmark on the first episode error")
    p.add_argument(
        "--test-config",
        default=str(Path(__file__).parent / "config" / "test.yaml"),
        help="path to the test mode config",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_steps < 1:
        raise ValueError("--max-steps must be >= 1")
    if args.max_no_op_calls < 1:
        raise ValueError("--max-no-op-calls must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")

    test_config_path = Path(args.test_config).expanduser().resolve()
    if args.mode == "benchmark":
        return run_benchmark(args, test_config_path)

    cfg = load_config(args.mode, test_config_path, args.scene_id, args.episode_id)

    log.info("===== eval_fast configuration =====")
    log.info("mode             : %s", args.mode)
    log.info("display          : %s", args.display)
    log.info("keyboard         : %s", args.keyboard)
    log.info("scene            : %s (%s)", cfg.scene_id, cfg.scene_path)
    log.info("episode_id       : %s", cfg.episode_id)
    log.info("sensors          : %s", [
        f"{s.name}({s.sensor_type},{s.width}x{s.height},hfov={s.hfov})"
        for s in cfg.sensors
    ])
    log.info("actions          : %s", cfg.action_names)
    log.info("forward_step     : %.3f m", cfg.forward_step_size)
    log.info("turn_angle       : %.1f deg", cfg.turn_angle_deg)
    log.info("====================================")

    if args.keyboard and not args.display:
        # We piggy-back on OpenCV's window for keyboard input. Without a
        # window there's nowhere for waitKey() to receive events.
        log.warning("--keyboard requires a display window; enabling --display")
        args.display = True

    if args.keyboard and (args.policy or args.policy_remote):
        log.warning("--keyboard takes precedence; disabling --policy/--policy-remote")
        args.policy = False
        args.policy_remote = None
    if args.policy and args.policy_remote:
        log.warning("--policy-remote takes precedence over in-process --policy")
        args.policy = False

    sim = build_env(cfg)
    policy_server = None
    policy_client = None
    try:
        if args.keyboard:
            provider: BaseActionProvider = KeyboardActionProvider(
                cfg.keyboard_map, cfg.action_names
            )
            run_test_loop(sim, cfg, provider, display=args.display)

        elif args.policy_remote:
            # Cross-venv path: only stdlib + numpy in this process.
            havana_root = Path(__file__).resolve().parents[2]
            if str(havana_root) not in sys.path:
                sys.path.insert(0, str(havana_root))
            from fast_system.scripts.policy_client import PolicyClient  # noqa: WPS433

            policy_client = PolicyClient(
                args.policy_remote,
                authkey=args.policy_authkey.encode(),
                action_names=cfg.action_names,
            )
            if args.instruction:
                policy_client.set_instruction(args.instruction)

            class _RemoteProvider(BaseActionProvider):
                def __init__(self, client):
                    self._client = client
                    self._sid = 0
                def get_action(self, obs):
                    sid = self._sid
                    self._sid += 1
                    return self._client.get_action(obs, sid)

            provider = _RemoteProvider(policy_client)
            run_policy_loop(sim, cfg, provider, display=args.display, max_steps=int(args.max_steps))

        elif args.policy:
            # In-process path: needs torch / cosmos in this venv.
            havana_root = Path(__file__).resolve().parents[2]
            if str(havana_root) not in sys.path:
                sys.path.insert(0, str(havana_root))
            from fast_system.scripts.run_pipeline import (  # noqa: WPS433
                HavanaPolicyServer,
                PolicyActionProvider,
                load_pipeline_config,
            )

            pipe_cfg_path = Path(args.policy_config).expanduser().resolve() if args.policy_config else None
            pipe_cfg = load_pipeline_config(pipe_cfg_path)
            if args.instruction:
                pipe_cfg.instruction = args.instruction
            policy_server = HavanaPolicyServer(pipe_cfg)
            policy_server.start()
            provider = PolicyActionProvider(policy_server, action_names=cfg.action_names)
            run_policy_loop(sim, cfg, provider, display=args.display, max_steps=int(args.max_steps))

        else:
            # Reserved hook for plugging in a policy ActionProvider later.
            provider = NullActionProvider()
            run_test_loop(sim, cfg, provider, display=args.display)
    finally:
        if policy_client is not None:
            policy_client.close()
        if policy_server is not None:
            policy_server.stop()
        if args.display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:  # noqa: BLE001
                pass
        sim.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
