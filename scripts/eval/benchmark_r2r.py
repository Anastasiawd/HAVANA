#!/usr/bin/env python3
"""Self-contained VLN-CE R2R benchmark — no InternNav internals needed.

Dependencies: habitat-sim, numpy, yaml, cv2 (optional, for display).

Usage:
    # 1. Start slow system (Cosmos venv):
    python slow_system/scripts/slow_server.py --serve /tmp/havana-slow.sock \
        --config slow_system/configs/slow_server.yaml

    # 2. Start fast system (torch venv):
    python fast_system/scripts/run_pipeline.py --serve /tmp/havana.sock \
        --config fast_system/configs/policy_server.yaml

    # 3. Run benchmark (habitat venv):
    python scripts/eval/benchmark_r2r.py \
        --split val_unseen \
        --policy-remote /tmp/havana.sock \
        --max-episodes 10 \
        --display
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s][%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark_r2r")


# =========================================================================== #
# Config — self-contained; reads vln_r2r.yaml for sensor / action defs only
# =========================================================================== #

VLN_R2R_YAML = Path(
    "/home/wangdzh/ffnav/InternNav/scripts/eval/configs/vln_r2r.yaml"
)
SCENES_DIR = Path("/home/wangdzh/data/mp3d")
R2R_DATA_PATH = "/home/wangdzh/data/InternData-N1/vln_ce/raw_data/r2r/{split}/{split}.json.gz"
SUCCESS_RADIUS = 3.0  # meters


@dataclass
class SensorSpec:
    name: str
    sensor_type: str
    width: int
    height: int
    hfov: float
    position: List[float]
    min_depth: float = 0.0
    max_depth: float = 10.0


@dataclass
class EnvConfig:
    scene_path: Path
    forward_step_size: float
    turn_angle_deg: float
    tilt_angle_deg: float
    gpu_device_id: int
    agent_height: float
    agent_radius: float
    sensors: List[SensorSpec]
    action_names: List[str]


def _read_yaml(path: Path) -> dict:
    with path.open("r") as f:
        return yaml.safe_load(f) or {}


def _bare_scene_id(scene_id: str) -> str:
    """Extract bare scene name from R2R's full path format.

    'mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb'  ->  '1LXtFkjw3qL'
    '1LXtFkjw3qL'                        ->  '1LXtFkjw3qL'
    """
    return Path(scene_id).stem


def load_env_config(scene_id: str, camera_height: float = 1.25,
                    camera_fov: Optional[float] = None) -> EnvConfig:
    """Parse vln_r2r.yaml to extract sensor + action specs.

    camera_height:  sensor position[1] in meters (default 1.25 = eye level)
    camera_fov:     override hfov in degrees (default from vln_r2r.yaml)
    """
    bare = _bare_scene_id(scene_id)
    vln = _read_yaml(VLN_R2R_YAML)
    sim = (vln.get("habitat", {}) or {}).get("simulator", {}) or {}
    agents = sim.get("agents", {}) or {}
    main = agents.get("main_agent", {}) or {}
    sim_sensors = main.get("sim_sensors", {}) or {}

    sensor_defaults = {
        "rgb_sensor":  dict(width=640, height=480, hfov=90,
                            position=[0.0, camera_height, 0.0], sensor_type="color"),
        "depth_sensor": dict(width=640, height=480, hfov=90,
                             position=[0.0, camera_height, 0.0], sensor_type="depth"),
    }
    sensors = []
    for name, overrides in sim_sensors.items():
        if name not in sensor_defaults:
            continue
        d = dict(sensor_defaults[name])
        d.update(overrides or {})
        if camera_fov is not None:
            d["hfov"] = camera_fov
        sensors.append(SensorSpec(
            name=name,
            sensor_type=d["sensor_type"],
            width=int(d["width"]), height=int(d["height"]),
            hfov=float(d["hfov"]),
            position=list(d.get("position", [0.0, 1.25, 0.0])),
            min_depth=float(d.get("min_depth", 0.0)),
            max_depth=float(d.get("max_depth", 10.0)),
        ))

    actions = (vln.get("habitat", {}) or {}).get("task", {}) or {}
    actions = actions.get("actions", {}) or {}
    action_names = list(actions.keys())

    scene_path = SCENES_DIR / bare / f"{bare}.glb"
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene not found: {scene_path}")

    return EnvConfig(
        scene_path=scene_path,
        forward_step_size=float(sim.get("forward_step_size", 0.10)),
        turn_angle_deg=float(sim.get("turn_angle", 15)),
        tilt_angle_deg=float(sim.get("tilt_angle", 60)),
        gpu_device_id=int((sim.get("habitat_sim_v0") or {}).get("gpu_device_id", 0)),
        agent_height=1.5, agent_radius=0.1,
        sensors=sensors, action_names=action_names,
    )


# =========================================================================== #
# R2R episode loader
# =========================================================================== #

def load_r2r_episodes(split: str, max_episodes: Optional[int] = None) -> List[dict]:
    path = R2R_DATA_PATH.format(split=split)
    log.info("Loading R2R episodes from %s", path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        data = json.load(f)
    episodes = data.get("episodes", data)
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    log.info("Loaded %d episodes (split=%s)", len(episodes), split)
    return episodes


# =========================================================================== #
# Habitat-Sim env
# =========================================================================== #

def build_env(cfg: EnvConfig):
    import habitat_sim

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(cfg.scene_path)
    sim_cfg.gpu_device_id = cfg.gpu_device_id
    sim_cfg.enable_physics = False

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.height = cfg.agent_height
    agent_cfg.radius = cfg.agent_radius

    specs = []
    for s in cfg.sensors:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = s.name
        spec.resolution = [s.height, s.width]
        spec.position = s.position
        spec.hfov = s.hfov
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        spec.sensor_type = (
            habitat_sim.SensorType.COLOR if s.sensor_type == "color"
            else habitat_sim.SensorType.DEPTH
        )
        specs.append(spec)
    agent_cfg.sensor_specifications = specs

    name_to_actuation = {
        "move_forward": ("move_forward", cfg.forward_step_size),
        "turn_left":    ("turn_left",    cfg.turn_angle_deg),
        "turn_right":   ("turn_right",   cfg.turn_angle_deg),
        "look_up":      ("look_up",      cfg.tilt_angle_deg),
        "look_down":    ("look_down",    cfg.tilt_angle_deg),
        "stop":         ("move_forward", 0.0),
    }
    action_space = {}
    for name in cfg.action_names:
        if name in name_to_actuation:
            sim_name, amount = name_to_actuation[name]
            action_space[name] = habitat_sim.agent.ActionSpec(
                sim_name, habitat_sim.agent.ActuationSpec(amount=amount)
            )
    agent_cfg.action_space = action_space

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def reset_to_episode(sim, episode: dict) -> dict:
    """Place agent at episode start pose; return initial obs."""
    import habitat_sim
    agent = sim.get_agent(0)
    state = habitat_sim.AgentState()
    state.position = np.array(episode["start_position"], dtype=np.float32)
    # vln_ce quaternion: [x, y, z, w] → numpy quaternion (w, x, y, z)
    x, y, z, w = episode["start_rotation"]
    import quaternion  # noqa: F401
    state.rotation = np.quaternion(w, x, y, z)
    agent.set_state(state)
    return sim.get_sensor_observations()


def get_agent_position(sim) -> np.ndarray:
    return sim.get_agent(0).get_state().position


# =========================================================================== #
# Policy client (remote RPC, stdlib + numpy only)
# =========================================================================== #

class PolicyClient:
    """Talks to run_pipeline.py --serve over multiprocessing.connection."""

    def __init__(self, address: str, authkey: bytes, action_names: List[str]):
        from multiprocessing.connection import Client
        addr = address
        if isinstance(address, str) and ":" in address and not address.startswith("/"):
            host, port = address.rsplit(":", 1)
            addr = (host, int(port))
        self._conn = Client(addr, authkey=authkey)
        # Attach action space
        self._conn.send(("attach_action_space", action_names))
        resp = self._conn.recv()
        if not isinstance(resp, dict) or not resp.get("ok"):
            raise RuntimeError(f"attach_action_space failed: {resp}")

    def set_instruction(self, text: str) -> None:
        self._conn.send(("set_instruction", text))
        resp = self._conn.recv()
        if not isinstance(resp, dict) or not resp.get("ok"):
            log.warning("set_instruction failed: %s", resp)

    def get_action(self, obs: dict, step_id: int) -> Optional[str]:
        # Strip non-numpy values for clean pickle
        clean = {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}
        self._conn.send(("get_action", clean, step_id))
        resp = self._conn.recv()
        if isinstance(resp, dict) and resp.get("ok"):
            return resp["result"]
        return None

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# =========================================================================== #
# Display
# =========================================================================== #

def render_observation(obs: dict, window_name: str = "habitat") -> bool:
    import cv2
    panels = []
    if "rgb_sensor" in obs:
        rgb = obs["rgb_sensor"]
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        panels.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if "depth_sensor" in obs:
        d = np.clip(obs["depth_sensor"] if obs["depth_sensor"].ndim == 2
                    else obs["depth_sensor"][..., 0], 0.0, None)
        d_max = d.max() if d.max() > 0 else 1.0
        panels.append(cv2.applyColorMap((d / d_max * 255).astype(np.uint8),
                                        cv2.COLORMAP_VIRIDIS))
    if not panels:
        return True
    h = max(p.shape[0] for p in panels)
    panels = [cv2.resize(p, (int(p.shape[1] * h / p.shape[0]), h))
              if p.shape[0] != h else p for p in panels]
    canvas = np.concatenate(panels, axis=1)
    cv2.imshow(window_name, canvas)
    return (cv2.waitKey(1) & 0xFF) not in (ord("q"), 27)


# =========================================================================== #
# Metrics
# =========================================================================== #

@dataclass
class EpisodeResult:
    episode_id: int
    scene_id: str
    success: bool
    steps: int
    path_length_m: float
    final_distance_to_goal_m: float
    instruction: str
    elapsed_s: float


def print_summary(results: List[EpisodeResult]) -> None:
    n = len(results)
    if n == 0:
        log.warning("No episodes evaluated")
        return
    sr = sum(1 for r in results if r.success) / n
    successes = [r for r in results if r.success]
    failures = [r for r in results if not r.success]
    log.info("=" * 60)
    log.info("R2R Benchmark Summary  (n=%d)", n)
    log.info("  Success Rate (SR)          : %.1f %%  (%d/%d)",
             sr * 100, len(successes), n)
    log.info("  Avg path length (success)  : %.2f m", np.mean(
        [r.path_length_m for r in successes]) if successes else float("nan"))
    log.info("  Avg steps    (success)     : %.1f", np.mean(
        [r.steps for r in successes]) if successes else float("nan"))
    log.info("  Avg final dist (failure)   : %.2f m", np.mean(
        [r.final_distance_to_goal_m for r in failures]) if failures else float("nan"))
    log.info("  Avg elapsed per episode    : %.2f s", np.mean(
        [r.elapsed_s for r in results]))
    log.info("=" * 60)

    # Save JSON
    out_path = Path("outputs/benchmark_r2r_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump([{
            "episode_id": r.episode_id,
            "scene_id": r.scene_id,
            "success": r.success,
            "steps": r.steps,
            "path_length_m": round(r.path_length_m, 3),
            "final_distance_to_goal_m": round(r.final_distance_to_goal_m, 3),
            "instruction": r.instruction[:120],
            "elapsed_s": round(r.elapsed_s, 2),
        } for r in results], f, indent=2)
    log.info("Detailed results saved to %s", out_path)


# =========================================================================== #
# Episode runner
# =========================================================================== #

def run_episode(
    sim,
    episode: dict,
    client: PolicyClient,
    max_steps: int,
    display: bool,
) -> EpisodeResult:
    instruction = episode.get("instruction", {})
    if isinstance(instruction, dict):
        instruction = instruction.get("instruction_text", "")
    goal_pos = np.array(episode["goals"][0]["position"], dtype=np.float32)

    log.info("[Ep %d] scene=%s  instruction=%s",
             episode["episode_id"], episode["scene_id"], instruction[:100])

    client.set_instruction(instruction)
    obs = reset_to_episode(sim, episode)

    prev_pos = get_agent_position(sim).copy()
    path_length = 0.0
    success = False
    step = 0
    t0 = time.time()

    while step < max_steps:
        if display and not render_observation(obs):
            log.info("Display window closed — aborting episode")
            break

        action = client.get_action(obs, step)

        if action is None:
            # Policy not ready yet (slow system warming up)
            obs = sim.get_sensor_observations()
            continue

        if action == "stop":
            log.info("[Ep %d] step=%d  policy emitted stop", episode["episode_id"], step)
            break

        obs = sim.step(action)
        step += 1

        cur_pos = get_agent_position(sim)
        path_length += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos.copy()

        dist = float(np.linalg.norm(cur_pos - goal_pos))
        if dist < SUCCESS_RADIUS:
            success = True
            log.info("[Ep %d] step=%d  GOAL REACHED (dist=%.2fm)", episode["episode_id"], step, dist)
            break

        if step % 50 == 0:
            log.info("[Ep %d] step=%d  dist_to_goal=%.2fm  path=%.2fm",
                     episode["episode_id"], step, dist, path_length)

    elapsed = time.time() - t0
    final_dist = float(np.linalg.norm(get_agent_position(sim) - goal_pos))

    return EpisodeResult(
        episode_id=episode["episode_id"],
        scene_id=episode["scene_id"],
        success=success,
        steps=step,
        path_length_m=path_length,
        final_distance_to_goal_m=final_dist,
        instruction=instruction,
        elapsed_s=elapsed,
    )


# =========================================================================== #
# Main
# =========================================================================== #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Self-contained VLN-CE R2R benchmark")
    p.add_argument("--split", default="val_unseen",
                   choices=["train", "val_seen", "val_unseen", "test"])
    p.add_argument("--scene-id", default=None,
                   help="Only evaluate episodes in this MP3D scene (e.g. 1LXtFkjw3qL). "
                        "Comma-separated for multiple scenes.")
    p.add_argument("--max-episodes", type=int, default=None,
                   help="Limit number of episodes (default: all)")
    p.add_argument("--max-steps", type=int, default=500,
                   help="Max steps per episode")
    p.add_argument("--policy-remote", default="/tmp/havana.sock",
                   help="Havana policy server address (unix socket or host:port)")
    p.add_argument("--policy-authkey", default="havana")
    p.add_argument("--list-scenes", action="store_true",
                   help="List available scenes in the split and exit")
    p.add_argument("--display", action="store_true",
                   help="Show RGB + Depth window")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--camera-height", type=float, default=1.25,
                   help="Camera height above agent origin in meters (default: 1.25)")
    p.add_argument("--camera-fov", type=float, default=None,
                   help="Override horizontal FOV in degrees (default: from vln_r2r.yaml)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # 1. Load R2R episodes
    episodes = load_r2r_episodes(args.split, max_episodes=None)  # load all, filter first
    if not episodes:
        log.error("No episodes loaded")
        return 1

    # 1a. --list-scenes
    if args.list_scenes:
        scenes: Dict[str, int] = {}
        for ep in episodes:
            bare = _bare_scene_id(ep["scene_id"])
            scenes[bare] = scenes.get(bare, 0) + 1
        log.info("Scenes in split '%s' (%d episodes total):", args.split, len(episodes))
        for sid, count in sorted(scenes.items()):
            log.info("  %s  (%d episodes)", sid, count)
        return 0

    # 1b. Filter by scene if --scene-id is given
    if args.scene_id:
        allowed = set(s.strip() for s in args.scene_id.split(","))
        before = len(episodes)
        episodes = [
            ep for ep in episodes
            if _bare_scene_id(ep["scene_id"]) in allowed
        ]
        log.info("Filtered by scene: %d -> %d episodes (scenes=%s)",
                 before, len(episodes), sorted(allowed))
        if not episodes:
            log.error("No episodes match --scene-id %s", args.scene_id)
            return 1

    # 2. Group by scene so we only load each scene once
    scene_groups: Dict[str, List[dict]] = {}
    for ep in episodes:
        scene_groups.setdefault(ep["scene_id"], []).append(ep)

    # 3. Apply --max-episodes AFTER filtering (truncate per-scene proportionally)
    if args.max_episodes is not None and len(episodes) > args.max_episodes:
        episodes = episodes[:args.max_episodes]
        scene_groups.clear()
        for ep in episodes:
            scene_groups.setdefault(ep["scene_id"], []).append(ep)
        log.info("Truncated to %d episodes after scene filter", len(episodes))

    # 3. Connect to policy server
    havana_root = Path(__file__).resolve().parents[2]
    if str(havana_root) not in sys.path:
        sys.path.insert(0, str(havana_root))

    env_cfg = load_env_config(episodes[0]["scene_id"])
    client = PolicyClient(
        args.policy_remote,
        authkey=args.policy_authkey.encode(),
        action_names=env_cfg.action_names,
    )
    log.info("Connected to policy server at %s", args.policy_remote)

    # 4. Run episodes
    all_results: List[EpisodeResult] = []
    try:
        for scene_id, scene_eps in scene_groups.items():
            log.info("--- Loading scene %s (%d episodes) ---", scene_id, len(scene_eps))
            env_cfg = load_env_config(scene_id)
            env_cfg.gpu_device_id = args.gpu_id
            sim = build_env(env_cfg)
            try:
                for ep in scene_eps:
                    result = run_episode(sim, ep, client, args.max_steps, args.display)
                    all_results.append(result)
                    log.info("[Ep %d] success=%s  steps=%d  path=%.2fm  final_dist=%.2fm  time=%.1fs",
                             ep["episode_id"], result.success, result.steps,
                             result.path_length_m, result.final_distance_to_goal_m,
                             result.elapsed_s)
            finally:
                sim.close()
    finally:
        client.close()
        if args.display:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    # 5. Report
    print_summary(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
