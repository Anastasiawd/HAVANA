#!/usr/bin/env python3
"""Build mixed SCAND + RECON + GS JSONL indexes from extracted scene folders.

Expected input layout for each scene:
  <scene>/
    images/
      *.jpg|*.png|...
    trajectory.npy   # shape [T, 4], columns: x, y, z, yaw
    timestamps.npy   # optional, shape [T]

The generated records are file-backed equivalents of the original SCAND bag
indexes.  Every 16-frame clip stores route supervision in the clip-start local
frame, plus a dense local pose trajectory that Stage-2 can re-root at later
fast-system observation offsets.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from tqdm import tqdm


DEFAULT_SOURCES = (
    ("recon", Path("/home/wangdzh/data/recon_dataset/recon_extracted")),
    ("scand", Path("/home/wangdzh/data/SCAND/scand_extracted")),
    ("gs", Path("/home/wangdzh/data/gs2_withres/gs_extracted")),
)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TURN_NAMES = ("left", "straight", "right")


@dataclass(frozen=True)
class Scene:
    dataset: str
    scene_dir: Path
    images_dir: Path
    image_paths: tuple[Path, ...]
    trajectory: np.ndarray
    timestamps: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare mixed extracted SCAND+RECON route-clip indexes.")
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help=(
            "Dataset source as NAME=PATH. May be repeated. "
            "Defaults to recon=/home/wangdzh/data/recon_dataset/recon_extracted and "
            "scand=/home/wangdzh/data/SCAND/scand_extracted and "
            "gs=/home/wangdzh/data/gs2_withres/gs_extracted."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for output JSONL indexes.",
    )
    parser.add_argument("--index-prefix", default="mix_stage1", help="Output prefix, e.g. mix_stage1_train_index.jsonl.")
    parser.add_argument("--clip-frames", type=int, default=16, help="Video frames per slow-system route clip.")
    parser.add_argument("--stride-frames", type=int, default=4, help="Frame stride between route clips.")
    parser.add_argument("--traj-points", type=int, default=32, help="Dense trajectory points stored per clip.")
    parser.add_argument("--num-anchors", type=int, default=4, help="Uniform Stage-1 route anchor count.")
    parser.add_argument("--min-distance", type=float, default=0.25, help="Skip clips moving less than this many meters.")
    parser.add_argument("--turn-threshold-deg", type=float, default=15.0, help="Goal bearing threshold for turn labels.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Scene-level validation split per source.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Scene-level test split per source.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-scenes-per-source", type=int, default=None)
    parser.add_argument("--max-samples-per-scene", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def natural_key(path: Path | str) -> list[object]:
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def parse_sources(values: Sequence[str] | None) -> list[tuple[str, Path]]:
    if not values:
        return [(name, path.expanduser()) for name, path in DEFAULT_SOURCES]

    sources: list[tuple[str, Path]] = []
    for value in values:
        if "=" in value:
            name, path = value.split("=", 1)
            name = name.strip()
        else:
            path_obj = Path(value).expanduser()
            name = path_obj.name
            path = value
        if not name:
            raise ValueError(f"Invalid --source {value!r}: dataset name is empty.")
        sources.append((sanitize_name(name), Path(path).expanduser()))
    return sources


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "dataset"


def normalize_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotate_world_to_local(delta_xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return delta_xy @ rot


def poses_to_local(poses: np.ndarray, origin_pose: np.ndarray | None = None) -> np.ndarray:
    if poses.ndim != 2 or poses.shape[1] < 4:
        raise ValueError(f"trajectory poses must have shape [T,4+], got {poses.shape}.")
    poses = poses.astype(np.float64, copy=False)
    origin = poses[0] if origin_pose is None else origin_pose.astype(np.float64, copy=False)
    yaw = np.unwrap(poses[:, 3])
    origin_yaw = float(origin[3])
    local_xy = rotate_world_to_local(poses[:, :2] - origin[:2], origin_yaw)
    local_yaw = normalize_angle(yaw - origin_yaw)
    return np.concatenate((local_xy, local_yaw[:, None]), axis=1).astype(np.float32)


def interpolate_poses(
    trajectory: np.ndarray,
    positions: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> np.ndarray:
    if timestamps is not None:
        domain = (timestamps.astype(np.float64) - float(timestamps[0])) / 1e9
        sample_domain = (positions.astype(np.float64) - float(timestamps[0])) / 1e9
    else:
        domain = np.arange(len(trajectory), dtype=np.float64)
        sample_domain = positions.astype(np.float64)

    source = trajectory.astype(np.float64, copy=False)
    yaw = np.unwrap(source[:, 3])
    out = np.empty((len(sample_domain), 4), dtype=np.float32)
    for col in range(3):
        out[:, col] = np.interp(sample_domain, domain, source[:, col]).astype(np.float32)
    out[:, 3] = normalize_angle(np.interp(sample_domain, domain, yaw)).astype(np.float32)
    return out


def sample_dense_poses(scene: Scene, start: int, end: int, traj_points: int) -> np.ndarray:
    if scene.timestamps is None:
        positions = np.linspace(float(start), float(end), int(traj_points), dtype=np.float64)
        return interpolate_poses(scene.trajectory, positions)

    start_ts = int(scene.timestamps[start])
    end_ts = int(scene.timestamps[end])
    positions = np.linspace(start_ts, end_ts, int(traj_points), dtype=np.float64)
    return interpolate_poses(scene.trajectory, positions, timestamps=scene.timestamps)


def turn_label(goal_xy: np.ndarray, threshold_rad: float) -> int:
    angle = math.atan2(float(goal_xy[1]), float(goal_xy[0]))
    if angle > threshold_rad:
        return 0
    if angle < -threshold_rad:
        return 2
    return 1


def sample_anchors(traj_local_xy: np.ndarray, num_anchors: int) -> np.ndarray:
    if len(traj_local_xy) == 1:
        return np.repeat(traj_local_xy, num_anchors, axis=0)
    indices = np.linspace(1, len(traj_local_xy) - 1, int(num_anchors))
    return traj_local_xy[np.rint(indices).astype(np.int64)]


def round_json_floats(value: object, decimals: int) -> object:
    if isinstance(value, list):
        return [round_json_floats(item, decimals) for item in value]
    return round(float(value), decimals)


def json_float_array(array: np.ndarray, decimals: int = 6) -> list[object]:
    rounded = np.round(array.astype(np.float32, copy=False), decimals=decimals).tolist()
    return round_json_floats(rounded, decimals)  # type: ignore[return-value]


def image_files_for_scene(images_dir: Path, timestamps: np.ndarray | None) -> tuple[Path, ...]:
    paths = sorted(
        (path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=natural_key,
    )
    if timestamps is None or not paths:
        return tuple(paths)

    by_stem = {path.stem: path for path in paths}
    timestamp_paths = [by_stem.get(str(int(ts))) for ts in timestamps]
    if all(path is not None for path in timestamp_paths):
        return tuple(path for path in timestamp_paths if path is not None)
    return tuple(paths)


def load_scene(dataset: str, scene_dir: Path) -> Scene | None:
    images_dir = scene_dir / "images"
    traj_path = scene_dir / "trajectory.npy"
    if not images_dir.is_dir() or not traj_path.exists():
        return None

    trajectory = np.load(traj_path)
    if trajectory.ndim != 2 or trajectory.shape[1] < 4:
        logging.warning("%s skipped: trajectory.npy must have shape [T,4+], got %s.", scene_dir, trajectory.shape)
        return None

    ts_path = scene_dir / "timestamps.npy"
    timestamps = np.load(ts_path).astype(np.int64) if ts_path.exists() else None
    if timestamps is not None and len(timestamps) != len(trajectory):
        logging.warning("%s: timestamps length does not match trajectory; falling back to frame order.", scene_dir)
        timestamps = None

    image_paths = image_files_for_scene(images_dir, timestamps)
    usable = min(len(image_paths), len(trajectory))
    if usable < 2:
        logging.warning("%s skipped: too few image/trajectory frames.", scene_dir)
        return None
    if usable != len(image_paths) or usable != len(trajectory):
        logging.warning(
            "%s: trimming to %d aligned frames (images=%d trajectory=%d).",
            scene_dir,
            usable,
            len(image_paths),
            len(trajectory),
        )
        image_paths = image_paths[:usable]
        trajectory = trajectory[:usable]
        if timestamps is not None:
            timestamps = timestamps[:usable]

    return Scene(
        dataset=dataset,
        scene_dir=scene_dir,
        images_dir=images_dir,
        image_paths=tuple(image_paths),
        trajectory=trajectory.astype(np.float32, copy=False),
        timestamps=timestamps,
    )


def discover_scenes(dataset: str, root: Path, max_scenes: int | None = None) -> list[Scene]:
    if not root.exists():
        logging.warning("%s source skipped: root does not exist: %s", dataset, root)
        return []

    traj_paths = sorted(root.rglob("trajectory.npy"), key=natural_key)
    scenes: list[Scene] = []
    for traj_path in traj_paths:
        scene = load_scene(dataset, traj_path.parent)
        if scene is None:
            continue
        scenes.append(scene)
        if max_scenes is not None and len(scenes) >= int(max_scenes):
            break
    return scenes


def make_records_for_scene(scene: Scene, args: argparse.Namespace) -> list[dict[str, object]]:
    clip_frames = int(args.clip_frames)
    stride_frames = int(args.stride_frames)
    traj_points = int(args.traj_points)
    threshold_rad = math.radians(float(args.turn_threshold_deg))

    if len(scene.image_paths) < clip_frames:
        return []

    records: list[dict[str, object]] = []
    sample_index = 0
    max_start = len(scene.image_paths) - clip_frames
    for start in range(0, max_start + 1, stride_frames):
        end = start + clip_frames - 1
        dense_poses = sample_dense_poses(scene, start=start, end=end, traj_points=traj_points)
        dense_local_pose = poses_to_local(dense_poses, origin_pose=dense_poses[0])
        frame_poses = scene.trajectory[start : start + clip_frames]
        frame_local_pose = poses_to_local(frame_poses, origin_pose=frame_poses[0])

        coarse_goal = dense_local_pose[-1, :2]
        if float(np.linalg.norm(coarse_goal)) < float(args.min_distance):
            continue

        route_anchors = sample_anchors(dense_local_pose[:, :2], int(args.num_anchors)).astype(np.float32)
        label = turn_label(coarse_goal, threshold_rad)
        scene_key = sanitize_name(scene.scene_dir.name)
        sample_id = f"{scene.dataset}_{scene_key}_{sample_index:06d}"
        record: dict[str, object] = {
            "schema_version": 2,
            "storage": "extracted",
            "dataset": scene.dataset,
            "sample_id": sample_id,
            "scene_name": scene.scene_dir.name,
            "scene_dir": str(scene.scene_dir),
            "images_dir": str(scene.images_dir),
            "start_frame_index": int(start),
            "end_frame_index": int(end),
            "frame_indices": list(range(start, start + clip_frames)),
            "frame_filenames": [scene.image_paths[idx].name for idx in range(start, start + clip_frames)],
            "traj_local": json_float_array(dense_local_pose[:, :2]),
            "traj_local_pose": json_float_array(dense_local_pose),
            "clip_frame_poses": json_float_array(frame_local_pose),
            "coarse_goal": json_float_array(coarse_goal),
            "route_anchors": json_float_array(route_anchors),
            "turn_label": int(label),
            "turn_label_name": TURN_NAMES[label],
            "clip_frames": clip_frames,
            "traj_points": traj_points,
            "num_anchors": int(args.num_anchors),
            "trajectory_columns": ["x", "y", "z", "yaw"],
            "traj_local_pose_columns": ["x", "y", "yaw"],
        }
        if scene.timestamps is not None:
            record["frame_timestamps_ns"] = [int(scene.timestamps[idx]) for idx in range(start, start + clip_frames)]
        records.append(record)
        sample_index += 1
        if args.max_samples_per_scene is not None and sample_index >= int(args.max_samples_per_scene):
            break
    return records


def assign_split(scenes: Sequence[Scene], val_ratio: float, test_ratio: float, seed: int) -> dict[Path, str]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1.")
    rng = random.Random(seed)
    shuffled = list(scenes)
    rng.shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_ratio))
    n_val = int(round(len(shuffled) * val_ratio))
    test_set = {scene.scene_dir for scene in shuffled[:n_test]}
    val_set = {scene.scene_dir for scene in shuffled[n_test : n_test + n_val]}
    return {
        scene.scene_dir: "test" if scene.scene_dir in test_set else "val" if scene.scene_dir in val_set else "train"
        for scene in scenes
    }


def write_jsonl(path: Path, records: Iterable[dict[str, object]], overwrite: bool) -> int:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists. Pass --overwrite to replace it.")
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":")) + "\n")
            count += 1
    return count


def main() -> None:
    setup_logging()
    args = parse_args()
    if args.clip_frames <= 1 or args.stride_frames <= 0:
        raise ValueError("--clip-frames must be > 1 and --stride-frames must be > 0.")
    if args.traj_points < 2 or args.num_anchors <= 0:
        raise ValueError("--traj-points must be >= 2 and --num-anchors must be > 0.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}
    sources = parse_sources(args.source)

    for source_index, (dataset, root) in enumerate(sources):
        scenes = discover_scenes(dataset, root, max_scenes=args.max_scenes_per_source)
        if not scenes:
            logging.warning("%s: no usable scenes found under %s.", dataset, root)
            continue
        split_for_scene = assign_split(
            scenes,
            val_ratio=float(args.val_ratio),
            test_ratio=float(args.test_ratio),
            seed=int(args.seed) + source_index,
        )
        logging.info("%s: building records from %d scene(s).", dataset, len(scenes))
        for scene in tqdm(scenes, desc=f"{dataset} scenes", unit="scene"):
            try:
                records = make_records_for_scene(scene, args)
            except Exception as exc:
                logging.warning("%s skipped: %s", scene.scene_dir, exc)
                continue
            split_records[split_for_scene[scene.scene_dir]].extend(records)

    for split, records in split_records.items():
        if split == "test" and not records:
            continue
        out_path = args.out_dir / f"{args.index_prefix}_{split}_index.jsonl"
        count = write_jsonl(out_path, records, overwrite=bool(args.overwrite))
        logging.info("Wrote %d %s samples to %s.", count, split, out_path)


if __name__ == "__main__":
    main()
