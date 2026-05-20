#!/usr/bin/env python3
"""Build Stage-1 SCAND JSONL indexes from ROS1 bag files."""

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
from rosbags.highlevel import AnyReader
from tqdm import tqdm


IMAGE_TOPICS = (
    "/image_raw/compressed",
    "/camera/rgb/image_raw/compressed",
    "/left/image_color/compressed",
)
ODOM_TOPICS = ("/odom", "/jackal_velocity_controller/odom")
TURN_NAMES = ("left", "straight", "right")


@dataclass(frozen=True)
class BagSignals:
    image_topic: str
    odom_topic: str
    image_ts: np.ndarray
    odom_ts: np.ndarray
    odom_xy: np.ndarray
    odom_yaw: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare SCAND Stage-1 route-clip indexes.")
    parser.add_argument("--dataset-root", type=Path, required=True, help="SCAND root containing ROS1 .bag files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for scand_stage1_*_index.jsonl.")
    parser.add_argument("--clip-frames", type=int, default=16, help="Video frames sampled per route clip.")
    parser.add_argument("--fps", type=float, default=8.0, help="Frame sampling FPS inside each clip.")
    parser.add_argument("--clip-duration", type=float, default=0.0, help="Override clip duration in seconds.")
    parser.add_argument("--stride-seconds", type=float, default=1.0, help="Start-time stride between route clips.")
    parser.add_argument("--traj-points", type=int, default=32, help="Trajectory points stored per sample.")
    parser.add_argument("--num-anchors", type=int, default=4, help="Uniform route anchor count.")
    parser.add_argument("--min-distance", type=float, default=0.25, help="Skip clips moving less than this many meters.")
    parser.add_argument("--turn-threshold-deg", type=float, default=15.0, help="Goal bearing threshold for turn labels.")
    parser.add_argument("--max-frame-dt", type=float, default=0.25, help="Max image sync error in seconds.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Bag-level validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=0.0, help="Bag-level test split ratio.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-bags", type=int, default=None)
    parser.add_argument("--max-samples-per-bag", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def choose_topic(available: set[str], candidates: Sequence[str]) -> str | None:
    for topic in candidates:
        if topic in available:
            return topic
    return None


def quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def read_bag_signals(bag_path: Path) -> BagSignals | None:
    with AnyReader([bag_path]) as reader:
        topics = {conn.topic for conn in reader.connections}
        image_topic = choose_topic(topics, IMAGE_TOPICS)
        odom_topic = choose_topic(topics, ODOM_TOPICS)
        if image_topic is None or odom_topic is None:
            logging.warning("%s skipped: missing image or odom topic.", bag_path.name)
            return None

        image_conn = [conn for conn in reader.connections if conn.topic == image_topic]
        odom_conn = [conn for conn in reader.connections if conn.topic == odom_topic]
        image_ts: list[int] = []
        odom_ts: list[int] = []
        xy: list[tuple[float, float]] = []
        yaw: list[float] = []

        for conn, ts, raw in reader.messages(connections=[*image_conn, *odom_conn]):
            if conn.topic == image_topic:
                image_ts.append(int(ts))
                continue

            msg = reader.deserialize(raw, conn.msgtype)
            pos = msg.pose.pose.position
            ori = msg.pose.pose.orientation
            odom_ts.append(int(ts))
            xy.append((float(pos.x), float(pos.y)))
            yaw.append(quat_to_yaw(float(ori.x), float(ori.y), float(ori.z), float(ori.w)))

    if len(image_ts) < 2 or len(odom_ts) < 2:
        logging.warning("%s skipped: too few image/odom messages.", bag_path.name)
        return None

    return BagSignals(
        image_topic=image_topic,
        odom_topic=odom_topic,
        image_ts=np.asarray(image_ts, dtype=np.int64),
        odom_ts=np.asarray(odom_ts, dtype=np.int64),
        odom_xy=np.asarray(xy, dtype=np.float64),
        odom_yaw=np.unwrap(np.asarray(yaw, dtype=np.float64)),
    )


def nearest_image_times(
    image_ts: np.ndarray,
    target_ts: np.ndarray,
    max_dt_ns: int,
) -> list[int] | None:
    indices = np.searchsorted(image_ts, target_ts)
    chosen: list[int] = []
    for target, idx in zip(target_ts, indices):
        best_idx = min(max(int(idx), 0), len(image_ts) - 1)
        if idx > 0 and abs(int(image_ts[idx - 1]) - int(target)) < abs(int(image_ts[best_idx]) - int(target)):
            best_idx = int(idx - 1)
        if abs(int(image_ts[best_idx]) - int(target)) > max_dt_ns:
            return None
        chosen.append(int(image_ts[best_idx]))
    return chosen


def interpolate_xy_yaw(signals: BagSignals, query_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    odom_t = signals.odom_ts.astype(np.float64)
    x = np.interp(query_ts, odom_t, signals.odom_xy[:, 0])
    y = np.interp(query_ts, odom_t, signals.odom_xy[:, 1])
    yaw = np.interp(query_ts, odom_t, signals.odom_yaw)
    return np.stack((x, y), axis=1), yaw


def to_start_local(xy: np.ndarray, origin: np.ndarray, yaw: float) -> np.ndarray:
    delta = xy - origin
    c, s = math.cos(yaw), math.sin(yaw)
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    return delta @ rot


def turn_label(goal_xy: np.ndarray, threshold_rad: float) -> int:
    angle = math.atan2(float(goal_xy[1]), float(goal_xy[0]))
    if angle > threshold_rad:
        return 0
    if angle < -threshold_rad:
        return 2
    return 1


def sample_anchors(traj_local: np.ndarray, num_anchors: int) -> np.ndarray:
    if len(traj_local) == 1:
        return np.repeat(traj_local, num_anchors, axis=0)
    indices = np.linspace(1, len(traj_local) - 1, num_anchors)
    return traj_local[np.rint(indices).astype(np.int64)]


def make_records_for_bag(
    bag_path: Path,
    signals: BagSignals,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    clip_duration = float(args.clip_duration)
    if clip_duration <= 0:
        clip_duration = (int(args.clip_frames) - 1) / float(args.fps)

    start_ns = int(max(signals.image_ts[0], signals.odom_ts[0]))
    end_ns = int(min(signals.image_ts[-1], signals.odom_ts[-1]) - clip_duration * 1e9)
    if end_ns <= start_ns:
        return []

    stride_ns = int(float(args.stride_seconds) * 1e9)
    clip_ns = int(clip_duration * 1e9)
    max_dt_ns = int(float(args.max_frame_dt) * 1e9)
    threshold_rad = math.radians(float(args.turn_threshold_deg))

    records: list[dict[str, object]] = []
    sample_index = 0
    for clip_start in range(start_ns, end_ns + 1, stride_ns):
        clip_end = clip_start + clip_ns
        frame_targets = np.linspace(clip_start, clip_end, int(args.clip_frames), dtype=np.int64)
        frame_ts = nearest_image_times(signals.image_ts, frame_targets, max_dt_ns)
        if frame_ts is None:
            continue

        traj_targets = np.linspace(clip_start, clip_end, int(args.traj_points), dtype=np.float64)
        xy, yaws = interpolate_xy_yaw(signals, traj_targets)
        traj_local = to_start_local(xy, xy[0], float(yaws[0])).astype(np.float32)
        coarse_goal = traj_local[-1]
        if float(np.linalg.norm(coarse_goal)) < float(args.min_distance):
            continue

        anchors = sample_anchors(traj_local, int(args.num_anchors)).astype(np.float32)
        label = turn_label(coarse_goal, threshold_rad)
        sample_id = f"{bag_path.stem}_{sample_index:06d}"
        records.append(
            {
                "schema_version": 1,
                "dataset": "scand",
                "sample_id": sample_id,
                "bag_name": bag_path.stem,
                "bag_path": str(bag_path),
                "image_topic": signals.image_topic,
                "odom_topic": signals.odom_topic,
                "start_timestamp_ns": int(clip_start),
                "end_timestamp_ns": int(clip_end),
                "frame_timestamps_ns": frame_ts,
                "traj_local": traj_local.tolist(),
                "coarse_goal": coarse_goal.astype(float).tolist(),
                "route_anchors": anchors.astype(float).tolist(),
                "turn_label": int(label),
                "turn_label_name": TURN_NAMES[label],
                "clip_frames": int(args.clip_frames),
                "traj_points": int(args.traj_points),
                "num_anchors": int(args.num_anchors),
            }
        )
        sample_index += 1
        if args.max_samples_per_bag is not None and sample_index >= int(args.max_samples_per_bag):
            break
    return records


def assign_split(bag_paths: Sequence[Path], val_ratio: float, test_ratio: float, seed: int) -> dict[Path, str]:
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1:
        raise ValueError("val_ratio and test_ratio must be >= 0 and sum to < 1.")
    rng = random.Random(seed)
    shuffled = list(bag_paths)
    rng.shuffle(shuffled)
    n_test = int(round(len(shuffled) * test_ratio))
    n_val = int(round(len(shuffled) * val_ratio))
    test_set = set(shuffled[:n_test])
    val_set = set(shuffled[n_test : n_test + n_val])
    return {path: "test" if path in test_set else "val" if path in val_set else "train" for path in bag_paths}


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
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root does not exist: {args.dataset_root}")
    if args.clip_frames <= 1 or args.fps <= 0 or args.stride_seconds <= 0:
        raise ValueError("clip_frames must be > 1, fps and stride_seconds must be > 0.")
    if args.traj_points < 2 or args.num_anchors <= 0:
        raise ValueError("traj_points must be >= 2 and num_anchors must be > 0.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    bag_paths = sorted(args.dataset_root.rglob("*.bag"), key=natural_key)
    if args.max_bags is not None:
        bag_paths = bag_paths[: int(args.max_bags)]
    if not bag_paths:
        raise FileNotFoundError(f"No .bag files found under {args.dataset_root}")

    split_for_bag = assign_split(bag_paths, float(args.val_ratio), float(args.test_ratio), int(args.seed))
    split_records: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}

    for bag_path in tqdm(bag_paths, desc="SCAND bags", unit="bag"):
        try:
            signals = read_bag_signals(bag_path)
            if signals is None:
                continue
            records = make_records_for_bag(bag_path, signals, args)
        except Exception as exc:
            logging.warning("%s skipped: %s", bag_path.name, exc)
            continue
        split_records[split_for_bag[bag_path]].extend(records)

    for split, records in split_records.items():
        if split == "test" and not records:
            continue
        out_path = args.out_dir / f"scand_stage1_{split}_index.jsonl"
        count = write_jsonl(out_path, records, overwrite=bool(args.overwrite))
        logging.info("Wrote %d %s samples to %s.", count, split, out_path)


if __name__ == "__main__":
    main()
