#!/usr/bin/env python3
"""Preprocess GS2 fisheye image/velocity lists into extracted scene folders.

GS2 stores left/right fisheye image lists and matching robot velocity lists:
  dataset_L_<building>pre_<seq>.txt
  dataset_refres_<building>pre_<seq>.txt

This script uses only the left camera list, reprojects each fisheye image into
a front rectilinear perspective image, integrates [linear, angular] velocities
at 3 FPS into an xy-yaw trajectory, and writes the same extracted layout used
by ``build_dataset.py``:

  <out-root>/<scene>/
    images/
      000000.jpg
      ...
    trajectory.npy   # [T, 4] columns: x, y, z, yaw
    timestamps.npy   # synthetic timestamps at --fps
    metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pickle
import re
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
from tqdm import tqdm


LIST_RE = re.compile(r"^dataset_L_(?P<key>.+)\.txt$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class GSSequence:
    key: str
    left_list: Path
    velocity_list: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GS2 left fisheye images into extracted scenes.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/wangdzh/data/gs2_withres"),
        help="GS2 root containing dataset_L_*.txt and dataset_refres_*.txt files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/wangdzh/data/gs2_withres/gs_extracted"),
        help="Output extracted scene root.",
    )
    parser.add_argument("--fps", type=float, default=3.0, help="Frame interval used for velocity integration.")
    parser.add_argument("--output-size", type=int, default=256, help="Rectilinear output image size.")
    parser.add_argument("--front-fov-deg", type=float, default=90.0, help="Rectilinear front camera horizontal FOV.")
    parser.add_argument("--fisheye-fov-deg", type=float, default=180.0, help="Input fisheye circular FOV.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-frames-per-sequence", type=int, default=None)
    parser.add_argument("--skip-flipped", action="store_true", help="Skip sequences whose id ends with F.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Parallel sequence workers. Use 1 for deterministic single-process preprocessing.",
    )
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


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return cleaned or "scene"


def discover_sequences(root: Path, skip_flipped: bool = False) -> list[GSSequence]:
    sequences: list[GSSequence] = []
    for left_list in sorted(root.glob("dataset_L_*.txt"), key=natural_key):
        if left_list.name.endswith("~"):
            continue
        match = LIST_RE.match(left_list.name)
        if match is None:
            continue
        key = match.group("key")
        if skip_flipped and key.endswith("F"):
            continue
        velocity_list = root / f"dataset_refres_{key}.txt"
        if not velocity_list.exists():
            logging.warning("%s skipped: missing %s.", left_list.name, velocity_list.name)
            continue
        sequences.append(GSSequence(key=key, left_list=left_list, velocity_list=velocity_list))
    return sequences


def read_list_paths(list_path: Path) -> list[str]:
    paths: list[str] = []
    with list_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                paths.append(parts[0])
    return paths


def load_velocity(path: Path) -> tuple[float, float]:
    with path.open("rb") as f:
        value = pickle.load(f, encoding="latin1")
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Velocity pickle must contain [linear, angular], got shape {arr.shape}: {path}")
    return float(arr[0]), float(arr[1])


def load_velocities(root: Path, relative_paths: Sequence[str]) -> np.ndarray:
    velocities = np.empty((len(relative_paths), 2), dtype=np.float32)
    for idx, rel_path in enumerate(relative_paths):
        velocities[idx] = load_velocity(root / rel_path)
    return velocities


def integrate_unicycle(velocities: np.ndarray, fps: float) -> np.ndarray:
    if fps <= 0:
        raise ValueError("--fps must be > 0.")
    dt = 1.0 / float(fps)
    trajectory = np.zeros((len(velocities), 4), dtype=np.float32)
    x = y = yaw = 0.0
    for idx in range(1, len(velocities)):
        v, w = float(velocities[idx - 1, 0]), float(velocities[idx - 1, 1])
        if abs(w) < 1e-6:
            x += v * dt * math.cos(yaw)
            y += v * dt * math.sin(yaw)
        else:
            next_yaw = yaw + w * dt
            radius = v / w
            x += radius * (math.sin(next_yaw) - math.sin(yaw))
            y -= radius * (math.cos(next_yaw) - math.cos(yaw))
            yaw = next_yaw
        yaw = float((yaw + math.pi) % (2.0 * math.pi) - math.pi)
        trajectory[idx] = (x, y, 0.0, yaw)
    return trajectory


def build_rectilinear_maps(
    in_width: int,
    in_height: int,
    out_size: int,
    front_fov_deg: float,
    fisheye_fov_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    if out_size <= 0:
        raise ValueError("--output-size must be > 0.")
    if not (0.0 < front_fov_deg < 180.0):
        raise ValueError("--front-fov-deg must be in (0, 180).")
    if not (0.0 < fisheye_fov_deg <= 360.0):
        raise ValueError("--fisheye-fov-deg must be in (0, 360].")

    cx = (in_width - 1) * 0.5
    cy = (in_height - 1) * 0.5
    radius = min(in_width, in_height) * 0.5
    fish_f = radius / math.radians(fisheye_fov_deg * 0.5)
    rect_f = (out_size * 0.5) / math.tan(math.radians(front_fov_deg * 0.5))

    yy, xx = np.meshgrid(np.arange(out_size, dtype=np.float32), np.arange(out_size, dtype=np.float32), indexing="ij")
    x = (xx - (out_size - 1) * 0.5) / rect_f
    y = (yy - (out_size - 1) * 0.5) / rect_f
    rho = np.sqrt(x * x + y * y)
    theta = np.arctan(rho)
    phi = np.arctan2(y, x)

    r = fish_f * theta
    map_x = (cx + r * np.cos(phi)).astype(np.float32)
    map_y = (cy + r * np.sin(phi)).astype(np.float32)
    return map_x, map_y


def reproject_fisheye_to_front_bgr(image_bgr: np.ndarray, maps: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    return cv2.remap(
        image_bgr,
        maps[0],
        maps[1],
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def write_sequence(seq: GSSequence, args: argparse.Namespace) -> int:
    cv2.setNumThreads(0)
    root = args.dataset_root
    image_rel_paths = read_list_paths(seq.left_list)
    velocity_rel_paths = read_list_paths(seq.velocity_list)
    n = min(len(image_rel_paths), len(velocity_rel_paths))
    if args.max_frames_per_sequence is not None:
        n = min(n, int(args.max_frames_per_sequence))
    if n < 2:
        logging.warning("%s skipped: too few frames.", seq.key)
        return 0
    if len(image_rel_paths) != len(velocity_rel_paths):
        logging.warning(
            "%s: trimming to %d aligned rows (images=%d velocities=%d).",
            seq.key,
            n,
            len(image_rel_paths),
            len(velocity_rel_paths),
        )
    image_rel_paths = image_rel_paths[:n]
    velocity_rel_paths = velocity_rel_paths[:n]

    scene_name = f"gs_{sanitize_name(seq.key)}"
    scene_dir = args.out_root / scene_name
    if scene_dir.exists():
        if not args.overwrite:
            logging.info("%s skipped: %s already exists. Use --overwrite to replace it.", seq.key, scene_dir)
            return 0
        shutil.rmtree(scene_dir)
    images_dir = scene_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    first_image_path = root / image_rel_paths[0]
    first_image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise FileNotFoundError(f"Missing or unreadable GS2 image: {first_image_path}")
    in_height, in_width = first_image.shape[:2]
    maps = build_rectilinear_maps(
        in_width=in_width,
        in_height=in_height,
        out_size=int(args.output_size),
        front_fov_deg=float(args.front_fov_deg),
        fisheye_fov_deg=float(args.fisheye_fov_deg),
    )
    jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    for idx, rel_path in enumerate(image_rel_paths):
        src_path = root / rel_path
        image = cv2.imread(str(src_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Missing or unreadable GS2 image: {src_path}")
        rect = reproject_fisheye_to_front_bgr(image, maps)
        ok = cv2.imwrite(str(images_dir / f"{idx:06d}.jpg"), rect, jpeg_params)
        if not ok:
            raise RuntimeError(f"Failed to write rectilinear image for {src_path}")

    velocities = load_velocities(root, velocity_rel_paths)
    trajectory = integrate_unicycle(velocities, fps=float(args.fps))
    timestamps = np.rint(np.arange(n, dtype=np.float64) * (1e9 / float(args.fps))).astype(np.int64)
    np.save(scene_dir / "trajectory.npy", trajectory)
    np.save(scene_dir / "timestamps.npy", timestamps)
    np.save(scene_dir / "velocities.npy", velocities)

    metadata = {
        "schema_version": 1,
        "dataset": "gs",
        "source_root": str(root),
        "scene_name": scene_name,
        "sequence_key": seq.key,
        "left_list": str(seq.left_list),
        "velocity_list": str(seq.velocity_list),
        "camera": "left",
        "projection": "front_rectilinear",
        "front_fov_deg": float(args.front_fov_deg),
        "fisheye_fov_deg": float(args.fisheye_fov_deg),
        "fps": float(args.fps),
        "num_frames": int(n),
        "trajectory_columns": ["x", "y", "z", "yaw"],
        "velocity_columns": ["linear", "angular"],
        "image_timestamp_unit": "ns",
        "timestamp_source": "synthetic_uniform_fps",
    }
    with (scene_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return int(n)


def worker_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dataset_root": args.dataset_root,
        "out_root": args.out_root,
        "fps": args.fps,
        "output_size": args.output_size,
        "front_fov_deg": args.front_fov_deg,
        "fisheye_fov_deg": args.fisheye_fov_deg,
        "jpeg_quality": args.jpeg_quality,
        "max_frames_per_sequence": args.max_frames_per_sequence,
        "overwrite": args.overwrite,
    }


def write_sequence_worker(seq: GSSequence, args_dict: dict[str, Any]) -> tuple[str, int, str | None]:
    args = argparse.Namespace(**args_dict)
    try:
        return seq.key, write_sequence(seq, args), None
    except Exception as exc:
        return seq.key, 0, str(exc)


def main() -> None:
    setup_logging()
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.out_root = args.out_root.expanduser().resolve()
    if not args.dataset_root.exists():
        raise FileNotFoundError(f"--dataset-root does not exist: {args.dataset_root}")
    args.out_root.mkdir(parents=True, exist_ok=True)

    sequences = discover_sequences(args.dataset_root, skip_flipped=bool(args.skip_flipped))
    if args.max_sequences is not None:
        sequences = sequences[: int(args.max_sequences)]
    if not sequences:
        raise FileNotFoundError(f"No GS2 left-camera dataset_L_*.txt files found in {args.dataset_root}")

    total_frames = 0
    written = 0
    num_workers = max(1, int(args.num_workers))
    if num_workers == 1:
        for seq in tqdm(sequences, desc="GS2 sequences", unit="seq"):
            try:
                frames = write_sequence(seq, args)
            except Exception as exc:
                logging.warning("%s skipped: %s", seq.key, exc)
                continue
            if frames > 0:
                written += 1
                total_frames += frames
    else:
        args_dict = worker_args(args)
        max_workers = min(num_workers, len(sequences), os.cpu_count() or num_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(write_sequence_worker, seq, args_dict) for seq in sequences]
            for future in tqdm(as_completed(futures), total=len(futures), desc="GS2 sequences", unit="seq"):
                key, frames, error = future.result()
                if error is not None:
                    logging.warning("%s skipped: %s", key, error)
                    continue
                if frames > 0:
                    written += 1
                    total_frames += frames

    logging.info("Wrote %d GS2 scene(s), %d frame(s) to %s.", written, total_frames, args.out_root)


if __name__ == "__main__":
    main()
