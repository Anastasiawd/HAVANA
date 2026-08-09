"""RGB-D InternData samples for offline or frozen-online Reason1 plans."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import OrderedDict
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from decord import VideoReader, cpu
except ImportError:  # Allows contract/collate tests without video dependencies.
    VideoReader = None
    cpu = None

from fast_system.model.plan_encoder import LEGACY_PLAN_SCHEMA_VERSION, PLAN_SCHEMA_VERSION


HAVANA_ROOT = Path(__file__).resolve().parents[2]
COSMOS_ROOT = HAVANA_ROOT / "cosmos-predict2.5"


def _torch_load(path: Path) -> object:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _pose_matrix(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32)
    if pose.size != 16:
        raise ValueError(f"Pose must contain 16 values, got {pose.shape}.")
    return pose.reshape(4, 4)


def _local_xy(poses: Sequence[np.ndarray], reference: np.ndarray) -> np.ndarray:
    """Re-root world poses as (x forward, y left) in the reference frame."""
    reference_inv = np.linalg.inv(reference)
    result = []
    for pose in poses:
        relative = reference_inv @ pose
        result.append([float(relative[0, 3]), float(relative[1, 3])])
    return np.asarray(result, dtype=np.float32)


def _wrap_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _local_xyt(poses: Sequence[np.ndarray], reference: np.ndarray) -> np.ndarray:
    """Re-root world poses as absolute ``(x forward, y left, yaw)`` waypoints."""

    reference_inv = np.linalg.inv(reference)
    result = []
    for pose in poses:
        relative = reference_inv @ pose
        result.append(
            [
                float(relative[0, 3]),
                float(relative[1, 3]),
                float(_wrap_to_pi(math.atan2(float(relative[1, 0]), float(relative[0, 0])))),
            ]
        )
    return np.asarray(result, dtype=np.float32)


def _resize_rgb(frame: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
    if image.size != size:
        image = image.resize(size, getattr(Image, "Resampling", Image).BILINEAR)
    value = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(value).permute(2, 0, 1).contiguous()


def _resize_depth(depth_m: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    image = Image.fromarray(np.asarray(depth_m, dtype=np.float32))
    if image.size != size:
        image = image.resize(size, getattr(Image, "Resampling", Image).NEAREST)
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).contiguous()


def _align_rgb_to_depth(frame: np.ndarray, depth_shape: tuple[int, int]) -> np.ndarray:
    """Remove the small codec padding around RGB so RGB and raw depth align."""

    rgb = np.asarray(frame)
    target_h, target_w = depth_shape
    height, width = rgb.shape[:2]
    if (height, width) == (target_h, target_w):
        return rgb
    if height < target_h or width < target_w:
        raise ValueError(
            f"RGB frame {height}x{width} is smaller than depth frame {target_h}x{target_w}."
        )
    top = (height - target_h) // 2
    left = (width - target_w) // 2
    return np.ascontiguousarray(rgb[top : top + target_h, left : left + target_w])


def _depth_image_path(video_path: Path, frame_index: int) -> Path:
    chunk_dir = video_path.parent.parent
    return chunk_dir / "observation.images.depth" / f"{video_path.stem}_{int(frame_index):03d}.png"


def _load_depth_m(path: Path, min_depth_m: float, max_depth_m: float) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Required InternData depth frame is missing: {path}")
    raw = np.asarray(Image.open(path))
    if raw.ndim == 3:
        raw = raw[..., 0]
    depth = raw.astype(np.float32) / 10000.0
    valid = np.isfinite(depth) & (depth >= float(min_depth_m)) & (depth <= float(max_depth_m))
    return np.where(valid, depth, 0.0).astype(np.float32, copy=False)


def _parquet_num_rows(path: str | Path) -> int:
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(str(path)).metadata.num_rows)
    except ImportError:
        return int(len(pd.read_parquet(path, columns=[])))


class _ParquetCache:
    """Bounded LRU of per-episode pose frames shared by dataset workers."""

    def __init__(self, capacity: int = 64) -> None:
        self._capacity = max(1, int(capacity))
        self._cache: OrderedDict[str, pd.DataFrame] = OrderedDict()

    def get(self, key: str) -> pd.DataFrame:
        value = self._cache.pop(key, None)
        if value is None:
            value = pd.read_parquet(key)
            self._cache[key] = value
            if len(self._cache) > self._capacity:
                self._cache.popitem(last=False)
        else:
            self._cache[key] = value
        return value


_PARQUET_CACHE = _ParquetCache()


class InternDataFastDataset(Dataset):
    """Fast-policy RGB-D/action samples from a plan JSONL manifest.

    Every FastNav offset, observation-context frame, and action target is taken
    at the native source frame rate. ``target_fps`` is only the Reason1 plan
    sampling rate used by the native temporal dataset.
    """

    def __init__(
        self,
        manifest: str | Path,
        obs_image_size: int = 256,
        context_size: int = 3,
        action_horizon: int = 16,
        async_offsets: Sequence[int] = tuple(range(1, 13)),
        target_fps: float = 10.0,
        obs_context_dt_frames: int = 1,
        pose_key: str = "action",
        min_depth_m: float = 0.1,
        max_depth_m: float = 5.0,
        require_full_horizon: bool = True,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.manifest_path = Path(manifest).expanduser().resolve()
        with self.manifest_path.open("r", encoding="utf-8") as stream:
            self.records = [json.loads(line) for line in stream if line.strip()]
        self.size = (int(obs_image_size), int(obs_image_size))
        self.context_size = int(context_size)
        self.action_horizon = int(action_horizon)
        self.async_offsets = tuple(int(value) for value in async_offsets)
        self.target_fps = float(target_fps)
        self.obs_context_dt_frames = max(1, int(obs_context_dt_frames))
        self.pose_key = str(pose_key)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.require_full_horizon = bool(require_full_horizon)
        if self.require_full_horizon:
            self.records = [record for record in self.records if self._record_has_full_horizon(record)]
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        if not self.records:
            raise ValueError(f"No full-horizon plan records found in {self.manifest_path}.")

    def _record_has_full_horizon(self, record: Mapping[str, Any]) -> bool:
        last = int(record["reference_frame"]) + max(self.async_offsets) + self.action_horizon
        return last < _parquet_num_rows(record["data_path"])

    def __len__(self) -> int:
        return len(self.records)

    def _load_video_and_poses(self, record: Mapping[str, Any]):
        if VideoReader is None:
            raise ImportError("decord is required to read InternData RGB videos.")
        video_path = Path(record["video_path"]).expanduser()
        data_path = Path(record["data_path"]).expanduser()
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=2)
        dataframe = _PARQUET_CACHE.get(str(data_path))
        pose_key = str(record.get("pose_key", self.pose_key))
        if pose_key not in dataframe:
            raise KeyError(f"Pose key {pose_key!r} not found in {data_path}.")
        count = min(len(reader), len(dataframe))
        return reader, dataframe, pose_key, count

    def _build_fast_fields(self, record: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        reader, dataframe, pose_key, total_frames = self._load_video_and_poses(record)
        reference_frame = int(record["reference_frame"])

        video_path = Path(record["video_path"]).expanduser()
        # FastNav runs at the native stream rate: offset=1 means exactly the
        # next recorded frame, regardless of the source FPS.
        current_indices = [reference_frame + offset for offset in self.async_offsets]
        all_obs_indices = {
            min(max(current - step * self.obs_context_dt_frames, 0), total_frames - 1)
            for current in current_indices
            for step in range(self.context_size, -1, -1)
        }
        rgb_batch = reader.get_batch(sorted(all_obs_indices)).asnumpy()
        rgb_cache = {index: frame for index, frame in zip(sorted(all_obs_indices), rgb_batch)}
        depth_cache = {
            current: _load_depth_m(
                _depth_image_path(video_path, current), self.min_depth_m, self.max_depth_m
            )
            for current in current_indices
        }

        obs_groups: list[torch.Tensor] = []
        depth_groups: list[torch.Tensor] = []
        action_groups: list[torch.Tensor] = []
        mask_groups: list[torch.Tensor] = []
        distances: list[torch.Tensor] = []
        for offset in self.async_offsets:
            current = reference_frame + offset
            obs_indices = np.asarray(
                [current - step * self.obs_context_dt_frames for step in range(self.context_size, -1, -1)],
                dtype=np.int64,
            )
            obs_indices = np.clip(obs_indices, 0, total_frames - 1)
            current_depth = depth_cache[current]
            frames = [
                _align_rgb_to_depth(rgb_cache[int(frame)], current_depth.shape)
                for frame in obs_indices
            ]
            obs_groups.append(torch.stack([_resize_rgb(frame, self.size) for frame in frames]))
            depth_groups.append(_resize_depth(current_depth, self.size))

            target_indices = current + np.arange(1, self.action_horizon + 1, dtype=np.int64)
            valid = target_indices < total_frames
            safe_indices = np.clip(target_indices, 0, total_frames - 1)
            reference_pose = _pose_matrix(dataframe.iloc[min(max(current, 0), total_frames - 1)][pose_key])
            target_poses = [_pose_matrix(dataframe.iloc[int(frame)][pose_key]) for frame in safe_indices]
            local_action = _local_xyt(target_poses, reference_pose)
            action_groups.append(torch.from_numpy(local_action))
            mask_groups.append(torch.from_numpy(valid.astype(np.float32)))
            if valid.any():
                distances.append(
                    torch.tensor(float(np.linalg.norm(local_action[np.flatnonzero(valid)[-1], :2])))
                )
            else:
                distances.append(torch.tensor(0.0))
        del reader

        return {
            "obs": torch.stack(obs_groups),
            "depth": torch.stack(depth_groups),
            "action": torch.stack(action_groups),
            "action_mask": torch.stack(mask_groups),
            "distance": torch.stack(distances),
            "async_offsets": torch.tensor(self.async_offsets, dtype=torch.long),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        plan_path = Path(record["plan_path"])
        if not plan_path.is_absolute():
            plan_path = self.manifest_path.parent / plan_path
        plan = _torch_load(plan_path)
        if not isinstance(plan, Mapping) or plan.get("schema_version") not in {
            PLAN_SCHEMA_VERSION,
            LEGACY_PLAN_SCHEMA_VERSION,
        }:
            raise ValueError(f"Invalid plan schema in {plan_path}.")

        fast_fields = self._build_fast_fields(record)
        reference_frame = int(record["reference_frame"])

        return {
            "reason_context": torch.as_tensor(plan["reason_context"]).half(),
            "context_mask": torch.as_tensor(plan["context_mask"]).bool(),
            "traj_latents": torch.as_tensor(plan["traj_latents"]).half(),
            **fast_fields,
            "metadata": {
                "dataset_index": int(record.get("dataset_index", index)),
                "scene_id": record.get("scene_id", ""),
                "episode_index": record.get("episode_index"),
                "subtask_index": record.get("subtask_index"),
                "reference_frame": reference_frame,
                "plan_path": str(plan_path),
            },
        }


def discover_scene_roots(roots: Sequence[Path]) -> list[tuple[str, Path]]:
    """Discover LeRobot scene roots before constructing the native dataset."""

    found: list[tuple[str, Path]] = []
    for dataset_root in roots:
        if (dataset_root / "meta" / "episodes.jsonl").exists():
            found.append((dataset_root.name, dataset_root))
            continue
        info_paths: list[Path] = []
        for pattern in ("*/meta/info.json", "*/*/meta/info.json", "*/*/*/meta/info.json"):
            info_paths.extend(sorted(dataset_root.glob(pattern)))
        for info in info_paths:
            scene = info.parent.parent
            if not (scene / "meta" / "episodes.jsonl").exists():
                continue
            relative = scene.relative_to(dataset_root)
            source = relative.parts[0] if relative.parts else dataset_root.name
            found.append((source, scene))
    unique = {(source, str(path.resolve())): (source, path.resolve()) for source, path in found}
    return list(unique.values())


def stratified_scene_split(
    scenes: Sequence[tuple[str, Path]], seed: int, val_fraction: float
) -> tuple[list[Path], list[Path]]:
    """Make a fixed source-stratified scene split with no overlap."""

    groups: dict[str, list[Path]] = defaultdict(list)
    for source, path in scenes:
        groups[source].append(path)
    rng = random.Random(seed)
    train: list[Path] = []
    val: list[Path] = []
    for source in sorted(groups):
        values = sorted(groups[source])
        rng.shuffle(values)
        count = int(round(len(values) * val_fraction))
        if len(values) > 1:
            count = min(max(count, 1), len(values) - 1)
        val.extend(values[:count])
        train.extend(values[count:])
    if set(train) & set(val):
        raise AssertionError("Scene split overlap detected.")
    return sorted(train), sorted(val)


def _native_dataset_class():
    cosmos_path = str(COSMOS_ROOT)
    if cosmos_path not in sys.path:
        sys.path.insert(0, cosmos_path)
    from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_vln_waypoint import (
        InternDataVLNWaypointDataset,
    )

    return InternDataVLNWaypointDataset


def build_native_intern_dataset(data_cfg: Mapping[str, Any], split: str):
    """Instantiate the deterministic native 17-frame Cosmos dataset."""

    root_values = data_cfg.get("dataset_roots", data_cfg.get("dataset_root"))
    if isinstance(root_values, (str, Path)):
        root_values = [root_values]
    if not root_values:
        raise ValueError("Online plan mode requires intern_data.dataset_roots.")
    roots = [Path(value).expanduser().resolve() for value in root_values]
    scenes = discover_scene_roots(roots)
    if not scenes:
        raise ValueError(f"No InternData scene roots found under {roots}.")
    train_scenes, val_scenes = stratified_scene_split(
        scenes,
        int(data_cfg.get("split_seed", data_cfg.get("seed", 0))),
        float(data_cfg.get("val_fraction", 0.1)),
    )
    selected = train_scenes if split == "train" else val_scenes
    filters = set(str(value) for value in data_cfg.get(f"{split}_scenes", data_cfg.get("scenes", ())))
    if filters:
        selected = [path for path in selected if path.name in filters or str(path) in filters]
    if not selected:
        raise ValueError(f"No scenes selected for online split={split!r}.")
    audit_manifest = Path(str(data_cfg.get("source_audit_manifest", ""))).expanduser().resolve()
    if not audit_manifest.is_file():
        raise FileNotFoundError(
            "Online plan mode requires the native dataset source audit manifest: "
            f"{audit_manifest}"
        )
    dataset_class = _native_dataset_class()
    prompt_fields = tuple(str(value) for value in data_cfg.get("prompt_fields", ("sub_instruction",)))
    dataset = dataset_class(
        dataset_dir=[str(path) for path in selected],
        num_frames=17,
        video_size=(272, 480),
        target_fps=10.0,
        pose_key=str(data_cfg.get("pose_key", "action")),
        waypoint_positions=(8, 12, 16),
        subtask_anchor_position=5,
        temporal_mode=True,
        condition_frame_positions=(0, 1, 2, 3, 4),
        subtask_start_position=5,
        waypoint_reference_position=4,
        post_subtask_tail_frames=4,
        straight_window_stride=8,
        source_audit_manifest=str(audit_manifest),
        source_audit_policy="report_only",
        prompt_fields=prompt_fields,
        structured_prompt=bool(data_cfg.get("structured_prompt", False)),
        horizontal_flip_prob=0.0,
        strict_target_fps=True,
        onset_jitter=0,
        seed=int(data_cfg.get("seed", 0)),
    )
    # Each selected scene is passed as an individual native dataset root.  The
    # native loader consequently reports ``root.name`` as its source id, which
    # is not unique when two sensor suites contain the same Matterport scene.
    # Keep the true source family alongside the native dataset for diagnostics
    # and for the online wrapper's metadata.
    source_by_scene = {str(path.resolve()): source for source, path in scenes}
    dataset.havana_source_by_scene = {
        str(path.resolve()): source_by_scene[str(path.resolve())]
        for path in selected
    }
    return dataset


class InternDataOnlinePlanDataset(InternDataFastDataset):
    """Native InternData samples with RGB/actions and inputs for online Reason1."""

    def __init__(self, data_cfg: Mapping[str, Any], split: str) -> None:
        Dataset.__init__(self)
        self.native_dataset = build_native_intern_dataset(data_cfg, split)
        image_size = int(data_cfg.get("obs_image_size", 256))
        self.size = (image_size, image_size)
        self.context_size = int(data_cfg.get("context_size", 3))
        self.action_horizon = int(data_cfg.get("action_horizon", 16))
        self.async_offsets = tuple(int(value) for value in data_cfg.get("fast_offsets", tuple(range(1, 13))))
        self.target_fps = float(data_cfg.get("target_fps", 10.0))
        self.obs_context_dt_frames = max(1, int(data_cfg.get("obs_context_dt_frames", 1)))
        self.pose_key = str(data_cfg.get("pose_key", "action"))
        self.min_depth_m = float(data_cfg.get("min_depth_m", 0.1))
        self.max_depth_m = float(data_cfg.get("max_depth_m", 5.0))
        self.require_full_horizon = bool(data_cfg.get("require_full_horizon", True))
        kept_samples = []
        kept_anchors = []
        for sample, anchor in zip(self.native_dataset.samples, self.native_dataset.sample_anchor_frames):
            episode, _ = sample
            # The native Reason1 sample is still 10 FPS, so its frame at
            # position 4 is one Reason1 stride before the position-5 anchor.
            reason_stride = max(1, int(round(float(episode.source_fps or 30.0) / self.target_fps)))
            reference_frame = int(anchor) - reason_stride
            # From that reference onward FastNav consumes every native frame.
            last = reference_frame + max(self.async_offsets) + self.action_horizon
            total_frames = (
                int(episode.length)
                if episode.length is not None
                else _parquet_num_rows(episode.data_path)
            )
            if self.require_full_horizon and last >= total_frames:
                continue

            # Fail fast while constructing the dataset instead of many batches
            # into training. FastNav consumes metric depth at every async tick.
            required_depth = (
                _depth_image_path(episode.video_path, reference_frame + offset)
                for offset in self.async_offsets
            )
            if not all(path.is_file() for path in required_depth):
                continue
            kept_samples.append(sample)
            kept_anchors.append(anchor)
        self.native_dataset.samples = kept_samples
        self.native_dataset.sample_anchor_frames = kept_anchors
        max_samples = data_cfg.get("max_samples")
        native_count = len(self.native_dataset)
        self.sample_count = native_count if max_samples is None else min(native_count, int(max_samples))
        self.source_by_scene = getattr(self.native_dataset, "havana_source_by_scene", {})

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Use the episode object paired with this exact sample.  Looking it up
        # by (source_id, scene_id, episode_index) is unsafe here: when scene
        # roots are passed individually, the native source id collapses to the
        # scene name and collides across sensor suites (e.g. zed vs d435i).
        episode, subtask = self.native_dataset.samples[index]
        load_sample = getattr(self.native_dataset, "_load_sample", None)
        if callable(load_sample):
            # Bypass the native __getitem__ random-retry fallback so an error
            # cannot silently replace this index with a different episode.
            item = load_sample(episode, subtask, index)
        else:  # Lightweight test doubles may expose only __getitem__.
            item = self.native_dataset[index]
        frame_indices = torch.as_tensor(item["frame_indices"]).long()
        record = {
            "video_path": str(episode.video_path),
            "data_path": str(episode.data_path),
            "source_fps": float(episode.source_fps or 30.0),
            "pose_key": self.native_dataset.pose_key,
            "reference_frame": int(frame_indices[4]),
        }
        fast_fields = self._build_fast_fields(record)
        return {
            "reason_video": torch.as_tensor(item["video"]).to(dtype=torch.uint8),
            "instruction": str(item["ai_caption"]),
            **fast_fields,
            "metadata": {
                "dataset_index": int(index),
                "source_id": self.source_by_scene.get(str(episode.root.resolve()), item["source_id"]),
                "scene_id": episode.root.name,
                "episode_index": int(item["episode_index"]),
                "subtask_index": int(item["subtask_index"]),
                "reference_frame": int(frame_indices[4]),
                "source_step": int(item["subtask_anchor_frame"]),
            },
        }


def collate_plan_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable Reason1 sequences and stack all fixed-shape fields."""
    if not items:
        raise ValueError("Cannot collate an empty plan batch.")
    max_length = max(int(item["reason_context"].shape[0]) for item in items)
    dim = int(items[0]["reason_context"].shape[-1])
    context = torch.zeros(len(items), max_length, dim, dtype=torch.float16)
    mask = torch.zeros(len(items), max_length, dtype=torch.bool)
    for row, item in enumerate(items):
        length = int(item["reason_context"].shape[0])
        context[row, :length] = item["reason_context"]
        mask[row, :length] = item["context_mask"]
    batch = {"reason_context": context, "context_mask": mask}
    for key in ("traj_latents", "obs", "depth", "action", "action_mask", "distance", "async_offsets"):
        batch[key] = torch.stack([item[key] for item in items])
    batch["metadata"] = [item["metadata"] for item in items]
    return batch


def collate_online_plan_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack training tensors while retaining captions for online Reason1."""

    if not items:
        raise ValueError("Cannot collate an empty online plan batch.")
    batch: dict[str, Any] = {
        "instruction": [str(item["instruction"]) for item in items],
        "metadata": [item["metadata"] for item in items],
    }
    for key in ("reason_video", "obs", "depth", "action", "action_mask", "distance", "async_offsets"):
        batch[key] = torch.stack([item[key] for item in items])
    return batch


def build_intern_plan_dataloader(
    data_cfg: Mapping[str, Any],
    loader_cfg: Mapping[str, Any],
    split: str,
) -> DataLoader:
    manifest = data_cfg[f"{split}_manifest"]
    dataset = InternDataFastDataset(
        manifest=manifest,
        obs_image_size=int(data_cfg.get("obs_image_size", 256)),
        context_size=int(data_cfg.get("context_size", 3)),
        action_horizon=int(data_cfg.get("action_horizon", 16)),
        async_offsets=data_cfg.get("fast_offsets", tuple(range(1, 13))),
        target_fps=float(data_cfg.get("target_fps", 10.0)),
        obs_context_dt_frames=int(data_cfg.get("obs_context_dt_frames", 1)),
        pose_key=str(data_cfg.get("pose_key", "action")),
        min_depth_m=float(data_cfg.get("min_depth_m", 0.1)),
        max_depth_m=float(data_cfg.get("max_depth_m", 5.0)),
        require_full_horizon=bool(data_cfg.get("require_full_horizon", True)),
        max_samples=data_cfg.get("max_samples"),
    )
    workers = int(loader_cfg.get("num_workers", 4))
    shuffle = bool(loader_cfg.get("shuffle", split == "train")) if split == "train" else False
    drop_last = bool(loader_cfg.get("drop_last", False)) if split == "train" else False
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", workers > 0)) and workers > 0,
        drop_last=drop_last,
        collate_fn=collate_plan_batch,
    )


def build_intern_online_dataloader(
    data_cfg: Mapping[str, Any],
    loader_cfg: Mapping[str, Any],
    split: str,
) -> DataLoader:
    dataset = InternDataOnlinePlanDataset(data_cfg, split)
    workers = int(loader_cfg.get("num_workers", 4))
    shuffle = bool(loader_cfg.get("shuffle", split == "train")) if split == "train" else False
    drop_last = bool(loader_cfg.get("drop_last", False)) if split == "train" else False
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", workers > 0)) and workers > 0,
        drop_last=drop_last,
        collate_fn=collate_online_plan_batch,
        multiprocessing_context="spawn" if workers > 0 else None,
    )
