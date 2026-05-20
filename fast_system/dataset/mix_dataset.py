"""Mixed SCAND/RECON datasets backed by JSONL indexes.

The dataset accepts both legacy SCAND ROS bag records and the extracted
``images/ + trajectory.npy`` records produced by ``data_preparation/build_dataset.py``.
Stage-2 items can expose several fast-system observation/action pairs for one
slow-system latent-goal clip.
"""

from __future__ import annotations

import io
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

try:
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag1.reader import Header, RecordType, read_bytes, read_uint32
except ImportError:  # Extracted-folder records do not need rosbags at runtime.
    AnyReader = None
    Header = None
    RecordType = None
    read_bytes = None
    read_uint32 = None


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_jsonl(paths: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: list[dict[str, Any]] = []
    for path_like in paths:
        path = Path(path_like).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Index file does not exist: {path}")
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None:
        return torch.float32
    lookup = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if dtype not in lookup:
        raise ValueError(f"Unsupported dtype {dtype!r}.")
    return lookup[dtype]


def natural_key(path: Path | str) -> list[object]:
    parts = re.split(r"(\d+)", str(path))
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def normalize_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def rotate_to_local(delta_xy: np.ndarray, yaw: float) -> np.ndarray:
    c, s = math.cos(float(yaw)), math.sin(float(yaw))
    rot = np.asarray([[c, -s], [s, c]], dtype=np.float32)
    return delta_xy @ rot


def compressed_msg_to_image(msg: Any) -> Image.Image:
    data = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(msg.data)
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def pil_to_tensor(image: Image.Image, size: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    if image.size != size:
        resample = getattr(Image, "Resampling", Image).BILINEAR
        image = image.resize(size, resample=resample)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor.to(dtype=dtype)


def image_file_to_tensor(path: Path, size: tuple[int, int], dtype: torch.dtype) -> torch.Tensor:
    with Image.open(path) as image:
        return pil_to_tensor(image.convert("RGB"), size=size, dtype=dtype)


def interpolate_local_pose(local_pose: np.ndarray, positions: np.ndarray) -> np.ndarray:
    domain = np.arange(len(local_pose), dtype=np.float32)
    pose = local_pose.astype(np.float32, copy=False)
    yaw = np.unwrap(pose[:, 2].astype(np.float64)).astype(np.float32)
    out = np.empty((len(positions), 3), dtype=np.float32)
    out[:, 0] = np.interp(positions, domain, pose[:, 0]).astype(np.float32)
    out[:, 1] = np.interp(positions, domain, pose[:, 1]).astype(np.float32)
    out[:, 2] = normalize_angle(np.interp(positions, domain, yaw)).astype(np.float32)
    return out


def infer_yaw_from_xy(xy: np.ndarray) -> np.ndarray:
    if len(xy) <= 1:
        return np.zeros((len(xy),), dtype=np.float32)
    grad = np.gradient(xy.astype(np.float32), axis=0)
    yaw = np.arctan2(grad[:, 1], grad[:, 0]).astype(np.float32)
    yaw[~np.isfinite(yaw)] = 0.0
    return yaw


class ScandStage1Dataset(Dataset):
    """Loads route clips and Stage-1 labels from bag-backed or extracted indexes."""

    def __init__(
        self,
        index_path: str | Path | Sequence[str | Path],
        image_size: int | None = 256,
        height: int | None = None,
        width: int | None = None,
        frame_tolerance_ns: int = 250_000_000,
        dtype: str | torch.dtype | None = "float32",
        cache_readers: bool = True,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        self.records = load_jsonl(index_path)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        if not self.records:
            raise ValueError(f"No samples found in index_path={index_path!r}.")

        if image_size is not None:
            height = width = int(image_size)
        if height is None or width is None:
            raise ValueError("Either image_size or both height and width must be provided.")
        self.size = (int(width), int(height))
        self.frame_tolerance_ns = int(frame_tolerance_ns)
        self.dtype = resolve_dtype(dtype)
        self.cache_readers = bool(cache_readers)
        self._readers: dict[str, AnyReader] = {}
        self._index_cache: dict[tuple[int, int], np.ndarray] = {}
        self._scene_image_cache: dict[str, tuple[Path, ...]] = {}

    def __len__(self) -> int:
        return len(self.records)

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _is_extracted_record(record: Mapping[str, Any]) -> bool:
        return record.get("storage") == "extracted" or "scene_dir" in record or "images_dir" in record

    def _scene_image_paths(self, record: Mapping[str, Any]) -> tuple[Path, ...]:
        image_paths = record.get("image_paths")
        if image_paths:
            return tuple(Path(path).expanduser() for path in image_paths)

        images_dir_value = record.get("images_dir")
        if images_dir_value is None and record.get("scene_dir") is not None:
            images_dir_value = str(Path(record["scene_dir"]) / "images")
        if images_dir_value is None:
            raise KeyError("Extracted record must contain images_dir, scene_dir, or image_paths.")

        images_dir = Path(images_dir_value).expanduser()
        cache_key = str(images_dir)
        cached = self._scene_image_cache.get(cache_key)
        if cached is None:
            cached = tuple(
                sorted(
                    (
                        path
                        for path in images_dir.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    ),
                    key=natural_key,
                )
            )
            if not cached:
                raise FileNotFoundError(f"No images found in {images_dir}")
            self._scene_image_cache[cache_key] = cached
        return cached

    def _read_extracted_indices(
        self,
        record: Mapping[str, Any],
        frame_indices: Sequence[int] | np.ndarray,
        size: tuple[int, int] | None = None,
        allow_padding: bool = False,
    ) -> torch.Tensor:
        paths = self._scene_image_paths(record)
        size = self.size if size is None else size
        max_index = len(paths) - 1
        frames: list[torch.Tensor] = []
        for raw_index in frame_indices:
            index = int(raw_index)
            if index < 0 or index > max_index:
                if not allow_padding:
                    raise IndexError(f"Frame index {index} is outside [0,{max_index}] for {record.get('sample_id')}.")
                index = min(max(index, 0), max_index)
            frames.append(image_file_to_tensor(paths[index], size=size, dtype=self.dtype))
        return torch.stack(frames, dim=0)

    def _get_reader(self, bag_path: str) -> AnyReader:
        if AnyReader is None:
            raise ImportError("rosbags is required for legacy bag-backed index records.")
        if not self.cache_readers:
            reader = AnyReader([Path(bag_path)])
            reader.open()
            return reader
        reader = self._readers.get(bag_path)
        if reader is None:
            reader = AnyReader([Path(bag_path)])
            reader.open()
            self._readers[bag_path] = reader
        return reader

    @staticmethod
    def _image_connections(reader: AnyReader, topic: str) -> list[Any]:
        conns = [conn for conn in reader.connections if conn.topic == topic]
        if not conns:
            raise KeyError(f"Image topic {topic!r} not found in bag.")
        return conns

    def _connection_times(self, conn: Any) -> np.ndarray | None:
        owner = getattr(conn, "owner", None)
        indexes = getattr(owner, "indexes", None)
        if indexes is None or conn.id not in indexes:
            return None
        key = (id(owner), int(conn.id))
        times = self._index_cache.get(key)
        if times is None:
            times = np.fromiter((entry.time for entry in indexes[conn.id]), dtype=np.int64)
            self._index_cache[key] = times
        return times

    @staticmethod
    def _read_rosbag1_raw(conn: Any, entry: Any) -> bytes:
        owner = conn.owner
        if owner.current_chunk[0] != entry.chunk_pos:
            owner.current_chunk[1].close()
            chunk_header = owner.chunks[entry.chunk_pos]
            owner.bio.seek(chunk_header.datapos)
            rawbytes = chunk_header.decompressor(read_bytes(owner.bio, chunk_header.datasize))
            owner.current_chunk = (entry.chunk_pos, io.BytesIO(rawbytes))

        chunk = owner.current_chunk[1]
        chunk.seek(entry.offset)
        while True:
            header = Header.read(chunk)
            have = header.get_uint8("op")
            if have != RecordType.CONNECTION:
                break
            chunk.seek(read_uint32(chunk), 1)
        if have != RecordType.MSGDATA:
            raise RuntimeError("Expected MSGDATA while reading rosbag1 index entry.")
        return read_bytes(chunk, read_uint32(chunk))

    def _iter_image_messages(
        self,
        reader: AnyReader,
        conns: Sequence[Any],
        start: int,
        stop: int,
    ) -> Iterable[tuple[Any, int, bytes]]:
        for conn in conns:
            times = self._connection_times(conn)
            owner = getattr(conn, "owner", None)
            indexes = getattr(owner, "indexes", None)
            if times is None or indexes is None:
                yield from reader.messages(connections=[conn], start=start, stop=stop)
                continue
            left = int(np.searchsorted(times, start, side="left"))
            for entry in indexes[conn.id][left:]:
                if int(entry.time) >= stop:
                    break
                yield conn, int(entry.time), self._read_rosbag1_raw(conn, entry)

    def _read_frames_at(
        self,
        record: Mapping[str, Any],
        frame_ts: Sequence[int] | np.ndarray,
        size: tuple[int, int] | None = None,
        allow_padding: bool = False,
    ) -> torch.Tensor:
        if self._is_extracted_record(record):
            raise TypeError("Extracted records must be read by frame indices, not timestamps.")

        bag_path = str(record["bag_path"])
        reader = self._get_reader(bag_path)
        try:
            frame_ts = np.asarray(frame_ts, dtype=np.int64)
            size = self.size if size is None else size
            conns = self._image_connections(reader, str(record["image_topic"]))
            start = int(frame_ts.min()) - self.frame_tolerance_ns
            stop = int(frame_ts.max()) + self.frame_tolerance_ns + 1
            best_raw: list[bytes | None] = [None] * len(frame_ts)
            best_conn: list[Any | None] = [None] * len(frame_ts)
            max_dt = np.iinfo(np.int64).max if allow_padding else self.frame_tolerance_ns
            best_dt = np.full(len(frame_ts), max_dt, dtype=np.int64)

            for conn, ts, raw in self._iter_image_messages(reader, conns, start=start, stop=stop):
                dt = np.abs(frame_ts - int(ts))
                for idx in np.flatnonzero(dt <= best_dt):
                    best_dt[idx] = int(dt[idx])
                    best_raw[idx] = raw
                    best_conn[idx] = conn

            frames: list[torch.Tensor] = []
            for idx, (raw, conn) in enumerate(zip(best_raw, best_conn)):
                if raw is None or conn is None:
                    raise RuntimeError(
                        f"Missing frame {idx} for {record.get('sample_id', bag_path)} "
                        f"within {self.frame_tolerance_ns} ns."
                    )
                msg = reader.deserialize(raw, conn.msgtype)
                frames.append(pil_to_tensor(compressed_msg_to_image(msg), size, self.dtype))
            return torch.stack(frames, dim=0)
        finally:
            if not self.cache_readers:
                reader.close()

    def _read_clip(self, record: Mapping[str, Any], size: tuple[int, int] | None = None) -> torch.Tensor:
        if self._is_extracted_record(record):
            indices = record.get("frame_indices")
            if indices is None:
                start = int(record.get("start_frame_index", 0))
                clip_frames = int(record.get("clip_frames", len(record.get("frame_filenames", []))))
                indices = list(range(start, start + clip_frames))
            return self._read_extracted_indices(record, indices, size=size)
        return self._read_frames_at(record, record["frame_timestamps_ns"], size=size)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        traj_local = record.get("traj_local")
        if traj_local is None and record.get("traj_local_pose") is not None:
            traj_local = np.asarray(record["traj_local_pose"], dtype=np.float32)[:, :2].tolist()
        sample: dict[str, Any] = {
            "video": self._read_clip(record),
            "coarse_goal": torch.as_tensor(record["coarse_goal"], dtype=torch.float32),
            "route_anchors": torch.as_tensor(record["route_anchors"], dtype=torch.float32),
            "turn_label": torch.as_tensor(record["turn_label"], dtype=torch.long),
            "traj_local": torch.as_tensor(traj_local, dtype=torch.float32),
            "sample_id": record.get("sample_id", str(index)),
            "bag_path": record.get("bag_path", record.get("scene_dir", "")),
            "bag_name": record.get("bag_name", record.get("scene_name", Path(record.get("bag_path", "")).stem)),
            "dataset": record.get("dataset", "unknown"),
        }
        return sample


class ScandStage2Dataset(ScandStage1Dataset):
    """Stage-2 samples for asynchronous slow-goal / fast-action training.

    One record contains a 16-frame future expert clip used by Stage-1 to compute
    z_g.  The Stage-2 item can pair that same latent goal with observations at
    several fast offsets inside the clip, e.g. t, t+4T, t+8T, and t+12T.  Each
    offset receives the remaining route segment re-rooted into that offset's
    local frame.
    """

    def __init__(
        self,
        index_path: str | Path | Sequence[str | Path],
        obs_image_size: int | None = 96,
        goal_image_size: int | None = 256,
        obs_height: int | None = None,
        obs_width: int | None = None,
        goal_height: int | None = None,
        goal_width: int | None = None,
        context_size: int = 3,
        context_dt_seconds: float | None = None,
        context_dt_frames: int = 1,
        action_horizon: int = 16,
        metric_waypoint_spacing: float = 0.38,
        normalize_actions: bool = True,
        allow_context_padding: bool = True,
        fast_stride_frames: int = 4,
        fast_offsets: Sequence[int] | None = None,
        return_async_steps: bool = True,
        **kwargs: Any,
    ) -> None:
        image_size = goal_image_size if goal_image_size is not None else obs_image_size
        super().__init__(index_path=index_path, image_size=image_size, height=goal_height, width=goal_width, **kwargs)
        self.context_size = int(context_size)
        self.context_dt_seconds = None if context_dt_seconds is None else float(context_dt_seconds)
        self.context_dt_frames = int(context_dt_frames)
        self.action_horizon = int(action_horizon)
        self.metric_waypoint_spacing = float(metric_waypoint_spacing)
        self.normalize_actions = bool(normalize_actions)
        self.allow_context_padding = bool(allow_context_padding)
        self.fast_stride_frames = int(fast_stride_frames)
        self.fast_offsets = None if fast_offsets is None else tuple(int(offset) for offset in fast_offsets)
        self.return_async_steps = bool(return_async_steps)
        if self.context_size < 0 or self.action_horizon <= 0:
            raise ValueError("context_size must be >= 0 and action_horizon must be > 0.")
        if self.context_dt_frames <= 0 or self.fast_stride_frames <= 0:
            raise ValueError("context_dt_frames and fast_stride_frames must be > 0.")

        if obs_image_size is not None:
            obs_height = obs_width = int(obs_image_size)
        if goal_image_size is not None:
            goal_height = goal_width = int(goal_image_size)
        if obs_height is None or obs_width is None or goal_height is None or goal_width is None:
            raise ValueError("Both obs and goal image sizes must be fully specified.")
        self.obs_size = (int(obs_width), int(obs_height))
        self.goal_size = (int(goal_width), int(goal_height))

    def _clip_frames(self, record: Mapping[str, Any]) -> int:
        if record.get("clip_frames") is not None:
            return int(record["clip_frames"])
        if record.get("frame_indices") is not None:
            return len(record["frame_indices"])
        return len(record["frame_timestamps_ns"])

    def _async_offsets(self, record: Mapping[str, Any]) -> list[int]:
        clip_frames = self._clip_frames(record)
        if not self.return_async_steps:
            return [0]
        offsets = self.fast_offsets if self.fast_offsets is not None else range(0, max(clip_frames - 1, 1), self.fast_stride_frames)
        valid = [int(offset) for offset in offsets if 0 <= int(offset) < clip_frames - 1]
        return valid or [0]

    def _context_timestamps(self, record: Mapping[str, Any], offset: int = 0) -> np.ndarray:
        frame_ts = np.asarray(record["frame_timestamps_ns"], dtype=np.int64)
        if self.context_dt_seconds is None:
            dt_ns = int(np.median(np.diff(frame_ts))) if len(frame_ts) > 1 else 125_000_000
        else:
            dt_ns = int(self.context_dt_seconds * 1e9)
        offset = min(max(int(offset), 0), len(frame_ts) - 1)
        current_ts = int(frame_ts[offset])
        offsets = np.arange(self.context_size, -1, -1, dtype=np.int64) * dt_ns
        return current_ts - offsets

    def _context_frame_indices(self, record: Mapping[str, Any], offset: int = 0) -> np.ndarray:
        frame_indices = record.get("frame_indices")
        if frame_indices is not None:
            offset = min(max(int(offset), 0), len(frame_indices) - 1)
            current_index = int(frame_indices[offset])
        else:
            current_index = int(record.get("start_frame_index", 0)) + int(offset)
        offsets = np.arange(self.context_size, -1, -1, dtype=np.int64) * self.context_dt_frames
        return current_index - offsets

    def _read_observation(self, record: Mapping[str, Any], offset: int) -> torch.Tensor:
        if self._is_extracted_record(record):
            return self._read_extracted_indices(
                record,
                self._context_frame_indices(record, offset=offset),
                size=self.obs_size,
                allow_padding=self.allow_context_padding,
            )
        return self._read_frames_at(
            record,
            self._context_timestamps(record, offset=offset),
            size=self.obs_size,
            allow_padding=self.allow_context_padding,
        )

    def _local_pose_trajectory(self, record: Mapping[str, Any]) -> np.ndarray:
        if record.get("traj_local_pose") is not None:
            pose = np.asarray(record["traj_local_pose"], dtype=np.float32)
            if pose.ndim == 2 and pose.shape[1] >= 3:
                return pose[:, :3]
        traj = np.asarray(record["traj_local"], dtype=np.float32)
        if traj.ndim != 2 or traj.shape[-1] < 2:
            raise ValueError(f"Invalid traj_local for {record.get('sample_id', '<unknown>')}.")
        xy = traj[:, :2]
        return np.concatenate((xy, infer_yaw_from_xy(xy)[:, None]), axis=1).astype(np.float32)

    def _current_pose_for_offset(self, record: Mapping[str, Any], offset: int, local_pose: np.ndarray) -> np.ndarray:
        clip_frame_poses = record.get("clip_frame_poses")
        if clip_frame_poses is not None:
            frame_pose = np.asarray(clip_frame_poses, dtype=np.float32)
            if frame_pose.ndim == 2 and frame_pose.shape[1] >= 3 and 0 <= int(offset) < len(frame_pose):
                return frame_pose[int(offset), :3]

        clip_frames = max(self._clip_frames(record), 2)
        dense_position = float(offset) / float(clip_frames - 1) * float(len(local_pose) - 1)
        return interpolate_local_pose(local_pose, np.asarray([dense_position], dtype=np.float32))[0]

    def _sample_actions(self, record: Mapping[str, Any], offset: int = 0) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        local_pose = self._local_pose_trajectory(record)
        if len(local_pose) == 1:
            actions = np.repeat(local_pose[:, :2], self.action_horizon, axis=0)
            raw_actions = actions.copy()
        else:
            clip_frames = max(self._clip_frames(record), 2)
            offset = min(max(int(offset), 0), clip_frames - 2)
            current_dense = float(offset) / float(clip_frames - 1) * float(len(local_pose) - 1)
            sample_positions = np.linspace(current_dense, float(len(local_pose) - 1), self.action_horizon + 1)[1:]
            future_pose = interpolate_local_pose(local_pose, sample_positions.astype(np.float32))
            current_pose = self._current_pose_for_offset(record, offset=offset, local_pose=local_pose)
            raw_actions = rotate_to_local(future_pose[:, :2] - current_pose[:2], float(current_pose[2]))
            actions = raw_actions.copy()

        distance = float(np.linalg.norm(raw_actions[-1])) / max(self.metric_waypoint_spacing, 1e-6)
        if self.normalize_actions:
            actions = actions / max(self.metric_waypoint_spacing, 1e-6)
        action_tensor = torch.as_tensor(actions, dtype=torch.float32)
        distance_tensor = torch.as_tensor(distance, dtype=torch.float32)
        goal_pos = action_tensor[-1]
        action_mask = torch.ones((), dtype=torch.float32)
        return action_tensor, distance_tensor, goal_pos, action_mask

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        offsets = self._async_offsets(record)
        obs_list: list[torch.Tensor] = []
        action_list: list[torch.Tensor] = []
        distance_list: list[torch.Tensor] = []
        goal_pos_list: list[torch.Tensor] = []
        action_mask_list: list[torch.Tensor] = []
        for offset in offsets:
            action, distance, goal_pos, action_mask = self._sample_actions(record, offset=offset)
            obs_list.append(self._read_observation(record, offset=offset))
            action_list.append(action)
            distance_list.append(distance)
            goal_pos_list.append(goal_pos)
            action_mask_list.append(action_mask)

        if self.return_async_steps:
            obs: torch.Tensor = torch.stack(obs_list, dim=0)
            action_out: torch.Tensor = torch.stack(action_list, dim=0)
            distance_out: torch.Tensor = torch.stack(distance_list, dim=0)
            goal_pos_out: torch.Tensor = torch.stack(goal_pos_list, dim=0)
            action_mask_out: torch.Tensor = torch.stack(action_mask_list, dim=0)
            async_offsets: torch.Tensor = torch.as_tensor(offsets, dtype=torch.long)
        else:
            obs = obs_list[0]
            action_out = action_list[0]
            distance_out = distance_list[0]
            goal_pos_out = goal_pos_list[0]
            action_mask_out = action_mask_list[0]
            async_offsets = torch.as_tensor(offsets[0], dtype=torch.long)

        sample: dict[str, Any] = {
            "obs": obs,
            "goal_video": self._read_clip(record, size=self.goal_size),
            "action": action_out,
            "distance": distance_out,
            "goal_pos": goal_pos_out,
            "action_mask": action_mask_out,
            "async_offsets": async_offsets,
            "sample_id": record.get("sample_id", str(index)),
            "bag_path": record.get("bag_path", record.get("scene_dir", "")),
            "bag_name": record.get("bag_name", record.get("scene_name", Path(record.get("bag_path", "")).stem)),
            "dataset": record.get("dataset", "unknown"),
        }
        return sample


def _stack_or_list(values: list[Any]) -> Any:
    if all(torch.is_tensor(value) for value in values):
        shapes = {tuple(value.shape) for value in values}
        if len(shapes) == 1:
            return torch.stack(values, dim=0)
    return values


def scand_stage1_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for sample in batch for key in sample.keys()})
    return {key: _stack_or_list([sample[key] for sample in batch]) for key in keys}


def build_scand_dataset_from_config(cfg: Mapping[str, Any], split: str = "train") -> ScandStage1Dataset:
    index_path = cfg.get(f"{split}_index") or cfg.get("index_path")
    if index_path is None:
        raise KeyError(f"Missing {split}_index or index_path in dataset config.")
    return ScandStage1Dataset(
        index_path=index_path,
        image_size=cfg.get("image_size", 256),
        height=cfg.get("height"),
        width=cfg.get("width"),
        frame_tolerance_ns=int(cfg.get("frame_tolerance_ns", 250_000_000)),
        dtype=cfg.get("dtype", "float32"),
        cache_readers=bool(cfg.get("cache_readers", True)),
        max_samples=cfg.get("max_samples"),
    )


def build_scand_stage2_dataset_from_config(cfg: Mapping[str, Any], split: str = "train") -> ScandStage2Dataset:
    index_path = cfg.get(f"{split}_index") or cfg.get("index_path")
    if index_path is None:
        raise KeyError(f"Missing {split}_index or index_path in dataset config.")
    return ScandStage2Dataset(
        index_path=index_path,
        obs_image_size=cfg.get("obs_image_size", cfg.get("image_size", 96)),
        goal_image_size=cfg.get("goal_image_size", cfg.get("stage1_image_size", 256)),
        obs_height=cfg.get("obs_height"),
        obs_width=cfg.get("obs_width"),
        goal_height=cfg.get("goal_height"),
        goal_width=cfg.get("goal_width"),
        context_size=int(cfg.get("context_size", 3)),
        context_dt_seconds=cfg.get("context_dt_seconds"),
        context_dt_frames=int(cfg.get("context_dt_frames", 1)),
        action_horizon=int(cfg.get("action_horizon", 16)),
        metric_waypoint_spacing=float(cfg.get("metric_waypoint_spacing", 0.38)),
        normalize_actions=bool(cfg.get("normalize_actions", True)),
        allow_context_padding=bool(cfg.get("allow_context_padding", True)),
        fast_stride_frames=int(cfg.get("fast_stride_frames", 4)),
        fast_offsets=cfg.get("fast_offsets"),
        return_async_steps=bool(cfg.get("return_async_steps", True)),
        frame_tolerance_ns=int(cfg.get("frame_tolerance_ns", 250_000_000)),
        dtype=cfg.get("dtype", "float32"),
        cache_readers=bool(cfg.get("cache_readers", True)),
        max_samples=cfg.get("max_samples"),
    )


def build_scand_dataloader_from_config(
    data_cfg: Mapping[str, Any],
    loader_cfg: Mapping[str, Any],
    split: str = "train",
) -> DataLoader:
    dataset = build_scand_dataset_from_config(data_cfg, split=split)
    num_workers = int(loader_cfg.get("num_workers", 4))
    shuffle_value = loader_cfg.get("shuffle", None)
    shuffle = split == "train" if shuffle_value is None else bool(shuffle_value)
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=bool(loader_cfg.get("drop_last", False)),
        timeout=int(loader_cfg.get("timeout", 0)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0,
        collate_fn=scand_stage1_collate_fn,
    )


def build_scand_stage2_dataloader_from_config(
    data_cfg: Mapping[str, Any],
    loader_cfg: Mapping[str, Any],
    split: str = "train",
) -> DataLoader:
    dataset = build_scand_stage2_dataset_from_config(data_cfg, split=split)
    num_workers = int(loader_cfg.get("num_workers", 4))
    if split == "train":
        shuffle = bool(loader_cfg.get("shuffle", True))
    else:
        shuffle = bool(loader_cfg.get(f"{split}_shuffle", False))
    return DataLoader(
        dataset,
        batch_size=int(loader_cfg.get("batch_size", 8)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=bool(loader_cfg.get("pin_memory", True)),
        drop_last=bool(loader_cfg.get("drop_last", False)),
        timeout=int(loader_cfg.get("timeout", 0)),
        persistent_workers=bool(loader_cfg.get("persistent_workers", False)) and num_workers > 0,
        collate_fn=scand_stage1_collate_fn,
    )


ScandDataset = ScandStage1Dataset
MixDataset = ScandStage1Dataset
MixStage2Dataset = ScandStage2Dataset
