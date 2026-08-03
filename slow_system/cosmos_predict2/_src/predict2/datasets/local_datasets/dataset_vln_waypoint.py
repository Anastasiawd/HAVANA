# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import random
import re
import traceback
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from decord import VideoReader, cpu
from megatron.core import parallel_state
from torch.utils.data import DataLoader, Dataset, DistributedSampler, Sampler
from torchvision import transforms as T

from cosmos_predict2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.datasets.local_datasets.dataset_utils import ResizePreprocess, ToTensorVideo


@dataclass(frozen=True)
class InternDataSubtask:
    task_index: int
    start_frame: int
    end_frame: int
    caption: str
    canonical_action: str = ""
    original_instruction: str = ""
    revised_instruction: str = ""
    revised_instruction_conflict: bool = False


@dataclass(frozen=True)
class InternDataEpisode:
    root: Path
    episode_index: int
    data_path: Path
    video_path: Path
    subtasks: tuple[InternDataSubtask, ...]
    length: Optional[int]
    source_fps: Optional[float]
    source_id: str = ""
    scene_id: str = ""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


def _first_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        for item in value:
            text = _first_text(item)
            if text:
                return text
    return ""


def _episode_caption(episode: dict[str, Any], priority: Sequence[str]) -> str:
    tasks = episode.get("tasks", [])
    for field in priority:
        if field in episode:
            text = _first_text(episode[field])
            if text:
                return text
        if isinstance(tasks, Sequence) and not isinstance(tasks, (bytes, bytearray, str)):
            for task in tasks:
                if isinstance(task, dict) and field in task:
                    text = _first_text(task[field])
                    if text:
                        return text
    return _first_text(tasks)


def _task_caption(task: dict[str, Any], priority: Sequence[str]) -> str:
    for field in priority:
        if field in task:
            text = _first_text(task[field])
            if text:
                return text
    return ""


ACTION_PARSER_VERSION = "explicit-motion-v2"

_CANONICAL_ACTION_PATTERN = re.compile(r"\b(TURN_LEFT|TURN_RIGHT|GO_STRAIGHT)\b", re.IGNORECASE)
_COUNTERCLOCKWISE_PATTERN = re.compile(r"\bcounter[- ]?clockwise\b", re.IGNORECASE)
_CLOCKWISE_PATTERN = re.compile(r"\bclockwise\b", re.IGNORECASE)
_EXPLICIT_TURN_PATTERNS = (
    re.compile(
        r"\b(?:turn(?:ing)?|veer(?:ing)?|bear(?:ing)?|pivot(?:ing)?|rotate|rotating)"
        r"(?:\s+(?:sharply|slightly|gently|gradually))?"
        r"(?:\s+\d+(?:\.\d+)?\s*degrees?)?"
        r"\s+(?:to\s+(?:your|the)\s+)?(?P<direction>left|right)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:make|take|hang|hook|swing)\s+"
        r"(?:(?:a|an|the|your)\s+)?(?:(?:first|second|next|sharp|slight)\s+)?"
        r"(?P<direction>left|right)(?:\s+turn)?\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(?P<direction>left|right)\s+turn\b", re.IGNORECASE),
    re.compile(r"\b(?:head|go|move)\s+(?:to\s+the\s+)?(?P<direction>left|right)\b", re.IGNORECASE),
)
_EXPLICIT_STRAIGHT_PATTERNS = (
    re.compile(
        r"\b(?:walk|go|move|proceed|continue|head|travel|advance)\s+"
        r"(?:(?:walking|moving|going|directly)\s+)?(?:straight|forward|ahead)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bkeep\s+(?:walking|moving|going\s+)?(?:straight|forward|ahead)\b", re.IGNORECASE),
    re.compile(r"\bstraight\s+ahead\b", re.IGNORECASE),
)


def canonical_action_from_instruction(instruction: str) -> Optional[str]:
    """Parse the commanded motion while ignoring landmark-side mentions.

    ``left`` and ``right`` are common landmark descriptors in VLN prompts.
    They become actions only when attached to an explicit motion predicate,
    such as ``turn right`` or ``take a left``. Conflicting explicit motion
    predicates remain unparseable instead of inventing a direction.
    """
    instruction = instruction or ""
    canonical_actions = {match.upper() for match in _CANONICAL_ACTION_PATTERN.findall(instruction)}
    if len(canonical_actions) == 1:
        return canonical_actions.pop()
    if len(canonical_actions) > 1:
        return None

    turn_directions: set[str] = set()
    if _COUNTERCLOCKWISE_PATTERN.search(instruction):
        turn_directions.add("left")
    without_counterclockwise = _COUNTERCLOCKWISE_PATTERN.sub("", instruction)
    if _CLOCKWISE_PATTERN.search(without_counterclockwise):
        turn_directions.add("right")
    for pattern in _EXPLICIT_TURN_PATTERNS:
        turn_directions.update(match.group("direction").lower() for match in pattern.finditer(instruction))

    if len(turn_directions) > 1:
        return None
    if turn_directions == {"left"}:
        return "TURN_LEFT"
    if turn_directions == {"right"}:
        return "TURN_RIGHT"
    if any(pattern.search(instruction) for pattern in _EXPLICIT_STRAIGHT_PATTERNS):
        return "GO_STRAIGHT"
    return None


def swap_left_right_text(text: str) -> str:
    """Swap left/right case-insensitively without replacing substrings."""
    action_placeholder = "__VLN_TURN_ACTION_SWAP__"
    swapped = re.sub(r"TURN_LEFT", action_placeholder, text, flags=re.IGNORECASE)
    swapped = re.sub(r"TURN_RIGHT", "TURN_LEFT", swapped, flags=re.IGNORECASE)
    swapped = swapped.replace(action_placeholder, "TURN_RIGHT")
    placeholder = "__VLN_LEFT_RIGHT_SWAP__"
    swapped = re.sub(r"\bleft\b", placeholder, swapped, flags=re.IGNORECASE)
    swapped = re.sub(r"\bright\b", "left", swapped, flags=re.IGNORECASE)
    return swapped.replace(placeholder, "right")


def _frame_bounds(value: Any) -> Optional[tuple[int, int]]:
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray, str)) or len(value) != 2:
        return None
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < start:
        return None
    return start, end


def _episode_subtasks(
    episode: dict[str, Any],
    priority: Sequence[str],
    structured_prompt: bool = False,
) -> tuple[InternDataSubtask, ...]:
    """Extract subtask-level captions and inclusive frame bounds.

    Prefer ``sub_indexes`` records. The episode-level ``sum_indexes`` record is
    only used when no subtask annotations exist, so it is not duplicated as an
    extra sample alongside the subtasks it summarizes.
    """
    tasks = episode.get("tasks", [])
    if not isinstance(tasks, Sequence) or isinstance(tasks, (bytes, bytearray, str)):
        tasks = []

    indexed_tasks = [
        (task_index, task, "sub_indexes")
        for task_index, task in enumerate(tasks)
        if isinstance(task, dict) and _frame_bounds(task.get("sub_indexes")) is not None
    ]
    if not indexed_tasks:
        indexed_tasks = [
            (task_index, task, "sum_indexes")
            for task_index, task in enumerate(tasks)
            if isinstance(task, dict) and _frame_bounds(task.get("sum_indexes")) is not None
        ]

    subtasks = []
    for task_index, task, bounds_field in indexed_tasks:
        bounds = _frame_bounds(task[bounds_field])
        assert bounds is not None
        original_instruction = _first_text(task.get("sub_instruction", ""))
        revised_instruction = _first_text(task.get("revised_sub_instruction", ""))
        canonical_action = canonical_action_from_instruction(original_instruction)
        revised_action = canonical_action_from_instruction(revised_instruction)
        revised_conflict = bool(canonical_action and revised_action and canonical_action != revised_action)
        caption = original_instruction if revised_conflict else _task_caption(task, priority)
        if not caption:
            continue
        if structured_prompt:
            if canonical_action is None:
                continue
            caption = f"Action: {canonical_action}\nInstruction: {caption}"
        subtasks.append(
            InternDataSubtask(
                task_index=task_index,
                start_frame=bounds[0],
                end_frame=bounds[1],
                caption=caption,
                canonical_action=canonical_action or "",
                original_instruction=original_instruction,
                revised_instruction=revised_instruction,
                revised_instruction_conflict=revised_conflict,
            )
        )

    if subtasks:
        return tuple(subtasks)
    if structured_prompt:
        # Temporal VLN actions must originate from a bounded original
        # sub_instruction; episode summaries are never promoted to labels.
        return ()

    caption = _episode_caption(episode, priority)
    length = episode.get("length")
    if caption and length is not None and int(length) > 0:
        canonical_action = canonical_action_from_instruction(caption)
        if structured_prompt and canonical_action is None:
            return ()
        prompt = f"Action: {canonical_action}\nInstruction: {caption}" if structured_prompt else caption
        return (
            InternDataSubtask(
                task_index=0,
                start_frame=0,
                end_frame=int(length) - 1,
                caption=prompt,
                canonical_action=canonical_action or "",
                original_instruction=caption,
            ),
        )
    return ()


def _wrap_to_pi(angle: float) -> float:
    return float((angle + np.pi) % (2 * np.pi) - np.pi)


def _pose_matrix(value: Any) -> np.ndarray:
    pose = np.asarray(value, dtype=np.float32)
    if pose.size != 16:
        raise ValueError(f"Pose must contain 16 values, got shape {pose.shape}")
    return pose.reshape(4, 4)


def _yaw_from_pose(pose: np.ndarray) -> float:
    return float(np.arctan2(pose[1, 0], pose[0, 0]))


def _relative_ego_waypoints(
    poses: Sequence[np.ndarray],
    ref_pose: np.ndarray,
    dx_scale: float,
    dy_scale: float,
    dyaw_scale: float,
) -> np.ndarray:
    ref_xy = ref_pose[:2, 3]
    ref_yaw = _yaw_from_pose(ref_pose)
    cos_yaw = np.cos(-ref_yaw)
    sin_yaw = np.sin(-ref_yaw)
    waypoints = []
    for pose in poses:
        delta = pose[:2, 3] - ref_xy
        dx = cos_yaw * delta[0] - sin_yaw * delta[1]
        dy = sin_yaw * delta[0] + cos_yaw * delta[1]
        dyaw = _wrap_to_pi(_yaw_from_pose(pose) - ref_yaw)
        waypoints.append([dx / dx_scale, dy / dy_scale, dyaw / dyaw_scale])
    return np.asarray(waypoints, dtype=np.float32)


def _subtask_aligned_indices(
    total_frames: int,
    sequence_length: int,
    stride: int,
    subtask_start: int,
    subtask_end: int,
    anchor_position: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a clip whose anchor frame belongs to the selected subtask.

    Short turn subtasks are retained: the clip may include approach and exit
    frames from adjacent subtasks, while the selected action is placed at the
    configured future waypoint position whenever episode boundaries allow it.
    """
    required_span = (sequence_length - 1) * stride
    if total_frames <= required_span:
        raise ValueError(
            f"Need at least {required_span + 1} aligned frames for {sequence_length} frames at stride {stride}, "
            f"got {total_frames}"
        )
    if not 0 <= anchor_position < sequence_length:
        raise ValueError(f"anchor_position={anchor_position} must be in [0, {sequence_length})")
    if subtask_end < 0 or subtask_start >= total_frames:
        raise ValueError(f"Subtask [{subtask_start}, {subtask_end}] is outside an episode with {total_frames} frames")

    subtask_start = max(0, subtask_start)
    subtask_end = min(total_frames - 1, subtask_end)
    if subtask_end < subtask_start:
        raise ValueError(f"Invalid clipped subtask bounds [{subtask_start}, {subtask_end}]")

    max_start = total_frames - required_span - 1
    anchor_offset = anchor_position * stride
    min_aligned_start = max(0, subtask_start - anchor_offset)
    max_aligned_start = min(max_start, subtask_end - anchor_offset)
    if min_aligned_start <= max_aligned_start:
        start = int(rng.integers(min_aligned_start, max_aligned_start + 1))
    else:
        # Near episode boundaries the requested anchor position is infeasible.
        # Keep the whole clip valid and place the beginning of the subtask as
        # close to the anchor position as possible.
        start = int(np.clip(subtask_start - anchor_offset, 0, max_start))
    return start + np.arange(sequence_length, dtype=np.int64) * stride


class InternDataVLNWaypointDataset(Dataset):
    """InternData/LeRobot VLN clip loader with normalized ego waypoint targets."""

    def __init__(
        self,
        dataset_dir: str | Sequence[str],
        num_frames: int = 29,
        video_size: tuple[int, int] = (272, 480),
        target_fps: Optional[float] = 10.0,
        video_key: str = "observation.video.rgb",
        pose_key: str = "action",
        waypoint_positions: Sequence[int] = (10, 16, 22, 28),
        subtask_anchor_position: Optional[int] = None,
        prompt_fields: Sequence[str] = ("revised_sub_instruction", "sub_instruction", "sum_instruction"),
        dx_scale: float = 1.0,
        dy_scale: float = 1.0,
        dyaw_scale: float = float(np.pi),
        limit_episodes: Optional[int] = None,
        skip_missing: bool = True,
        recursive_scene_discovery: bool = True,
        seed: int = 0,
        temporal_mode: bool = False,
        condition_frame_positions: Sequence[int] = (0, 1, 2, 3, 4),
        subtask_start_position: int = 5,
        waypoint_reference_position: int = 4,
        post_subtask_tail_frames: int = 4,
        straight_window_stride: int = 8,
        source_audit_manifest: Optional[str] = None,
        source_audit_policy: str = "strict",
        structured_prompt: bool = False,
        horizontal_flip_prob: float = 0.0,
        strict_target_fps: bool = False,
        onset_jitter: int = 1,
    ) -> None:
        super().__init__()
        self.dataset_dirs = [Path(dataset_dir)] if isinstance(dataset_dir, str) else [Path(p) for p in dataset_dir]
        self.sequence_length = num_frames
        self.target_fps = target_fps
        self.video_key = video_key
        self.pose_key = pose_key
        self.waypoint_positions = tuple(int(pos) for pos in waypoint_positions)
        self.subtask_anchor_position = int(
            self.waypoint_positions[0] if subtask_anchor_position is None else subtask_anchor_position
        )
        self.prompt_fields = tuple(prompt_fields)
        self.dx_scale = dx_scale
        self.dy_scale = dy_scale
        self.dyaw_scale = dyaw_scale
        self.recursive_scene_discovery = recursive_scene_discovery
        self.seed = seed
        self.temporal_mode = temporal_mode
        self.condition_frame_positions = tuple(int(pos) for pos in condition_frame_positions)
        self.subtask_start_position = int(subtask_start_position)
        self.waypoint_reference_position = int(waypoint_reference_position)
        self.post_subtask_tail_frames = int(post_subtask_tail_frames)
        self.straight_window_stride = int(straight_window_stride)
        self.structured_prompt = structured_prompt
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.strict_target_fps = strict_target_fps
        self.onset_jitter = int(onset_jitter)
        self.num_failed_loads = 0
        self.filter_counts: Counter[str] = Counter()
        self.source_audit_policy = source_audit_policy
        self.allowed_sources = self._load_allowed_sources(source_audit_manifest, source_audit_policy)
        self.mirror_allowed_sources = self._load_mirror_allowed_sources(source_audit_manifest)
        self.preprocess = T.Compose([ToTensorVideo(), ResizePreprocess((video_size[0], video_size[1]))])

        if not self.waypoint_positions or min(self.waypoint_positions) < 0:
            raise ValueError(f"waypoint_positions must be non-empty and non-negative, got {self.waypoint_positions}")
        if max(self.waypoint_positions) >= self.sequence_length:
            raise ValueError(f"waypoint_positions {self.waypoint_positions} exceed num_frames={self.sequence_length}")
        if not 0 <= self.subtask_anchor_position < self.sequence_length:
            raise ValueError(
                f"subtask_anchor_position={self.subtask_anchor_position} must be in [0, {self.sequence_length})"
            )
        if self.temporal_mode:
            expected = {
                "num_frames": 17,
                "condition_frame_positions": (0, 1, 2, 3, 4),
                "subtask_start_position": 5,
                "waypoint_reference_position": 4,
                "waypoint_positions": (8, 12, 16),
            }
            actual = {
                "num_frames": self.sequence_length,
                "condition_frame_positions": self.condition_frame_positions,
                "subtask_start_position": self.subtask_start_position,
                "waypoint_reference_position": self.waypoint_reference_position,
                "waypoint_positions": self.waypoint_positions,
            }
            mismatches = [f"{key}={actual[key]!r}" for key in expected if actual[key] != expected[key]]
            if mismatches:
                raise ValueError(
                    "temporal_mode implements the fixed 17-frame contract; invalid " + ", ".join(mismatches)
                )
            if self.target_fps != 10.0:
                raise ValueError(f"temporal_mode requires target_fps=10.0, got {self.target_fps}")
            if source_audit_manifest is None:
                raise ValueError("temporal_mode requires source_audit_manifest; source coordinates are never guessed")

        self.episodes = self._discover_episodes(skip_missing=skip_missing)
        if limit_episodes is not None:
            self.episodes = self.episodes[:limit_episodes]
        self.samples: list[tuple[InternDataEpisode, InternDataSubtask]] = []
        self.sample_anchor_frames: list[Optional[int]] = []
        for episode in self.episodes:
            for subtask in episode.subtasks:
                if self.temporal_mode:
                    anchors = self._temporal_sample_anchors(episode, subtask)
                    self.samples.extend((episode, subtask) for _ in anchors)
                    self.sample_anchor_frames.extend(anchors)
                else:
                    self.samples.append((episode, subtask))
                    self.sample_anchor_frames.append(None)
        if not self.samples:
            raise ValueError(f"No valid InternData VLN subtasks found under {self.dataset_dirs}")
        log.info(
            f"{len(self.episodes)} InternData VLN episodes and {len(self.samples)} subtask-aligned samples in total"
        )
        if self.filter_counts:
            log.info(f"VLN temporal dataset filtering counts: {dict(sorted(self.filter_counts.items()))}")

    def __str__(self) -> str:
        return f"{len(self.samples)} InternData VLN waypoint subtasks from {self.dataset_dirs}"

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_allowed_sources(
        manifest_path: Optional[str],
        source_audit_policy: str = "strict",
    ) -> Optional[set[str]]:
        if source_audit_policy not in {"report_only", "strict"}:
            raise ValueError(f"source_audit_policy must be 'report_only' or 'strict', got {source_audit_policy!r}")
        if manifest_path is None:
            return None
        manifest = json.loads(Path(manifest_path).read_text())
        if source_audit_policy == "report_only":
            return None
        if "allowed_sources" in manifest:
            return set(manifest["allowed_sources"])
        return {
            source_id
            for source_id, record in manifest.get("sources", {}).items()
            if record.get("allowed_for_training", False)
        }

    @staticmethod
    def _load_mirror_allowed_sources(manifest_path: Optional[str]) -> Optional[set[str]]:
        """Return sources whose pose convention is reliable enough to mirror.

        ``report_only`` keeps every source for training, but it must not turn
        an unreliable coordinate source into twice as many contradictory turn
        examples. A missing manifest preserves the legacy non-temporal
        behavior.
        """
        if manifest_path is None:
            return None
        manifest = json.loads(Path(manifest_path).read_text())
        if "recommended_sources" in manifest:
            return set(manifest["recommended_sources"])
        return {
            source_id
            for source_id, record in manifest.get("sources", {}).items()
            if record.get("passes_recommended_turn_audit", record.get("allowed_for_training", False))
        }

    @staticmethod
    def _source_scene_ids(root: Path, dataset_root: Path) -> tuple[str, str]:
        try:
            parts = root.relative_to(dataset_root).parts
        except ValueError:
            parts = ()
        if not parts:
            return root.name, root.name
        if len(parts) == 1:
            return parts[0], parts[0]
        return parts[0], "/".join(parts[1:])

    def _target_stride(self, source_fps: float) -> int:
        if self.target_fps is None or self.target_fps <= 0:
            return 1
        stride = max(1, int(round(source_fps / self.target_fps)))
        adjusted_fps = source_fps / stride
        if getattr(self, "strict_target_fps", False) and not np.isclose(adjusted_fps, self.target_fps, atol=0.25):
            raise ValueError(
                f"Source fps={source_fps} cannot produce fixed {self.target_fps} FPS with integer stride "
                f"(nearest is {adjusted_fps:.3f})"
            )
        return stride

    def _temporal_sample_anchors(
        self,
        episode: InternDataEpisode,
        subtask: InternDataSubtask,
    ) -> list[int]:
        if subtask.canonical_action not in {"TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT"}:
            self.filter_counts["unparseable_action"] += 1
            return []
        if episode.source_fps is None or episode.length is None:
            self.filter_counts["missing_fps_or_length"] += 1
            return []
        try:
            stride = self._target_stride(episode.source_fps)
        except ValueError:
            self.filter_counts["invalid_target_fps"] += 1
            return []

        min_anchor = self.subtask_start_position * stride
        max_anchor = int(episode.length) - 1 - (self.sequence_length - 1 - self.subtask_start_position) * stride
        if max_anchor < min_anchor:
            self.filter_counts["insufficient_history_or_future"] += 1
            return []

        if subtask.canonical_action in {"TURN_LEFT", "TURN_RIGHT"}:
            anchor = subtask.start_frame
            feasible = any(
                0 <= anchor - (self.subtask_start_position + jitter) * stride
                and anchor - (self.subtask_start_position + jitter) * stride + (self.sequence_length - 1) * stride
                < int(episode.length)
                for jitter in range(-self.onset_jitter, self.onset_jitter + 1)
            )
            if not feasible:
                self.filter_counts["insufficient_history_or_future"] += 1
                return []
            return [anchor]

        first_anchor = max(subtask.start_frame, min_anchor)
        last_anchor = min(subtask.end_frame, max_anchor)
        if first_anchor > last_anchor:
            self.filter_counts["insufficient_history_or_future"] += 1
            return []
        source_window_stride = self.straight_window_stride * stride
        return list(range(first_anchor, last_anchor + 1, source_window_stride))

    def _iter_dataset_roots(self, root: Path) -> list[Path]:
        if (root / "meta" / "info.json").exists() and (root / "meta" / "episodes.jsonl").exists():
            return [root]
        if not self.recursive_scene_discovery:
            return [root]

        scene_roots = []
        seen = set()
        for pattern in ("*/meta/info.json", "*/*/meta/info.json", "*/*/*/meta/info.json"):
            for info_path in sorted(root.glob(pattern)):
                scene_root = info_path.parent.parent
                if scene_root in seen:
                    continue
                if (scene_root / "meta" / "episodes.jsonl").exists():
                    scene_roots.append(scene_root)
                    seen.add(scene_root)
        return scene_roots or [root]

    def _discover_episodes(self, skip_missing: bool) -> list[InternDataEpisode]:
        episodes: list[InternDataEpisode] = []
        skipped_short_episodes = 0
        for dataset_root in self.dataset_dirs:
            for root in self._iter_dataset_roots(dataset_root):
                source_id, scene_id = self._source_scene_ids(root, dataset_root)
                if self.allowed_sources is not None and source_id not in self.allowed_sources:
                    self.filter_counts["source_not_audited"] += 1
                    continue
                info_path = root / "meta" / "info.json"
                episodes_path = root / "meta" / "episodes.jsonl"
                if not info_path.exists() or not episodes_path.exists():
                    if skip_missing:
                        log.warning(f"Skipping {root}: missing meta/info.json or meta/episodes.jsonl")
                        continue
                    raise FileNotFoundError(f"Missing metadata under {root}")
                info = json.loads(info_path.read_text())
                chunk_size = int(info.get("chunks_size", 1000))
                data_pattern = info["data_path"]
                video_pattern = info["video_path"]
                source_fps = float(info["fps"]) if "fps" in info else None
                for episode in _read_jsonl(episodes_path):
                    episode_index = int(episode["episode_index"])
                    subtasks = _episode_subtasks(
                        episode,
                        self.prompt_fields,
                        structured_prompt=self.structured_prompt,
                    )
                    if not subtasks:
                        message = f"Episode {episode_index} in {episodes_path} has no bounded instruction"
                        if skip_missing:
                            log.warning(f"Skipping {message}")
                            continue
                        raise ValueError(message)
                    episode_length = episode.get("length")
                    if episode_length is None:
                        episode_length = max(subtask.end_frame for subtask in subtasks) + 1
                    episode_length = int(episode_length)
                    if episode_length < self.sequence_length and not self.temporal_mode:
                        if skip_missing:
                            skipped_short_episodes += 1
                            continue
                        raise ValueError(
                            f"Episode {episode_index} in {episodes_path} has only {episode_length} frames, "
                            f"fewer than num_frames={self.sequence_length}"
                        )
                    chunk = episode_index // chunk_size
                    data_path = root / data_pattern.format(episode_chunk=chunk, episode_index=episode_index)
                    video_path = root / video_pattern.format(
                        episode_chunk=chunk,
                        episode_index=episode_index,
                        video_key=self.video_key,
                    )
                    if not video_path.exists() and self.video_key.startswith("observation."):
                        alt_key = self.video_key.replace("observation.video.", "observation.images.")
                        video_path = root / video_pattern.format(
                            episode_chunk=chunk,
                            episode_index=episode_index,
                            video_key=alt_key,
                        )
                    if skip_missing and (not data_path.exists() or not video_path.exists()):
                        continue
                    if not data_path.exists():
                        raise FileNotFoundError(data_path)
                    if not video_path.exists():
                        raise FileNotFoundError(video_path)
                    episodes.append(
                        InternDataEpisode(
                            root=root,
                            episode_index=episode_index,
                            data_path=data_path,
                            video_path=video_path,
                            subtasks=subtasks,
                            length=episode_length,
                            source_fps=source_fps,
                            source_id=source_id,
                            scene_id=scene_id,
                        )
                    )
        if skipped_short_episodes:
            log.warning(
                f"Skipped {skipped_short_episodes} InternData VLN episodes shorter than "
                f"num_frames={self.sequence_length}"
            )
        return episodes

    def _sample_indices(
        self,
        total_frames: int,
        source_fps: float,
        subtask: InternDataSubtask,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        stride = self._target_stride(source_fps)
        adjusted_fps = source_fps / stride
        if getattr(self, "temporal_mode", False):
            raise RuntimeError("Temporal samples must use _sample_temporal_indices with a fixed anchor")
        max_feasible_stride = (total_frames - 1) // (self.sequence_length - 1)
        if max_feasible_stride < 1:
            raise ValueError(f"Need at least {self.sequence_length} frames, got {total_frames}")
        stride = min(stride, max_feasible_stride)
        adjusted_fps = source_fps / stride
        frame_indices = _subtask_aligned_indices(
            total_frames=total_frames,
            sequence_length=self.sequence_length,
            stride=stride,
            subtask_start=subtask.start_frame,
            subtask_end=subtask.end_frame,
            anchor_position=self.subtask_anchor_position,
            rng=rng,
        )
        return frame_indices, adjusted_fps

    def _sample_temporal_indices(
        self,
        total_frames: int,
        source_fps: float,
        subtask: InternDataSubtask,
        anchor_frame: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, float]:
        stride = self._target_stride(source_fps)
        if subtask.canonical_action in {"TURN_LEFT", "TURN_RIGHT"}:
            feasible_jitters = [
                jitter
                for jitter in range(-self.onset_jitter, self.onset_jitter + 1)
                if 0 <= anchor_frame - (self.subtask_start_position + jitter) * stride
                and anchor_frame - (self.subtask_start_position + jitter) * stride + (self.sequence_length - 1) * stride
                < total_frames
            ]
            if not feasible_jitters:
                raise ValueError("Turn sample lost its full history/future window")
            jitter = int(rng.choice(feasible_jitters))
        else:
            jitter = 0
        start = anchor_frame - (self.subtask_start_position + jitter) * stride
        indices = start + np.arange(self.sequence_length, dtype=np.int64) * stride
        if indices[0] < 0 or indices[-1] >= total_frames:
            raise ValueError(f"Strict temporal window [{indices[0]}, {indices[-1]}] exceeds [0, {total_frames - 1}]")
        return indices, source_fps / stride

    def _load_sample(self, episode: InternDataEpisode, subtask: InternDataSubtask, index: int) -> dict[str, Any]:
        dataframe = pd.read_parquet(episode.data_path)
        if self.pose_key not in dataframe.columns:
            raise KeyError(f"Pose key {self.pose_key!r} not found in {episode.data_path}")

        vr = VideoReader(str(episode.video_path), ctx=cpu(0), num_threads=2)
        source_fps = episode.source_fps
        if source_fps is None:
            try:
                source_fps = float(vr.get_avg_fps())
            except Exception:
                source_fps = 30.0
        total_frames = min(len(vr), len(dataframe))
        if episode.length is not None:
            total_frames = min(total_frames, int(episode.length))
        rng = np.random.default_rng(self.seed + index + random.randint(0, 2**16))
        anchor_frame = self.sample_anchor_frames[index]
        if self.temporal_mode:
            assert anchor_frame is not None
            frame_indices, adjusted_fps = self._sample_temporal_indices(
                total_frames, source_fps, subtask, anchor_frame, rng
            )
        else:
            frame_indices, adjusted_fps = self._sample_indices(total_frames, source_fps, subtask, rng)

        frame_data = vr.get_batch(frame_indices.tolist()).asnumpy().astype(np.uint8)
        vr.seek(0)
        del vr

        frames = torch.from_numpy(frame_data).permute(0, 3, 1, 2)
        frames = self.preprocess(frames)
        frames = torch.clamp(frames * 255.0, 0, 255).to(torch.uint8)
        video = frames.permute(1, 0, 2, 3).contiguous()

        reference_position = self.waypoint_reference_position if self.temporal_mode else 0
        ref_pose = _pose_matrix(dataframe.iloc[int(frame_indices[reference_position])][self.pose_key])
        semantic_end = min(
            total_frames - 1,
            subtask.end_frame + self.post_subtask_tail_frames * self._target_stride(source_fps),
        )
        if self.temporal_mode:
            target_frame_indices = np.asarray(
                [min(int(frame_indices[pos]), semantic_end) for pos in self.waypoint_positions],
                dtype=np.int64,
            )
            clamped_mask = target_frame_indices != frame_indices[np.asarray(self.waypoint_positions)]
        else:
            target_frame_indices = frame_indices[np.asarray(self.waypoint_positions)]
            clamped_mask = np.zeros(len(self.waypoint_positions), dtype=bool)
        target_poses = [
            _pose_matrix(dataframe.iloc[int(target_frame_idx)][self.pose_key])
            for target_frame_idx in target_frame_indices
        ]
        waypoints = _relative_ego_waypoints(target_poses, ref_pose, self.dx_scale, self.dy_scale, self.dyaw_scale)

        video_frame_loss_mask = np.ones(self.sequence_length, dtype=np.float32)
        if self.temporal_mode:
            video_frame_loss_mask[:] = 0.0
            generated_positions = np.arange(self.sequence_length) >= self.subtask_start_position
            semantically_valid = (frame_indices >= subtask.start_frame) & (frame_indices <= semantic_end)
            video_frame_loss_mask[generated_positions & semantically_valid] = 1.0

        mirror_source_is_safe = self.mirror_allowed_sources is None or episode.source_id in self.mirror_allowed_sources
        mirrored = bool(
            self.temporal_mode
            and mirror_source_is_safe
            and self.horizontal_flip_prob > 0
            and rng.random() < self.horizontal_flip_prob
        )
        caption = subtask.caption
        canonical_action = subtask.canonical_action
        if mirrored:
            video = torch.flip(video, dims=(-1,))
            waypoints[:, 0] *= -1
            waypoints[:, 2] *= -1
            caption = swap_left_right_text(caption)
            canonical_action = {
                "TURN_LEFT": "TURN_RIGHT",
                "TURN_RIGHT": "TURN_LEFT",
            }.get(canonical_action, canonical_action)

        _, _, h, w = video.shape
        return {
            "video": video,
            "ai_caption": caption,
            "fps": adjusted_fps,
            "image_size": torch.tensor([h, w, h, w]),
            "num_frames": self.sequence_length,
            "padding_mask": torch.zeros(1, h, w),
            "frame_indices": torch.from_numpy(frame_indices.copy()).long(),
            "ego_waypoints_norm": torch.from_numpy(waypoints),
            "ego_waypoints_mask": torch.ones(len(self.waypoint_positions), dtype=torch.float32),
            "video_frame_loss_mask": torch.from_numpy(video_frame_loss_mask),
            "waypoint_target_frame_indices": torch.from_numpy(target_frame_indices.copy()).long(),
            "waypoint_clamped_mask": torch.from_numpy(clamped_mask.copy()),
            "canonical_action": canonical_action,
            "source_id": episode.source_id,
            "scene_id": episode.scene_id,
            "mirrored": mirrored,
            "revised_instruction_conflict": subtask.revised_instruction_conflict,
            "episode_index": episode.episode_index,
            "subtask_index": subtask.task_index,
            "subtask_frame_bounds": torch.tensor([subtask.start_frame, subtask.end_frame], dtype=torch.long),
            "subtask_anchor_frame": int(
                anchor_frame if self.temporal_mode else frame_indices[self.subtask_anchor_position]
            ),
            "subtask_anchor_in_bounds": bool(
                subtask.start_frame
                <= int(anchor_frame if self.temporal_mode else frame_indices[self.subtask_anchor_position])
                <= subtask.end_frame
            ),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        try:
            episode, subtask = self.samples[index]
            return self._load_sample(episode, subtask, index)
        except Exception as e:
            self.num_failed_loads += 1
            log.warning(
                f"Failed to load VLN waypoint sample {index} (total failures: {self.num_failed_loads}): {e}\n"
                f"{traceback.format_exc()}",
                rank0_only=False,
            )
            return self[np.random.randint(len(self.samples))]


class DistributedActionSourceBalancedSampler(Sampler[int]):
    """Deterministic 2:1:1 sampler that cycles through every source bucket.

    Rare action/source buckets repeat as needed, while abundant buckets use a
    deterministic shuffled cursor that advances across epochs. This preserves
    the requested balance without repeatedly drawing the same random subset and
    starving valid straight clips.
    """

    _ACTION_PROBABILITIES = {
        "GO_STRAIGHT": 0.5,
        "TURN_LEFT": 0.25,
        "TURN_RIGHT": 0.25,
    }

    def __init__(
        self,
        dataset: InternDataVLNWaypointDataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ) -> None:
        if num_replicas is None:
            num_replicas = parallel_state.get_data_parallel_world_size() if parallel_state.is_initialized() else 1
        if rank is None:
            rank = parallel_state.get_data_parallel_rank() if parallel_state.is_initialized() else 0
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank={rank} must be in [0, {num_replicas})")
        self.dataset = dataset
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0
        self.num_samples = int(np.ceil(len(dataset) / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.buckets = self._build_buckets()
        self.action_sources = {
            action: sorted(source_id for bucket_action, source_id in self.buckets if bucket_action == action)
            for action in self._ACTION_PROBABILITIES
        }
        missing = [action for action, sources in self.action_sources.items() if not sources]
        if missing:
            raise ValueError(f"Balanced sampler requires all action buckets; missing {missing}")
        self.weights = self._build_weights()

    def _build_buckets(self) -> dict[tuple[str, str], list[int]]:
        buckets: dict[tuple[str, str], list[int]] = {}
        for index, (episode, subtask) in enumerate(self.dataset.samples):
            if subtask.canonical_action not in self._ACTION_PROBABILITIES:
                continue
            buckets.setdefault((subtask.canonical_action, episode.source_id), []).append(index)
        return buckets

    def _build_weights(self) -> torch.Tensor:
        weights = []
        for episode, subtask in self.dataset.samples:
            action = subtask.canonical_action
            if action not in self._ACTION_PROBABILITIES:
                weights.append(0.0)
                continue
            source_id = episode.source_id
            weight = (
                self._ACTION_PROBABILITIES[action]
                / len(self.action_sources[action])
                / len(self.buckets[(action, source_id)])
            )
            weights.append(weight)
        result = torch.tensor(weights, dtype=torch.double)
        return result / result.sum()

    def _action_draw_counts(self) -> dict[str, int]:
        counts = {
            action: int(self.total_size * probability) for action, probability in self._ACTION_PROBABILITIES.items()
        }
        remainder = self.total_size - sum(counts.values())
        for action in self._ACTION_PROBABILITIES:
            if remainder <= 0:
                break
            counts[action] += 1
            remainder -= 1
        return counts

    def _source_draw_counts(self, action: str, action_count: int, epoch: int) -> dict[str, int]:
        sources = self.action_sources[action]
        base, remainder = divmod(action_count, len(sources))
        counts = {source_id: base for source_id in sources}
        start = epoch % len(sources)
        for offset in range(remainder):
            counts[sources[(start + offset) % len(sources)]] += 1
        return counts

    def _source_draw_offset(self, action: str, action_count: int, source_id: str) -> int:
        return sum(
            self._source_draw_counts(action, action_count, previous_epoch)[source_id]
            for previous_epoch in range(self.epoch)
        )

    def _draw_bucket(
        self,
        action: str,
        source_id: str,
        count: int,
        start: int,
    ) -> list[int]:
        bucket = self.buckets[(action, source_id)]
        bucket_size = len(bucket)
        stable_bucket_seed = zlib.crc32(f"{action}\0{source_id}".encode())
        selected: list[int] = []
        cursor = start
        while len(selected) < count:
            cycle, cycle_offset = divmod(cursor, bucket_size)
            generator = torch.Generator()
            generator.manual_seed((self.seed + stable_bucket_seed + cycle * 104_729) % (2**63 - 1))
            permutation = torch.randperm(bucket_size, generator=generator).tolist()
            take = min(count - len(selected), bucket_size - cycle_offset)
            selected.extend(bucket[permutation[index]] for index in range(cycle_offset, cycle_offset + take))
            cursor += take
        return selected

    def __iter__(self):
        action_counts = self._action_draw_counts()
        indices: list[int] = []
        for action, action_count in action_counts.items():
            source_counts = self._source_draw_counts(action, action_count, self.epoch)
            for source_id, count in source_counts.items():
                start = self._source_draw_offset(action, action_count, source_id)
                indices.extend(self._draw_bucket(action, source_id, count, start))

        if len(indices) != self.total_size:
            raise RuntimeError(f"Balanced sampler produced {len(indices)} indices, expected {self.total_size}")
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(self.total_size, generator=generator).tolist()
        indices = [indices[index] for index in permutation]
        rank_indices = indices[self.rank : self.total_size : self.num_replicas]
        return iter(rank_indices[self.start_index :])

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def set_start_index(self, start_index: int) -> None:
        if not 0 <= start_index <= self.num_samples:
            raise ValueError(f"start_index={start_index} must be in [0, {self.num_samples}]")
        self.start_index = int(start_index)


def get_generic_dataloader(
    dataset: Dataset,
    batch_size: int = 1,
    sampler: Optional[Any] = None,
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable] = None,
    **kwargs,
) -> DataLoader:
    del kwargs
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )


def get_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=True,
        seed=0,
    )


def get_vln_action_source_balanced_sampler(dataset) -> DistributedActionSourceBalancedSampler:
    return DistributedActionSourceBalancedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        seed=0,
    )


def get_vln_waypoint_eval_sampler(dataset) -> DistributedSampler:
    return DistributedSampler(
        dataset,
        num_replicas=parallel_state.get_data_parallel_world_size(),
        rank=parallel_state.get_data_parallel_rank(),
        shuffle=False,
        seed=0,
    )


def get_vln_waypoint_dataloader(
    dataset_dir: str | Sequence[str],
    batch_size: int = 1,
    num_frames: int = 29,
    video_size: tuple[int, int] = (272, 480),
    num_workers: int = 8,
    target_fps: Optional[float] = 10.0,
    drop_last: bool = True,
) -> DataLoader:
    dataset = L(InternDataVLNWaypointDataset)(
        dataset_dir=dataset_dir,
        num_frames=num_frames,
        video_size=video_size,
        target_fps=target_fps,
    )
    return L(get_generic_dataloader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
