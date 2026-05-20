"""
goal_extractor.py
-----------------
Extract goal frames from a demonstration video or a folder of demo frames.

Key convention
~~~~~~~~~~~~~~
The demo video's **frame 0 ≈ the current observation's starting frame**.
Therefore, valid goal frames start from **frame_id >= 1**.

Supported inputs:
    - A video file  (e.g. ``demo.mp4``)
    - A folder of frame images  (e.g. ``demo_frames/0.jpg, 1.jpg, …``)

Goal selection strategies:
    - ``"next"``   : return only frame 1  (the immediate next goal)
    - ``"stride"`` : return frames 1, 1+stride, 1+2*stride, …
    - ``"last"``   : return only the last frame
    - ``"all"``    : return every frame from frame 1 onward
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Literal, Sequence

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _is_video_file(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _load_frames_from_video(video_path: Path) -> List[Image.Image]:
    """Decode *all* frames from a video file using OpenCV."""
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "OpenCV is required to decode video files. "
            "Install it with:  pip install opencv-python"
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames: List[Image.Image] = []
    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames decoded from video: {video_path}")
    return frames


def _load_frames_from_folder(
    folder: Path,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> List[Image.Image]:
    """Load all image frames from a folder, sorted numerically by filename."""
    if not folder.is_dir():
        raise FileNotFoundError(f"Demo frame folder not found: {folder}")

    paths: List[Path] = []
    for p in folder.iterdir():
        if p.suffix.lower() in extensions:
            paths.append(p)
    import re
    paths.sort(key=lambda p: int(re.findall(r"\d+", p.stem)[-1]) if re.findall(r"\d+", p.stem) else 0)

    if len(paths) == 0:
        raise RuntimeError(f"No image files found in {folder}")

    frames = [Image.open(p).convert("RGB") for p in paths]
    return frames


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_goal_frames(
    demo_video_path: str | Path,
    goal_strategy: Literal["next", "stride", "last", "all"] = "next",
    goal_stride: int = 5,
    image_extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> List[Image.Image]:
    """
    Extract goal frames from a demonstration video / frame folder.

    Frame 0 of the demo is treated as equivalent to the robot's current
    observation, so **all returned goal frames have frame_id >= 1**.

    Parameters
    ----------
    demo_video_path : str | Path
        Path to a video file **or** a folder of frame images.
    goal_strategy : {"next", "stride", "last", "all"}
        How to select goal frames from the demo.
    goal_stride : int
        Frame stride when ``goal_strategy == "stride"``.
    image_extensions : sequence of str
        Extensions to consider when loading from a folder.

    Returns
    -------
    list[PIL.Image.Image]
        The selected goal frame(s), each as an RGB PIL image.
    """
    path = Path(demo_video_path)

    # 1. Load all frames -------------------------------------------------
    if path.is_file() and _is_video_file(path):
        all_frames = _load_frames_from_video(path)
    elif path.is_dir():
        all_frames = _load_frames_from_folder(path, extensions=image_extensions)
    elif path.is_file():
        # single image treated as the only goal
        return [Image.open(path).convert("RGB")]
    else:
        raise FileNotFoundError(
            f"Demo video path does not exist: {demo_video_path}"
        )

    total = len(all_frames)
    print(f"[goal_extractor] Loaded {total} total demo frames from {path}")

    if total < 2:
        raise ValueError(
            "Demo video must have at least 2 frames "
            "(frame 0 = start, frame 1+ = goals)."
        )

    # 2. Skip frame 0 (== current observation) ---------------------------
    candidate_frames = all_frames[1:]  # frame_id 1 .. N-1
    candidate_ids = list(range(1, total))

    # 3. Apply selection strategy -----------------------------------------
    if goal_strategy == "next":
        selected_ids = [candidate_ids[0]]
    elif goal_strategy == "last":
        selected_ids = [candidate_ids[-1]]
    elif goal_strategy == "all":
        selected_ids = candidate_ids
    elif goal_strategy == "stride":
        selected_ids = candidate_ids[::goal_stride]
        # always include the last frame if not already
        if selected_ids[-1] != candidate_ids[-1]:
            selected_ids.append(candidate_ids[-1])
    else:
        raise ValueError(f"Unknown goal_strategy: {goal_strategy}")

    # map ids back to candidate_frames (id 1 -> index 0 in candidates)
    goals = [candidate_frames[fid - 1] for fid in selected_ids]

    print(
        f"[goal_extractor] Strategy='{goal_strategy}' → "
        f"selected {len(goals)} goal frame(s): ids={selected_ids}"
    )
    return goals
