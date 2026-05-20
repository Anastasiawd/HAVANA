"""
obs_loader.py
-------------
Load observation frames from a folder of images.
Images are sorted by the last numeric group in the filename,
so both ``0.jpg`` and ``episode_000000_2.jpg`` styles work.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Sequence

from PIL import Image


def _numeric_key(p: Path) -> int:
    """Extract the last integer from a filename stem for sorting."""
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


def sorted_image_paths(
    folder: str | Path,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> List[Path]:
    """Return image file paths sorted by the trailing number in filename."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Directory not found: {folder}")

    files = [p for p in folder.iterdir() if p.suffix.lower() in extensions]
    files.sort(key=_numeric_key)
    return files


def load_all_images(
    folder: str | Path,
    extensions: Sequence[str] = (".jpg", ".jpeg", ".png"),
) -> List[Image.Image]:
    """Load *all* images from a folder, sorted by numeric filename.

    This is the generic loader used for both obs and goal folders.
    """
    paths = sorted_image_paths(folder, extensions=extensions)
    if len(paths) == 0:
        raise RuntimeError(f"No images found in {folder}")
    images = [Image.open(p).convert("RGB") for p in paths]
    print(f"[loader] Loaded {len(images)} images from {folder}")
    print(f"  first: {paths[0].name}  last: {paths[-1].name}")
    return images
