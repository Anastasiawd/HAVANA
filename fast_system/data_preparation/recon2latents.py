#!/usr/bin/env python3
"""
Offline RECON image-sequence feature extraction.

Expected RECON layout:
  recon_extracted/
    <trajectory_id>/
      images/
        0000.jpg
        0001.jpg
        ...

The encoder boundary is intentionally small: use DummyVideoEncoder for fast
pipeline checks or VJEPA2VideoEncoder for real V-JEPA2 patch tokens.
"""

from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_DIR_NAMES = {"images", "image", "rgb", "rgb_left", "rgb_right", "images_left", "images_right"}
DEFAULT_VJEPA2_ROOT = Path("/home/wangdzh/ffnav/vjepa2")
DEFAULT_VJEPA2_CHECKPOINT = Path(
    "/home/wangdzh/ffnav/havana/data_preparation/model_weights/vjepa2_1_vitb_dist_vitG_384.pt"
)
torch = None
F = None


def require_torch() -> None:
    global torch, F
    if torch is not None and F is not None:
        return
    try:
        import torch as torch_module
        import torch.nn.functional as functional_module
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required to extract and save video features. "
            "Please run this script in an environment with torch installed."
        ) from exc
    torch = torch_module
    F = functional_module


@dataclass
class VideoRecord:
    video_id: str
    video_path: Path
    frame_paths: List[Path]


class BaseVideoEncoder:
    name = "base"

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        """Encode one clip.

        Args:
            clip: Tensor with shape [T, C, H, W], RGB, float32, normalized to [0, 1].

        Returns:
            A feature tensor. Concrete encoders define the exact output shape.
        """
        raise NotImplementedError


class DummyVideoEncoder(BaseVideoEncoder):
    """Deterministic lightweight placeholder encoder.

    It converts each frame to grayscale and adaptively pools it into K tokens
    with D dimensions, producing [T, K, D]. This keeps the offline pipeline
    debuggable without depending on V-JEPA2.
    """

    name = "dummy"

    def __init__(self, device: torch.device, num_tokens: int = 8, feature_dim: int = 256):
        self.device = device
        self.num_tokens = num_tokens
        self.feature_dim = feature_dim

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        require_torch()
        with torch.inference_mode():
            if clip.ndim != 4:
                raise ValueError(f"Expected clip shape [T, C, H, W], got {tuple(clip.shape)}")
            if clip.shape[1] != 3:
                raise ValueError(f"Expected RGB clip with C=3, got C={clip.shape[1]}")

            clip = clip.to(self.device, non_blocking=True).float()
            gray = clip.mean(dim=1, keepdim=True)
            feature = F.adaptive_avg_pool2d(gray, (self.num_tokens, self.feature_dim)).squeeze(1)
            return feature.contiguous()


class VJEPA2VideoEncoder(BaseVideoEncoder):
    name = "vjepa2"

    def __init__(
        self,
        device: torch.device,
        repo: Path | str,
        checkpoint: Path,
        model_name: str = "vjepa2_1_vit_base_384",
        checkpoint_key: str = "ema_encoder",
        num_frames: int = 16,
    ):
        require_torch()
        self.device = device
        self.repo = repo
        self.checkpoint = checkpoint
        self.model_name = model_name
        self.checkpoint_key = checkpoint_key
        self.name = f"vjepa2:{model_name}"

        self.model = self._load_model(num_frames=num_frames)
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.embed_dim = int(getattr(self.model, "embed_dim", 0))
        self.patch_size = int(getattr(self.model, "patch_size", 16))
        self.tubelet_size = int(getattr(self.model, "tubelet_size", 2))

        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.mean = mean.to(self.device)
        self.std = std.to(self.device)

    def _load_model(self, num_frames: int):
        repo = Path(self.repo).expanduser()
        source = "local" if repo.exists() else "github"
        repo_arg = str(repo) if source == "local" else str(self.repo)

        logging.info(
            "Loading V-JEPA2 model %s from %s (source=%s, num_frames=%d).",
            self.model_name,
            repo_arg,
            source,
            num_frames,
        )
        loaded = torch.hub.load(
            repo_arg,
            self.model_name,
            source=source,
            pretrained=False,
            num_frames=num_frames,
        )
        if isinstance(loaded, (tuple, list)):
            model = loaded[0]
        else:
            model = loaded
        del loaded

        checkpoint_path = self.checkpoint.expanduser()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"V-JEPA2 checkpoint not found: {checkpoint_path}")

        logging.info("Loading V-JEPA2 weights from %s [%s].", checkpoint_path, self.checkpoint_key)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if self.checkpoint_key not in checkpoint:
            available = ", ".join(sorted(checkpoint.keys()))
            raise KeyError(f"Checkpoint key {self.checkpoint_key!r} not found. Available keys: {available}")

        state_dict = checkpoint[self.checkpoint_key]
        state_dict = {
            key.replace("module.", "").replace("backbone.", ""): value
            for key, value in state_dict.items()
        }
        msg = model.load_state_dict(state_dict, strict=False)
        missing = getattr(msg, "missing_keys", [])
        unexpected = getattr(msg, "unexpected_keys", [])
        if missing or unexpected:
            logging.warning(
                "Loaded V-JEPA2 with missing=%d unexpected=%d keys.",
                len(missing),
                len(unexpected),
            )
            if missing:
                logging.debug("Missing keys: %s", missing)
            if unexpected:
                logging.debug("Unexpected keys: %s", unexpected)
        else:
            logging.info("V-JEPA2 checkpoint loaded with all keys matched.")

        del checkpoint, state_dict
        return model

    def encode(self, clip: torch.Tensor) -> torch.Tensor:
        require_torch()
        if clip.ndim != 4:
            raise ValueError(f"Expected clip shape [T, C, H, W], got {tuple(clip.shape)}")
        if clip.shape[1] != 3:
            raise ValueError(f"Expected RGB clip with C=3, got C={clip.shape[1]}")

        with torch.inference_mode():
            # Official eval feeds V-JEPA2 clips as [B, C, T, H, W] with ImageNet normalization.
            x = clip.to(self.device, non_blocking=True).float()
            x = x.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            x = (x - self.mean) / self.std
            tokens = self.model(x).squeeze(0).contiguous()

            t_tokens = max(1, int(clip.shape[0]) // self.tubelet_size)
            h_tokens = int(clip.shape[2]) // self.patch_size
            w_tokens = int(clip.shape[3]) // self.patch_size
            spatial_tokens = h_tokens * w_tokens
            if spatial_tokens > 0 and tokens.shape[0] == t_tokens * spatial_tokens:
                tokens = tokens.view(t_tokens, spatial_tokens, tokens.shape[-1])
            return tokens.contiguous()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract offline video features/latents from RECON images.")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "RECON extracted root containing many <trajectory_id>/images folders, "
            "or one single trajectory images folder."
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for .pt clip features.")
    parser.add_argument("--fps", type=float, default=8.0, help="Target resampled FPS.")
    parser.add_argument(
        "--source-fps",
        type=float,
        default=0,
        help="FPS of the extracted image sequence before resampling. Set <= 0 to keep all frames.",
    )
    parser.add_argument("--clip-frames", type=int, default=16, help="Frames per clip.")
    parser.add_argument("--stride-frames", type=int, default=8, help="Sliding-window stride in resampled frames.")
    parser.add_argument("--height", type=int, default=256, help="Resize height.")
    parser.add_argument("--width", type=int, default=256, help="Resize width.")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device for encoder inference.")
    parser.add_argument("--encoder", choices=["dummy", "vjepa2"], default="dummy", help="Video encoder backend.")
    parser.add_argument("--vjepa2-root", type=Path, default=DEFAULT_VJEPA2_ROOT, help="Local V-JEPA2 repo path.")
    parser.add_argument(
        "--vjepa2-checkpoint",
        type=Path,
        default=DEFAULT_VJEPA2_CHECKPOINT,
        help="V-JEPA2 checkpoint path.",
    )
    parser.add_argument(
        "--vjepa2-model-name",
        default="vjepa2_1_vit_base_384",
        help="torch.hub model name, e.g. vjepa2_1_vit_base_384.",
    )
    parser.add_argument(
        "--vjepa2-checkpoint-key",
        default="ema_encoder",
        help="Checkpoint state-dict key for the encoder weights.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing clip .pt files.")
    parser.add_argument("--max-videos", type=int, default=None, help="Only process the first N videos for debug.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_root.exists():
        raise FileNotFoundError(f"--data-root does not exist: {args.data_root}")
    if args.fps <= 0:
        raise ValueError("--fps must be > 0")
    if args.clip_frames <= 0:
        raise ValueError("--clip-frames must be > 0")
    if args.stride_frames <= 0:
        raise ValueError("--stride-frames must be > 0")
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be > 0")
    if args.max_videos is not None and args.max_videos < 0:
        raise ValueError("--max-videos must be >= 0")
    if args.encoder == "vjepa2":
        if args.height % 16 != 0 or args.width % 16 != 0:
            raise ValueError("V-JEPA2 expects --height and --width to be multiples of patch size 16")
        if args.clip_frames < 2:
            raise ValueError("V-JEPA2 video encoder expects --clip-frames >= 2")
        if args.vjepa2_root != DEFAULT_VJEPA2_ROOT and not args.vjepa2_root.exists():
            logging.warning(
                "V-JEPA2 root %s does not exist; torch.hub will treat it as a remote repo id.",
                args.vjepa2_root,
            )
        if not args.vjepa2_checkpoint.exists():
            raise FileNotFoundError(f"--vjepa2-checkpoint does not exist: {args.vjepa2_checkpoint}")
        if (args.height, args.width) != (384, 384):
            logging.warning(
                "Using V-JEPA2 2.1 384 model with resize %dx%d. Official eval uses 384x384.",
                args.height,
                args.width,
            )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def sanitize_video_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("._") or "video"


def infer_video_id(frame_dir: Path, data_root: Path, used_ids: Dict[str, int]) -> str:
    if frame_dir.name.lower() in IMAGE_DIR_NAMES and frame_dir.parent != frame_dir:
        if frame_dir.name.lower() == "images":
            base = frame_dir.parent.name
        else:
            base = f"{frame_dir.parent.name}_{frame_dir.name}"
    else:
        try:
            rel = frame_dir.relative_to(data_root)
            base = data_root.name if str(rel) == "." else str(rel)
        except ValueError:
            base = frame_dir.name

    base = sanitize_video_id(base.replace("/", "__"))
    count = used_ids.get(base, 0)
    used_ids[base] = count + 1
    return base if count == 0 else f"{base}_{count:04d}"


def discover_videos(data_root: Path) -> List[VideoRecord]:
    groups: Dict[Path, List[Path]] = {}
    for path in data_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            groups.setdefault(path.parent, []).append(path)

    used_ids: Dict[str, int] = {}
    videos: List[VideoRecord] = []
    for frame_dir in sorted(groups):
        frames = sorted(groups[frame_dir], key=natural_key)
        if not frames:
            continue
        video_id = infer_video_id(frame_dir, data_root, used_ids)
        videos.append(VideoRecord(video_id=video_id, video_path=frame_dir, frame_paths=frames))
    return videos


def resolve_device(device_name: str) -> torch.device:
    require_torch()
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    return device


def resample_frame_paths(
    frame_paths: Sequence[Path],
    target_fps: float,
    source_fps: Optional[float],
) -> List[Path]:
    if source_fps is None or source_fps <= 0 or target_fps >= source_fps:
        return list(frame_paths)

    sampled: List[Path] = []
    seen = set()
    step = source_fps / target_fps
    sample_pos = 0.0

    while True:
        idx = int(round(sample_pos))
        if idx >= len(frame_paths):
            break
        if idx not in seen:
            sampled.append(frame_paths[idx])
            seen.add(idx)
        sample_pos += step

    return sampled


def iter_clip_windows(num_frames: int, clip_frames: int, stride_frames: int) -> List[Tuple[int, int]]:
    if num_frames < clip_frames:
        return []
    return [(start, start + clip_frames) for start in range(0, num_frames - clip_frames + 1, stride_frames)]


def load_clip(frame_paths: Sequence[Path], height: int, width: int) -> torch.Tensor:
    require_torch()
    frames = []
    resample_mode = getattr(Image, "Resampling", Image).BILINEAR

    for path in frame_paths:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image = image.resize((width, height), resample=resample_mode)
                array = np.asarray(image, dtype=np.float32) / 255.0
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            raise RuntimeError(f"failed to read image {path}: {exc}") from exc

        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        frames.append(tensor)

    return torch.stack(frames, dim=0)


def build_encoder(args: argparse.Namespace, device: torch.device) -> BaseVideoEncoder:
    if args.encoder == "dummy":
        return DummyVideoEncoder(device=device)
    if args.encoder == "vjepa2":
        return VJEPA2VideoEncoder(
            device=device,
            repo=args.vjepa2_root,
            checkpoint=args.vjepa2_checkpoint,
            model_name=args.vjepa2_model_name,
            checkpoint_key=args.vjepa2_checkpoint_key,
            num_frames=args.clip_frames,
        )
    raise ValueError(f"Unknown encoder: {args.encoder}")


def process_video(
    video: VideoRecord,
    args: argparse.Namespace,
    encoder: BaseVideoEncoder,
) -> Tuple[int, int, int, int]:
    resampled_frames = resample_frame_paths(video.frame_paths, args.fps, args.source_fps)
    windows = iter_clip_windows(len(resampled_frames), args.clip_frames, args.stride_frames)

    if not windows:
        logging.info(
            "%s: %d frames -> %d resampled frames, skipped because it is shorter than one clip.",
            video.video_id,
            len(video.frame_paths),
            len(resampled_frames),
        )
        return 0, 0, 0, 0

    out_video_dir = args.out_dir / video.video_id
    out_video_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_existing = 0
    failed = 0

    for clip_index, (start, end) in enumerate(windows):
        out_path = out_video_dir / f"clip_{clip_index:06d}.pt"
        if out_path.exists() and not args.overwrite:
            skipped_existing += 1
            continue

        try:
            clip = load_clip(resampled_frames[start:end], args.height, args.width)
            feature = encoder.encode(clip)
            payload = {
                "feature": feature.cpu(),
                "video_path": str(video.video_path),
                "video_id": video.video_id,
                "clip_index": clip_index,
                "start_frame_resampled": start,
                "end_frame_resampled": end - 1,
                "fps": float(args.fps),
                "source_fps": float(args.source_fps) if args.source_fps and args.source_fps > 0 else None,
                "clip_frames": int(args.clip_frames),
                "stride_frames": int(args.stride_frames),
                "height": int(args.height),
                "width": int(args.width),
                "encoder": encoder.name,
            }
            torch.save(payload, out_path)
            saved += 1
        except Exception as exc:
            failed += 1
            logging.warning("%s clip %06d skipped: %s", video.video_id, clip_index, exc)
        finally:
            if "clip" in locals():
                del clip
            if "feature" in locals():
                del feature

    logging.info(
        "%s: %d frames -> %d resampled frames -> %d clips (%d saved, %d existing, %d failed).",
        video.video_id,
        len(video.frame_paths),
        len(resampled_frames),
        len(windows),
        saved,
        skipped_existing,
        failed,
    )
    return len(windows), saved, skipped_existing, failed


def main() -> None:
    setup_logging()
    args = parse_args()
    validate_args(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(args.data_root)
    if args.max_videos is not None:
        videos = videos[: args.max_videos]

    logging.info("Found %d video(s) under %s.", len(videos), args.data_root)
    if not videos:
        logging.warning("No image files found. Supported extensions: %s", sorted(IMAGE_EXTENSIONS))
        return

    require_torch()
    device = resolve_device(args.device)
    encoder = build_encoder(args, device)
    logging.info("Using encoder=%s on device=%s.", encoder.name, device)

    total_clips = 0
    total_saved = 0
    total_existing = 0
    total_failed = 0

    for video in tqdm(videos, desc="Videos", unit="video"):
        try:
            num_clips, saved, existing, failed = process_video(video, args, encoder)
        except Exception as exc:
            logging.warning("%s skipped because processing failed: %s", video.video_id, exc)
            continue
        total_clips += num_clips
        total_saved += saved
        total_existing += existing
        total_failed += failed

    logging.info(
        "Done. Generated %d clip window(s); saved %d new .pt file(s), skipped %d existing, failed %d.",
        total_clips,
        total_saved,
        total_existing,
        total_failed,
    )


if __name__ == "__main__":
    main()
