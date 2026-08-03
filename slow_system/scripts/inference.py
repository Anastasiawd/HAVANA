#!/usr/bin/env python3
"""Direct inference script for slow system.

Run inference with image + instruction input directly.

Usage:
    # Single image + instruction:
    python slow_system/scripts/inference.py \
        --config slow_system/configs/inference.yaml \
        --image path/to/image1.jpg \
        --image path/to/image2.jpg \
        --instruction "Walk forward and turn left"
    
    # Or use config file settings:
    python slow_system/scripts/inference.py --config slow_system/configs/inference.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

_SLOW_ROOT = Path(__file__).resolve().parents[2] / "slow_system"
_SLOW_OSS_ROOT = _SLOW_ROOT / "packages" / "cosmos-oss"
# Prepend so the local slow_system/cosmos_predict2 tree (which contains
# inference_lora.py and the registered VLN experiments) takes priority over any
# other cosmos_predict2 install on sys.path (e.g. an editable venv .pth).
for p in (_SLOW_OSS_ROOT, _SLOW_ROOT):
    p_str = str(p)
    if p.is_dir():
        if p_str in sys.path:
            sys.path.remove(p_str)
        sys.path.insert(0, p_str)

LOG = logging.getLogger("havana.slow.inference")


def _setup_logger(level: int = logging.INFO) -> None:
    if LOG.handlers:
        LOG.setLevel(level)
        return
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter(
        "[%(asctime)s][%(name)s][%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    LOG.addHandler(h)
    LOG.setLevel(level)
    LOG.propagate = False


@dataclass
class InferenceConfig:
    # Checkpoint. `checkpoint_path` may either be a directory (joined with
    # `checkpoint_filename`) or a full path to the `.pt` file directly.
    checkpoint_path: str = "/home/wangdzh/Downloads/augmented_iter_26000"
    checkpoint_filename: str = "model_ema_bf16.pt"
    # Hydra experiment that matches this fine-tuned checkpoint's architecture
    # (state_t=8 -> 29 frames, reason1p1_7B online text encoder, no LoRA).
    experiment: str = "predict2_video2world_training_2b_vln_online_reason2_align_29f_10fps_272_480"
    config_file: str = "cosmos_predict2/_src/predict2/configs/video2world/config.py"
    # Local Cosmos-Reason1 text encoder checkpoint (dir or HF snapshot). Leave
    # empty to auto-discover from the HuggingFace cache (nvidia/Cosmos-Reason1-7B).
    reason1_ckpt_path: str = ""
    input_images: List[str] = None
    output_dir: str = "./outputs"
    save_video: bool = True
    save_frames: bool = False
    resolution: str = "272,480"       # H,W (training resolution of this checkpoint)
    num_conditional_frames: int = 2   # 1 (image2world) or 2 (video2world)
    num_output_frames: int = 29       # (state_t-1)*4+1 = 29 for state_t=8
    num_steps: int = 35
    guidance: int = 7
    seed: int = 0
    offload_diffusion_model: bool = True
    offload_text_encoder: bool = True
    offload_tokenizer: bool = True
    disable_guardrails: bool = True
    aggressive_unload: bool = False


def _load_config(path: Optional[Path]) -> InferenceConfig:
    cfg = InferenceConfig()
    if path is None:
        return cfg
    path = Path(path)
    if not path.exists():
        LOG.warning("Config %s not found — using defaults", path)
        return cfg
    data = yaml.safe_load(path.read_text()) or {}
    for k, v in data.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _load_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img)


def _parse_resolution(resolution: str) -> Tuple[int, int]:
    parts = [int(x) for x in resolution.split(",")]
    if len(parts) != 2:
        raise ValueError(f"resolution must be 'H,W'; got '{resolution}'")
    return parts[0], parts[1]


def _strip_to_rgb_u8(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _build_input_video(
    images: List[np.ndarray],
    resolution: str,
    num_conditional_frames: int,
    total_frames: int,
) -> torch.Tensor:
    H, W = _parse_resolution(resolution)
    cond_frames = []
    for f in images:
        img = Image.fromarray(_strip_to_rgb_u8(f)).resize((W, H), Image.BILINEAR)
        cond_frames.append(
            torch.from_numpy(np.asarray(img, dtype=np.uint8)).permute(2, 0, 1).contiguous()
        )
    # cosmos expects 4*(N-1)+1 raw frames for num_latent_conditional_frames=N.
    cond_count = max(1, 4 * (num_conditional_frames - 1) + 1)
    if not cond_frames:
        cond_frames = [torch.zeros(3, H, W, dtype=torch.uint8)] * cond_count
    elif len(cond_frames) < cond_count:
        cond_frames = [cond_frames[0]] * (cond_count - len(cond_frames)) + cond_frames
    else:
        cond_frames = cond_frames[-cond_count:]

    full = torch.zeros(3, total_frames, H, W, dtype=torch.uint8)
    for i, fr in enumerate(cond_frames):
        full[:, i] = fr
    if cond_count < total_frames:
        full[:, cond_count:] = cond_frames[-1].unsqueeze(1)
    return full.unsqueeze(0)  # (1, 3, T, H, W)


def _save_video(video: np.ndarray, output_path: Path, fps: int = 16) -> None:
    try:
        import cv2
    except ImportError:
        LOG.warning("OpenCV not available; skipping video save")
        return

    h, w = video.shape[1:3]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for frame in video:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)
    writer.release()
    LOG.info("Saved video to %s", output_path)


def _save_frames(video: np.ndarray, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(video):
        Image.fromarray(frame).save(output_dir / f"frame_{i:04d}.png")
    LOG.info("Saved %d frames to %s", len(video), output_dir)


def _resolve_checkpoint(cfg: InferenceConfig) -> Path:
    """Resolve the checkpoint file path.

    `checkpoint_path` may be a full path to a `.pt` file, or a directory that
    is joined with `checkpoint_filename`.
    """
    base = Path(cfg.checkpoint_path).expanduser()
    ckpt = base if base.suffix == ".pt" else base / cfg.checkpoint_filename
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return ckpt


def _setup_reason1_text_encoder(cfg: InferenceConfig) -> None:
    """Point LoraVideo2WorldInference at a local Cosmos-Reason1 checkpoint.

    This model uses an online reason1p1_7B text encoder whose weights are NOT
    stored in the diffusion checkpoint. `LoraVideo2WorldInference` loads it from
    `COSMOS_PREDICT2_REASON1_CKPT_PATH` or the HuggingFace cache. We set the env
    var explicitly when the user provides a path.
    """
    if cfg.reason1_ckpt_path:
        path = Path(cfg.reason1_ckpt_path).expanduser()
        if not path.exists():
            LOG.warning("reason1_ckpt_path %s does not exist", path)
        os.environ["COSMOS_PREDICT2_REASON1_CKPT_PATH"] = str(path)
        LOG.info("Using Reason1 text encoder from %s", path)
    elif "COSMOS_PREDICT2_REASON1_CKPT_PATH" not in os.environ:
        LOG.info(
            "reason1_ckpt_path not set; will auto-discover nvidia/Cosmos-Reason1-7B "
            "from the HuggingFace cache."
        )


def run_inference(
    cfg: InferenceConfig,
    images: List[np.ndarray],
    instruction: str,
) -> np.ndarray:
    from cosmos_oss.init import init_environment
    from cosmos_predict2.inference_lora import LoraVideo2WorldInference

    init_environment()
    os.environ.setdefault("COSMOS_PREDICT2_LORA_AUTO_OFFLOAD", "1")
    _setup_reason1_text_encoder(cfg)

    ckpt_path = _resolve_checkpoint(cfg)
    config_file = cfg.config_file or "cosmos_predict2/_src/predict2/configs/video2world/config.py"

    LOG.info("Loading experiment '%s'", cfg.experiment)
    LOG.info("Loading checkpoint %s", ckpt_path)
    t0 = time.time()
    pipe = LoraVideo2WorldInference(
        experiment_name=cfg.experiment,
        ckpt_path=str(ckpt_path),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=config_file,
        offload_diffusion_model=cfg.offload_diffusion_model,
        offload_text_encoder=cfg.offload_text_encoder,
        offload_tokenizer=cfg.offload_tokenizer,
    )
    LOG.info("Model loaded in %.2fs", time.time() - t0)

    try:
        total_frames = int(
            pipe.model.tokenizer.get_pixel_num_frames(pipe.model.config.state_t)
        )
    except Exception:
        total_frames = cfg.num_output_frames

    vid_input = _build_input_video(
        images, cfg.resolution, cfg.num_conditional_frames, total_frames
    )

    LOG.info("Running generation...")
    t0 = time.time()
    video_tensor = pipe.generate_vid2world(
        prompt=instruction,
        input_path=vid_input,
        guidance=cfg.guidance,
        num_video_frames=cfg.num_output_frames,
        num_latent_conditional_frames=cfg.num_conditional_frames,
        resolution=cfg.resolution,
        seed=cfg.seed,
        num_steps=cfg.num_steps,
    )
    elapsed = time.time() - t0
    LOG.info("Generation completed in %.2fs", elapsed)

    video_np = (video_tensor[0].permute(1, 2, 3, 0).cpu().numpy() * 0.5 + 0.5) * 255
    video_np = np.clip(video_np, 0, 255).astype(np.uint8)

    if cfg.aggressive_unload:
        del pipe
        torch.cuda.empty_cache()

    return video_np


def main():
    parser = argparse.ArgumentParser(description="Slow system direct inference")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--image", "-i", action="append", dest="images",
                        help="Input image path (repeat for multiple images)")
    parser.add_argument("--instruction", "-t", type=str, help="Navigation instruction")
    parser.add_argument("--output-dir", "-o", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, help="Random seed")
    args = parser.parse_args()

    _setup_logger()

    cfg = _load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed

    images = []
    if args.images:
        images = [_load_image(p) for p in args.images]
    elif cfg.input_images:
        images = [_load_image(p) for p in cfg.input_images]

    if len(images) == 0:
        LOG.error("No input images provided")
        sys.exit(1)

    if len(images) < cfg.num_conditional_frames:
        LOG.warning("Only %d images provided, but num_conditional_frames=%d",
                    len(images), cfg.num_conditional_frames)

    instruction = args.instruction
    if not instruction:
        LOG.error("No instruction provided")
        sys.exit(1)

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Input: %d images, instruction='%s'", len(images), instruction)

    video = run_inference(cfg, images, instruction)

    LOG.info("Generated video shape: %s", video.shape)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if cfg.save_video:
        _save_video(video, output_dir / f"generated_{timestamp}.mp4")
    if cfg.save_frames:
        _save_frames(video, output_dir / f"frames_{timestamp}")

    LOG.info("Done!")


if __name__ == "__main__":
    main()
