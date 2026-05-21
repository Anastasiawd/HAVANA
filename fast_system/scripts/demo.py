#!/usr/bin/env python3
"""Run the Stage-1 + Stage-2 pipeline on an arbitrary MP4 and dump a side-by-side video.

Pipeline:
    arbitrary mp4  --[uniformly sample N frames]--> Stage-1 --> z_g
                                                                  |
                                  for each frame t of the input video:
                                      obs_history = video[t-3..t]  --+--> Stage-2 --> trajectory
                                                                  |
                              left = current frame, right = BEV with predicted trajectory

There is no GT for an arbitrary mp4, so only the prediction (red) is shown
on the BEV. Origin = robot at the current frame; x is forward, +y is left
(displayed on the left side of the plot).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

FAST_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(FAST_SYSTEM_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(FAST_SYSTEM_ROOT.parent))

from fast_system.scripts.eval_pipeline import (
    action_stats_from_cfg,
    load_stage1,
    load_stage2,
    make_autocast,
    render_bev,
    sample_traj,
)
from fast_system.trainer.train_stage_2 import load_config, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the slow+fast pipeline on a single MP4 or image folder.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video-path", type=Path, default=None, help="Input MP4 file (any length / fps).")
    src.add_argument("--image-folder", type=Path, default=None,
                     help="Folder of jpg/png/bmp/webp frames (read in natural order).")
    p.add_argument("--source-fps", type=float, default=None,
                   help="Frame rate to assume for --image-folder (required in folder mode).")
    p.add_argument("--output", type=Path, default=None,
                   help="Output MP4 path (default: <input>.demo.mp4).")
    p.add_argument("--config", type=Path, default=FAST_SYSTEM_ROOT / "configs" / "stage2.yaml")
    p.add_argument("--stage1-ckpt", type=Path, default=FAST_SYSTEM_ROOT / "model_weights" / "stage1.pt")
    p.add_argument("--stage2-ckpt", type=Path, default=FAST_SYSTEM_ROOT / "model_weights" / "stage2.pt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cfm-steps", type=int, default=10)
    p.add_argument("--num-trajectory-samples", type=int, default=4)
    p.add_argument("--route-window", type=int, default=16,
                   help="Number of consecutive (target-fps) frames forming the route window "
                        "fed to Stage-1 at each timestep. Matches training (16 frames @ 4 fps = 4 s).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Truncate input to first N frames (debug / speed).")
    p.add_argument("--target-fps", type=float, default=5.8,
                   help="Resample the input to this fps before running the pipeline. "
                        "Stage-2 was trained at 4 fps; matching this avoids OOD obs context. "
                        "Set to 0 to keep the source fps unchanged.")
    p.add_argument("--frame-stride", type=int, default=4,
                   help="z_g segment length, in target-fps frames. Stage-1 re-encodes z_g once "
                        "every K processed frames; Stage-2 still runs on every processed frame "
                        "but reuses the segment's z_g. K = how long the slow system 'looks ahead' "
                        "before re-planning. Default 4 (= 1 s @ 4 fps, matches training fast_offsets).")
    p.add_argument("--inference-batch", type=int, default=16,
                   help="Number of obs frames pushed through Stage-2 per forward pass.")
    p.add_argument("--output-fps", type=float, default=None,
                   help="Output mp4 fps (default: source fps).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--display-debug", action="store_true",
                   help="Overlay intermediate Stage-1 outputs (route anchors / coarse goal / "
                        "turn probabilities / z_g + sigma heatmaps) on the rendered video. "
                        "Adds an extra side panel; baseline flow is untouched when off.")
    return p.parse_args()


def read_video(path: Path, max_frames: int | None) -> tuple[np.ndarray, float]:
    """Return (frames[T,H,W,3] uint8, fps)."""
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 30.0))
    frames: list[np.ndarray] = []
    try:
        for i, frame in enumerate(reader):
            if max_frames is not None and i >= int(max_frames):
                break
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            elif frame.shape[-1] == 4:
                frame = frame[..., :3]
            frames.append(np.ascontiguousarray(frame, dtype=np.uint8))
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames, axis=0), fps


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _natural_key(path: Path) -> list[object]:
    import re
    parts = re.split(r"(\d+)", path.name)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def read_image_folder(folder: Path, max_frames: int | None) -> np.ndarray:
    """Return frames[T,H,W,3] uint8 from a folder of jpg/png images, sorted naturally."""
    paths = sorted(
        (p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES),
        key=_natural_key,
    )
    if not paths:
        raise FileNotFoundError(f"No image files found in {folder}")
    if max_frames is not None:
        paths = paths[: int(max_frames)]
    frames: list[np.ndarray] = []
    target_shape: tuple[int, int] | None = None
    for path in paths:
        with Image.open(path) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.uint8)
        if target_shape is None:
            target_shape = arr.shape[:2]
        elif arr.shape[:2] != target_shape:
            arr = np.asarray(
                Image.fromarray(arr).resize((target_shape[1], target_shape[0]), Image.BILINEAR),
                dtype=np.uint8,
            )
        frames.append(np.ascontiguousarray(arr))
    return np.stack(frames, axis=0)


def to_tensor_chw(frame_uint8: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    img = Image.fromarray(frame_uint8)
    if img.size != size:
        img = img.resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def build_obs_context(
    chw_frames: list[torch.Tensor],
    current_idx: int,
    context_size: int,
    context_dt_frames: int,
) -> torch.Tensor:
    """Mirror dataset behaviour: take frames at [t - C*dt, ..., t-dt, t], pad-by-clamp."""
    n = len(chw_frames)
    indices = [current_idx - off * context_dt_frames for off in range(context_size, -1, -1)]
    indices = [min(max(i, 0), n - 1) for i in indices]
    return torch.stack([chw_frames[i] for i in indices], dim=0)


TURN_NAMES = ("left", "straight", "right")


def render_bev_with_debug(
    pred_xy_m: np.ndarray,
    side_px: int,
    extent: float,
    anchors_xy: np.ndarray | None,
    goal_xy: np.ndarray | None,
    title: str,
) -> np.ndarray:
    """Like render_bev() but also overlays Stage-1 anchors and coarse goal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = plt.figure(figsize=(4, 4), dpi=max(64, side_px // 4))
    ax = fig.add_subplot(111)
    ax.plot([0.0, *pred_xy_m[:, 1].tolist()], [0.0, *pred_xy_m[:, 0].tolist()],
            "-o", color="#d62728", markersize=3, linewidth=2, label="Pred")
    if anchors_xy is not None and anchors_xy.size > 0:
        ax.plot([0.0, *anchors_xy[:, 1].tolist()], [0.0, *anchors_xy[:, 0].tolist()],
                "--", color="#ff7f0e", linewidth=1.5, alpha=0.9, label="anchors (z_g)")
        ax.scatter(anchors_xy[:, 1], anchors_xy[:, 0], c="#ff7f0e", s=15, zorder=4)
    if goal_xy is not None:
        ax.scatter([goal_xy[1]], [goal_xy[0]], marker="*", c="#ffbf00",
                   s=140, edgecolors="black", linewidths=0.8, zorder=6, label="goal (z_g)")
    ax.scatter([0.0], [0.0], c="k", marker="s", s=50, zorder=5, label="origin")
    ax.set_xlim(extent, -extent)
    ax.set_ylim(-0.5, extent * 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("y (left, m)")
    ax.set_ylabel("x (forward, m)")
    ax.legend(loc="upper left", fontsize=7)
    ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.4)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
    plt.close(fig)
    return np.asarray(Image.fromarray(img).resize((side_px, side_px), Image.BILINEAR))


def render_debug_panel(
    seg_data: dict[str, np.ndarray],
    side_px: int,
    z_value_range: tuple[float, float],
    sigma_value_range: tuple[float, float],
    seg_id: int,
) -> np.ndarray:
    """Vertical stack: turn-prob bars, z_g heatmap strip, sigma heatmap strip, brief stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = plt.figure(figsize=(3.2, 4), dpi=max(64, side_px // 4))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.2, 1.0, 1.0, 0.7], hspace=0.5)

    ax_turn = fig.add_subplot(gs[0])
    turn_probs = seg_data["turn_probs"]
    ax_turn.bar(np.arange(len(turn_probs)), turn_probs,
                color=["#1f77b4", "#2ca02c", "#d62728"], edgecolor="black", linewidth=0.5)
    ax_turn.set_xticks(np.arange(len(turn_probs)))
    ax_turn.set_xticklabels(TURN_NAMES[: len(turn_probs)], fontsize=8)
    ax_turn.set_ylim(0, 1.05)
    ax_turn.set_title("Stage-1 turn prob", fontsize=9)
    ax_turn.tick_params(axis="y", labelsize=7)

    ax_z = fig.add_subplot(gs[1])
    ax_z.imshow(seg_data["z_g"][None, :], aspect="auto", cmap="coolwarm",
                vmin=z_value_range[0], vmax=z_value_range[1])
    ax_z.set_yticks([])
    ax_z.set_xticks([])
    ax_z.set_title(f"z_g (dim={seg_data['z_g'].size})", fontsize=9)

    ax_s = fig.add_subplot(gs[2])
    sigma = np.exp(0.5 * seg_data["logvar"])
    ax_s.imshow(sigma[None, :], aspect="auto", cmap="viridis",
                vmin=sigma_value_range[0], vmax=sigma_value_range[1])
    ax_s.set_yticks([])
    ax_s.set_xticks([])
    ax_s.set_title("VIB sigma (uncertainty per dim)", fontsize=9)

    ax_t = fig.add_subplot(gs[3])
    ax_t.axis("off")
    text = (
        f"seg #{seg_id}\n"
        f"z_g  mean={seg_data['z_g'].mean():+.3f}  std={seg_data['z_g'].std():.3f}\n"
        f"sigma mean={sigma.mean():.3f}  max={sigma.max():.3f}\n"
        f"goal_pred={seg_data['goal'][0]:+.2f}, {seg_data['goal'][1]:+.2f} m"
    )
    ax_t.text(0.0, 1.0, text, fontsize=8, va="top", family="monospace")

    fig.tight_layout(pad=0.4)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
    plt.close(fig)
    return np.asarray(Image.fromarray(img).resize((int(side_px * 0.8), side_px), Image.BILINEAR))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    seed_everything(int(args.seed))
    device = resolve_device(args.device or str(cfg.get("device", "cuda")))
    use_amp = (not args.no_amp) and device.type == "cuda"

    if args.video_path is not None:
        input_path = args.video_path
        print(f"[demo] reading video {input_path}", flush=True)
        frames_u8_src, src_fps = read_video(input_path, args.max_frames)
    else:
        input_path = args.image_folder
        if args.source_fps is None:
            raise SystemExit("--source-fps is required when using --image-folder.")
        print(f"[demo] reading image folder {input_path} @ {args.source_fps:.2f} fps", flush=True)
        frames_u8_src = read_image_folder(input_path, args.max_frames)
        src_fps = float(args.source_fps)

    if args.output is not None:
        output_path = args.output
    else:
        anchor = input_path.resolve()
        stem = anchor.stem if args.video_path is not None else anchor.name
        output_path = anchor.parent / f"{stem}.demo.mp4"
    if output_path.exists() and output_path.is_dir():
        raise SystemExit(f"--output must be a file path, not a directory: {output_path}")
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    n_src = int(frames_u8_src.shape[0])

    # Resample to target_fps so obs context spacing matches training.
    if args.target_fps and args.target_fps > 0 and abs(src_fps - args.target_fps) > 1e-3:
        step = src_fps / float(args.target_fps)
        proc_src_idx = np.unique(np.floor(np.arange(0, n_src, step)).astype(int))
        proc_src_idx = proc_src_idx[proc_src_idx < n_src]
        frames_u8 = frames_u8_src[proc_src_idx]
        effective_fps = float(args.target_fps)
        print(f"[demo] resampled {n_src} src frames @ {src_fps:.2f} fps -> "
              f"{len(frames_u8)} frames @ {effective_fps:.2f} fps", flush=True)
    else:
        frames_u8 = frames_u8_src
        proc_src_idx = np.arange(n_src)
        effective_fps = src_fps
        print(f"[demo] {n_src} frames @ {src_fps:.2f} fps (no resampling)", flush=True)

    n_total = int(frames_u8.shape[0])
    print(f"[demo] -> output: {output_path}", flush=True)

    data_cfg = cfg.get("scand_data", {})
    obs_size_v = int(data_cfg.get("obs_image_size", 256))
    goal_size_v = int(data_cfg.get("goal_image_size", 256))
    obs_size = (obs_size_v, obs_size_v)
    goal_size = (goal_size_v, goal_size_v)
    context_size = int(data_cfg.get("context_size", cfg.get("model", {}).get("context_size", 3)))
    context_dt_frames = int(data_cfg.get("context_dt_frames", 1))
    horizon = int(data_cfg.get("action_horizon", 16))
    action_dim = int(cfg.get("model", {}).get("action_dim", 2))
    waypoint_spacing = float(data_cfg.get("metric_waypoint_spacing", 0.38))
    pred_to_meter = waypoint_spacing if bool(data_cfg.get("normalize_actions", True)) else 1.0

    print(f"[demo] resizing frames: obs={obs_size} goal={goal_size}", flush=True)
    obs_chw_src = [to_tensor_chw(f, obs_size) for f in frames_u8_src]
    goal_chw = [to_tensor_chw(f, goal_size) for f in frames_u8]

    stage1 = load_stage1(args.stage1_ckpt, device)
    stage2 = load_stage2(args.stage2_ckpt, cfg, device)
    action_stats = action_stats_from_cfg(cfg)

    window = max(2, int(args.route_window))
    src_dt = max(1, int(round(src_fps / effective_fps)))  # source frames between obs-context entries
    print(f"[demo] route window = {window} frames @ {effective_fps:.2f} fps "
          f"({window / effective_fps:.2f} s look-ahead); obs context spacing = "
          f"{src_dt} src frame(s) ({src_dt / src_fps:.3f} s)", flush=True)

    def build_route_window(start_idx: int) -> torch.Tensor:
        """Take target-fps frames[start..start+window-1]; pad by repeating the last frame."""
        idx = [min(start_idx + k, n_total - 1) for k in range(window)]
        return torch.stack([goal_chw[i] for i in idx], dim=0)

    def build_obs_context_src(s: int) -> torch.Tensor:
        """Obs at src frame s: walk back at training temporal spacing (= src_dt source frames)."""
        idx = [max(0, s - off * src_dt) for off in range(context_size, -1, -1)]
        idx = [min(i, n_src - 1) for i in idx]
        return torch.stack([obs_chw_src[i] for i in idx], dim=0)

    # z_g segments live on the target-fps grid (slow system replans at this cadence).
    seg_size = max(1, int(args.frame_stride))
    seg_starts = list(range(0, n_total, seg_size))
    print(f"[demo] {len(seg_starts)} z_g segments "
          f"(stride={seg_size} target-fps frames = {seg_size / effective_fps:.2f} s replanning)",
          flush=True)

    bs = max(1, int(args.inference_batch))

    # --- Stage-1: encode one z_g per segment, batched ---
    z_chunks: list[torch.Tensor] = []
    seg_debug: dict[int, dict[str, np.ndarray]] = {}
    for chunk_start in tqdm(range(0, len(seg_starts), bs), desc="stage1", dynamic_ncols=True):
        chunk = seg_starts[chunk_start:chunk_start + bs]
        goal_batch = torch.stack([build_route_window(s) for s in chunk], dim=0).to(device)
        with make_autocast(device, use_amp):
            stage1_out = stage1(goal_batch, sample=False)
        z = stage1_out["z_g"].detach().float()
        z_chunks.append(z)
        if args.display_debug:
            anchors = stage1_out["anchor_pred"].detach().float().cpu().numpy()
            goals = stage1_out["goal_pred"].detach().float().cpu().numpy()
            turn_probs = F.softmax(stage1_out["turn_logits"].detach().float(), dim=-1).cpu().numpy()
            logvars = stage1_out["logvar"].detach().float().cpu().numpy()
            z_np = z.cpu().numpy()
            for k_i in range(len(chunk)):
                seg_debug[chunk_start + k_i] = {
                    "z_g": z_np[k_i],
                    "logvar": logvars[k_i],
                    "anchor": anchors[k_i],
                    "goal": goals[k_i],
                    "turn_probs": turn_probs[k_i],
                }
    z_segments = torch.cat(z_chunks, dim=0)  # [num_segments, z_dim]

    # Map each src frame to its processed-timeline index, then to a segment id.
    proc_src_arr = np.asarray(proc_src_idx, dtype=np.int64)
    proc_t_for_src = np.clip(
        np.searchsorted(proc_src_arr, np.arange(n_src, dtype=np.int64), side="right") - 1,
        0, n_total - 1,
    )
    seg_id_for_src = (proc_t_for_src // seg_size).astype(np.int64)

    # --- Stage-2: every SOURCE frame (n_src inferences) ---
    pred_per_src: dict[int, np.ndarray] = {}
    process_indices = list(range(n_src))
    print(f"[demo] running Stage-2 over {n_src} source frames", flush=True)
    for chunk_start in tqdm(range(0, len(process_indices), bs), desc="stage2", dynamic_ncols=True):
        chunk = process_indices[chunk_start:chunk_start + bs]
        obs_batch = torch.stack([build_obs_context_src(s) for s in chunk], dim=0).to(device)
        seg_ids = torch.as_tensor([int(seg_id_for_src[s]) for s in chunk], device=device, dtype=torch.long)
        z_per_t = z_segments.index_select(0, seg_ids)
        pred = sample_traj(
            stage2, obs_batch, z_per_t,
            horizon=horizon, action_dim=action_dim, steps=args.cfm_steps,
            action_stats=action_stats, input_goal_mask=None,
            num_samples=args.num_trajectory_samples, device=device, amp=use_amp,
        )
        pred_m = (pred * pred_to_meter).detach().cpu().numpy()
        for k, s in enumerate(chunk):
            pred_per_src[s] = pred_m[k]

    all_xy = np.concatenate([v.reshape(-1, 2) for v in pred_per_src.values()], axis=0)
    extent = float(np.max(np.abs(all_xy))) * 1.2 if all_xy.size else 1.0
    extent = max(extent, 1.0)

    # Pre-compute global ranges for debug heatmaps so they don't jitter across frames.
    if args.display_debug and seg_debug:
        all_z = np.stack([d["z_g"] for d in seg_debug.values()], axis=0)
        all_sigma = np.exp(0.5 * np.stack([d["logvar"] for d in seg_debug.values()], axis=0))
        z_range = (float(all_z.min()), float(all_z.max()))
        sigma_range = (float(all_sigma.min()), float(all_sigma.max()))
        all_anchor_xy = np.concatenate([d["anchor"].reshape(-1, 2) for d in seg_debug.values()] +
                                       [d["goal"].reshape(-1, 2) for d in seg_debug.values()], axis=0)
        if all_anchor_xy.size:
            extent = max(extent, float(np.max(np.abs(all_anchor_xy))) * 1.2)
    else:
        z_range = sigma_range = (0.0, 1.0)

    # --- Render at SOURCE fps, on the original frames; one fresh BEV per src frame ---
    import imageio.v2 as imageio

    out_fps = float(args.output_fps) if args.output_fps is not None else src_fps
    output_path.parent.mkdir(parents=True, exist_ok=True)

    side_px = int(frames_u8_src.shape[1])
    empty_gt = np.zeros((0, 2), dtype=np.float32)

    print(f"[demo] writing {n_src} frames @ {out_fps:.2f} fps"
          f"{' with debug panel' if args.display_debug else ''}", flush=True)
    with imageio.get_writer(str(output_path), fps=out_fps, macro_block_size=1) as writer:
        for s in tqdm(range(n_src), desc="render", dynamic_ncols=True):
            seg_id = int(seg_id_for_src[s])
            title = f"src {s}/{n_src - 1}  proc {int(proc_t_for_src[s])}  seg {seg_id}"
            if args.display_debug and seg_id in seg_debug:
                dbg = seg_debug[seg_id]
                bev = render_bev_with_debug(
                    pred_per_src[s], side_px=side_px, extent=extent,
                    anchors_xy=dbg["anchor"], goal_xy=dbg["goal"], title=title,
                )
                panel = render_debug_panel(
                    dbg, side_px=side_px, z_value_range=z_range,
                    sigma_value_range=sigma_range, seg_id=seg_id,
                )
            else:
                bev = render_bev(
                    pred_per_src[s], empty_gt, side_px=side_px, extent=extent, title=title,
                )
                panel = None
            left = frames_u8_src[s]
            if left.shape[0] != bev.shape[0]:
                new_w = int(left.shape[1] * bev.shape[0] / left.shape[0])
                left = np.asarray(Image.fromarray(left).resize((new_w, bev.shape[0]), Image.BILINEAR))
            row = [left, bev] if panel is None else [left, bev, panel]
            frame = np.concatenate(row, axis=1)
            pad_h = frame.shape[0] % 2
            pad_w = frame.shape[1] % 2
            if pad_h or pad_w:
                frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            writer.append_data(frame)

    print(f"[demo] done -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
