#!/usr/bin/env python3
"""Cross-source variant of demo.py: Stage-1's route video and Stage-2's obs stream come from
DIFFERENT sources (e.g. you film a route with a hand-held camera while a robot uses its own
front-facing camera as the obs feed). Everything else mirrors demo.py.

Pipeline:
    route source   --[resample to route-target-fps]--> z_g segments  (slow system)
    obs source     --[per-source-frame]-------------> Stage-2 obs    (fast system)
    Stage-2 cond = (obs context, z_g segment looked up by proportional time-mapping)
    Output mp4: left = current obs frame at obs source fps,
                right = BEV with predicted trajectory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

FAST_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(FAST_SYSTEM_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(FAST_SYSTEM_ROOT.parent))

from fast_system.scripts.demo import read_video, read_image_folder, to_tensor_chw
from fast_system.scripts.eval_pipeline import (
    action_stats_from_cfg,
    load_stage1,
    load_stage2,
    make_autocast,
    render_bev,
    sample_traj,
)
from fast_system.model.nav_model import normalized_deltas_to_actions
from fast_system.trainer.train_stage_2 import load_config, resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the slow+fast pipeline with route (z_g) and obs streams from different sources."
    )
    route = p.add_mutually_exclusive_group(required=True)
    route.add_argument("--route-video", type=Path, default=None,
                       help="Video file used as the slow-system route (encoded to z_g).")
    route.add_argument("--route-folder", type=Path, default=None,
                       help="Folder of frames used as the slow-system route.")
    p.add_argument("--route-source-fps", type=float, default=None,
                   help="Source fps for --route-folder (ignored for video).")

    obs = p.add_mutually_exclusive_group(required=True)
    obs.add_argument("--obs-video", type=Path, default=None,
                     help="Video file used as the fast-system obs stream.")
    obs.add_argument("--obs-folder", type=Path, default=None,
                     help="Folder of frames used as the fast-system obs stream.")
    p.add_argument("--obs-source-fps", type=float, default=None,
                   help="Source fps for --obs-folder (ignored for video).")

    p.add_argument("--output", type=Path, default=None,
                   help="Output MP4 path (default: <obs>.xdemo.mp4).")
    p.add_argument("--config", type=Path, default=FAST_SYSTEM_ROOT / "configs" / "stage2.yaml")
    p.add_argument("--stage1-ckpt", type=Path, default=FAST_SYSTEM_ROOT / "model_weights" / "stage1.pt")
    p.add_argument("--stage2-ckpt", type=Path, default=FAST_SYSTEM_ROOT / "model_weights" / "stage2.pt")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--cfm-steps", type=int, default=16)
    p.add_argument("--num-trajectory-samples", type=int, default=4)
    p.add_argument("--route-window", type=int, default=16,
                   help="Number of consecutive route target-fps frames per z_g segment "
                        "(default 16, matches training).")
    p.add_argument("--target-fps", type=float, default=4.0,
                   help="Target fps used for BOTH route resampling (Stage-1 input) and "
                        "obs context spacing (Stage-2 input). Default 4 matches training.")
    p.add_argument("--frame-stride", type=int, default=8,
                   help="z_g segment length in route target-fps frames (= slow-system replanning period).")
    p.add_argument("--max-route-frames", type=int, default=None,
                   help="Truncate route input to first N source frames.")
    p.add_argument("--max-obs-frames", type=int, default=None,
                   help="Truncate obs input to first N source frames.")
    p.add_argument("--inference-batch", type=int, default=16)
    p.add_argument("--output-fps", type=float, default=None,
                   help="Output mp4 fps (default: obs source fps).")
    p.add_argument("--mask-zg", action="store_true",
                   help="Mask out z_g during Stage-2 inference so the model predicts "
                        "trajectories from observations only (input_goal_mask=1). "
                        "Equivalent to --cfg-scale 0; mutually exclusive with --cfg-scale.")
    p.add_argument("--cfg-scale", type=float, default=None,
                   help="Classifier-free guidance scale s for velocity-field mixing: "
                        "v = v_uncond + s*(v_cond - v_uncond). "
                        "Unset = pure conditional (default, single forward). "
                        "0 = obs-only (same as --mask-zg). 1 = conditional (but 2x forward cost). "
                        ">1 amplifies z_g guidance. Typical sweep: 0.5, 1.5, 2.0.")
    p.add_argument("--seed", type=int, default=5)
    return p.parse_args()


def read_source(
    video_path: Path | None,
    folder_path: Path | None,
    folder_fps: float | None,
    max_frames: int | None,
    label: str,
) -> tuple[np.ndarray, float, Path]:
    """Read either a video file or an image folder and return (frames_uint8, fps, input_path)."""
    if video_path is not None:
        frames, fps = read_video(video_path, max_frames)
        return frames, fps, video_path
    if folder_path is None:
        raise SystemExit(f"[{label}] need either --{label}-video or --{label}-folder.")
    if folder_fps is None:
        raise SystemExit(f"[{label}] --{label}-source-fps is required when using --{label}-folder.")
    frames = read_image_folder(folder_path, max_frames)
    return frames, float(folder_fps), folder_path


def resample_to_target(
    frames: np.ndarray, src_fps: float, target_fps: float
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (resampled_frames, proc_src_idx, effective_fps)."""
    n = int(frames.shape[0])
    if target_fps and target_fps > 0 and abs(src_fps - target_fps) > 1e-3:
        step = src_fps / float(target_fps)
        proc_src_idx = np.unique(np.floor(np.arange(0, n, step)).astype(int))
        proc_src_idx = proc_src_idx[proc_src_idx < n]
        return frames[proc_src_idx], proc_src_idx, float(target_fps)
    return frames, np.arange(n), float(src_fps)


@torch.no_grad()
def sample_traj_cfg(
    model,
    obs: torch.Tensor,
    z_goal: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
    steps: int,
    action_stats,
    cfg_scale: float,
    num_samples: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    """Euler-CFM sampler with classifier-free guidance on the velocity field.

    v = v_uncond + s * (v_cond - v_uncond), where v_cond uses z_g and
    v_uncond masks z_g out (input_goal_mask=1). Both branches share the same
    UNet weights — the model was trained with random goal-mask dropout.
    """
    B = obs.shape[0]
    zero_mask = torch.zeros(B, device=device, dtype=torch.long)
    one_mask = torch.ones(B, device=device, dtype=torch.long)
    with make_autocast(device, amp):
        cond_with = model.encode_condition(obs, z_goal, input_goal_mask=zero_mask)
        cond_without = model.encode_condition(obs, z_goal, input_goal_mask=one_mask)
    cond_both = torch.cat([cond_with, cond_without], dim=0)

    n_steps = max(int(steps), 1)
    dt = 1.0 / n_steps
    s = float(cfg_scale)
    accum = None
    for _ in range(max(1, int(num_samples))):
        x = torch.randn((B, horizon, action_dim), device=device, dtype=cond_with.dtype)
        for i in range(n_steps):
            t = torch.full((B,), i * dt, device=device, dtype=x.dtype)
            x_both = torch.cat([x, x], dim=0)
            t_both = torch.cat([t, t], dim=0)
            with make_autocast(device, amp):
                v_both = model.noise_pred_net(
                    sample=x_both, timestep=t_both, global_cond=cond_both,
                )
            v_cond, v_uncond = v_both[:B], v_both[B:]
            v = v_uncond + s * (v_cond - v_uncond)
            x = x + v * dt
        traj = normalized_deltas_to_actions(x, action_stats).float()
        accum = traj if accum is None else accum + traj
    return accum / max(1, int(num_samples))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.cfg_scale is not None and args.mask_zg:
        raise SystemExit("--cfg-scale and --mask-zg are mutually exclusive (use --cfg-scale 0 instead).")
    cfg = load_config(args.config)
    seed_everything(int(args.seed))
    device = resolve_device(args.device or str(cfg.get("device", "cuda")))
    use_amp = (not args.no_amp) and device.type == "cuda"

    route_frames_src, route_fps, route_path = read_source(
        args.route_video, args.route_folder, args.route_source_fps, args.max_route_frames, "route",
    )
    obs_frames_src, obs_fps, obs_path = read_source(
        args.obs_video, args.obs_folder, args.obs_source_fps, args.max_obs_frames, "obs",
    )
    print(f"[demo] route: {route_path} -> {route_frames_src.shape[0]} frames @ {route_fps:.2f} fps", flush=True)
    print(f"[demo] obs  : {obs_path} -> {obs_frames_src.shape[0]} frames @ {obs_fps:.2f} fps",   flush=True)

    if args.output is not None:
        output_path = args.output
    else:
        anchor = obs_path.resolve()
        stem = anchor.stem if args.obs_video is not None else anchor.name
        output_path = anchor.parent / f"{stem}.xdemo.mp4"
    if output_path.exists() and output_path.is_dir():
        raise SystemExit(f"--output must be a file path, not a directory: {output_path}")
    if output_path.suffix.lower() != ".mp4":
        output_path = output_path.with_suffix(".mp4")
    print(f"[demo] -> output: {output_path}", flush=True)

    target_fps = float(args.target_fps)
    route_frames, _, route_effective_fps = resample_to_target(route_frames_src, route_fps, target_fps)
    n_route = int(route_frames.shape[0])
    n_obs_src = int(obs_frames_src.shape[0])

    # obs context spacing is computed against obs source fps so it matches training (~250 ms / 4 fps).
    obs_dt_src = max(1, int(round(obs_fps / target_fps))) if target_fps > 0 else 1

    print(f"[demo] route resampled -> {n_route} frames @ {route_effective_fps:.2f} fps", flush=True)
    print(f"[demo] obs context spacing = {obs_dt_src} src frames "
          f"({obs_dt_src / obs_fps:.3f} s, ~{target_fps:.1f} fps)", flush=True)

    data_cfg = cfg.get("scand_data", {})
    obs_size_v = int(data_cfg.get("obs_image_size", 256))
    goal_size_v = int(data_cfg.get("goal_image_size", 256))
    obs_size = (obs_size_v, obs_size_v)
    goal_size = (goal_size_v, goal_size_v)
    context_size = int(data_cfg.get("context_size", cfg.get("model", {}).get("context_size", 3)))
    horizon = int(data_cfg.get("action_horizon", 16))
    action_dim = int(cfg.get("model", {}).get("action_dim", 2))
    waypoint_spacing = float(data_cfg.get("metric_waypoint_spacing", 0.38))
    pred_to_meter = waypoint_spacing if bool(data_cfg.get("normalize_actions", True)) else 1.0

    print(f"[demo] resizing frames: obs={obs_size} goal={goal_size}", flush=True)
    goal_chw = [to_tensor_chw(f, goal_size) for f in route_frames]
    obs_chw_src = [to_tensor_chw(f, obs_size) for f in obs_frames_src]

    stage1 = load_stage1(args.stage1_ckpt, device)
    stage2 = load_stage2(args.stage2_ckpt, cfg, device)
    action_stats = action_stats_from_cfg(cfg)

    window = max(2, int(args.route_window))
    seg_size = max(1, int(args.frame_stride))
    seg_starts = list(range(0, n_route, seg_size))
    print(f"[demo] route window = {window} frames @ {route_effective_fps:.2f} fps "
          f"({window / route_effective_fps:.2f} s look-ahead); "
          f"{len(seg_starts)} z_g segments (stride={seg_size} = "
          f"{seg_size / route_effective_fps:.2f} s replanning)", flush=True)

    def build_route_window(start_idx: int) -> torch.Tensor:
        idx = [min(start_idx + k, n_route - 1) for k in range(window)]
        return torch.stack([goal_chw[i] for i in idx], dim=0)

    def build_obs_context_src(s: int) -> torch.Tensor:
        idx = [max(0, s - off * obs_dt_src) for off in range(context_size, -1, -1)]
        idx = [min(i, n_obs_src - 1) for i in idx]
        return torch.stack([obs_chw_src[i] for i in idx], dim=0)

    bs = max(1, int(args.inference_batch))

    # Stage-1: one z_g per route segment, batched.
    z_chunks: list[torch.Tensor] = []
    for chunk_start in tqdm(range(0, len(seg_starts), bs), desc="stage1", dynamic_ncols=True):
        chunk = seg_starts[chunk_start:chunk_start + bs]
        goal_batch = torch.stack([build_route_window(s) for s in chunk], dim=0).to(device)
        with make_autocast(device, use_amp):
            z = stage1(goal_batch, sample=False)["z_g"].detach().float()
        z_chunks.append(z)
    z_segments = torch.cat(z_chunks, dim=0)
    num_segments = int(z_segments.shape[0])

    # Map each obs source frame -> route segment by proportional time alignment.
    if n_obs_src == 1:
        proportional = np.zeros(1, dtype=np.float64)
    else:
        proportional = np.arange(n_obs_src, dtype=np.float64) / float(n_obs_src - 1)
    route_proc_for_obs = np.clip(
        np.round(proportional * max(n_route - 1, 0)).astype(np.int64), 0, max(n_route - 1, 0),
    )
    seg_id_for_obs = np.clip(route_proc_for_obs // seg_size, 0, num_segments - 1).astype(np.int64)

    # Stage-2: one inference per obs source frame.
    pred_per_src: dict[int, np.ndarray] = {}
    process_indices = list(range(n_obs_src))
    if args.cfg_scale is not None:
        print(f"[demo] CFG mixing: v = v_uncond + {args.cfg_scale:g}*(v_cond - v_uncond) "
              "(2 forwards/step)", flush=True)
    elif args.mask_zg:
        print("[demo] z_g MASKED: Stage-2 will predict from observations only", flush=True)
    print(f"[demo] running Stage-2 over {n_obs_src} obs source frames", flush=True)
    for chunk_start in tqdm(range(0, len(process_indices), bs), desc="stage2", dynamic_ncols=True):
        chunk = process_indices[chunk_start:chunk_start + bs]
        obs_batch = torch.stack([build_obs_context_src(s) for s in chunk], dim=0).to(device)
        seg_ids = torch.as_tensor([int(seg_id_for_obs[s]) for s in chunk], device=device, dtype=torch.long)
        z_per_t = z_segments.index_select(0, seg_ids)
        if args.cfg_scale is not None:
            pred = sample_traj_cfg(
                stage2, obs_batch, z_per_t,
                horizon=horizon, action_dim=action_dim, steps=args.cfm_steps,
                action_stats=action_stats, cfg_scale=args.cfg_scale,
                num_samples=args.num_trajectory_samples, device=device, amp=use_amp,
            )
        else:
            goal_mask = (torch.ones(len(chunk), device=device, dtype=torch.long)
                         if args.mask_zg else None)
            pred = sample_traj(
                stage2, obs_batch, z_per_t,
                horizon=horizon, action_dim=action_dim, steps=args.cfm_steps,
                action_stats=action_stats, input_goal_mask=goal_mask,
                num_samples=args.num_trajectory_samples, device=device, amp=use_amp,
            )
        pred_m = (pred * pred_to_meter).detach().cpu().numpy()
        for k, s in enumerate(chunk):
            pred_per_src[s] = pred_m[k]

    all_xy = np.concatenate([v.reshape(-1, 2) for v in pred_per_src.values()], axis=0)
    extent = float(np.max(np.abs(all_xy))) * 1.2 if all_xy.size else 1.0
    extent = max(extent, 1.0)

    import imageio.v2 as imageio

    out_fps = float(args.output_fps) if args.output_fps is not None else obs_fps
    output_path.parent.mkdir(parents=True, exist_ok=True)
    side_px = int(obs_frames_src.shape[1])
    empty_gt = np.zeros((0, 2), dtype=np.float32)

    print(f"[demo] writing {n_obs_src} frames @ {out_fps:.2f} fps", flush=True)
    with imageio.get_writer(str(output_path), fps=out_fps, macro_block_size=1) as writer:
        for s in tqdm(range(n_obs_src), desc="render", dynamic_ncols=True):
            seg_id = int(seg_id_for_obs[s])
            route_t = int(route_proc_for_obs[s])
            bev = render_bev(
                pred_per_src[s], empty_gt, side_px=side_px, extent=extent,
                title=f"obs {s}/{n_obs_src - 1}  route {route_t}/{max(n_route - 1, 0)}  seg {seg_id}",
            )
            left = obs_frames_src[s]
            if left.shape[0] != bev.shape[0]:
                new_w = int(left.shape[1] * bev.shape[0] / left.shape[0])
                left = np.asarray(Image.fromarray(left).resize((new_w, bev.shape[0]), Image.BILINEAR))
            writer.append_data(np.concatenate([left, bev], axis=1))

    print(f"[demo] done -> {output_path}", flush=True)


if __name__ == "__main__":
    main()
