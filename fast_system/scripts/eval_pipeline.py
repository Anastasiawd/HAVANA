#!/usr/bin/env python3
"""End-to-end evaluation of the slow (Stage-1) + fast (Stage-2) navigation pipeline.

Pipeline:
    route video --[frozen V-JEPA2 + Route Adapter]--> z_g
    (z_g, obs_history) --[CFM trajectory decoder]--> future trajectory  vs  GT trajectory

This script answers three questions:
    1. Does the full system predict reasonable future trajectories from
       (route_video, current_obs)?  -> ADE / FDE against GT.
    2. Does z_g actually help Stage-2?  -> compare predictions conditioned on
       the real z_g vs. with z_g masked out (input_goal_mask=1) and vs. a
       random z_g drawn from N(0, I).
    3. With one fixed route z_g, can Stage-2 keep producing sensible trajectories
       at multiple async offsets along the route?  -> per-offset ADE/FDE and
       cross-offset variance.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from tqdm.auto import tqdm

FAST_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(FAST_SYSTEM_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(FAST_SYSTEM_ROOT.parent))

from fast_system.dataset.mix_dataset import build_scand_stage2_dataloader_from_config
from fast_system.model.adapter import build_stage1_route_model
from fast_system.model.nav_model import (
    DEFAULT_ACTION_STATS,
    DEFAULT_FLOWNAV_CHECKPOINT,
    build_fast_nav_model,
    load_flownav_checkpoint,
)
from fast_system.trainer.train_stage_2 import (
    apply_overrides as _apply_overrides,
    load_config,
    move_to_device,
    resolve_device,
    seed_everything,
    torch_load,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Stage-1 + Stage-2 navigation pipeline.")
    parser.add_argument("--config", type=Path, default=FAST_SYSTEM_ROOT / "configs" / "stage2.yaml")
    parser.add_argument("--stage1-ckpt", type=Path, default=FAST_SYSTEM_ROOT / "model_weights" / "stage1.pt")
    parser.add_argument("--stage2-ckpt", type=Path, default=FAST_SYSTEM_ROOT / "model_weights" / "stage2.pt")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--index", type=Path, default=None, help="Override JSONL index path.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cfm-steps", type=int, default=10)
    parser.add_argument("--num-trajectory-samples", type=int, default=8,
                        help="Sample multiple trajectories per condition; ADE/FDE are reported on the mean prediction.")
    parser.add_argument("--ablation", nargs="+", default=["zg", "masked", "random"],
                        choices=["zg", "masked", "random"],
                        help="Conditioning variants to evaluate.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=FAST_SYSTEM_ROOT / "outputs" / "eval_pipeline")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save per-sample predictions / GT to .npz for later plotting.")
    parser.add_argument("--num-sequences", type=int, default=None,
                        help="Limit to the first N route sequences (slow-system clips). "
                             "Combined with --max-samples; whichever is smaller wins.")
    parser.add_argument("--save-video", action="store_true",
                        help="Render an MP4 per (sequence, offset): left = route video, "
                             "right = BEV with predicted (red) and GT (green) trajectories.")
    parser.add_argument("--video-fps", type=int, default=4)
    parser.add_argument("--video-limit", type=int, default=None,
                        help="Cap on number of MP4s written (default: all evaluated).")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle the dataset before slicing with --num-sequences. "
                             "Use this to mix recon/scand/gs (the index file is sorted by dataset).")
    return parser.parse_args()


def load_stage1(ckpt_path: Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch_load(ckpt_path, torch.device("cpu"))
    if not isinstance(ckpt, Mapping) or "cfg" not in ckpt or "model" not in ckpt:
        raise TypeError(f"Stage-1 checkpoint missing cfg/model: {ckpt_path}")
    model = build_stage1_route_model(ckpt["cfg"])
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[stage1] loaded {ckpt_path} epoch={ckpt.get('epoch','?')} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model


def load_stage2(ckpt_path: Path, cfg: Mapping[str, Any], device: torch.device) -> torch.nn.Module:
    ckpt = torch_load(ckpt_path, torch.device("cpu"))
    model_cfg = cfg.get("model", {})
    model = build_fast_nav_model(model_cfg).to(device)
    flownav_ckpt = cfg.get("flownav", {}).get("checkpoint", DEFAULT_FLOWNAV_CHECKPOINT)
    load_flownav_checkpoint(model, flownav_ckpt)
    state_dict = ckpt["model"] if isinstance(ckpt, Mapping) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"[stage2] loaded {ckpt_path} epoch={ckpt.get('epoch','?') if isinstance(ckpt, Mapping) else '?'} "
          f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    return model


def make_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def action_stats_from_cfg(cfg: Mapping[str, Any]) -> dict[str, Sequence[float]]:
    stats = cfg.get("model", {}).get("action_stats", DEFAULT_ACTION_STATS)
    return {"min": tuple(stats["min"]), "max": tuple(stats["max"])}


def flatten_async(batch: dict[str, Any], z_goal: torch.Tensor) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Flatten [B, S, ...] async batches to [B*S, ...]; replicate z_g across offsets.

    Returns (flat_batch, flat_z_goal, slow_index) where slow_index[i] is the
    original slow-system clip id for flattened item i (used for grouping).
    """
    obs = batch["obs"]
    if not torch.is_tensor(obs) or obs.ndim != 6:
        slow_index = torch.arange(int(obs.shape[0]), device=obs.device)
        return batch, z_goal, slow_index

    slow_b, fast_s = int(obs.shape[0]), int(obs.shape[1])
    flat = dict(batch)
    for key in ("obs", "action", "distance", "goal_pos", "action_mask", "async_offsets"):
        v = batch.get(key)
        if torch.is_tensor(v) and v.shape[:2] == (slow_b, fast_s):
            flat[key] = v.reshape(slow_b * fast_s, *v.shape[2:])
    z_goal_flat = z_goal.unsqueeze(1).expand(-1, fast_s, -1).reshape(slow_b * fast_s, -1)
    slow_index = torch.arange(slow_b, device=obs.device).unsqueeze(1).expand(-1, fast_s).reshape(-1)
    return flat, z_goal_flat, slow_index


def trajectory_metrics(pred: torch.Tensor, gt: torch.Tensor, scale: float) -> dict[str, np.ndarray]:
    """Per-sample ADE / FDE in meters.  pred, gt: [N, H, 2] in normalized units."""
    pred_m = pred * scale
    gt_m = gt * scale
    err = torch.linalg.norm(pred_m - gt_m, dim=-1)  # [N, H]
    ade = err.mean(dim=-1).cpu().numpy()
    fde = err[:, -1].cpu().numpy()
    return {"ade": ade, "fde": fde}


@torch.no_grad()
def sample_traj(
    model: torch.nn.Module,
    obs: torch.Tensor,
    z_goal: torch.Tensor,
    *,
    horizon: int,
    action_dim: int,
    steps: int,
    action_stats: Mapping[str, Sequence[float]],
    input_goal_mask: torch.Tensor | None,
    num_samples: int,
    device: torch.device,
    amp: bool,
) -> torch.Tensor:
    """Sample `num_samples` trajectories, return mean prediction [N, H, 2]."""
    accum = None
    for _ in range(max(1, num_samples)):
        with make_autocast(device, amp):
            traj = model.sample_actions(
                obs_img=obs,
                z_goal=z_goal,
                pred_horizon=horizon,
                action_dim=action_dim,
                steps=steps,
                action_stats=action_stats,
                input_goal_mask=input_goal_mask,
            )
        accum = traj.float() if accum is None else accum + traj.float()
    return accum / max(1, num_samples)


def aggregate(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "n": 0}
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std()),
        "n": int(values.size),
    }


def render_bev(
    pred_xy_m: np.ndarray,
    gt_xy_m: np.ndarray,
    side_px: int,
    extent: float | None = None,
    title: str | None = None,
) -> np.ndarray:
    """Render a BEV plot with predicted (red) vs GT (green) trajectory, both rooted at origin.

    Convention (matches the dataset's `rotate_to_local`): x is forward, y is left.
    The plot uses y on the horizontal axis and x on the vertical axis so the robot
    looks "up". Pass `extent` to keep the axes fixed across frames.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    fig = plt.figure(figsize=(4, 4), dpi=max(64, side_px // 4))
    ax = fig.add_subplot(111)
    if gt_xy_m is not None and gt_xy_m.size > 0:
        ax.plot([0.0, *gt_xy_m[:, 1].tolist()], [0.0, *gt_xy_m[:, 0].tolist()],
                "-o", color="#2ca02c", markersize=3, linewidth=2, label="GT")
    ax.plot([0.0, *pred_xy_m[:, 1].tolist()], [0.0, *pred_xy_m[:, 0].tolist()],
            "-o", color="#d62728", markersize=3, linewidth=2, label="Pred")
    ax.scatter([0.0], [0.0], c="k", marker="s", s=50, zorder=5, label="origin")

    if extent is None:
        extent = float(np.max(np.abs(np.concatenate([gt_xy_m, pred_xy_m], axis=0)))) if gt_xy_m.size else 1.0
        extent = max(extent * 1.2, 1.0)
    ax.set_xlim(extent, -extent)  # flip so +y (left in body frame) appears on the left
    ax.set_ylim(-0.5, extent * 2)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("y (left, m)")
    ax.set_ylabel("x (forward, m)")
    ax.legend(loc="upper left", fontsize=8)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout(pad=0.4)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
    plt.close(fig)

    from PIL import Image
    return np.asarray(Image.fromarray(img).resize((side_px, side_px), Image.BILINEAR))


@torch.no_grad()
def render_dynamic_video(
    *,
    path: Path,
    dataset,
    dataset_idx: int,
    stage2: torch.nn.Module,
    z_goal: torch.Tensor,
    goal_frames: torch.Tensor,
    action_stats: Mapping[str, Sequence[float]],
    horizon: int,
    action_dim: int,
    cfm_steps: int,
    num_samples: int,
    pred_to_meter: float,
    device: torch.device,
    amp: bool,
    fps: int,
) -> None:
    """Per-frame inference: for each frame of the route clip rebuild obs + GT,
    run Stage-2 with the same z_g, and render left=route frame / right=BEV."""

    import imageio.v2 as imageio

    record = dataset.records[int(dataset_idx)]
    n_frames = int(goal_frames.shape[0])
    obs_list: list[torch.Tensor] = []
    gt_list: list[torch.Tensor] = []
    valid_offsets: list[int] = []
    for offset in range(n_frames):
        try:
            obs = dataset._read_observation(record, offset=offset)
            action, _, _, _ = dataset._sample_actions(record, offset=offset)
        except Exception:
            continue
        obs_list.append(obs)
        gt_list.append(action)
        valid_offsets.append(offset)

    if not obs_list:
        print(f"[video] no valid offsets for sequence {dataset_idx}", flush=True)
        return

    obs_batch = torch.stack(obs_list, dim=0).to(device)
    gt_batch = torch.stack(gt_list, dim=0).to(device).float()
    z_rep = z_goal.detach().unsqueeze(0).expand(obs_batch.shape[0], -1).contiguous().to(device)

    pred = sample_traj(
        stage2, obs_batch, z_rep,
        horizon=horizon, action_dim=action_dim, steps=cfm_steps,
        action_stats=action_stats, input_goal_mask=None,
        num_samples=num_samples, device=device, amp=amp,
    )
    pred_m = (pred * pred_to_meter).detach().cpu().numpy()
    gt_m = (gt_batch * pred_to_meter).detach().cpu().numpy()

    all_xy = np.concatenate([pred_m.reshape(-1, 2), gt_m.reshape(-1, 2)], axis=0)
    extent = float(np.max(np.abs(all_xy))) * 1.2 if all_xy.size else 1.0
    extent = max(extent, 1.0)

    side = int(goal_frames.shape[-1])
    frames_np = goal_frames.detach().float().cpu().numpy()
    offset_to_i = {off: i for i, off in enumerate(valid_offsets)}
    last_i = 0

    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=int(fps), macro_block_size=1) as writer:
        for t in range(n_frames):
            i = offset_to_i.get(t, last_i)
            last_i = i
            ade_t = float(np.mean(np.linalg.norm(pred_m[i] - gt_m[i], axis=-1)))
            bev = render_bev(pred_m[i], gt_m[i], side_px=side, extent=extent,
                             title=f"frame {t}/{n_frames-1}  ADE={ade_t:.2f}m")
            rgb = (np.transpose(frames_np[t], (1, 2, 0)) * 255.0).clip(0, 255).astype(np.uint8)
            writer.append_data(np.concatenate([rgb, bev], axis=1))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    overrides = argparse.Namespace(
        train_index=args.index if args.split == "train" else None,
        val_index=args.index if args.split == "val" else None,
        output_dir=None, epochs=None, batch_size=args.batch_size, grad_accum_steps=None,
        lr=None, device=args.device, resume=None, max_samples=args.max_samples,
        disable_lingbot_depth=False,
    )
    cfg = _apply_overrides(cfg, overrides)
    cfg.setdefault("dataloader", {})["num_workers"] = int(args.num_workers)
    cfg["dataloader"]["shuffle"] = False
    cfg["dataloader"]["persistent_workers"] = False

    if args.num_sequences is not None and not args.shuffle:
        existing = cfg["scand_data"].get("max_samples")
        cfg["scand_data"]["max_samples"] = (
            int(args.num_sequences) if existing is None else min(int(existing), int(args.num_sequences))
        )
    elif args.shuffle:
        cfg["scand_data"]["max_samples"] = None

    seed_everything(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "cuda")))
    use_amp = (not args.no_amp) and device.type == "cuda"

    loader = build_scand_stage2_dataloader_from_config(cfg["scand_data"], cfg["dataloader"], split=args.split)
    if args.shuffle:
        import random as _random
        rng = _random.Random(int(cfg.get("seed", 0)))
        rng.shuffle(loader.dataset.records)
        if args.num_sequences is not None:
            loader.dataset.records = loader.dataset.records[: int(args.num_sequences)]
    from collections import Counter as _Counter
    ds_counts = _Counter(r.get("dataset", "?") for r in loader.dataset.records)
    print(f"[data] split={args.split} samples={len(loader.dataset)} batches={len(loader)} "
          f"dataset_mix={dict(ds_counts)}", flush=True)

    stage1 = load_stage1(args.stage1_ckpt, device)
    stage2 = load_stage2(args.stage2_ckpt, cfg, device)

    action_stats = action_stats_from_cfg(cfg)
    horizon = int(cfg.get("scand_data", {}).get("action_horizon", 16))
    action_dim = int(cfg.get("model", {}).get("action_dim", 2))
    waypoint_spacing = float(cfg.get("scand_data", {}).get("metric_waypoint_spacing", 0.38))
    normalize_actions = bool(cfg.get("scand_data", {}).get("normalize_actions", True))
    pred_to_meter = waypoint_spacing if normalize_actions else 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    saved = {variant: defaultdict(list) for variant in args.ablation}
    per_offset = {variant: defaultdict(lambda: {"ade": [], "fde": []}) for variant in args.ablation}
    per_sample_rows: list[dict[str, Any]] = []
    pred_dump: list[dict[str, Any]] = [] if args.save_trajectories else []
    video_dir = args.output_dir / "videos"
    videos_written = 0
    video_limit = args.video_limit if args.video_limit is not None else float("inf")
    seq_seen = 0
    seq_limit = args.num_sequences if args.num_sequences is not None else float("inf")

    for batch in tqdm(loader, desc=f"eval[{args.split}]", dynamic_ncols=True):
        batch = move_to_device(batch, device)
        with make_autocast(device, use_amp):
            z_goal = stage1(batch["goal_video"], sample=False)["z_g"].detach().float()
        flat_batch, z_goal_flat, slow_index = flatten_async(batch, z_goal)

        obs = flat_batch["obs"]
        gt = flat_batch["action"].float()
        offsets = flat_batch.get("async_offsets")
        offsets_np = (offsets.detach().cpu().numpy().astype(int)
                      if torch.is_tensor(offsets) else np.zeros(obs.shape[0], dtype=int))
        slow_index_np = slow_index.detach().cpu().numpy().astype(int)
        n_flat = obs.shape[0]
        batch_slow_b = int(batch["goal_video"].shape[0]) if torch.is_tensor(batch.get("goal_video")) else n_flat
        slow_offset = seq_seen
        seq_seen += batch_slow_b

        for variant in args.ablation:
            if variant == "zg":
                z = z_goal_flat
                mask = None
            elif variant == "masked":
                z = z_goal_flat
                mask = torch.ones(n_flat, device=device, dtype=torch.long)
            elif variant == "random":
                z = torch.randn_like(z_goal_flat)
                mask = None
            else:
                raise ValueError(variant)

            pred = sample_traj(
                stage2, obs, z,
                horizon=horizon, action_dim=action_dim, steps=args.cfm_steps,
                action_stats=action_stats, input_goal_mask=mask,
                num_samples=args.num_trajectory_samples, device=device, amp=use_amp,
            )
            metrics = trajectory_metrics(pred, gt, scale=pred_to_meter)
            saved[variant]["ade"].append(metrics["ade"])
            saved[variant]["fde"].append(metrics["fde"])
            for i in range(n_flat):
                key = int(offsets_np[i])
                per_offset[variant][key]["ade"].append(float(metrics["ade"][i]))
                per_offset[variant][key]["fde"].append(float(metrics["fde"][i]))
                per_sample_rows.append({
                    "variant": variant,
                    "slow_index": int(slow_offset + slow_index_np[i]),
                    "offset": key,
                    "ade_m": float(metrics["ade"][i]),
                    "fde_m": float(metrics["fde"][i]),
                })

            if variant == "zg":
                pred_zg_m = (pred * pred_to_meter).detach().cpu().numpy()
                gt_m = (gt * pred_to_meter).detach().cpu().numpy()

                if args.save_trajectories:
                    pred_dump.append({
                        "slow_index": (slow_offset + slow_index_np).copy(),
                        "offsets": offsets_np.copy(),
                        "pred": pred_zg_m,
                        "gt":   gt_m,
                    })

                if args.save_video and videos_written < video_limit:
                    seen_clip_locals: set[int] = set()
                    for i in range(n_flat):
                        if videos_written >= video_limit:
                            break
                        local_slow_idx = int(slow_index_np[i])
                        if local_slow_idx in seen_clip_locals:
                            continue
                        seen_clip_locals.add(local_slow_idx)
                        global_slow_idx = int(slow_offset + local_slow_idx)
                        video_path = video_dir / f"seq{global_slow_idx:05d}.mp4"
                        try:
                            render_dynamic_video(
                                path=video_path,
                                dataset=loader.dataset,
                                dataset_idx=global_slow_idx,
                                stage2=stage2,
                                z_goal=z_goal[local_slow_idx],
                                goal_frames=batch["goal_video"][local_slow_idx],
                                action_stats=action_stats,
                                horizon=horizon,
                                action_dim=action_dim,
                                cfm_steps=args.cfm_steps,
                                num_samples=args.num_trajectory_samples,
                                pred_to_meter=pred_to_meter,
                                device=device,
                                amp=use_amp,
                                fps=args.video_fps,
                            )
                            videos_written += 1
                        except Exception as exc:
                            print(f"[video] failed {video_path}: {exc}", flush=True)

        if seq_seen >= seq_limit:
            break

    summary: dict[str, Any] = {
        "config": str(args.config),
        "stage1_ckpt": str(args.stage1_ckpt),
        "stage2_ckpt": str(args.stage2_ckpt),
        "split": args.split,
        "num_samples": int(len(loader.dataset)),
        "horizon": horizon,
        "cfm_steps": args.cfm_steps,
        "num_trajectory_samples": args.num_trajectory_samples,
        "waypoint_spacing_m": waypoint_spacing,
        "results": {},
        "per_offset": {},
    }
    print("\n=== Aggregate results (meters) ===")
    for variant in args.ablation:
        ade = np.concatenate(saved[variant]["ade"]) if saved[variant]["ade"] else np.zeros(0)
        fde = np.concatenate(saved[variant]["fde"]) if saved[variant]["fde"] else np.zeros(0)
        summary["results"][variant] = {"ade": aggregate(ade), "fde": aggregate(fde)}
        print(f"  [{variant:>6}] ADE={ade.mean():.3f}±{ade.std():.3f}  "
              f"FDE={fde.mean():.3f}±{fde.std():.3f}  n={ade.size}")

    print("\n=== Per-offset (variant=zg, meters) ===")
    if "zg" in args.ablation:
        for offset in sorted(per_offset["zg"].keys()):
            ade = np.asarray(per_offset["zg"][offset]["ade"])
            fde = np.asarray(per_offset["zg"][offset]["fde"])
            print(f"  offset={offset:>3}  ADE={ade.mean():.3f}  FDE={fde.mean():.3f}  n={ade.size}")
    for variant in args.ablation:
        summary["per_offset"][variant] = {
            int(off): {"ade_mean": float(np.mean(d["ade"])), "fde_mean": float(np.mean(d["fde"])), "n": len(d["ade"])}
            for off, d in per_offset[variant].items()
        }

    # Cross-offset consistency (Q3): for each route clip, std of ADE across offsets.
    if "zg" in args.ablation:
        per_route_ade: dict[int, list[float]] = defaultdict(list)
        for row in per_sample_rows:
            if row["variant"] == "zg":
                per_route_ade[row["slow_index"]].append(row["ade_m"])
        stds = np.asarray([np.std(v) for v in per_route_ade.values() if len(v) > 1])
        if stds.size > 0:
            print(f"\n=== Async consistency (variant=zg) ===")
            print(f"  mean std(ADE) across offsets per route = {stds.mean():.3f} m  "
                  f"(median={np.median(stds):.3f}, n_routes={stds.size})")
            summary["async_consistency"] = {
                "mean_std_ade_m": float(stds.mean()),
                "median_std_ade_m": float(np.median(stds)),
                "num_routes": int(stds.size),
            }

    # Persist artifacts.
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (args.output_dir / "per_sample.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "slow_index", "offset", "ade_m", "fde_m"])
        writer.writeheader()
        writer.writerows(per_sample_rows)
    if args.save_trajectories and pred_dump:
        np.savez(
            args.output_dir / "trajectories.npz",
            slow_index=np.concatenate([d["slow_index"] for d in pred_dump]),
            offsets=np.concatenate([d["offsets"] for d in pred_dump]),
            pred=np.concatenate([d["pred"] for d in pred_dump], axis=0),
            gt=np.concatenate([d["gt"] for d in pred_dump], axis=0),
        )
    print(f"\nWrote: {args.output_dir/'summary.json'}")
    print(f"Wrote: {args.output_dir/'per_sample.csv'}")
    if args.save_trajectories and pred_dump:
        print(f"Wrote: {args.output_dir/'trajectories.npz'}")
    if args.save_video:
        print(f"Wrote: {videos_written} videos under {video_dir}")


if __name__ == "__main__":
    main()
