#!/usr/bin/env python3
"""Train Stage-2 z_g-conditioned CFM navigation policy on SCAND."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm.auto import tqdm

try:
    from torch.utils.tensorboard import SummaryWriter
except (ImportError, ModuleNotFoundError):
    SummaryWriter = None


FAST_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
if str(FAST_SYSTEM_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(FAST_SYSTEM_ROOT.parent))

from fast_system.dataset.mix_dataset import build_scand_stage2_dataloader_from_config
from fast_system.model.adapter import build_stage1_route_model
from fast_system.model.nav_model import (
    DEFAULT_ACTION_STATS,
    DEFAULT_FLOWNAV_CHECKPOINT,
    action_reduce,
    actions_to_deltas,
    build_fast_nav_model,
    load_flownav_checkpoint,
    normalize_deltas,
    sample_cfm_location,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage-2 fast navigation policy.")
    parser.add_argument("--config", type=Path, default=FAST_SYSTEM_ROOT / "configs" / "stage2.yaml")
    parser.add_argument("--train-index", type=Path, default=None)
    parser.add_argument("--val-index", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--disable-lingbot-depth", action="store_true")
    return parser.parse_args()


def get_by_path(root: Mapping[str, Any], dotted: str) -> Any:
    value: Any = root
    for part in dotted.split("."):
        value = value[part]
    return value


def resolve_refs(obj: Any, root: Mapping[str, Any]) -> Any:
    pattern = re.compile(r"\$\{([^}]+)\}")
    if isinstance(obj, dict):
        return {key: resolve_refs(value, root) for key, value in obj.items()}
    if isinstance(obj, list):
        return [resolve_refs(value, root) for value in obj]
    if isinstance(obj, str):
        exact = re.fullmatch(pattern, obj)
        if exact:
            return get_by_path(root, exact.group(1))
        return pattern.sub(lambda m: str(get_by_path(root, m.group(1))), obj)
    return obj


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for _ in range(8):
        resolved = resolve_refs(cfg, cfg)
        if resolved == cfg:
            return resolved
        cfg = resolved
    return cfg


def apply_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg.setdefault("scand_data", {})
    cfg.setdefault("dataloader", {})
    cfg.setdefault("training", {})
    cfg.setdefault("model", {}).setdefault("vision_encoder", {}).setdefault("lingbot_depth", {})
    if args.train_index is not None:
        cfg["scand_data"]["train_index"] = str(args.train_index)
    if args.val_index is not None:
        cfg["scand_data"]["val_index"] = str(args.val_index)
    if args.output_dir is not None:
        cfg["training"]["output_dir"] = str(args.output_dir)
    if args.epochs is not None:
        cfg["training"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["dataloader"]["batch_size"] = int(args.batch_size)
    if args.grad_accum_steps is not None:
        cfg["training"]["grad_accum_steps"] = int(args.grad_accum_steps)
    if args.lr is not None:
        cfg["training"]["lr"] = float(args.lr)
    if args.device is not None:
        cfg["device"] = args.device
    if args.max_samples is not None:
        cfg["scand_data"]["max_samples"] = int(args.max_samples)
    if args.disable_lingbot_depth:
        cfg["model"]["vision_encoder"]["lingbot_depth"]["enabled"] = False
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU.", flush=True)
        return torch.device("cpu")
    return device


def torch_load(path: str | Path, device: torch.device) -> object:
    try:
        return torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=device)


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, enabled=True)


def action_stats_from_config(cfg: Mapping[str, Any]) -> dict[str, Sequence[float]]:
    stats = cfg.get("action_stats", DEFAULT_ACTION_STATS)
    return {"min": tuple(stats["min"]), "max": tuple(stats["max"])}


def load_stage1_model(checkpoint: str | Path, device: torch.device) -> torch.nn.Module:
    ckpt = torch_load(checkpoint, torch.device("cpu"))
    if not isinstance(ckpt, Mapping) or "cfg" not in ckpt or "model" not in ckpt:
        raise TypeError(f"Stage-1 checkpoint must contain cfg and model: {checkpoint}")
    model = build_stage1_route_model(ckpt["cfg"])
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(
        f"loaded Stage-1 checkpoint: path={checkpoint} "
        f"epoch={ckpt.get('epoch', 'unknown')} missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )
    return model


@torch.no_grad()
def encode_z_goal(stage1_model: torch.nn.Module, goal_video: torch.Tensor, device: torch.device, amp: bool) -> torch.Tensor:
    stage1_model.eval()
    with make_autocast(device, amp):
        outputs = stage1_model(goal_video, sample=False)
    return outputs["z_g"].detach()


def flatten_async_stage2_batch(
    batch: dict[str, Any],
    z_goal: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor, int]:
    """Flatten fast-system offsets while sharing one slow-system z_g per clip."""

    obs = batch["obs"]
    if not torch.is_tensor(obs) or obs.ndim != 6:
        return batch, z_goal, int(batch["action"].shape[0])

    slow_batch, fast_steps = int(obs.shape[0]), int(obs.shape[1])
    flat_batch = dict(batch)
    for key in ("obs", "action", "distance", "goal_pos", "action_mask", "async_offsets"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[:2] == (slow_batch, fast_steps):
            flat_batch[key] = value.reshape(slow_batch * fast_steps, *value.shape[2:])

    z_goal = z_goal.unsqueeze(1).expand(-1, fast_steps, -1).reshape(slow_batch * fast_steps, -1)
    return flat_batch, z_goal, slow_batch * fast_steps


def compute_stage2_loss(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    z_goal: torch.Tensor,
    action_stats: Mapping[str, Sequence[float]],
    goal_mask_prob: float,
    distance_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    obs = batch["obs"]
    actions = batch["action"]
    action_mask = batch.get("action_mask")
    batch_size = actions.shape[0]
    goal_mask = None
    if goal_mask_prob > 0:
        goal_mask = (torch.rand((batch_size,), device=obs.device) < goal_mask_prob).long()

    cond = model("vision_encoder", obs_img=obs, z_goal=z_goal, input_goal_mask=goal_mask)
    naction = normalize_deltas(actions_to_deltas(actions), action_stats)
    noise = torch.randn_like(naction)
    t, xt, ut = sample_cfm_location(noise, naction)
    vt = model("noise_pred_net", sample=xt, timestep=t, global_cond=cond)

    if action_mask is None:
        action_mask = torch.ones(batch_size, device=actions.device, dtype=actions.dtype)
    flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)
    loss = flow_loss

    dist_loss = cond.new_zeros(())
    if distance_loss_weight > 0:
        dist_pred = model("dist_pred_net", obsgoal_cond=cond).squeeze(-1)
        dist_loss = F.mse_loss(dist_pred, batch["distance"].float())
        loss = loss + float(distance_loss_weight) * dist_loss

    return loss, {
        "loss": float(loss.detach().cpu()),
        "flow_loss": float(flow_loss.detach().cpu()),
        "dist_loss": float(dist_loss.detach().cpu()),
    }


def update_meter(meter: dict[str, float], metrics: Mapping[str, float], n: int) -> None:
    meter["count"] = meter.get("count", 0.0) + n
    for key, value in metrics.items():
        meter[key] = meter.get(key, 0.0) + float(value) * n


def average_meter(meter: Mapping[str, float]) -> dict[str, float]:
    count = max(float(meter.get("count", 1.0)), 1.0)
    return {key: value / count for key, value in meter.items() if key != "count"}


def format_metrics(metrics: Mapping[str, float]) -> str:
    return " ".join(f"{key}={value:.4f}" for key, value in metrics.items())


def log_scalars(writer: Any, prefix: str, metrics: Mapping[str, float], step: int) -> None:
    if writer is None:
        return
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)
    writer.flush()


def has_index(data_cfg: Mapping[str, Any], split: str) -> bool:
    value = data_cfg.get(f"{split}_index") or data_cfg.get("index_path")
    if not value:
        return False
    path = Path(value).expanduser()
    return path.exists() and path.stat().st_size > 0


def run_epoch(
    model: torch.nn.Module,
    stage1_model: torch.nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    action_stats: Mapping[str, Sequence[float]],
    amp: bool,
    grad_clip: float,
    grad_accum_steps: int,
    goal_mask_prob: float,
    distance_loss_weight: float,
    log_interval: int,
    epoch: int,
    writer: Any = None,
    global_step: int = 0,
    progress: bool = True,
    split: str = "train",
) -> tuple[dict[str, float], int]:
    train = optimizer is not None
    grad_accum_steps = max(1, int(grad_accum_steps))
    num_batches = len(dataloader)
    remainder = num_batches % grad_accum_steps
    model.train(train)
    stage1_model.eval()
    meter: dict[str, float] = {}
    if train:
        optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(dataloader, desc=f"{split} epoch {epoch}", dynamic_ncols=True, leave=False, disable=not progress)
    for step, batch in enumerate(iterator, start=1):
        batch = move_to_device(batch, device)
        z_goal = encode_z_goal(stage1_model, batch["goal_video"], device=device, amp=amp)
        batch, z_goal, batch_size = flatten_async_stage2_batch(batch, z_goal)

        in_tail_group = remainder > 0 and step > num_batches - remainder
        accum_denom = remainder if in_tail_group else grad_accum_steps
        should_update = train and (step % grad_accum_steps == 0 or step == num_batches)
        mask_prob = goal_mask_prob if train else 0.0

        with torch.set_grad_enabled(train), make_autocast(device, amp):
            loss, metrics = compute_stage2_loss(
                model=model,
                batch=batch,
                z_goal=z_goal,
                action_stats=action_stats,
                goal_mask_prob=mask_prob,
                distance_loss_weight=distance_loss_weight,
            )

        if train:
            scaler.scale(loss / accum_denom).backward()
            if should_update:
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        update_meter(meter, metrics, batch_size)
        avg = average_meter(meter)
        if progress:
            iterator.set_postfix(loss=f"{avg['loss']:.3f}", flow=f"{avg['flow_loss']:.3f}")
        if train and should_update and log_interval > 0 and (global_step == 1 or global_step % log_interval == 0):
            log_scalars(writer, "train_step", avg, global_step)

    return average_meter(meter), global_step


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    cfg: Mapping[str, Any],
    best_val: float,
    global_step: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "cfg": dict(cfg),
            "best_val": best_val,
            "global_step": global_step,
        },
        path,
    )


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    seed_everything(int(cfg.get("seed", 0)))
    device = resolve_device(str(cfg.get("device", "cuda")))

    data_cfg = cfg.get("scand_data")
    if not isinstance(data_cfg, Mapping):
        raise KeyError("Config must contain scand_data with train_index/val_index.")
    loader_cfg = cfg.get("dataloader", {})
    train_loader = build_scand_stage2_dataloader_from_config(data_cfg, loader_cfg, split="train")
    val_loader = build_scand_stage2_dataloader_from_config(data_cfg, loader_cfg, split="val") if has_index(data_cfg, "val") else None

    stage1_ckpt = cfg.get("stage1", {}).get("checkpoint", FAST_SYSTEM_ROOT / "model_weights" / "stage1.pt")
    stage1_model = load_stage1_model(stage1_ckpt, device=device)

    model_cfg = cfg.get("model", {})
    depth_cfg = model_cfg.get("vision_encoder", {}).get("lingbot_depth", {})
    if bool(depth_cfg.get("enabled", True)):
        print(
            f"loading LingBot-Depth checkpoint: path={depth_cfg.get('checkpoint', 'default')} "
            f"freeze={depth_cfg.get('freeze', True)}",
            flush=True,
        )
    else:
        print("LingBot-Depth disabled; using zero depth token.", flush=True)
    model = build_fast_nav_model(model_cfg).to(device)
    flownav_ckpt = cfg.get("flownav", {}).get("checkpoint", DEFAULT_FLOWNAV_CHECKPOINT)
    load_info = load_flownav_checkpoint(model, flownav_ckpt)
    print(
        f"loaded FlowNav checkpoint: path={flownav_ckpt} loaded={load_info['loaded']} skipped={load_info['skipped']} "
        f"missing={len(load_info['missing'])}",
        flush=True,
    )

    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=float(train_cfg.get("lr", 1e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.05)),
    )
    use_amp = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch = 1
    global_step = 0
    best_val = float("inf")
    if args.resume is not None:
        ckpt = torch_load(args.resume, device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", best_val))

    out_dir = Path(train_cfg.get("output_dir", FAST_SYSTEM_ROOT / "checkpoints" / "stage2")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    writer = None
    if bool(train_cfg.get("tensorboard", True)):
        if SummaryWriter is None:
            print("TensorBoard is unavailable; install tensorboard to enable event logging.", flush=True)
        else:
            writer = SummaryWriter(log_dir=str(Path(train_cfg.get("tensorboard_dir", out_dir / "tb")).expanduser()))

    epochs = int(train_cfg.get("epochs", 50))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    log_interval = int(train_cfg.get("log_interval", 20))
    progress = bool(train_cfg.get("progress", True))
    goal_mask_prob = float(train_cfg.get("goal_mask_prob", 0.5))
    distance_loss_weight = float(train_cfg.get("distance_loss_weight", 0.0))
    action_stats = action_stats_from_config(cfg.get("model", {}))

    print(
        f"train samples={len(train_loader.dataset)} batches={len(train_loader)} "
        f"val samples={len(val_loader.dataset) if val_loader is not None else 0} device={device} "
        f"effective_batch={int(loader_cfg.get('batch_size', 8)) * grad_accum_steps}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        train_metrics, global_step = run_epoch(
            model=model,
            stage1_model=stage1_model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            action_stats=action_stats,
            amp=use_amp,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
            goal_mask_prob=goal_mask_prob,
            distance_loss_weight=distance_loss_weight,
            log_interval=log_interval,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            progress=progress,
            split="train",
        )
        log_scalars(writer, "train_epoch", train_metrics, epoch)
        if writer is not None:
            writer.add_scalar("train_epoch/lr", optimizer.param_groups[0]["lr"], epoch)
        print(f"epoch={epoch} train {format_metrics(train_metrics)}", flush=True)

        val_metrics = None
        if val_loader is not None:
            val_metrics, _ = run_epoch(
                model=model,
                stage1_model=stage1_model,
                dataloader=val_loader,
                optimizer=None,
                scaler=scaler,
                device=device,
                action_stats=action_stats,
                amp=use_amp,
                grad_clip=0.0,
                grad_accum_steps=1,
                goal_mask_prob=0.0,
                distance_loss_weight=distance_loss_weight,
                log_interval=0,
                epoch=epoch,
                writer=None,
                global_step=global_step,
                progress=progress,
                split="val",
            )
            log_scalars(writer, "val_epoch", val_metrics, epoch)
            print(f"epoch={epoch} val   {format_metrics(val_metrics)}", flush=True)

        score = val_metrics["loss"] if val_metrics is not None else train_metrics["loss"]
        is_best = score < best_val
        if is_best:
            best_val = score
        save_checkpoint(out_dir / "last.pt", model, optimizer, scaler, epoch, cfg, best_val, global_step)
        if is_best:
            save_checkpoint(out_dir / "best.pt", model, optimizer, scaler, epoch, cfg, best_val, global_step)

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
