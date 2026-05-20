#!/usr/bin/env python3
"""Train Stage-1 route adapter supervision on SCAND clips."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

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

from fast_system.dataset.scand_dataset import build_scand_dataloader_from_config
from fast_system.model.adapter import build_stage1_route_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the SCAND Stage-1 route adapter.")
    parser.add_argument("--config", type=Path, default=FAST_SYSTEM_ROOT / "configs" / "stage1.yaml")
    parser.add_argument("--train-index", type=Path, default=None)
    parser.add_argument("--val-index", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=Path, default=None)
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
        print("CUDA requested but unavailable; using CPU.")
        return torch.device("cpu")
    return device


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def stage1_loss(outputs: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor], beta_kl: float = 1e-4):
    anchor_gt = batch["route_anchors"]
    goal_gt = batch["coarse_goal"]
    turn_gt = batch["turn_label"]

    loss_anchor = F.smooth_l1_loss(outputs["anchor_pred"], anchor_gt)
    loss_goal = F.smooth_l1_loss(outputs["goal_pred"], goal_gt)
    loss_turn = F.cross_entropy(outputs["turn_logits"], turn_gt)
    loss_kl = kl_loss(outputs["mu"], outputs["logvar"])

    loss = 2.0 * loss_anchor + loss_goal + 0.5 * loss_turn + beta_kl * loss_kl
    return loss, {
        "loss": float(loss.detach().cpu()),
        "loss_anchor": float(loss_anchor.detach().cpu()),
        "loss_goal": float(loss_goal.detach().cpu()),
        "loss_turn": float(loss_turn.detach().cpu()),
        "loss_kl": float(loss_kl.detach().cpu()),
    }


def make_autocast(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "autocast"):
        return torch.autocast(device_type=device.type, enabled=True)
    return torch.cuda.amp.autocast(enabled=device.type == "cuda")


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
    dataloader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    beta_kl: float,
    amp: bool,
    grad_clip: float,
    grad_accum_steps: int,
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
    meter: dict[str, float] = {}
    if train:
        optimizer.zero_grad(set_to_none=True)
    iterator = tqdm(
        dataloader,
        desc=f"{split} epoch {epoch}",
        dynamic_ncols=True,
        leave=False,
        disable=not progress,
    )

    for step, batch in enumerate(iterator, start=1):
        batch = move_to_device(batch, device)
        batch_size = int(batch["video"].shape[0])

        in_tail_group = remainder > 0 and step > num_batches - remainder
        accum_denom = remainder if in_tail_group else grad_accum_steps
        should_update = train and (step % grad_accum_steps == 0 or step == num_batches)

        with torch.set_grad_enabled(train), make_autocast(device, amp):
            outputs = model(batch["video"], sample=train)
            loss, metrics = stage1_loss(outputs, batch, beta_kl=beta_kl)

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
        avg_metrics = average_meter(meter)
        if progress:
            postfix = dict(
                loss=f"{avg_metrics['loss']:.3f}",
                anchor=f"{avg_metrics['loss_anchor']:.3f}",
                goal=f"{avg_metrics['loss_goal']:.3f}",
                turn=f"{avg_metrics['loss_turn']:.3f}",
                kl=f"{avg_metrics['loss_kl']:.3f}",
            )
            if train and grad_accum_steps > 1:
                postfix["accum"] = f"{((step - 1) % grad_accum_steps) + 1}/{accum_denom}"
            iterator.set_postfix(**postfix)
        if train and should_update:
            if log_interval > 0 and (global_step == 1 or global_step % log_interval == 0):
                log_scalars(writer, "train_step", avg_metrics, global_step)

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
    train_loader = build_scand_dataloader_from_config(data_cfg, loader_cfg, split="train")
    val_loader = build_scand_dataloader_from_config(data_cfg, loader_cfg, split="val") if has_index(data_cfg, "val") else None

    model = build_stage1_route_model(cfg).to(device)
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
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt.get("scaler", {}))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_val = float(ckpt.get("best_val", best_val))

    out_dir = Path(train_cfg.get("output_dir", FAST_SYSTEM_ROOT / "checkpoints" / "stage1")).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    epochs = int(train_cfg.get("epochs", 20))
    beta_kl = float(train_cfg.get("beta_kl", 1e-4))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    log_interval = int(train_cfg.get("log_interval", 20))
    progress = bool(train_cfg.get("progress", True))

    writer = None
    if bool(train_cfg.get("tensorboard", True)):
        if SummaryWriter is None:
            print("TensorBoard is unavailable; install tensorboard to enable event logging.", flush=True)
        else:
            tb_dir = Path(train_cfg.get("tensorboard_dir", out_dir / "tb")).expanduser()
            writer = SummaryWriter(log_dir=str(tb_dir))
            print(f"tensorboard log_dir={tb_dir}", flush=True)

    print(
        f"train samples={len(train_loader.dataset)} batches={len(train_loader)} "
        f"val samples={len(val_loader.dataset) if val_loader is not None else 0} device={device} "
        f"effective_batch={int(loader_cfg.get('batch_size', 8)) * grad_accum_steps}",
        flush=True,
    )

    for epoch in range(start_epoch, epochs + 1):
        print(f"epoch={epoch} starting", flush=True)
        train_metrics, global_step = run_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            beta_kl=beta_kl,
            amp=use_amp,
            grad_clip=grad_clip,
            grad_accum_steps=grad_accum_steps,
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
                dataloader=val_loader,
                optimizer=None,
                scaler=scaler,
                device=device,
                beta_kl=beta_kl,
                amp=use_amp,
                grad_clip=0.0,
                grad_accum_steps=1,
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
