#!/usr/bin/env python3
"""Train token-conditioned RGB-D FastNav with offline or frozen-online Reason1."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

try:
    import wandb
except ImportError:  # A clear error is raised at startup when logging is enabled.
    wandb = None

HAVANA_ROOT = Path(__file__).resolve().parents[2]
if str(HAVANA_ROOT) not in sys.path:
    sys.path.insert(0, str(HAVANA_ROOT))

from fast_system.dataset.intern_plan_dataset import (  # noqa: E402
    ResumableEpochSampler,
    build_intern_online_dataloader,
    build_intern_plan_dataloader,
)
from fast_system.model.nav_model import (  # noqa: E402
    DEFAULT_ACTION_STATS,
    actions_to_deltas,
    build_fast_nav_model,
    normalize_deltas,
    sample_cfm_location,
)
from fast_system.model.plan_encoder import PLAN_SCHEMA_VERSION, build_plan_encoder  # noqa: E402
from fast_system.model.online_plan import FrozenReason1Planner  # noqa: E402
from fast_system.trainer.train_stage_2 import load_config, resolve_device, seed_everything  # noqa: E402


STAGE2_BACKBONE_PREFIXES = (
    "vision_encoder.obs_encoder.",
    "vision_encoder.compress_obs_enc.",
    "vision_encoder.depth_token_proj.",
    "vision_encoder.compress_depth_enc.",
    "vision_encoder.sa_encoder.",
)
STAGE2_FULL_PREFIXES = STAGE2_BACKBONE_PREFIXES + ("noise_pred_net.", "dist_pred_net.")
STAGE2_ALWAYS_EXCLUDED_PREFIXES = (
    "vision_encoder.goal_encoder.",
    "vision_encoder.z_goal_projector.",
    "vision_encoder.z_plan_projector.",
    "vision_encoder.depth_encoder.",
)


class TrainableEMA:
    """EMA only for trainable parameters, excluding the frozen LingBot model.

    The decay warmup follows the diffusion-policy EMAModel convention.  A
    complete inference state is assembled only while checkpointing, so EMA
    does not require a second copy of frozen model parameters in GPU memory.
    """

    def __init__(
        self,
        modules: Mapping[str, torch.nn.Module],
        *,
        update_after_step: int = 0,
        inv_gamma: float = 1.0,
        power: float = 0.75,
        min_value: float = 0.0,
        max_value: float = 0.9999,
    ) -> None:
        if inv_gamma <= 0:
            raise ValueError("EMA inv_gamma must be positive.")
        if not 0.0 <= min_value <= max_value < 1.0:
            raise ValueError("EMA decay bounds must satisfy 0 <= min <= max < 1.")
        self.modules = dict(modules)
        self.update_after_step = int(update_after_step)
        self.inv_gamma = float(inv_gamma)
        self.power = float(power)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.optimization_step = 0
        self.decay = 0.0
        self.parameters = {
            f"{module_name}.{parameter_name}": parameter
            for module_name, module in self.modules.items()
            for parameter_name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        if not self.parameters:
            raise ValueError("EMA requires at least one trainable parameter.")
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }

    def get_decay(self, optimization_step: int) -> float:
        step = max(0, int(optimization_step) - self.update_after_step - 1)
        value = 1.0 - (1.0 + step / self.inv_gamma) ** (-self.power)
        if step <= 0:
            return 0.0
        return max(self.min_value, min(value, self.max_value))

    @torch.no_grad()
    def update(self) -> None:
        self.decay = self.get_decay(self.optimization_step)
        for name, parameter in self.parameters.items():
            average = self.shadow[name]
            average.mul_(self.decay).add_(parameter.detach(), alpha=1.0 - self.decay)
        self.optimization_step += 1

    @torch.no_grad()
    def reset_from_modules(self, optimization_step: int = 0) -> None:
        for name, parameter in self.parameters.items():
            self.shadow[name].copy_(parameter.detach())
        self.optimization_step = int(optimization_step)
        self.decay = self.get_decay(max(0, self.optimization_step - 1))

    @torch.no_grad()
    def load_averaged_state_dicts(
        self,
        states: Mapping[str, Mapping[str, torch.Tensor]],
        optimization_step: int,
    ) -> None:
        for full_name, average in self.shadow.items():
            module_name, parameter_name = full_name.split(".", 1)
            try:
                value = states[module_name][parameter_name]
            except KeyError as error:
                raise KeyError(f"EMA checkpoint is missing {full_name!r}.") from error
            average.copy_(value.to(device=average.device, dtype=average.dtype))
        self.optimization_step = int(optimization_step)
        self.decay = self.get_decay(max(0, self.optimization_step - 1))

    def averaged_state_dict(self, module_name: str) -> dict[str, torch.Tensor]:
        module = self.modules[module_name]
        state = dict(module.state_dict())
        prefix = f"{module_name}."
        for full_name, average in self.shadow.items():
            if full_name.startswith(prefix):
                state[full_name[len(prefix) :]] = average.detach().to(device="cpu", copy=True)
        return state

    def raw_trainable_state_dict(self, module_name: str) -> dict[str, torch.Tensor]:
        module = self.modules[module_name]
        return {
            name: parameter.detach().to(device="cpu", copy=True)
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "optimization_step": int(self.optimization_step),
            "decay": float(self.decay),
            "update_after_step": self.update_after_step,
            "inv_gamma": self.inv_gamma,
            "power": self.power,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "tracked_parameters": len(self.shadow),
        }

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        backup = {
            name: parameter.detach().clone()
            for name, parameter in self.parameters.items()
        }
        try:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(self.shadow[name])
            yield
        finally:
            with torch.no_grad():
                for name, parameter in self.parameters.items():
                    parameter.copy_(backup[name])


def build_constant_with_warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by a constant base learning rate."""

    warmup_steps = int(warmup_steps)
    if warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative.")

    def lr_factor(step: int) -> float:
        if warmup_steps == 0:
            return 1.0
        # LambdaLR evaluates step=0 during construction. This makes optimizer
        # update 1 use base_lr / warmup_steps and update N reach base_lr at N.
        return min(float(step + 1) / float(warmup_steps), 1.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)


def capture_rng_state() -> dict[str, Any]:
    """Capture stochastic training state needed for an exact continuation."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore RNG state saved immediately after a completed optimizer step."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda_states = state.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        for device_index, cuda_state in enumerate(cuda_states[: torch.cuda.device_count()]):
            torch.cuda.set_rng_state(cuda_state.cpu(), device=device_index)


def configure_loader_epoch(
    loader,
    epoch: int,
    start_index: int = 0,
    *,
    permutation_seed: int | None = None,
    legacy_global_seed: bool = False,
) -> None:
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, ResumableEpochSampler):
        raise TypeError("FastNav training requires ResumableEpochSampler for checkpointable data order.")
    sampler.set_epoch(
        epoch,
        start_index,
        permutation_seed=permutation_seed,
        legacy_global_seed=legacy_global_seed,
    )


def checkpoint_data_state(
    loader,
    epoch: int,
    samples_completed: int,
    batches_completed: int,
) -> dict[str, Any]:
    sampler = getattr(loader, "sampler", None)
    if not isinstance(sampler, ResumableEpochSampler):
        raise TypeError("Cannot checkpoint a loader without ResumableEpochSampler.")
    sampler_state = sampler.state_dict()
    sampler_state["start_index"] = int(samples_completed)
    return {
        "version": 1,
        "epoch": int(epoch),
        "samples_completed_in_epoch": int(samples_completed),
        "batches_completed_in_epoch": int(batches_completed),
        "dataset_size": int(sampler.dataset_size),
        "batch_size": int(loader.batch_size),
        "drop_last": bool(loader.drop_last),
        "sampler": sampler_state,
    }


def resolve_resume_data_state(checkpoint: Mapping[str, Any], loader) -> tuple[dict[str, Any], bool]:
    """Resolve a checkpoint cursor, including legacy batch-only checkpoints."""

    dataset_size = len(loader.dataset)
    epoch = int(checkpoint["epoch"])
    epoch_complete = bool(checkpoint.get("epoch_complete", True))
    saved = checkpoint.get("data_state")
    legacy = not isinstance(saved, Mapping)
    if legacy:
        saved_loader = checkpoint.get("config", {}).get("dataloader", {})
        saved_batch_size = int(saved_loader.get("batch_size", loader.batch_size))
        saved_batches_completed = int(checkpoint.get("batch_index", 0))
        samples_completed = min(dataset_size, saved_batches_completed * saved_batch_size)
        batches_completed = (samples_completed + int(loader.batch_size) - 1) // int(loader.batch_size)
        permutation_seed = None
    else:
        saved_size = int(saved.get("dataset_size", dataset_size))
        if saved_size != dataset_size:
            raise ValueError(
                "Cannot exactly resume DataLoader because the filtered dataset size changed: "
                f"checkpoint={saved_size}, current={dataset_size}."
            )
        samples_completed = int(saved.get("samples_completed_in_epoch", 0))
        batches_completed = (samples_completed + int(loader.batch_size) - 1) // int(loader.batch_size)
        sampler_state = saved.get("sampler", {})
        permutation_seed = (
            sampler_state.get("permutation_seed")
            if isinstance(sampler_state, Mapping)
            else None
        )
        if not 0 <= samples_completed <= dataset_size:
            raise ValueError(
                f"Invalid checkpoint DataLoader cursor {samples_completed}/{dataset_size}."
            )

    resume_same_epoch = not epoch_complete and samples_completed < dataset_size
    if not resume_same_epoch:
        epoch += 1
        samples_completed = 0
        batches_completed = 0
        permutation_seed = None
    return {
        "data_epoch": epoch,
        "samples_completed_in_epoch": samples_completed,
        "batches_completed_in_epoch": batches_completed,
        "sampler_permutation_seed": permutation_seed,
        "legacy_sampler_seed": legacy and resume_same_epoch,
    }, legacy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).parents[1] / "configs" / "plan_stage2.yaml")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path, default=None)
    checkpoint_group.add_argument(
        "--stage2-init",
        type=Path,
        default=None,
        help="Initialize compatible weights from a legacy Stage-2 checkpoint or directory; never restores optimizer state.",
    )
    parser.add_argument(
        "--stage2-init-mode",
        choices=("backbone", "full"),
        default="backbone",
        help="backbone excludes the CFM/distance heads; full loads them when shapes match.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--plan-source", choices=("online", "offline"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def resolve_stage2_init_checkpoint(path: Path) -> Path:
    """Resolve a legacy checkpoint without guessing among multiple unrelated files."""

    path = path.expanduser().resolve()
    if path.is_file():
        if path.stat().st_size == 0:
            raise ValueError(f"Stage-2 initialization checkpoint is empty: {path}")
        return path
    if not path.exists():
        raise FileNotFoundError(f"Stage-2 initialization path does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"Stage-2 initialization path must be a file or directory: {path}")

    for preferred_name in ("best.pt", "last.pt", "best.pth", "last.pth"):
        preferred = path / preferred_name
        if preferred.is_file() and preferred.stat().st_size > 0:
            return preferred
    candidates = sorted(
        candidate
        for pattern in ("*.pt", "*.pth", "*.ckpt")
        for candidate in path.glob(pattern)
        if candidate.is_file() and candidate.stat().st_size > 0
    )
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(f"No non-empty .pt/.pth/.ckpt file found in {path}")
    names = ", ".join(candidate.name for candidate in candidates)
    raise ValueError(f"Multiple Stage-2 checkpoints found in {path}; pass one file explicitly: {names}")


def _legacy_model_state(checkpoint: object, path: Path) -> Mapping[str, torch.Tensor]:
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Legacy Stage-2 checkpoint must be a mapping: {path}")
    for key in ("model", "state_dict"):
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    available = ", ".join(str(key) for key in checkpoint.keys())
    raise KeyError(f"Legacy Stage-2 checkpoint has no model/state_dict tensor mapping: {path}; keys={available}")


def _canonical_stage2_key(key: str, current: Mapping[str, torch.Tensor]) -> str:
    while key.startswith(("module.", "_orig_mod.")):
        key = key.split(".", 1)[1]
    if key.startswith("model.") and key.removeprefix("model.") in current:
        key = key.removeprefix("model.")
    return key


def select_stage2_pretrained_state(
    current: Mapping[str, torch.Tensor],
    legacy: Mapping[str, torch.Tensor],
    mode: str,
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Select only semantically compatible Stage-2 tensors with matching shapes."""

    allowed_prefixes = STAGE2_BACKBONE_PREFIXES if mode == "backbone" else STAGE2_FULL_PREFIXES
    if mode not in {"backbone", "full"}:
        raise ValueError(f"Unsupported Stage-2 initialization mode: {mode}")
    selected: dict[str, torch.Tensor] = {}
    counts = {"loaded": 0, "excluded": 0, "not_allowed": 0, "missing": 0, "shape_mismatch": 0}
    for raw_key, value in legacy.items():
        if not isinstance(raw_key, str) or not torch.is_tensor(value):
            counts["not_allowed"] += 1
            continue
        key = _canonical_stage2_key(raw_key, current)
        if key.startswith(STAGE2_ALWAYS_EXCLUDED_PREFIXES):
            counts["excluded"] += 1
        elif not key.startswith(allowed_prefixes):
            counts["not_allowed"] += 1
        elif key not in current:
            counts["missing"] += 1
        elif tuple(value.shape) != tuple(current[key].shape):
            counts["shape_mismatch"] += 1
        else:
            selected[key] = value
            counts["loaded"] += 1
    return selected, counts


def load_stage2_pretrained(model: torch.nn.Module, path: Path, mode: str = "backbone") -> dict[str, Any]:
    """Partially initialize the new plan policy from a legacy Stage-2 checkpoint."""

    checkpoint_path = resolve_stage2_init_checkpoint(path)
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    legacy = _legacy_model_state(checkpoint, checkpoint_path)
    current = model.state_dict()
    selected, counts = select_stage2_pretrained_state(current, legacy, mode)
    if not selected:
        raise ValueError(f"No compatible Stage-2 tensors found in {checkpoint_path} for mode={mode}.")
    if not any(key.startswith("vision_encoder.obs_encoder.") for key in selected):
        raise ValueError(f"Checkpoint did not provide a compatible RGB observation encoder: {checkpoint_path}")
    if not any(key.startswith("vision_encoder.sa_encoder.") for key in selected):
        raise ValueError(f"Checkpoint did not provide a compatible Stage-2 Transformer: {checkpoint_path}")
    model.load_state_dict(selected, strict=False)
    return {
        "path": str(checkpoint_path),
        "mode": mode,
        "source_epoch": checkpoint.get("epoch") if isinstance(checkpoint, Mapping) else None,
        **counts,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {name: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for name, value in batch.items()}


def init_wandb(config: Mapping[str, Any], output_dir: Path):
    logging_cfg = config.get("wandb", {})
    if not bool(logging_cfg.get("enabled", True)):
        return None
    if wandb is None:
        raise RuntimeError("W&B logging is enabled, but wandb is not installed in this environment.")
    mode = str(logging_cfg.get("mode", "online"))
    if mode == "online" and not getattr(wandb.api, "api_key", None):
        raise RuntimeError(
            "W&B online logging requires authentication. Run "
            "`cosmos-predict2.5/.venv/bin/wandb login` or set WANDB_API_KEY before training."
        )
    init_kwargs = {
        "project": str(logging_cfg.get("project", "havana-fast-nav")),
        "name": logging_cfg.get("name"),
        "entity": logging_cfg.get("entity"),
        "group": logging_cfg.get("group"),
        "tags": list(logging_cfg.get("tags", ())),
        "mode": mode,
        "dir": str(output_dir),
        "config": dict(config),
    }
    return wandb.init(**{key: value for key, value in init_kwargs.items() if value is not None})


def wandb_batch_payload(
    split: str,
    epoch: int,
    batch_index: int,
    metrics: Mapping[str, float],
    lr: float | None = None,
) -> dict[str, float | int]:
    payload: dict[str, float | int] = {
        "epoch": int(epoch),
        f"{split}/batch": int(batch_index),
        **{f"{split}/{name}": float(value) for name, value in metrics.items()},
    }
    if lr is not None:
        payload["train/learning_rate"] = float(lr)
    return payload


def flatten_async(
    batch: Mapping[str, torch.Tensor], plan_tokens: torch.Tensor
) -> tuple[dict[str, Any], torch.Tensor]:
    obs = batch["obs"]
    if obs.ndim != 6:
        return dict(batch), plan_tokens
    batch_size, offsets = obs.shape[:2]
    flat = dict(batch)
    for name in (
        "obs",
        "depth",
        "action",
        "action_mask",
        "distance",
        "async_offsets",
        "plan_times",
    ):
        value = batch[name]
        flat[name] = value.reshape(batch_size * offsets, *value.shape[2:])
    if plan_tokens.shape[:2] != (batch_size, offsets):
        raise ValueError(
            f"plan token offsets {tuple(plan_tokens.shape[:2])} do not match obs {(batch_size, offsets)}."
        )
    plan_tokens = plan_tokens.reshape(batch_size * offsets, *plan_tokens.shape[2:])
    return flat, plan_tokens


def compute_loss(
    model: torch.nn.Module,
    plan_encoder: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    goal_mask_prob: float,
    distance_weight: float,
    action_stats: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    plan_tokens = plan_encoder(
        batch["reason_context"],
        batch["context_mask"],
        batch["traj_latents"],
        batch["plan_times"],
    )
    slow_batch_size = int(batch["obs"].shape[0])
    num_offsets = int(batch["obs"].shape[1]) if batch["obs"].ndim == 6 else 1
    flat, plan_tokens = flatten_async(batch, plan_tokens)
    sample_count = flat["obs"].shape[0]
    input_goal_mask = None
    if goal_mask_prob > 0:
        slow_drop = torch.rand(slow_batch_size, device=plan_tokens.device) < float(goal_mask_prob)
        input_goal_mask = slow_drop[:, None].expand(-1, num_offsets).reshape(-1).long()
    cond = model(
        "vision_encoder",
        obs_img=flat["obs"],
        depth=flat["depth"],
        plan_tokens=plan_tokens,
        input_goal_mask=input_goal_mask,
    )

    normalized = normalize_deltas(actions_to_deltas(flat["action"]), action_stats)
    noise = torch.randn_like(normalized)
    t, xt, velocity = sample_cfm_location(noise, normalized)
    prediction = model(
        "noise_pred_net",
        sample=xt,
        timestep=t,
        global_cond=cond["global_cond"],
        context=cond["context"],
        context_mask=cond["context_mask"],
    )
    per_step = F.mse_loss(prediction, velocity, reduction="none").mean(dim=-1)
    action_mask = flat["action_mask"].to(per_step)
    flow_loss = (per_step * action_mask).sum() / action_mask.sum().clamp_min(1.0)
    distance_prediction = model("dist_pred_net", obsgoal_cond=cond["global_cond"]).squeeze(-1)
    distance_loss = F.mse_loss(distance_prediction, flat["distance"].float())
    loss = flow_loss + float(distance_weight) * distance_loss
    return loss, {
        "loss": float(loss.detach()),
        "flow_loss": float(flow_loss.detach()),
        "distance_loss": float(distance_loss.detach()),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    plan_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    config: Mapping[str, Any],
    *,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: TrainableEMA | None = None,
    global_step: int = 0,
    batch_index: int = 0,
    epoch_complete: bool = False,
    data_state: Mapping[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if ema is None:
        model_state = model.state_dict()
        plan_encoder_state = plan_encoder.state_dict()
        raw_model_state = None
        raw_plan_encoder_state = None
        ema_metadata = {"enabled": False}
    else:
        # Existing inference code loads these two complete states directly;
        # expose EMA here while retaining only the trainable raw delta needed
        # to resume optimization. Frozen LingBot weights are stored once.
        model_state = ema.averaged_state_dict("model")
        plan_encoder_state = ema.averaged_state_dict("plan_encoder")
        raw_model_state = ema.raw_trainable_state_dict("model")
        raw_plan_encoder_state = ema.raw_trainable_state_dict("plan_encoder")
        ema_metadata = ema.metadata()
    payload = {
        "model": model_state,
        "plan_encoder": plan_encoder_state,
        "raw_model": raw_model_state,
        "raw_plan_encoder": raw_plan_encoder_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict(),
        "epoch": int(epoch),
        "batch_index": int(batch_index),
        "epoch_complete": bool(epoch_complete),
        "global_step": int(global_step),
        "data_state": dict(data_state) if data_state is not None else None,
        "rng_state": capture_rng_state(),
        "ema": ema_metadata,
        "config": dict(config),
        "plan_schema_version": PLAN_SCHEMA_VERSION,
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def update_last_checkpoint(checkpoint_path: Path, last_path: Path) -> None:
    """Atomically point ``last.pt`` at a step checkpoint without duplicating it."""

    checkpoint_path = checkpoint_path.resolve()
    last_path = last_path.resolve()
    if checkpoint_path == last_path:
        return
    temporary = last_path.with_name(f".{last_path.name}.{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        try:
            os.link(checkpoint_path, temporary)
        except OSError:
            shutil.copy2(checkpoint_path, temporary)
        os.replace(temporary, last_path)
    finally:
        temporary.unlink(missing_ok=True)


def save_step_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    plan_encoder: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    ema: TrainableEMA | None,
    *,
    epoch: int,
    batch_index: int,
    global_step: int,
    config: Mapping[str, Any],
    epoch_complete: bool = False,
    data_state: Mapping[str, Any] | None = None,
    name: str | None = None,
) -> Path:
    checkpoint_path = output_dir / (name or f"step_{int(global_step):09d}.pt")
    save_checkpoint(
        checkpoint_path,
        model,
        plan_encoder,
        optimizer,
        scaler,
        epoch,
        config,
        scheduler=scheduler,
        ema=ema,
        global_step=global_step,
        batch_index=batch_index,
        epoch_complete=epoch_complete,
        data_state=data_state,
    )
    update_last_checkpoint(checkpoint_path, output_dir / "last.pt")
    return checkpoint_path


def run_epoch(
    model: torch.nn.Module,
    plan_encoder: torch.nn.Module,
    loader,
    optimizer,
    scaler,
    device: torch.device,
    amp: bool,
    config: Mapping[str, Any],
    train: bool,
    reason1_planner: FrozenReason1Planner | None = None,
    epoch: int = 1,
    total_epochs: int = 1,
    wandb_run=None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ema: TrainableEMA | None = None,
    training_state: dict[str, Any] | None = None,
    output_dir: Path | None = None,
) -> dict[str, float]:
    model.train(train)
    plan_encoder.train(train)
    # The LingBot RGB-D backbone is frozen and must remain in eval mode.
    depth_encoder = getattr(model.vision_encoder, "depth_encoder", None)
    if depth_encoder is not None:
        depth_encoder.eval()
    totals = {
        "loss": 0.0,
        "flow_loss": 0.0,
        "distance_loss": 0.0,
        "reason1_latency": 0.0,
        "count": 0.0,
    }
    if train:
        optimizer.zero_grad(set_to_none=True)
    training = config["training"]
    if training_state is None:
        training_state = {
            "global_step": 0,
            "last_saved_step": 0,
            "data_epoch": epoch,
            "samples_completed_in_epoch": 0,
            "batches_completed_in_epoch": 0,
        }
    save_every_steps = max(0, int(training.get("save_every_steps", 0)))
    split = "train" if train else "val"
    log_every = max(1, int(config.get("wandb", {}).get("log_every", 1)))
    stats_cfg = config["model"].get("action_stats", DEFAULT_ACTION_STATS)
    action_stats = {
        "min": tuple(stats_cfg["min"]),
        "max": tuple(stats_cfg["max"]),
        "clip": bool(stats_cfg.get("clip", False)),
    }
    samples_completed = int(training_state.get("samples_completed_in_epoch", 0)) if train else 0
    loader_batch_size = max(1, int(getattr(loader, "batch_size", 1) or 1))
    display_initial = samples_completed // loader_batch_size
    progress = tqdm(
        loader,
        total=display_initial + len(loader),
        initial=display_initial,
        desc=f"{split} epoch {epoch}/{total_epochs}",
        dynamic_ncols=True,
        leave=True,
    )
    for relative_batch_index, batch in enumerate(progress, start=1):
        source_batch_size = int(batch["obs"].shape[0])
        batch_index = relative_batch_index
        reason1_latency = 0.0
        if reason1_planner is not None:
            reason_video = batch.pop("reason_video")
            instructions = batch.pop("instruction")
            online_plan, reason1_latency = reason1_planner.generate_batch(reason_video, instructions)
            batch.update(online_plan)
        batch = move_batch(batch, device)
        autocast = torch.autocast(device_type=device.type, enabled=amp) if amp else nullcontext()
        with torch.set_grad_enabled(train), autocast:
            loss_args = (
                batch,
                float(training.get("plan_dropout_prob", 0.5)) if train else 0.0,
                float(training.get("distance_loss_weight", 1e-4)),
                action_stats,
            )
            loss, metrics = compute_loss(model, plan_encoder, *loss_args)
        metrics["reason1_latency"] = float(reason1_latency)
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [parameter for group in optimizer.param_groups for parameter in group["params"]],
                float(training.get("grad_clip", 1.0)),
            )
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            training_state["data_epoch"] = int(epoch)
            training_state["samples_completed_in_epoch"] = int(
                training_state.get("samples_completed_in_epoch", 0)
            ) + source_batch_size
            training_state["batches_completed_in_epoch"] = int(
                training_state.get("batches_completed_in_epoch", 0)
            ) + 1
            batch_index = int(training_state["batches_completed_in_epoch"])
            optimizer_updated = not scaler.is_enabled() or float(scaler.get_scale()) >= scale_before
            if optimizer_updated:
                if scheduler is not None:
                    scheduler.step()
                if ema is not None:
                    ema.update()
                training_state["global_step"] = int(training_state.get("global_step", 0)) + 1
                global_step = training_state["global_step"]
                if save_every_steps > 0 and global_step % save_every_steps == 0:
                    if output_dir is None:
                        raise ValueError("output_dir is required for step-based checkpointing.")
                    checkpoint_path = save_step_checkpoint(
                        output_dir,
                        model,
                        plan_encoder,
                        optimizer,
                        scaler,
                        scheduler,
                        ema,
                        epoch=epoch,
                        batch_index=batch_index,
                        global_step=global_step,
                        config=config,
                        data_state=checkpoint_data_state(
                            loader,
                            epoch,
                            int(training_state["samples_completed_in_epoch"]),
                            int(training_state["batches_completed_in_epoch"]),
                        ),
                    )
                    training_state["last_saved_step"] = global_step
                    print(f"saved checkpoint: {checkpoint_path}", flush=True)
        log_metrics = metrics
        count = int(batch["obs"].shape[0] * batch["obs"].shape[1])
        totals["count"] += count
        for name, value in metrics.items():
            totals[name] += value * count
        progress.set_postfix(
            loss=f"{log_metrics['loss']:.4f}",
            flow=f"{log_metrics['flow_loss']:.4f}",
            dist=f"{log_metrics['distance_loss']:.4f}",
            reason1=f"{log_metrics['reason1_latency']:.3f}s",
        )
        if wandb_run is not None and batch_index % log_every == 0:
            lr = float(optimizer.param_groups[0]["lr"]) if train else None
            payload = wandb_batch_payload(split, epoch, batch_index, log_metrics, lr)
            payload["global_step"] = int(training_state.get("global_step", 0))
            if ema is not None:
                payload["train/ema_decay"] = float(ema.decay)
            wandb_run.log(payload)
    count = max(totals.pop("count"), 1.0)
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.max_samples is not None:
        config["intern_data"]["max_samples"] = args.max_samples
    if args.plan_source is not None:
        config["intern_data"]["plan_source"] = args.plan_source
    device = resolve_device(str(config.get("device", "cuda")))
    output_dir = Path(config["training"]["output_dir"]).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_run = init_wandb(config, output_dir)
    seed_everything(int(config.get("seed", 0)))
    data_config = config["intern_data"]
    plan_source = str(data_config.get("plan_source", "offline"))
    reason1_planner: FrozenReason1Planner | None = None
    if plan_source == "online":
        train_loader = build_intern_online_dataloader(
            data_config,
            config["dataloader"],
            "train",
        )
        val_loader = build_intern_online_dataloader(
            data_config,
            config["dataloader"],
            "val",
        )
        reason1_cfg = dict(config["online_reason1"])
        reason1_planner = FrozenReason1Planner(reason1_cfg, device)
        config["online_reason1_runtime"] = reason1_planner.info
        print(f"online Reason1 ready: {reason1_planner.info}", flush=True)
    elif plan_source == "offline":
        train_loader = build_intern_plan_dataloader(
            data_config,
            config["dataloader"],
            "train",
        )
        val_loader = (
            build_intern_plan_dataloader(
                data_config,
                config["dataloader"],
                "val",
            )
            if data_config.get("val_manifest") else None
        )
    else:
        raise ValueError(f"Unsupported intern_data.plan_source={plan_source!r}.")
    model = build_fast_nav_model(config["model"]).to(device)
    plan_encoder = build_plan_encoder(config.get("plan_encoder", {})).to(device)
    if args.stage2_init is not None:
        init_info = load_stage2_pretrained(model, args.stage2_init, args.stage2_init_mode)
        config["stage2_initialization"] = init_info
        print(
            "initialized legacy Stage-2: "
            + " ".join(f"{name}={value}" for name, value in init_info.items()),
            flush=True,
        )
        if args.stage2_init_mode == "full":
            print(
                "full initialization loaded compatible CFM/distance tensors; "
                "this assumes identical action normalization, units, horizon, and distance targets.",
                flush=True,
            )
    parameters = [
        parameter
        for module in (model, plan_encoder)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(config["training"].get("lr", 1e-4)),
        weight_decay=float(config["training"].get("weight_decay", 0.05)),
    )
    scheduler_name = str(config["training"].get("lr_scheduler", "constant")).lower()
    if scheduler_name != "constant":
        raise ValueError(f"FastNav supports lr_scheduler='constant', got {scheduler_name!r}.")
    scheduler = build_constant_with_warmup_scheduler(
        optimizer,
        int(config["training"].get("lr_warmup_steps", 500)),
    )
    amp = bool(config["training"].get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    ema_cfg = config.get("ema", {})
    ema = None
    if bool(ema_cfg.get("enabled", True)):
        ema = TrainableEMA(
            {"model": model, "plan_encoder": plan_encoder},
            update_after_step=int(ema_cfg.get("update_after_step", 0)),
            inv_gamma=float(ema_cfg.get("inv_gamma", 1.0)),
            power=float(ema_cfg.get("power", 0.75)),
            min_value=float(ema_cfg.get("min_value", 0.0)),
            max_value=float(ema_cfg.get("max_value", 0.9999)),
        )
    start_epoch = 1
    training_state = {
        "global_step": 0,
        "last_saved_step": 0,
        "data_epoch": 1,
        "samples_completed_in_epoch": 0,
        "batches_completed_in_epoch": 0,
        "sampler_permutation_seed": None,
        "legacy_sampler_seed": False,
    }
    resume_rng_state = None
    if args.resume is not None:
        checkpoint = torch.load(str(args.resume), map_location=device, weights_only=False)
        if checkpoint.get("plan_schema_version") != PLAN_SCHEMA_VERSION:
            raise ValueError("Refusing to resume a checkpoint that does not use the v2 token plan schema.")
        # `model` and `plan_encoder` are EMA inference weights in new
        # checkpoints. Load them first so the tracker can recover its average,
        # then overlay raw trainable parameters for continued optimization.
        model.load_state_dict(checkpoint["model"], strict=True)
        plan_encoder.load_state_dict(checkpoint["plan_encoder"], strict=True)
        raw_model = checkpoint.get("raw_model")
        raw_plan_encoder = checkpoint.get("raw_plan_encoder")
        if ema is not None:
            ema_info = checkpoint.get("ema", {}) or {}
            if bool(ema_info.get("enabled", False)):
                ema.load_averaged_state_dicts(
                    {"model": checkpoint["model"], "plan_encoder": checkpoint["plan_encoder"]},
                    int(ema_info.get("optimization_step", checkpoint.get("global_step", 0))),
                )
        if isinstance(raw_model, Mapping) and isinstance(raw_plan_encoder, Mapping):
            model.load_state_dict(raw_model, strict=False)
            plan_encoder.load_state_dict(raw_plan_encoder, strict=False)
        elif ema is not None:
            # Compatibility with v2 checkpoints written before EMA support.
            ema.reset_from_modules(int(checkpoint.get("global_step", 0)))
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler") is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        global_step = int(checkpoint.get("global_step", 0))
        data_resume, legacy_data_state = resolve_resume_data_state(checkpoint, train_loader)
        training_state = {
            "global_step": global_step,
            "last_saved_step": global_step,
            **data_resume,
        }
        start_epoch = int(data_resume["data_epoch"])
        samples_completed = int(data_resume["samples_completed_in_epoch"])
        current_batch = (
            samples_completed + int(train_loader.batch_size) - 1
        ) // int(train_loader.batch_size)
        total_batches = (
            len(train_loader.dataset) + int(train_loader.batch_size) - 1
        ) // int(train_loader.batch_size)
        legacy_note = (
            " Legacy checkpoint has no saved sampler/RNG state; its sample cursor was "
            "reconstructed from saved batch_index and batch_size."
            if legacy_data_state
            else ""
        )
        print(
            f"resuming at global_step={global_step}, epoch={start_epoch}, "
            f"samples={samples_completed}/{len(train_loader.dataset)} "
            f"(current batches={current_batch}/{total_batches}).{legacy_note}",
            flush=True,
        )
        resume_rng_state = checkpoint.get("rng_state")
        del checkpoint

    (output_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    if wandb_run is not None:
        wandb_run.config.update(dict(config), allow_val_change=True)
        print(f"wandb run: {wandb_run.url}", flush=True)
    if isinstance(resume_rng_state, Mapping):
        restore_rng_state(resume_rng_state)
    total_epochs = int(config["training"].get("epochs", 50))
    current_epoch = start_epoch
    train_epoch_complete = False
    try:
        for epoch in range(start_epoch, total_epochs + 1):
            current_epoch = epoch
            if int(training_state.get("data_epoch", epoch)) != epoch:
                training_state["data_epoch"] = epoch
                training_state["samples_completed_in_epoch"] = 0
                training_state["batches_completed_in_epoch"] = 0
                training_state["sampler_permutation_seed"] = None
                training_state["legacy_sampler_seed"] = False
            configure_loader_epoch(
                train_loader,
                epoch,
                int(training_state.get("samples_completed_in_epoch", 0)),
                permutation_seed=training_state.get("sampler_permutation_seed"),
                legacy_global_seed=bool(training_state.get("legacy_sampler_seed", False)),
            )
            train_epoch_complete = False
            train_metrics = run_epoch(
                model,
                plan_encoder,
                train_loader,
                optimizer,
                scaler,
                device,
                amp,
                config,
                True,
                reason1_planner,
                epoch,
                total_epochs,
                wandb_run,
                scheduler,
                ema,
                training_state,
                output_dir,
            )
            train_epoch_complete = True
            print(f"epoch={epoch} train={train_metrics}", flush=True)
            if wandb_run is not None:
                wandb_run.log({f"epoch/train_{name}": value for name, value in train_metrics.items()} | {"epoch": epoch})
            if val_loader is not None:
                ema_context = ema.average_parameters() if ema is not None else nullcontext()
                with ema_context:
                    val_metrics = run_epoch(
                        model,
                        plan_encoder,
                        val_loader,
                        optimizer,
                        scaler,
                        device,
                        amp,
                        config,
                        False,
                        reason1_planner,
                        epoch,
                        total_epochs,
                        wandb_run,
                        scheduler,
                        ema,
                        training_state,
                        output_dir,
                    )
                print(f"epoch={epoch} val={val_metrics}", flush=True)
                if wandb_run is not None:
                    wandb_run.log({f"epoch/val_{name}": value for name, value in val_metrics.items()} | {"epoch": epoch})
        final_step = int(training_state["global_step"])
        final_path = save_step_checkpoint(
            output_dir,
            model,
            plan_encoder,
            optimizer,
            scaler,
            scheduler,
            ema,
            epoch=total_epochs,
            batch_index=len(train_loader),
            global_step=final_step,
            config=config,
            epoch_complete=True,
            data_state=checkpoint_data_state(
                train_loader,
                total_epochs,
                len(train_loader.dataset),
                int(training_state.get("batches_completed_in_epoch", len(train_loader))),
            ),
            name="final.pt",
        )
        training_state["last_saved_step"] = final_step
        print(f"saved final checkpoint: {final_path}", flush=True)
    except BaseException:
        emergency_step = int(training_state.get("global_step", 0))
        if emergency_step > int(training_state.get("last_saved_step", 0)):
            try:
                emergency_path = save_step_checkpoint(
                    output_dir,
                    model,
                    plan_encoder,
                    optimizer,
                    scaler,
                    scheduler,
                    ema,
                    epoch=current_epoch,
                    batch_index=int(training_state.get("batches_completed_in_epoch", 0)),
                    global_step=emergency_step,
                    config=config,
                    epoch_complete=train_epoch_complete,
                    data_state=checkpoint_data_state(
                        train_loader,
                        current_epoch,
                        int(training_state.get("samples_completed_in_epoch", 0)),
                        int(training_state.get("batches_completed_in_epoch", 0)),
                    ),
                )
                training_state["last_saved_step"] = emergency_step
                print(f"saved emergency checkpoint: {emergency_path}", flush=True)
            except Exception as checkpoint_error:
                print(f"failed to save emergency checkpoint: {checkpoint_error}", flush=True)
        raise
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
