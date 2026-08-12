"""Frozen Reason1 plan generation for online fast-policy training."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from fast_system.model.plan_encoder import PLAN_SCHEMA_VERSION


class FrozenReason1Planner:
    """Keep Reason1 resident on GPU and generate detached plans per batch."""

    def __init__(self, cfg: Mapping[str, Any], device: torch.device) -> None:
        if device.type != "cuda":
            raise ValueError("Online Reason1 training requires a CUDA device.")
        if device.index is not None:
            torch.cuda.set_device(device)

        # Import lazily so offline training/tests do not initialize Cosmos.
        from slow_system.scripts.slow_server import CosmosSlowEngine, SlowServerConfig

        checkpoint = Path(str(cfg["checkpoint"])).expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Reason1 checkpoint not found: {checkpoint}")
        slow_cfg = SlowServerConfig(
            checkpoint_path=str(checkpoint.parent),
            checkpoint_filename=checkpoint.name,
            experiment=str(
                cfg.get(
                    "experiment",
                    "predict2_video2world_training_2b_vln_temporal_waypoint_17f_full_finetune",
                )
            ),
            config_file=str(cfg.get("config_file", "")),
            resolution="272,480",
            num_conditional_frames=2,
            num_steps=5,
            offload_diffusion_model=True,
            offload_text_encoder=False,
            offload_tokenizer=True,
            aggressive_unload=False,
        )
        caller_grad_enabled = torch.is_grad_enabled()
        self.engine = CosmosSlowEngine(slow_cfg)
        # Cosmos inference initialization disables grad globally. Restore the
        # caller's state so the subsequent fast-policy graph is unaffected.
        torch.set_grad_enabled(caller_grad_enabled)
        self.model = self.engine._pipe.model
        self.device = device
        self.log_interval = max(1, int(cfg.get("log_interval", 10)))
        self.log_enabled = bool(cfg.get("log_enabled", True))
        self.calls = 0

        self._offload_unused_video_modules()
        modules = (
            self.model.trajectory_reason_encoder,
            self.model.reason2_to_dit_proj,
        )
        self.model.trajectory_reason_encoder.to(device=device)
        self.model.reason2_to_dit_proj.to(**self.model.tensor_kwargs)
        for module in modules:
            module.requires_grad_(False)
            module.eval()
        if any(parameter.requires_grad for module in modules for parameter in module.parameters()):
            raise RuntimeError("Frozen Reason1 planner still has trainable parameters.")

    def _offload_unused_video_modules(self) -> None:
        """Keep DiT/video tokenizer off GPU; online training only uses Reason1."""

        for name in ("net", "conditioner", "waypoint_head"):
            module = getattr(self.model, name, None)
            if module is not None:
                setattr(self.model, name, module.to(device="cpu"))
        tokenizer = getattr(self.model, "tokenizer", None)
        if tokenizer is not None:
            for name in ("encoder", "decoder"):
                module = getattr(tokenizer, name, None)
                if module is not None:
                    setattr(tokenizer, name, module.to(device="cpu"))
        text_encoder = getattr(self.model, "text_encoder", None)
        text_model = getattr(text_encoder, "model", None)
        if text_model is not None:
            text_encoder.model = text_model.to(device="cpu")
        torch.cuda.empty_cache()

    @property
    def info(self) -> dict[str, Any]:
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "checkpoint": str(
                Path(self.engine.cfg.checkpoint_path).expanduser().resolve()
                / self.engine.cfg.checkpoint_filename
            ),
            "frozen": True,
            "resident": True,
        }

    def generate_batch(
        self,
        reason_video: torch.Tensor,
        instructions: Sequence[str],
    ) -> tuple[dict[str, torch.Tensor], float]:
        if reason_video.ndim != 5 or tuple(reason_video.shape[1:3]) != (3, 5):
            raise ValueError(
                "reason_video must contain the five Reason1 condition frames [B,3,5,H,W], "
                f"got {tuple(reason_video.shape)}."
            )
        if tuple(reason_video.shape[-2:]) != (272, 480):
            raise ValueError(f"reason_video must be 272x480, got {tuple(reason_video.shape[-2:])}.")
        if reason_video.dtype != torch.uint8:
            raise TypeError(f"reason_video must be uint8, got {reason_video.dtype}.")
        batch_size = int(reason_video.shape[0])
        if len(instructions) != batch_size:
            raise ValueError(f"Expected {batch_size} instructions, got {len(instructions)}.")

        # TrajectoryReason1Encoder converts these five condition frames to PIL,
        # so keep the video on CPU and avoid a pointless GPU round-trip.
        video = reason_video.detach().to(device="cpu", dtype=torch.uint8).contiguous()
        data_batch = {
            "dataset_name": "video_data",
            "video": video,
            "fps": torch.full((batch_size,), 10.0),
            "padding_mask": torch.zeros(batch_size, 1, 272, 480),
            "num_conditional_frames": 2,
            "ai_caption": [str(instruction) for instruction in instructions],
        }
        with torch.inference_mode():
            torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            reason_output = self.model.trajectory_reason_encoder(
                data_batch, self.model.input_caption_key
            )
            reason_context = self.model.reason2_to_dit_proj(
                reason_output.context_tokens.to(**self.model.tensor_kwargs)
            )
            torch.cuda.synchronize(self.device)
            elapsed = time.perf_counter() - started
            raw_plan = {
                "reason_context": reason_context,
                "context_mask": reason_output.context_mask,
                "traj_latents": reason_output.traj_latents,
            }
        # Tensors created inside inference_mode retain an inference-only flag.
        # Clone after leaving the context so trainable PlanEncoder layers can
        # save these constant inputs for their own weight gradients.
        plan = {
            "reason_context": raw_plan["reason_context"].detach().clone(),
            "context_mask": raw_plan["context_mask"].detach().clone().bool(),
            "traj_latents": raw_plan["traj_latents"].detach().clone(),
        }
        self._validate(plan, batch_size)
        self.calls += 1
        if self.log_enabled and self.calls % self.log_interval == 0:
            valid_lengths = plan["context_mask"].sum(dim=1).tolist()
            print(
                f"online_reason1 calls={self.calls} batch={batch_size} "
                f"elapsed={elapsed:.3f}s context_lengths={valid_lengths}",
                flush=True,
            )
        return plan, elapsed

    @staticmethod
    def _validate(plan: Mapping[str, torch.Tensor], batch_size: int) -> None:
        context = plan["reason_context"]
        mask = plan["context_mask"]
        trajectories = plan["traj_latents"]
        if context.ndim != 3 or context.shape[0] != batch_size or context.shape[-1] != 1024:
            raise ValueError(f"Unexpected online reason_context shape: {tuple(context.shape)}")
        if mask.shape != context.shape[:2] or not mask.any(dim=1).all():
            raise ValueError("Every online plan must have at least one valid context token.")
        if tuple(trajectories.shape) != (batch_size, 3, 3584):
            raise ValueError(f"Unexpected online traj_latents shape: {tuple(trajectories.shape)}")
        if any(value.requires_grad for value in plan.values()):
            raise RuntimeError("Online Reason1 outputs must be detached from autograd.")
        if any(torch.is_inference(value) for value in plan.values()):
            raise RuntimeError("Online Reason1 outputs must be normal tensors, not inference tensors.")
