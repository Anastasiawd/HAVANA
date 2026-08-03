# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""17-frame temporal-waypoint VLN training model."""

from __future__ import annotations

import re
from typing import Any, Optional

import attrs
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.context_parallel import broadcast_split_tensor
from cosmos_predict2._src.imaginaire.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2._src.predict2.conditioner import Text2WorldCondition
from cosmos_predict2._src.predict2.models.vln_temporal_waypoint_utils import (
    global_latent_loss_weights,
    mask_projected_context,
    masked_mean_or_graph_zero,
    random_cyclic_captions,
)
from cosmos_predict2._src.predict2.models.vln_waypoint_model_rectified_flow import (
    TrajectoryReason1Output,
    VLNWaypointVideo2WorldModelRectifiedFlow,
    VLNWaypointVideo2WorldModelRectifiedFlowConfig,
    _TrainableModules,
)


@attrs.define(slots=False)
class VLNTemporalWaypointVideo2WorldModelRectifiedFlowConfig(VLNWaypointVideo2WorldModelRectifiedFlowConfig):
    dit_context_mode: str = "full"
    waypoint_condition_dropout_prob: float = 0.1
    mirrored_waypoint_loss_weight: float = 0.05
    mirrored_waypoint_margin: float = 0.02
    language_instruction_loss_weight: float = 0.05
    language_instruction_margin: float = 0.05
    language_instruction_t_min: float = 0.2
    language_instruction_t_max: float = 0.8
    teacher_forcing_gt_steps: int = 0
    teacher_forcing_decay_steps: int = 0
    detach_predicted_waypoints_for_denoise: bool = False
    freeze_dit_backbone: bool = False
    backbone_lr: float = 3e-6
    conditioning_lr: float = 3e-5

    def __attrs_post_init__(self):
        super().__attrs_post_init__()
        if self.dit_context_mode not in {"vision_only", "full"}:
            raise ValueError(f"Unsupported dit_context_mode={self.dit_context_mode}")
        if self.num_waypoints != 3:
            raise ValueError("Temporal waypoint model requires exactly three waypoints")
        if self.state_t != 5:
            raise ValueError("17-frame temporal waypoint model requires state_t=5")
        if self.min_num_conditional_frames != 2 or self.max_num_conditional_frames != 2:
            raise ValueError("Temporal waypoint model requires exactly two conditional latents")


class VLNTemporalWaypointVideo2WorldModelRectifiedFlow(VLNWaypointVideo2WorldModelRectifiedFlow):
    """Independent temporal experiment; the legacy 29-frame model stays unchanged."""

    def __init__(self, config: VLNTemporalWaypointVideo2WorldModelRectifiedFlowConfig):
        super().__init__(config)
        self._current_iteration = 0
        self._inference_reason_cache: tuple[TrajectoryReason1Output, torch.Tensor, torch.Tensor] | None = None
        self._configure_trainable_modules()

    @torch.inference_mode()
    def prepare_reason_conditioning_for_inference(self, data_batch: dict[str, Any]) -> None:
        """Precompute Reason1 outputs before the diffusion network is loaded."""
        self._inference_reason_cache = None
        self.trajectory_reason_encoder.to(device="cuda")
        self.trajectory_reason_encoder.model.eval()
        self.reason2_to_dit_proj.to(**self.tensor_kwargs)
        self.waypoint_head.to(**self.tensor_kwargs)
        self._inference_reason_cache = self._predict_reason_context_and_waypoints(data_batch)

    def offload_reason_conditioner_for_inference(self) -> None:
        """Move temporal-waypoint conditioning modules to CPU."""
        self.trajectory_reason_encoder.to(device="cpu")
        self.reason2_to_dit_proj.to(device="cpu")
        self.waypoint_head.to(device="cpu")
        torch.cuda.empty_cache()

    def clear_reason_conditioning_cache_for_inference(self) -> None:
        """Release any cached Reason1 tensors after sampling or an error."""
        self._inference_reason_cache = None

    def _configure_trainable_modules(self) -> None:
        self.trajectory_reason_encoder.model.requires_grad_(False)
        # This pipeline always supplies already-projected 1024-D Reason1
        # context. The original 100352->1024 projection is outside the active
        # graph and should not be placed in the full-finetune optimizer.
        if hasattr(self.net, "crossattn_proj"):
            self.net.crossattn_proj.requires_grad_(False)
        if not self.config.freeze_dit_backbone:
            return
        self.net.requires_grad_(False)
        temporal_module = getattr(self.net, "temporal_waypoint_embedder", None)
        if temporal_module is None:
            raise TypeError("freeze_dit_backbone=True requires TemporalWaypointMinimalV1LVGDiT")
        temporal_module.requires_grad_(True)

    def training_step(
        self,
        data_batch: dict[str, torch.Tensor],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        self._current_iteration = int(iteration)
        self._update_train_stats(data_batch)
        return self.forward(data_batch)

    def init_optimizer_scheduler(
        self,
        optimizer_config: LazyDict,
        scheduler_config: LazyDict,
    ) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
        self._optimizer_module = _TrainableModules(
            self.net,
            self.trajectory_reason_encoder,
            self.reason2_to_dit_proj,
            self.waypoint_head,
        )
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            DTensor = ()

        grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
        for name, parameter in self._optimizer_module.named_parameters():
            if not parameter.requires_grad:
                continue
            is_conditioning = (
                "temporal_waypoint_embedder" in name
                or name.startswith("trajectory_encoder.traj_queries")
                or name.startswith("reason2_to_dit_proj")
                or name.startswith("waypoint_head")
            )
            kind = "conditioning" if is_conditioning else "backbone"
            grouped.setdefault((kind, isinstance(parameter, DTensor)), []).append(parameter)

        parameter_groups = []
        for kind in ("backbone", "conditioning"):
            for is_dtensor in (True, False):
                parameters = grouped.get((kind, is_dtensor), [])
                if parameters:
                    parameter_groups.append(
                        {
                            "params": parameters,
                            "lr": (self.config.conditioning_lr if kind == "conditioning" else self.config.backbone_lr),
                            "group_name": f"{kind}_{'dtensor' if is_dtensor else 'tensor'}",
                        }
                    )
        optimizer = lazy_instantiate(optimizer_config, params=parameter_groups)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        log.info(
            "Temporal VLN optimizer groups: "
            + ", ".join(
                f"{group['group_name']}={len(group['params'])} params @ {group['lr']:.2e}" for group in parameter_groups
            )
        )
        return optimizer, scheduler

    @staticmethod
    def _select_tokens(
        context: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        selected = [context[index][token_mask[index]] for index in range(context.shape[0])]
        max_length = max(tokens.shape[0] for tokens in selected)
        output = context.new_zeros(context.shape[0], max_length, context.shape[-1])
        output_mask = torch.zeros(
            context.shape[0],
            max_length,
            dtype=torch.bool,
            device=context.device,
        )
        for index, tokens in enumerate(selected):
            output[index, : tokens.shape[0]] = tokens
            output_mask[index, : tokens.shape[0]] = True
        return output, output_mask

    def _dit_reason_context(
        self,
        reason_output: TrajectoryReason1Output,
        projected_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.config.dit_context_mode == "full":
            context_mask = reason_output.context_mask.to(device=projected_context.device)
            # The DiT cross-attention API has no padding-mask input. At least
            # keep padded tokens neutral after the bias-bearing projection;
            # otherwise every padded position becomes the same learned token.
            projected_context = mask_projected_context(projected_context, context_mask)
            return (
                projected_context,
                context_mask,
            )
        return self._select_tokens(
            projected_context,
            reason_output.vision_mask.to(device=projected_context.device),
        )

    def _condition_with_waypoints(
        self,
        condition: Text2WorldCondition,
        waypoints_norm: torch.Tensor,
        waypoint_condition_mask: Optional[torch.Tensor] = None,
    ) -> Text2WorldCondition:
        kwargs = condition.to_dict(skip_underscore=False)
        kwargs["crossattn_emb_preprojected"] = True
        kwargs["waypoints_norm"] = waypoints_norm
        kwargs["waypoint_condition_mask"] = waypoint_condition_mask
        return type(condition)(**kwargs)

    def _teacher_forcing_probability(self) -> float:
        step = self._current_iteration
        if step < self.config.teacher_forcing_gt_steps:
            return 1.0
        if self.config.teacher_forcing_decay_steps <= 0:
            return 0.0
        decay_step = step - self.config.teacher_forcing_gt_steps
        return float(max(0.0, 1.0 - decay_step / self.config.teacher_forcing_decay_steps))

    def _waypoints_for_denoise(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        probability = self._teacher_forcing_probability() if self.training else 0.0
        if probability <= 0:
            selected = predicted
        elif probability >= 1:
            selected = target
        else:
            use_target = (
                torch.rand(
                    predicted.shape[0],
                    1,
                    1,
                    device=predicted.device,
                )
                < probability
            )
            selected = torch.where(use_target, target, predicted)
        if self.config.detach_predicted_waypoints_for_denoise:
            selected = selected.detach()
        return selected, probability

    def _waypoint_condition_mask(self, waypoints: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.waypoint_condition_dropout_prob <= 0:
            return torch.ones(waypoints.shape[0], device=waypoints.device, dtype=waypoints.dtype)
        keep = torch.rand(waypoints.shape[0], device=waypoints.device)
        return (keep >= self.config.waypoint_condition_dropout_prob).to(dtype=waypoints.dtype)

    @staticmethod
    def _random_cyclic_captions(captions: Any, batch_size: int) -> Optional[list[str]]:
        return random_cyclic_captions(captions, batch_size)

    @staticmethod
    def _clone_condition(
        condition: Text2WorldCondition,
        *,
        context: Optional[torch.Tensor] = None,
        waypoints: Optional[torch.Tensor] = None,
        waypoint_mask: Optional[torch.Tensor] = None,
    ) -> Text2WorldCondition:
        kwargs = condition.to_dict(skip_underscore=False)
        if context is not None:
            kwargs["crossattn_emb"] = context
        if waypoints is not None:
            kwargs["waypoints_norm"] = waypoints
        if waypoint_mask is not None:
            kwargs["waypoint_condition_mask"] = waypoint_mask
        return type(condition)(**kwargs)

    def _negative_instruction_condition(
        self,
        data_batch: dict[str, Any],
        base_condition: Text2WorldCondition,
        batch_size: int,
        waypoint_condition_mask: torch.Tensor,
    ) -> Optional[Text2WorldCondition]:
        if self.config.language_instruction_loss_weight <= 0:
            return None
        negative_captions = self._random_cyclic_captions(
            data_batch[self.input_caption_key],
            batch_size,
        )
        if negative_captions is None:
            return None
        negative_batch = dict(data_batch)
        negative_batch[self.input_caption_key] = negative_captions
        with torch.no_grad():
            reason_output, projected_context, predicted_waypoints = self._predict_reason_context_and_waypoints(
                negative_batch
            )
            dit_context, _ = self._dit_reason_context(reason_output, projected_context)
        return self._clone_condition(
            base_condition,
            context=dit_context.detach(),
            waypoints=predicted_waypoints.detach(),
            waypoint_mask=waypoint_condition_mask.detach().to(
                device=predicted_waypoints.device,
                dtype=predicted_waypoints.dtype,
            ),
        )

    @staticmethod
    def _global_latent_loss_weights(frame_mask: torch.Tensor) -> torch.Tensor:
        return global_latent_loss_weights(frame_mask)

    def _local_latent_loss_weights(
        self,
        data_batch: dict[str, Any],
        local_latent_t: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        frame_mask = data_batch["video_frame_loss_mask"].to(device=device, dtype=dtype)
        weights = self._global_latent_loss_weights(frame_mask)
        context_group = self.get_context_parallel_group()
        context_size = 1 if context_group is None else context_group.size()
        if context_size > 1:
            if context_size > weights.shape[1] or weights.shape[1] % context_size != 0:
                raise ValueError(
                    "Temporal masked loss supports context-parallel temporal splits only; "
                    f"state_t=5 is incompatible with context_parallel_size={context_size}"
                )
            weights = broadcast_split_tensor(weights, seq_dim=1, process_group=context_group)
        if weights.shape[1] != local_latent_t:
            raise ValueError(f"Latent loss weights T={weights.shape[1]} do not match local latent T={local_latent_t}")
        return weights

    @staticmethod
    def _masked_per_instance_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        latent_weights: torch.Tensor,
    ) -> torch.Tensor:
        error_by_time = (prediction - target).square().mean(dim=(1, 3, 4))
        denominator = latent_weights.sum(dim=1).clamp_min(1e-6)
        return (error_by_time * latent_weights).sum(dim=1) / denominator

    def _mirrored_waypoint_loss(
        self,
        predicted_waypoints: torch.Tensor,
        condition: Text2WorldCondition,
        actions: Any,
        noise: torch.Tensor,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        target_velocity: torch.Tensor,
        latent_weights: torch.Tensor,
        positive_per_instance: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zero = predicted_waypoints.new_zeros(())
        if self.config.mirrored_waypoint_loss_weight <= 0:
            return zero, zero
        if isinstance(actions, str):
            actions = [actions] * predicted_waypoints.shape[0]
        valid = torch.tensor(
            [action in {"TURN_LEFT", "TURN_RIGHT"} for action in actions],
            device=predicted_waypoints.device,
            dtype=torch.bool,
        )
        condition_mask = condition.waypoint_condition_mask
        if condition_mask is not None:
            condition_mask = condition_mask.to(device=valid.device).reshape(valid.shape[0], -1)
            valid &= condition_mask.bool().any(dim=1)
        # Every rank in an FSDP shard group must execute the same sequence of
        # DiT forwards. Validation follows the natural (mostly-straight) data
        # distribution, so branching here on a rank-local action batch can
        # otherwise desynchronize FSDP all-gathers. Compute the mirrored
        # prediction on every rank and mask the loss to turn samples below.
        mirrored = predicted_waypoints.detach().clone()
        mirrored[..., 0] *= -1
        mirrored[..., 2] *= -1
        mirrored_condition = self._clone_condition(condition, waypoints=mirrored)
        mirrored_prediction = self.denoise(
            noise=noise,
            xt_B_C_T_H_W=xt.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=mirrored_condition,
        )
        mirrored_per_instance = self._masked_per_instance_loss(
            mirrored_prediction,
            target_velocity,
            latent_weights,
        )
        sensitivity = mirrored_per_instance.detach() - positive_per_instance.detach()
        margin_loss = F.relu(self.config.mirrored_waypoint_margin + positive_per_instance - mirrored_per_instance)
        return masked_mean_or_graph_zero(margin_loss, valid), masked_mean_or_graph_zero(sensitivity, valid)

    def _language_instruction_loss(
        self,
        negative_condition: Optional[Text2WorldCondition],
        noise: torch.Tensor,
        xt: torch.Tensor,
        timesteps: torch.Tensor,
        continuous_time: torch.Tensor,
        target_velocity: torch.Tensor,
        latent_weights: torch.Tensor,
        positive_per_instance: torch.Tensor,
    ) -> torch.Tensor:
        if negative_condition is None:
            return positive_per_instance.new_zeros(())
        context_group = self.get_context_parallel_group()
        if context_group is not None:
            negative_condition = negative_condition.broadcast(context_group)
        negative_prediction = self.denoise(
            noise=noise,
            xt_B_C_T_H_W=xt.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=negative_condition,
        )
        negative_per_instance = self._masked_per_instance_loss(
            negative_prediction,
            target_velocity,
            latent_weights,
        )
        time = continuous_time.flatten()
        valid = (time >= self.config.language_instruction_t_min) & (time <= self.config.language_instruction_t_max)
        hinge = F.relu(self.config.language_instruction_margin + positive_per_instance - negative_per_instance)
        # The sampled time window is rank-local. Preserve an autograd edge to
        # the negative DiT pass when this rank has no valid item, otherwise
        # FSDP backward hooks can diverge across ranks.
        return masked_mean_or_graph_zero(hinge, valid)

    def _temporal_waypoint_metrics(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        data_batch: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        metrics: dict[str, torch.Tensor] = {}
        difference = (predicted.detach() - target.detach()).abs()
        actions = data_batch.get("canonical_action", [])
        if isinstance(actions, str):
            actions = [actions] * predicted.shape[0]
        sources = data_batch.get("source_id", [])
        if isinstance(sources, str):
            sources = [sources] * predicted.shape[0]

        def add_bucket(prefix: str, labels: Any, label: str) -> None:
            indices = [index for index, value in enumerate(labels) if value == label]
            if not indices:
                return
            selector = torch.tensor(indices, device=predicted.device, dtype=torch.long)
            key = re.sub(r"[^a-zA-Z0-9_]+", "_", label.lower())
            bucket_diff = difference.index_select(0, selector)
            metrics[f"{prefix}_{key}_waypoint_mae"] = bucket_diff.mean()
            target_yaw = target.index_select(0, selector)[..., 2]
            predicted_yaw = predicted.detach().index_select(0, selector)[..., 2]
            informative = target_yaw.abs() >= (0.1 / np.pi)
            if informative.any():
                metrics[f"{prefix}_{key}_yaw_sign_accuracy"] = (
                    (torch.sign(target_yaw) == torch.sign(predicted_yaw))[informative].float().mean()
                )

        for action in ("GO_STRAIGHT", "TURN_LEFT", "TURN_RIGHT"):
            add_bucket("action", actions, action)
        for source_id in sorted(set(sources)):
            add_bucket("source", sources, source_id)

        clamped = data_batch.get("waypoint_clamped_mask")
        if clamped is not None:
            metrics["waypoint_clamp_rate"] = clamped.to(device=predicted.device).float().mean()
        return metrics

    @torch.inference_mode()
    def get_velocity_fn_from_batch(
        self,
        data_batch: dict[str, Any],
        guidance: float = 1.0,
        is_negative_prompt: bool = False,
    ):
        del is_negative_prompt
        if self._inference_reason_cache is None:
            reason_output, projected_context, predicted_waypoints = self._predict_reason_context_and_waypoints(data_batch)
        else:
            reason_output, projected_context, predicted_waypoints = self._inference_reason_cache
            self._inference_reason_cache = None
        dit_context, dit_mask = self._dit_reason_context(reason_output, projected_context)
        data_batch["t5_text_embeddings"] = dit_context
        data_batch["t5_text_mask"] = dit_mask
        _, x0, condition = self.get_data_and_condition(data_batch)
        waypoint_on = torch.ones(
            predicted_waypoints.shape[0],
            device=predicted_waypoints.device,
            dtype=predicted_waypoints.dtype,
        )
        waypoint_off = torch.zeros_like(waypoint_on)
        condition = self._condition_with_waypoints(condition, predicted_waypoints, waypoint_on)
        context_only_condition = self._clone_condition(condition, waypoint_mask=waypoint_off)
        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, context_only_condition, _, _ = self.broadcast_split_for_model_parallelsim(
            x0,
            context_only_condition,
            None,
            None,
        )
        self.last_pred_waypoints_norm = predicted_waypoints.detach()

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            waypoint_velocity = self.denoise(noise, noise_x, timestep, condition)
            context_velocity = self.denoise(noise, noise_x, timestep, context_only_condition)
            return context_velocity + guidance * (waypoint_velocity - context_velocity)

        return velocity_fn

    def forward(
        self,
        data_batch: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        reason_output, projected_context, predicted_waypoints = self._predict_reason_context_and_waypoints(data_batch)
        dit_context, dit_mask = self._dit_reason_context(reason_output, projected_context)
        data_batch["t5_text_embeddings"] = dit_context
        data_batch["t5_text_mask"] = dit_mask

        _, x0, base_condition = self.get_data_and_condition(data_batch)
        target_waypoints = data_batch["ego_waypoints_norm"].to(
            device=predicted_waypoints.device,
            dtype=predicted_waypoints.dtype,
        )
        denoise_waypoints, teacher_forcing_probability = self._waypoints_for_denoise(
            predicted_waypoints,
            target_waypoints,
        )
        waypoint_condition_mask = self._waypoint_condition_mask(denoise_waypoints)
        condition = self._condition_with_waypoints(
            base_condition,
            denoise_waypoints,
            waypoint_condition_mask,
        )
        negative_condition = self._negative_instruction_condition(
            data_batch,
            base_condition,
            x0.shape[0],
            waypoint_condition_mask,
        )

        epsilon = torch.randn(x0.size(), **self.tensor_kwargs_fp32)
        continuous_time = rearrange(
            self.rectified_flow.sample_train_time(x0.shape[0]).to(**self.tensor_kwargs_fp32),
            "b -> b 1",
        )
        x0, condition, epsilon, continuous_time = self.broadcast_split_for_model_parallelsim(
            x0,
            condition,
            epsilon,
            continuous_time,
        )
        timesteps = self.rectified_flow.get_discrete_timestamp(
            continuous_time,
            self.tensor_kwargs_fp32,
        )
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)
        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt, target_velocity = self.rectified_flow.get_interpolation(epsilon, x0, sigmas)

        prediction = self.denoise(
            noise=epsilon,
            xt_B_C_T_H_W=xt.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )
        latent_weights = self._local_latent_loss_weights(
            data_batch,
            prediction.shape[2],
            prediction.device,
            prediction.dtype,
        )
        positive_per_instance = self._masked_per_instance_loss(
            prediction,
            target_velocity,
            latent_weights,
        )
        time_weights = self.rectified_flow.train_time_weight(
            timesteps,
            self.tensor_kwargs_fp32,
        )
        denoise_loss = (time_weights.flatten() * positive_per_instance).mean()
        waypoint_loss, waypoint_metrics = self._waypoint_loss_and_metrics(
            predicted_waypoints,
            data_batch,
        )
        mirrored_loss, sensitivity_margin = self._mirrored_waypoint_loss(
            predicted_waypoints,
            condition,
            data_batch.get("canonical_action", []),
            epsilon,
            xt,
            timesteps,
            target_velocity,
            latent_weights,
            positive_per_instance,
        )
        language_loss = self._language_instruction_loss(
            negative_condition,
            epsilon,
            xt,
            timesteps,
            continuous_time,
            target_velocity,
            latent_weights,
            positive_per_instance,
        )
        loss = (
            denoise_loss
            + self.config.waypoint_loss_weight * waypoint_loss
            + self.config.mirrored_waypoint_loss_weight * mirrored_loss
            + self.config.language_instruction_loss_weight * language_loss
        )

        metrics = self._temporal_waypoint_metrics(
            predicted_waypoints,
            target_waypoints,
            data_batch,
        )
        metrics.update(waypoint_metrics)
        metrics.update(
            {
                "denoise_loss": denoise_loss.detach(),
                "mirrored_waypoint_margin_loss": mirrored_loss.detach(),
                "waypoint_sensitivity_margin": sensitivity_margin.detach(),
                "language_instruction_loss": language_loss.detach(),
                "valid_latent_ratio": latent_weights[:, 2:].mean().detach(),
                "teacher_forcing_probability": prediction.new_tensor(teacher_forcing_probability),
                "waypoint_condition_keep_rate": waypoint_condition_mask.float().mean().detach(),
            }
        )
        output_batch = {
            "x0": x0,
            "xt": xt,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": prediction,
            "edm_loss": denoise_loss,
            "timesteps": timesteps,
            "per_instance_loss": positive_per_instance,
            "n_cond_frames": condition.num_conditional_frames_B,
            "pred_waypoints_norm": predicted_waypoints.detach(),
            **metrics,
        }
        return output_batch, loss
