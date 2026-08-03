# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import collections
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import attrs
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from megatron.core import parallel_state
from omegaconf import open_dict
from PIL import Image
from torch import nn
from torch.nn.modules.module import _IncompatibleKeys
from torch.nn.utils.clip_grad import clip_grad_norm_

from cosmos_predict2._src.imaginaire.lazy_config import LazyDict
from cosmos_predict2._src.imaginaire.lazy_config import instantiate as lazy_instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.misc import get_local_tensor_if_DTensor
from cosmos_predict2._src.imaginaire.utils.optim_instantiate import get_base_scheduler
from cosmos_predict2._src.predict2.conditioner import Text2WorldCondition
from cosmos_predict2._src.predict2.models.video2world_model_rectified_flow import (
    Video2WorldModelRectifiedFlow,
    Video2WorldModelRectifiedFlowConfig,
)
from cosmos_predict2._src.predict2.text_encoders.text_encoder import TextEncoder, TextEncoderConfig


@dataclass
class TrajectoryReason1Output:
    context_tokens: torch.Tensor
    context_mask: torch.Tensor
    text_mask: torch.Tensor
    vision_mask: torch.Tensor
    traj_latents: torch.Tensor


class TrajectoryReason1Encoder(nn.Module):
    """Frozen Reason1 encoder with trainable trajectory query embeddings."""

    def __init__(
        self,
        config: TextEncoderConfig,
        num_traj_tokens: int = 4,
        hidden_size: int = 3584,
        condition_frame_positions: Sequence[int] = (0, 1, 2, 3, 4),
        device: str = "cuda",
    ) -> None:
        super().__init__()
        self.config = config
        self.num_traj_tokens = num_traj_tokens
        self.hidden_size = hidden_size
        self.condition_frame_positions = tuple(condition_frame_positions)
        self.device = device

        self._disable_reason1_fsdp2(config)
        text_encoder = TextEncoder(config, device=device)
        self.model = text_encoder.model
        self._assert_reason1_is_not_dtensor_parallelized()
        self.model.requires_grad_(False)
        self.model.eval()
        self.traj_queries = nn.Parameter(torch.empty(num_traj_tokens, hidden_size, device=device))
        nn.init.normal_(self.traj_queries, mean=0.0, std=0.02)

    @staticmethod
    def _disable_reason1_fsdp2(config: TextEncoderConfig) -> None:
        reason_model_config = getattr(getattr(config, "model_config", None), "model_config", None)
        if reason_model_config is None:
            return
        try:
            with open_dict(reason_model_config):
                reason_model_config.use_fsdp2 = False
        except Exception:
            reason_model_config.use_fsdp2 = False

    def _assert_reason1_is_not_dtensor_parallelized(self) -> None:
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            return
        for name, param in self.model.named_parameters():
            if isinstance(param, DTensor):
                raise RuntimeError(
                    "Reason1 was still tensor-parallelized into DTensor parameters "
                    f"(first DTensor parameter: {name}). Ensure "
                    "trajectory_reason_encoder_config.model_config.model_config.use_fsdp2=False "
                    "before QwenVLBaseModel is instantiated."
                )

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _sample_frames_as_pil(self, video_B_C_T_H_W: torch.Tensor, sample_idx: int) -> list[Image.Image]:
        video = video_B_C_T_H_W[sample_idx]
        if video.dtype != torch.uint8:
            video = torch.clamp((video.float() + 1.0) * 127.5, 0, 255).to(torch.uint8)
        max_t = video.shape[1]
        positions = [min(pos, max_t - 1) for pos in self.condition_frame_positions]
        frames = video[:, positions].permute(1, 2, 3, 0).detach().cpu().numpy()
        return [Image.fromarray(frame) for frame in frames]

    def _feature_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device)
        return value

    def _build_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor],
        pixel_values_videos: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        video_grid_thw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        with torch.no_grad():
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            if pixel_values is not None:
                pixel_values = pixel_values.type(self.model.visual.dtype)
                image_embeds = self.model.visual(pixel_values, grid_thw=image_grid_thw)
                n_image_tokens = (input_ids == self.model.config.image_token_id).sum().item()
                if n_image_tokens != image_embeds.shape[0]:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens {n_image_tokens}, features {image_embeds.shape[0]}"
                    )
                image_mask = (input_ids == self.model.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds.to(inputs_embeds))

            if pixel_values_videos is not None:
                pixel_values_videos = pixel_values_videos.type(self.model.visual.dtype)
                video_embeds = self.model.visual(pixel_values_videos, grid_thw=video_grid_thw)
                n_video_tokens = (input_ids == self.model.config.video_token_id).sum().item()
                if n_video_tokens != video_embeds.shape[0]:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens {n_video_tokens}, features {video_embeds.shape[0]}"
                    )
                video_mask = (input_ids == self.model.config.video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds.to(inputs_embeds))

        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[:, -self.num_traj_tokens :] = self.traj_queries.to(
            device=inputs_embeds.device, dtype=inputs_embeds.dtype
        ).unsqueeze(0)
        return inputs_embeds

    def _encode_one(
        self,
        caption: str,
        frames: list[Image.Image],
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        conversations = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant who will provide prompts to an image generator.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames, "fps": fps},
                    {"type": "text", "text": caption},
                ],
            },
        ]
        features = self.model.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=False,
            add_vision_id=False,
        )
        features = {key: self._feature_to_device(value) for key, value in features.items()}

        input_ids_base = features["input_ids"].unsqueeze(0).to(device=self.device)
        attention_mask_base = features["attention_mask"].unsqueeze(0).to(device=self.device)
        eos_id = self.model.tokenizer.eos_id if self.model.tokenizer.eos_id is not None else self.model.tokenizer.pad_id
        traj_ids = torch.full(
            (1, self.num_traj_tokens), eos_id, dtype=input_ids_base.dtype, device=input_ids_base.device
        )
        input_ids = torch.cat([input_ids_base, traj_ids], dim=1)
        attention_mask = torch.cat([attention_mask_base, torch.ones_like(traj_ids)], dim=1)

        image_grid_thw = features.get("image_grid_thw")
        video_grid_thw = features.get("video_grid_thw")
        second_per_grid_ts = features.get("second_per_grid_ts")
        inputs_embeds = self._build_inputs_embeds(
            input_ids=input_ids,
            pixel_values=features.get("pixel_values"),
            pixel_values_videos=features.get("pixel_values_videos"),
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
        )
        _, outputs = self.model._forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = outputs.hidden_states[-1] if outputs.hidden_states is not None else outputs.last_hidden_state
        base_len = input_ids_base.shape[1]
        context_hidden = hidden[0, :base_len]
        traj_latents = hidden[:, -self.num_traj_tokens :].squeeze(0)

        base_attention = attention_mask_base.squeeze(0).bool()
        base_ids = input_ids_base.squeeze(0)
        vision_mask_full = torch.zeros_like(base_attention)
        if getattr(self.model.config, "video_token_id", None) is not None:
            vision_mask_full |= base_ids == self.model.config.video_token_id
        if getattr(self.model.config, "image_token_id", None) is not None:
            vision_mask_full |= base_ids == self.model.config.image_token_id

        context_hidden = context_hidden[base_attention]
        vision_mask = vision_mask_full[base_attention]
        text_mask = ~vision_mask
        context_mask = torch.ones(context_hidden.shape[0], dtype=torch.bool, device=context_hidden.device)
        return context_hidden, context_mask, text_mask, vision_mask, traj_latents

    def forward(self, data_batch: dict[str, Any], input_caption_key: str) -> TrajectoryReason1Output:
        video = data_batch["video"]
        captions = data_batch[input_caption_key]
        fps_values = data_batch.get("fps", [10.0] * video.shape[0])
        if isinstance(fps_values, torch.Tensor):
            fps_values = fps_values.detach().cpu().tolist()

        context_tokens = []
        context_masks = []
        text_masks = []
        vision_masks = []
        traj_latents = []
        for sample_idx in range(video.shape[0]):
            frames = self._sample_frames_as_pil(video, sample_idx)
            caption = captions[sample_idx] if isinstance(captions, (list, tuple)) else captions
            outputs = self._encode_one(caption=str(caption), frames=frames, fps=float(fps_values[sample_idx]))
            context_tokens.append(outputs[0])
            context_masks.append(outputs[1])
            text_masks.append(outputs[2])
            vision_masks.append(outputs[3])
            traj_latents.append(outputs[4])

        max_len = max(tokens.shape[0] for tokens in context_tokens)
        dtype = context_tokens[0].dtype
        device = context_tokens[0].device
        padded_context = torch.zeros(len(context_tokens), max_len, self.hidden_size, dtype=dtype, device=device)
        padded_context_mask = torch.zeros(len(context_tokens), max_len, dtype=torch.bool, device=device)
        padded_text_mask = torch.zeros_like(padded_context_mask)
        padded_vision_mask = torch.zeros_like(padded_context_mask)
        for idx, tokens in enumerate(context_tokens):
            length = tokens.shape[0]
            padded_context[idx, :length] = tokens
            padded_context_mask[idx, :length] = context_masks[idx]
            padded_text_mask[idx, :length] = text_masks[idx]
            padded_vision_mask[idx, :length] = vision_masks[idx]

        return TrajectoryReason1Output(
            context_tokens=padded_context,
            context_mask=padded_context_mask,
            text_mask=padded_text_mask,
            vision_mask=padded_vision_mask,
            traj_latents=torch.stack(traj_latents, dim=0),
        )


class _TrainableModules(nn.Module):
    def __init__(self, net: nn.Module, trajectory_encoder: nn.Module, reason2_to_dit_proj: nn.Module, waypoint_head: nn.Module):
        super().__init__()
        self.net = net
        self.trajectory_encoder = trajectory_encoder
        self.reason2_to_dit_proj = reason2_to_dit_proj
        self.waypoint_head = waypoint_head


def _default_trajectory_reason_encoder_config() -> TextEncoderConfig:
    config = TextEncoderConfig(compute_online=True)
    TrajectoryReason1Encoder._disable_reason1_fsdp2(config)
    return config


@attrs.define(slots=False)
class VLNWaypointVideo2WorldModelRectifiedFlowConfig(Video2WorldModelRectifiedFlowConfig):
    trajectory_reason_encoder_config: Optional[TextEncoderConfig] = attrs.field(
        factory=_default_trajectory_reason_encoder_config
    )
    reason_hidden_dim: int = 3584
    dit_context_dim: int = 1024
    num_waypoints: int = 4
    waypoint_dim: int = 3
    condition_frame_positions: Tuple[int, ...] = (0, 1, 2, 3, 4)
    waypoint_head_hidden_dim: int = 1024
    waypoint_loss_weight: float = 1.0
    waypoint_swap_loss_weight: float = 0.05
    waypoint_swap_margin: float = 0.02
    waypoint_swap_min_distance: float = 0.5
    waypoint_only_dropout_prob: float = 0.1
    waypoint_grad_log_interval: int = 0


class VLNWaypointVideo2WorldModelRectifiedFlow(Video2WorldModelRectifiedFlow):
    def __init__(self, config: VLNWaypointVideo2WorldModelRectifiedFlowConfig):
        super().__init__(config)
        assert config.trajectory_reason_encoder_config is not None
        self.trajectory_reason_encoder = TrajectoryReason1Encoder(
            config=config.trajectory_reason_encoder_config,
            num_traj_tokens=config.num_waypoints,
            hidden_size=config.reason_hidden_dim,
            condition_frame_positions=config.condition_frame_positions,
            device="cuda",
        )
        self.reason2_to_dit_proj = nn.Sequential(
            nn.LayerNorm(config.reason_hidden_dim),
            nn.Linear(config.reason_hidden_dim, config.dit_context_dim),
        )
        self.waypoint_head = nn.Sequential(
            nn.LayerNorm(config.reason_hidden_dim),
            nn.Linear(config.reason_hidden_dim, config.waypoint_head_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.waypoint_head_hidden_dim, config.waypoint_dim),
        )
        self._optimizer_module = None


    def params_and_buffers_to_ignore_for_sync(self) -> set[str]:
        ignored = set()
        for name, _ in self.trajectory_reason_encoder.model.named_parameters():
            ignored.add(f"trajectory_reason_encoder.model.{name}")
        for name, _ in self.trajectory_reason_encoder.model.named_buffers():
            ignored.add(f"trajectory_reason_encoder.model.{name}")
        return ignored

    def init_optimizer_scheduler(
        self, optimizer_config: LazyDict, scheduler_config: LazyDict
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

        tensor_params = []
        dtensor_params = []
        for param in self._optimizer_module.parameters():
            if not param.requires_grad:
                continue
            if isinstance(param, DTensor):
                dtensor_params.append(param)
            else:
                tensor_params.append(param)

        param_groups = []
        if dtensor_params:
            param_groups.append({"params": dtensor_params})
        if tensor_params:
            param_groups.append({"params": tensor_params})
        optimizer = lazy_instantiate(optimizer_config, params=param_groups)
        scheduler = get_base_scheduler(optimizer, self, scheduler_config)
        return optimizer, scheduler

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        super().on_train_start(memory_format=memory_format)
        self.trajectory_reason_encoder.to(device="cuda")
        self.trajectory_reason_encoder.model.eval()
        self.reason2_to_dit_proj.to(**self.tensor_kwargs)
        self.waypoint_head.to(**self.tensor_kwargs)

    def _extra_trainable_parameters(self):
        yield self.trajectory_reason_encoder.traj_queries
        yield from self.reason2_to_dit_proj.parameters()
        yield from self.waypoint_head.parameters()

    def on_after_backward(self, iteration: int = 0) -> None:
        super().on_after_backward(iteration=iteration)
        if self.config.fsdp_shard_size > 1 and parallel_state.is_initialized():
            group = parallel_state.get_data_parallel_group(with_context_parallel=True)
            world_size = dist.get_world_size(group)
            if world_size > 1:
                for param in self._extra_trainable_parameters():
                    if param.requires_grad and param.grad is not None:
                        dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
                        param.grad.div_(world_size)

        interval = self.config.waypoint_grad_log_interval
        if interval > 0 and iteration % interval == 0:
            try:
                from torch.distributed._tensor import DTensor
            except ImportError:
                DTensor = ()
            for name, param in self.net.named_parameters():
                if "waypoint" not in name:
                    continue
                grad = param.grad
                if isinstance(grad, DTensor):
                    grad = grad.to_local()
                grad_norm = None if grad is None else grad.detach().float().norm().item()
                log.info(f"Waypoint gradient iteration={iteration} name={name} norm={grad_norm}")

    def _apply_waypoint_only_dropout(
        self,
        context: torch.Tensor,
        text_mask: torch.Tensor,
    ) -> torch.Tensor:
        dropout_prob = self.config.waypoint_only_dropout_prob
        if not self.training or dropout_prob <= 0:
            return context
        drop_sample = torch.rand(context.shape[0], device=context.device) < dropout_prob
        if not drop_sample.any():
            return context
        drop_mask = drop_sample[:, None] & text_mask.to(device=context.device)
        context = context.clone()
        context[drop_mask] = 0
        return context

    def _condition_with_waypoints(
        self,
        condition: Text2WorldCondition,
        waypoints_norm: torch.Tensor,
    ) -> Text2WorldCondition:
        kwargs = condition.to_dict(skip_underscore=False)
        kwargs["crossattn_emb_preprojected"] = True
        kwargs["waypoints_norm"] = waypoints_norm
        return type(condition)(**kwargs)

    def _predict_reason_context_and_waypoints(
        self,
        data_batch: dict[str, Any],
    ) -> tuple[TrajectoryReason1Output, torch.Tensor, torch.Tensor]:
        reason_output = self.trajectory_reason_encoder(data_batch, self.input_caption_key)
        reason_context = self.reason2_to_dit_proj(reason_output.context_tokens.to(**self.tensor_kwargs))
        pred_waypoints_norm = self.waypoint_head(reason_output.traj_latents.to(**self.tensor_kwargs))
        return reason_output, reason_context, pred_waypoints_norm

    @torch.inference_mode()
    def predict_waypoints_from_batch(
        self,
        data_batch: dict[str, Any],
        denormalize: bool = True,
    ) -> torch.Tensor:
        """Predict ego-frame [dx, dy, dyaw] waypoints without running video sampling."""
        _, _, pred_waypoints_norm = self._predict_reason_context_and_waypoints(data_batch)
        if not denormalize:
            return pred_waypoints_norm
        scales = pred_waypoints_norm.new_tensor([1.0, 1.0, np.pi])
        return pred_waypoints_norm * scales

    @torch.inference_mode()
    def get_velocity_fn_from_batch(
        self,
        data_batch: dict[str, Any],
        guidance: float = 1.5,
        is_negative_prompt: bool = False,
    ):
        """Build the VLN sampling conditions used by the waypoint training path."""
        del is_negative_prompt
        reason_output, reason_context, pred_waypoints_norm = self._predict_reason_context_and_waypoints(data_batch)

        data_batch["t5_text_embeddings"] = reason_context
        data_batch["t5_text_mask"] = reason_output.context_mask.to(device=reason_context.device)
        _, x0, condition = self.get_data_and_condition(data_batch)
        condition = self._condition_with_waypoints(condition, pred_waypoints_norm)

        # Match waypoint_only_dropout: retain visual tokens and waypoints while
        # removing instruction tokens for the CFG-unconditional branch.
        uncond_context = reason_context.clone()
        uncond_context[reason_output.text_mask.to(device=uncond_context.device)] = 0
        uncond_kwargs = condition.to_dict(skip_underscore=False)
        uncond_kwargs["crossattn_emb"] = uncond_context
        uncondition = type(condition)(**uncond_kwargs)

        _, condition, _, _ = self.broadcast_split_for_model_parallelsim(x0, condition, None, None)
        _, uncondition, _, _ = self.broadcast_split_for_model_parallelsim(x0, uncondition, None, None)
        self.last_pred_waypoints_norm = pred_waypoints_norm.detach()
        waypoint_scales = pred_waypoints_norm.new_tensor([1.0, 1.0, np.pi])
        log.info(
            f"Predicted VLN waypoints [dx_m, dy_m, dyaw_rad]: "
            f"{(pred_waypoints_norm * waypoint_scales).detach().float().cpu().tolist()}",
            rank0_only=True,
        )

        if not parallel_state.is_initialized():
            assert not self.net.is_context_parallel_enabled, (
                "parallel_state is not initialized, context parallel should be turned off."
            )

        def velocity_fn(noise: torch.Tensor, noise_x: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            cond_v = self.denoise(noise, noise_x, timestep, condition)
            uncond_v = self.denoise(noise, noise_x, timestep, uncondition)
            return uncond_v + guidance * (cond_v - uncond_v)

        return velocity_fn

    def _waypoint_loss_and_metrics(
        self,
        pred_waypoints_norm: torch.Tensor,
        data_batch: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = data_batch["ego_waypoints_norm"].to(device=pred_waypoints_norm.device, dtype=pred_waypoints_norm.dtype)
        mask = data_batch.get("ego_waypoints_mask", torch.ones_like(target[..., 0]))
        mask = mask.to(device=pred_waypoints_norm.device, dtype=pred_waypoints_norm.dtype)
        denom = mask.sum().clamp_min(1.0)
        smooth_l1 = F.smooth_l1_loss(pred_waypoints_norm, target, reduction="none").mean(dim=-1)
        waypoint_loss = (smooth_l1 * mask).sum() / denom

        diff = pred_waypoints_norm.detach() - target.detach()
        xy_mae = (diff[..., :2].abs().mean(dim=-1) * mask).sum() / denom
        yaw_mae = (diff[..., 2].abs() * np.pi * mask).sum() / denom
        metrics = {
            "waypoint_loss": waypoint_loss.detach(),
            "waypoint_mae_xy": xy_mae,
            "waypoint_mae_yaw": yaw_mae,
        }
        for idx in range(self.config.num_waypoints):
            wp_mask = mask[:, idx].clamp_min(0.0)
            wp_denom = wp_mask.sum().clamp_min(1.0)
            wp_xy = (diff[:, idx, :2].abs().mean(dim=-1) * wp_mask).sum() / wp_denom
            metrics[f"wp{idx + 1}_mae"] = wp_xy
        return waypoint_loss, metrics

    def _swap_waypoint_loss(
        self,
        pred_waypoints_norm: torch.Tensor,
        target_waypoints_norm: torch.Tensor,
        condition: Text2WorldCondition,
        epsilon_B_C_T_H_W: torch.Tensor,
        xt_B_C_T_H_W: torch.Tensor,
        timesteps: torch.Tensor,
        vt_B_C_T_H_W: torch.Tensor,
        per_instance_loss: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.waypoint_swap_loss_weight <= 0 or pred_waypoints_norm.shape[0] < 2:
            return pred_waypoints_norm.new_zeros(())
        swapped_waypoints = pred_waypoints_norm.detach().roll(1, dims=0)
        pair_distance = torch.linalg.vector_norm(
            (target_waypoints_norm[..., :2] - target_waypoints_norm[..., :2].roll(1, dims=0)).flatten(1), dim=-1
        )
        valid = pair_distance > self.config.waypoint_swap_min_distance
        if not valid.any():
            return pred_waypoints_norm.new_zeros(())
        swapped_condition = self._condition_with_waypoints(condition, swapped_waypoints)
        swapped_pred = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=swapped_condition,
        )
        swapped_per_instance_loss = torch.mean(
            (swapped_pred - vt_B_C_T_H_W) ** 2,
            dim=list(range(1, swapped_pred.dim())),
        )
        swap_margin = self.config.waypoint_swap_margin + per_instance_loss.detach() - swapped_per_instance_loss
        return F.relu(swap_margin[valid]).mean()

    def forward(self, data_batch: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        reason_output, reason_context, pred_waypoints_norm = self._predict_reason_context_and_waypoints(data_batch)
        reason_context = self._apply_waypoint_only_dropout(reason_context, reason_output.text_mask)
        data_batch["t5_text_embeddings"] = reason_context
        data_batch["t5_text_mask"] = reason_output.context_mask.to(device=reason_context.device)

        _, x0_B_C_T_H_W, condition = self.get_data_and_condition(data_batch)
        condition = self._condition_with_waypoints(condition, pred_waypoints_norm)

        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), **self.tensor_kwargs_fp32)
        batch_size = x0_B_C_T_H_W.size()[0]
        t_B = self.rectified_flow.sample_train_time(batch_size).to(**self.tensor_kwargs_fp32)
        t_B = rearrange(t_B, "b -> b 1")

        x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B = self.broadcast_split_for_model_parallelsim(
            x0_B_C_T_H_W, condition, epsilon_B_C_T_H_W, t_B
        )
        timesteps = self.rectified_flow.get_discrete_timestamp(t_B, self.tensor_kwargs_fp32)
        sigmas = self.rectified_flow.get_sigmas(timesteps, self.tensor_kwargs_fp32)

        timesteps = rearrange(timesteps, "b -> b 1")
        sigmas = rearrange(sigmas, "b -> b 1")
        xt_B_C_T_H_W, vt_B_C_T_H_W = self.rectified_flow.get_interpolation(epsilon_B_C_T_H_W, x0_B_C_T_H_W, sigmas)

        vt_pred_B_C_T_H_W = self.denoise(
            noise=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=timesteps,
            condition=condition,
        )

        time_weights_B = self.rectified_flow.train_time_weight(timesteps, self.tensor_kwargs_fp32)
        per_instance_loss = torch.mean(
            (vt_pred_B_C_T_H_W - vt_B_C_T_H_W) ** 2, dim=list(range(1, vt_pred_B_C_T_H_W.dim()))
        )
        denoise_loss = torch.mean(time_weights_B * per_instance_loss)

        waypoint_loss, waypoint_metrics = self._waypoint_loss_and_metrics(pred_waypoints_norm, data_batch)
        target_waypoints_norm = data_batch["ego_waypoints_norm"].to(
            device=pred_waypoints_norm.device, dtype=pred_waypoints_norm.dtype
        )
        swap_loss = self._swap_waypoint_loss(
            pred_waypoints_norm=pred_waypoints_norm,
            target_waypoints_norm=target_waypoints_norm,
            condition=condition,
            epsilon_B_C_T_H_W=epsilon_B_C_T_H_W,
            xt_B_C_T_H_W=xt_B_C_T_H_W,
            timesteps=timesteps,
            vt_B_C_T_H_W=vt_B_C_T_H_W,
            per_instance_loss=per_instance_loss,
        )
        loss = (
            denoise_loss
            + self.config.waypoint_loss_weight * waypoint_loss
            + self.config.waypoint_swap_loss_weight * swap_loss
        )

        output_batch = {
            "x0": x0_B_C_T_H_W,
            "xt": xt_B_C_T_H_W,
            "sigma": sigmas,
            "condition": condition,
            "model_pred": vt_pred_B_C_T_H_W,
            "edm_loss": denoise_loss,
            "denoise_loss": denoise_loss.detach(),
            "timesteps": timesteps,
            "per_instance_loss": per_instance_loss,
            "n_cond_frames": condition.num_conditional_frames_B,
            "pred_waypoints_norm": pred_waypoints_norm.detach(),
            "waypoint_swap_loss": swap_loss.detach(),
        }
        output_batch.update(waypoint_metrics)
        return output_batch, loss

    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["trajectory_reason_encoder.traj_queries"] = self.trajectory_reason_encoder.traj_queries.detach().clone()
        for key, value in self.reason2_to_dit_proj.state_dict().items():
            state[f"reason2_to_dit_proj.{key}"] = value
        for key, value in self.waypoint_head.state_dict().items():
            state[f"waypoint_head.{key}"] = value
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        state_dict = collections.OrderedDict(state_dict)
        extra_keys = [
            key
            for key in state_dict
            if key.startswith("trajectory_reason_encoder.")
            or key.startswith("reason2_to_dit_proj.")
            or key.startswith("waypoint_head.")
        ]
        extra_state = {key: state_dict.pop(key) for key in extra_keys}
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)

        if "trajectory_reason_encoder.traj_queries" in extra_state:
            self.trajectory_reason_encoder.traj_queries.data.copy_(extra_state["trajectory_reason_encoder.traj_queries"])
        proj_state = {
            key.replace("reason2_to_dit_proj.", ""): value
            for key, value in extra_state.items()
            if key.startswith("reason2_to_dit_proj.")
        }
        if proj_state:
            self.reason2_to_dit_proj.load_state_dict(proj_state, strict=False)
        head_state = {
            key.replace("waypoint_head.", ""): value
            for key, value in extra_state.items()
            if key.startswith("waypoint_head.")
        }
        if head_state:
            self.waypoint_head.load_state_dict(head_state, strict=False)

        if result is None:
            return _IncompatibleKeys(missing_keys=[], unexpected_keys=[])
        return result

    def model_param_stats(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total_learnable_param_num": trainable}

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
    ):
        try:
            from torch.distributed._tensor import DTensor
        except ImportError:
            DTensor = ()

        module = self._optimizer_module if self._optimizer_module is not None else self
        tensor_params = []
        dtensor_params = []
        for param in module.parameters():
            if not param.requires_grad or param.grad is None:
                continue
            if isinstance(param, DTensor) or isinstance(param.grad, DTensor):
                dtensor_params.append(param)
            else:
                tensor_params.append(param)

        norms = []
        for params in (dtensor_params, tensor_params):
            if not params:
                continue
            norms.append(
                clip_grad_norm_(
                    params,
                    max_norm,
                    norm_type=norm_type,
                    error_if_nonfinite=error_if_nonfinite,
                    foreach=foreach,
                )
            )

        if not norms:
            return torch.zeros((), **self.tensor_kwargs_fp32)
        if len(norms) == 1:
            return norms[0]

        local_norms = [
            get_local_tensor_if_DTensor(norm).to(device=self.tensor_kwargs["device"], dtype=torch.float32)
            for norm in norms
        ]
        norm_stack = torch.stack(local_norms)
        if norm_type == float("inf"):
            return norm_stack.max()
        return torch.linalg.vector_norm(norm_stack, ord=norm_type)
