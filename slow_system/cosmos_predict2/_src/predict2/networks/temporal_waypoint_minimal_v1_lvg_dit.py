# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Temporal waypoint conditioning for the independent 17-frame VLN experiment."""

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
from einops import rearrange
from torch import nn
from torch.distributed._composable.fsdp import fully_shard

from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT


class _TemporalWaypointEmbedder(nn.Module):
    def __init__(
        self,
        num_waypoints: int,
        waypoint_dim: int,
        hidden_dim: int,
        model_channels: int,
    ) -> None:
        super().__init__()
        self.num_waypoints = num_waypoints
        self.step_embedding = nn.Parameter(torch.empty(num_waypoints, hidden_dim))
        self.to_timestep = nn.Sequential(
            nn.Linear(waypoint_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, model_channels),
        )
        self.to_adaln_lora = nn.Sequential(
            nn.Linear(waypoint_dim, hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_dim, 3 * model_channels),
        )

    def reset_parameters(self) -> None:
        nn.init.normal_(self.step_embedding, mean=0.0, std=0.02)
        for branch in (self.to_timestep, self.to_adaln_lora):
            first = branch[0]
            final = branch[-1]
            assert isinstance(first, nn.Linear) and isinstance(final, nn.Linear)
            first.reset_parameters()
            # A small non-zero random output keeps the official backbone close
            # to its initialization while satisfying the independent random
            # initialization contract and giving all layers gradients at step 0.
            nn.init.normal_(final.weight, mean=0.0, std=1e-3)
            if final.bias is not None:
                nn.init.zeros_(final.bias)

    def _forward_branch(self, waypoints: torch.Tensor, branch: nn.Sequential) -> torch.Tensor:
        hidden = branch[0](waypoints)
        hidden = hidden + self.step_embedding.unsqueeze(0).to(device=hidden.device, dtype=hidden.dtype)
        hidden = branch[1](hidden)
        return branch[2](hidden)

    def forward(self, waypoints: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self._forward_branch(waypoints, self.to_timestep),
            self._forward_branch(waypoints, self.to_adaln_lora),
        )


class TemporalWaypointMinimalV1LVGDiT(MinimalV1LVGDiT):
    """Inject three ordered waypoints into latent positions 2/3/4.

    Unlike :class:`WaypointMinimalV1LVGDiT`, this module creates no
    cross-attention waypoint tokens and no all-time per-block modulation.
    The temporal embeddings are added to both timestep and AdaLN-LoRA inputs,
    so every DiT block sees the correctly aligned action step.
    """

    def __init__(
        self,
        *args,
        num_waypoints: int = 3,
        waypoint_dim: int = 3,
        num_condition_latents: int = 2,
        waypoint_hidden_dim: Optional[int] = None,
        **kwargs,
    ) -> None:
        self.num_waypoints = int(num_waypoints)
        self.waypoint_dim = int(waypoint_dim)
        self.num_condition_latents = int(num_condition_latents)
        super().__init__(*args, **kwargs)
        hidden_dim = waypoint_hidden_dim or self.model_channels * 2
        self.temporal_waypoint_embedder = _TemporalWaypointEmbedder(
            num_waypoints=self.num_waypoints,
            waypoint_dim=self.waypoint_dim,
            hidden_dim=hidden_dim,
            model_channels=self.model_channels,
        )
        self.init_temporal_waypoint_weights()

    def init_temporal_waypoint_weights(self) -> None:
        self.temporal_waypoint_embedder.reset_parameters()

    def init_weights(self) -> None:
        super().init_weights()
        if hasattr(self, "temporal_waypoint_embedder"):
            self.init_temporal_waypoint_weights()

    def fully_shard(self, mesh, **fsdp_kwargs):
        super().fully_shard(mesh, **fsdp_kwargs)
        fully_shard(
            self.temporal_waypoint_embedder,
            mesh=mesh,
            reshard_after_forward=False,
            **fsdp_kwargs,
        )

    def _temporal_waypoint_embeddings(
        self,
        waypoints_norm: Optional[torch.Tensor],
        waypoint_condition_mask: Optional[torch.Tensor],
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        total_latents = self.num_condition_latents + self.num_waypoints
        if waypoints_norm is None:
            return (
                torch.zeros(batch_size, total_latents, self.model_channels, dtype=dtype, device=device),
                torch.zeros(batch_size, total_latents, 3 * self.model_channels, dtype=dtype, device=device),
            )
        if waypoints_norm.shape[-2:] != (self.num_waypoints, self.waypoint_dim):
            raise ValueError(
                f"Expected waypoints_norm [B,{self.num_waypoints},{self.waypoint_dim}], "
                f"got {tuple(waypoints_norm.shape)}"
            )
        param = next(self.temporal_waypoint_embedder.parameters())
        waypoint_input = waypoints_norm.to(device=param.device, dtype=param.dtype)
        waypoint_D, waypoint_3D = self.temporal_waypoint_embedder(waypoint_input)
        if waypoint_condition_mask is not None:
            mask = waypoint_condition_mask.to(device=waypoint_D.device, dtype=waypoint_D.dtype)
            if mask.ndim == 1:
                mask = mask[:, None]
            if mask.ndim != 2 or mask.shape[1] not in (1, self.num_waypoints):
                raise ValueError(
                    f"waypoint_condition_mask must be [B], [B,1], or [B,{self.num_waypoints}], got {tuple(mask.shape)}"
                )
            waypoint_D = waypoint_D * mask.unsqueeze(-1)
            waypoint_3D = waypoint_3D * mask.unsqueeze(-1)
        waypoint_D = waypoint_D.to(device=device, dtype=dtype)
        waypoint_3D = waypoint_3D.to(device=device, dtype=dtype)
        zero_D = waypoint_D.new_zeros(batch_size, self.num_condition_latents, self.model_channels)
        zero_3D = waypoint_3D.new_zeros(batch_size, self.num_condition_latents, 3 * self.model_channels)
        return torch.cat([zero_D, waypoint_D], dim=1), torch.cat([zero_3D, waypoint_3D], dim=1)

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: Optional[torch.Tensor] = None,
        fps: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        data_type: Optional[DataType] = DataType.VIDEO,
        intermediate_feature_ids: Optional[List[int]] = None,
        img_context_emb: Optional[torch.Tensor] = None,
        crossattn_emb_preprojected: bool = False,
        waypoints_norm: Optional[torch.Tensor] = None,
        waypoint_condition_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        # Match the official action-conditioned implementation: the base
        # denoiser passes bookkeeping kwargs that this DiT does not consume.
        del kwargs
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)],
                dim=1,
            )
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)],
                dim=1,
            )
        assert isinstance(data_type, DataType)
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )
        if x_B_T_H_W_D.shape[1] != self.num_condition_latents + self.num_waypoints:
            raise ValueError(
                f"Temporal waypoint experiment requires {self.num_condition_latents + self.num_waypoints} "
                f"latent frames, got {x_B_T_H_W_D.shape[1]}"
            )

        if self.use_crossattn_projection and not crossattn_emb_preprojected:
            if crossattn_emb.shape[-1] == self.crossattn_proj_in_channels:
                crossattn_emb = self.crossattn_proj(crossattn_emb)
            elif crossattn_emb.shape[-1] != self.crossattn_emb_channels:
                raise ValueError(f"Unexpected crossattn embedding dimension {crossattn_emb.shape[-1]}")
        elif crossattn_emb_preprojected and crossattn_emb.shape[-1] != self.crossattn_emb_channels:
            raise ValueError(f"Preprojected context dim {crossattn_emb.shape[-1]} != {self.crossattn_emb_channels}")

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None
            context_input = (crossattn_emb, self.img_context_proj(img_context_emb))
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T * self.timestep_scale)
            waypoint_D, waypoint_3D = self._temporal_waypoint_embeddings(
                waypoints_norm,
                waypoint_condition_mask,
                batch_size=x_B_T_H_W_D.shape[0],
                dtype=t_embedding_B_T_D.dtype,
                device=t_embedding_B_T_D.device,
            )
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D + waypoint_D)
            adaln_lora_B_T_3D = adaln_lora_B_T_3D + waypoint_3D

        self.affline_scale_log_info = {"t_embedding_B_T_D": t_embedding_B_T_D.detach()}
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb
        if extra_pos_emb is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb.shape

        intermediate_features_outputs = []
        for index, block in enumerate(self.blocks):
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb,
            )
            if intermediate_feature_ids and index in intermediate_feature_ids:
                intermediate_features_outputs.append(rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d"))

        output = self.unpatchify(
            self.final_layer(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
            )
        )
        if intermediate_feature_ids:
            return output, intermediate_features_outputs
        return output
