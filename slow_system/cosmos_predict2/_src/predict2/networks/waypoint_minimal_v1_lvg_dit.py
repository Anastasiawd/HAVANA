# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional, Tuple

import torch
import torch.amp as amp
from einops import rearrange
from torch import nn
from torch.distributed._composable.fsdp import fully_shard

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.conditioner import DataType
from cosmos_predict2._src.predict2.networks.minimal_v1_lvg_dit import MinimalV1LVGDiT


class WaypointMinimalV1LVGDiT(MinimalV1LVGDiT):
    """Minimal V1 LVG DiT with waypoint tokens and zero-init per-block cross-attn modulation."""

    def __init__(
        self,
        *args,
        num_waypoints: int = 4,
        waypoint_dim: int = 3,
        waypoint_hidden_dim: Optional[int] = None,
        **kwargs,
    ):
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim
        super().__init__(*args, **kwargs)

        waypoint_hidden_dim = waypoint_hidden_dim or self.crossattn_emb_channels
        self.waypoint_token_proj = nn.Sequential(
            nn.Linear(waypoint_dim, waypoint_hidden_dim),
            nn.GELU(),
            nn.Linear(waypoint_hidden_dim, self.crossattn_emb_channels),
        )
        self.block_waypoint_mlp = nn.Sequential(
            nn.Linear(num_waypoints * waypoint_dim, self.model_channels),
            nn.SiLU(),
            nn.Linear(self.model_channels, self.num_blocks * 3 * self.model_channels),
        )
        self.init_waypoint_weights()

    def init_waypoint_weights(self) -> None:
        """Initialize waypoint layers after meta-device materialization."""
        for module in self.waypoint_token_proj:
            if isinstance(module, nn.Linear):
                module.reset_parameters()

        first_linear = self.block_waypoint_mlp[0]
        final_linear = self.block_waypoint_mlp[-1]
        assert isinstance(first_linear, nn.Linear)
        assert isinstance(final_linear, nn.Linear)

        # W0 must be non-zero so the zero-initialized W2 receives a gradient on
        # the first optimization step. W0 begins receiving gradients after W2
        # has moved away from zero.
        first_linear.reset_parameters()
        if first_linear.bias is not None:
            nn.init.zeros_(first_linear.bias)
        nn.init.zeros_(final_linear.weight)
        if final_linear.bias is not None:
            nn.init.zeros_(final_linear.bias)

    def init_weights(self) -> None:
        """Initialize base and waypoint weights after meta materialization.

        MiniTrainDIT.__init__ invokes this before the waypoint modules have
        been attached. The model builder invokes it again after materializing
        the meta-device model, when the waypoint modules must be initialized.
        """
        super().init_weights()
        if hasattr(self, "waypoint_token_proj") and hasattr(self, "block_waypoint_mlp"):
            self.init_waypoint_weights()

    def fully_shard(self, mesh, **fsdp_kwargs):
        super().fully_shard(mesh, **fsdp_kwargs)
        fully_shard(self.waypoint_token_proj, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)
        fully_shard(self.block_waypoint_mlp, mesh=mesh, reshard_after_forward=False, **fsdp_kwargs)

    @staticmethod
    def _module_dtype_device(module: nn.Module) -> tuple[torch.dtype, torch.device]:
        param = next(iter(module.parameters()), None)
        if param is None:
            return torch.float32, torch.device("cpu")
        return param.dtype, param.device

    def _prepend_waypoint_tokens(
        self,
        crossattn_emb: torch.Tensor,
        waypoints_norm: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if waypoints_norm is None:
            return crossattn_emb, None
        if crossattn_emb.shape[-1] != self.crossattn_emb_channels:
            raise ValueError(
                f"Waypoint token injection expects crossattn_emb dim {self.crossattn_emb_channels}, "
                f"got {crossattn_emb.shape[-1]}. Use crossattn_emb_preprojected=True or enable crossattn_proj."
            )
        if waypoints_norm.shape[-2:] != (self.num_waypoints, self.waypoint_dim):
            raise ValueError(
                f"Expected waypoints_norm shape [B, {self.num_waypoints}, {self.waypoint_dim}], "
                f"got {tuple(waypoints_norm.shape)}"
            )

        proj_dtype, proj_device = self._module_dtype_device(self.waypoint_token_proj)
        waypoint_proj_input = waypoints_norm.to(device=proj_device, dtype=proj_dtype)
        waypoint_tokens = self.waypoint_token_proj(waypoint_proj_input)
        waypoint_tokens = waypoint_tokens.to(device=crossattn_emb.device, dtype=crossattn_emb.dtype)
        crossattn_emb = torch.cat([waypoint_tokens, crossattn_emb], dim=1)
        return crossattn_emb, waypoints_norm.to(device=crossattn_emb.device)

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
        **kwargs,
    ) -> torch.Tensor | List[torch.Tensor] | Tuple[torch.Tensor, List[torch.Tensor]]:
        if kwargs:
            log.debug(f"Ignoring unused WaypointMinimalV1LVGDiT kwargs: {sorted(kwargs.keys())}")

        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat([x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1)
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device)],
                dim=1,
            )

        assert isinstance(data_type, DataType), (
            f"Expected DataType, got {type(data_type)}. We need discuss this flag later."
        )
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        if self.use_crossattn_projection and not crossattn_emb_preprojected:
            if crossattn_emb.shape[-1] == self.crossattn_proj_in_channels:
                crossattn_emb = self.crossattn_proj(crossattn_emb)
            elif crossattn_emb.shape[-1] != self.crossattn_emb_channels:
                raise ValueError(
                    f"crossattn_emb dim {crossattn_emb.shape[-1]} does not match either "
                    f"crossattn_proj_in_channels={self.crossattn_proj_in_channels} or "
                    f"crossattn_emb_channels={self.crossattn_emb_channels}"
                )
        elif crossattn_emb_preprojected:
            assert crossattn_emb.shape[-1] == self.crossattn_emb_channels, (
                f"preprojected crossattn_emb dim {crossattn_emb.shape[-1]} "
                f"!= expected {self.crossattn_emb_channels}"
            )

        crossattn_emb, projected_waypoints_norm = self._prepend_waypoint_tokens(crossattn_emb, waypoints_norm)

        if img_context_emb is not None:
            assert self.extra_image_context_dim is not None, "extra_image_context_dim must be set if img_context_emb is provided"
            img_context_emb = self.img_context_proj(img_context_emb)
            context_input = (crossattn_emb, img_context_emb)
        else:
            context_input = crossattn_emb

        with amp.autocast("cuda", enabled=self.use_wan_fp32_strategy, dtype=torch.float32):
            if timesteps_B_T.ndim == 1:
                timesteps_B_T = timesteps_B_T.unsqueeze(1)
            t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T * self.timestep_scale)
            t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        waypoint_mod_B_N_3D = None
        if projected_waypoints_norm is not None:
            mlp_dtype, mlp_device = self._module_dtype_device(self.block_waypoint_mlp)
            waypoint_flat = projected_waypoints_norm.to(device=mlp_device, dtype=mlp_dtype).flatten(1)
            waypoint_mod_B_N_3D = self.block_waypoint_mlp(waypoint_flat)
            waypoint_mod_B_N_3D = waypoint_mod_B_N_3D.view(-1, self.num_blocks, 3 * self.model_channels)

        affline_scale_log_info = {}
        affline_scale_log_info["t_embedding_B_T_D"] = t_embedding_B_T_D.detach()
        self.affline_scale_log_info = affline_scale_log_info
        self.affline_emb = t_embedding_B_T_D
        self.crossattn_emb = crossattn_emb

        if extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D is not None:
            assert x_B_T_H_W_D.shape == extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape, (
                f"{x_B_T_H_W_D.shape} != {extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D.shape}"
            )

        intermediate_features_outputs = []
        for i, block in enumerate(self.blocks):
            waypoint_mod_B_3D = None if waypoint_mod_B_N_3D is None else waypoint_mod_B_N_3D[:, i]
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                context_input,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                waypoint_crossattn_mod_B_3D=waypoint_mod_B_3D,
                extra_per_block_pos_emb=extra_pos_emb_B_T_H_W_D_or_T_H_W_B_D,
            )
            if intermediate_feature_ids and i in intermediate_feature_ids:
                x_reshaped_for_disc = rearrange(x_B_T_H_W_D, "b tp hp wp d -> b (tp hp wp) d")
                intermediate_features_outputs.append(x_reshaped_for_disc)

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        x_B_C_Tt_Hp_Wp = self.unpatchify(x_B_T_H_W_O)
        if intermediate_feature_ids:
            if len(intermediate_features_outputs) != len(intermediate_feature_ids):
                log.warning(
                    f"Collected {len(intermediate_features_outputs)} intermediate features, "
                    f"but expected {len(intermediate_feature_ids)}. "
                    f"Requested IDs: {intermediate_feature_ids}"
                )
            return x_B_C_Tt_Hp_Wp, intermediate_features_outputs

        return x_B_C_Tt_Hp_Wp
