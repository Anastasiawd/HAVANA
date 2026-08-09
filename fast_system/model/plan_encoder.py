"""Token-preserving Reason1 plan encoder used by the fast navigation policy."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import nn

from fast_system.model.adapter import RouteQFormer


PLAN_SCHEMA_VERSION = "havana.reason1-plan.v2"
LEGACY_PLAN_SCHEMA_VERSION = "havana.reason1-plan.v1"


class PlanEncoder(nn.Module):
    """Encode Reason1 outputs as a sequence instead of a pooled plan vector.

    The variable-length Cosmos conditioning sequence is distilled into fixed
    route tokens.  The three Reason1 trajectory latent tokens remain distinct
    and receive both anchor-index and signed time-to-anchor embeddings.  When
    ``plan_offsets`` is ``[B,O]`` the expensive route encoding is computed once
    per slow plan and then expanded to the O fast-policy observation times.
    """

    def __init__(
        self,
        context_dim: int = 1024,
        trajectory_dim: int = 3584,
        hidden_dim: int = 384,
        num_route_queries: int = 16,
        num_trajectory_tokens: int = 3,
        trajectory_anchor_steps: Sequence[int] = (4, 8, 12),
        time_normalizer: float = 12.0,
        num_layers: int = 3,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        anchors = tuple(int(value) for value in trajectory_anchor_steps)
        if num_trajectory_tokens != 3 or len(anchors) != num_trajectory_tokens:
            raise ValueError("The plan v2 contract requires three trajectory tokens and three anchors.")
        if time_normalizer <= 0:
            raise ValueError("time_normalizer must be positive.")

        self.context_dim = int(context_dim)
        self.trajectory_dim = int(trajectory_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_route_queries = int(num_route_queries)
        self.num_trajectory_tokens = int(num_trajectory_tokens)
        self.num_plan_tokens = self.num_route_queries + self.num_trajectory_tokens
        self.time_normalizer = float(time_normalizer)

        self.qformer = RouteQFormer(
            input_dim=self.context_dim,
            hidden_dim=self.hidden_dim,
            num_queries=self.num_route_queries,
            num_layers=int(num_layers),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
        )
        self.trajectory_proj = nn.Sequential(
            nn.LayerNorm(self.trajectory_dim),
            nn.Linear(self.trajectory_dim, self.hidden_dim),
        )
        self.trajectory_index_embed = nn.Embedding(self.num_trajectory_tokens, self.hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.output_norm = nn.LayerNorm(self.hidden_dim)
        self.register_buffer(
            "trajectory_anchor_steps",
            torch.tensor(anchors, dtype=torch.float32),
            persistent=True,
        )
        nn.init.trunc_normal_(self.trajectory_index_embed.weight, std=0.02)

    @staticmethod
    def _batched(value: torch.Tensor, rank: int, name: str) -> torch.Tensor:
        if value.ndim == rank - 1:
            value = value.unsqueeze(0)
        if value.ndim != rank:
            raise ValueError(f"{name} must have rank {rank - 1} or {rank}, got {tuple(value.shape)}.")
        return value

    def forward(
        self,
        reason_context: torch.Tensor,
        context_mask: torch.Tensor,
        traj_latents: torch.Tensor,
        plan_offsets: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reason_context = self._batched(reason_context, 3, "reason_context")
        context_mask = self._batched(context_mask, 2, "context_mask")
        traj_latents = self._batched(traj_latents, 3, "traj_latents")
        batch_size = int(reason_context.shape[0])

        if reason_context.shape[-1] != self.context_dim:
            raise ValueError(f"Expected reason_context dim {self.context_dim}, got {reason_context.shape[-1]}.")
        if context_mask.shape != reason_context.shape[:2]:
            raise ValueError("context_mask must match reason_context's batch and sequence dimensions.")
        if traj_latents.shape != (batch_size, self.num_trajectory_tokens, self.trajectory_dim):
            raise ValueError(
                f"traj_latents must be {(batch_size, self.num_trajectory_tokens, self.trajectory_dim)}, "
                f"got {tuple(traj_latents.shape)}."
            )
        context_mask = context_mask.to(device=reason_context.device, dtype=torch.bool)
        invalid = ~context_mask.any(dim=1)
        if invalid.any():
            indices = invalid.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(f"Plan context must contain at least one valid token; invalid samples={indices}.")

        parameter = self.trajectory_proj[1].weight
        device, dtype = parameter.device, parameter.dtype
        reason_context = reason_context.to(device=device, dtype=dtype)
        context_mask = context_mask.to(device=device)
        traj_latents = traj_latents.to(device=device, dtype=dtype)
        route_tokens = self.qformer(reason_context, token_mask=context_mask)

        if plan_offsets is None:
            offsets = torch.zeros((batch_size, 1), device=device, dtype=dtype)
            squeeze_offsets = True
        else:
            offsets = torch.as_tensor(plan_offsets, device=device, dtype=dtype)
            if offsets.ndim == 1:
                offsets = offsets.unsqueeze(1)
                squeeze_offsets = True
            elif offsets.ndim == 2:
                squeeze_offsets = False
            else:
                raise ValueError(f"plan_offsets must be [B] or [B,O], got {tuple(offsets.shape)}.")
            if offsets.shape[0] != batch_size:
                raise ValueError(f"plan_offsets batch {offsets.shape[0]} does not match {batch_size}.")

        num_offsets = int(offsets.shape[1])
        route_tokens = route_tokens[:, None].expand(-1, num_offsets, -1, -1)
        trajectory_tokens = self.trajectory_proj(traj_latents)
        token_ids = torch.arange(self.num_trajectory_tokens, device=device)
        trajectory_tokens = trajectory_tokens + self.trajectory_index_embed(token_ids).unsqueeze(0)
        trajectory_tokens = trajectory_tokens[:, None].expand(-1, num_offsets, -1, -1)

        relative_steps = self.trajectory_anchor_steps.to(device=device, dtype=dtype).view(1, 1, -1)
        relative_steps = (relative_steps - offsets.unsqueeze(-1)) / self.time_normalizer
        trajectory_tokens = trajectory_tokens + self.time_embed(relative_steps.unsqueeze(-1))
        plan_tokens = self.output_norm(torch.cat((route_tokens, trajectory_tokens), dim=2))
        if squeeze_offsets:
            return plan_tokens[:, 0]
        return plan_tokens


def build_plan_encoder(cfg: Mapping[str, Any] | None = None) -> PlanEncoder:
    cfg = {} if cfg is None else cfg
    return PlanEncoder(
        context_dim=int(cfg.get("context_dim", 1024)),
        trajectory_dim=int(cfg.get("trajectory_dim", 3584)),
        hidden_dim=int(cfg.get("hidden_dim", 384)),
        num_route_queries=int(cfg.get("num_route_queries", 16)),
        num_trajectory_tokens=int(cfg.get("num_trajectory_tokens", 3)),
        trajectory_anchor_steps=cfg.get("trajectory_anchor_steps", (4, 8, 12)),
        time_normalizer=float(cfg.get("time_normalizer", 12.0)),
        num_layers=int(cfg.get("num_layers", 3)),
        num_heads=int(cfg.get("num_heads", 6)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.1)),
    )
