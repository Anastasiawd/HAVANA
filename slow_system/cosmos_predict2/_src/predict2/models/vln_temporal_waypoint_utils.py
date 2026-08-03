# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lightweight tensor/list helpers for temporal VLN training and CPU tests."""

from typing import Any, Optional

import torch


def random_cyclic_captions(captions: Any, batch_size: int) -> Optional[list[str]]:
    if batch_size < 2:
        return None
    if isinstance(captions, str):
        captions = [captions] * batch_size
    captions = [str(caption) for caption in captions]
    shift = int(torch.randint(1, batch_size, (1,)).item())
    return [captions[(index + shift) % batch_size] for index in range(batch_size)]


def global_latent_loss_weights(frame_mask: torch.Tensor) -> torch.Tensor:
    if frame_mask.ndim != 2 or frame_mask.shape[1] != 17:
        raise ValueError(f"video_frame_loss_mask must be [B,17], got {tuple(frame_mask.shape)}")
    zeros = frame_mask.new_zeros(frame_mask.shape[0], 2)
    future = torch.stack(
        [
            frame_mask[:, 5:9].mean(dim=1),
            frame_mask[:, 9:13].mean(dim=1),
            frame_mask[:, 13:17].mean(dim=1),
        ],
        dim=1,
    )
    return torch.cat([zeros, future], dim=1)


def masked_mean_or_graph_zero(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return a masked mean without disconnecting empty rank-local batches.

    FSDP ranks can have different local masks. Returning ``values.new_zeros``
    for an empty mask would skip that rank's backward path, so the zero result
    must retain an autograd edge to the corresponding forward computation.
    """

    mask = mask.to(device=values.device, dtype=torch.bool)
    if mask.any():
        return values[mask].mean()
    return values.sum() * 0.0


def mask_projected_context(projected_context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
    """Keep padded Reason1 tokens neutral after a bias-bearing projection."""
    context_mask = context_mask.to(device=projected_context.device, dtype=torch.bool)
    if context_mask.shape != projected_context.shape[:2]:
        raise ValueError(
            f"context_mask shape {tuple(context_mask.shape)} does not match "
            f"projected context {tuple(projected_context.shape[:2])}"
        )
    return projected_context.masked_fill(~context_mask.unsqueeze(-1), 0)
