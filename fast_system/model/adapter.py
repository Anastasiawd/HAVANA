"""Route Q-Former adapter for frozen V-JEPA tokens.

The adapter follows the BLIP-2 Q-Former idea: a small set of learnable route
queries cross-attends to frozen visual tokens and emits compact route tokens.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch import nn


DEFAULT_VJEPA2_ROOT = Path("/home/wangdzh/ffnav/vjepa2")
DEFAULT_VJEPA2_CHECKPOINT = Path(
    "/home/wangdzh/ffnav/havana/RouteQueryAdapter/data_preparation/model_weights/vjepa2_1_vitb_dist_vitG_384.pt"
)


VJEPA2_1_ARCH_MAP = {
    "vjepa2_1_vit_base_384": "vit_base",
    "vjepa2_1_vit_large_384": "vit_large",
    "vjepa2_1_vit_giant_384": "vit_giant_xformers",
    "vjepa2_1_vit_gigantic_384": "vit_gigantic_xformers",
}


def _torch_load(path: Path | str, map_location: str | torch.device = "cpu") -> object:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _clean_vjepa_state_dict(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state_dict.items()
    }


class FrozenVJEPA2Encoder(nn.Module):
    """Frozen V-JEPA2 encoder used online inside the route adapter.

    Official V-JEPA2 PyTorch inference feeds normalized video tensors as
    [B, C, T, H, W]. This wrapper accepts common raw video layouts, performs
    optional resize and ImageNet normalization, runs only the frozen encoder,
    and returns patch tokens shaped [B, T', N, D] by default.
    """

    def __init__(
        self,
        repo: str | Path = DEFAULT_VJEPA2_ROOT,
        checkpoint: str | Path = DEFAULT_VJEPA2_CHECKPOINT,
        model_name: str = "vjepa2_1_vit_base_384",
        checkpoint_key: str = "ema_encoder",
        image_size: int = 384,
        num_frames: int = 16,
        patch_size: int = 16,
        tubelet_size: int = 2,
        resize: bool = True,
        normalize: bool = True,
        input_range: str = "auto",
        freeze: bool = True,
        strict: bool = False,
        output_3d: bool = True,
    ) -> None:
        super().__init__()
        if image_size <= 0 or num_frames <= 0:
            raise ValueError("image_size and num_frames must be positive.")
        if input_range not in {"auto", "0_1", "0_255"}:
            raise ValueError("input_range must be one of auto, 0_1, or 0_255.")

        self.repo = Path(repo).expanduser()
        self.checkpoint = Path(checkpoint).expanduser()
        self.model_name = model_name
        self.checkpoint_key = checkpoint_key
        self.image_size = int(image_size)
        self.num_frames = int(num_frames)
        self.resize = bool(resize)
        self.normalize = bool(normalize)
        self.input_range = input_range
        self.output_3d = bool(output_3d)

        self.encoder = self._build_encoder(
            model_name=model_name,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            strict=strict,
        )
        self.encoder.eval()
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.embed_dim = int(getattr(self.encoder, "embed_dim", 0))
        self.patch_size = int(getattr(self.encoder, "patch_size", patch_size))
        self.tubelet_size = int(getattr(self.encoder, "tubelet_size", tubelet_size))

        mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, 3, 1, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, 3, 1, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _build_encoder(
        self,
        model_name: str,
        patch_size: int,
        tubelet_size: int,
        strict: bool,
    ) -> nn.Module:
        if not self.repo.exists():
            raise FileNotFoundError(f"V-JEPA2 repo not found: {self.repo}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"V-JEPA2 checkpoint not found: {self.checkpoint}")
        if model_name not in VJEPA2_1_ARCH_MAP:
            available = ", ".join(sorted(VJEPA2_1_ARCH_MAP))
            raise ValueError(f"Unsupported V-JEPA2 model_name {model_name!r}. Available: {available}")

        repo_str = str(self.repo)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from app.vjepa_2_1.models import vision_transformer as vit_encoder

        arch_name = VJEPA2_1_ARCH_MAP[model_name]
        encoder = vit_encoder.__dict__[arch_name](
            patch_size=patch_size,
            img_size=(self.image_size, self.image_size),
            num_frames=self.num_frames,
            tubelet_size=tubelet_size,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
            img_temporal_dim_size=1,
            interpolate_rope=True,
        )

        checkpoint = _torch_load(self.checkpoint, map_location="cpu")
        if not isinstance(checkpoint, Mapping) or self.checkpoint_key not in checkpoint:
            keys = ", ".join(sorted(checkpoint.keys())) if isinstance(checkpoint, Mapping) else type(checkpoint).__name__
            raise KeyError(f"Checkpoint key {self.checkpoint_key!r} not found. Available: {keys}")
        state_dict = _clean_vjepa_state_dict(checkpoint[self.checkpoint_key])
        encoder.load_state_dict(state_dict, strict=strict)
        return encoder

    @staticmethod
    def _to_bcthw(video: torch.Tensor) -> torch.Tensor:
        if video.ndim == 5:
            if video.shape[1] == 3:
                return video
            if video.shape[2] == 3:
                return video.permute(0, 2, 1, 3, 4).contiguous()
            if video.shape[-1] == 3:
                return video.permute(0, 4, 1, 2, 3).contiguous()
        elif video.ndim == 4:
            if video.shape[0] == 3:
                return video.unsqueeze(0)
            if video.shape[1] == 3:
                return video.permute(1, 0, 2, 3).unsqueeze(0).contiguous()
            if video.shape[-1] == 3:
                return video.permute(3, 0, 1, 2).unsqueeze(0).contiguous()
        raise ValueError(
            "video must have shape [B,C,T,H,W], [B,T,C,H,W], [B,T,H,W,C], "
            "[C,T,H,W], [T,C,H,W], or [T,H,W,C]; "
            f"got {tuple(video.shape)}."
        )

    def _preprocess(self, video: torch.Tensor) -> torch.Tensor:
        x = self._to_bcthw(video)
        was_integer = not x.is_floating_point()
        x = x.float()

        if self.input_range == "0_255" or (self.input_range == "auto" and was_integer):
            x = x / 255.0
        elif self.input_range == "auto" and x.numel() > 0 and x.detach().amax() > 2.0:
            x = x / 255.0

        if self.resize and (x.shape[-2] != self.image_size or x.shape[-1] != self.image_size):
            batch_size, channels, frames, height, width = x.shape
            x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * frames, channels, height, width)
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            x = x.reshape(batch_size, frames, channels, self.image_size, self.image_size)
            x = x.permute(0, 2, 1, 3, 4).contiguous()

        if self.normalize:
            x = (x - self.mean.to(device=x.device, dtype=x.dtype)) / self.std.to(device=x.device, dtype=x.dtype)
        return x

    def _reshape_tokens(self, tokens: torch.Tensor, video_bcthw: torch.Tensor) -> torch.Tensor:
        if not self.output_3d or tokens.ndim != 3:
            return tokens

        _, _, frames, height, width = video_bcthw.shape
        t_tokens = max(1, int(frames) // self.tubelet_size)
        h_tokens = int(height) // self.patch_size
        w_tokens = int(width) // self.patch_size
        spatial_tokens = h_tokens * w_tokens
        if spatial_tokens > 0 and tokens.shape[1] == t_tokens * spatial_tokens:
            return tokens.view(tokens.shape[0], t_tokens, spatial_tokens, tokens.shape[-1]).contiguous()
        return tokens

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        video_bcthw = self._preprocess(video)
        self.encoder.eval()
        with torch.no_grad():
            tokens = self.encoder(video_bcthw)
        return self._reshape_tokens(tokens, video_bcthw)


class MLP(nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0) -> None:
        super().__init__()
        mlp_dim = int(hidden_dim * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RouteQFormerLayer(nn.Module):
    """One query-transformer block with query self-attention and visual cross-attention."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.use_self_attention = use_self_attention
        if use_self_attention:
            self.self_norm = nn.LayerNorm(hidden_dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim=hidden_dim, mlp_ratio=mlp_ratio, dropout=dropout)

    def forward(
        self,
        queries: torch.Tensor,
        visual_tokens: torch.Tensor,
        visual_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.use_self_attention:
            q_norm = self.self_norm(queries)
            self_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
            queries = queries + self_out

        q_norm = self.cross_norm(queries)
        cross_out, _ = self.cross_attn(
            q_norm,
            visual_tokens,
            visual_tokens,
            key_padding_mask=visual_key_padding_mask,
            need_weights=False,
        )
        queries = queries + cross_out
        queries = queries + self.mlp(self.mlp_norm(queries))
        return queries


class RouteQFormer(nn.Module):
    """Compress V-JEPA tokens into learnable route tokens.

    Args:
        input_dim: Feature dimension of frozen image/video encoder tokens.
        hidden_dim: Internal query and route-token dimension.
        num_queries: Number of learnable route queries M.
        num_layers: Number of Q-Former blocks.
        num_heads: Attention heads in each block.
        mlp_ratio: Expansion ratio for the feed-forward block.
        dropout: Dropout used in attention and MLP blocks.
        use_self_attention: Whether route queries self-attend before cross-attention.

    Input shapes:
        visual_tokens can be [B, T, N, D], [B, S, D], [T, N, D], or [S, D].
        token_mask is optional, True for valid visual tokens, with shape matching
        [B, T, N] or [B, S].

    Returns:
        Route tokens with shape [B, M, hidden_dim].
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 384,
        num_queries: int = 16,
        num_layers: int = 3,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        if num_queries <= 0:
            raise ValueError("num_queries must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.num_layers = int(num_layers)

        self.visual_proj = nn.Linear(input_dim, hidden_dim)
        self.visual_norm = nn.LayerNorm(hidden_dim)
        self.route_queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        self.query_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            RouteQFormerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                use_self_attention=use_self_attention,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.route_queries, std=0.02)

    @staticmethod
    def _flatten_visual_tokens(
        visual_tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if visual_tokens.ndim == 4:
            batch_size, time_steps, num_tokens, dim = visual_tokens.shape
            visual_tokens = visual_tokens.reshape(batch_size, time_steps * num_tokens, dim)
            if token_mask is not None:
                token_mask = token_mask.reshape(batch_size, time_steps * num_tokens)
        elif visual_tokens.ndim == 3:
            if token_mask is None and visual_tokens.shape[0] <= 32 and visual_tokens.shape[1] >= 32:
                time_steps, num_tokens, dim = visual_tokens.shape
                visual_tokens = visual_tokens.reshape(1, time_steps * num_tokens, dim)
        elif visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
            if token_mask is not None and token_mask.ndim == 1:
                token_mask = token_mask.unsqueeze(0)
        else:
            raise ValueError(
                "visual_tokens must have shape [B,T,N,D], [B,S,D], [T,N,D], or [S,D]; "
                f"got {tuple(visual_tokens.shape)}."
            )

        if token_mask is not None and token_mask.shape != visual_tokens.shape[:2]:
            raise ValueError(
                f"token_mask shape {tuple(token_mask.shape)} does not match token shape "
                f"{tuple(visual_tokens.shape[:2])}."
            )
        return visual_tokens, token_mask

    def forward(
        self,
        visual_tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        visual_tokens, token_mask = self._flatten_visual_tokens(visual_tokens, token_mask)
        if visual_tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Expected token dim {self.input_dim}, got {visual_tokens.shape[-1]}.")

        visual_tokens = self.visual_norm(self.visual_proj(visual_tokens))
        batch_size = visual_tokens.shape[0]
        queries = self.route_queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_dropout(queries)

        key_padding_mask = None
        if token_mask is not None:
            key_padding_mask = ~token_mask.to(dtype=torch.bool, device=visual_tokens.device)

        for layer in self.layers:
            queries = layer(
                queries=queries,
                visual_tokens=visual_tokens,
                visual_key_padding_mask=key_padding_mask,
            )

        return self.output_norm(queries)


class RouteVideoQFormer(nn.Module):
    """Online video adapter: frozen V-JEPA2 encoder followed by Route Q-Former."""

    def __init__(
        self,
        vjepa_encoder: FrozenVJEPA2Encoder,
        qformer: Optional[RouteQFormer] = None,
        hidden_dim: int = 384,
        num_queries: int = 16,
        num_layers: int = 3,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_self_attention: bool = True,
    ) -> None:
        super().__init__()
        self.vjepa = vjepa_encoder
        self.qformer = qformer or RouteQFormer(
            input_dim=self.vjepa.embed_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_self_attention=use_self_attention,
        )

    def forward(
        self,
        video: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        vjepa_tokens = self.vjepa(video)
        route_tokens = self.qformer(vjepa_tokens, token_mask=token_mask)
        if return_features:
            return {
                "route_tokens": route_tokens,
                "vjepa_tokens": vjepa_tokens,
            }
        return route_tokens


def build_route_qformer(cfg: Mapping[str, object]) -> RouteQFormer:
    """Build RouteQFormer from a Hydra/OmegaConf-compatible mapping."""

    return RouteQFormer(
        input_dim=int(cfg.get("input_dim", 384)),
        hidden_dim=int(cfg.get("hidden_dim", 384)),
        num_queries=int(cfg.get("num_queries", 16)),
        num_layers=int(cfg.get("num_layers", 3)),
        num_heads=int(cfg.get("num_heads", 6)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.1)),
        use_self_attention=bool(cfg.get("use_self_attention", True)),
    )


def build_frozen_vjepa2_encoder(cfg: Mapping[str, object]) -> FrozenVJEPA2Encoder:
    """Build FrozenVJEPA2Encoder from a Hydra/OmegaConf-compatible mapping."""

    return FrozenVJEPA2Encoder(
        repo=cfg.get("repo", DEFAULT_VJEPA2_ROOT),
        checkpoint=cfg.get("checkpoint", DEFAULT_VJEPA2_CHECKPOINT),
        model_name=str(cfg.get("model_name", "vjepa2_1_vit_base_384")),
        checkpoint_key=str(cfg.get("checkpoint_key", "ema_encoder")),
        image_size=int(cfg.get("image_size", 384)),
        num_frames=int(cfg.get("num_frames", 16)),
        patch_size=int(cfg.get("patch_size", 16)),
        tubelet_size=int(cfg.get("tubelet_size", 2)),
        resize=bool(cfg.get("resize", True)),
        normalize=bool(cfg.get("normalize", True)),
        input_range=str(cfg.get("input_range", "auto")),
        freeze=bool(cfg.get("freeze", True)),
        strict=bool(cfg.get("strict", False)),
        output_3d=bool(cfg.get("output_3d", True)),
    )


def build_route_video_qformer(
    vjepa_cfg: Mapping[str, object],
    qformer_cfg: Mapping[str, object],
) -> RouteVideoQFormer:
    """Build the online frozen-V-JEPA + Route-Q-Former adapter."""

    vjepa = build_frozen_vjepa2_encoder(vjepa_cfg)
    qformer = RouteQFormer(
        input_dim=int(qformer_cfg.get("input_dim", vjepa.embed_dim)),
        hidden_dim=int(qformer_cfg.get("hidden_dim", 384)),
        num_queries=int(qformer_cfg.get("num_queries", 16)),
        num_layers=int(qformer_cfg.get("num_layers", 3)),
        num_heads=int(qformer_cfg.get("num_heads", 6)),
        mlp_ratio=float(qformer_cfg.get("mlp_ratio", 4.0)),
        dropout=float(qformer_cfg.get("dropout", 0.1)),
        use_self_attention=bool(qformer_cfg.get("use_self_attention", True)),
    )
    return RouteVideoQFormer(vjepa_encoder=vjepa, qformer=qformer)


class TemporalTransformer(nn.Module):
    """Model route-token sequences across clips.

    Q-Former extracts route tokens per clip. This transformer lets those tokens
    communicate across the clip axis so the adapter can encode ordered route
    structure such as "go straight, turn right, then approach the goal".

    Input shapes:
        route_tokens: [K, M, d] or [B, K, M, d]
        clip_mask: Optional bool mask with shape [K] or [B, K], True for valid clips.

    Returns:
        Temporal tokens H with shape [K*M, d] for unbatched input or
        [B, K*M, d] for batched input.
    """

    def __init__(
        self,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_clips: int = 64,
        num_queries: int = 16,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_queries <= 0 or max_clips <= 0:
            raise ValueError("num_queries and max_clips must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.hidden_dim = int(hidden_dim)
        self.max_clips = int(max_clips)
        self.num_queries = int(num_queries)

        self.clip_time_embed = nn.Embedding(max_clips, hidden_dim)
        self.query_index_embed = nn.Embedding(num_queries, hidden_dim)
        self.pos_dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=norm_first,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.clip_time_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.query_index_embed.weight, std=0.02)

    @staticmethod
    def _ensure_batched(route_tokens: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if route_tokens.ndim == 3:
            return route_tokens.unsqueeze(0), True
        if route_tokens.ndim == 4:
            return route_tokens, False
        raise ValueError(
            "route_tokens must have shape [K,M,d] or [B,K,M,d]; "
            f"got {tuple(route_tokens.shape)}."
        )

    @staticmethod
    def _ensure_clip_mask(
        clip_mask: Optional[torch.Tensor],
        batch_size: int,
        num_clips: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if clip_mask is None:
            return None
        if clip_mask.ndim == 1:
            clip_mask = clip_mask.unsqueeze(0)
        if clip_mask.shape != (batch_size, num_clips):
            raise ValueError(
                f"clip_mask must have shape [K] or [B,K], expected {(batch_size, num_clips)}, "
                f"got {tuple(clip_mask.shape)}."
            )
        return clip_mask.to(device=device, dtype=torch.bool)

    def forward(
        self,
        route_tokens: torch.Tensor,
        clip_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        route_tokens, squeeze_batch = self._ensure_batched(route_tokens)
        batch_size, num_clips, num_queries, hidden_dim = route_tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden dim {self.hidden_dim}, got {hidden_dim}.")
        if num_clips > self.max_clips:
            raise ValueError(f"num_clips={num_clips} exceeds max_clips={self.max_clips}.")
        if num_queries > self.num_queries:
            raise ValueError(f"num_queries={num_queries} exceeds configured num_queries={self.num_queries}.")

        device = route_tokens.device
        clip_ids = torch.arange(num_clips, device=device, dtype=torch.long).view(1, num_clips, 1)
        query_ids = torch.arange(num_queries, device=device, dtype=torch.long).view(1, 1, num_queries)
        pos_embed = self.clip_time_embed(clip_ids) + self.query_index_embed(query_ids)

        tokens = route_tokens + pos_embed
        tokens = self.pos_dropout(tokens.reshape(batch_size, num_clips * num_queries, hidden_dim))

        clip_mask = self._ensure_clip_mask(clip_mask, batch_size, num_clips, device)
        key_padding_mask = None
        if clip_mask is not None:
            key_padding_mask = ~clip_mask.repeat_interleave(num_queries, dim=1)

        temporal_tokens = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        temporal_tokens = self.output_norm(temporal_tokens)
        return temporal_tokens.squeeze(0) if squeeze_batch else temporal_tokens


class AttentionPool(nn.Module):
    """Attention-pool temporal route tokens into one global route vector."""

    def __init__(
        self,
        hidden_dim: int = 384,
        num_heads: int = 6,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads}).")

        self.hidden_dim = int(hidden_dim)
        self.pool_query = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pool_query, std=0.02)

    def forward(
        self,
        tokens: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        return_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if tokens.ndim == 2:
            tokens = tokens.unsqueeze(0)
            squeeze_batch = True
        elif tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [S,d] or [B,S,d], got {tuple(tokens.shape)}.")

        batch_size, sequence_len, hidden_dim = tokens.shape
        if hidden_dim != self.hidden_dim:
            raise ValueError(f"Expected hidden dim {self.hidden_dim}, got {hidden_dim}.")

        key_padding_mask = None
        if token_mask is not None:
            if token_mask.ndim == 1:
                token_mask = token_mask.unsqueeze(0)
            if token_mask.shape != (batch_size, sequence_len):
                raise ValueError(
                    f"token_mask must have shape [S] or [B,S], expected {(batch_size, sequence_len)}, "
                    f"got {tuple(token_mask.shape)}."
                )
            key_padding_mask = ~token_mask.to(device=tokens.device, dtype=torch.bool)

        query = self.pool_query.expand(batch_size, -1, -1)
        pooled, weights = self.attn(
            query,
            tokens,
            tokens,
            key_padding_mask=key_padding_mask,
            need_weights=return_weights,
            average_attn_weights=False,
        )
        pooled = self.output_norm(pooled.squeeze(1))
        if squeeze_batch:
            pooled = pooled.squeeze(0)
            if weights is not None:
                weights = weights.squeeze(0)
        if return_weights:
            return pooled, weights
        return pooled


class RouteTemporalEncoder(nn.Module):
    """Convenience wrapper: TemporalTransformer followed by AttentionPool."""

    def __init__(
        self,
        hidden_dim: int = 384,
        num_layers: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        max_clips: int = 64,
        num_queries: int = 16,
        pool_heads: Optional[int] = None,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.temporal = TemporalTransformer(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_clips=max_clips,
            num_queries=num_queries,
            norm_first=norm_first,
        )
        self.pool = AttentionPool(
            hidden_dim=hidden_dim,
            num_heads=pool_heads or num_heads,
            dropout=dropout,
        )

    def forward(
        self,
        route_tokens: torch.Tensor,
        clip_mask: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        temporal_tokens = self.temporal(route_tokens, clip_mask=clip_mask)
        token_mask = None
        if clip_mask is not None:
            if route_tokens.ndim == 3:
                num_queries = route_tokens.shape[1]
            else:
                num_queries = route_tokens.shape[2]
            if clip_mask.ndim == 1:
                token_mask = clip_mask.repeat_interleave(num_queries)
            else:
                token_mask = clip_mask.repeat_interleave(num_queries, dim=1)
        pooled = self.pool(temporal_tokens, token_mask=token_mask)
        if return_tokens:
            return pooled, temporal_tokens
        return pooled


class VIBBottleneck(nn.Module):
    """Variational information bottleneck for global route vectors.

    The bottleneck discourages the expert-video encoder from carrying local
    appearance details and trajectory noise into Stage 2. It maps the global
    route representation h_g into a Gaussian posterior q(z_g | expert video),
    samples z_g with the reparameterization trick, and exposes the KL term
    against a unit Gaussian prior.

    Input shapes:
        h_g: [d] or [B, d]

    Returns:
        A dictionary with z, mu, logvar, kl, and loss.
    """

    def __init__(
        self,
        input_dim: int = 384,
        z_dim: int = 128,
        beta: float = 1e-3,
        min_logvar: float = -10.0,
        max_logvar: float = 10.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or z_dim <= 0:
            raise ValueError("input_dim and z_dim must be positive.")
        if beta < 0:
            raise ValueError("beta must be non-negative.")
        if min_logvar > max_logvar:
            raise ValueError("min_logvar must be <= max_logvar.")

        self.input_dim = int(input_dim)
        self.z_dim = int(z_dim)
        self.beta = float(beta)
        self.min_logvar = float(min_logvar)
        self.max_logvar = float(max_logvar)

        self.mu = nn.Linear(input_dim, z_dim)
        self.logvar = nn.Linear(input_dim, z_dim)

    def encode(self, h_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h_g.shape[-1] != self.input_dim:
            raise ValueError(f"Expected h_g dim {self.input_dim}, got {h_g.shape[-1]}.")
        mu = self.mu(h_g)
        logvar = self.logvar(h_g).clamp(min=self.min_logvar, max=self.max_logvar)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, sample: bool = True) -> torch.Tensor:
        if not sample:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_divergence(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        kl = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
        if reduction == "none":
            return kl
        if reduction == "mean":
            return kl.mean()
        if reduction == "sum":
            return kl.sum()
        raise ValueError(f"Unsupported KL reduction {reduction!r}. Use none, mean, or sum.")

    def forward(
        self,
        h_g: torch.Tensor,
        sample: Optional[bool] = None,
        beta: Optional[float] = None,
        reduction: str = "mean",
        return_dict: bool = True,
    ) -> dict[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        squeeze_batch = False
        if h_g.ndim == 1:
            h_g = h_g.unsqueeze(0)
            squeeze_batch = True
        elif h_g.ndim != 2:
            raise ValueError(f"h_g must have shape [d] or [B,d], got {tuple(h_g.shape)}.")

        if sample is None:
            sample = self.training
        beta_value = self.beta if beta is None else float(beta)

        mu, logvar = self.encode(h_g)
        z = self.reparameterize(mu, logvar, sample=sample)
        kl = self.kl_divergence(mu, logvar, reduction="none")
        kl_loss = beta_value * self.kl_divergence(mu, logvar, reduction=reduction)

        if squeeze_batch:
            z = z.squeeze(0)
            mu = mu.squeeze(0)
            logvar = logvar.squeeze(0)
            kl = kl.squeeze(0)
            if reduction == "none":
                kl_loss = kl_loss.squeeze(0)

        if return_dict:
            return {
                "z": z,
                "mu": mu,
                "logvar": logvar,
                "kl": kl,
                "loss": kl_loss,
            }
        return z, mu, logvar, kl_loss


class Stage1RouteDecoder(nn.Module):
    """Decode z_g into coarse route supervision targets."""

    def __init__(
        self,
        z_dim: int = 128,
        hidden_dim: int = 256,
        num_anchors: int = 8,
        num_turn_classes: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if z_dim <= 0 or hidden_dim <= 0:
            raise ValueError("z_dim and hidden_dim must be positive.")
        if num_anchors <= 0 or num_turn_classes <= 1:
            raise ValueError("num_anchors must be > 0 and num_turn_classes must be > 1.")

        self.num_anchors = int(num_anchors)
        trunk = [
            nn.LayerNorm(z_dim),
            nn.Linear(z_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        ]
        self.trunk = nn.Sequential(*trunk)
        self.goal_head = nn.Linear(hidden_dim, 2)
        self.anchor_head = nn.Linear(hidden_dim, self.num_anchors * 2)
        self.turn_head = nn.Linear(hidden_dim, num_turn_classes)

    def forward(self, z_g: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.trunk(z_g)
        return {
            "goal_pred": self.goal_head(hidden),
            "anchor_pred": self.anchor_head(hidden).view(z_g.shape[0], self.num_anchors, 2),
            "turn_logits": self.turn_head(hidden),
        }


class Stage1RouteModel(nn.Module):
    """Frozen V-JEPA + Route Q-Former + VIB + Stage-1 decoder."""

    def __init__(
        self,
        video_adapter: RouteVideoQFormer,
        pool: Optional[AttentionPool] = None,
        vib: Optional[VIBBottleneck] = None,
        decoder: Optional[Stage1RouteDecoder] = None,
        z_dim: int = 128,
        num_anchors: int = 8,
    ) -> None:
        super().__init__()
        self.video_adapter = video_adapter
        hidden_dim = int(video_adapter.qformer.hidden_dim)
        self.pool = pool or AttentionPool(hidden_dim=hidden_dim)
        self.vib = vib or VIBBottleneck(input_dim=hidden_dim, z_dim=z_dim)
        self.decoder = decoder or Stage1RouteDecoder(z_dim=self.vib.z_dim, num_anchors=num_anchors)

    def forward(
        self,
        video: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
        sample: Optional[bool] = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        features = self.video_adapter(video, token_mask=token_mask, return_features=return_features)
        if isinstance(features, dict):
            route_tokens = features["route_tokens"]
        else:
            route_tokens = features

        h_g = self.pool(route_tokens)
        vib_out = self.vib(h_g, sample=sample, return_dict=True)
        outputs = self.decoder(vib_out["z"])
        outputs.update(
            {
                "z_g": vib_out["z"],
                "mu": vib_out["mu"],
                "logvar": vib_out["logvar"],
                "kl": vib_out["kl"],
                "h_g": h_g,
            }
        )
        if return_features:
            outputs["route_tokens"] = route_tokens
            if isinstance(features, dict) and "vjepa_tokens" in features:
                outputs["vjepa_tokens"] = features["vjepa_tokens"]
        return outputs


def build_temporal_transformer(cfg: Mapping[str, object]) -> TemporalTransformer:
    """Build TemporalTransformer from a Hydra/OmegaConf-compatible mapping."""

    return TemporalTransformer(
        hidden_dim=int(cfg.get("hidden_dim", 384)),
        num_layers=int(cfg.get("num_layers", 4)),
        num_heads=int(cfg.get("num_heads", 6)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_clips=int(cfg.get("max_clips", 64)),
        num_queries=int(cfg.get("num_queries", 16)),
        norm_first=bool(cfg.get("norm_first", True)),
    )


def build_attention_pool(cfg: Mapping[str, object]) -> AttentionPool:
    """Build AttentionPool from a Hydra/OmegaConf-compatible mapping."""

    return AttentionPool(
        hidden_dim=int(cfg.get("hidden_dim", 384)),
        num_heads=int(cfg.get("num_heads", 6)),
        dropout=float(cfg.get("dropout", 0.1)),
    )


def build_route_temporal_encoder(cfg: Mapping[str, object]) -> RouteTemporalEncoder:
    """Build RouteTemporalEncoder from a Hydra/OmegaConf-compatible mapping."""

    pool_heads = cfg.get("pool_heads")
    return RouteTemporalEncoder(
        hidden_dim=int(cfg.get("hidden_dim", 384)),
        num_layers=int(cfg.get("num_layers", 4)),
        num_heads=int(cfg.get("num_heads", 6)),
        mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
        dropout=float(cfg.get("dropout", 0.1)),
        max_clips=int(cfg.get("max_clips", 64)),
        num_queries=int(cfg.get("num_queries", 16)),
        pool_heads=None if pool_heads is None else int(pool_heads),
        norm_first=bool(cfg.get("norm_first", True)),
    )


def build_vib_bottleneck(cfg: Mapping[str, object]) -> VIBBottleneck:
    """Build VIBBottleneck from a Hydra/OmegaConf-compatible mapping."""

    return VIBBottleneck(
        input_dim=int(cfg.get("input_dim", 384)),
        z_dim=int(cfg.get("z_dim", 128)),
        beta=float(cfg.get("beta", 1e-3)),
        min_logvar=float(cfg.get("min_logvar", -10.0)),
        max_logvar=float(cfg.get("max_logvar", 10.0)),
    )


def build_stage1_route_decoder(cfg: Mapping[str, object]) -> Stage1RouteDecoder:
    return Stage1RouteDecoder(
        z_dim=int(cfg.get("z_dim", 128)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_anchors=int(cfg.get("num_anchors", 8)),
        num_turn_classes=int(cfg.get("num_turn_classes", 3)),
        dropout=float(cfg.get("dropout", 0.1)),
    )


def build_stage1_route_model(cfg: Mapping[str, object]) -> Stage1RouteModel:
    """Build the complete Stage-1 supervised route adapter."""

    vjepa_cfg = cfg.get("vjepa2", {})
    qformer_cfg = cfg.get("route_qformer", {})
    pool_cfg = cfg.get("attention_pool", {})
    vib_cfg = cfg.get("vib", {})
    head_cfg = cfg.get("stage1_head", {})

    if not isinstance(vjepa_cfg, Mapping) or not isinstance(qformer_cfg, Mapping):
        raise TypeError("vjepa2 and route_qformer configs must be mappings.")
    if not isinstance(pool_cfg, Mapping) or not isinstance(vib_cfg, Mapping) or not isinstance(head_cfg, Mapping):
        raise TypeError("attention_pool, vib, and stage1_head configs must be mappings.")

    vjepa = build_frozen_vjepa2_encoder(vjepa_cfg)
    qformer = RouteQFormer(
        input_dim=int(qformer_cfg.get("input_dim", vjepa.embed_dim)),
        hidden_dim=int(qformer_cfg.get("hidden_dim", 384)),
        num_queries=int(qformer_cfg.get("num_queries", 16)),
        num_layers=int(qformer_cfg.get("num_layers", 3)),
        num_heads=int(qformer_cfg.get("num_heads", 6)),
        mlp_ratio=float(qformer_cfg.get("mlp_ratio", 4.0)),
        dropout=float(qformer_cfg.get("dropout", 0.1)),
        use_self_attention=bool(qformer_cfg.get("use_self_attention", True)),
    )
    video_adapter = RouteVideoQFormer(vjepa_encoder=vjepa, qformer=qformer)
    pool = AttentionPool(
        hidden_dim=qformer.hidden_dim,
        num_heads=int(pool_cfg.get("num_heads", qformer_cfg.get("num_heads", 6))),
        dropout=float(pool_cfg.get("dropout", qformer_cfg.get("dropout", 0.1))),
    )
    vib = VIBBottleneck(
        input_dim=int(vib_cfg.get("input_dim", qformer.hidden_dim)),
        z_dim=int(vib_cfg.get("z_dim", head_cfg.get("z_dim", 128))),
        beta=float(vib_cfg.get("beta", 1e-3)),
        min_logvar=float(vib_cfg.get("min_logvar", -10.0)),
        max_logvar=float(vib_cfg.get("max_logvar", 10.0)),
    )
    decoder_cfg = dict(head_cfg)
    decoder_cfg.setdefault("z_dim", vib.z_dim)
    decoder = build_stage1_route_decoder(decoder_cfg)
    return Stage1RouteModel(video_adapter=video_adapter, pool=pool, vib=vib, decoder=decoder)
