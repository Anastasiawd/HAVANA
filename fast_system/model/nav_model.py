"""Stage-2 fast navigation model.

This module keeps the FlowNav policy structure and replaces the image-goal
condition with the Stage-1 route latent z_g.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torch import nn


FAST_SYSTEM_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = FAST_SYSTEM_ROOT / "model"
LINGBOT_ROOT = MODEL_ROOT / "lingbot-depth"
for path in (MODEL_ROOT, LINGBOT_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from flownav.models.attention import PositionalEncoding
from flownav.models.nomad import DenseNetwork
from flownav.models.nomad_vint import replace_bn_with_gn


DEFAULT_FLOWNAV_CHECKPOINT = FAST_SYSTEM_ROOT / "model_weights" / "flownav.pth"
DEFAULT_LINGBOT_CHECKPOINT = FAST_SYSTEM_ROOT / "model_weights" / "lingbot-depth-pretrain-vitl-14-v0p5.pt"
DEFAULT_ACTION_STATS = {
    "min": (-2.5, -4.0),
    "max": (5.0, 4.0),
}


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> object:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _as_pair_tensor(values: Sequence[float], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(values, device=device, dtype=dtype).view(1, 1, 2)


def actions_to_deltas(actions: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros_like(actions[:, :1])
    previous = torch.cat((origin, actions[:, :-1]), dim=1)
    return actions - previous


def normalize_deltas(deltas: torch.Tensor, stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS) -> torch.Tensor:
    data_min = _as_pair_tensor(stats["min"], deltas.device, deltas.dtype)
    data_max = _as_pair_tensor(stats["max"], deltas.device, deltas.dtype)
    return 2.0 * (deltas - data_min) / (data_max - data_min) - 1.0


def unnormalize_deltas(ndeltas: torch.Tensor, stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS) -> torch.Tensor:
    data_min = _as_pair_tensor(stats["min"], ndeltas.device, ndeltas.dtype)
    data_max = _as_pair_tensor(stats["max"], ndeltas.device, ndeltas.dtype)
    return 0.5 * (ndeltas + 1.0) * (data_max - data_min) + data_min


def normalized_deltas_to_actions(
    ndeltas: torch.Tensor,
    stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS,
) -> torch.Tensor:
    return torch.cumsum(unnormalize_deltas(ndeltas, stats), dim=1)


def action_reduce(unreduced_loss: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
    while unreduced_loss.dim() > 1:
        unreduced_loss = unreduced_loss.mean(dim=-1)
    action_mask = action_mask.to(device=unreduced_loss.device, dtype=unreduced_loss.dtype).view_as(unreduced_loss)
    return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)


def sample_cfm_location(x0: torch.Tensor, x1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    view_shape = (x1.shape[0],) + (1,) * (x1.ndim - 1)
    xt = (1.0 - t.view(view_shape)) * x0 + t.view(view_shape) * x1
    ut = x1 - x0
    return t, xt, ut


class LingBotDepthFeatureEncoder(nn.Module):
    """Frozen LingBot-Depth backbone exposed as a token feature extractor."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_LINGBOT_CHECKPOINT,
        resolution_level: int = 5,
        num_tokens: int | None = None,
        default_depth: float = 0.0,
        use_fp16: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from mdm.model.v2 import MDMModel

        self.model = MDMModel.from_pretrained(Path(checkpoint).expanduser())
        if hasattr(self.model, "enable_pytorch_native_sdpa"):
            self.model.enable_pytorch_native_sdpa()
        self.resolution_level = int(resolution_level)
        self.num_tokens = None if num_tokens is None else int(num_tokens)
        self.default_depth = float(default_depth)
        self.use_fp16 = bool(use_fp16)
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        encoder = self.model.encoder
        self.out_dim = int(getattr(encoder, "dim_features", 1024))

    def forward(self, image: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}.")
        if depth is None:
            depth = image.new_full((image.shape[0], image.shape[-2], image.shape[-1]), self.default_depth)
        elif depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]

        self.model.eval()
        with torch.no_grad():
            features, _ = self.model.infer_feat(
                image,
                depth_in=depth,
                num_tokens=self.num_tokens,
                resolution_level=self.resolution_level,
                use_fp16=self.use_fp16 and image.device.type == "cuda",
                enable_depth_mask=False,
            )
        return features.flatten(2).transpose(1, 2).contiguous()


class Stage2VisionEncoder(nn.Module):
    """FlowNav-style observation encoder conditioned on z_g instead of goal image."""

    def __init__(
        self,
        context_size: int = 3,
        obs_encoder: str = "efficientnet-b0",
        obs_encoding_size: int = 256,
        z_dim: int = 128,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        depth_encoder: nn.Module | None = None,
        depth_token_dim: int = 1024,
        depth_proj_dim: int = 384,
        depth_pool_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_size = int(context_size)
        self.obs_encoding_size = int(obs_encoding_size)
        self.z_dim = int(z_dim)

        if not obs_encoder.startswith("efficientnet"):
            raise NotImplementedError(f"Unsupported obs_encoder={obs_encoder!r}.")
        self.obs_encoder = EfficientNet.from_name(obs_encoder, in_channels=3)
        self.obs_encoder = replace_bn_with_gn(self.obs_encoder)
        self.num_obs_features = self.obs_encoder._fc.in_features
        self.compress_obs_enc = (
            nn.Linear(self.num_obs_features, self.obs_encoding_size)
            if self.num_obs_features != self.obs_encoding_size
            else nn.Identity()
        )

        self.depth_encoder = depth_encoder
        self.depth_token_proj = nn.Linear(depth_token_dim, depth_proj_dim)
        self.depth_pool_dim = int(depth_pool_dim)
        self.compress_depth_enc = nn.Sequential(
            nn.AdaptiveAvgPool1d(self.depth_pool_dim),
            nn.Flatten(),
            nn.Linear(depth_proj_dim * self.depth_pool_dim, self.obs_encoding_size),
        )

        self.z_goal_projector = nn.Sequential(
            nn.LayerNorm(self.z_dim),
            nn.Linear(self.z_dim, self.obs_encoding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.obs_encoding_size, self.obs_encoding_size),
        )

        mask_size = self.context_size + 3
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=mask_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(encoder_layer, num_layers=mha_num_attention_layers)

        no_mask = torch.zeros((1, mask_size), dtype=torch.bool)
        goal_mask = torch.zeros((1, mask_size), dtype=torch.bool)
        goal_mask[:, -1] = True
        self.register_buffer("all_masks", torch.cat([no_mask, goal_mask], dim=0), persistent=False)
        self.register_buffer(
            "avg_pool_mask",
            torch.cat(
                [
                    1 - no_mask.float(),
                    (1 - goal_mask.float()) * ((self.context_size + 2) / (self.context_size + 1)),
                ],
                dim=0,
            ),
            persistent=False,
        )

    def _to_context(self, obs_img: torch.Tensor) -> torch.Tensor:
        expected_frames = self.context_size + 1
        if obs_img.ndim == 5:
            if obs_img.shape[2] == 3:
                return obs_img
            if obs_img.shape[1] == 3:
                return obs_img.permute(0, 2, 1, 3, 4).contiguous()
        if obs_img.ndim == 4:
            if obs_img.shape[1] == 3 * expected_frames:
                b, _, h, w = obs_img.shape
                return obs_img.view(b, expected_frames, 3, h, w)
            if obs_img.shape[1] == 3:
                return obs_img.unsqueeze(1).expand(-1, expected_frames, -1, -1, -1).contiguous()
        raise ValueError(
            "obs_img must be [B,T,3,H,W], [B,3,T,H,W], [B,3*(context+1),H,W], or [B,3,H,W]; "
            f"got {tuple(obs_img.shape)}."
        )

    def _encode_obs(self, obs_context: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = obs_context.shape
        flat_obs = obs_context.reshape(batch_size * num_frames, channels, height, width)
        obs_encoding = self.obs_encoder.extract_features(flat_obs)
        obs_encoding = self.obs_encoder._avg_pooling(obs_encoding)
        if self.obs_encoder._global_params.include_top:
            obs_encoding = obs_encoding.flatten(start_dim=1)
            obs_encoding = self.obs_encoder._dropout(obs_encoding)
        obs_encoding = self.compress_obs_enc(obs_encoding)
        return obs_encoding.view(batch_size, num_frames, self.obs_encoding_size)

    def _encode_depth(self, image: torch.Tensor, depth: torch.Tensor | None = None) -> torch.Tensor:
        if self.depth_encoder is None:
            return image.new_zeros((image.shape[0], 1, self.obs_encoding_size))
        depth_tokens = self.depth_encoder(image, depth=depth)
        depth_tokens = self.depth_token_proj(depth_tokens).transpose(1, 2).contiguous()
        depth_encoding = self.compress_depth_enc(depth_tokens)
        return depth_encoding.unsqueeze(1)

    def forward(
        self,
        obs_img: torch.Tensor,
        z_goal: torch.Tensor,
        input_goal_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        obs_context = self._to_context(obs_img)
        if z_goal.ndim != 2 or z_goal.shape[-1] != self.z_dim:
            raise ValueError(f"z_goal must have shape [B,{self.z_dim}], got {tuple(z_goal.shape)}.")

        obs_encoding = self._encode_obs(obs_context)
        depth_encoding = self._encode_depth(obs_context[:, -1], depth=depth)
        goal_encoding = self.z_goal_projector(z_goal).unsqueeze(1)
        tokens = torch.cat((obs_encoding, depth_encoding, goal_encoding), dim=1)
        tokens = self.positional_encoding(tokens)

        src_key_padding_mask = None
        no_goal_mask = None
        if input_goal_mask is not None:
            no_goal_mask = input_goal_mask.to(device=tokens.device, dtype=torch.long).clamp(0, 1)
            src_key_padding_mask = torch.index_select(self.all_masks.to(tokens.device), 0, no_goal_mask)

        tokens = self.sa_encoder(tokens, src_key_padding_mask=src_key_padding_mask)
        if no_goal_mask is not None:
            avg_mask = torch.index_select(self.avg_pool_mask.to(tokens.device), 0, no_goal_mask).unsqueeze(-1)
            tokens = tokens * avg_mask
        return tokens.mean(dim=1)


class FastNavModel(nn.Module):
    """FlowNav-compatible wrapper with z_g conditioning."""

    def __init__(self, vision_encoder: Stage2VisionEncoder, noise_pred_net: nn.Module, dist_pred_net: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net

    def encode_condition(
        self,
        obs_img: torch.Tensor,
        z_goal: torch.Tensor,
        input_goal_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.vision_encoder(obs_img, z_goal, input_goal_mask=input_goal_mask, depth=depth)

    def forward(self, func_name: str, **kwargs: Any) -> torch.Tensor:
        if func_name == "vision_encoder":
            z_goal = kwargs.get("z_goal", kwargs.get("goal_latent", kwargs.get("goal_img")))
            if z_goal is None:
                raise KeyError("vision_encoder requires z_goal.")
            return self.encode_condition(
                obs_img=kwargs["obs_img"],
                z_goal=z_goal,
                input_goal_mask=kwargs.get("input_goal_mask"),
                depth=kwargs.get("depth"),
            )
        if func_name == "noise_pred_net":
            return self.noise_pred_net(
                sample=kwargs["sample"],
                timestep=kwargs["timestep"],
                global_cond=kwargs["global_cond"],
            )
        if func_name == "dist_pred_net":
            return self.dist_pred_net(kwargs["obsgoal_cond"].flatten(start_dim=1))
        raise NotImplementedError(f"Unknown func_name={func_name!r}.")

    def cfm_loss(
        self,
        obs_img: torch.Tensor,
        z_goal: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action_stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS,
        input_goal_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.encode_condition(obs_img, z_goal, input_goal_mask=input_goal_mask, depth=depth)
        naction = normalize_deltas(actions_to_deltas(actions), action_stats)
        noise = torch.randn_like(naction)
        t, xt, ut = sample_cfm_location(noise, naction)
        vt = self.noise_pred_net(sample=xt, timestep=t, global_cond=cond)
        if action_mask is None:
            action_mask = torch.ones(actions.shape[0], device=actions.device, dtype=actions.dtype)
        flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)
        return flow_loss, {"flow_loss": flow_loss.detach(), "cond": cond.detach()}

    @torch.no_grad()
    def sample_actions(
        self,
        obs_img: torch.Tensor,
        z_goal: torch.Tensor,
        pred_horizon: int = 16,
        action_dim: int = 2,
        steps: int = 10,
        action_stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS,
        input_goal_mask: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.encode_condition(obs_img, z_goal, input_goal_mask=input_goal_mask, depth=depth)
        x = torch.randn((obs_img.shape[0], pred_horizon, action_dim), device=obs_img.device, dtype=cond.dtype)
        dt = 1.0 / max(int(steps), 1)
        for i in range(max(int(steps), 1)):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + self.noise_pred_net(sample=x, timestep=t, global_cond=cond) * dt
        return normalized_deltas_to_actions(x, action_stats)


def build_lingbot_depth_encoder(cfg: Mapping[str, Any]) -> LingBotDepthFeatureEncoder:
    return LingBotDepthFeatureEncoder(
        checkpoint=cfg.get("checkpoint", DEFAULT_LINGBOT_CHECKPOINT),
        resolution_level=int(cfg.get("resolution_level", 5)),
        num_tokens=cfg.get("num_tokens"),
        default_depth=float(cfg.get("default_depth", 0.0)),
        use_fp16=bool(cfg.get("use_fp16", True)),
        freeze=bool(cfg.get("freeze", True)),
    )


def build_fast_nav_model(cfg: Mapping[str, Any] | None = None) -> FastNavModel:
    cfg = {} if cfg is None else cfg
    vision_cfg = cfg.get("vision_encoder", {})
    depth_cfg = vision_cfg.get("lingbot_depth", {})
    use_depth = bool(depth_cfg.get("enabled", True))
    depth_encoder = build_lingbot_depth_encoder(depth_cfg) if use_depth else None
    depth_token_dim = getattr(depth_encoder, "out_dim", int(depth_cfg.get("out_dim", 1024))) if depth_encoder else int(depth_cfg.get("out_dim", 1024))

    encoding_size = int(cfg.get("encoding_size", vision_cfg.get("obs_encoding_size", 256)))
    context_size = int(cfg.get("context_size", vision_cfg.get("context_size", 3)))
    z_dim = int(cfg.get("z_dim", 128))
    vision_encoder = Stage2VisionEncoder(
        context_size=context_size,
        obs_encoder=str(vision_cfg.get("obs_encoder", "efficientnet-b0")),
        obs_encoding_size=encoding_size,
        z_dim=z_dim,
        mha_num_attention_heads=int(vision_cfg.get("mha_num_attention_heads", 4)),
        mha_num_attention_layers=int(vision_cfg.get("mha_num_attention_layers", 4)),
        mha_ff_dim_factor=int(vision_cfg.get("mha_ff_dim_factor", 4)),
        depth_encoder=depth_encoder,
        depth_token_dim=depth_token_dim,
        depth_proj_dim=int(vision_cfg.get("depth_proj_dim", 384)),
        depth_pool_dim=int(vision_cfg.get("depth_pool_dim", 64)),
        dropout=float(vision_cfg.get("dropout", 0.0)),
    )

    unet_cfg = cfg.get("noise_pred_net", {})
    noise_pred_net = ConditionalUnet1D(
        input_dim=int(cfg.get("action_dim", 2)),
        global_cond_dim=encoding_size,
        diffusion_step_embed_dim=int(unet_cfg.get("diffusion_step_embed_dim", 256)),
        down_dims=tuple(unet_cfg.get("down_dims", (64, 128, 256))),
        kernel_size=int(unet_cfg.get("kernel_size", 3)),
        n_groups=int(unet_cfg.get("n_groups", 8)),
        cond_predict_scale=bool(unet_cfg.get("cond_predict_scale", False)),
    )
    dist_pred_net = DenseNetwork(embedding_dim=encoding_size)
    return FastNavModel(vision_encoder=vision_encoder, noise_pred_net=noise_pred_net, dist_pred_net=dist_pred_net)


def load_flownav_checkpoint(
    model: nn.Module,
    checkpoint: str | Path = DEFAULT_FLOWNAV_CHECKPOINT,
    skip_prefixes: Sequence[str] = ("vision_encoder.goal_encoder", "vision_encoder.depth_encoder"),
) -> dict[str, Any]:
    state = _torch_load(checkpoint, map_location="cpu")
    if isinstance(state, Mapping) and "model" in state:
        state = state["model"]
    if not isinstance(state, Mapping):
        raise TypeError(f"Unsupported checkpoint format at {checkpoint}.")

    current = model.state_dict()
    filtered: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for key, value in state.items():
        if any(key.startswith(prefix) for prefix in skip_prefixes):
            skipped.append(key)
            continue
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            filtered[key] = value
        else:
            skipped.append(key)
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    return {
        "loaded": len(filtered),
        "skipped": len(skipped),
        "missing": list(missing),
        "unexpected": list(unexpected),
    }
