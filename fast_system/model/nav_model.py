"""RGB-D fast navigation model conditioned on a Reason1 plan-token sequence."""

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
DEFAULT_LINGBOT_CHECKPOINT = "robbyant/lingbot-depth-postrain-dc-vitl14"
DEFAULT_ACTION_STATS = {
    "min": (-0.15, -0.025, -0.25),
    "max": (0.15, 0.075, 0.25),
    "clip": True,
}


def _torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> object:
    try:
        return torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location=map_location)


def _as_action_tensor(
    values: Sequence[float], action_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    result = torch.as_tensor(values, device=device, dtype=dtype)
    if result.numel() != action_dim:
        raise ValueError(f"Action statistics contain {result.numel()} values for action_dim={action_dim}.")
    return result.view(1, 1, action_dim)


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle + torch.pi, 2.0 * torch.pi) - torch.pi


def actions_to_deltas(actions: torch.Tensor) -> torch.Tensor:
    origin = torch.zeros_like(actions[:, :1])
    previous = torch.cat((origin, actions[:, :-1]), dim=1)
    deltas = actions - previous
    if actions.shape[-1] >= 3:
        deltas = deltas.clone()
        deltas[..., 2] = wrap_to_pi(deltas[..., 2])
    return deltas


def normalize_deltas(deltas: torch.Tensor, stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS) -> torch.Tensor:
    action_dim = int(deltas.shape[-1])
    data_min = _as_action_tensor(stats["min"], action_dim, deltas.device, deltas.dtype)
    data_max = _as_action_tensor(stats["max"], action_dim, deltas.device, deltas.dtype)
    normalized = 2.0 * (deltas - data_min) / (data_max - data_min) - 1.0
    return normalized.clamp(-1.0, 1.0) if bool(stats.get("clip", False)) else normalized


def unnormalize_deltas(ndeltas: torch.Tensor, stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS) -> torch.Tensor:
    action_dim = int(ndeltas.shape[-1])
    data_min = _as_action_tensor(stats["min"], action_dim, ndeltas.device, ndeltas.dtype)
    data_max = _as_action_tensor(stats["max"], action_dim, ndeltas.device, ndeltas.dtype)
    values = ndeltas.clamp(-1.0, 1.0) if bool(stats.get("clip", False)) else ndeltas
    return 0.5 * (values + 1.0) * (data_max - data_min) + data_min


def normalized_deltas_to_actions(
    ndeltas: torch.Tensor,
    stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS,
) -> torch.Tensor:
    actions = torch.cumsum(unnormalize_deltas(ndeltas, stats), dim=1)
    if actions.shape[-1] >= 3:
        actions = actions.clone()
        actions[..., 2] = wrap_to_pi(actions[..., 2])
    return actions


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
        use_fp16: bool = True,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        from mdm.model.v2 import MDMModel

        checkpoint_text = str(checkpoint)
        local_checkpoint = Path(checkpoint_text).expanduser()
        source: str | Path = local_checkpoint if local_checkpoint.is_file() else checkpoint_text
        self.model = MDMModel.from_pretrained(source)
        if hasattr(self.model, "enable_pytorch_native_sdpa"):
            self.model.enable_pytorch_native_sdpa()
        self.resolution_level = int(resolution_level)
        self.num_tokens = None if num_tokens is None else int(num_tokens)
        self.use_fp16 = bool(use_fp16)
        if freeze:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        encoder = self.model.encoder
        self.out_dim = int(getattr(encoder, "dim_features", 1024))

    def forward(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}.")
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        expected = (image.shape[0], image.shape[-2], image.shape[-1])
        if tuple(depth.shape) != expected:
            raise ValueError(f"depth must have shape {expected}, got {tuple(depth.shape)}.")
        if not torch.isfinite(depth).all():
            raise ValueError("depth contains non-finite values.")
        depth = depth.to(device=image.device, dtype=image.dtype)

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


class PlanObservationCrossAttention(nn.Module):
    """Observation queries attend to plan memory with safe all-masked handling."""

    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.memory_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.mlp_norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, observation: torch.Tensor, plan: torch.Tensor, plan_valid: torch.Tensor
    ) -> torch.Tensor:
        fully_masked = ~plan_valid.any(dim=1)
        key_padding_mask = ~plan_valid
        if fully_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[fully_masked, 0] = False
        attended, _ = self.attention(
            self.query_norm(observation),
            self.memory_norm(plan),
            self.memory_norm(plan),
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        if fully_masked.any():
            attended = attended * (~fully_masked).to(attended.dtype).view(-1, 1, 1)
        observation = observation + attended
        return observation + self.mlp(self.mlp_norm(observation))


class Stage2VisionEncoder(nn.Module):
    """Encode current RGB-D and fuse it with the token-preserving Reason1 plan."""

    def __init__(
        self,
        context_size: int = 0,
        obs_encoder: str = "efficientnet-b0",
        obs_encoding_size: int = 256,
        plan_dim: int = 384,
        mha_num_attention_heads: int = 4,
        mha_num_attention_layers: int = 4,
        mha_ff_dim_factor: int = 4,
        plan_fusion_layers: int = 2,
        depth_encoder: nn.Module | None = None,
        depth_token_dim: int = 1024,
        depth_proj_dim: int = 384,
        depth_pool_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.context_size = int(context_size)
        self.obs_encoding_size = int(obs_encoding_size)
        self.plan_dim = int(plan_dim)

        if not obs_encoder.startswith("efficientnet"):
            raise NotImplementedError(f"Unsupported obs_encoder={obs_encoder!r}.")
        self.obs_encoder = replace_bn_with_gn(EfficientNet.from_name(obs_encoder, in_channels=3))
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

        obs_token_count = self.context_size + 2
        self.positional_encoding = PositionalEncoding(self.obs_encoding_size, max_seq_len=obs_token_count)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.obs_encoding_size,
            nhead=mha_num_attention_heads,
            dim_feedforward=mha_ff_dim_factor * self.obs_encoding_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sa_encoder = nn.TransformerEncoder(encoder_layer, num_layers=mha_num_attention_layers)
        self.plan_to_obs = nn.Sequential(nn.LayerNorm(self.plan_dim), nn.Linear(self.plan_dim, self.obs_encoding_size))
        self.plan_fusion = nn.ModuleList(
            PlanObservationCrossAttention(
                self.obs_encoding_size, mha_num_attention_heads, mha_ff_dim_factor, dropout
            )
            for _ in range(int(plan_fusion_layers))
        )
        self.obs_to_context = nn.Sequential(
            nn.LayerNorm(self.obs_encoding_size), nn.Linear(self.obs_encoding_size, self.plan_dim)
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
            obs_encoding = self.obs_encoder._dropout(obs_encoding.flatten(start_dim=1))
        return self.compress_obs_enc(obs_encoding).view(batch_size, num_frames, self.obs_encoding_size)

    def _encode_depth(self, image: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        if self.depth_encoder is None:
            return image.new_zeros((image.shape[0], 1, self.obs_encoding_size))
        depth_tokens = self.depth_encoder(image, depth)
        depth_tokens = self.depth_token_proj(depth_tokens).transpose(1, 2).contiguous()
        return self.compress_depth_enc(depth_tokens).unsqueeze(1)

    def forward(
        self,
        obs_img: torch.Tensor,
        depth: torch.Tensor,
        plan_tokens: torch.Tensor,
        input_goal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        obs_context = self._to_context(obs_img)
        batch_size = int(obs_context.shape[0])
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if tuple(depth.shape) != (batch_size, obs_context.shape[-2], obs_context.shape[-1]):
            raise ValueError(
                "depth must be [B,H,W] and spatially match obs_img; "
                f"got depth={tuple(depth.shape)} obs={tuple(obs_context.shape)}."
            )
        if plan_tokens.ndim != 3 or plan_tokens.shape[0] != batch_size or plan_tokens.shape[-1] != self.plan_dim:
            raise ValueError(
                f"plan_tokens must have shape [B,N,{self.plan_dim}], got {tuple(plan_tokens.shape)}."
            )

        observation = torch.cat(
            (self._encode_obs(obs_context), self._encode_depth(obs_context[:, -1], depth)), dim=1
        )
        observation = self.sa_encoder(self.positional_encoding(observation))
        projected_plan = self.plan_to_obs(plan_tokens)
        plan_valid = torch.ones(plan_tokens.shape[:2], device=plan_tokens.device, dtype=torch.bool)
        if input_goal_mask is not None:
            drop_plan = input_goal_mask.to(device=plan_tokens.device, dtype=torch.bool).view(-1)
            if drop_plan.shape[0] != batch_size:
                raise ValueError("input_goal_mask must have one value per batch item.")
            plan_valid = plan_valid & ~drop_plan.unsqueeze(1)
        for layer in self.plan_fusion:
            observation = layer(observation, projected_plan, plan_valid)

        global_cond = observation.mean(dim=1)
        context = torch.cat((self.obs_to_context(observation), plan_tokens), dim=1)
        context_mask = torch.cat((torch.zeros(observation.shape[:2], device=observation.device, dtype=torch.bool), ~plan_valid), dim=1)
        return {"global_cond": global_cond, "context": context, "context_mask": context_mask}


class FastNavModel(nn.Module):
    """FlowNav-compatible wrapper with Reason1 plan conditioning."""

    def __init__(self, vision_encoder: Stage2VisionEncoder, noise_pred_net: nn.Module, dist_pred_net: nn.Module) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net

    def encode_condition(
        self,
        obs_img: torch.Tensor,
        depth: torch.Tensor,
        plan_tokens: torch.Tensor,
        input_goal_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        return self.vision_encoder(
            obs_img, depth, plan_tokens, input_goal_mask=input_goal_mask
        )

    def forward(self, func_name: str, **kwargs: Any) -> Any:
        if func_name == "vision_encoder":
            plan_tokens = kwargs.get("plan_tokens", kwargs.get("z_plan"))
            if plan_tokens is None:
                raise KeyError("vision_encoder requires plan_tokens.")
            if kwargs.get("depth") is None:
                raise KeyError("vision_encoder requires depth.")
            return self.encode_condition(
                obs_img=kwargs["obs_img"],
                depth=kwargs["depth"],
                plan_tokens=plan_tokens,
                input_goal_mask=kwargs.get("input_goal_mask"),
            )
        if func_name == "noise_pred_net":
            return self.noise_pred_net(
                sample=kwargs["sample"],
                timestep=kwargs["timestep"],
                global_cond=kwargs["global_cond"],
                context=kwargs.get("context"),
                context_mask=kwargs.get("context_mask"),
            )
        if func_name == "dist_pred_net":
            return self.dist_pred_net(kwargs["obsgoal_cond"].flatten(start_dim=1))
        raise NotImplementedError(f"Unknown func_name={func_name!r}.")

    def cfm_loss(
        self,
        obs_img: torch.Tensor,
        depth: torch.Tensor,
        plan_tokens: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        action_stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS,
        input_goal_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        cond = self.encode_condition(obs_img, depth, plan_tokens, input_goal_mask=input_goal_mask)
        naction = normalize_deltas(actions_to_deltas(actions), action_stats)
        noise = torch.randn_like(naction)
        t, xt, ut = sample_cfm_location(noise, naction)
        vt = self.noise_pred_net(
            sample=xt,
            timestep=t,
            global_cond=cond["global_cond"],
            context=cond["context"],
            context_mask=cond["context_mask"],
        )
        if action_mask is None:
            action_mask = torch.ones(actions.shape[0], device=actions.device, dtype=actions.dtype)
        flow_loss = action_reduce(F.mse_loss(vt, ut, reduction="none"), action_mask)
        return flow_loss, {"flow_loss": flow_loss.detach(), "cond": cond["global_cond"].detach()}

    @torch.no_grad()
    def sample_actions(
        self,
        obs_img: torch.Tensor,
        depth: torch.Tensor,
        plan_tokens: torch.Tensor,
        pred_horizon: int = 16,
        action_dim: int = 3,
        steps: int = 10,
        action_stats: Mapping[str, Sequence[float]] = DEFAULT_ACTION_STATS,
        input_goal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cond = self.encode_condition(obs_img, depth, plan_tokens, input_goal_mask=input_goal_mask)
        x = torch.randn(
            (obs_img.shape[0], pred_horizon, action_dim),
            device=obs_img.device,
            dtype=cond["global_cond"].dtype,
        )
        dt = 1.0 / max(int(steps), 1)
        for i in range(max(int(steps), 1)):
            t = torch.full((x.shape[0],), i * dt, device=x.device, dtype=x.dtype)
            x = x + self.noise_pred_net(
                sample=x,
                timestep=t,
                global_cond=cond["global_cond"],
                context=cond["context"],
                context_mask=cond["context_mask"],
            ) * dt
        return normalized_deltas_to_actions(x, action_stats)


def build_lingbot_depth_encoder(cfg: Mapping[str, Any]) -> LingBotDepthFeatureEncoder:
    return LingBotDepthFeatureEncoder(
        checkpoint=cfg.get("checkpoint", DEFAULT_LINGBOT_CHECKPOINT),
        resolution_level=int(cfg.get("resolution_level", 5)),
        num_tokens=cfg.get("num_tokens"),
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
    context_size = int(cfg.get("context_size", vision_cfg.get("context_size", 0)))
    plan_dim = int(cfg.get("plan_dim", cfg.get("z_dim", 384)))
    vision_encoder = Stage2VisionEncoder(
        context_size=context_size,
        obs_encoder=str(vision_cfg.get("obs_encoder", "efficientnet-b0")),
        obs_encoding_size=encoding_size,
        plan_dim=plan_dim,
        mha_num_attention_heads=int(vision_cfg.get("mha_num_attention_heads", 4)),
        mha_num_attention_layers=int(vision_cfg.get("mha_num_attention_layers", 4)),
        mha_ff_dim_factor=int(vision_cfg.get("mha_ff_dim_factor", 4)),
        plan_fusion_layers=int(vision_cfg.get("plan_fusion_layers", 2)),
        depth_encoder=depth_encoder,
        depth_token_dim=depth_token_dim,
        depth_proj_dim=int(vision_cfg.get("depth_proj_dim", 384)),
        depth_pool_dim=int(vision_cfg.get("depth_pool_dim", 64)),
        dropout=float(vision_cfg.get("dropout", 0.0)),
    )

    unet_cfg = cfg.get("noise_pred_net", {})
    noise_pred_net = ConditionalUnet1D(
        input_dim=int(cfg.get("action_dim", 3)),
        global_cond_dim=encoding_size,
        diffusion_step_embed_dim=int(unet_cfg.get("diffusion_step_embed_dim", 256)),
        down_dims=tuple(unet_cfg.get("down_dims", (64, 128, 256))),
        kernel_size=int(unet_cfg.get("kernel_size", 3)),
        n_groups=int(unet_cfg.get("n_groups", 8)),
        cond_predict_scale=bool(unet_cfg.get("cond_predict_scale", False)),
        context_dim=plan_dim,
        cross_attn_heads=int(unet_cfg.get("cross_attn_heads", 4)),
        action_horizon=int(cfg.get("action_horizon", 16)),
        cross_attn_on_all_resolutions=bool(unet_cfg.get("cross_attn_on_all_resolutions", True)),
    )
    dist_pred_net = DenseNetwork(embedding_dim=encoding_size)
    return FastNavModel(vision_encoder=vision_encoder, noise_pred_net=noise_pred_net, dist_pred_net=dist_pred_net)


def load_flownav_checkpoint(
    model: nn.Module,
    checkpoint: str | Path = DEFAULT_FLOWNAV_CHECKPOINT,
    skip_prefixes: Sequence[str] = (
        "vision_encoder.goal_encoder",
        "vision_encoder.z_goal_projector",
        "vision_encoder.z_plan_projector",
        "vision_encoder.depth_encoder",
    ),
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
