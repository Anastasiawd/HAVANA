from typing import Optional, Union
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops.layers.torch import Rearrange

from diffusion_policy.model.diffusion.conv1d_components import (
    Downsample1d, Upsample1d, Conv1dBlock)
from diffusion_policy.model.diffusion.positional_embedding import SinusoidalPosEmb

logger = logging.getLogger(__name__)


class VideoCrossAttention(nn.Module):
    """
    Cross-attention from UNet hidden [B, L, C] to a video-token sequence
    [B, N, D]. The video tokens are projected from D -> C with a per-layer
    linear. A temporal alignment bias [L, N] is added to the attention
    logits; alpha is a learnable scalar shared per layer, initialised to 0.1.

    Output projection is zero-initialised (ControlNet-style) so that at init
    the layer is exactly a residual no-op regardless of the cross-attn inputs.

    If ``context is None`` the layer is a strict identity (residual skip),
    which is required for bit-exact parity with the legacy use_video_goal=False
    code path.

    ``context_mask`` is a [B, N] bool tensor where True means "mask this key".
    Samples whose entire mask row is True (fully masked) would produce NaNs
    from a softmax over all -inf logits; we handle that by zeroing the
    per-sample cross-attn residual output for those rows explicitly (equivalent
    to routing zero information through the residual for fully-masked samples,
    which is exactly the CFG unconditional branch).
    """

    def __init__(
        self,
        channels: int,
        context_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert channels % num_heads == 0, (
            f"channels={channels} must be divisible by num_heads={num_heads}"
        )
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        self.norm = nn.GroupNorm(8, channels)

        self.to_q = nn.Linear(channels, channels, bias=False)
        self.context_proj = nn.Linear(context_dim, channels * 2, bias=False)  # K and V

        # learnable temporal-bias scale, initialised to 0.1 per spec
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # zero-init output projection (ControlNet zero-conv idea)
        self.out_proj = nn.Linear(channels, channels)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.attn_dropout = dropout

    @staticmethod
    def _action_position_encoding(length: int, channels: int, device, dtype) -> torch.Tensor:
        """Fixed 1-D encoding so equal-valued action slots remain distinguishable."""
        half = max(1, channels // 2)
        positions = torch.linspace(-1.0, 1.0, length, device=device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(half, device=device, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0, device=device)) / max(half - 1, 1))
        ).unsqueeze(0)
        encoding = torch.cat((torch.sin(positions * frequencies), torch.cos(positions * frequencies)), dim=1)
        return encoding[:, :channels].to(dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,                       # [B, C, L]
        context: Optional[torch.Tensor],       # [B, N, D]
        context_mask: Optional[torch.Tensor],  # [B, N] bool, True => mask
        temporal_bias: Optional[torch.Tensor], # [L, N] float
    ) -> torch.Tensor:
        # early-exit identity when no context provided -- preserves legacy path
        if context is None:
            return x

        B, C, L = x.shape
        _, N, D = context.shape
        assert C == self.channels, f"channels {C} != {self.channels}"
        assert D == self.context_dim, f"context_dim {D} != {self.context_dim}"

        # normalize in channel dim, then move to [B, L, C] for linear layers
        h = self.norm(x)
        h = h.transpose(1, 2)  # [B, L, C]
        h = h + self._action_position_encoding(L, C, h.device, h.dtype).unsqueeze(0)

        q = self.to_q(h)  # [B, L, C]
        kv = self.context_proj(context)  # [B, N, 2C]
        k, v = kv.chunk(2, dim=-1)  # [B, N, C] each

        # split heads: [B, H, L, head_dim]
        def _split(t, seq):
            return t.reshape(B, seq, self.num_heads, self.head_dim).transpose(1, 2)

        q = _split(q, L)
        k = _split(k, N)
        v = _split(v, N)

        scale = 1.0 / (self.head_dim ** 0.5)
        # [B, H, L, N]
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale

        if temporal_bias is not None:
            # broadcast [L, N] -> [1, 1, L, N]
            attn_logits = attn_logits + temporal_bias.to(dtype=attn_logits.dtype).view(1, 1, L, N)

        # build the key-padding mask. We also detect rows that are *fully* masked
        # and later zero their residual -- avoids softmax NaNs and matches the
        # CFG "unconditional" semantics (no info from context).
        fully_masked_rows = None
        if context_mask is not None:
            assert context_mask.dtype == torch.bool, "context_mask must be bool"
            assert context_mask.shape == (B, N), f"context_mask shape {context_mask.shape}"
            fully_masked_rows = context_mask.all(dim=-1)  # [B]
            # broadcast to [B, 1, 1, N]
            mask_expand = context_mask.view(B, 1, 1, N)
            # For fully-masked rows, temporarily "unmask" one key to avoid NaNs;
            # we'll zero out the residual for those rows after the attention.
            if fully_masked_rows.any():
                safe_mask = context_mask.clone()
                safe_mask[fully_masked_rows, 0] = False
                mask_expand = safe_mask.view(B, 1, 1, N)
            attn_logits = attn_logits.masked_fill(mask_expand, float("-inf"))

        attn = F.softmax(attn_logits, dim=-1)
        if self.attn_dropout > 0 and self.training:
            attn = F.dropout(attn, p=self.attn_dropout)

        out = torch.matmul(attn, v)  # [B, H, L, head_dim]
        out = out.transpose(1, 2).reshape(B, L, C)  # [B, L, C]

        out = self.out_proj(out)  # zero-init at start

        if fully_masked_rows is not None and fully_masked_rows.any():
            # [B, 1, 1] -> broadcast to [B, L, C]
            keep = (~fully_masked_rows).to(out.dtype).view(B, 1, 1)
            out = out * keep

        # residual back in [B, C, L] layout
        out = out.transpose(1, 2)  # [B, C, L]
        return x + out

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, 
            in_channels, 
            out_channels, 
            cond_dim,
            kernel_size=3,
            n_groups=8,
            cond_predict_scale=False):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange('batch t -> batch t 1'),
        )

        # make sure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            embed = embed.reshape(
                embed.shape[0], 2, self.out_channels, 1)
            scale = embed[:,0,...]
            bias = embed[:,1,...]
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    def __init__(self,
        input_dim,
        local_cond_dim=None,
        global_cond_dim=None,
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=3,
        n_groups=8,
        cond_predict_scale=False,
        # --- video cross-attention (new) ---
        context_dim: Optional[int] = None,
        cross_attn_heads: int = 4,
        action_horizon: Optional[int] = None,
        video_num_frames: Optional[int] = None,
        video_num_spatial_per_frame: Optional[int] = None,
        cross_attn_on_mid: bool = True,
        cross_attn_on_deepest_down: bool = True,
        cross_attn_on_deepest_up: bool = True,
        cross_attn_on_all_resolutions: bool = False,
        ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        local_cond_encoder = None
        if local_cond_dim is not None:
            _, dim_out = in_out[0]
            dim_in = local_cond_dim
            local_cond_encoder = nn.ModuleList([
                # down encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                # up encoder
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale)
            ])

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim, 
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
        
        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.local_cond_encoder = local_cond_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

        # -----------------------------------------------------------------
        # Video cross-attention layers (optional).
        #
        # Insertion points per spec: mid block, deepest down block, shallowest
        # (= deepest-channel) up block. For down_dims=[64,128,256] these have
        # channel counts 256, 256, 128 respectively.
        #
        # All cross-attn modules share the convention that when ``context is
        # None`` they are a strict identity, so a model instantiated with
        # context_dim=None behaves exactly like the legacy ConditionalUnet1D.
        # -----------------------------------------------------------------
        self.context_dim = context_dim
        self.video_num_frames = video_num_frames
        self.video_num_spatial_per_frame = video_num_spatial_per_frame
        self.action_horizon = action_horizon

        self.mid_cross_attn = None
        self.down_cross_attn = None
        self.up_cross_attn = None
        self.down_cross_attn_layers = nn.ModuleList()
        self.up_cross_attn_layers = nn.ModuleList()

        if context_dim is not None:
            mid_channels = all_dims[-1]
            if cross_attn_on_mid:
                self.mid_cross_attn = VideoCrossAttention(
                    channels=mid_channels,
                    context_dim=context_dim,
                    num_heads=cross_attn_heads,
                )
            if cross_attn_on_deepest_down:
                # Channel count of the deepest down block output (before downsample)
                deepest_down_channels = in_out[-1][1]
                self.down_cross_attn = VideoCrossAttention(
                    channels=deepest_down_channels,
                    context_dim=context_dim,
                    num_heads=cross_attn_heads,
                )
            if cross_attn_on_deepest_up:
                # First entry of up_modules corresponds to the deepest up block;
                # its resnet2 output has `dim_in` channels (i.e. one step smaller
                # than mid_channels). For down_dims=[64,128,256] -> 128.
                deepest_up_channels = list(reversed(in_out[1:]))[0][0]
                self.up_cross_attn = VideoCrossAttention(
                    channels=deepest_up_channels,
                    context_dim=context_dim,
                    num_heads=cross_attn_heads,
                )
            if cross_attn_on_all_resolutions:
                self.down_cross_attn = None
                self.up_cross_attn = None
                self.down_cross_attn_layers = nn.ModuleList(
                    VideoCrossAttention(
                        channels=dim_out,
                        context_dim=context_dim,
                        num_heads=cross_attn_heads,
                    )
                    for _, dim_out in in_out
                )
                self.up_cross_attn_layers = nn.ModuleList(
                    VideoCrossAttention(
                        channels=dim_in,
                        context_dim=context_dim,
                        num_heads=cross_attn_heads,
                    )
                    for dim_in, _ in reversed(in_out[1:])
                )

        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def _build_temporal_bias_for(self, L: int, alpha: torch.Tensor, device, dtype) -> Optional[torch.Tensor]:
        """Build the [L, N] temporal bias for this UNet's configured video layout."""
        if self.context_dim is None or self.video_num_frames is None or self.video_num_spatial_per_frame is None:
            return None
        # local import to avoid a hard circular dep during partial package reloads
        from vint_train.models.nomad.video_latent_projector import build_temporal_bias
        T_v = self.video_num_frames
        N = T_v * self.video_num_spatial_per_frame
        return build_temporal_bias(
            L=L,
            N=N,
            T_v=T_v,
            num_spatial_per_frame=self.video_num_spatial_per_frame,
            alpha=alpha,
            device=device,
            dtype=dtype,
        )

    def forward(self,
            sample: torch.Tensor,
            timestep: Union[torch.Tensor, float, int],
            local_cond=None, global_cond=None,
            context: Optional[torch.Tensor] = None,
            context_mask: Optional[torch.Tensor] = None,
            **kwargs):
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        context: (B,N,context_dim) video-token sequence for cross-attention (optional)
        context_mask: (B,N) bool, True = mask that key (optional)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, 'b h t -> b t h')

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)
        
        # encode local features
        h_local = list()
        if local_cond is not None:
            local_cond = einops.rearrange(local_cond, 'b h t -> b t h')
            resnet, resnet2 = self.local_cond_encoder
            x = resnet(local_cond, global_feature)
            h_local.append(x)
            x = resnet2(local_cond, global_feature)
            h_local.append(x)
        
        x = sample
        h = []
        num_down = len(self.down_modules)
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            if idx == 0 and len(h_local) > 0:
                x = x + h_local[0]
            x = resnet2(x, global_feature)
            if self.down_cross_attn_layers:
                cross_attn = self.down_cross_attn_layers[idx]
                x = cross_attn(x, context, context_mask, None)
            # Cross-attention after the deepest down block, before its (Identity) downsample.
            if idx == num_down - 1 and self.down_cross_attn is not None:
                tbias = self._build_temporal_bias_for(
                    L=x.shape[-1], alpha=self.down_cross_attn.alpha,
                    device=x.device, dtype=x.dtype,
                )
                x = self.down_cross_attn(x, context, context_mask, tbias)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        if self.mid_cross_attn is not None:
            tbias = self._build_temporal_bias_for(
                L=x.shape[-1], alpha=self.mid_cross_attn.alpha,
                device=x.device, dtype=x.dtype,
            )
            x = self.mid_cross_attn(x, context, context_mask, tbias)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            # The correct condition should be:
            # if idx == (len(self.up_modules)-1) and len(h_local) > 0:
            # However this change will break compatibility with published checkpoints.
            # Therefore it is left as a comment.
            if idx == len(self.up_modules) and len(h_local) > 0:
                x = x + h_local[1]
            x = resnet2(x, global_feature)
            if self.up_cross_attn_layers:
                cross_attn = self.up_cross_attn_layers[idx]
                x = cross_attn(x, context, context_mask, None)
            # Cross-attention after the deepest (idx=0) up block's resnet2, before upsample.
            if idx == 0 and self.up_cross_attn is not None:
                tbias = self._build_temporal_bias_for(
                    L=x.shape[-1], alpha=self.up_cross_attn.alpha,
                    device=x.device, dtype=x.dtype,
                )
                x = self.up_cross_attn(x, context, context_mask, tbias)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, 'b t h -> b h t')
        return x
