import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from einops import rearrange


class SinusoidalPosEmb(Module):
    """Classic Transformer-style time embedding used in DDPM."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResidualBlock(Module):
    """DDPM residual block with GroupNorm, dropout, and FiLM-style time conditioning."""

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        time_emb_dim: int,
        dropout: float,
        groups: int = 32,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, dim_in)
        self.conv1 = nn.Conv2d(dim_in, dim_out, 3, padding=1)

        self.norm2 = nn.GroupNorm(groups, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(dim_out, dim_out, 3, padding=1)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out),
        )

        self.res_conv = (
            nn.Conv2d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        time_term = self.time_mlp(time_emb)
        h = h + time_term[..., None, None]

        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.res_conv(x)


class AttentionBlock(Module):
    """Self-attention used at 16×16 resolution in the original DDPM."""

    def __init__(self, dim: int, heads: int = 1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.norm = nn.GroupNorm(32, dim)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head (h w) c", head=self.heads)
        k = rearrange(k, "b (head c) h w -> b head (h w) c", head=self.heads)
        v = rearrange(v, "b (head c) h w -> b head (h w) c", head=self.heads)

        attn = torch.softmax(torch.matmul(q, k.transpose(-1, -2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b head (h w) c -> b (head c) h w", head=self.heads, h=h, w=w)
        return self.proj(out) + x


def Downsample(dim: int) -> Module:
    return nn.Conv2d(dim, dim, 3, stride=2, padding=1)


def Upsample(dim: int, dim_out: int) -> Module:
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(dim, dim_out, 3, padding=1),
    )


class UNet(Module):
    """
    Backbone that mirrors the 35.7M-parameter CIFAR-10 model from
    Ho et al. (2020): GroupNorm residual blocks, dropout, and a single
    attention resolution at 16×16.
    """

    def __init__(
        self,
        *,
        dim: int = 128,
        channels: int = 3,
        dim_mults: Tuple[int, ...] = (1, 2, 4, 8),
        image_size: int = 32,
        attention_resolutions: Tuple[int, ...] = (16,),
        dropout: float = 0.1,
        time_emb_dim: int = None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size

        self.init_conv = nn.Conv2d(channels, dim, 3, padding=1)

        time_dim = dim * 4 if time_emb_dim is None else time_emb_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        dims = [dim]
        for mult in dim_mults:
            dims.append(dim * mult)
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = ModuleList([])
        current_res = image_size
        self.resolutions = []

        for idx, (dim_in, dim_out) in enumerate(in_out):
            self.resolutions.append(current_res)
            is_last = idx == (len(in_out) - 1)
            use_attn = current_res in attention_resolutions

            self.downs.append(
                ModuleList(
                    [
                        ResidualBlock(dim_in, dim_out, time_dim, dropout),
                        ResidualBlock(dim_out, dim_out, time_dim, dropout),
                        AttentionBlock(dim_out) if use_attn else nn.Identity(),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

            if not is_last:
                current_res //= 2

        self.mid_block1 = ResidualBlock(dims[-1], dims[-1], time_dim, dropout)
        self.mid_attn = AttentionBlock(dims[-1])
        self.mid_block2 = ResidualBlock(dims[-1], dims[-1], time_dim, dropout)

        up_resolutions = list(reversed(self.resolutions))
        self.ups = ModuleList([])

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            res = up_resolutions[idx]
            is_last = idx == (len(in_out) - 1)
            use_attn = res in attention_resolutions

            self.ups.append(
                ModuleList(
                    [
                        ResidualBlock(dim_out * 2, dim_out, time_dim, dropout),
                        ResidualBlock(dim_out, dim_out, time_dim, dropout),
                        AttentionBlock(dim_out) if use_attn else nn.Identity(),
                        Upsample(dim_out, dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.out_norm = nn.GroupNorm(32, dim)
        self.out_conv = nn.Conv2d(dim, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.image_size and x.shape[-2] == self.image_size, (
            f"expected input spatial size {self.image_size}, got {x.shape[-2:]}"
        )

        time = time.to(x.device)
        if not torch.is_floating_point(time):
            time = time.float()

        x = self.init_conv(x)
        time_emb = self.time_mlp(time)

        skips = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        for block1, block2, attn, upsample in self.ups:
            skip = skips.pop()
            x = torch.cat((x, skip), dim=1)
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            x = upsample(x)

        x = F.silu(self.out_norm(x))
        return self.out_conv(x)
