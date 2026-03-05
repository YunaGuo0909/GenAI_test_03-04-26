"""
DDPM (Denoising Diffusion Probabilistic Model) for animal image generation.

Implements Ho et al. (2020) with a U-Net denoising backbone.
Supports optional class-conditional generation via Classifier-Free Guidance.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEmbedding(nn.Module):
    """Encodes diffusion timestep t into a vector via sinusoidal embeddings."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class ResBlock(nn.Module):
    """Residual block with timestep conditioning and optional class embedding."""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, num_classes: int = 0):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.class_mlp = nn.Sequential(nn.SiLU(), nn.Linear(num_classes, out_ch)) if num_classes > 0 else None
        self.conv2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor = None) -> torch.Tensor:
        h = self.conv1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        if self.class_mlp is not None and c_emb is not None:
            h = h + self.class_mlp(c_emb)[:, :, None, None]
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Self-attention at lower spatial resolutions for global context."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.einsum("bci,bcj->bij", q, k) * (C ** -0.5)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v).reshape(B, C, H, W)
        return x + self.proj(out)


class UNet(nn.Module):
    """
    U-Net denoising network for DDPM.

    Predicts noise epsilon given noisy image x_t, timestep t, and optional class label.
    Architecture: 4-level encoder-decoder with skip connections.
    Spatial resolutions: 64 -> 32 -> 16 -> 8 -> 16 -> 32 -> 64
    """

    def __init__(self, in_ch: int = 3, base_ch: int = 64, num_classes: int = 0):
        super().__init__()
        time_dim = base_ch * 4
        ch_mult = [1, 2, 4, 8]
        self.num_classes = num_classes

        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.init_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        prev_ch = base_ch
        for mult in ch_mult:
            out_ch = base_ch * mult
            self.down_blocks.append(nn.ModuleList([
                ResBlock(prev_ch, out_ch, time_dim, num_classes),
                ResBlock(out_ch, out_ch, time_dim, num_classes),
                AttentionBlock(out_ch) if mult >= 4 else nn.Identity(),
            ]))
            self.down_samples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            prev_ch = out_ch

        # Bottleneck
        mid_ch = base_ch * ch_mult[-1]
        self.mid_block1 = ResBlock(mid_ch, mid_ch, time_dim, num_classes)
        self.mid_attn = AttentionBlock(mid_ch)
        self.mid_block2 = ResBlock(mid_ch, mid_ch, time_dim, num_classes)

        # Decoder (spatial upsampling preserves channels; ResBlocks handle channel reduction)
        self.up_blocks = nn.ModuleList()
        for mult in reversed(ch_mult):
            out_ch = base_ch * mult
            self.up_blocks.append(nn.ModuleList([
                ResBlock(prev_ch + out_ch, out_ch, time_dim, num_classes),
                ResBlock(out_ch, out_ch, time_dim, num_classes),
                AttentionBlock(out_ch) if mult >= 4 else nn.Identity(),
            ]))
            prev_ch = out_ch

        self.final = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, in_ch, 3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        t_emb = self.time_embed(t)
        c_emb = c if self.num_classes > 0 else None

        h = self.init_conv(x)

        # Encoder with skip connections
        skips = []
        for (res1, res2, attn), downsample in zip(self.down_blocks, self.down_samples):
            h = res1(h, t_emb, c_emb)
            h = res2(h, t_emb, c_emb)
            h = attn(h) if not isinstance(attn, nn.Identity) else h
            skips.append(h)
            h = downsample(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb, c_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, c_emb)

        # Decoder
        for res1, res2, attn in self.up_blocks:
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            skip = skips.pop()
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            h = res1(h, t_emb, c_emb)
            h = res2(h, t_emb, c_emb)
            h = attn(h) if not isinstance(attn, nn.Identity) else h

        return self.final(h)


class GaussianDiffusion:
    """
    Manages the forward (noising) and reverse (denoising) diffusion process.

    Forward: q(x_t | x_0) = N(x_t; sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    Reverse: p(x_{t-1} | x_t) parameterized by UNet predicting noise epsilon.
    """

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02,
                 device: torch.device = None):
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")

        betas = torch.linspace(beta_start, beta_end, timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.betas = betas
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor = None) -> torch.Tensor:
        """Forward diffusion: add noise to x_0 at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def p_losses(self, model: nn.Module, x_0: torch.Tensor, t: torch.Tensor,
                 c: torch.Tensor = None) -> torch.Tensor:
        """Compute MSE loss between predicted and actual noise."""
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t, c)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x_t: torch.Tensor, t: int,
                 c: torch.Tensor = None) -> torch.Tensor:
        """Single reverse diffusion step."""
        t_batch = torch.full((x_t.shape[0],), t, device=self.device, dtype=torch.long)
        predicted_noise = model(x_t, t_batch, c)

        beta_t = self.betas[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]

        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_alpha * predicted_noise)

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(self.posterior_variance[t])
            return mean + sigma * noise
        return mean

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple, c: torch.Tensor = None) -> torch.Tensor:
        """Generate images via full reverse diffusion from pure noise."""
        x = torch.randn(shape, device=self.device)
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(model, x, t, c)
        return x
