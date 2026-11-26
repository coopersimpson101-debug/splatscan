"""Image metrics and losses: PSNR, SSIM (Wang et al. 2004, 11x11 Gaussian window, sigma 1.5)."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gauss_window(size: int, sigma: float, dtype, device):
    x = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2
    g = torch.exp(-x * x / (2 * sigma * sigma))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None]


def ssim(a: torch.Tensor, b: torch.Tensor, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Mean SSIM of two (H, W, 3) images in [0, 1] (valid-region, per-channel, then averaged)."""
    x = a.permute(2, 0, 1)[:, None]
    y = b.permute(2, 0, 1)[:, None]
    w = _gauss_window(size, sigma, a.dtype, a.device)
    mx, my = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mx * mx
    syy = F.conv2d(y * y, w) - my * my
    sxy = F.conv2d(x * y, w) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))
    return s.mean()


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.clamp(0, 1) - b) ** 2).item()
    return 10 * math.log10(1.0 / max(mse, 1e-12))


def photometric_loss(rgb, gt, lam: float = 0.2):
    """3DGS loss: (1 - lam) L1 + lam (1 - SSIM)."""
    return (1 - lam) * (rgb - gt).abs().mean() + lam * (1 - ssim(rgb, gt))
