"""Online (incremental) Gaussian-splat mapping from VIO poses.

The loop, per incoming frame (strictly causal - no frame ever sees the future):

1. **Keyframe selection** from the VIO pose: a frame becomes a keyframe when the camera has
   moved more than ``kf_trans`` metres or rotated more than ``kf_rot_deg`` since the last
   keyframe (MonoGS and SplaTAM use similar covisibility / motion tests).
2. **Growth** (SplaTAM's silhouette densification, Keetha et al. CVPR 2024): render the map at
   the keyframe pose; pixels whose accumulated opacity is below ``silhouette`` - or whose
   observed depth is clearly in front of the rendered surface - are back-projected from the
   depth map into new isotropic splats.
3. **Optimisation** over a **sliding window** of the most recent ``window`` keyframes (plus,
   with probability ``p_old``, a random older keyframe to limit forgetting, as in MonoGS,
   Matsuki et al. CVPR 2024). Loss: ``(1-l) L1 + l (1-SSIM)`` on colour plus a weighted L1 on
   rendered depth. Parameters are updated with the per-splat Adam in :mod:`splatscan.optim`.
4. **Densify / prune** (3DGS, Kerbl et al. 2023): clone / split splats with large mean
   screen-space gradient, drop near-transparent and oversized splats.

Poses are taken from VIO as-is (tracking and mapping are decoupled; the map does not refine
poses). Held-out frames are never used for mapping; they are rendered for evaluation only.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .gaussians import GaussianMap
from .losses import photometric_loss, psnr, ssim
from .raster import Camera


@dataclass
class MapperConfig:
    sh_degree: int = 1
    tile: int = 4
    kf_trans: float = 0.20
    kf_rot_deg: float = 12.0
    window: int = 8
    p_old: float = 0.25
    iters_per_kf: int = 20
    first_kf_iters: int = 60
    lam_ssim: float = 0.2
    depth_weight: float = 0.1
    silhouette: float = 0.5
    depth_front_rel: float = 0.15
    add_stride: int = 2
    sigma_scale: float = 0.6
    densify_every_kf: int = 2
    grad_thresh: float = 2e-5
    split_scale: float = 0.05
    min_opacity: float = 0.01
    max_scale: float = 0.5
    bg: tuple = (0.0, 0.0, 0.0)
    device: str = "cpu"
    seed: int = 0


@dataclass
class Keyframe:
    index: int
    t: float
    cam: Camera
    rgb: torch.Tensor
    depth: torch.Tensor


@dataclass
class MapperLog:
    events: list = field(default_factory=list)

    def add(self, **kw):
        self.events.append(kw)


def camera_from(R_wc, p_wc, intr, device="cpu") -> Camera:
    return Camera.from_pose(np.asarray(R_wc, np.float32), np.asarray(p_wc, np.float32), intr.width, intr.height,
                            intr.fx, intr.fy, intr.cx, intr.cy, device=device)


def _rot_angle_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class OnlineMapper:
    def __init__(self, intr, cfg: MapperConfig | None = None):
        self.cfg = cfg or MapperConfig()
        self.intr = intr
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        self.rand = random.Random(self.cfg.seed)
        self.map = GaussianMap(self.cfg.sh_degree, device=self.cfg.device)
        self.keyframes: list[Keyframe] = []
        self.bg = torch.tensor(self.cfg.bg, device=self.cfg.device)
        self.iterations = 0
        self.log = MapperLog()
        self.last_kf_pose = None

    # -------------------------------------------------------------- keyframes
    def is_keyframe(self, R_wc, p_wc) -> bool:
        if self.last_kf_pose is None:
            return True
        R0, p0 = self.last_kf_pose
        return (np.linalg.norm(p_wc - p0) > self.cfg.kf_trans or
                _rot_angle_deg(R0, R_wc) > self.cfg.kf_rot_deg)

    def _to_tensor(self, a):
        return torch.as_tensor(np.ascontiguousarray(a), device=self.cfg.device, dtype=torch.float32)

    # -------------------------------------------------------------- main entry
    def process(self, frame) -> dict | None:
        """Feed one frame; returns a dict of what happened if it became a keyframe."""
        if frame.R_wc is None or not self.is_keyframe(frame.R_wc, frame.p_wc):
            return None
        t0 = time.perf_counter()
        cam = camera_from(frame.R_wc, frame.p_wc, self.intr, self.cfg.device)
        kf = Keyframe(frame.index, frame.t, cam, self._to_tensor(frame.rgb), self._to_tensor(frame.depth))
        self.keyframes.append(kf)
        self.last_kf_pose = (frame.R_wc, frame.p_wc)
        n_added = self._grow(kf)
        iters = self.cfg.first_kf_iters if len(self.keyframes) == 1 else self.cfg.iters_per_kf
        loss = self._optimise(iters)
        dens = (0, 0, 0)
        if len(self.keyframes) % self.cfg.densify_every_kf == 0:
            dens = self.map.densify_and_prune(self.cfg.grad_thresh, self.cfg.split_scale,
                                              self.cfg.min_opacity, self.cfg.max_scale)
        info = dict(kf=len(self.keyframes) - 1, frame=frame.index, t=frame.t, added=n_added,
                    cloned=dens[0], split=dens[1], pruned=dens[2], splats=len(self.map),
                    loss=loss, iters=iters, seconds=time.perf_counter() - t0)
        self.log.add(**info)
        return info

    @torch.no_grad()
    def _grow(self, kf: Keyframe) -> int:
        if len(self.map) == 0:
            mask = torch.ones_like(kf.depth, dtype=torch.bool)
        else:
            _, d, a = self.map.render(kf.cam, self.bg, tile=self.cfg.tile)
            d_surf = d / a.clamp_min(1e-6)
            front = torch.nan_to_num(d_surf - kf.depth, nan=0.0) > self.cfg.depth_front_rel * torch.nan_to_num(kf.depth, nan=1e9)
            mask = (a < self.cfg.silhouette) | front
        return len(self.map.add_from_depth(kf.rgb, kf.depth, kf.cam, mask, self.cfg.add_stride,
                                           self.cfg.sigma_scale))

    def _sample_kf(self, it: int, iters: int) -> Keyframe:
        kfs = self.keyframes
        if it < max(1, iters // 3):
            return kfs[-1]                       # current keyframe first
        old = kfs[: -self.cfg.window]
        if old and self.rand.random() < self.cfg.p_old:
            return self.rand.choice(old)
        return self.rand.choice(kfs[-self.cfg.window:])

    def _optimise(self, iters: int) -> float:
        loss_v = float("nan")
        for it in range(iters):
            kf = self._sample_kf(it, iters)
            stats = {}
            rgb, depth, alpha = self.map.render(kf.cam, self.bg, stats=stats, tile=self.cfg.tile)
            loss = photometric_loss(rgb, kf.rgb, self.cfg.lam_ssim)
            if self.cfg.depth_weight > 0:
                ok = torch.isfinite(kf.depth)              # sparse depth (stereo features) has holes
                dgt = torch.where(ok, kf.depth, torch.zeros_like(kf.depth))
                loss = loss + self.cfg.depth_weight * ((depth - dgt * alpha.detach()).abs() * ok).mean()
            self.map.opt.zero_grad()
            loss.backward()
            self.map.accumulate_grad_stats(stats)
            self.map.opt.step()
            self.iterations += 1
            loss_v = loss.item()
        return loss_v

    # -------------------------------------------------------------- evaluation
    @torch.no_grad()
    def evaluate(self, views) -> dict:
        """PSNR / SSIM of the current map against held-out ``views`` [(R_wc, p_wc, rgb)]."""
        ps, ss = [], []
        for R_wc, p_wc, rgb in views:
            cam = camera_from(R_wc, p_wc, self.intr, self.cfg.device)
            out, _, _ = self.map.render(cam, self.bg, tile=self.cfg.tile)
            gt = self._to_tensor(rgb)
            out = out.clamp(0, 1)
            ps.append(psnr(out, gt))
            ss.append(ssim(out, gt).item())
        return {"psnr": float(np.mean(ps)) if ps else float("nan"),
                "ssim": float(np.mean(ss)) if ss else float("nan"), "n_views": len(ps)}

    @torch.no_grad()
    def render_rgb(self, R_wc, p_wc, intr=None) -> np.ndarray:
        cam = camera_from(R_wc, p_wc, intr or self.intr, self.cfg.device)
        out, _, _ = self.map.render(cam, self.bg, tile=self.cfg.tile)
        return out.clamp(0, 1).cpu().numpy()
