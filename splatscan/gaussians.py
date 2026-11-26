"""The splat map: parameters, stable ids, growth from depth, densify (clone/split) and prune.

Parameters are stored unconstrained, as in 3DGS: log-scales, raw quaternions, opacity
logits and SH coefficients (DC and higher bands in separate optimiser groups). Every splat has
a stable integer ``id`` so the streaming layer can send additions / updates / removals.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .optim import SplatAdam
from .raster import Camera, quat_to_rotmat, rasterize
from .sh import num_coeffs, rgb_to_dc

DEFAULT_LRS = {
    "means": 4e-4,        # metres per step (Adam is scale-invariant in the gradient)
    "log_scales": 5e-3,
    "quats": 1e-3,
    "logits": 5e-2,
    "sh_dc": 2.5e-3,
    "sh_rest": 2.5e-3 / 20,
}


class GaussianMap:
    def __init__(self, sh_degree: int = 1, device="cpu", dtype=torch.float32, lrs: dict | None = None):
        self.degree = sh_degree
        self.K = num_coeffs(sh_degree)
        self.device, self.dtype = torch.device(device), dtype
        z = lambda *s: torch.zeros(*s, device=self.device, dtype=dtype)  # noqa: E731
        self.p = {
            "means": z(0, 3), "log_scales": z(0, 3), "quats": z(0, 4), "logits": z(0),
            "sh_dc": z(0, 1, 3), "sh_rest": z(0, self.K - 1, 3),
        }
        for t in self.p.values():
            t.requires_grad_()
        self.ids = torch.zeros(0, dtype=torch.int64, device=self.device)
        self.next_id = 0
        self.opt = SplatAdam(self.p, lrs or DEFAULT_LRS)
        self.grad_accum = z(0)
        self.grad_count = z(0)

    # ------------------------------------------------------------------ basics
    def __len__(self):
        return self.p["means"].shape[0]

    def sh(self):
        return torch.cat([self.p["sh_dc"], self.p["sh_rest"]], 1)

    def render(self, cam: Camera, bg=None, stats=None, tile: int = 8):
        p = self.p
        return rasterize(p["means"], p["log_scales"], p["quats"], p["logits"], self.sh(), cam,
                         bg, degree=self.degree, tile=tile, stats=stats)

    def _replace(self, new: dict):
        for k in self.p:
            self.p[k] = new[k].detach().contiguous().requires_grad_()
        self.opt.params = self.p

    # ------------------------------------------------------------------ growth
    @torch.no_grad()
    def add(self, means, sigma, rgb, opacity: float = 0.5):
        """Append isotropic splats with linear std ``sigma`` (N,), colour ``rgb`` (N, 3)."""
        n = means.shape[0]
        if n == 0:
            return torch.zeros(0, dtype=torch.int64)
        dt, dev = self.dtype, self.device
        new = {
            "means": means.to(dev, dt),
            "log_scales": torch.log(sigma.to(dev, dt))[:, None].expand(n, 3).clone(),
            "quats": torch.tensor([1.0, 0, 0, 0], device=dev, dtype=dt).expand(n, 4).clone(),
            "logits": torch.full((n,), math.log(opacity / (1 - opacity)), device=dev, dtype=dt),
            "sh_dc": rgb_to_dc(rgb.to(dev, dt))[:, None, :],
            "sh_rest": torch.zeros(n, self.K - 1, 3, device=dev, dtype=dt),
        }
        self._replace({k: torch.cat([self.p[k], new[k]]) for k in self.p})
        self.opt.append(n)
        ids = torch.arange(self.next_id, self.next_id + n, device=dev)
        self.next_id += n
        self.ids = torch.cat([self.ids, ids])
        self.grad_accum = torch.cat([self.grad_accum, self.grad_accum.new_zeros(n)])
        self.grad_count = torch.cat([self.grad_count, self.grad_count.new_zeros(n)])
        return ids

    @torch.no_grad()
    def add_from_depth(self, rgb, depth, cam: Camera, mask, stride: int = 2, sigma_scale: float = 0.6,
                       opacity: float = 0.5):
        """Back-project masked pixels (on a ``stride`` grid) of a depth map into new splats.

        Each splat gets sigma = sigma_scale * stride * depth / f: roughly its pixel footprint,
        so neighbouring seeds overlap (SplaTAM-style initialisation).
        """
        H, W = depth.shape
        sel = torch.zeros(H, W, dtype=torch.bool, device=depth.device)
        oy, ox = np.random.randint(stride), np.random.randint(stride)
        sel[oy::stride, ox::stride] = True
        sel &= mask & torch.isfinite(depth) & (depth > cam.near)
        vv, uu = torch.nonzero(sel, as_tuple=True)
        z = depth[vv, uu]
        xc = torch.stack([(uu - cam.cx) / cam.fx * z, (vv - cam.cy) / cam.fy * z, z], 1).to(self.dtype)
        R_cw, t_cw = cam.R_cw.to(xc), cam.t_cw.to(xc)
        xw = (xc - t_cw) @ R_cw                      # R_cw^T (x_c - t)
        sigma = sigma_scale * stride * z / cam.fx
        return self.add(xw, sigma.to(self.dtype), rgb[vv, uu], opacity)

    # ------------------------------------------------------------------ densify / prune
    @torch.no_grad()
    def accumulate_grad_stats(self, stats: dict):
        g = stats.get("grad2d")
        if g is None:
            return
        vis = stats["radii"] > 0
        self.grad_accum += torch.where(vis, g, torch.zeros_like(g))
        self.grad_count += vis.to(self.grad_count.dtype)

    @torch.no_grad()
    def densify_and_prune(self, grad_thresh: float, split_scale: float, min_opacity: float,
                          max_scale: float, min_views: int = 3):
        """Clone small / split large splats with high mean screen-space gradient (3DGS Sec. 5.2);
        prune nearly transparent or oversized ones. Returns (n_cloned, n_split, n_pruned)."""
        n = len(self)
        if n == 0:
            return 0, 0, 0
        avg = self.grad_accum / self.grad_count.clamp_min(1)
        hot = (avg > grad_thresh) & (self.grad_count >= min_views)
        smax = self.p["log_scales"].exp().amax(1)
        clone = hot & (smax <= split_scale)
        split = hot & (smax > split_scale)
        keep_idx = torch.arange(n, device=self.device)
        # --- split: two children sampled from the parent's Gaussian, scales / 1.6
        si = torch.nonzero(split).squeeze(1)
        R = quat_to_rotmat(torch.nn.functional.normalize(self.p["quats"][si], dim=1))
        s = self.p["log_scales"][si].exp()
        children = []
        for _ in range(2):
            eps = torch.randn_like(s) * s
            children.append(self.p["means"][si] + (R @ eps[:, :, None])[:, :, 0])
        ci = torch.nonzero(clone).squeeze(1)
        src = torch.cat([keep_idx, ci, si, si])
        new = {k: v[src].clone() for k, v in self.p.items()}
        if len(si):
            new["means"][n + len(ci):] = torch.cat(children)
            new["log_scales"][n + len(ci):] -= math.log(1.6)
        self.opt.gather(src)
        self._replace(new)
        new_ids = torch.arange(self.next_id, self.next_id + len(ci) + 2 * len(si), device=self.device)
        self.next_id += len(new_ids)
        self.ids = torch.cat([self.ids, new_ids])
        # parents that were split are removed below together with the prune set
        m = len(self)
        drop = torch.zeros(m, dtype=torch.bool, device=self.device)
        drop[si] = True
        opac = torch.sigmoid(self.p["logits"])
        smax = self.p["log_scales"].exp().amax(1)
        prune = (opac < min_opacity) | (smax > max_scale)
        n_pruned = int((prune & ~drop).sum())
        self.remove(drop | prune)
        self.grad_accum = torch.zeros(len(self), device=self.device, dtype=self.dtype)
        self.grad_count = torch.zeros_like(self.grad_accum)
        return len(ci), len(si), n_pruned

    @torch.no_grad()
    def remove(self, mask: torch.Tensor):
        keep = torch.nonzero(~mask).squeeze(1)
        self.opt.gather(keep)
        self._replace({k: v[keep] for k, v in self.p.items()})
        self.ids = self.ids[keep]
        if len(self.grad_accum) == len(mask):
            self.grad_accum, self.grad_count = self.grad_accum[keep], self.grad_count[keep]

    # ------------------------------------------------------------------ export
    @torch.no_grad()
    def activated(self) -> dict:
        """numpy arrays with activations applied (what streaming and .ply export carry)."""
        p = self.p
        return {
            "ids": self.ids.cpu().numpy().astype(np.uint32),
            "pos": p["means"].float().cpu().numpy(),
            "scale": p["log_scales"].exp().float().cpu().numpy(),
            "rot": torch.nn.functional.normalize(p["quats"], dim=1).float().cpu().numpy(),
            "opacity": torch.sigmoid(p["logits"]).float().cpu().numpy(),
            "sh": self.sh().float().cpu().numpy().reshape(len(self), -1),
        }
