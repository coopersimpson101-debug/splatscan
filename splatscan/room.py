"""Procedurally textured "lab" room, ray-cast in numpy to RGB + metric depth.

The room is the same 8 x 8 x 3 m box insight-vio's simulator uses (world frame: z up, floor at
z = 0), furnished with a few axis-aligned boxes (benches, a cabinet, a crate, a pillar) so the
map has real 3D structure and occlusions. Surfaces carry colour value-noise plus sharp-edged
"posters"/panels (corners for the FAST/KLT front-end) and baked, view-independent lighting
from two ceiling lights, so a Gaussian-splat map can represent the scene exactly in principle.

Pixel centres are at integer coordinates (``cx = W/2 - 0.5``), matching both insight-vio's
camera model and :mod:`splatscan.raster`. No lens distortion, motion blur or rolling shutter.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Box:
    lo: tuple
    hi: tuple
    base: tuple          # base RGB
    pattern: str         # "wall" | "floor" | "wood" | "metal" | "crate"
    inside: bool = False  # the room itself is seen from the inside


LIGHTS = (np.array([-1.5, 0.5, 2.9]), np.array([1.8, -1.0, 2.9]))


def default_boxes(half: float = 4.0, height: float = 3.0) -> list:
    return [
        Box((-half, -half, 0.0), (half, half, height), (0.80, 0.78, 0.72), "wall", inside=True),
        Box((2.6, -2.5, 0.0), (3.6, 1.0, 0.9), (0.55, 0.38, 0.22), "wood"),       # bench, +x wall
        Box((-3.7, -1.2, 0.0), (-3.0, 1.6, 2.0), (0.30, 0.42, 0.62), "metal"),    # cabinet, -x wall
        Box((-1.0, 2.8, 0.0), (1.4, 3.7, 0.75), (0.62, 0.30, 0.25), "wood"),      # bench, +y wall
        Box((0.6, -3.6, 0.0), (1.5, -2.8, 0.8), (0.70, 0.60, 0.30), "crate"),     # crate, -y wall
        Box((-3.2, -3.2, 0.0), (-2.6, -2.6, height), (0.65, 0.65, 0.68), "metal"),  # pillar
        Box((2.7, 2.7, 0.0), (3.2, 3.2, height), (0.65, 0.65, 0.68), "metal"),      # pillar
    ]


class LabRoom:
    def __init__(self, seed: int = 0, boxes: list | None = None):
        self.boxes = boxes or default_boxes()
        rng = np.random.default_rng(seed)
        self.noise = [rng.uniform(0, 1, (256, 256)).astype(np.float32) for _ in range(6)]
        self.poster = rng.uniform(0, 1, (256, 256)).astype(np.float32)
        self.palette = np.array([
            [0.85, 0.20, 0.20], [0.20, 0.55, 0.85], [0.95, 0.75, 0.15], [0.20, 0.70, 0.35],
            [0.55, 0.25, 0.70], [0.95, 0.50, 0.15], [0.10, 0.10, 0.12], [0.95, 0.95, 0.92],
        ], np.float32)

    # ---------------------------------------------------------------- textures
    def _vnoise(self, k, u, v, scale):
        x, y = u / scale, v / scale
        x0, y0 = np.floor(x), np.floor(y)
        fx, fy = x - x0, y - y0
        fx = fx * fx * (3 - 2 * fx)
        fy = fy * fy * (3 - 2 * fy)
        t = self.noise[k]
        i0, j0 = x0.astype(np.int64) & 255, y0.astype(np.int64) & 255
        i1, j1 = (i0 + 1) & 255, (j0 + 1) & 255
        return (t[i0, j0] * (1 - fx) + t[i1, j0] * fx) * (1 - fy) + (t[i0, j1] * (1 - fx) + t[i1, j1] * fx) * fy

    def _texture(self, box: Box, face, u, v, normal_axis):
        n = len(u)
        base = np.broadcast_to(np.asarray(box.base, np.float32), (n, 3)).copy()
        off = face * 31.7
        lum = 0.75 + 0.35 * self._vnoise(0, u + off, v, 0.45) + 0.15 * self._vnoise(1, u, v + off, 0.11)
        col = base * lum[:, None]
        if box.pattern == "wall":
            is_floor = (normal_axis == 2) & (face % 2 == 0)
            is_ceil = (normal_axis == 2) & (face % 2 == 1)
            # floor: large two-tone tiles with grout lines
            tu, tv = np.floor(u / 0.6), np.floor(v / 0.6)
            checker = ((tu + tv) % 2 == 0)
            fl = np.where(checker[:, None], [0.42, 0.44, 0.48], [0.60, 0.62, 0.64]).astype(np.float32)
            grout = (np.abs(u / 0.6 - np.round(u / 0.6)) < 0.04) | (np.abs(v / 0.6 - np.round(v / 0.6)) < 0.04)
            fl = np.where(grout[:, None], 0.25, fl) * lum[:, None]
            col = np.where(is_floor[:, None], fl, col)
            col = np.where(is_ceil[:, None], np.float32(0.9) * lum[:, None], col)
            # posters / panels on the walls: sharp coloured rectangles
            wall = normal_axis != 2
            cu, cv = np.floor(u / 0.7).astype(np.int64), np.floor(v / 0.55).astype(np.int64)
            cell = self.poster[(cu + face * 13) & 255, cv & 255]
            fu, fv = u / 0.7 - cu, v / 0.55 - cv
            inner = (fu > 0.12) & (fu < 0.88) & (fv > 0.15) & (fv < 0.85) & (v > 0.5) & (v < 2.5)
            pidx = (cell * 997).astype(np.int64) % len(self.palette)
            pc = self.palette[pidx] * (0.85 + 0.25 * self._vnoise(2, u * 3, v * 3, 0.3))[:, None]
            show = wall & inner & (cell > 0.45)
            col = np.where(show[:, None], pc, col)
            # a dark skirting board
            col = np.where((wall & (v < 0.12))[:, None], np.float32(0.18), col)
        elif box.pattern == "wood":
            grain = 0.8 + 0.25 * np.sin(u * 40 + 6 * self._vnoise(3, u, v, 0.25))
            col = base * grain[:, None] * lum[:, None]
        elif box.pattern == "metal":
            stripe = (np.floor(v / 0.4) % 2 == 0)
            col = base * np.where(stripe, 1.0, 0.82)[:, None] * (0.9 + 0.1 * lum[:, None])
            label = (np.abs(u - np.round(u * 2) / 2) < 0.12) & (np.abs(v - 1.2) < 0.1)
            col = np.where(label[:, None], [0.95, 0.85, 0.1], col)
        elif box.pattern == "crate":
            plank = np.floor(v / 0.2)
            col = base * (0.75 + 0.3 * self.noise[4][(plank.astype(np.int64) * 7) & 255, 3])[:, None] * lum[:, None]
            edge = (np.abs(v / 0.2 - np.round(v / 0.2)) < 0.06)
            col = np.where(edge[:, None], col * 0.45, col)
        return col

    # ---------------------------------------------------------------- ray casting
    def cast(self, origin: np.ndarray, dirs: np.ndarray):
        """Ray-cast unit-free directions (N, 3) from ``origin``. Returns (rgb (N,3), t (N,))."""
        n = len(dirs)
        best_t = np.full(n, np.inf)
        best_box = np.full(n, -1)
        best_axis = np.zeros(n, np.int64)
        best_side = np.zeros(n, np.int64)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / dirs
            for bi, b in enumerate(self.boxes):
                lo, hi = np.asarray(b.lo), np.asarray(b.hi)
                t0 = (lo - origin) * inv
                t1 = (hi - origin) * inv
                tmin = np.minimum(t0, t1)
                tmax = np.maximum(t0, t1)
                if b.inside:
                    tmax = np.where(np.isnan(tmax), np.inf, tmax)
                    t = np.min(tmax, axis=1)
                    axis = np.argmin(tmax, axis=1)
                    side = (dirs[np.arange(n), axis] > 0).astype(np.int64)  # 1 -> hit the hi face
                    hit = np.isfinite(t) & (t > 0)
                else:
                    tmin = np.where(np.isnan(tmin), -np.inf, tmin)
                    tmax = np.where(np.isnan(tmax), np.inf, tmax)
                    tn = np.max(tmin, axis=1)
                    tf = np.min(tmax, axis=1)
                    axis = np.argmax(tmin, axis=1)
                    side = (dirs[np.arange(n), axis] < 0).astype(np.int64)  # entered through hi face
                    hit = (tn <= tf) & (tn > 1e-6)
                    t = tn
                better = hit & (t < best_t)
                best_t = np.where(better, t, best_t)
                best_box = np.where(better, bi, best_box)
                best_axis = np.where(better, axis, best_axis)
                best_side = np.where(better, side, best_side)
        P = origin + dirs * best_t[:, None]
        rgb = np.zeros((n, 3), np.float32)
        for bi, b in enumerate(self.boxes):
            sel = np.where(best_box == bi)[0]
            if len(sel) == 0:
                continue
            ax = best_axis[sel]
            p = P[sel]
            u = np.where(ax == 0, p[:, 1], p[:, 0])
            v = np.where(ax == 2, p[:, 1], p[:, 2])
            face = ax * 2 + best_side[sel]
            col = self._texture(b, face, u, v, ax)
            # baked diffuse lighting from two ceiling lights (view independent)
            nrm = np.zeros((len(sel), 3))
            sign = np.where(best_side[sel] == 1, 1.0, -1.0)
            if b.inside:
                sign = -sign                                 # room normals point inwards
            nrm[np.arange(len(sel)), ax] = sign
            shade = np.full(len(sel), 0.45)
            for L in LIGHTS:
                d = L - p
                dist = np.linalg.norm(d, axis=1)
                shade += 0.55 * np.clip((nrm * d).sum(1) / dist, 0, 1) / (1 + 0.08 * dist ** 2) * 1.3
            rgb[sel] = col * shade[:, None]
        return np.clip(rgb, 0, 1), best_t

    def render(self, R_wc, p_wc, width, height, fx, fy, cx, cy):
        """RGB (H, W, 3) float32 in [0,1] and z-depth (H, W) for a CV camera (z forward)."""
        vv, uu = np.mgrid[0:height, 0:width].astype(np.float64)
        d_c = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], -1).reshape(-1, 3)
        d_w = d_c @ np.asarray(R_wc).T
        rgb, t = self.cast(np.asarray(p_wc, float), d_w)
        return rgb.reshape(height, width, 3), t.reshape(height, width).astype(np.float32)  # t = z because d_c.z = 1


def downsample(img: np.ndarray, k: int = 2) -> np.ndarray:
    """k x k box filter (consistent with f/k and cx' = (cx + 0.5)/k - 0.5)."""
    if k == 1:
        return img
    H, W = img.shape[:2]
    return img[: H // k * k, : W // k * k].reshape(H // k, k, W // k, k, *img.shape[2:]).mean((1, 3))


def to_gray_u8(rgb: np.ndarray) -> np.ndarray:
    return np.clip((0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) * 255, 0, 255).astype(np.uint8)
