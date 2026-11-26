"""Sparse depth from insight-vio's stereo feature matches (the depth-sensor-free init path).

insight-vio's front-end returns, per frame, ``{feature_id: [(cam, xy_normalized), ...]}``.
For every feature seen by both cameras of the rig in the *same* frame we triangulate with
the linear (DLT) method in the left camera frame and keep it if it is in front of both
cameras, the reprojection residual is small and the disparity is large enough. The result is
a sparse depth map the mapper can seed splats from (``add_from_depth`` with a sparse mask),
instead of a dense depth sensor.
"""
from __future__ import annotations

import numpy as np


def stereo_extrinsics(cam_left, cam_right):
    """(R_rl, t_rl): maps left-camera coordinates into the right camera."""
    R_rl = cam_right.R_bc.T @ cam_left.R_bc
    t_rl = cam_right.R_bc.T @ (cam_left.p_bc - cam_right.p_bc)
    return R_rl, t_rl


def triangulate_stereo(xl: np.ndarray, xr: np.ndarray, R_rl: np.ndarray, t_rl: np.ndarray,
                       max_residual: float = 3e-3, min_depth: float = 0.2, max_depth: float = 15.0):
    """Linear triangulation of normalized left/right observations (N, 2) -> points (N, 3), valid (N,)."""
    n = len(xl)
    P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = np.hstack([R_rl, t_rl[:, None]])
    X = np.zeros((n, 3))
    for i in range(n):
        A = np.stack([xl[i, 0] * P1[2] - P1[0], xl[i, 1] * P1[2] - P1[1],
                      xr[i, 0] * P2[2] - P2[0], xr[i, 1] * P2[2] - P2[1]])
        _, _, Vt = np.linalg.svd(A)
        h = Vt[-1]
        X[i] = h[:3] / h[3] if abs(h[3]) > 1e-12 else np.nan
    Xr = X @ R_rl.T + t_rl
    with np.errstate(invalid="ignore", divide="ignore"):
        rl = np.linalg.norm(X[:, :2] / X[:, 2:3] - xl, axis=1)
        rr = np.linalg.norm(Xr[:, :2] / Xr[:, 2:3] - xr, axis=1)
        ok = np.isfinite(X).all(1) & (X[:, 2] > min_depth) & (X[:, 2] < max_depth) & (Xr[:, 2] > min_depth)
        ok &= (rl < max_residual) & (rr < max_residual)
    return X, ok


def sparse_depth_map(meas: dict, cam_left, cam_right, width: int, height: int, fx: float, fy: float,
                     cx: float, cy: float):
    """Rasterise triangulated stereo features into an (H, W) depth map (NaN where unknown)."""
    xl, xr = [], []
    for obs in meas.values():
        d = dict(obs)
        if 0 in d and 1 in d:
            xl.append(d[0])
            xr.append(d[1])
    depth = np.full((height, width), np.nan, np.float32)
    if not xl:
        return depth, 0
    R_rl, t_rl = stereo_extrinsics(cam_left, cam_right)
    X, ok = triangulate_stereo(np.asarray(xl, float), np.asarray(xr, float), R_rl, t_rl)
    X = X[ok]
    u = np.round(fx * X[:, 0] / X[:, 2] + cx).astype(int)
    v = np.round(fy * X[:, 1] / X[:, 2] + cy).astype(int)
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    depth[v[inside], u[inside]] = X[inside, 2]
    return depth, int(inside.sum())
