"""Differentiable tile-based Gaussian splatting rasteriser with a hand-derived backward pass.

This is a from-scratch PyTorch re-implementation of the forward model of 3D Gaussian
Splatting (Kerbl, Kopanas, Leimkuehler, Drettakis, SIGGRAPH 2023):

1. **Projection.** Each splat ``i`` has a mean ``mu``, log-scales ``s``, a quaternion ``q``,
   an opacity logit and SH colour coefficients. Its 3D covariance is
   ``Sigma = R(q) S S^T R(q)^T`` with ``S = diag(exp s)``. With the world->camera rotation
   ``W`` and the Jacobian ``J`` of the pinhole projection at the camera-space mean, the
   screen-space covariance is the EWA approximation ``Sigma' = J W Sigma W^T J^T + 0.3 I``
   (Zwicker et al. 2001), whose inverse (the *conic*) gives the 2D falloff.
2. **Binning.** The image is cut into ``tile x tile`` pixel tiles. Every splat is duplicated
   once per tile its 3-sigma box touches, and the (tile, depth) pairs are sorted - the same
   "duplicate with keys + sort" scheme as the CUDA reference, done with ``torch.sort``.
3. **Compositing.** Inside a tile every pixel composites the tile's splats front to back,
   ``C = sum_i f_i a_i T_i + T_N bg`` with ``a_i = min(0.99, o_i exp(-m_i/2))``, the
   transmittance ``T_i = prod_{j<i} (1 - a_j)``, the 1/255 alpha cut-off and the
   ``T < 1e-4`` early stop of the reference. ``f_i`` is the SH colour *and* the camera depth,
   so the same pass produces an expected-depth image (used for SplaTAM-style depth loss).
   Transmittance is a segmented exclusive cumulative sum of ``log(1 - a)`` in float64, which
   vectorises the per-pixel loop over all pixels of all tiles at once.

The backward pass (:class:`_RasterizeFn`) is derived by hand, stage by stage, in
``docs/derivation.md``. It is verified against PyTorch autograd on an independent dense
reference (:func:`rasterize_reference`) and against central finite differences in
``tests/test_raster_grad.py``. Everything is device-agnostic (CPU here, CUDA on a Jetson).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .sh import num_coeffs, sh_basis_and_grad

ALPHA_MIN = 1.0 / 255.0
ALPHA_MAX = 0.99
T_MIN = 1e-4
DILATION = 0.3


@dataclass
class Camera:
    """Pinhole camera. Pixel centres are at integer coordinates (``cx = W/2 - 0.5`` is centred).

    ``R_cw``/``t_cw`` map world points into the camera frame (x right, y down, z forward):
    ``p_c = R_cw p_w + t_cw``.
    """
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    R_cw: torch.Tensor
    t_cw: torch.Tensor
    near: float = 0.05

    @staticmethod
    def from_pose(R_wc, p_wc, width, height, fx, fy, cx, cy, near=0.05, dtype=torch.float32,
                  device="cpu") -> "Camera":
        R_wc = torch.as_tensor(R_wc, dtype=dtype, device=device)
        p_wc = torch.as_tensor(p_wc, dtype=dtype, device=device)
        R_cw = R_wc.T.contiguous()
        return Camera(width, height, fx, fy, cx, cy, R_cw, -(R_cw @ p_wc), near)

    @property
    def campos(self) -> torch.Tensor:
        return -(self.R_cw.T @ self.t_cw)

    def to(self, ref: torch.Tensor) -> "Camera":
        return Camera(self.width, self.height, self.fx, self.fy, self.cx, self.cy,
                      self.R_cw.to(ref), self.t_cw.to(ref), self.near)


def quat_to_rotmat(q: torch.Tensor) -> torch.Tensor:
    """Unit quaternions (N, 4) as [w, x, y, z] -> rotation matrices (N, 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], -1).reshape(*q.shape[:-1], 3, 3)


def _quat_rotmat_vjp(q: torch.Tensor, gR: torch.Tensor) -> torch.Tensor:
    """Gradient w.r.t. a unit quaternion given dL/dR (N, 3, 3)."""
    w, x, y, z = q.unbind(-1)
    g = gR.reshape(-1, 9)
    g00, g01, g02, g10, g11, g12, g20, g21, g22 = g.unbind(-1)
    gw = 2 * (-z * g01 + y * g02 + z * g10 - x * g12 - y * g20 + x * g21)
    gx = 2 * (y * g01 + z * g02 + y * g10 - 2 * x * g11 - w * g12 + z * g20 + w * g21 - 2 * x * g22)
    gy = 2 * (-2 * y * g00 + x * g01 + w * g02 + x * g10 + z * g12 - w * g20 + z * g21 - 2 * y * g22)
    gz = 2 * (-2 * z * g00 - w * g01 + x * g02 + w * g10 - 2 * z * g11 + y * g12 + x * g20 + y * g21)
    return torch.stack([gw, gx, gy, gz], -1)


# --------------------------------------------------------------------------- forward pieces
def _preprocess(means, log_scales, quats, opac_logits, sh, cam: Camera, degree: int):
    R, t = cam.R_cw, cam.t_cw
    pc = means @ R.T + t
    x, y, z = pc.unbind(1)
    valid = z > cam.near
    one = torch.ones_like(z)
    zs = torch.where(valid, z, one)
    limx = 1.3 * 0.5 * cam.width / cam.fx
    limy = 1.3 * 0.5 * cam.height / cam.fy
    xz, yz = x / zs, y / zs
    mx = (xz > -limx) & (xz < limx)
    my = (yz > -limy) & (yz < limy)
    tx = xz.clamp(-limx, limx) * zs
    ty = yz.clamp(-limy, limy) * zs
    zero = torch.zeros_like(z)
    J = torch.stack([cam.fx / zs, zero, -cam.fx * tx / zs ** 2,
                     zero, cam.fy / zs, -cam.fy * ty / zs ** 2], -1).reshape(-1, 2, 3)
    Tm = J @ R                                             # (N, 2, 3)
    qn = quats.norm(dim=1, keepdim=True)
    qh = quats / qn
    Rq = quat_to_rotmat(qh)
    s = log_scales.exp()
    M = Rq * s[:, None, :]
    S3 = M @ M.transpose(1, 2)
    S2 = Tm @ S3 @ Tm.transpose(1, 2)
    A = S2[:, 0, 0] + DILATION
    B = S2[:, 0, 1]
    C = S2[:, 1, 1] + DILATION
    det = A * C - B * B
    valid = valid & (det > 0)
    dets = torch.where(valid, det, one)
    conic = torch.stack([C / dets, -B / dets, A / dets], -1)
    u = cam.fx * x / zs + cam.cx
    v = cam.fy * y / zs + cam.cy
    mid = 0.5 * (A + C)
    lam = mid + torch.sqrt((mid * mid - dets).clamp_min(0.1))
    radius = torch.ceil(3.0 * torch.sqrt(lam))
    # view-dependent colour
    dvec = means - cam.campos
    dn = dvec.norm(dim=1, keepdim=True).clamp_min(1e-12)
    dirs = dvec / dn
    Y, dY = sh_basis_and_grad(degree, dirs)
    K = num_coeffs(degree)
    rgb = torch.einsum("nk,nkc->nc", Y, sh[:, :K]) + 0.5
    cmask = rgb > 0
    rgb = rgb.clamp_min(0)
    o = torch.sigmoid(opac_logits)
    return dict(pc=pc, zs=zs, valid=valid, mx=mx, my=my, tx=tx, ty=ty, J=J, Tm=Tm, qn=qn, qh=qh,
                Rq=Rq, s=s, M=M, S3=S3, A=A, B=B, C=C, det=dets, conic=conic, u=u, v=v,
                radius=radius, dn=dn, dirs=dirs, Y=Y, dY=dY, rgb=rgb, cmask=cmask, o=o)


def _bin(u, v, radius, valid, z, W, H, tile):
    """Duplicate splats per overlapped tile and sort by (tile, depth)."""
    TX, TY = (W + tile - 1) // tile, (H + tile - 1) // tile
    dev = u.device
    x0 = ((u - radius) / tile).floor().clamp(0, TX).long()
    x1 = ((u + radius) / tile).floor().add(1).clamp(0, TX).long()
    y0 = ((v - radius) / tile).floor().clamp(0, TY).long()
    y1 = ((v + radius) / tile).floor().add(1).clamp(0, TY).long()
    w = (x1 - x0).clamp_min(0)
    n = w * (y1 - y0).clamp_min(0) * valid.long()
    N = u.shape[0]
    gid = torch.repeat_interleave(torch.arange(N, device=dev), n)
    if gid.numel() == 0:
        return gid, gid, torch.zeros(TX * TY + 1, dtype=torch.long, device=dev), TX, TY
    start = torch.cumsum(n, 0) - n
    k = torch.arange(gid.numel(), device=dev) - start[gid]
    wg = w[gid]
    tile_id = (y0[gid] + k // wg) * TX + (x0[gid] + k % wg)
    rank = torch.empty(N, dtype=torch.long, device=dev)
    rank[torch.argsort(z)] = torch.arange(N, device=dev)
    key = tile_id * N + rank[gid]
    order = torch.argsort(key)
    gid, tile_id = gid[order], tile_id[order]
    counts = torch.bincount(tile_id, minlength=TX * TY)
    tile_start = torch.zeros(TX * TY + 1, dtype=torch.long, device=dev)
    tile_start[1:] = torch.cumsum(counts, 0)
    return gid, tile_id, tile_start, TX, TY


def _seg_cumsum_excl(x: torch.Tensor, seg_first: torch.Tensor) -> torch.Tensor:
    """Exclusive cumulative sum along dim 0 that restarts at each segment's first row."""
    c = torch.cumsum(x, 0)
    excl = c - x
    return excl - excl[seg_first]


def _pixel_grid(tile_id, TX, tile, W, H):
    lx = torch.arange(tile * tile, device=tile_id.device) % tile
    ly = torch.arange(tile * tile, device=tile_id.device) // tile
    px = (tile_id % TX)[:, None] * tile + lx[None, :]
    py = (tile_id // TX)[:, None] * tile + ly[None, :]
    inside = (px < W) & (py < H)
    return px, py, inside


def _tiles_to_image(t, TX, TY, tile, W, H):
    F = t.shape[-1]
    img = t.reshape(TY, TX, tile, tile, F).permute(0, 2, 1, 3, 4).reshape(TY * tile, TX * tile, F)
    return img[:H, :W]


def _image_to_tiles(img, TX, TY, tile):
    H, W, F = img.shape
    pad = img.new_zeros(TY * tile, TX * tile, F)
    pad[:H, :W] = img
    return pad.reshape(TY, tile, TX, tile, F).permute(0, 2, 1, 3, 4).reshape(TY * TX, tile * tile, F)


class _RasterizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means, log_scales, quats, opac_logits, sh, bg, cam, degree, tile, stats):
        W, H = cam.width, cam.height
        pre = _preprocess(means, log_scales, quats, opac_logits, sh, cam, degree)
        u, v, conic, o = pre["u"], pre["v"], pre["conic"], pre["o"]
        gid, tile_id, tile_start, TX, TY = _bin(u, v, pre["radius"], pre["valid"], pre["zs"], W, H, tile)
        feat = torch.cat([pre["rgb"], pre["zs"][:, None]], 1)          # (N, 4): rgb + depth
        bgf = torch.cat([bg, bg.new_zeros(1)])
        n_tiles = TX * TY
        BB = tile * tile
        if gid.numel() > 0:
            px, py, inside = _pixel_grid(tile_id, TX, tile, W, H)
            dx = px - u[gid, None]
            dy = py - v[gid, None]
            a, b, c = conic[gid, 0, None], conic[gid, 1, None], conic[gid, 2, None]
            power = -0.5 * (a * dx * dx + c * dy * dy) - b * dx * dy
            G = torch.exp(power.clamp_max(0))
            raw = o[gid, None] * G
            alpha = raw.clamp_max(ALPHA_MAX)
            keep = inside & (power <= 0) & (alpha >= ALPHA_MIN)
            alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
            log1m = torch.log1p(-alpha.double())
            seg_first = tile_start[tile_id]
            excl = _seg_cumsum_excl(log1m, seg_first)
            keep = keep & ((excl + log1m) >= math.log(T_MIN))     # reference early termination
            alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
            log1m = torch.where(keep, log1m, torch.zeros_like(log1m))
            Ti = torch.exp(excl).to(alpha.dtype)
            wgt = alpha * Ti
            acc = means.new_zeros(n_tiles, BB, 4)
            acc.index_add_(0, tile_id, wgt[:, :, None] * feat[gid, None, :])
            logT = torch.zeros(n_tiles, BB, dtype=torch.float64, device=means.device)
            logT.index_add_(0, tile_id, log1m)
            Tfin = torch.exp(logT).to(means.dtype)
        else:
            acc = means.new_zeros(n_tiles, BB, 4)
            Tfin = means.new_ones(n_tiles, BB)
            px = py = alpha = Ti = keep = G = None
        out = acc + Tfin[:, :, None] * bgf
        img = _tiles_to_image(out, TX, TY, tile, W, H)
        alpha_img = 1 - _tiles_to_image(Tfin[:, :, None], TX, TY, tile, W, H)[..., 0]
        ctx.cam, ctx.degree, ctx.tile, ctx.stats = cam, degree, tile, stats
        ctx.geom = (TX, TY)
        ctx.pre = pre
        ctx.pairs = (gid, tile_id, tile_start, alpha, Ti, keep, G, px, py, Tfin)
        ctx.save_for_backward(means, sh, bgf, feat)
        if stats is not None:
            stats["radii"] = torch.where(pre["valid"], pre["radius"], torch.zeros_like(pre["radius"]))
            stats["n_pairs"] = int(gid.numel())
        ctx.mark_non_differentiable(alpha_img)
        return img[..., :3].contiguous(), img[..., 3].contiguous(), alpha_img.contiguous()

    @staticmethod
    def backward(ctx, g_rgb, g_depth, _g_alpha):
        means, sh, bgf, feat = ctx.saved_tensors
        pre, cam, degree, tile = ctx.pre, ctx.cam, ctx.degree, ctx.tile
        gid, tile_id, tile_start, alpha, Ti, keep, G, px, py, Tfin = ctx.pairs
        TX, TY = ctx.geom
        N = means.shape[0]
        dt = means.dtype
        g_feat = means.new_zeros(N, 4)
        g_u = means.new_zeros(N)
        g_v = means.new_zeros(N)
        g_conic = means.new_zeros(N, 3)
        g_o = means.new_zeros(N)
        g_bg = None
        gimg = torch.cat([g_rgb, g_depth[..., None]], -1)
        gt = _image_to_tiles(gimg, TX, TY, tile)                  # (tiles, BB, 4)
        if ctx.needs_input_grad[5]:
            g_bg = (gt[..., :3] * Tfin[:, :, None]).sum((0, 1))
        if gid is not None and gid.numel() > 0:
            gp = gt[tile_id]                                        # (P, BB, 4)
            wgt = alpha * Ti
            fg = (feat[gid, None, :] * gp).sum(-1)                  # f_i . g   (P, BB)
            g_feat.index_add_(0, gid, (wgt[:, :, None] * gp).sum(1))
            # tail_i = sum_{j>i} w_j f_j.g + T_N bg.g  = total - inclusive prefix
            s = (wgt * fg).double()
            seg_first = tile_start[tile_id]
            incl = _seg_cumsum_excl(s, seg_first) + s
            bg_term = (Tfin * (gt * bgf).sum(-1)).double()         # per tile pixel
            total = torch.zeros_like(bg_term)
            total.index_add_(0, tile_id, s)
            total = total + bg_term
            tail = (total[tile_id] - incl).to(dt)
            g_alpha = Ti * fg - tail / (1 - alpha)
            g_alpha = torch.where(keep, g_alpha, torch.zeros_like(g_alpha))
            unclamped = keep & (pre["o"][gid, None] * G < ALPHA_MAX)
            g_raw = torch.where(unclamped, g_alpha, torch.zeros_like(g_alpha))   # d/d(o*G)
            o_g = pre["o"][gid, None]
            g_o.index_add_(0, gid, (g_raw * G).sum(1))
            g_pow = g_raw * o_g * G
            u, v, conic = pre["u"], pre["v"], pre["conic"]
            dx = px - u[gid, None]
            dy = py - v[gid, None]
            a, b, c = conic[gid, 0, None], conic[gid, 1, None], conic[gid, 2, None]
            g_u.index_add_(0, gid, (g_pow * (a * dx + b * dy)).sum(1))
            g_v.index_add_(0, gid, (g_pow * (b * dx + c * dy)).sum(1))
            g_conic.index_add_(0, gid, torch.stack([(g_pow * (-0.5 * dx * dx)).sum(1),
                                                     (g_pow * (-dx * dy)).sum(1),
                                                     (g_pow * (-0.5 * dy * dy)).sum(1)], -1))
        if ctx.stats is not None:
            ctx.stats["grad2d"] = torch.sqrt(g_u * g_u + g_v * g_v)
        valid = pre["valid"]
        vm = valid.to(dt)
        # ---- conic -> 2D covariance: Q = S2^-1, dL/dS2 = -Q Gq Q (symmetric matrices)
        Qa, Qb, Qc = pre["conic"].unbind(1)
        Gq = torch.stack([g_conic[:, 0], 0.5 * g_conic[:, 1], 0.5 * g_conic[:, 1], g_conic[:, 2]], -1).reshape(-1, 2, 2)
        Q = torch.stack([Qa, Qb, Qb, Qc], -1).reshape(-1, 2, 2)
        GS2 = -(Q @ Gq @ Q) * vm[:, None, None]
        Tm, S3 = pre["Tm"], pre["S3"]
        GS3 = Tm.transpose(1, 2) @ GS2 @ Tm
        GT = 2 * GS2 @ Tm @ S3
        R = cam.R_cw
        GJ = GT @ R.T
        zs, tx, ty = pre["zs"], pre["tx"], pre["ty"]
        x, y = pre["pc"][:, 0], pre["pc"][:, 1]
        fx, fy = cam.fx, cam.fy
        mx, my = pre["mx"].to(dt), pre["my"].to(dt)
        g_pc = means.new_zeros(N, 3)
        g_pc[:, 0] += mx * (-fx / zs ** 2) * GJ[:, 0, 2]
        g_pc[:, 1] += my * (-fy / zs ** 2) * GJ[:, 1, 2]
        g_pc[:, 2] += (-fx / zs ** 2) * GJ[:, 0, 0] + (-fy / zs ** 2) * GJ[:, 1, 1] \
            + (1 + mx) * fx * tx / zs ** 3 * GJ[:, 0, 2] + (1 + my) * fy * ty / zs ** 3 * GJ[:, 1, 2]
        # ---- screen-space mean
        g_pc[:, 0] += fx / zs * g_u
        g_pc[:, 1] += fy / zs * g_v
        g_pc[:, 2] += -fx * x / zs ** 2 * g_u - fy * y / zs ** 2 * g_v
        # ---- depth feature
        g_pc[:, 2] += g_feat[:, 3]
        g_pc = g_pc * vm[:, None]
        g_means = g_pc @ R
        # ---- 3D covariance -> scale / rotation
        M, Rq, s = pre["M"], pre["Rq"], pre["s"]
        GM = 2 * GS3 @ M
        g_s = (Rq * GM).sum(1)
        g_logs = g_s * s
        GRq = GM * s[:, None, :]
        gqh = _quat_rotmat_vjp(pre["qh"], GRq)
        qh, qn = pre["qh"], pre["qn"]
        g_q = (gqh - qh * (qh * gqh).sum(1, keepdim=True)) / qn
        # ---- opacity
        o = pre["o"]
        g_logit = g_o * o * (1 - o)
        # ---- SH colour (+ view-direction gradient into the means)
        g_rgb_s = g_feat[:, :3] * pre["cmask"].to(dt)
        K = num_coeffs(degree)
        g_sh = torch.zeros_like(sh)
        g_sh[:, :K] = pre["Y"][:, :, None] * g_rgb_s[:, None, :]
        if degree > 0:
            gY = (sh[:, :K] * g_rgb_s[:, None, :]).sum(-1)           # (N, K)
            g_dir = (gY[:, :, None] * pre["dY"]).sum(1)
            d = pre["dirs"]
            g_dvec = (g_dir - d * (d * g_dir).sum(1, keepdim=True)) / pre["dn"]
            g_means = g_means + g_dvec
        return g_means, g_logs, g_q, g_logit, g_sh, g_bg, None, None, None, None


def rasterize(means, log_scales, quats, opac_logits, sh, cam: Camera, bg=None, degree: int = 0,
              tile: int = 8, stats: dict | None = None):
    """Render ``(rgb (H,W,3), depth (H,W), alpha (H,W))``. Differentiable in all splat params.

    ``sh`` is (N, K, 3) with K >= (degree+1)^2. ``bg`` is the background colour (3,).
    If ``stats`` is a dict it receives ``radii`` / ``n_pairs`` (forward) and ``grad2d``
    (backward: per-splat |dL/d(u,v)| in pixels, used for densification).
    """
    if bg is None:
        bg = means.new_zeros(3)
    cam = cam.to(means)
    return _RasterizeFn.apply(means, log_scales, quats, opac_logits, sh, bg, cam, degree, tile, stats)


# --------------------------------------------------------------------------- reference
def rasterize_reference(means, log_scales, quats, opac_logits, sh, cam: Camera, bg=None,
                        degree: int = 0, tile: int = 8):
    """Dense, autograd-differentiable reference with identical semantics (tests only).

    Every pixel looks at every splat in depth order (no binning, no segmented sums); the
    tile-coverage mask reproduces which pixels the tiled renderer evaluates. O(pixels x splats).
    """
    if bg is None:
        bg = means.new_zeros(3)
    cam = cam.to(means)
    W, H = cam.width, cam.height
    R, t = cam.R_cw, cam.t_cw
    pc = means @ R.T + t
    x, y, z = pc.unbind(1)
    with torch.no_grad():
        valid = z > cam.near
    zs = torch.where(valid, z, torch.ones_like(z))
    limx = 1.3 * 0.5 * W / cam.fx
    limy = 1.3 * 0.5 * H / cam.fy
    tx = (x / zs).clamp(-limx, limx) * zs
    ty = (y / zs).clamp(-limy, limy) * zs
    zero = torch.zeros_like(z)
    J = torch.stack([cam.fx / zs, zero, -cam.fx * tx / zs ** 2,
                     zero, cam.fy / zs, -cam.fy * ty / zs ** 2], -1).reshape(-1, 2, 3)
    q = quats / quats.norm(dim=1, keepdim=True)
    M = quat_to_rotmat(q) * log_scales.exp()[:, None, :]
    S2 = (J @ R) @ (M @ M.transpose(1, 2)) @ (J @ R).transpose(1, 2)
    S2 = S2 + DILATION * torch.eye(2, dtype=means.dtype, device=means.device)
    det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
    with torch.no_grad():
        valid = valid & (det > 0)
    dets = torch.where(valid, det, torch.ones_like(det))
    a, b, c = S2[:, 1, 1] / dets, -S2[:, 0, 1] / dets, S2[:, 0, 0] / dets
    u = cam.fx * x / zs + cam.cx
    v = cam.fy * y / zs + cam.cy
    dvec = means - cam.campos
    dirs = dvec / dvec.norm(dim=1, keepdim=True)
    Y, _ = sh_basis_and_grad(degree, dirs)
    K = num_coeffs(degree)
    rgb = (torch.einsum("nk,nkc->nc", Y, sh[:, :K]) + 0.5).clamp_min(0)
    o = torch.sigmoid(opac_logits)
    with torch.no_grad():
        A_, C_ = S2[:, 0, 0], S2[:, 1, 1]
        mid = 0.5 * (A_ + C_)
        radius = torch.ceil(3.0 * torch.sqrt(mid + torch.sqrt((mid * mid - dets).clamp_min(0.1))))
        TX, TY = (W + tile - 1) // tile, (H + tile - 1) // tile
        x0 = ((u - radius) / tile).floor().clamp(0, TX)
        x1 = ((u + radius) / tile).floor().add(1).clamp(0, TX)
        y0 = ((v - radius) / tile).floor().clamp(0, TY)
        y1 = ((v + radius) / tile).floor().add(1).clamp(0, TY)
        order = torch.argsort(zs)
    pyy, pxx = torch.meshgrid(torch.arange(H, dtype=means.dtype, device=means.device),
                              torch.arange(W, dtype=means.dtype, device=means.device), indexing="ij")
    pxx, pyy = pxx.reshape(-1, 1), pyy.reshape(-1, 1)
    o_, u_, v_, a_, b_, c_ = (t_[order][None, :] for t_ in (o, u, v, a, b, c))
    dx, dy = pxx - u_, pyy - v_
    power = -0.5 * (a_ * dx * dx + c_ * dy * dy) - b_ * dx * dy
    with torch.no_grad():
        ptx, pty = torch.floor(pxx / tile), torch.floor(pyy / tile)
        cover = (ptx >= x0[order][None]) & (ptx < x1[order][None]) & \
                (pty >= y0[order][None]) & (pty < y1[order][None]) & valid[order][None]
    alpha = (o_ * torch.exp(power.clamp_max(0))).clamp_max(ALPHA_MAX)
    with torch.no_grad():
        keep = cover & (power <= 0) & (alpha >= ALPHA_MIN)
    alpha = torch.where(keep, alpha, torch.zeros_like(alpha))
    T_incl = torch.cumprod(1 - alpha, 1)
    with torch.no_grad():
        keep2 = keep & (T_incl >= T_MIN)
    alpha = torch.where(keep2, alpha, torch.zeros_like(alpha))
    T_incl = torch.cumprod(1 - alpha, 1)
    T_excl = torch.cat([torch.ones_like(T_incl[:, :1]), T_incl[:, :-1]], 1)
    w = alpha * T_excl
    feat = torch.cat([rgb, zs[:, None]], 1)[order]
    out = w @ feat
    Tn = T_incl[:, -1:] if T_incl.shape[1] else torch.ones_like(pxx)
    col = out[:, :3] + Tn * bg
    return col.reshape(H, W, 3), out[:, 3].reshape(H, W), (1 - Tn).reshape(H, W)
