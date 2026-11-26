"""Real spherical harmonics up to degree 3 and their hand-written Jacobians.

Constants and sign convention are those of the reference 3D Gaussian Splatting code
(Kerbl et al. 2023) and of splatweb's ``src/core/sh.ts``, so a splat optimised here renders
with the same colour in the browser:

    colour = max(0, sum_k Y_k(d) * c_k + 0.5),   d = unit vector camera -> splat.

``sh_basis_and_grad`` returns both Y_k(d) and dY_k/dd (for a *unit* d; the projection onto
the unit sphere is handled by the caller), which the rasteriser's backward pass needs to send
colour gradients back into the splat means.
"""
from __future__ import annotations

import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = (1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792,
      0.5462742152960396)
C3 = (-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
      -0.4570457994644658, 1.445305721320277, -0.5900435899266435)


def num_coeffs(degree: int) -> int:
    return (degree + 1) ** 2


def sh_basis_and_grad(degree: int, d: torch.Tensor):
    """Basis ``Y`` (N, K) and Jacobian ``dY`` (N, K, 3) for unit directions ``d`` (N, 3)."""
    if not 0 <= degree <= 3:
        raise ValueError(f"SH degree must be 0..3, got {degree}")
    n = d.shape[0]
    K = num_coeffs(degree)
    Y = d.new_zeros(n, K)
    dY = d.new_zeros(n, K, 3)
    Y[:, 0] = C0
    if degree == 0:
        return Y, dY
    x, y, z = d[:, 0], d[:, 1], d[:, 2]
    Y[:, 1] = -C1 * y
    Y[:, 2] = C1 * z
    Y[:, 3] = -C1 * x
    dY[:, 1, 1] = -C1
    dY[:, 2, 2] = C1
    dY[:, 3, 0] = -C1
    if degree == 1:
        return Y, dY
    xx, yy, zz = x * x, y * y, z * z
    Y[:, 4] = C2[0] * x * y
    dY[:, 4, 0], dY[:, 4, 1] = C2[0] * y, C2[0] * x
    Y[:, 5] = C2[1] * y * z
    dY[:, 5, 1], dY[:, 5, 2] = C2[1] * z, C2[1] * y
    Y[:, 6] = C2[2] * (2 * zz - xx - yy)
    dY[:, 6, 0], dY[:, 6, 1], dY[:, 6, 2] = -2 * C2[2] * x, -2 * C2[2] * y, 4 * C2[2] * z
    Y[:, 7] = C2[3] * x * z
    dY[:, 7, 0], dY[:, 7, 2] = C2[3] * z, C2[3] * x
    Y[:, 8] = C2[4] * (xx - yy)
    dY[:, 8, 0], dY[:, 8, 1] = 2 * C2[4] * x, -2 * C2[4] * y
    if degree == 2:
        return Y, dY
    Y[:, 9] = C3[0] * y * (3 * xx - yy)
    dY[:, 9, 0] = C3[0] * 6 * x * y
    dY[:, 9, 1] = C3[0] * (3 * xx - 3 * yy)
    Y[:, 10] = C3[1] * x * y * z
    dY[:, 10, 0], dY[:, 10, 1], dY[:, 10, 2] = C3[1] * y * z, C3[1] * x * z, C3[1] * x * y
    Y[:, 11] = C3[2] * y * (4 * zz - xx - yy)
    dY[:, 11, 0] = C3[2] * (-2 * x * y)
    dY[:, 11, 1] = C3[2] * (4 * zz - xx - 3 * yy)
    dY[:, 11, 2] = C3[2] * 8 * y * z
    Y[:, 12] = C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
    dY[:, 12, 0] = C3[3] * (-6 * x * z)
    dY[:, 12, 1] = C3[3] * (-6 * y * z)
    dY[:, 12, 2] = C3[3] * (6 * zz - 3 * xx - 3 * yy)
    Y[:, 13] = C3[4] * x * (4 * zz - xx - yy)
    dY[:, 13, 0] = C3[4] * (4 * zz - 3 * xx - yy)
    dY[:, 13, 1] = C3[4] * (-2 * x * y)
    dY[:, 13, 2] = C3[4] * 8 * x * z
    Y[:, 14] = C3[5] * z * (xx - yy)
    dY[:, 14, 0], dY[:, 14, 1], dY[:, 14, 2] = C3[5] * 2 * x * z, -C3[5] * 2 * y * z, C3[5] * (xx - yy)
    Y[:, 15] = C3[6] * x * (xx - 3 * yy)
    dY[:, 15, 0] = C3[6] * (3 * xx - 3 * yy)
    dY[:, 15, 1] = C3[6] * (-6 * x * y)
    return Y, dY


def rgb_to_dc(rgb):
    """DC coefficient that renders as ``rgb`` (0..1) from every direction."""
    return (rgb - 0.5) / C0


def dc_to_rgb(dc):
    return dc * C0 + 0.5
