# The backward pass, derived by hand

This is the derivation implemented in `splatscan/raster.py` (`_RasterizeFn.backward`). Every
step is checked in `tests/test_raster_grad.py`, both against PyTorch autograd running through an
independent dense reference renderer (`rasterize_reference`, float64, 12 configurations: SH
degree 0-3 × tile size 4/8/16) and against central finite differences (`torch.autograd.gradcheck`).

Notation: splat `i` has mean `μ`, log-scales `s`, raw quaternion `q`, opacity logit `ℓ` and SH
coefficients `c_k` (RGB each). The camera maps world to camera by `p = W μ + t` (`W = R_cw`),
with focal lengths `f_x, f_y` and principal point `(c_x, c_y)`. `g_X` means `∂L/∂X`. Matrix
gradients follow the convention `dL = tr(G_Xᵀ dX)`.

## 1. Compositing (per pixel)

Splats covering a pixel are visited front to back. With `α_i = min(0.99, o_i G_i)`,
`G_i = exp(−½ m_i)` and `T_i = Π_{j<i} (1 − α_j)`, the pixel value for any per-splat feature
`f_i` (RGB colour **and** camera depth `z_i`, giving an expected-depth image) is

    C = Σ_i f_i α_i T_i + T_N b                       (b = background; 0 for depth)

Splats with `α < 1/255` are skipped and the loop stops before the splat that would push the
transmittance below `10⁻⁴` (exactly as the CUDA reference). Writing `w_i = α_i T_i` and the
incoming pixel gradient `g = ∂L/∂C`:

    g_{f_i}  += w_i g
    g_{α_i}   = T_i (f_i · g) − S_i / (1 − α_i),     S_i = Σ_{j>i} w_j (f_j · g) + T_N (b · g)

because `α_i` enters its own term directly and every later term (and the background) through
the factor `(1 − α_i)` in `T_j`. `S_i` is computed without a per-pixel loop as
`S_i = (Σ_all w_j f_j·g + T_N b·g) − Σ_{j≤i} w_j f_j·g`, using the same segmented prefix sum
as the forward pass. Clamped splats (`o G ≥ 0.99`) and skipped ones get zero gradient.

`g_{o_i} += g_{α_i} G_i` and, with `G = e^P`, `P = −½ m`, `g_{P_i} = g_{α_i} o_i G_i`.

**Vectorisation.** All (splat, tile) pairs are sorted by (tile, depth). For a tile of `B×B`
pixels the pairs form a `(P, B²)` array; the per-pixel exclusive product is
`T = exp(cumsum(log(1 − α)) − value at the tile's first row)`, a *segmented* exclusive
prefix sum along the pair axis, in float64 to keep the running sum exact. The forward pass
therefore never loops over pixels or splats in Python.

## 2. The 2D Gaussian

`m = a dx² + 2 b dx dy + c dy²` with `(dx, dy) = (pixel − (u, v))` and the conic
`Q = [[a, b], [b, c]] = Σ'⁻¹`:

    g_u = Σ_px g_P (a dx + b dy),     g_v = Σ_px g_P (b dx + c dy)
    g_a = Σ_px g_P (−½ dx²),   g_b = Σ_px g_P (−dx dy),   g_c = Σ_px g_P (−½ dy²)

(The sign flips twice: `∂P/∂dx = −(a dx + b dy)` and `∂dx/∂u = −1`.) `|(g_u, g_v)|` is also
accumulated per splat; it is the "view-space positional gradient" that 3DGS densification
thresholds.

## 3. Conic → 2D covariance

With `G_Q = [[g_a, g_b/2], [g_b/2, g_c]]` (the off-diagonal `b` appears twice in `Q`) and
`dQ = −Q dΣ' Q`:

    G_Σ' = −Q G_Q Q

The 0.3 px² low-pass dilation is additive, so it passes the gradient through unchanged.

## 4. EWA projection

`Σ' = T Σ Tᵀ` with `T = J W` and the projection Jacobian evaluated at `p = (x, y, z)`:

    J = [[f_x/z, 0, −f_x t_x/z²], [0, f_y/z, −f_y t_y/z²]],   t_x = z · clamp(x/z, ±1.3 tan(fov_x/2))

Since `G_Σ'` is symmetric:

    G_Σ = Tᵀ G_Σ' T,     G_T = 2 G_Σ' T Σ,     G_J = G_T Wᵀ

and into the camera-space mean (`m_x = 1` if `x/z` is inside the clamp, else 0):

    g_x += m_x (−f_x/z²) G_J[0,2]
    g_z += (−f_x/z²) G_J[0,0] + (−f_y/z²) G_J[1,1] + (1 + m_x) f_x t_x/z³ G_J[0,2] + (1 + m_y) f_y t_y/z³ G_J[1,2]

The `(1 + m_x)` factor is the exact derivative of `−f_x t_x/z²` in both branches: `2 f_x x/z³`
inside the clamp, `f_x t_x/z³` when `t_x = ±k z` is clamped (the CUDA reference uses 2 in both
cases; this implementation is exact, and `test_x_clamp_branch_gradient` exercises the clamped
branch with finite differences).

The screen-space mean `u = f_x x/z + c_x`, `v = f_y y/z + c_y` adds

    g_x += f_x/z g_u,   g_y += f_y/z g_v,   g_z += −f_x x/z² g_u − f_y y/z² g_v

and the depth feature adds `g_z += g_{f_z}`. Finally `g_μ += Wᵀ g_p`.

## 5. Covariance → scale and rotation

`Σ = M Mᵀ`, `M = R(q̂) S`, `S = diag(e^s)`, `q̂ = q/|q|`:

    G_M = 2 G_Σ M,     g_{σ_k} = Σ_r R_rk G_M[r,k],     g_{s_k} = g_{σ_k} e^{s_k},     G_R = G_M S

The rotation-matrix derivatives of the unit quaternion `(w, x, y, z)` are written out in
`_quat_rotmat_vjp` (e.g. `∂R/∂w = 2 [[0, −z, y], [z, 0, −x], [−y, x, 0]]`), and the
normalisation is undone with `g_q = (g_q̂ − q̂ (q̂ · g_q̂)) / |q|`.

## 6. Opacity and colour

`o = σ(ℓ)` ⇒ `g_ℓ = g_o o (1 − o)`.

Colour `c = max(0, Σ_k Y_k(d) c_k + 0.5)` with `d = (μ − camera centre)/|μ − camera centre|`:

    g_{c_k} = Y_k(d) g_rgb · 1[c > 0]
    g_d = Σ_k (c_k · g_rgb) ∂Y_k/∂d,     g_μ += (g_d − d (d · g_d)) / |μ − camera centre|

`∂Y_k/∂d` for all 16 real SH basis functions (degree ≤ 3, the 3DGS constants) is in
`splatscan/sh.py` and checked against autograd in `test_sh_jacobian_matches_autograd`.

## What is *not* differentiated

Tile assignment, the 3σ radius, the `α < 1/255` and `T < 10⁻⁴` cut-offs and the near-plane cull
are discrete decisions; like the reference implementation, they are treated as constants. The
accumulated-opacity image (used for the silhouette test when growing the map) is returned as a
non-differentiable output. Camera poses are inputs, not parameters (poses come from VIO).

## References

* B. Kerbl, G. Kopanas, T. Leimkühler, G. Drettakis. *3D Gaussian Splatting for Real-Time
  Radiance Field Rendering.* ACM TOG (SIGGRAPH) 2023.
* M. Zwicker, H. Pfister, J. van Baar, M. Gross. *EWA Splatting.* IEEE TVCG 2002.
* N. Keetha et al. *SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM.* CVPR 2024.
* H. Matsuki, R. Murai, P. H. J. Kelly, A. J. Davison. *Gaussian Splatting SLAM* (MonoGS). CVPR 2024.
