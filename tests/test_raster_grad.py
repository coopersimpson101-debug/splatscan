"""The hand-derived backward pass vs PyTorch autograd (dense reference) and finite differences."""
import math

import pytest
import torch

from splatscan.raster import Camera, quat_to_rotmat, rasterize, rasterize_reference
from splatscan.sh import num_coeffs, sh_basis_and_grad


def _scene(n=40, degree=1, seed=0, dtype=torch.float64, spread=0.6):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.rand(*s, generator=g, dtype=dtype)  # noqa: E731
    means = torch.stack([(r(n) - 0.5) * 2 * spread, (r(n) - 0.5) * 1.5 * spread, 2.0 + 1.5 * r(n)], 1)
    log_scales = torch.log(0.04 + 0.12 * r(n, 3))
    quats = r(n, 4) - 0.5
    logits = (r(n) - 0.3) * 4
    sh = (r(n, 16, 3) - 0.5) * 0.8
    sh[:, 0] += 0.6
    return [means, log_scales, quats, logits, sh]


def _cam(W=37, H=29, dtype=torch.float64, yaw=0.15):
    c, s = math.cos(yaw), math.sin(yaw)
    R_wc = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=dtype)
    return Camera.from_pose(R_wc, torch.tensor([0.1, -0.05, -0.2], dtype=dtype), W, H,
                            30.0, 31.0, W / 2 - 0.5, H / 2 - 0.5, dtype=dtype)


def _loss_weights(H, W, dtype, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(H, W, 3, generator=g, dtype=dtype), torch.randn(H, W, generator=g, dtype=dtype)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("tile", [4, 8, 16])
def test_forward_and_grads_match_autograd_reference(degree, tile):
    params = [p.requires_grad_() for p in _scene(degree=degree, seed=degree + tile)]
    bg = torch.tensor([0.2, 0.3, 0.1], dtype=torch.float64, requires_grad=True)
    cam = _cam()
    wc, wd = _loss_weights(cam.height, cam.width, torch.float64)
    rgb, depth, alpha = rasterize(*params, cam, bg, degree=degree, tile=tile)
    rgb_r, depth_r, alpha_r = rasterize_reference(*params, cam, bg, degree=degree, tile=tile)
    assert torch.allclose(rgb, rgb_r, atol=1e-10)
    assert torch.allclose(depth, depth_r, atol=1e-10)
    assert torch.allclose(alpha, alpha_r, atol=1e-10)
    assert alpha.max() > 0.5, "scene should actually cover the image"
    L = (rgb * wc).sum() + 0.3 * (depth * wd).sum()
    Lr = (rgb_r * wc).sum() + 0.3 * (depth_r * wd).sum()
    g = torch.autograd.grad(L, params + [bg])
    gr = torch.autograd.grad(Lr, params + [bg])
    for name, a, b in zip(["means", "log_scales", "quats", "logits", "sh", "bg"], g, gr):
        assert torch.allclose(a, b, rtol=1e-7, atol=1e-9), \
            f"{name}: max abs diff {(a - b).abs().max():.3e} (scale {b.abs().max():.3e})"


def test_gradcheck_finite_differences():
    """Central finite differences on the custom autograd.Function (float64)."""
    params = _scene(n=12, degree=2, seed=7)
    cam = _cam(W=20, H=16)
    bg = torch.tensor([0.1, 0.2, 0.3], dtype=torch.float64)
    wc, wd = _loss_weights(cam.height, cam.width, torch.float64, seed=3)

    def f(*p):
        rgb, depth, _ = rasterize(*p, cam, bg, degree=2, tile=8)
        return (rgb * wc).sum() + 0.3 * (depth * wd).sum()

    ok = torch.autograd.gradcheck(f, [p.clone().requires_grad_() for p in params],
                                  eps=1e-6, atol=1e-5, rtol=1e-4)
    assert ok


def test_x_clamp_branch_gradient():
    """A splat far off-axis exercises the J-clamping branch (|x/z| > 1.3 tan(fov/2))."""
    dtype = torch.float64
    means = torch.tensor([[1.9, 0.0, 1.0], [0.0, 0.0, 2.0]], dtype=dtype)
    log_scales = torch.log(torch.tensor([[0.6, 0.3, 0.2], [0.2, 0.2, 0.2]], dtype=dtype))
    quats = torch.tensor([[0.9, 0.1, 0.3, 0.2], [1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    logits = torch.tensor([1.0, 0.5], dtype=dtype)
    sh = torch.zeros(2, 1, 3, dtype=dtype)
    sh[:, 0] = 1.0
    cam = Camera.from_pose(torch.eye(3, dtype=dtype), torch.zeros(3, dtype=dtype), 24, 20, 12.0, 12.0,
                           11.5, 9.5, dtype=dtype)
    wc, wd = _loss_weights(20, 24, dtype, seed=4)

    def f(*p):
        rgb, depth, _ = rasterize(*p, cam, None, degree=0, tile=8)
        return (rgb * wc).sum() + (depth * wd).sum()

    p = [t.clone().requires_grad_() for t in (means, log_scales, quats, logits, sh)]
    x_over_z = means[0, 0] / means[0, 2]
    assert x_over_z > 1.3 * 0.5 * 24 / 12.0
    assert torch.autograd.gradcheck(f, p, eps=1e-6, atol=1e-5, rtol=1e-4)


def test_sh_jacobian_matches_autograd():
    d = torch.nn.functional.normalize(torch.randn(50, 3, dtype=torch.float64), dim=1).requires_grad_()
    Y, dY = sh_basis_and_grad(3, d)
    for k in range(num_coeffs(3)):
        (gk,) = torch.autograd.grad(Y[:, k].sum(), d, retain_graph=True, allow_unused=True)
        gk = torch.zeros_like(d) if gk is None else gk
        assert torch.allclose(gk, dY[:, k], atol=1e-12), k


def test_quat_to_rotmat_is_rotation():
    q = torch.nn.functional.normalize(torch.randn(100, 4, dtype=torch.float64), dim=1)
    R = quat_to_rotmat(q)
    eye = torch.eye(3, dtype=torch.float64).expand(100, 3, 3)
    assert torch.allclose(R @ R.transpose(1, 2), eye, atol=1e-12)
    assert torch.allclose(torch.linalg.det(R), torch.ones(100, dtype=torch.float64), atol=1e-12)


def test_float32_matches_float64():
    params64 = _scene(n=200, degree=1, seed=11)
    params32 = [p.float() for p in params64]
    rgb64, d64, _ = rasterize(*params64, _cam(64, 48), degree=1)
    rgb32, d32, _ = rasterize(*params32, _cam(64, 48, dtype=torch.float32), degree=1)
    assert (rgb64 - rgb32.double()).abs().max() < 2e-4
    assert (d64 - d32.double()).abs().max() < 2e-3


def test_stats_report_radii_and_grad2d():
    params = [p.requires_grad_() for p in _scene(n=30, degree=0, seed=2)]
    stats = {}
    rgb, _, _ = rasterize(*params, _cam(), degree=0, stats=stats)
    rgb.sum().backward()
    assert stats["radii"].shape == (30,) and stats["n_pairs"] > 0
    assert stats["grad2d"].shape == (30,) and torch.isfinite(stats["grad2d"]).all()
    assert (stats["grad2d"][stats["radii"] > 0] > 0).any()


def test_empty_and_behind_camera():
    params = _scene(n=5, degree=0)
    params[0][:, 2] = -1.0                       # all behind the camera
    params = [p.requires_grad_() for p in params]
    bg = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float64)
    rgb, depth, alpha = rasterize(*params, _cam(), bg, degree=0)
    assert torch.allclose(rgb, bg.expand_as(rgb)) and depth.abs().max() == 0 and alpha.max() == 0
    rgb.sum().backward()
    assert params[0].grad.abs().max() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device on this machine")
def test_cuda_matches_cpu():
    params = _scene(n=300, degree=1, seed=5, dtype=torch.float32)
    cam = _cam(64, 48, dtype=torch.float32)
    outs = []
    grads = []
    for dev in ["cpu", "cuda"]:
        p = [t.detach().clone().to(dev).requires_grad_() for t in params]
        c = Camera(cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, cam.R_cw.to(dev), cam.t_cw.to(dev))
        rgb, depth, _ = rasterize(*p, c, degree=1)
        (rgb.sum() + depth.sum()).backward()
        outs.append(rgb.detach().cpu())
        grads.append([t.grad.cpu() for t in p])
    assert (outs[0] - outs[1]).abs().max() < 1e-4
    for a, b in zip(*grads):
        assert (a - b).abs().max() <= 1e-3 * max(1.0, b.abs().max().item())
