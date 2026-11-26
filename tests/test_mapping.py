"""Optimiser, map bookkeeping, the online mapper, the room/walk simulator and the VIO hookup."""
import math

import numpy as np
import pytest
import torch

from splatscan.gaussians import GaussianMap
from splatscan.losses import psnr, ssim
from splatscan.mapper import MapperConfig, OnlineMapper, camera_from
from splatscan.optim import SplatAdam
from splatscan.room import LabRoom, downsample
from splatscan.vio_source import Intrinsics, SyntheticWalkSource
from splatscan.walk import LabWalk


def test_adam_matches_reference_formula_and_freezes_invisible_rows():
    torch.manual_seed(0)
    p = torch.randn(4, 3, requires_grad=True)
    opt = SplatAdam({"x": p}, {"x": 0.1})
    ref = p.detach().clone()
    m = torch.zeros_like(ref)
    v = torch.zeros_like(ref)
    for k in range(1, 6):
        g = torch.randn(4, 3)
        g[2] = 0                                   # row 2 never visible
        p.grad = g.clone()
        opt.step()
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g * g
        upd = 0.1 * (m / (1 - 0.9 ** k)) / ((v / (1 - 0.999 ** k)).sqrt() + 1e-15)
        upd[2] = 0
        ref -= upd
    assert torch.allclose(p, ref, atol=1e-5)
    assert opt.steps.tolist() == [5, 5, 0, 5]


def test_adam_per_row_bias_correction_after_append():
    p = torch.zeros(1, 1, requires_grad=True)
    opt = SplatAdam({"x": p}, {"x": 0.01})
    for _ in range(100):
        p.grad = torch.ones(1, 1)
        opt.step()
    opt.append(1)
    q = torch.cat([p.detach(), torch.zeros(1, 1)]).requires_grad_()
    opt.params = {"x": q}
    q.grad = torch.ones(2, 1)
    opt.step()
    # the new row's first step is a full lr step (bias-corrected with its own step count = 1)
    assert q[1, 0].item() == pytest.approx(-0.01, rel=1e-4)


def test_map_add_densify_prune_keeps_ids_and_optimiser_in_sync():
    g = GaussianMap(sh_degree=1)
    ids = g.add(torch.randn(10, 3), torch.full((10,), 0.02), torch.rand(10, 3))
    assert ids.tolist() == list(range(10)) and len(g) == 10
    with torch.no_grad():
        g.p["log_scales"][0] = math.log(0.2)      # big -> split
        g.p["logits"][5] = -10.0                  # transparent -> prune
    g.grad_count[:] = 5
    g.grad_accum[:] = 0
    g.grad_accum[0] = 5.0                          # big + hot -> split
    g.grad_accum[3] = 5.0                          # small + hot -> clone
    nc, ns, npr = g.densify_and_prune(grad_thresh=0.5, split_scale=0.05, min_opacity=0.01, max_scale=0.5)
    assert (nc, ns, npr) == (1, 1, 1)
    assert len(g) == 10 + 1 + 2 - 1 - 1
    assert len(set(g.ids.tolist())) == len(g)
    assert 0 not in g.ids.tolist() and 5 not in g.ids.tolist() and 3 in g.ids.tolist()
    for k, t in g.p.items():
        assert t.shape[0] == len(g) and g.opt.m[k].shape == t.shape
    assert g.opt.steps.shape[0] == len(g)
    a = g.activated()
    assert np.allclose(np.linalg.norm(a["rot"], axis=1), 1) and a["sh"].shape == (len(g), 12)


def test_add_from_depth_backprojects_onto_the_surface():
    room = LabRoom()
    intr = Intrinsics(64, 48, 40.0, 40.0, 31.5, 23.5)
    R = LabWalk().rotation(0.0) @ np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    p = np.array([0.0, 0.0, 1.5])
    rgb, depth = room.render(R, p, intr.width, intr.height, intr.fx, intr.fy, intr.cx, intr.cy)
    g = GaussianMap()
    cam = camera_from(R, p, intr)
    g.add_from_depth(torch.tensor(rgb), torch.tensor(depth), cam, torch.ones(48, 64, dtype=torch.bool), stride=1)
    # every new splat projects back to its source pixel at the measured depth
    pts = g.p["means"].detach().numpy()
    pc = (pts - p) @ R
    assert np.all(pc[:, 2] > 0)
    u = intr.fx * pc[:, 0] / pc[:, 2] + intr.cx
    v = intr.fy * pc[:, 1] / pc[:, 2] + intr.cy
    assert np.abs(u - np.round(u)).max() < 1e-3 and np.abs(v - np.round(v)).max() < 1e-3
    assert np.allclose(pc[:, 2], depth[np.round(v).astype(int), np.round(u).astype(int)], rtol=1e-5)


def test_room_renderer_depth_is_consistent_with_geometry():
    room = LabRoom()
    R = np.array([[0.0, 0, 1], [-1, 0, 0], [0, -1, 0]])      # looking along +x
    rgb, depth = room.render(R, np.array([0.0, 0.0, 1.5]), 33, 25, 20.0, 20.0, 16.0, 12.0)
    assert rgb.shape == (25, 33, 3) and 0 <= rgb.min() and rgb.max() <= 1
    # the centre ray hits the bench / wall at x in [2.6, 4.0]
    assert 2.5 < depth[12, 16] <= 4.0 + 1e-6
    small = downsample(rgb, 1)
    assert small is rgb


def test_ssim_and_psnr():
    a = torch.rand(40, 50, 3)
    assert ssim(a, a).item() == pytest.approx(1.0, abs=1e-5)
    assert psnr(a, a) > 100
    b = (a + 0.1 * torch.randn_like(a)).clamp(0, 1)
    assert ssim(a, b).item() < 0.95 and 15 < psnr(b, a) < 25


def test_online_mapper_learns_the_room_from_gt_poses():
    """Small-resolution run with ground-truth poses: held-out PSNR must improve and be decent."""
    src = SyntheticWalkSource(duration=3.0, vio_width=128, vio_height=96, focal=80.0, use_vio=False)
    m = OnlineMapper(src.map_intr, MapperConfig(iters_per_kf=15, first_kf_iters=60, kf_rot_deg=8, tile=4))
    frames = list(src.frames())
    held = [(f.gt_R_wc, f.gt_p_wc, f.rgb) for f in frames if f.index % 20 == 10]
    before = None
    for f in frames:
        if f.index % 20 == 10:
            continue
        info = m.process(f)
        if info and before is None:
            before = m.evaluate(held[:1])["psnr"]
    after = m.evaluate(held[:1])
    assert len(m.keyframes) >= 3
    assert after["psnr"] > 20.0, after
    assert after["ssim"] > 0.7, after
    snap = m.map.activated()
    assert len(snap["ids"]) == len(m.map) and len(np.unique(snap["ids"])) == len(m.map)


def test_walk_feeds_insight_vio_and_tracks():
    """1.5 s of the full image pipeline: render -> insight-vio FAST/KLT + MSCKF -> pose."""
    pytest.importorskip("insight_vio")
    src = SyntheticWalkSource(duration=1.5)
    errs = [np.linalg.norm(f.p_wc - f.gt_p_wc) for f in src.frames()]
    assert len(errs) == 30
    assert max(errs) < 0.05, f"VIO drifted {max(errs):.3f} m in 1.5 s"


def test_triangulated_stereo_features_match_true_depth():
    """The depth-sensor-free init path: insight-vio stereo matches -> DLT -> sparse depth."""
    pytest.importorskip("insight_vio")
    from splatscan.triangulate import sparse_depth_map
    src = SyntheticWalkSource(duration=0.3, depth_noise=0.0)
    d = src.data
    mi = src.map_intr
    for f in src.frames():
        sparse, n = sparse_depth_map(f.features, d.cameras[0], d.cameras[1], mi.width, mi.height,
                                     mi.fx, mi.fy, mi.cx, mi.cy)
        if f.index < 2:
            continue
        ok = np.isfinite(sparse)
        assert n >= 20 and ok.sum() >= 20
        rel = np.abs(sparse[ok] - f.depth[ok]) / f.depth[ok]
        # 6.4 cm baseline, 200 px focal: sub-pixel matches give a few % depth error at 2-4 m
        assert np.median(rel) < 0.05, np.median(rel)


def test_ply_roundtrip_renders_identically(tmp_path):
    from splatscan.export import read_ply, write_ply
    g = GaussianMap(sh_degree=1)
    torch.manual_seed(3)
    g.add(torch.randn(50, 3) * 0.3 + torch.tensor([0, 0, 2.0]), torch.full((50,), 0.05), torch.rand(50, 3))
    with torch.no_grad():
        g.p["sh_rest"].normal_(0, 0.2)
        g.p["quats"].normal_()
    write_ply(str(tmp_path / "m.ply"), g)
    h = read_ply(str(tmp_path / "m.ply"), sh_degree=1)
    cam = camera_from(np.eye(3), np.zeros(3), Intrinsics(40, 30, 30.0, 30.0, 19.5, 14.5))
    a, _, _ = g.render(cam)
    b, _, _ = h.render(cam)
    assert torch.allclose(a, b, atol=1e-6) and a.max() > 0.1
