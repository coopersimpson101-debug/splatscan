"""Pose source: insight-vio's stereo MSCKF fed by the synthetic lab walk.

``SyntheticWalkSource`` yields one :class:`Frame` per camera frame, in time order, exactly
as a live device would: IMU samples up to the frame time are pushed into
``insight_vio.system.VioSystem`` (FAST/KLT front-end + MSCKF back-end, unchanged), the stereo
pair is rendered by :class:`splatscan.room.LabRoom`, and the filter's pose estimate comes
back with the colour image and a depth map for the mapper.

Depth stands in for what a stereo-depth or ToF module would deliver on the device; it is the
renderer's z-buffer plus multiplicative Gaussian noise (``depth_noise``, default 1 %) and is
only used to *initialise* new splats and as a weak depth loss, never for tracking.

On hardware the same interface is fed by ``splatscan.hw`` instead (see deploy/jetson/).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from .room import LabRoom, downsample, to_gray_u8
from .walk import LabWalk


@dataclass
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def scaled(self, k: int) -> "Intrinsics":
        """Intrinsics after a k x k box-filter downsample (pixel centres at integers)."""
        return Intrinsics(self.width // k, self.height // k, self.fx / k, self.fy / k,
                          (self.cx + 0.5) / k - 0.5, (self.cy + 0.5) / k - 0.5)


@dataclass
class Frame:
    index: int
    t: float
    rgb: np.ndarray            # (h, w, 3) float32, mapping resolution
    depth: np.ndarray          # (h, w) float32 metres (noisy), mapping resolution
    R_wc: np.ndarray           # camera->world rotation from VIO (None while initialising)
    p_wc: np.ndarray
    gt_R_wc: np.ndarray        # ground truth (evaluation only)
    gt_p_wc: np.ndarray
    vio_seconds: float         # wall-clock of the VIO update for this frame (this machine)
    render_seconds: float      # wall-clock of the synthetic renderer (not part of the system)
    features: dict | None = None   # insight-vio front-end output for this frame (stereo tracks)


class SyntheticWalkSource:
    def __init__(self, duration: float = 36.0, seed: int = 0, vio_width: int = 320,
                 vio_height: int = 240, focal: float = 200.0, map_downsample: int = 2,
                 depth_noise: float = 0.01, cam_rate: float = 20.0, walk: LabWalk | None = None,
                 room: LabRoom | None = None, use_vio: bool = True):
        from insight_vio.sim import SimConfig, simulate
        self.walk = walk or LabWalk()
        self.room = room or LabRoom(seed=seed)
        self.cfg = SimConfig(duration=duration, seed=seed, width=vio_width, height=vio_height,
                             focal=focal, cam_rate=cam_rate)
        self.data = simulate(self.cfg, traj=self.walk)
        c0 = self.data.cameras[0]
        self.vio_intr = Intrinsics(c0.width, c0.height, c0.fx, c0.fy, c0.cx, c0.cy)
        self.map_intr = self.vio_intr.scaled(map_downsample)
        self.k = map_downsample
        self.depth_noise = depth_noise
        self.rng = np.random.default_rng(seed + 1)
        self.use_vio = use_vio
        self.vio = self._make_vio() if use_vio else None

    def _make_vio(self):
        from insight_vio.config import VioConfig
        from insight_vio.frontend import FrontendConfig
        from insight_vio.pipeline import filter_config_for_sim
        from insight_vio.system import VioSystem
        d = self.data
        mcfg = filter_config_for_sim(d, stereo=True, pixel_sigma=1.5)
        cfg = VioConfig(cameras=d.cameras, noise=mcfg.noise, time_offset=0.0, filter=mcfg,
                        frontend=FrontendConfig(max_features=120, fast_threshold=12, min_distance=10,
                                                pyramid_levels=3))
        vio = VioSystem(cfg, stereo=True)
        # keep a copy of the front-end's stereo matches for the triangulated-feature init path
        process = vio.frontend.process

        def capture(*args, **kw):
            self.last_features = process(*args, **kw)
            return self.last_features

        vio.frontend.process = capture
        self.last_features = None
        k0 = d.cam_idx[0]
        sig = np.r_[[1e-3] * 3, [1e-3] * 3, [0.02] * 3, [0.02] * 3, [0.1] * 3]
        # Initialised from the ground-truth state at t0 (as insight-vio's own synthetic
        # pipeline does); this fixes the gauge so map and ground truth share a frame.
        vio.initialize_from_state(d.gt_t[k0], d.gt_R[k0], d.gt_p[k0], d.gt_v[k0],
                                  np.zeros(3), np.zeros(3), np.diag(sig ** 2))
        return vio

    def __len__(self):
        return len(self.data.cam_t)

    def gt_camera_pose(self, frame_index: int, cam: int = 0):
        d = self.data
        k = d.cam_idx[frame_index]
        c = d.cameras[cam]
        return d.gt_R[k] @ c.R_bc, d.gt_p[k] + d.gt_R[k] @ c.p_bc

    def render_view(self, R_wc, p_wc):
        """Noise-free colour + depth at mapping resolution (used for held-out evaluation)."""
        vi, mi = self.vio_intr, self.map_intr
        rgb, _ = self.room.render(R_wc, p_wc, vi.width, vi.height, vi.fx, vi.fy, vi.cx, vi.cy)
        _, depth = self.room.render(R_wc, p_wc, mi.width, mi.height, mi.fx, mi.fy, mi.cx, mi.cy)
        return downsample(rgb, self.k).astype(np.float32), depth

    def frames(self, max_frames: int | None = None):
        d = self.data
        n = len(d.cam_t) if max_frames is None else min(max_frames, len(d.cam_t))
        imu_ptr = 0
        cl, cr = d.cameras[0], d.cameras[1]
        vi, mi = self.vio_intr, self.map_intr
        for fi in range(n):
            tc, k = d.cam_t[fi], d.cam_idx[fi]
            while imu_ptr < len(d.imu_t) and d.imu_t[imu_ptr] <= tc + 1e-9:
                if self.vio is not None:
                    self.vio.add_imu(d.imu_t[imu_ptr], d.gyro[imu_ptr], d.accel[imu_ptr])
                imu_ptr += 1
            a = time.perf_counter()
            R_l, p_l = d.gt_R[k] @ cl.R_bc, d.gt_p[k] + d.gt_R[k] @ cl.p_bc
            rgb_hi, _ = self.room.render(R_l, p_l, vi.width, vi.height, vi.fx, vi.fy, vi.cx, vi.cy)
            _, depth = self.room.render(R_l, p_l, mi.width, mi.height, mi.fx, mi.fy, mi.cx, mi.cy)
            right = None
            if self.vio is not None:
                R_r, p_r = d.gt_R[k] @ cr.R_bc, d.gt_p[k] + d.gt_R[k] @ cr.p_bc
                right, _ = self.room.render(R_r, p_r, vi.width, vi.height, vi.fx, vi.fy, vi.cx, vi.cy)
            b = time.perf_counter()
            if self.vio is not None:
                out = self.vio.process_images(tc, to_gray_u8(rgb_hi), to_gray_u8(right))
                _, R_wb, p_wb, _ = out
                R_wc, p_wc = R_wb @ cl.R_bc, p_wb + R_wb @ cl.p_bc
            else:
                R_wc, p_wc = R_l, p_l
            c = time.perf_counter()
            noisy = depth * (1 + self.depth_noise * self.rng.standard_normal(depth.shape).astype(np.float32))
            rgb = downsample(rgb_hi, self.k)
            yield Frame(fi, float(tc), rgb.astype(np.float32), noisy.astype(np.float32), R_wc, p_wc,
                        R_l, p_l, c - b, b - a, getattr(self, "last_features", None))
