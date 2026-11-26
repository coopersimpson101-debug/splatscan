"""Live device source for a Jetson: insight-vio's hardware layer + a colour camera.

**NOT VALIDATED ON HARDWARE** (status: pending, see README "Hardware validation").

Tracking uses exactly insight-vio's live wiring (``scripts/run_live.py`` there):

* ``insight_vio.hw.cameras.OpenCVStereo(jetson_csi_pipeline(0), jetson_csi_pipeline(1))`` -
  two OV9281 global-shutter mono modules through Argus/GStreamer,
* ``insight_vio.hw.imu_spi.ICM42688`` (or ``BMI088``) on SPI, polled by ``ImuReader``,
* ``insight_vio.system.VioSystem`` with static initialisation.

Mapping additionally needs **colour** and **depth**:

* colour - a third, rigidly mounted CSI/USB RGB camera (``rgb.pipeline`` in the config), whose
  extrinsic ``T_imu_rgb`` comes from Kalibr like the stereo pair;
* depth - either ``depth.mode: stereo_features`` (sparse, triangulated from insight-vio's own
  stereo matches, :mod:`splatscan.triangulate`; no extra sensor) or ``depth.mode: sensor`` with
  a depth camera aligned to the RGB image (the driver is left to the integrator).

``LiveDeviceSource.frames()`` yields the same :class:`~splatscan.vio_source.Frame` objects as
the synthetic source, so ``OnlineMapper`` and the stream server are unchanged.
"""
from __future__ import annotations

import queue
import threading
import time

import numpy as np

from .vio_source import Frame, Intrinsics


def rgb_csi_pipeline(sensor_id: int, width: int, height: int, fps: int) -> str:
    """GStreamer pipeline for a colour CSI camera on Jetson (Argus), BGR for OpenCV."""
    return (f"nvarguscamerasrc sensor-id={sensor_id} ! video/x-raw(memory:NVMM),width={width},"
            f"height={height},framerate={fps}/1 ! nvvidconv ! video/x-raw,format=BGRx ! "
            f"videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=2")


def load_device_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def map_intrinsics(cfg: dict) -> Intrinsics:
    r = cfg["rgb"]
    fx, fy, cx, cy = r["intrinsics"]
    k = int(r.get("map_downsample", 1))
    w, h = r["resolution"]
    return Intrinsics(w, h, fx, fy, cx, cy).scaled(k) if k > 1 else Intrinsics(w, h, fx, fy, cx, cy)


def rgb_pose_from_body(R_wb, p_wb, T_imu_rgb):
    T = np.asarray(T_imu_rgb, float)
    return R_wb @ T[:3, :3], p_wb + R_wb @ T[:3, 3]


class LiveDeviceSource:  # pragma: no cover - needs a Jetson, cameras and an IMU
    def __init__(self, device_cfg: dict):
        from insight_vio.config import load_config
        from insight_vio.hw import cameras
        from insight_vio.hw.imu_spi import BMI088, ICM42688, ImuReader, SpidevBus
        from insight_vio.system import VioSystem
        import cv2

        self.cfg = device_cfg
        self.vio_cfg = load_config(device_cfg["vio_config"])
        hw = self.vio_cfg.hardware
        if hw.get("imu_driver", "icm42688") == "icm42688":
            imu = ICM42688(SpidevBus(hw.get("spi_bus", 0), hw.get("spi_cs", 0)))
        else:
            imu = BMI088(SpidevBus(hw.get("spi_bus", 0), hw.get("spi_cs_accel", 0)),
                         SpidevBus(hw.get("spi_bus", 0), hw.get("spi_cs_gyro", 1)))
        imu.configure()
        self.imu_q: queue.Queue = queue.Queue()
        self.reader = ImuReader(imu, device_cfg.get("imu_rate_hz", 400.0), hw.get("R_body_imu"))
        w, h, fps = hw.get("width", 1280), hw.get("height", 800), hw.get("fps", 30)
        self.stereo = cameras.OpenCVStereo(cameras.jetson_csi_pipeline(0, w, h, fps),
                                           cameras.jetson_csi_pipeline(1, w, h, fps))
        r = device_cfg["rgb"]
        rw, rh = r["resolution"]
        self.rgb = cv2.VideoCapture(rgb_csi_pipeline(r.get("sensor_id", 2), rw, rh, r.get("fps", 15)),
                                    cv2.CAP_GSTREAMER)
        self.cv2 = cv2
        self.vio = VioSystem(self.vio_cfg, stereo=True)
        process = self.vio.frontend.process

        def capture(*args, **kw):
            self.last_features = process(*args, **kw)
            return self.last_features

        self.vio.frontend.process = capture
        self.last_features = None
        self.map_intr = map_intrinsics(device_cfg)
        self.T_imu_rgb = np.asarray(r["T_imu_rgb"], float)

    def frames(self, max_frames=None):
        from .room import downsample
        from .triangulate import sparse_depth_map
        threading.Thread(target=self.reader.run, args=(lambda t, w, a: self.imu_q.put((t, w, a)),),
                         daemon=True).start()
        k = int(self.cfg["rgb"].get("map_downsample", 1))
        mi = self.map_intr
        for i, (t_ns, left, right) in enumerate(self.stereo.frames()):
            if max_frames is not None and i >= max_frames:
                return
            while not self.imu_q.empty():
                ti, w, a = self.imu_q.get()
                self.vio.add_imu(ti * 1e-9, w, a)
            a0 = time.perf_counter()
            pose = self.vio.process_images(t_ns * 1e-9, left, right)
            vio_s = time.perf_counter() - a0
            ok, bgr = self.rgb.read()
            if pose is None or not ok:
                continue                                   # initialising / no colour frame
            _, R_wb, p_wb, _ = pose
            R_wc, p_wc = rgb_pose_from_body(R_wb, p_wb, self.T_imu_rgb)
            rgb = downsample(bgr[..., ::-1].astype(np.float32) / 255.0, k)
            cams = self.vio_cfg.cameras
            # NOTE: features are triangulated in the *left mono* camera; with a separate RGB camera
            # they must be transformed into it - done here by re-projecting through T_imu_rgb.
            depth = np.full((mi.height, mi.width), np.nan, np.float32)
            if self.cfg.get("depth", {}).get("mode", "stereo_features") == "stereo_features" and self.last_features:
                depth = self._features_in_rgb(sparse_depth_map, cams, mi)
            yield Frame(i, t_ns * 1e-9, rgb, depth, R_wc, p_wc, None, None, vio_s, 0.0, self.last_features)

    def _features_in_rgb(self, sparse_depth_map, cams, mi):
        from .triangulate import stereo_extrinsics, triangulate_stereo
        xl, xr = [], []
        for obs in self.last_features.values():
            d = dict(obs)
            if 0 in d and 1 in d:
                xl.append(d[0])
                xr.append(d[1])
        depth = np.full((mi.height, mi.width), np.nan, np.float32)
        if not xl:
            return depth
        R_rl, t_rl = stereo_extrinsics(cams[0], cams[1])
        X, ok = triangulate_stereo(np.asarray(xl), np.asarray(xr), R_rl, t_rl)
        Xb = X[ok] @ cams[0].R_bc.T + cams[0].p_bc                    # left cam -> body (IMU)
        T = self.T_imu_rgb
        Xc = (Xb - T[:3, 3]) @ T[:3, :3]                               # body -> rgb camera
        front = Xc[:, 2] > 0.2
        Xc = Xc[front]
        u = np.round(mi.fx * Xc[:, 0] / Xc[:, 2] + mi.cx).astype(int)
        v = np.round(mi.fy * Xc[:, 1] / Xc[:, 2] + mi.cy).astype(int)
        inside = (u >= 0) & (u < mi.width) & (v >= 0) & (v < mi.height)
        depth[v[inside], u[inside]] = Xc[inside, 2]
        return depth
