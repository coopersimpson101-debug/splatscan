"""The Jetson deployment files parse and wire up (no hardware needed)."""
import configparser
import os
import subprocess
import sys

import numpy as np

from splatscan.hw import load_device_config, map_intrinsics, rgb_csi_pipeline, rgb_pose_from_body

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = os.path.join(ROOT, "deploy", "jetson", "splatscan.yaml")


def test_device_config_parses_and_maps_to_mapper_options():
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from run_device import mapper_config
    cfg = load_device_config(CFG)
    mc = mapper_config(cfg)
    assert mc.device == "cuda" and mc.add_stride == 1
    intr = map_intrinsics(cfg)
    assert (intr.width, intr.height) == (320, 240) and intr.cx == (319.5 + 0.5) / 2 - 0.5
    T = np.asarray(cfg["rgb"]["T_imu_rgb"])
    assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3))


def test_dry_run_prints_wiring():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_device.py"), "--config", CFG, "--dry-run"],
                         capture_output=True, text=True, check=True).stdout
    assert "ws://0.0.0.0:8765" in out and "stereo_features" in out


def test_systemd_unit_is_well_formed():
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.read(os.path.join(ROOT, "deploy", "jetson", "splatscan.service"))
    assert cp["Service"]["ExecStart"].endswith("--config /etc/splatscan/splatscan.yaml")
    assert cp["Install"]["WantedBy"] == "multi-user.target"


def test_rgb_pipeline_and_pose_helpers():
    assert "nvarguscamerasrc sensor-id=2" in rgb_csi_pipeline(2, 640, 480, 15)
    R, p = rgb_pose_from_body(np.eye(3), np.array([1.0, 2, 3]), np.eye(4))
    assert np.allclose(R, np.eye(3)) and np.allclose(p, [1, 2, 3])
