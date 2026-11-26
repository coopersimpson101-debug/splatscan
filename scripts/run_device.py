#!/usr/bin/env python
"""On-device scanner loop (Jetson): live cameras + IMU -> insight-vio -> online splats -> WebSocket.

**NOT VALIDATED ON HARDWARE.** See deploy/jetson/ and docs/jetson.md.

    python scripts/run_device.py --config deploy/jetson/splatscan.yaml [--dry-run]

``--dry-run`` only parses the config and prints the wiring (works on any machine).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from splatscan.hw import load_device_config, map_intrinsics  # noqa: E402
from splatscan.mapper import MapperConfig  # noqa: E402


def mapper_config(cfg: dict) -> MapperConfig:
    mc = MapperConfig()
    for k, v in cfg.get("mapper", {}).items():
        if not hasattr(mc, k):
            raise KeyError(f"unknown mapper option {k!r}")
        setattr(mc, k, v)
    return mc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = load_device_config(a.config)
    mc = mapper_config(cfg)
    intr = map_intrinsics(cfg)
    st = cfg.get("stream", {})
    print(f"vio config : {cfg['vio_config']}")
    print(f"rgb        : sensor {cfg['rgb'].get('sensor_id')} {cfg['rgb']['resolution']} -> map {intr.width}x{intr.height}")
    print(f"depth      : {cfg.get('depth', {}).get('mode', 'stereo_features')}")
    print(f"mapper     : device={mc.device} tile={mc.tile} window={mc.window} iters/kf={mc.iters_per_kf}")
    print(f"stream     : ws://{st.get('host', '0.0.0.0')}:{st.get('port', 8765)}  log={st.get('log')}")
    if a.dry_run:
        return
    from splatscan.hw import LiveDeviceSource
    from splatscan.mapper import OnlineMapper
    from splatscan.server import StreamServer
    src = LiveDeviceSource(cfg)
    mapper = OnlineMapper(intr, mc)
    srv = StreamServer(st.get("host", "0.0.0.0"), st.get("port", 8765), mc.sh_degree, st.get("log")).start()
    print(f"streaming on ws://{st.get('host', '0.0.0.0')}:{srv.port}", flush=True)
    t0 = time.perf_counter()
    for f in src.frames():
        info = mapper.process(f)
        if info:
            srv.publish_snapshot(mapper.map.activated(), frame=dict(t=f.t, frame_index=f.index, splat_count=len(mapper.map),
                                                                    R_wc=f.R_wc, p_wc=f.p_wc, intr=intr))
            print(f"[kf {info['kf']}] {info['splats']} splats, {info['seconds'] * 1e3:.0f} ms "
                  f"(t={time.perf_counter() - t0:.0f} s)", flush=True)
        elif f.index % 3 == 0:
            srv.publish_frame(t=f.t, frame_index=f.index, splat_count=len(mapper.map), R_wc=f.R_wc, p_wc=f.p_wc, intr=intr)


if __name__ == "__main__":
    main()
