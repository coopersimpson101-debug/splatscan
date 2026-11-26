"""Run the VIO front of the pipeline once and cache its output (frames + VIO poses).

The mapper is strictly causal either way - frame k only ever sees frames <= k - but the
numpy FAST/KLT front-end and the ray-caster are slow on a desktop CPU, so for experiments it
is convenient to run them once::

    python -m splatscan.record --duration 36 --out runs/walk.npz

and replay the cached stream into the mapper. ``scripts/run_scan.py`` can also run both live
in one process (``--live``); the numbers in the README say which was used.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from .vio_source import Frame, Intrinsics, SyntheticWalkSource


def record(out: str, duration: float = 36.0, seed: int = 0, log_every: int = 40, use_vio: bool = True):
    src = SyntheticWalkSource(duration=duration, seed=seed, use_vio=use_vio)
    rgb, depth, Rv, pv, Rg, pg, t, tv, tr = [], [], [], [], [], [], [], [], []
    t0 = time.perf_counter()
    for f in src.frames():
        rgb.append(np.clip(f.rgb * 255 + 0.5, 0, 255).astype(np.uint8))
        depth.append(f.depth.astype(np.float16))
        Rv.append(f.R_wc), pv.append(f.p_wc), Rg.append(f.gt_R_wc), pg.append(f.gt_p_wc)
        t.append(f.t), tv.append(f.vio_seconds), tr.append(f.render_seconds)
        if log_every and f.index % log_every == 0:
            err = np.linalg.norm(f.p_wc - f.gt_p_wc) * 100
            print(f"frame {f.index:4d}  t={f.t:5.2f}s  |p_vio - p_gt| = {err:.2f} cm  "
                  f"({time.perf_counter() - t0:.0f} s elapsed)", flush=True)
    mi = src.map_intr
    np.savez_compressed(out, rgb=np.stack(rgb), depth=np.stack(depth), R_vio=np.stack(Rv), p_vio=np.stack(pv),
                        R_gt=np.stack(Rg), p_gt=np.stack(pg), t=np.array(t), vio_s=np.array(tv),
                        render_s=np.array(tr), intr=np.array([mi.width, mi.height, mi.fx, mi.fy, mi.cx, mi.cy]),
                        duration=duration, seed=seed, use_vio=use_vio)
    return out


class CachedSource:
    """Replays a recording made by :func:`record` as a stream of :class:`Frame`."""

    def __init__(self, path: str):
        z = np.load(path)
        self.z = {k: z[k] for k in z.files}
        w, h, fx, fy, cx, cy = self.z["intr"]
        self.map_intr = Intrinsics(int(w), int(h), float(fx), float(fy), float(cx), float(cy))
        self.seed = int(self.z["seed"])

    def __len__(self):
        return len(self.z["t"])

    def frames(self, max_frames: int | None = None):
        z = self.z
        n = len(self) if max_frames is None else min(max_frames, len(self))
        for i in range(n):
            yield Frame(i, float(z["t"][i]), z["rgb"][i].astype(np.float32) / 255.0,
                        z["depth"][i].astype(np.float32), z["R_vio"][i], z["p_vio"][i], z["R_gt"][i],
                        z["p_gt"][i], float(z["vio_s"][i]), float(z["render_s"][i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--duration", type=float, default=36.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-poses", action="store_true", help="skip VIO and use ground-truth poses")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    record(a.out, a.duration, a.seed, use_vio=not a.gt_poses)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
