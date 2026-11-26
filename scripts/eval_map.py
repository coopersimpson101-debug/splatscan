#!/usr/bin/env python
"""Offline evaluation of a finished map (map.ply) on the held-out views of its recording.

    python scripts/eval_map.py --run runs/scan --recording runs/walk.npz

Renders every held-out frame (index % 20 == 10) from (a) its ground-truth pose and (b) the pose
insight-vio estimated for it, and reports PSNR / SSIM for both. (b) measures how well the map
explains the images *in its own frame*; (a) additionally includes the VIO drift.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from splatscan.export import read_ply  # noqa: E402
from splatscan.losses import psnr, ssim  # noqa: E402
from splatscan.mapper import camera_from  # noqa: E402
from splatscan.record import CachedSource  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--recording", required=True)
    a = ap.parse_args()
    torch.set_num_threads(1)
    g = read_ply(os.path.join(a.run, "map.ply"))
    src = CachedSource(a.recording)
    res = {"gt_pose": [], "vio_pose": []}
    with torch.no_grad():
        for f in src.frames():
            if f.index % 20 != 10:
                continue
            gt = torch.tensor(f.rgb)
            for key, (R, p) in (("gt_pose", (f.gt_R_wc, f.gt_p_wc)), ("vio_pose", (f.R_wc, f.p_wc))):
                out, _, _ = g.render(camera_from(R, p, src.map_intr), tile=4)
                out = out.clamp(0, 1)
                res[key].append((psnr(out, gt), ssim(out, gt).item()))
    summary = {k: dict(psnr=float(np.mean([x[0] for x in v])), ssim=float(np.mean([x[1] for x in v])), n=len(v))
               for k, v in res.items()}
    print(json.dumps(summary, indent=1))
    json.dump(summary, open(os.path.join(a.run, "eval_heldout.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
