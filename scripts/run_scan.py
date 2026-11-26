#!/usr/bin/env python
"""End-to-end online scan: VIO poses -> online splat mapping -> live stream (+ metrics).

    # replay a cached VIO run (python -m splatscan.record --out runs/walk.npz) into the mapper
    python scripts/run_scan.py --recording runs/walk.npz --out runs/scan
    # or run VIO + rendering live in the same process
    python scripts/run_scan.py --live --duration 36 --out runs/scan
    # stream to browsers while scanning: open splatweb with ?live=ws://127.0.0.1:8765
    python scripts/run_scan.py --recording runs/walk.npz --out runs/scan --serve 8765

Writes to --out: metrics.jsonl (held-out PSNR/SSIM over time), keyframes.jsonl, summary.json,
stream.sslog (every protocol message, replayable with ``python -m splatscan.server``),
frames/cam_*.png + frames/map_*.png (for the hero GIF) and map.ply.

Every 20th frame (index % 20 == 10) is held out: never used for mapping, only rendered from its
ground-truth pose and compared with the true image. "seen" metrics use the held-out views
whose timestamp is <= the current time (views of places already walked past).
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from splatscan.export import write_ply  # noqa: E402
from splatscan.mapper import MapperConfig, OnlineMapper  # noqa: E402
from splatscan.server import StreamServer  # noqa: E402


def save_png(path, rgb):
    from PIL import Image
    Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8)).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src_g = ap.add_mutually_exclusive_group(required=True)
    src_g.add_argument("--recording", help="npz written by python -m splatscan.record")
    src_g.add_argument("--live", action="store_true", help="run VIO + renderer in-process")
    ap.add_argument("--duration", type=float, default=36.0, help="(--live) seconds of walk")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--iters", type=int, default=None, help="optimisation steps per keyframe")
    ap.add_argument("--first-iters", type=int, default=None)
    ap.add_argument("--kf-trans", type=float, default=None)
    ap.add_argument("--kf-rot", type=float, default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--eval-every", type=float, default=3.0, help="seconds of sequence time between evals")
    ap.add_argument("--frame-every", type=int, default=4, help="publish a FRAME marker / dump GIF frames every N frames")
    ap.add_argument("--serve", type=int, default=None, help="also serve the stream on this port")
    ap.add_argument("--no-frames", action="store_true")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    os.makedirs(os.path.join(a.out, "frames"), exist_ok=True)
    if a.recording:
        from splatscan.record import CachedSource
        src = CachedSource(a.recording)
    else:
        from splatscan.vio_source import SyntheticWalkSource
        src = SyntheticWalkSource(duration=a.duration)
    intr = src.map_intr
    cfg = MapperConfig(device=a.device)
    for k, v in dict(iters_per_kf=a.iters, first_kf_iters=a.first_iters, kf_trans=a.kf_trans,
                     kf_rot_deg=a.kf_rot, window=a.window).items():
        if v is not None:
            setattr(cfg, k, v)
    mapper = OnlineMapper(intr, cfg)
    server = StreamServer(port=a.serve or 0, sh_degree=cfg.sh_degree, log_path=os.path.join(a.out, "stream.sslog"))
    if a.serve is not None:
        server.start()
        print(f"streaming on ws://127.0.0.1:{server.port}", flush=True)

    held, est_p, gt_p, vio_s = [], [], [], []
    metrics_f = open(os.path.join(a.out, "metrics.jsonl"), "w")
    kf_f = open(os.path.join(a.out, "keyframes.jsonl"), "w")
    next_eval = a.eval_every
    last_psnr = float("nan")
    map_seconds = 0.0
    stream_bytes = 0
    t_wall0 = time.perf_counter()
    last_t = 0.0

    def do_eval(t):
        nonlocal last_psnr
        seen = [(R, p, rgb) for (tt, R, p, rgb) in held if tt <= t + 1e-9]
        e = mapper.evaluate(seen)
        last_psnr = e["psnr"]
        rec = dict(t=round(t, 3), wall_s=round(time.perf_counter() - t_wall0, 2), splats=len(mapper.map),
                   keyframes=len(mapper.keyframes), iterations=mapper.iterations, psnr_seen=e["psnr"],
                   ssim_seen=e["ssim"], n_seen=e["n_views"])
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        print(f"[eval] t={t:5.1f}s splats={len(mapper.map):6d} kfs={len(mapper.keyframes):3d} "
              f"PSNR(seen)={e['psnr']:.2f} dB SSIM={e['ssim']:.3f} on {e['n_views']} views "
              f"(wall {rec['wall_s']:.0f} s)", flush=True)

    for f in src.frames(a.max_frames):
        last_t = f.t
        est_p.append(f.p_wc)
        gt_p.append(f.gt_p_wc)
        vio_s.append(f.vio_seconds)
        if f.index % 20 == 10:
            held.append((f.t, f.gt_R_wc, f.gt_p_wc, f.rgb))
        else:
            t0 = time.perf_counter()
            info = mapper.process(f)
            map_seconds += time.perf_counter() - t0
            if info:
                counts = server.publish_snapshot(mapper.map.activated(), frame=dict(
                    t=f.t, frame_index=f.index, splat_count=len(mapper.map), R_wc=f.R_wc, p_wc=f.p_wc,
                    intr=intr, psnr=last_psnr))
                stream_bytes += counts["bytes"]
                info.update({f"stream_{k}": v for k, v in counts.items()})
                kf_f.write(json.dumps(info) + "\n")
                kf_f.flush()
                print(f"[kf {info['kf']:3d}] frame {f.index:4d} t={f.t:5.2f}s +{info['added']} splats "
                      f"(clone {info['cloned']}, split {info['split']}, prune {info['pruned']}) -> {info['splats']}; "
                      f"loss {info['loss']:.4f}; {info['seconds']:.1f} s; stream +{counts['added']} "
                      f"~{counts['updated']} -{counts['removed']} ({counts['bytes'] / 1024:.0f} KiB)", flush=True)
        if f.index % a.frame_every == 0:
            server.publish_frame(t=f.t, frame_index=f.index, splat_count=len(mapper.map), R_wc=f.R_wc,
                                 p_wc=f.p_wc, intr=intr, psnr=last_psnr)
            if not a.no_frames:
                save_png(os.path.join(a.out, "frames", f"cam_{f.index:04d}.png"), f.rgb)
                if len(mapper.map):
                    save_png(os.path.join(a.out, "frames", f"map_{f.index:04d}.png"),
                             mapper.render_rgb(f.R_wc, f.p_wc))
        if f.t >= next_eval:
            do_eval(f.t)
            next_eval += a.eval_every

    do_eval(last_t)
    final_all = mapper.evaluate([(R, p, rgb) for (_, R, p, rgb) in held])
    wall = time.perf_counter() - t_wall0
    from insight_vio.metrics import ate
    est_p, gt_p = np.array(est_p), np.array(gt_p)
    raw_err = np.linalg.norm(est_p - gt_p, axis=1)
    A = ate(est_p, gt_p)
    summary = dict(
        machine=f"{platform.processor() or platform.machine()} ({os.cpu_count()} logical CPUs), torch {torch.__version__}, "
                f"device={a.device}, threads={a.threads}",
        date=time.strftime("%Y-%m-%d"), sequence_s=round(last_t, 2), frames=len(est_p), heldout_views=len(held),
        keyframes=len(mapper.keyframes), iterations=mapper.iterations, splats=len(mapper.map),
        final_psnr_all_heldout=final_all["psnr"], final_ssim_all_heldout=final_all["ssim"],
        ate_rmse_m=A.rmse, ate_max_m=A.max, raw_pos_err_rmse_m=float(np.sqrt(np.mean(raw_err ** 2))),
        raw_pos_err_max_m=float(raw_err.max()),
        vio_ms_per_frame=float(np.mean(vio_s) * 1e3), mapping_s_total=map_seconds,
        mapping_s_per_keyframe=map_seconds / max(1, len(mapper.keyframes)), wall_s=wall,
        realtime_factor=last_t / wall, stream_bytes=server.stats["bytes"], stream_messages=server.stats["messages"],
        config={k: v for k, v in vars(cfg).items() if isinstance(v, (int, float, str, tuple))},
        source="recording " + os.path.basename(a.recording) if a.recording else "live in-process",
    )
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    json.dump(dict(p_vio=est_p.round(4).tolist(), p_gt=gt_p.round(4).tolist()),
              open(os.path.join(a.out, "trajectory.json"), "w"))
    write_ply(os.path.join(a.out, "map.ply"), mapper.map)
    server.stop()
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
