#!/usr/bin/env python
"""README figures from a real run directory (written by scripts/run_scan.py).

    python scripts/make_figures.py --run runs/scan --live runs/scan/live --out docs/assets

* psnr_over_time.png - held-out PSNR / SSIM of the online map vs sequence time, + splat count
* hero.gif           - left: the synthetic camera walk; right: the streamed map rendered by
                       splatweb headless (scripts/live_capture.ts in splatweb) at the same frame
* banner.svg         - 1280x400 banner whose motif is the run's real map (top-down splat means)
                       and VIO trajectory
"""
import argparse
import glob
import json
import os
import struct

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ACCENT = "#FB7185"


def load_jsonl(p):
    return [json.loads(ln) for ln in open(p) if ln.strip()]


def read_ply_xyz_rgb(path):
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        lines = head.decode().splitlines()
        n = int([ln for ln in lines if ln.startswith("element vertex")][0].split()[-1])
        props = [ln.split()[-1] for ln in lines if ln.startswith("property")]
        data = np.frombuffer(f.read(), "<f4").reshape(n, len(props))
    ix = {p: i for i, p in enumerate(props)}
    xyz = data[:, [ix["x"], ix["y"], ix["z"]]]
    rgb = np.clip(0.5 + 0.28209479177387814 * data[:, [ix["f_dc_0"], ix["f_dc_1"], ix["f_dc_2"]]], 0, 1)
    op = 1 / (1 + np.exp(-data[:, ix["opacity"]]))
    return xyz, rgb, op


def psnr_plot(run, out, compare=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = load_jsonl(os.path.join(run, "metrics.jsonl"))
    s = json.load(open(os.path.join(run, "summary.json")))
    t = [r["t"] for r in m]
    fig, ax = plt.subplots(figsize=(8, 3.4), dpi=110)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(t, [r["psnr_seen"] for r in m], "o-", color=ACCENT, lw=2, label="PSNR, insight-vio poses")
    if compare:
        mc = load_jsonl(os.path.join(compare, "metrics.jsonl"))
        ax.plot([r["t"] for r in mc], [r["psnr_seen"] for r in mc], "o:", color="#fda4af", lw=1.5, ms=3,
                label="PSNR, ground-truth poses (ablation)")
    ax.set_xlabel("sequence time [s]", color="#e6edf3")
    ax.set_ylabel("PSNR [dB]", color=ACCENT)
    ax2 = ax.twinx()
    ax2.plot(t, [r["splats"] / 1000 for r in m], "s--", color="#8b949e", lw=1.2, ms=3, label="splats (k)")
    ax2.set_ylabel("splats [thousands]", color="#8b949e")
    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("axes", 1.13))
    ax3.plot(t, [r["ssim_seen"] for r in m], "^-", color="#7dd3fc", lw=1.2, ms=3, label="SSIM")
    ax3.set_ylabel("SSIM", color="#7dd3fc")
    for a in (ax, ax2, ax3):
        a.tick_params(colors="#c9d1d9")
        for sp in a.spines.values():
            sp.set_color("#30363d")
    ax.grid(color="#21262d")
    lines = sum((a.get_legend_handles_labels()[0] for a in (ax, ax2, ax3)), [])
    labels = sum((a.get_legend_handles_labels()[1] for a in (ax, ax2, ax3)), [])
    leg = ax.legend(lines, labels, loc="lower left", fontsize=8, facecolor="#161b22", edgecolor="#30363d")
    for txt in leg.get_texts():
        txt.set_color("#e6edf3")
    ax.set_title(f"Held-out views seen so far, rendered from their true pose ({s['date']})",
                 color="#e6edf3", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def trajectory_plot(run, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tj = json.load(open(os.path.join(run, "trajectory.json")))
    s = json.load(open(os.path.join(run, "summary.json")))
    pv, pg = np.array(tj["p_vio"]), np.array(tj["p_gt"])
    xyz, rgb, op = read_ply_xyz_rgb(os.path.join(run, "map.ply"))
    keep = (op > 0.3) & (xyz[:, 2] < 2.6)
    fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=110)
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")
    ax.scatter(xyz[keep, 0], xyz[keep, 1], s=0.3, c=rgb[keep], alpha=0.6, linewidths=0)
    ax.plot(pg[:, 0], pg[:, 1], color="#e6edf3", lw=1.5, label="ground truth")
    ax.plot(pv[:, 0], pv[:, 1], color=ACCENT, lw=1.5, ls="--", label="insight-vio (MSCKF)")
    ax.set_aspect("equal")
    ax.set_xlim(-4.2, 4.2)
    ax.set_ylim(-4.2, 4.2)
    ax.tick_params(colors="#c9d1d9")
    for sp in ax.spines.values():
        sp.set_color("#30363d")
    leg = ax.legend(loc="upper left", fontsize=8, facecolor="#161b22", edgecolor="#30363d")
    for txt in leg.get_texts():
        txt.set_color("#e6edf3")
    err = np.linalg.norm(pv - pg, axis=1)
    ax.set_title("top-down: map splats + camera path\n"
                 f"VIO error RMSE {np.sqrt(np.mean(err ** 2)) * 100:.1f} cm (raw, "
                 f"max {err.max() * 100:.1f} cm), ATE(aligned) {s['ate_rmse_m'] * 100:.1f} cm", color="#e6edf3", fontsize=9)
    ax.set_xlabel("x [m]", color="#e6edf3")
    ax.set_ylabel("y [m]", color="#e6edf3")
    fig.tight_layout()
    fig.savefig(out, facecolor=fig.get_facecolor())
    plt.close(fig)


def _font(size):
    for name in ("consola.ttf", "DejaVuSansMono.ttf", "Menlo.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def hero_gif(run, live, out, every=12, ms=240, panel=(288, 216)):
    fr = json.load(open(os.path.join(live, "frames.json")))
    W, H = fr["width"], fr["height"]
    info = {f["frame"]: f for f in fr["frames"]}
    metrics = load_jsonl(os.path.join(run, "metrics.jsonl"))
    cams = sorted(glob.glob(os.path.join(run, "frames", "cam_*.png")))
    font, small = _font(14), _font(12)
    frames = []
    pw, ph = panel
    for p in cams:
        k = int(os.path.basename(p)[4:8])
        if k % every or k not in info:
            continue
        live_p = os.path.join(live, f"live_{k:04d}.rgba")
        if not os.path.exists(live_p):
            continue
        left = Image.open(p).convert("RGB").resize((pw, ph), Image.NEAREST)
        right = Image.fromarray(np.fromfile(live_p, np.uint8).reshape(H, W, 4)[..., :3])
        d = ImageDraw.Draw(right)
        fz = info[k]["frustum"]
        if fz[0] and all(fz[1:]):
            c = tuple(fz[0])
            for q in fz[1:]:
                d.line([c, tuple(q)], fill=ACCENT, width=2)
            d.line([tuple(q) for q in fz[1:]] + [tuple(fz[1])], fill=ACCENT, width=2)
        right = right.resize((int(ph * W / H), ph), Image.BILINEAR)
        bar_h, top = 24, 18
        img = Image.new("RGB", (pw + right.width, ph + bar_h + top), (13, 17, 23))
        img.paste(left, (0, top))
        img.paste(right, (pw, top))
        t = info[k]["t"]
        ps = [m["psnr_seen"] for m in metrics if m["t"] <= t + 1e-6]
        dd = ImageDraw.Draw(img)
        dd.text((6, 2), "camera walk (synthetic)", fill=(230, 237, 243), font=small)
        dd.text((pw + 6, 2), "live map via WebSocket (splatweb)", fill=(230, 237, 243), font=small)
        txt = f"t={t:4.1f}s  splats={info[k]['splats']:6d}  " + (f"held-out PSNR {ps[-1]:.1f} dB" if ps else "")
        dd.text((6, top + ph + 4), txt, fill=(251, 113, 133), font=font)
        frames.append(img.convert("P", palette=Image.ADAPTIVE, colors=72))
    if not frames:
        raise SystemExit("no matching frames")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    return len(frames)


def banner(run, out_svg, n_pts=2600, seed=0):
    xyz, rgb, op = read_ply_xyz_rgb(os.path.join(run, "map.ply"))
    rng = np.random.default_rng(seed)
    keep = (op > 0.3) & (xyz[:, 2] < 2.6)
    xyz, rgb = xyz[keep], rgb[keep]
    sel = rng.choice(len(xyz), size=min(n_pts, len(xyz)), replace=False)
    pts = xyz[sel]
    # top-down view of the room, fitted into the right part of the banner
    x0, x1, y0, y1 = -4.1, 4.1, -4.1, 4.1
    ox, oy, sz = 870, 12, 352
    to = lambda x, y: (ox + (x - x0) / (x1 - x0) * sz, oy + (y1 - y) / (y1 - y0) * sz)  # noqa: E731
    tj = os.path.join(run, "trajectory.json")
    est = np.array(json.load(open(tj))["p_vio"]) if os.path.exists(tj) else None
    s = json.load(open(os.path.join(run, "summary.json")))
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 400" width="1280" height="400">',
             '<defs><pattern id="g" width="32" height="32" patternUnits="userSpaceOnUse">'
             '<path d="M32 0H0V32" fill="none" stroke="#161b22" stroke-width="1"/></pattern></defs>',
             '<rect width="1280" height="400" fill="#0d1117"/><rect width="1280" height="400" fill="url(#g)"/>']
    for (x, y, _), c in zip(pts, rgb[sel]):
        X, Y = to(x, y)
        # accent-tinted by the splat's own luminance (real map data)
        lum = float(0.3 * c[0] + 0.59 * c[1] + 0.11 * c[2])
        parts.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="1.6" fill="{ACCENT}" fill-opacity="{0.25 + 0.6 * lum:.2f}"/>')
    if est is not None:
        path = " ".join(f"{'M' if i == 0 else 'L'}{to(p[0], p[1])[0]:.1f},{to(p[0], p[1])[1]:.1f}"
                        for i, p in enumerate(est[::3]))
        parts.append(f'<path d="{path}" fill="none" stroke="#e6edf3" stroke-width="2" stroke-opacity="0.85"/>')
    mono = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
    parts += [
        f'<text x="64" y="170" font-family="{mono}" font-weight="700" font-size="88" fill="#e6edf3">splatscan</text>',
        f'<text x="68" y="222" font-family="{mono}" font-size="24" fill="#8b949e">live 3D scanning: VIO + online Gaussian splatting</text>',
        f'<text x="68" y="258" font-family="{mono}" font-size="24" fill="#8b949e">streamed to the browser as you walk</text>',
        f'<text x="68" y="330" font-family="{mono}" font-size="18" fill="{ACCENT}">'
        f'{s["splats"]:,} splats · {s["keyframes"]} keyframes · VIO ATE {s["ate_rmse_m"] * 100:.1f} cm · hand-written backward pass</text>',
        f'<text x="1236" y="392" text-anchor="end" font-family="{mono}" font-size="12" fill="#484f58">'
        f'motif: top-down map + VIO path from a real run</text>',
        "</svg>"]
    open(out_svg, "w", encoding="utf-8").write("\n".join(parts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--live", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=12)
    ap.add_argument("--compare", default=None, help="second run dir (e.g. ground-truth poses) for the PSNR plot")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    psnr_plot(a.run, os.path.join(a.out, "psnr_over_time.png"), a.compare)
    trajectory_plot(a.run, os.path.join(a.out, "trajectory.png"))
    n = hero_gif(a.run, a.live, os.path.join(a.out, "hero.gif"), every=a.every)
    banner(a.run, os.path.join(a.out, "banner.svg"))
    for f in ("psnr_over_time.png", "trajectory.png", "hero.gif", "banner.svg"):
        print(f, os.path.getsize(os.path.join(a.out, f)) // 1024, "KiB")
    print("gif frames", n)


if __name__ == "__main__":
    main()
