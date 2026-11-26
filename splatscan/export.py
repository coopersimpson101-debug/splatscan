"""Export the map as a standard 3DGS .ply (loads in splatweb, the reference viewer, SuperSplat...)."""
from __future__ import annotations

import numpy as np


def write_ply(path: str, gmap) -> None:
    p = gmap.p
    n = len(gmap)
    means = p["means"].detach().float().cpu().numpy()
    dc = p["sh_dc"].detach().float().cpu().numpy().reshape(n, 3)
    rest = p["sh_rest"].detach().float().cpu().numpy()            # (n, K-1, 3) -> channel-major
    rest = rest.transpose(0, 2, 1).reshape(n, -1)
    logit = p["logits"].detach().float().cpu().numpy()[:, None]
    ls = p["log_scales"].detach().float().cpu().numpy()
    q = p["quats"].detach().float().cpu().numpy()
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    cols = [means, np.zeros((n, 3), np.float32), dc, rest, logit, ls, q]
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    names += [f"f_rest_{i}" for i in range(rest.shape[1])] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                              "rot_0", "rot_1", "rot_2", "rot_3"]
    data = np.concatenate(cols, 1).astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % n
    header += "".join(f"property float {nm}\n" for nm in names) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(data.tobytes())


def read_ply(path: str, sh_degree: int = 1, device="cpu"):
    """Load a .ply written by :func:`write_ply` back into a GaussianMap (for offline evaluation)."""
    import torch

    from .gaussians import GaussianMap
    with open(path, "rb") as f:
        head = b""
        while not head.endswith(b"end_header\n"):
            head += f.readline()
        lines = head.decode().splitlines()
        n = int([ln for ln in lines if ln.startswith("element vertex")][0].split()[-1])
        props = [ln.split()[-1] for ln in lines if ln.startswith("property")]
        data = np.frombuffer(f.read(), "<f4").reshape(n, len(props)).copy()
    ix = {p: i for i, p in enumerate(props)}
    col = lambda *names: torch.tensor(data[:, [ix[nm] for nm in names]])  # noqa: E731
    g = GaussianMap(sh_degree, device=device)
    K = g.K
    rest = col(*[f"f_rest_{i}" for i in range(3 * (K - 1))]).reshape(n, 3, K - 1).transpose(1, 2)
    new = {"means": col("x", "y", "z"), "log_scales": col("scale_0", "scale_1", "scale_2"),
           "quats": col("rot_0", "rot_1", "rot_2", "rot_3"), "logits": col("opacity")[:, 0],
           "sh_dc": col("f_dc_0", "f_dc_1", "f_dc_2")[:, None, :], "sh_rest": rest.contiguous()}
    g._replace({k: v.to(device) for k, v in new.items()})
    g.ids = torch.arange(n)
    return g
