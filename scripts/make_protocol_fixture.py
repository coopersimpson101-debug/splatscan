#!/usr/bin/env python
"""Write a small deterministic protocol-v1 stream + its expected final state.

The pair is committed to splatweb as ``tests/fixtures/splatscan_v1.{sslog,expected.json}``
so the TypeScript decoder is tested against bytes produced by this (Python) encoder::

    python scripts/make_protocol_fixture.py --out ../splatweb/tests/fixtures
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from splatscan import protocol as P  # noqa: E402
from splatscan.vio_source import Intrinsics  # noqa: E402


def snapshot(rng, ids, deg=1):
    n = len(ids)
    q = rng.normal(size=(n, 4))
    return dict(ids=np.asarray(ids, np.uint32), pos=rng.normal(size=(n, 3)).astype(np.float32),
                scale=rng.uniform(0.01, 0.2, (n, 3)).astype(np.float32),
                rot=(q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32),
                opacity=rng.uniform(0.05, 0.95, n).astype(np.float32),
                sh=rng.normal(size=(n, 3 * (deg + 1) ** 2)).astype(np.float32))


def build():
    rng = np.random.default_rng(1234)
    enc = P.DeltaEncoder(sh_degree=1)
    intr = Intrinsics(160, 120, 100.0, 100.0, 79.5, 59.5)
    msgs = [enc.hello(capacity_hint=8)]
    s = snapshot(rng, range(20))
    m, _ = enc.diff(s)
    msgs += m + [enc.frame(0.5, 10, 20, np.eye(3), [0.1, 0.2, 1.5], intr, 21.5)]
    # move a few, drop a few, add a few
    s2 = {k: v.copy() for k, v in s.items()}
    s2["pos"][[2, 5, 7]] += 0.5
    s2["opacity"][3] = 0.01
    keep = np.setdiff1d(np.arange(20), [0, 9, 13])
    s2 = {k: v[keep] for k, v in s2.items()}
    extra = snapshot(rng, range(20, 26))
    s2 = {k: np.concatenate([s2[k], extra[k]]) for k in s2}
    m, counts = enc.diff(s2)
    msgs += m + [enc.frame(1.0, 20, len(s2["ids"]), np.eye(3), [0.3, 0.2, 1.5], intr, 23.0)]
    assert counts == dict(added=6, updated=4, removed=3), counts
    return msgs, enc.sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    msgs, final = build()
    with open(os.path.join(a.out, "splatscan_v1.sslog"), "wb") as f:
        P.write_log_header(f)
        for m in msgs:
            P.append_log(f, m)
    exp = dict(frames=2, last_frame=dict(frame_index=20, p_wc=[0.3, 0.2, 1.5]), splats=[
        dict(id=int(final["ids"][i]), pos=final["pos"][i].tolist(), scale=final["scale"][i].tolist(),
             rot=final["rot"][i].tolist(), opacity=float(final["opacity"][i]), sh=final["sh"][i].tolist())
        for i in range(len(final["ids"]))])
    with open(os.path.join(a.out, "splatscan_v1.expected.json"), "w") as f:
        json.dump(exp, f, indent=1)
    print(f"wrote {len(msgs)} messages, {len(final['ids'])} splats to {a.out}")


if __name__ == "__main__":
    main()
