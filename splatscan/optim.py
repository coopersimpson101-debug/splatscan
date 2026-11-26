"""Hand-written Adam with per-parameter groups and *per-splat* state that survives resizing.

torch.optim.Adam keeps one step counter per tensor, which is wrong for an online map where
splats are born at different times: a splat added at global step 5000 would get no bias
correction at all and its first updates would be ~sqrt(1/(1-beta2)) too small. Here every
row (splat) has its own step count, and the moment buffers are appended to / gathered
together with the parameters when splats are added, cloned, split or pruned.

Update (Kingma & Ba 2015), per element, with ``k`` the row's own step count:

    m <- b1 m + (1 - b1) g
    v <- b2 v + (1 - b2) g^2
    p <- p - lr * (m / (1 - b1^k)) / (sqrt(v / (1 - b2^k)) + eps)

Rows whose gradient is exactly zero this step (splats not visible in the sampled keyframe)
are skipped entirely - their moments and step count are frozen - like the "sparse Adam"
used by recent 3DGS trainers, so invisible splats do not drift on stale momentum.
"""
from __future__ import annotations

import torch


class SplatAdam:
    def __init__(self, params: dict, lrs: dict, betas=(0.9, 0.999), eps: float = 1e-15):
        self.params = params            # name -> leaf tensor (N, ...)
        self.lrs = dict(lrs)
        self.b1, self.b2 = betas
        self.eps = eps
        n = next(iter(params.values())).shape[0]
        dev = next(iter(params.values())).device
        self.m = {k: torch.zeros_like(p) for k, p in params.items()}
        self.v = {k: torch.zeros_like(p) for k, p in params.items()}
        self.steps = torch.zeros(n, dtype=torch.int64, device=dev)

    @torch.no_grad()
    def step(self):
        active = None
        for p in self.params.values():
            if p.grad is None:
                continue
            nz = p.grad.reshape(p.shape[0], -1).abs().amax(1) > 0
            active = nz if active is None else active | nz
        if active is None or not active.any():
            return
        self.steps += active.long()
        k = self.steps.clamp_min(1).to(next(iter(self.params.values())).dtype)
        for name, p in self.params.items():
            g = p.grad
            if g is None:
                continue
            shape = (-1,) + (1,) * (p.dim() - 1)
            a = active.view(shape)
            m, v = self.m[name], self.v[name]
            m.copy_(torch.where(a, self.b1 * m + (1 - self.b1) * g, m))
            v.copy_(torch.where(a, self.b2 * v + (1 - self.b2) * g * g, v))
            mh = m / (1 - self.b1 ** k).view(shape)
            vh = v / (1 - self.b2 ** k).view(shape)
            upd = self.lrs[name] * mh / (vh.sqrt() + self.eps)
            p.sub_(torch.where(a, upd, torch.zeros_like(upd)))

    def zero_grad(self):
        for p in self.params.values():
            p.grad = None

    # -------------------------------------------------- resizing alongside the map
    def append(self, n: int):
        for name, p in self.params.items():
            z = p.new_zeros((n,) + tuple(p.shape[1:]))
            self.m[name] = torch.cat([self.m[name], z])
            self.v[name] = torch.cat([self.v[name], z.clone()])
        self.steps = torch.cat([self.steps, self.steps.new_zeros(n)])

    def gather(self, idx: torch.Tensor):
        """Keep / duplicate rows ``idx`` (clones and splits inherit their parent's moments)."""
        for name in self.params:
            self.m[name] = self.m[name][idx]
            self.v[name] = self.v[name][idx]
        self.steps = self.steps[idx]
