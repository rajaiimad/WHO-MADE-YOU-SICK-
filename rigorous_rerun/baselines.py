"""
rigorous_rerun.baselines
========================
Stage-2 baselines pre-registered in PREREGISTRATION_v3.md. Importing this module
registers the families RULE, GCN, GAT in models.FAMILIES / models.GRIDS.

Graph inputs follow run_detection.graph(["bb"], ["chan", "int", "bb", "geo"]):
edge columns 0:5 channels/max, 5:10 excess over benign band, 10:15 peer backbone,
15:18 geometry (distance, view overlap, audio proximity).
"""
from __future__ import annotations

import numpy as np

import models
from models import _Std, _predict, _torch, _train_torch

CH, EX, BB, GEO = slice(0, 5), slice(5, 10), slice(10, 15), slice(15, 18)


# ---------------------------------------------------------------- rule (no learning)
def rule_score(E):
    ex, d, v = E[..., EX], E[..., 15], E[..., 16]
    return (ex.max(-1) * (0.5 + 0.5 * v) / (d + 0.5)).sum(-1)


def fit_RULE(d, cfg, seed):
    s = lambda E: rule_score(E)
    return s(d["Eva"]), {k: s(E) for k, (V, E) in d["Gte"].items()}


# ---------------------------------------------------------------- per-edge interaction features (for HGB)
def edge_interaction_features(t):
    """Explicit per-edge interaction terms aggregated over peers (sum and max) + victim backbone."""
    from common import peer_intensity
    ex = peer_intensity(t["E_chan"])                       # (n,K,5)
    d, v = t["E_geo"][..., 0:1], t["E_geo"][..., 1:2]
    a = ex * v / (d + 0.1)
    b = ex / (d + 0.1)
    return np.concatenate([t["V_bb"], a.sum(1), a.max(1), b.sum(1), b.max(1)], 1).astype(np.float32)


# ---------------------------------------------------------------- GCN / GAT on the star graph
def _nets():
    torch = _torch()
    import torch.nn as nn

    class GCN(nn.Module):
        """One graph-convolution layer on the star graph (victim + K peers, self-loops,
        symmetric normalisation), then an MLP readout. Messages are LINEAR in peer features."""
        def __init__(self, dv, dn, h, use_geo):
            super().__init__()
            self.use_geo = use_geo
            self.w_self = nn.Linear(dv, h)
            self.w_nb = nn.Linear(dn + (3 if use_geo else 0), h, bias=False)
            self.rho = nn.Sequential(nn.Linear(dv + h, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

        def forward(self, V, E):
            K = E.shape[1]
            x = torch.cat([E[..., :15], E[..., GEO]], -1) if self.use_geo else E[..., :15]
            c = 1.0 / np.sqrt(2.0 * (K + 1))
            h = torch.relu(self.w_self(V) / (K + 1) + c * self.w_nb(x).sum(1))
            return self.rho(torch.cat([V, h], -1)).squeeze(-1)

    class GAT(nn.Module):
        """GATv2-style attention with edge features: attention logits depend on
        victim, peer and geometry; messages are linear in peer features; 4 heads."""
        def __init__(self, dv, dn, h, heads=4):
            super().__init__()
            self.heads, self.hd = heads, max(4, h // heads)
            H = heads * self.hd
            self.w_v = nn.Linear(dv, H); self.w_n = nn.Linear(dn, H); self.w_e = nn.Linear(3, H)
            self.att = nn.Parameter(torch.randn(heads, self.hd) * 0.1)
            self.w_msg = nn.Linear(dn, H)
            self.rho = nn.Sequential(nn.Linear(dv + H, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

        def forward(self, V, E):
            B, K, _ = E.shape
            xn, ge = E[..., :15], E[..., GEO]
            z = self.w_v(V).unsqueeze(1) + self.w_n(xn) + self.w_e(ge)          # (B,K,H)
            z = torch.nn.functional.leaky_relu(z, 0.2).view(B, K, self.heads, self.hd)
            alpha = torch.softmax((z * self.att).sum(-1), dim=1)                # (B,K,heads)
            m = self.w_msg(xn).view(B, K, self.heads, self.hd)
            agg = (alpha.unsqueeze(-1) * m).sum(1).reshape(B, -1)
            return self.rho(torch.cat([V, torch.relu(agg)], -1)).squeeze(-1)
    return GCN, GAT


def _fit_graph(d, cfg, seed, build):
    torch = _torch()
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sv, se = _Std(d["Vtr"]), _Std(d["Etr"])
    TV = lambda V: torch.tensor(sv(V), device=dev)
    TE = lambda E: torch.tensor(se(E), device=dev)
    Vtr, Etr, Vva, Eva = TV(d["Vtr"]), TE(d["Etr"]), TV(d["Vva"]), TE(d["Eva"])
    net = build(Vtr.shape[1]).to(dev)
    _train_torch(net, lambda bi: net(Vtr[bi], Etr[bi]), len(Vtr), d["ytr"],
                 lambda: _predict(net, lambda s: net(Vva[s], Eva[s]), len(Vva)), d["yva"], cfg, seed)
    pv = _predict(net, lambda s: net(Vva[s], Eva[s]), len(Vva))
    tests = {}
    for k, (V, E) in d["Gte"].items():
        Vt, Et = TV(V), TE(E)
        tests[k] = _predict(net, lambda s: net(Vt[s], Et[s]), len(Vt))
    d.setdefault("_meta", {}).update(net=net, scalers=(sv, se))
    return pv, tests


def fit_GCN(d, cfg, seed):
    GCN, _ = _nets()
    return _fit_graph(d, cfg, seed, lambda dv: GCN(dv, 15, cfg["h"], cfg.get("geo", False)))


def fit_GAT(d, cfg, seed):
    _, GAT = _nets()
    return _fit_graph(d, cfg, seed, lambda dv: GAT(dv, 15, cfg["h"]))


_NEURAL = [dict(h=h, lr=lr, pw=pw, wd=1e-4) for h in (32, 64) for lr in (1e-3, 3e-3) for pw in ("none", "sqrt")]
models.GRIDS.update({"RULE": [dict()], "GCN": _NEURAL, "GCNGEO": [dict(c, geo=True) for c in _NEURAL], "GAT": _NEURAL})
models.FAMILIES.update({"RULE": fit_RULE, "GCN": fit_GCN, "GCNGEO": fit_GCN, "GAT": fit_GAT})


if __name__ == "__main__":          # self-check on random tensors
    rng = np.random.default_rng(0)
    E = rng.random((7, 3, 18)).astype(np.float32); V = rng.random((7, 5)).astype(np.float32)
    assert rule_score(E).shape == (7,)
    GCN, GAT = _nets()
    torch = _torch()
    for net in (GCN(5, 15, 32, False), GCN(5, 15, 32, True), GAT(5, 15, 32)):
        net.eval()
        out = net(torch.tensor(V), torch.tensor(E))
        assert out.shape == (7,)
        # permutation invariance over peers
        perm = net(torch.tensor(V), torch.tensor(E[:, [2, 0, 1]]))
        assert torch.allclose(out, perm, atol=1e-5), type(net).__name__
    print("baselines self-check OK")
