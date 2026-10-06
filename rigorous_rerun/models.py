"""
rigorous_rerun.models
=====================
Model families evaluated under the nested cohort-disjoint protocol. Every
family exposes fit_predict(data, cfg, seed) -> (p_val, {name: p_test}) where
`data` holds train / val / test(s) inputs. Scalers are fitted on TRAIN only;
early stopping and model selection use VAL only; TEST is only predicted.

Inputs are either tabular  X: (n, d)  or graph  (V: (n, dv), E: (n, K, de)).
"""
from __future__ import annotations

import itertools

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Hyper-parameter grids (identical search budget philosophy: <= 16 configs,
# fixed before any test fold is scored).
# ---------------------------------------------------------------------------
GRIDS = {
    "LR": [dict(C=C, cw=cw) for C in (0.01, 0.1, 1.0, 10.0) for cw in (None, "balanced")],
    "HGB": [dict(lr=lr, leaves=lv, msl=msl, cw=cw, it=300)
            for lr in (0.05, 0.1) for lv in (15, 31) for msl in (20, 100) for cw in (None, "balanced")],
    "MLP": [dict(h=h, lr=lr, pw=pw, wd=1e-4) for h in (64, 128) for lr in (1e-3, 3e-3) for pw in ("none", "sqrt")],
    "MPNN": [dict(h=h, lr=lr, pw=pw, wd=1e-4, logd=ld) for h in (32, 64) for lr in (1e-3, 3e-3)
             for pw in ("none", "sqrt") for ld in (False, True)],
    "MLPpub": [dict()],      # the manuscript's sklearn MLP(64,32), as published -- no tuning
}


# ---------------------------------------------------------------------------
# sklearn families (tabular)
# ---------------------------------------------------------------------------
def _sk_fit_predict(model, Xtr, ytr, Xva, tests):
    model.fit(Xtr, ytr)
    pv = model.predict_proba(Xva)[:, 1]
    return pv, {k: model.predict_proba(X)[:, 1] for k, X in tests.items()}


def fit_LR(d, cfg, seed):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=cfg["C"], class_weight=cfg["cw"], max_iter=2000))
    d.setdefault("_meta", {})["model"] = m
    return _sk_fit_predict(m, d["Xtr"], d["ytr"], d["Xva"], d["Xte"])


def fit_HGB(d, cfg, seed):
    # Early stopping on a frame-level random split would mix frames of one cohort;
    # we instead pick the number of boosting rounds on the VALIDATION cohorts.
    m = HistGradientBoostingClassifier(learning_rate=cfg["lr"], max_leaf_nodes=cfg["leaves"],
                                       min_samples_leaf=cfg["msl"], class_weight=cfg["cw"],
                                       max_iter=cfg["it"], early_stopping=False, random_state=seed)
    m.fit(d["Xtr"], d["ytr"])
    d.setdefault("_meta", {})["model"] = m
    best, best_k = -1, cfg["it"]
    for k, pv in enumerate(m.staged_predict_proba(d["Xva"]), 1):
        if k % 10 == 0 or k == cfg["it"]:
            ap = average_precision_score(d["yva"], pv[:, 1])
            if ap > best:
                best, best_k = ap, k
    def at_k(X):
        for k, p in enumerate(m.staged_predict_proba(X), 1):
            if k == best_k:
                return p[:, 1]
    d.setdefault("_meta", {})["n_iter"] = best_k
    return at_k(d["Xva"]), {k: at_k(X) for k, X in d["Xte"].items()}


def fit_MLPpub(d, cfg, seed):
    """Manuscript prototype detector exactly as in coper_vr (sklearn MLP(64,32),
    max_iter 40, frame-level early stopping)."""
    m = make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=40, random_state=seed,
                                                      early_stopping=True, validation_fraction=0.1,
                                                      n_iter_no_change=6))
    d.setdefault("_meta", {})["model"] = m
    return _sk_fit_predict(m, d["Xtr"], d["ytr"], d["Xva"], d["Xte"])


# ---------------------------------------------------------------------------
# torch families
# ---------------------------------------------------------------------------
def _torch():
    import torch
    return torch


class _Std:
    """Feature standardiser fitted on TRAIN rows (edges flattened over slots)."""
    def __init__(self, A):
        flat = A.reshape(-1, A.shape[-1])
        self.mu = flat.mean(0)
        self.sd = flat.std(0) + 1e-6

    def __call__(self, A):
        return ((A - self.mu) / self.sd).astype(np.float32)


def _make_mlp(din, h, dropout=0.1):
    import torch.nn as nn
    return nn.Sequential(nn.Linear(din, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, h // 2), nn.ReLU(),
                         nn.Linear(h // 2, 1))


class MPNN:
    """Edge-conditioned message passing over the incoming co-presence edges of a
    victim:  m_j = phi([v, e_j]);  z = [v, sum_j m_j, max_j m_j];  logit = rho(z).
    Permutation-invariant in the peers and defined for any cohort size."""
    def __new__(cls, dv, de, h, agg="summax", dropout=0.1):
        import torch
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.phi = nn.Sequential(nn.Linear(dv + de, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU())
                k = 2 if agg == "summax" else 1
                self.rho = nn.Sequential(nn.Linear(dv + k * h, h), nn.ReLU(), nn.Dropout(dropout),
                                         nn.Linear(h, 1))
                self.agg = agg

            def forward(self, V, E, mask=None):
                B, K, _ = E.shape
                x = torch.cat([V.unsqueeze(1).expand(B, K, V.shape[1]), E], -1)
                m = self.phi(x)
                if mask is not None:
                    m = m * mask.unsqueeze(-1)
                if self.agg == "summax":
                    z = torch.cat([m.sum(1), m.max(1).values], -1)
                elif self.agg == "sum":
                    z = m.sum(1)
                elif self.agg == "mean":
                    z = m.mean(1)
                else:
                    z = m.max(1).values
                return self.rho(torch.cat([V, z], -1)).squeeze(-1)
        return _Net()


def _train_torch(net, fwd, n_tr, ytr, va_fn, yva, cfg, seed, max_epochs=60, patience=10, batch=512):
    torch = _torch()
    dev = next(net.parameters()).device
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    pos = max(1, int(ytr.sum()))
    pw = {"none": 1.0, "sqrt": float(np.sqrt((len(ytr) - pos) / pos)),
          "full": float((len(ytr) - pos) / pos)}[cfg.get("pw", "none")]
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=dev))
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)
    g = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, bad = -1.0, None, 0
    for ep in range(max_epochs):
        net.train()
        perm = torch.randperm(n_tr, generator=g).to(dev)
        for s in range(0, n_tr, batch):
            bi = perm[s:s + batch]
            opt.zero_grad()
            loss = lossf(fwd(bi), yt[bi])
            loss.backward()
            opt.step()
        net.eval()
        pv = va_fn()
        ap = average_precision_score(yva, pv)
        if ap > best + 1e-5:
            best, bad = ap, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return best


def _predict(net, fn, n, batch=8192):
    torch = _torch()
    out = []
    with torch.no_grad():
        for s in range(0, n, batch):
            out.append(torch.sigmoid(fn(slice(s, s + batch))).float().cpu().numpy())
    return np.concatenate(out)


def fit_MLP(d, cfg, seed):
    torch = _torch()
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sc = _Std(d["Xtr"])
    T = lambda X: torch.tensor(sc(X), device=dev)
    Xtr, Xva = T(d["Xtr"]), T(d["Xva"])
    net = _make_mlp(Xtr.shape[1], cfg["h"]).to(dev)
    _train_torch(net, lambda bi: net(Xtr[bi]).squeeze(-1), len(Xtr), d["ytr"],
                 lambda: _predict(net, lambda s: net(Xva[s]).squeeze(-1), len(Xva)), d["yva"], cfg, seed)
    pv = _predict(net, lambda s: net(Xva[s]).squeeze(-1), len(Xva))
    d.setdefault("_meta", {}).update(net=net, scalers=(sc,))
    tests = {}
    for k, X in d["Xte"].items():
        Xt = T(X)
        tests[k] = _predict(net, lambda s: net(Xt[s]).squeeze(-1), len(Xt))
    return pv, tests


def _logd(E, on):
    """Optional generic log-transform of the distance column (edge feature 0)."""
    if not on:
        return E
    return np.concatenate([E, np.log(E[..., :1] + 0.05)], -1)


def fit_MPNN(d, cfg, seed, agg="summax"):
    torch = _torch()
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Etr = _logd(d["Etr"], cfg["logd"])
    sv, se = _Std(d["Vtr"]), _Std(Etr)
    TV = lambda V: torch.tensor(sv(V), device=dev)
    TE = lambda E: torch.tensor(se(_logd(E, cfg["logd"])), device=dev)
    Vtr, Etr_t, Vva, Eva = TV(d["Vtr"]), TE(d["Etr"]), TV(d["Vva"]), TE(d["Eva"])
    net = MPNN(Vtr.shape[1], Etr_t.shape[2], cfg["h"], agg=cfg.get("agg", agg)).to(dev)
    _train_torch(net, lambda bi: net(Vtr[bi], Etr_t[bi]), len(Vtr), d["ytr"],
                 lambda: _predict(net, lambda s: net(Vva[s], Eva[s]), len(Vva)), d["yva"], cfg, seed)
    pv = _predict(net, lambda s: net(Vva[s], Eva[s]), len(Vva))
    tests = {}
    for k, (V, E) in d["Gte"].items():
        Vt, Et = TV(V), TE(E)
        tests[k] = _predict(net, lambda s: net(Vt[s], Et[s]), len(Vt))
    d.setdefault("_meta", {}).update(net=net, scalers=(sv, se))
    return pv, tests


FAMILIES = {"LR": fit_LR, "HGB": fit_HGB, "MLP": fit_MLP, "MPNN": fit_MPNN, "MLPpub": fit_MLPpub}


def select_and_predict(family, d, seed, grid=None):
    """Nested selection: fit every config on TRAIN, keep the one with the best
    VALIDATION average precision. Returns (p_val, tests, cfg, val_ap, all_val_aps)."""
    grid = grid if grid is not None else GRIDS[family]
    best = None
    log = []
    for cfg in grid:
        dd = dict(d)
        pv, pt = FAMILIES[family](dd, cfg, seed)
        ap = float(average_precision_score(d["yva"], pv))
        log.append((cfg, ap, dd.get("_meta", {}).get("n_iter")))
        if best is None or ap > best[3]:
            best = (pv, pt, dict(cfg, **({"n_iter": dd["_meta"]["n_iter"]} if "n_iter" in dd.get("_meta", {}) else {})), ap)
    return best[0], best[1], best[2], best[3], [(c, a) for c, a, _ in log]


def platt(pv, yv):
    """Platt scaling fitted on VALIDATION scores; returns a callable."""
    z = np.log(np.clip(pv, 1e-7, 1 - 1e-7) / np.clip(1 - pv, 1e-7, 1)).reshape(-1, 1)
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(z, yv)

    def f(p):
        zz = np.log(np.clip(p, 1e-7, 1 - 1e-7) / np.clip(1 - p, 1e-7, 1)).reshape(-1, 1)
        return lr.predict_proba(zz)[:, 1]
    return f
