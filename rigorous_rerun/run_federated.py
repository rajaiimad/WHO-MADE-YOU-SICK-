"""
rigorous_rerun.run_federated
============================
RQ6 (federated layer), corrected re-run. Fixes relative to coper_vr/e9*:
  * no pooled-data initialisation: the global model starts from a seeded random
    initialisation;
  * no pooled scaler: feature standardisation uses federated sufficient
    statistics (per-client sums / sums of squares, aggregated once) -- what a
    secure-aggregation deployment can compute;
  * per-seed AUCs are saved, so every statistic can be recomputed;
  * attacks are applied to the client UPDATE u = w_local - w_global.

Clients: the 48 outer-train cohorts of outer fold 0 of each repeat (one cohort
= one client); test = the 12 held-out cohorts of that fold. Model: the same
detector family as the primary model (argument: MPNN or MLP), fixed config.

Attacks (on a random byz_frac of clients, re-drawn per seed):
  gaussian   u + 5 N(0, I)                        (as in the manuscript)
  signflip   -u                                    (as in the manuscript)
  labelflip  u computed after swapping safe/unsafe labels on the client
  alie       "a little is enough" (Baruch et al. 2019): mean - z*std of the
             BENIGN updates, z = 1.0 -- colluding + omniscient, i.e. OUTSIDE the
             manuscript's threat model; reported as a stress test only.
Aggregators: fedavg (mean), coordinate-wise median, trimmed mean (beta = 0.2).
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from common import OUT, SEEDS, jdump, main_tensors, outer_inner_splits, prototype_features
from models import MPNN, _make_mlp
from run_detection import SPECS, fixed_cfg

ROUNDS, LOCAL_EPOCHS, BATCH = 30, 2, 256
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def flat(net):
    return torch.cat([p.data.view(-1) for p in net.parameters()])


def load_flat(net, v):
    o = 0
    for p in net.parameters():
        n = p.numel(); p.data.copy_(v[o:o + n].view_as(p)); o += n


def aggregate(U, how):
    if how == "fedavg":
        return U.mean(0)
    if how == "median":
        return U.median(0).values
    if how == "trimmed":
        k = int(0.2 * len(U))
        S = U.sort(0).values
        return S[k:len(U) - k].mean(0)
    raise ValueError(how)


def run_one(kind, cfg, data, clients, te, y, agg, attack, frac, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 7)
    # federated standardisation from client sufficient statistics
    def suff(A):
        F = A.reshape(-1, A.shape[-1]).astype(np.float64)
        return F.sum(0), (F ** 2).sum(0), len(F)
    def fed_std(A_list):
        s = [suff(a) for a in A_list]
        n = sum(x[2] for x in s); mu = sum(x[0] for x in s) / n
        sd = np.sqrt(np.maximum(sum(x[1] for x in s) / n - mu ** 2, 0)) + 1e-6
        return mu.astype(np.float32), sd.astype(np.float32)
    if kind == "MLP":
        X = data
        mu, sd = fed_std([X[c] for c in clients])
        Z = torch.tensor((X - mu) / sd, device=DEV)
        net = _make_mlp(Z.shape[1], cfg["h"]).to(DEV)
        fwd = lambda idx: net(Z[idx]).squeeze(-1)
    else:
        V, E = data
        if cfg.get("logd"):
            E = np.concatenate([E, np.log(E[..., :1] + 0.05)], -1)
        mv, sv = fed_std([V[c] for c in clients]); me, se = fed_std([E[c] for c in clients])
        Vt = torch.tensor((V - mv) / sv, device=DEV); Et = torch.tensor((E - me) / se, device=DEV)
        net = MPNN(Vt.shape[1], Et.shape[2], cfg["h"]).to(DEV)
        fwd = lambda idx: net(Vt[idx], Et[idx])
    yt = torch.tensor(y, dtype=torch.float32, device=DEV)
    n_byz = int(round(frac * len(clients)))
    byz = set(rng.choice(len(clients), n_byz, replace=False).tolist()) if attack != "none" else set()
    g = flat(net).clone()
    lossf = torch.nn.BCEWithLogitsLoss()
    for r in range(ROUNDS):
        U, benign = [], []
        for ci, idx in enumerate(clients):
            load_flat(net, g)
            net.train()
            opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
            idx_t = torch.tensor(idx, device=DEV)
            for _ in range(LOCAL_EPOCHS):
                perm = idx_t[torch.randperm(len(idx_t), device=DEV)]
                for s in range(0, len(perm), BATCH):
                    b = perm[s:s + BATCH]
                    lb = (1 - yt[b]) if (ci in byz and attack == "labelflip") else yt[b]
                    opt.zero_grad(); lossf(fwd(b), lb).backward(); opt.step()
            u = flat(net) - g
            U.append(u)
            if ci not in byz:
                benign.append(u)
        Bm = torch.stack(benign)
        for ci in byz:
            if attack == "gaussian":
                U[ci] = U[ci] + 5.0 * torch.randn_like(U[ci])
            elif attack == "signflip":
                U[ci] = -U[ci]
            elif attack == "alie":
                U[ci] = Bm.mean(0) - 1.0 * Bm.std(0)
        g = g + aggregate(torch.stack(U), agg)
    load_flat(net, g)
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(torch.cat([fwd(torch.tensor(te[s:s + 8192], device=DEV))
                                     for s in range(0, len(te), 8192)])).cpu().numpy()
    p = np.nan_to_num(p, nan=0.5)
    return {"roc_auc": float(roc_auc_score(y[te], p)), "pr_auc": float(average_precision_score(y[te], p))}


def main(kind="MPNN"):
    t = main_tensors()
    y, ctx = t["y"], t["ctx"]
    ref = {"MPNN": "mpnn__MPNN", "MLP": "proto__MLP"}[kind]
    cfg = fixed_cfg(kind, ref)
    if kind == "MLP":
        data = prototype_features(t)
    else:
        _, V, E = SPECS[ref][1](t)
        data = (V, E)
    grid = [("none", 0.0)] + [(a, f) for a in ("gaussian", "signflip", "labelflip", "alie") for f in (0.2, 0.4)]
    res = []
    fp = OUT / "federated" / f"federated_{kind}.json"
    if fp.exists():   # resume: keep only seeds whose whole grid finished
        import json
        old = json.loads(fp.read_text())["runs"]
        res = [r for r in old if sum(o["seed"] == r["seed"] for o in old) == 3 * len(grid)]
    done = {r["seed"] for r in res}
    t0 = time.time()
    for s in SEEDS:
        if s in done:
            continue
        f, tr, va, te = next(outer_inner_splits(ctx, y, s))
        trv = np.concatenate([tr, va])
        clients = [trv[ctx[trv, 0] == c] for c in np.unique(ctx[trv, 0])]
        for attack, frac in grid:
            for agg in ("fedavg", "median", "trimmed"):
                r = run_one(kind, cfg, data, clients, te, y, agg, attack, frac, s)
                res.append(dict(r, seed=s, attack=attack, frac=frac, agg=agg))
                print(f"seed {s} {attack:9s} {frac:.1f} {agg:8s} ROC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        jdump({"model": ref, "cfg": cfg, "rounds": ROUNDS, "local_epochs": LOCAL_EPOCHS, "runs": res},
              OUT / "federated" / f"federated_{kind}.json")


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["MPNN"]))
