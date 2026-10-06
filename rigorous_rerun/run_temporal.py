"""
rigorous_rerun.run_temporal
===========================
RQ3: does causal temporal context help beyond per-frame aggregation?

Same nested folds as detection. Model: HistGB with the fixed Tier-1 proto__HGB
configuration. Inputs: the 20-D prototype at frame t, plus (K > 1) the causal
mean and max of the prototype over frames t-K+1..t of the SAME victim track
(rows are stored contiguously per (cohort, track, victim); windows never cross
a track, hence never a split). Writes outputs/detection/temporal_K<K>__HGB/.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from common import OUT, SEEDS, jdump, main_tensors, outer_inner_splits, prototype_features, thresholds_from_val
from models import FAMILIES, platt
from run_detection import fixed_cfg


def causal_window(X, T, K):
    n = len(X) // T
    S = X.reshape(n, T, -1)
    mean = np.zeros_like(S); mx = np.zeros_like(S)
    for t in range(T):
        w = S[:, max(0, t - K + 1): t + 1]
        mean[:, t], mx[:, t] = w.mean(1), w.max(1)
    return np.concatenate([S, mean, mx], 2).reshape(n * T, -1)


def main(Ks=(1, 4, 8, 16)):
    t = main_tensors()
    y, ctx, T = t["y"], t["ctx"], int(t["T"])
    # contiguity check: every block of T rows is one (cohort, track, victim) with t = 0..T-1
    assert (ctx[:, 2].reshape(-1, T) == np.arange(T)).all()
    assert (np.ptp(ctx[:, [0, 1, 3]].reshape(-1, T, 3), axis=1) == 0).all()
    X0 = prototype_features(t)
    cfg = fixed_cfg("HGB", "proto__HGB")
    cfg.pop("n_iter", None)
    for K in Ks:
        t0 = time.time()
        X = X0 if K == 1 else causal_window(X0, T, K)
        name = f"temporal_K{K}__HGB"
        od = OUT / "detection" / name
        od.mkdir(parents=True, exist_ok=True)
        preds = {k: [] for k in ("seed", "fold", "row", "p", "p_cal")}
        sel = []
        for s in SEEDS:
            for f, tr, va, te in outer_inner_splits(ctx, y, s):
                d = {"Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va], "Xte": {"test": X[te]}}
                pv, pt = FAMILIES["HGB"](d, cfg, s)
                cal = platt(pv, y[va])
                sel.append({"seed": s, "fold": f, "cfg": dict(cfg, n_iter=d["_meta"]["n_iter"]),
                            "thresholds": thresholds_from_val(y[va], pv)})
                for k, v in (("seed", np.full(len(te), s)), ("fold", np.full(len(te), f)), ("row", te),
                             ("p", pt["test"]), ("p_cal", cal(pt["test"]))):
                    preds[k].append(v)
        np.savez_compressed(od / "preds.npz", **{k: np.concatenate(v) for k, v in preds.items()})
        jdump(sel, od / "selection.json")
        print(f"{name} done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
