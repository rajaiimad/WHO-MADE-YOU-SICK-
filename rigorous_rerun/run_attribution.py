"""
rigorous_rerun.run_attribution
==============================
Candidate source attribution (per directed edge peer -> victim, frame t) under
the same repeated nested cohort-disjoint folds as detection.

Label: the generator's victim-specific `attributed_peer` (fixed in the Sep-2026
audit). Positives exist only for victims the attacker measurably affects.

Feature sets (all frame-t only; benign bands are public platform bounds):
  paper     : victim logits(4) | peer logits(4) | peer channels(5) | geometry(3)
              (the manuscript's per-edge feature set; constant edge_type dropped)
  relative  : paper + peer excess-over-benign intensity(5) + per-channel contrast
              against the other incoming peers of the same victim-frame(5)
  agnostic  : channel-AGNOSTIC summary only -- max/sum/count of excess intensity
              over channels, contrast of the max against the other peers, and
              geometry. Contains no channel identity, so it is the only set that
              can in principle transfer to an unseen attack channel.
Controls  : geometry-only, peer-channels-only, logits-only, permuted training
            labels, randomised geometry.
LOCO      : leave-one-attack-channel-out (train/val never contain tracks attacked
            through the held-out channel; test = held-out-channel tracks only).

Model: HistGB with the manuscript's configuration (max_iter 80, depth 6,
lr 0.1) for every feature set -- identical capacity, so differences come from the
representation. Operating threshold: F1-optimal on VALIDATION edges.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from common import OUT, PEER_NAMES, SEEDS, jdump, main_tensors, outer_inner_splits, peer_intensity, thresholds_from_val


def edge_tables(t):
    n, K = t["y_att"].shape
    vb = np.repeat(t["V_bb"][:, None, :4], K, 1)
    pb = t["E_bb"][..., :4]
    ch = t["E_chan"]
    geo = t["E_geo"]
    it = peer_intensity(ch)                                     # (n,K,5)
    others_max = np.stack([np.delete(it, k, 1).max(1) for k in range(K)], 1)   # (n,K,5)
    contrast = it - others_max
    mx, sm, cnt = it.max(2), it.sum(2), (it > 0).sum(2)
    mx_others = np.stack([np.delete(mx, k, 1).max(1) for k in range(K)], 1)
    agn = np.stack([mx, sm, cnt, mx - mx_others], 2)
    F = {
        "paper": np.concatenate([vb, pb, ch, geo], 2),
        "relative": np.concatenate([vb, pb, ch, geo, it, contrast], 2),
        "agnostic": np.concatenate([agn, geo], 2),
        "ctl_geometry_only": geo,
        "ctl_peer_channels_only": ch,
        "ctl_logits_only": np.concatenate([vb, pb], 2),
    }
    return {k: v.reshape(n * K, -1).astype(np.float32) for k, v in F.items()}


def hgb(seed):
    return HistGradientBoostingClassifier(max_iter=80, max_depth=6, learning_rate=0.1, random_state=seed)


def ranking_metrics(y_edge, p_edge, K, thr):
    """y/p reshaped (frames, K). Top-k over victim-frames with a true source;
    spurious-attribution rate over victim-frames WITHOUT a true source."""
    Y, P = y_edge.reshape(-1, K), p_edge.reshape(-1, K)
    has = Y.max(1) == 1
    order = np.argsort(-P[has], 1)
    hit = np.take_along_axis(Y[has], order, 1)
    top = {f"top{k}": float(hit[:, :k].max(1).mean()) for k in (1, 2)} if has.any() else {}
    yh = (P >= thr).astype(int)
    tp = int((yh & Y).sum()); fp = int((yh & (1 - Y)).sum()); fn = int(((1 - yh) & Y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # macro-F1 over the two edge classes
    tn = int(((1 - yh) & (1 - Y)).sum())
    p0 = tn / (tn + fn) if tn + fn else 0.0; r0 = tn / (tn + fp) if tn + fp else 0.0
    f0 = 2 * p0 * r0 / (p0 + r0) if p0 + r0 else 0.0
    spurious = float(yh[~has].max(1).mean()) if (~has).any() else float("nan")
    return dict(top, precision=prec, recall=rec, f1=f1, macro_f1=(f0 + f1) / 2,
                false_attribution_rate=1 - prec if tp + fp else 0.0, spurious_rate_no_source=spurious)


def evaluate(F, y, tr, va, te, K, seed, perm_labels=False):
    ytr = y[tr]
    if perm_labels:
        ytr = np.random.default_rng(seed).permutation(ytr)
    m = hgb(seed).fit(F[tr], ytr)
    pv, pt = m.predict_proba(F[va])[:, 1], m.predict_proba(F[te])[:, 1]
    thr = thresholds_from_val(y[va], pv)["f1opt"] if y[va].any() else 0.5
    r = {"roc_auc": float(roc_auc_score(y[te], pt)), "pr_auc": float(average_precision_score(y[te], pt)),
         "prevalence": float(y[te].mean())}
    r.update(ranking_metrics(y[te], pt, K, thr))
    return r, pt


def edge_idx(frame_idx, K):
    return (frame_idx[:, None] * K + np.arange(K)[None]).ravel()


def main():
    t0 = time.time()
    t = main_tensors()
    K = t["y_att"].shape[1]
    ya = t["y_att"].ravel()
    F = edge_tables(t)
    # randomised-geometry control: geometry rows shuffled across all edges
    rg = F["paper"].copy()
    rg[:, 13:16] = rg[np.random.default_rng(7).permutation(len(rg)), 13:16]
    F["ctl_randomised_geometry"] = rg
    ch_frame = t["atk_ch"]                                   # 0 none, 1..5
    res = {k: [] for k in list(F) + ["ctl_permuted_labels"]}
    loco = {f"{fs}|{PEER_NAMES[c - 1]}": [] for fs in ("paper", "relative", "agnostic") for c in range(1, 6)}
    pred_store = {}
    for s in SEEDS:
        for f, tr, va, te in outer_inner_splits(t["ctx"], t["y"], s):
            etr, eva, ete = edge_idx(tr, K), edge_idx(va, K), edge_idx(te, K)
            for name, A in F.items():
                r, pt = evaluate(A, ya, etr, eva, ete, K, s)
                res[name].append(dict(r, seed=s, fold=f))
                if name in ("paper", "relative", "agnostic"):
                    pred_store.setdefault(name, []).append((s, f, ete, pt))
            r, _ = evaluate(F["paper"], ya, etr, eva, ete, K, s, perm_labels=True)
            res["ctl_permuted_labels"].append(dict(r, seed=s, fold=f))
            # LOCO
            for c in range(1, 6):
                trc, vac, tec = tr[ch_frame[tr] != c], va[ch_frame[va] != c], te[ch_frame[te] == c]
                if t["y_att"][tec].sum() == 0 or t["y_att"][vac].sum() == 0:
                    continue
                for fs in ("paper", "relative", "agnostic"):
                    r, _ = evaluate(F[fs], ya, edge_idx(trc, K), edge_idx(vac, K), edge_idx(tec, K), K, s)
                    loco[f"{fs}|{PEER_NAMES[c - 1]}"].append(dict(r, seed=s, fold=f))
            print(f"seed {s} fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    od = OUT / "attribution"
    od.mkdir(parents=True, exist_ok=True)
    jdump({"in_distribution": res, "loco": loco}, od / "per_fold.json")
    for name, L in pred_store.items():
        np.savez_compressed(od / f"preds_{name}.npz", seed=np.concatenate([np.full(len(e), s) for s, f, e, p in L]),
                            fold=np.concatenate([np.full(len(e), f) for s, f, e, p in L]),
                            edge=np.concatenate([e for *_, e, p in L]), p=np.concatenate([p for *_, p in L]))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
