"""Stage-2 attribution rule baselines (PREREGISTRATION_v3): no learning, same 25 folds,
threshold for precision/recall chosen on validation edges (F1-optimal)."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import OUT, PEER_NAMES, SEEDS, jdump, main_tensors, outer_inner_splits, peer_intensity, thresholds_from_val
from run_attribution import edge_idx, ranking_metrics


def main():
    t = main_tensors()
    K = t["y_att"].shape[1]
    ya = t["y_att"].ravel()
    ex = peer_intensity(t["E_chan"]).max(-1)                       # (n,K)
    d, v = t["E_geo"][..., 0], t["E_geo"][..., 1]
    S = {"rule_excess": ex.ravel(), "rule_excess_geo": (ex * (0.5 + 0.5 * v) / (d + 0.5)).ravel()}
    ch = t["atk_ch"]
    res = {k: [] for k in S}; loco = {f"{k}|{c}": [] for k in S for c in PEER_NAMES}
    for s in SEEDS:
        for f, tr, va, te in outer_inner_splits(t["ctx"], t["y"], s):
            for k, sc in S.items():
                ev, et = edge_idx(va, K), edge_idx(te, K)
                thr = thresholds_from_val(ya[ev], sc[ev])["f1opt"]
                r = {"roc_auc": float(roc_auc_score(ya[et], sc[et])), "pr_auc": float(average_precision_score(ya[et], sc[et])),
                     "seed": s, "fold": f}
                r.update(ranking_metrics(ya[et], sc[et], K, thr)); res[k].append(r)
                for c in range(1, 6):            # rules see no training data, so 'held-out' = test tracks of channel c
                    tec = te[ch[te] == c]
                    if t["y_att"][tec].sum() == 0:
                        continue
                    ec = edge_idx(tec, K)
                    rr = {"roc_auc": float(roc_auc_score(ya[ec], sc[ec])), "seed": s, "fold": f}
                    rr.update(ranking_metrics(ya[ec], sc[ec], K, thr)); loco[f"{k}|{PEER_NAMES[c-1]}"].append(rr)
    summ = lambda L, m: [float(np.mean([x[m] for x in L])), float(np.std([x[m] for x in L], ddof=1))]
    out = {"in_distribution": {k: {m: summ(v, m) for m in ("roc_auc", "pr_auc", "top1", "top2", "precision", "recall", "macro_f1", "spurious_rate_no_source")} for k, v in res.items()},
           "per_channel": {k: {m: summ(v, m) for m in ("roc_auc", "top1")} for k, v in loco.items()}}
    jdump(out, OUT / "attribution_rules.json")
    for k, v in out["in_distribution"].items():
        print(k, {m: round(x[0], 3) for m, x in v.items()})


if __name__ == "__main__":
    main()
