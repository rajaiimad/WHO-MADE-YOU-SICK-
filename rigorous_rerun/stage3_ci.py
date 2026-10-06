"""Cohort-bootstrap 95% CIs for the Stage-3 test results (no retraining).

For every variant: resample the 60 test cohorts with replacement (B = 1000, same
draws for every model, so differences are paired), compute the seed-averaged
ROC-AUC / PR-AUC of each model and of the generator ceiling, and the CI of the
paired differences listed in DIFFS. Weighted metrics on one pre-sorted order
make the bootstrap fast; with unit weights they equal sklearn's values
(checked below). Writes outputs_v2/stage3/ci.json.
"""
import json

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from common import OUT

S3 = OUT / "stage3"
VARIANTS = ["g2", "g3", "g4", "g5", "g6", "max", "dose", "spikes", "moving"]
ROWS_PER_COHORT = 8 * 4 * 48          # tracks x avatars x frames, cohort-major order
B = 1000
DIFFS = [("mpnn__MPNN", "rule__RULE"), ("mpnn__MPNN", "gat__GAT"), ("mpnn__MPNN", "edgefeat__HGB"),
         ("mpnn_temporal__MPNN", "mpnn__MPNN"), ("edgefeat_temporal__HGB", "edgefeat__HGB")]


def prep(score, y, g):
    o = np.argsort(-score, kind="stable")
    return y[o].astype(float), g[o]


def wmetrics(ys, w):
    """Weighted PR-AUC (average precision) and ROC-AUC on a pre-sorted order."""
    tp = np.cumsum(w * ys)
    fp = np.cumsum(w * (1 - ys))
    P, N = tp[-1], fp[-1]
    ap = np.sum(w * ys * tp / np.maximum(tp + fp, 1e-12)) / P
    roc = np.sum(w * (1 - ys) * tp) / (P * N)     # P(score_pos > score_neg)
    return roc, ap


def main():
    out = {}
    for v in VARIANTS:
        z = np.load(OUT / "cache" / f"s3_{v}_test.npz")
        y = z["y"]
        g = np.arange(len(y)) // ROWS_PER_COHORT
        nc = g.max() + 1
        sev_score = norm.cdf((z["oracle_score"] - 0.30) / 0.05)
        models = {"ceiling": [sev_score]}
        for f in sorted((S3 / v).glob("*_preds.npy")):
            models[f.name[:-len("_preds.npy")]] = list(np.load(f))
        sorted_ = {m: [prep(p, y, g) for p in ps] for m, ps in models.items()}
        # sanity check: unit weights reproduce sklearn
        ys, gs = sorted_["mpnn__MPNN"][0]
        r, a = wmetrics(ys, np.ones_like(ys))
        assert abs(a - average_precision_score(y, models["mpnn__MPNN"][0])) < 2e-3
        assert abs(r - roc_auc_score(y, models["mpnn__MPNN"][0])) < 1e-3

        rng = np.random.default_rng(0)
        boot = {m: [] for m in models}
        for _ in range(B):
            cnt = np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)
            for m, lst in sorted_.items():
                vals = [wmetrics(ys, cnt[gs]) for ys, gs in lst]
                boot[m].append(np.mean(vals, axis=0))
        boot = {m: np.array(b) for m, b in boot.items()}       # B x (roc, pr)
        point = {m: np.mean([wmetrics(ys, np.ones_like(ys)) for ys, gs in lst], axis=0)
                 for m, lst in sorted_.items()}
        res = {"models": {}, "diffs": {}}
        for m in models:
            res["models"][m] = {k: [float(point[m][i]), *np.percentile(boot[m][:, i], [2.5, 97.5]).tolist()]
                                for i, k in enumerate(["roc_auc", "pr_auc"])}
        for a, b in DIFFS:
            if a in boot and b in boot:
                d = boot[a][:, 1] - boot[b][:, 1]
                res["diffs"][f"{a} - {b}"] = {"pr_auc_diff": float(point[a][1] - point[b][1]),
                                              "ci": np.percentile(d, [2.5, 97.5]).tolist(),
                                              "p_le0": float(np.mean(d <= 0))}
        out[v] = res
        print(v, " ".join(f"{m.split('__')[0]}={r['pr_auc'][0]:.3f}[{r['pr_auc'][1]:.3f},{r['pr_auc'][2]:.3f}]"
                          for m, r in res["models"].items()))
        for k, d in res["diffs"].items():
            print("   ", k, f"{d['pr_auc_diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] p<=0 {d['p_le0']:.3f}")
    (S3 / "ci.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
