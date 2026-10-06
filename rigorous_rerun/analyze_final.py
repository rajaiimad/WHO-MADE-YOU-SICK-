"""
rigorous_rerun.analyze_final
============================
Metrics for Stage B (models trained once per seed on the main corpus) on every
external corpus, computed from outputs/final/<model>/seed<s>/preds.npz.

Per (model, corpus): mean +- SD over the 5 seeds of ROC-AUC, PR-AUC, calibration
and operating-point metrics at the VALIDATION-selected thresholds (main-corpus
validation cohorts), plus a cohort bootstrap 95% CI of the seed-averaged metric.
Adaptive-attacker corpora additionally report harm (unsafe-frame rate on
attacked tracks) and missed harm at the operating point.
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from common import OUT, SEEDS, jdump, main_tensors, op_metrics, score_metrics, thresholds_from_val
from run_final import FINAL_MODELS, external_corpora


THRESH2 = {"shift_thresh": 0.40}


def boot(groups, y, P, fn, n, seed=0):
    """P: (n_seeds, n_rows) scores; resample cohorts, average fn over seeds."""
    rng = np.random.default_rng(seed)
    cs = np.unique(groups)
    by = {c: np.where(groups == c)[0] for c in cs}
    v = []
    for _ in range(n):
        idx = np.concatenate([by[c] for c in rng.choice(cs, len(cs))])
        if 0 < y[idx].sum() < len(idx):
            v.append(np.mean([fn(y[idx], p[idx]) for p in P]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def oracle_p(c, name):
    thr = THRESH2.get(name, 0.30)
    return 1 - norm.cdf((thr - c["oracle_score"]) / 0.05)


def summarise(rows):
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))] for k in keys}


def main(models=None, n_boot_lock=1000, n_boot=0):
    C = external_corpora()
    main_t = main_tensors()
    res = {}
    for m in (models or FINAL_MODELS + ["ORACLE"]):
        res[m] = {}
        for cname, c in C.items():
            y, g = c["y"], c["ctx"][:, 0]
            if m == "ORACLE":
                if cname in ("shift_decouple", "lockbox_peerperm"):
                    continue    # hidden score not recoverable after the shuffle
                # (param corpora store oracle_score with their own coefficients: common.param_tensors)
            rows, P = [], []
            for s in SEEDS:
                if m == "ORACLE":
                    meta = json.loads((OUT / "final" / FINAL_MODELS[0] / f"seed{s}" / "meta.json").read_text())
                    va = np.isin(main_t["ctx"][:, 0], meta["val_cohorts"])
                    pm = 1 - norm.cdf((0.30 - main_t["oracle_score"]) / 0.05)
                    thr = thresholds_from_val(main_t["y"][va], pm[va])
                    p = pc = oracle_p(c, cname)
                else:
                    d = OUT / "final" / m / f"seed{s}"
                    z = np.load(d / "preds.npz")
                    thr = json.loads((d / "meta.json").read_text())["thresholds"]
                    p, pc = z[f"p__{cname}"], z[f"pcal__{cname}"]
                r = score_metrics(y, p)
                cal = score_metrics(y, pc)
                r.update({f"cal_{k}": cal[k] for k in ("brier", "log_loss", "ece15", "ece15_adaptive")})
                for op in ("f1opt", "fpr10"):
                    r.update({f"{op}_{k}": v for k, v in op_metrics(y, p, thr[op]).items()})
                if cname.startswith("adapt") or cname == "lockbox":
                    atk = c["atk_id"] >= 0
                    yh = p >= thr["fpr10"]
                    n_atk_tracks = len({(a, b) for a, b in c["ctx"][atk][:, :2]})
                    r["harm_unsafe_rate_attacked_tracks"] = float(y[atk].mean())
                    r["missed_unsafe_per_attacked_track_fpr10"] = float((y[atk] & ~yh[atk]).sum() / n_atk_tracks)
                    r["unsafe_per_attacked_track"] = float(y[atk].sum() / n_atk_tracks)
                rows.append(r); P.append(p)
                if m == "ORACLE":
                    break
            out = {"by_seed": summarise(rows) if len(rows) > 1 else {k: [v, 0.0] for k, v in rows[0].items()}}
            nb = n_boot_lock if cname == "lockbox" else n_boot
            if y.sum() > 0 and nb > 0:
                out["ci95_roc_auc"] = boot(g, y, np.array(P), roc_auc_score, nb)
                out["ci95_pr_auc"] = boot(g, y, np.array(P), average_precision_score, nb)
            res[m][cname] = out
            print(f"{m:20s} {cname:18s} ROC {out['by_seed']['roc_auc'][0]:.3f}±{out['by_seed']['roc_auc'][1]:.3f} "
                  f"PR {out['by_seed']['pr_auc'][0]:.3f}±{out['by_seed']['pr_auc'][1]:.3f} "
                  f"prev {out['by_seed']['prevalence'][0]:.4f}", flush=True)
        jdump(res[m], OUT / "final_external" / f"{m}.json")   # one file per model


if __name__ == "__main__":
    main(sys.argv[1:] or None)
