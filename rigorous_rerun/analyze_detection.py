"""
rigorous_rerun.analyze_detection
================================
All detection numbers are computed HERE from the saved out-of-fold test
predictions (outputs/detection/<model>/preds.npz) and validation-selected
thresholds (selection.json). Nothing is typed in by hand.

Reported per model:
  * per-fold metrics (25 = 5 repeats x 5 outer folds): mean +- SD
  * per-repeat pooled out-of-fold ROC-AUC / PR-AUC: mean +- SD over 5 repeats
  * hierarchical cohort bootstrap 95% CI (resample the 60 cohorts, recompute the
    pooled metric in every repeat, average over repeats; B = 1000)
  * operating points at validation-selected thresholds (F1-optimal, FPR budgets)
  * calibration (Brier, log-loss, ECE) of raw and Platt-calibrated scores
  * paired comparisons vs a reference model: Nadeau-Bengio corrected
    resampled t-test over the 25 folds + cohort-bootstrap CI of the difference
"""
from __future__ import annotations

import json
import sys

import numpy as np
from scipy import stats
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from common import OUT, SEEDS, confusion_metrics, jdump, main_tensors, op_metrics, outer_inner_splits, \
    score_metrics, thresholds_from_val

B = 1000
OPS = ("f1opt", "fpr1", "fpr5", "fpr10", "fpr20")


def load_model(name, t):
    d = OUT / "detection" / name
    z = np.load(d / "preds.npz")
    sel = json.loads((d / "selection.json").read_text())
    return {k: z[k] for k in z.files}, sel


def oracle_model(t):
    """Bayes reference: P(score + N(0,.05) > .30) from the generator's own hidden
    score. Not a model -- a ceiling. Thresholds chosen on the same validation
    cohorts as every model."""
    y, ctx = t["y"], t["ctx"]
    p_all = 1 - norm.cdf((0.30 - t["oracle_score"]) / 0.05)
    preds = {k: [] for k in ("seed", "fold", "row", "p", "p_cal")}
    sel = []
    for s in SEEDS:
        for f, tr, va, te in outer_inner_splits(ctx, y, s):
            sel.append({"seed": s, "fold": f, "thresholds": thresholds_from_val(y[va], p_all[va]), "cfg": {}})
            for k, v in (("seed", np.full(len(te), s)), ("fold", np.full(len(te), f)), ("row", te),
                         ("p", p_all[te]), ("p_cal", p_all[te])):
                preds[k].append(v)
    return {k: np.concatenate(v) for k, v in preds.items()}, sel


def prevalence_model(t):
    preds, sel = oracle_model(t)
    preds["p"] = np.full(len(preds["p"]), t["y"].mean()) + 1e-9 * np.random.default_rng(0).random(len(preds["p"]))
    preds["p_cal"] = preds["p"]
    return preds, sel


def per_fold(P, sel, y):
    rows = []
    th = {(s["seed"], s["fold"]): s["thresholds"] for s in sel}
    for s in SEEDS:
        for f in range(5):
            m = (P["seed"] == s) & (P["fold"] == f)
            if not m.any():
                continue
            yy, pp, pc = y[P["row"][m]], P["p"][m], P["p_cal"][m]
            r = {"seed": s, "fold": f}
            r.update(score_metrics(yy, pp))
            cal = score_metrics(yy, pc)
            r.update({f"cal_{k}": cal[k] for k in ("brier", "log_loss", "ece15", "ece15_adaptive")})
            for op in OPS:
                for k, v in op_metrics(yy, pp, th[(s, f)][op]).items():
                    r[f"{op}_{k}"] = v
            rows.append(r)
    return rows


def pooled(P, sel, y):
    """Per repeat: pooled OOF score metrics, and pooled confusion counts at
    the per-fold validation thresholds."""
    th = {(s["seed"], s["fold"]): s["thresholds"] for s in sel}
    out = []
    for s in SEEDS:
        m = P["seed"] == s
        yy, pp = y[P["row"][m]], P["p"][m]
        r = {"seed": s, "roc_auc": roc_auc_score(yy, pp), "pr_auc": average_precision_score(yy, pp)}
        for op in OPS:
            c = np.zeros(4, int)
            for f in range(5):
                mf = m & (P["fold"] == f)
                o = op_metrics(y[P["row"][mf]], P["p"][mf], th[(s, f)][op])
                c += [o["tp"], o["fp"], o["fn"], o["tn"]]
            for k, v in confusion_metrics(*c).items():
                r[f"{op}_{k}"] = v
        out.append(r)
    return out


def _rank_by_seed(P, y, groups):
    """Pre-index per repeat for the bootstrap."""
    per = []
    for s in SEEDS:
        m = np.where(P["seed"] == s)[0]
        rows = P["row"][m]
        per.append((y[rows], P["p"][m], groups[rows]))
    return per


def hier_boot(P, y, groups, fn, n=B, seed=0, P2=None):
    """Cohort bootstrap of the repeat-averaged pooled metric (or of the
    difference model - P2 when P2 is given; P and P2 share rows)."""
    rng = np.random.default_rng(seed)
    cs = np.unique(groups)
    per = _rank_by_seed(P, y, groups)
    per2 = _rank_by_seed(P2, y, groups) if P2 is not None else None
    idx_by = []
    for k, (yy, pp, gg) in enumerate(per):
        by = {c: np.where(gg == c)[0] for c in cs}
        idx_by.append(by)
        if per2 is not None:   # align P2 rows to P rows
            o1 = np.argsort(P["row"][P["seed"] == SEEDS[k]]); o2 = np.argsort(P2["row"][P2["seed"] == SEEDS[k]])
            p2 = np.empty_like(per2[k][1]); p2[o1] = per2[k][1][o2]
            per2[k] = (yy, p2, gg)
    vals = []
    for _ in range(n):
        pick = rng.choice(cs, len(cs))
        v = []
        for k, (yy, pp, gg) in enumerate(per):
            idx = np.concatenate([idx_by[k][c] for c in pick])
            if yy[idx].sum() == 0:
                continue
            a = fn(yy[idx], pp[idx])
            if per2 is not None:
                a -= fn(yy[idx], per2[k][1][idx])
            v.append(a)
        if v:
            vals.append(np.mean(v))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def nb_ttest(diffs, ratio=12 / 36):
    """Nadeau & Bengio (2003) corrected resampled t-test; ratio = n_test/n_train."""
    d = np.asarray(diffs); J = len(d)
    var = d.var(ddof=1)
    t_ = d.mean() / np.sqrt((1 / J + ratio) * var) if var > 0 else np.inf
    p = 2 * stats.t.sf(abs(t_), J - 1)
    return {"mean_diff": float(d.mean()), "sd_diff": float(d.std(ddof=1)), "t": float(t_), "p": float(p),
            "wins": int((d > 0).sum()), "n": J}


def summarise(rows):
    keys = [k for k in rows[0] if k not in ("seed", "fold")]
    return {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))] for k in keys}


def analyse(names, ref="proto__MLPpub", boot=True):
    t = main_tensors()
    y, groups = t["y"], t["ctx"][:, 0]
    res = {}
    loaded = {}
    for n in names:
        if n == "ORACLE":
            P, sel = oracle_model(t)
        elif n == "PREVALENCE":
            P, sel = prevalence_model(t)
        else:
            P, sel = load_model(n, t)
        assert all((np.sort(P["row"][P["seed"] == s]) == np.arange(len(y))).all() for s in SEEDS), \
            f"{n}: every frame must be scored exactly once per repeat"
        loaded[n] = P
        pf, pl = per_fold(P, sel, y), pooled(P, sel, y)
        r = {"per_fold": summarise(pf), "pooled_by_repeat": summarise(pl),
             "selected_cfgs": [s.get("cfg") for s in sel]}
        if boot:
            r["ci95_roc_auc"] = hier_boot(P, y, groups, roc_auc_score)
            r["ci95_pr_auc"] = hier_boot(P, y, groups, average_precision_score)
        r["_fold_rows"] = pf
        res[n] = r
        print(f"{n:32s} ROC {r['pooled_by_repeat']['roc_auc'][0]:.3f}±{r['pooled_by_repeat']['roc_auc'][1]:.3f} "
              f"PR {r['pooled_by_repeat']['pr_auc'][0]:.3f}±{r['pooled_by_repeat']['pr_auc'][1]:.3f} "
              f"| fold ROC {r['per_fold']['roc_auc'][0]:.3f}±{r['per_fold']['roc_auc'][1]:.3f} "
              f"PR {r['per_fold']['pr_auc'][0]:.3f}±{r['per_fold']['pr_auc'][1]:.3f} "
              f"| F1opt F1 {r['pooled_by_repeat']['f1opt_f1'][0]:.3f} MCC {r['pooled_by_repeat']['f1opt_mcc'][0]:.3f} "
              + (f"CI ROC {r['ci95_roc_auc']} PR {r['ci95_pr_auc']}" if boot else ""), flush=True)
    if ref in loaded:
        for n in names:
            if n == ref or n in ("ORACLE", "PREVALENCE"):
                continue
            a = {(r["seed"], r["fold"]): r for r in res[n]["_fold_rows"]}
            b = {(r["seed"], r["fold"]): r for r in res[ref]["_fold_rows"]}
            comp = {}
            for met in ("roc_auc", "pr_auc", "f1opt_f1", "f1opt_mcc", "brier"):
                comp[met] = nb_ttest([a[k][met] - b[k][met] for k in a])
            if boot:
                comp["boot_ci95_delta_roc_auc"] = hier_boot(loaded[n], y, groups, roc_auc_score, P2=loaded[ref], n=500)
                comp["boot_ci95_delta_pr_auc"] = hier_boot(loaded[n], y, groups, average_precision_score, P2=loaded[ref], n=500)
            res[n][f"vs_{ref}"] = comp
    return res


if __name__ == "__main__":
    args = sys.argv[1:]
    out = args[0]
    names = args[1:]
    res = analyse(names)
    jdump(res, OUT / f"{out}.json")
