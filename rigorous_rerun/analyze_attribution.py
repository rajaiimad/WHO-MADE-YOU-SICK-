"""Summarise outputs/attribution/per_fold.json -> outputs/attribution_summary.json
(mean +- SD over the 25 nested folds; LOCO per held-out channel and the
channel mean; Nadeau-Bengio test of relative/agnostic vs paper features)."""
from __future__ import annotations

import json

import numpy as np

from analyze_detection import nb_ttest
from common import OUT, PEER_NAMES, jdump

KEYS = ("roc_auc", "pr_auc", "top1", "top2", "precision", "recall", "f1", "macro_f1",
        "false_attribution_rate", "spurious_rate_no_source", "prevalence")


def summ(rows):
    return {k: [float(np.nanmean([r.get(k, np.nan) for r in rows])), float(np.nanstd([r.get(k, np.nan) for r in rows], ddof=1)), len(rows)]
            for k in KEYS if any(k in r for r in rows)}


def main():
    R = json.loads((OUT / "attribution" / "per_fold.json").read_text())
    out = {"in_distribution": {k: summ(v) for k, v in R["in_distribution"].items()}, "loco": {}, "loco_mean": {},
           "paired": {}}
    for k, v in R["loco"].items():
        out["loco"][k] = summ(v)
    for fs in ("paper", "relative", "agnostic"):
        out["loco_mean"][fs] = {m: float(np.mean([out["loco"][f"{fs}|{c}"][m][0] for c in PEER_NAMES]))
                                for m in ("roc_auc", "pr_auc", "top1")}
    key = lambda r: (r["seed"], r["fold"])
    for fs in ("relative", "agnostic"):
        a = {key(r): r for r in R["in_distribution"][fs]}; b = {key(r): r for r in R["in_distribution"]["paper"]}
        out["paired"][f"{fs}_vs_paper_in_dist"] = {m: nb_ttest([a[k][m] - b[k][m] for k in a]) for m in ("roc_auc", "pr_auc", "top1")}
        # LOCO: pair per (channel, seed, fold)
        da = {m: [] for m in ("roc_auc", "top1")}
        for c in PEER_NAMES:
            a = {key(r): r for r in R["loco"][f"{fs}|{c}"]}; b = {key(r): r for r in R["loco"][f"paper|{c}"]}
            for m in da:
                da[m] += [a[k][m] - b[k][m] for k in a if k in b]
        out["paired"][f"{fs}_vs_paper_loco_all_channels"] = {m: nb_ttest(v) for m, v in da.items()}
    jdump(out, OUT / "attribution_summary.json")
    for k, v in out["in_distribution"].items():
        print(f"{k:28s} ROC {v['roc_auc'][0]:.3f}±{v['roc_auc'][1]:.3f} PR {v['pr_auc'][0]:.3f} top1 {v['top1'][0]:.3f} "
              f"top2 {v['top2'][0]:.3f} P {v['precision'][0]:.3f} R {v['recall'][0]:.3f} mF1 {v['macro_f1'][0]:.3f} "
              f"spur {v['spurious_rate_no_source'][0]:.3f}")
    for k, v in out["loco"].items():
        print(f"LOCO {k:40s} ROC {v['roc_auc'][0]:.3f}±{v['roc_auc'][1]:.3f} PR {v['pr_auc'][0]:.3f} top1 {v['top1'][0]:.3f} n={v['roc_auc'][2]}")
    print("LOCO mean", out["loco_mean"])
    print(json.dumps(out["paired"], indent=1))


if __name__ == "__main__":
    main()
