"""EXPLORATORY (not pre-registered; decided after seeing the lock-box result that the raw-channel HGB
leads on the unsafe target but trails on the attacker-induced target).

Scores the saved Stage-7 test predictions against the attacker-induced target (Stage-4 definition) for
every generator in which that target is defined exactly: sum aggregation over peers and no dose
accumulation (the counterfactual "remove the attacker's term" is then exact). Writes
outputs_v2/stage7/attacker_target.json. Run with XRP_GEN=v2.
"""
import numpy as np

from common import CACHE, OUT, jdump
from stage7 import ALL, DOSE
from stage7_extra import boot

OD = OUT / "stage7"
res = {}
for name, v in ALL.items():
    if name in DOSE or v.get("agg", "sum") != "sum":
        continue
    t = {k: x for k, x in np.load(CACHE / f"s7_{name}_test.npz").items()}
    s, a = t["s_noisy"].astype(np.float64), t["atk_term"].astype(np.float64)
    y_ai = ((s > 0.30) & (s - a <= 0.30)).astype(np.int64)        # atk_term is 0 on benign tracks and attacker rows
    P = {f.name[:-10]: np.load(f) for f in (OD / name).glob("*_preds.npy")}
    others = [m for m in P if m != "mpnn__MPNN"]
    r = boot(y_ai, P, [("mpnn__MPNN", m) for m in others])
    r["prevalence_attacker"] = float(y_ai.mean())
    r["share_of_unsafe"] = float(y_ai.sum() / max(1, t["y"].sum()))
    res[name] = r
    best = max(P, key=lambda m: r["models"][m]["pr_auc"][0])
    d = r["diffs"]["mpnn__MPNN - edgefeatraw__HGB"]
    print(f"{name:8s} attacker share {r['share_of_unsafe']:.2f} | MPNN {r['models']['mpnn__MPNN']['pr_auc'][0]:.3f} "
          f"rawHGB {r['models']['edgefeatraw__HGB']['pr_auc'][0]:.3f} best {best} | diff {d['diff']:+.3f} "
          f"[{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]", flush=True)
jdump(res, OD / "attacker_target.json")
