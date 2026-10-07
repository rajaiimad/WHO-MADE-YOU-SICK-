"""Round-4 review: uncertainty of the family win count (exploratory reporting addition).

For each detectable family generator, a paired cohort bootstrap (B = 1000, the Stage-7 resampling) of
  best full-state detector (max of MPNN and raw-channel HGB, chosen within each resample)
  minus best other detector (max of prototype, rule, excess-only HGB, GAT, chosen within each resample),
on the unsafe target. Counts the generators in which the 95% CI excludes 0. Also reports each family
generator's development prevalence (to explain why some transferred models beat retrained ones).
-> outputs_v2/stage9/wincount.json. Run with XRP_GEN=v2.
"""
import json

import numpy as np

from common import CACHE, OUT, jdump
from stage3_ci import B, prep, wmetrics
from stage7 import RANDOM, STRUCT, DOSE

S7 = OUT / "stage7"
ROWS = 8 * 4 * 48
FULL = ["mpnn__MPNN", "edgefeatraw__HGB"]
OTHER = ["proto__MLPpub", "rule__RULE", "edgefeat__HGB", "gat__GAT"]


def main():
    ci7 = json.loads((S7 / "ci.json").read_text())
    res = {}
    for n in list(STRUCT) + list(RANDOM) + list(DOSE):
        if ci7[n]["models"]["ceiling"]["pr_auc"][0] < 0.15:
            continue
        y = np.load(CACHE / f"s7_{n}_test.npz")["y"]
        g = np.arange(len(y)) // ROWS
        nc = g.max() + 1
        srt = {m: [prep(p, y, g) for p in np.load(S7 / n / f"{m}_preds.npy")] for m in FULL + OTHER}
        rng = np.random.default_rng(0)
        d = []
        for _ in range(B):
            cnt = np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)
            pr = {m: np.mean([wmetrics(ys, cnt[gs])[1] for ys, gs in L]) for m, L in srt.items()}
            d.append(max(pr[m] for m in FULL) - max(pr[m] for m in OTHER))
        summ = json.loads((S7 / n / "summary.json").read_text())
        res[n] = {"ci": np.percentile(d, [2.5, 97.5]).tolist(), "mean": float(np.mean(d)),
                  "prevalence_dev": summ["prevalence_dev"]}
        print(n, np.round(res[n]["ci"], 3), round(summ["prevalence_dev"], 4), flush=True)
    res["_n"] = len([k for k in res if not k.startswith("_")])
    res["_ci_excludes_0"] = [k for k, v in res.items() if not k.startswith("_") and v["ci"][0] > 0]
    res["_v2_prevalence_dev"] = float(np.load(CACHE / "s4_main.npz")["y"].mean())
    jdump(res, OUT / "stage9" / "wincount.json")
    print("CI>0 in", len(res["_ci_excludes_0"]), "of", res["_n"], res["_ci_excludes_0"])


if __name__ == "__main__":
    main()
