"""
rigorous_rerun.stage8_lockbox2
==============================
PREREGISTRATION_v8.md: a fresh confirmation lock-box (seeds 80000-80119), scored once.
Every model is trained once per seed (5 seeds, 45/15 cohort split) on the v2 main corpus with its
modal nested-CV configuration, exactly as for Table V. Run with XRP_GEN=v2.
"""
import time

import numpy as np

from common import CACHE, OUT, SEED_RANGES, jdump
from stage3 import S3, V2_GAIN, generate
from stage7 import train
from stage7_extra import boot

SEED0, N_COH = 80_000, 120
for r in list(SEED_RANGES.values()) + list(S3.values()) + [(70_000, 70_060)]:
    assert SEED0 + N_COH <= r[0] or SEED0 >= r[1], "lock-box 2 seeds overlap an earlier corpus"
MODELS = ["mpnn__MPNN", "edgefeatraw__HGB", "edgefeat__HGB", "gat__GAT", "rule__RULE", "proto__MLPpub"]


def main():
    import baselines  # noqa: F401
    import stage5
    from run_detection import SPECS
    t0 = time.time()
    f = CACHE / "s8_lockbox2.npz"
    if f.exists():
        lb2 = {k: v for k, v in np.load(f).items()}
    else:
        lb2 = generate(SEED0, N_COH, {}, V2_GAIN)
        np.savez_compressed(f, **lb2)
    dev = {k: v for k, v in np.load(CACHE / "s4_main.npz").items()}
    print(f"lock-box 2: {len(lb2['y'])} victim-frames, prevalence {lb2['y'].mean():.4f} ({time.time() - t0:.0f}s)", flush=True)
    P = {}
    od = OUT / "stage8"
    od.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        fam, builder = SPECS[m]
        src = None if fam in ("MLPpub", "RULE") else m
        rs = train(fam, builder, src, dev, {"lb2": lb2})
        P[m] = np.array([r["pred"]["lb2"] for r in rs], np.float32)
        np.save(od / f"{m}_lb2.npy", P[m])
        print(f"{m} done ({time.time() - t0:.0f}s)", flush=True)
    y = lb2["y"]
    y_ai, _ = stage5.labels(lb2)
    pairs = [("mpnn__MPNN", m) for m in MODELS if m != "mpnn__MPNN"]
    from scipy.stats import norm
    P_ceil = dict(P, ceiling=norm.cdf((lb2["oracle_score"] - 0.30) / 0.05)[None])
    res = {"prevalence_unsafe": float(y.mean()), "prevalence_attacker": float(y_ai.mean()),
           "unsafe": boot(y, P_ceil, pairs), "attacker": boot(y_ai, P, pairs)}
    jdump(res, od / "lockbox2.json")
    for tgt in ("unsafe", "attacker"):
        print(tgt, {m: round(v["pr_auc"][0], 3) for m, v in res[tgt]["models"].items()})
        for k, d in res[tgt]["diffs"].items():
            print("   ", k, f"{d['diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
