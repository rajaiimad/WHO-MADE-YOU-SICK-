"""
rigorous_rerun.stage9_near_benign
=================================
PREREGISTRATION_v9.md: proximity-only attacker (`near_benign`). The attacker stands where the v7
within-band attacker stands, but all its channels are drawn from the benign band like a benign peer's.
Detectors (modal nested-CV configurations, 5 seeds) and the attribution model are trained once on the
v2 main corpus; the standard attack and the v7 within-band attacker (top-of-band) are scored in the
same run for reference. Run with XRP_GEN=v2. -> outputs_v2/stage9/near_benign.json
"""
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CACHE, OUT, SEEDS, jdump, peer_intensity, thresholds_from_val
from stage3 import V2_GAIN, generate
from stage7 import train

MODELS = ["mpnn__MPNN", "edgefeatraw__HGB", "edgefeat__HGB", "gat__GAT", "rule__RULE", "proto__MLPpub"]


def main():
    import baselines  # noqa: F401
    import stage5
    from run_attribution import edge_tables, hgb, ranking_metrics
    from run_detection import SPECS
    from run_final import train_val_split
    t0 = time.time()
    f = CACHE / "s9_near_benign.npz"
    if f.exists():
        nb = {k: v for k, v in np.load(f).items()}
    else:
        nb = generate(70_000, 60, {"near": True, "near_clamp": False}, V2_GAIN)
        np.savez_compressed(f, **nb)
    load = lambda p: {k: v for k, v in np.load(CACHE / p).items()}
    tests = {"standard": load("s5_standard.npz"), "near_top": load("s7_near_near.npz"), "near_benign": nb}
    dev = load("s4_main.npz")
    res = {"harm": {}, "detection": {}, "attribution": {}}
    for c, t in tests.items():
        y_ai, p_ai = stage5.labels(t)
        res["harm"][c] = {"attacker_induced": int(y_ai.sum()), "expected": float(p_ai.sum())}
    for c in tests:
        res["harm"][c]["relative"] = res["harm"][c]["attacker_induced"] / res["harm"]["standard"]["attacker_induced"]
    print("harm", res["harm"], flush=True)
    for m in MODELS:
        fam, builder = SPECS[m]
        rs = train(fam, builder, None if fam in ("MLPpub", "RULE") else m, dev, tests)
        res["detection"][m] = {}
        for c, t in tests.items():
            ev = [stage5.evaluate(t, r["pred"][c], r["thr"]) for r in rs]
            res["detection"][m][c] = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]}
        print(m, {c: round(v["pr_attacker"], 3) for c, v in res["detection"][m].items()},
              {c: round(v["recall_attacker_fpr1"], 3) for c, v in res["detection"][m].items()}, f"({time.time() - t0:.0f}s)", flush=True)
    Kp = dev["y_att"].shape[1]
    Fd, yad = edge_tables(dev)["paper"], dev["y_att"].ravel()
    models = []
    for s in SEEDS:
        tr, va = train_val_split(dev["ctx"], dev["y"], s)
        etr = (tr[:, None] * Kp + np.arange(Kp)).ravel()
        eva = (va[:, None] * Kp + np.arange(Kp)).ravel()
        m = hgb(s).fit(Fd[etr], yad[etr])
        models.append((m, thresholds_from_val(yad[eva], m.predict_proba(Fd[eva])[:, 1])["f1opt"]))
    for c, t in tests.items():
        Ft, yat = edge_tables(t)["paper"], t["y_att"].ravel()
        ex = peer_intensity(t["E_chan"]).max(-1)
        rule = (ex * (0.5 + 0.5 * t["E_geo"][..., 1]) / (t["E_geo"][..., 0] + 0.5)).ravel()
        rr = {"roc_auc": float(roc_auc_score(yat, rule)), "pr_auc": float(average_precision_score(yat, rule))}
        rr.update({k: v for k, v in ranking_metrics(yat, rule, Kp, np.inf).items() if k in ("top1", "top2")})
        L = []
        for m, thr in models:
            pt = m.predict_proba(Ft)[:, 1]
            r = {"roc_auc": float(roc_auc_score(yat, pt)), "pr_auc": float(average_precision_score(yat, pt))}
            r.update({k: v for k, v in ranking_metrics(yat, pt, Kp, thr).items() if k in ("top1", "top2")})
            L.append(r)
        res["attribution"][c] = {"rule": rr, "learned": {k: float(np.mean([x[k] for x in L])) for k in L[0]}}
        print("attribution", c, {k: round(v["top1"], 3) for k, v in res["attribution"][c].items()}, flush=True)
    (OUT / "stage9").mkdir(parents=True, exist_ok=True)
    jdump(res, OUT / "stage9" / "near_benign.json")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
