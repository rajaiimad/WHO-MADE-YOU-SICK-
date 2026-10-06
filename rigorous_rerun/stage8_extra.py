"""
rigorous_rerun.stage8_extra
===========================
Analyses requested by the round-3 review that need no new training except attribution on the
within-band attacker (trained on the v2 main corpus exactly as in Stage 3). Run with XRP_GEN=v2.

  1. Holm correction of the paired MPNN - raw-channel HGB comparisons over the distinct detectable
     generators (two-sided bootstrap p = 2 min(P(d<=0), P(d>0)), floor 1/B).
  2. Median training-seed SD of PR-AUC per detector over the detectable family generators.
  3. Family-level reference scores per detector: mean and worst-case ceiling-normalised PR-AUC
     (PR-AUC / ceiling PR-AUC) over the detectable family generators, within-generator and transferred
     from v2, and the number of generators won.
  4. Source attribution on the within-band attacker: rule vs learned per-edge HGB (channel-specific
     features, trained on the v2 main corpus), top-1 and edge PR-AUC, with the standard attack as reference.
-> outputs_v2/stage8/extra.json
"""
import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CACHE, OUT, SEEDS, jdump, peer_intensity, thresholds_from_val

S7 = OUT / "stage7"
J = lambda p: json.loads(p.read_text())
B = 1000
DET = 0.15
BASE = ["proto__MLPpub", "rule__RULE", "edgefeat__HGB", "gat__GAT", "mpnn__MPNN", "edgefeatraw__HGB"]


def holm(ps):
    order = np.argsort(ps)
    m = len(ps)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * ps[i]))
        adj[i] = run
    return adj


def main():
    from stage7 import ALL, DOSE, RANDOM, STRUCT
    ci7 = J(S7 / "ci.json")
    ex = J(S7 / "extra.json")
    ci3 = J(OUT / "stage3" / "ci.json")
    res = {}
    # 1. Holm over distinct detectable generators (stage-3 'dose' == stage-7 'dose_ramp': counted once)
    rows = []
    for n in list(STRUCT) + list(RANDOM) + list(DOSE):
        if ci7[n]["models"]["ceiling"]["pr_auc"][0] >= DET:
            rows.append((n, ci7[n]["diffs"]["mpnn__MPNN - edgefeatraw__HGB"]))
    for n in ["g2", "g3", "g4", "g5", "g6", "max", "spikes", "moving"]:
        rows.append((n, ex["stage3"][n]["diffs"]["mpnn__MPNN - edgefeatraw__HGB"]))
    p = np.array([max(1.0 / B, 2 * min(d["p_le0"], 1 - d["p_le0"])) for _, d in rows])
    adj = holm(p)
    res["holm"] = {"n": len(rows), "rows": [{"gen": n, "diff": d["diff"], "ci": d["ci"], "p": float(pp), "p_holm": float(a)}
                                           for (n, d), pp, a in zip(rows, p, adj)]}
    res["holm"]["mpnn_sig"] = [r["gen"] for r in res["holm"]["rows"] if r["p_holm"] < 0.05 and r["diff"] > 0]
    res["holm"]["raw_sig"] = [r["gen"] for r in res["holm"]["rows"] if r["p_holm"] < 0.05 and r["diff"] < 0]
    # per-edge winner count over the distinct detectable generators
    win = {}
    for n in list(STRUCT) + list(RANDOM) + list(DOSE):
        c = ci7[n]
        if c["models"]["ceiling"]["pr_auc"][0] >= DET:
            pr = {m: c["models"][m]["pr_auc"][0] for m in BASE}
            win[n] = max(pr, key=pr.get)
    for n in ["g2", "g3", "g4", "g5", "g6", "max", "spikes", "moving"]:
        pr = {m: ci3[n]["models"][m]["pr_auc"][0] for m in BASE if m in ci3[n]["models"]}
        pr["edgefeatraw__HGB"] = ex["stage3"][n]["models"]["edgefeatraw__HGB"]["pr_auc"][0]
        win[n] = max(pr, key=pr.get)
    res["winners"] = win
    res["n_distinct"] = len(win)
    res["n_full_state_wins"] = sum(w in ("mpnn__MPNN", "edgefeatraw__HGB") for w in win.values())
    # 2. median seed SD
    det = [n for n in list(STRUCT) + list(RANDOM) + list(DOSE) if ci7[n]["models"]["ceiling"]["pr_auc"][0] >= DET]
    res["seed_sd_median"] = {m: float(np.median([J(S7 / n / "summary.json")["models"][m]["pr_auc"][1] for n in det])) for m in BASE}
    # 3. family-level reference scores
    tr = J(S7 / "main" / "summary.json")["transfer"]
    ref = {}
    for m in BASE:
        within = [ci7[n]["models"][m]["pr_auc"][0] / ci7[n]["models"]["ceiling"]["pr_auc"][0] for n in det]
        trans = [tr[n][m] / ci7[n]["models"]["ceiling"]["pr_auc"][0] for n in det if n in tr]
        ref[m] = {"mean_norm_within": float(np.mean(within)), "min_norm_within": float(np.min(within)),
                  "mean_norm_transfer": float(np.mean(trans)), "min_norm_transfer": float(np.min(trans)),
                  "wins": sum(win[n] == m for n in det)}
    res["reference"] = ref
    res["n_det_family"] = len(det)
    res["n_transfer"] = len([n for n in det if n in tr])
    # 4. attribution on the within-band attacker
    from run_attribution import edge_tables, hgb, ranking_metrics
    from run_final import train_val_split
    dev = {k: v for k, v in np.load(CACHE / "s4_main.npz").items()}
    tests = {"standard70k": {k: v for k, v in np.load(CACHE / "s5_standard.npz").items()},
             "near70k": {k: v for k, v in np.load(CACHE / "s7_near_near.npz").items()}}
    Kp = dev["y_att"].shape[1]
    Fd = edge_tables(dev)["paper"]
    yad = dev["y_att"].ravel()
    att = {}
    for c, t in tests.items():
        Ft = edge_tables(t)["paper"]
        yat = t["y_att"].ravel()
        ex_ = peer_intensity(t["E_chan"]).max(-1)
        rule = (ex_ * (0.5 + 0.5 * t["E_geo"][..., 1]) / (t["E_geo"][..., 0] + 0.5)).ravel()
        rr = {"roc_auc": float(roc_auc_score(yat, rule)), "pr_auc": float(average_precision_score(yat, rule))}
        rr.update({k: v for k, v in ranking_metrics(yat, rule, Kp, np.inf).items() if k in ("top1", "top2")})
        L = []
        for s in SEEDS:
            tr_, va = train_val_split(dev["ctx"], dev["y"], s)
            etr = (tr_[:, None] * Kp + np.arange(Kp)).ravel()
            eva = (va[:, None] * Kp + np.arange(Kp)).ravel()
            m = hgb(s).fit(Fd[etr], yad[etr])
            pt = m.predict_proba(Ft)[:, 1]
            thr = thresholds_from_val(yad[eva], m.predict_proba(Fd[eva])[:, 1])["f1opt"]
            r = {"roc_auc": float(roc_auc_score(yat, pt)), "pr_auc": float(average_precision_score(yat, pt))}
            r.update({k: v for k, v in ranking_metrics(yat, pt, Kp, thr).items() if k in ("top1", "top2")})
            L.append(r)
        att[c] = {"rule": rr, "learned": {k: float(np.mean([x[k] for x in L])) for k in L[0]},
                  "n_sources": int(yat.sum())}
        print(c, att[c], flush=True)
    res["attribution_near"] = att
    jdump(res, OUT / "stage8" / "extra.json")
    print("holm", res["holm"]["n"], "mpnn sig", res["holm"]["mpnn_sig"], "raw sig", res["holm"]["raw_sig"])
    print("winners", res["n_full_state_wins"], "/", res["n_distinct"], {n: w for n, w in win.items() if w not in ("mpnn__MPNN", "edgefeatraw__HGB")})
    print("seed sd", res["seed_sd_median"])
    print("reference", json.dumps(ref, indent=0))


if __name__ == "__main__":
    main()
