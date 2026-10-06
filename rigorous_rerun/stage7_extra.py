"""
rigorous_rerun.stage7_extra
===========================
Completes pre-registered item E of PREREGISTRATION_v7.md ("edgefeatraw is added to every variant"):
  1. lock-box row for the raw-channel interaction HGB (all Table V metrics) and both targets,
     paired against the MPNN trained in the same Stage-7 run;
  2. the raw-channel HGB on every Stage-3 variant (paired against the saved Stage-3 MPNN predictions);
  3. the raw-channel HGB on the deterministic Stage-5 attacks (standard, out-of-view, out-of-view >= 1 m,
     channel switching, colluding split). The black-box corpus depends on a queried MPNN and is not rebuilt.
Writes outputs_v2/stage7/extra.json. Run with XRP_GEN=v2.
"""
from __future__ import annotations

import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CACHE, OUT, jdump
from stage3_ci import B, prep, wmetrics
from stage7 import edge_raw_features, train

OD = OUT / "stage7"
ROWS = 8 * 4 * 48


def boot(Y, P, pairs, metric=1):
    g = np.arange(len(Y)) // ROWS
    nc = g.max() + 1
    srt = {m: [prep(p, Y, g) for p in ps] for m, ps in P.items()}
    point = {m: np.mean([wmetrics(ys, np.ones_like(ys)) for ys, _ in L], 0) for m, L in srt.items()}
    rng = np.random.default_rng(0)
    bs = {m: [] for m in srt}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)
        for m, L in srt.items():
            bs[m].append(np.mean([wmetrics(ys, cnt[gs]) for ys, gs in L], 0))
    bs = {m: np.array(b) for m, b in bs.items()}
    out = {"models": {m: {k: [float(point[m][i]), *np.percentile(bs[m][:, i], [2.5, 97.5]).tolist()]
                          for i, k in enumerate(("roc_auc", "pr_auc"))} for m in P}, "diffs": {}}
    for a, b in pairs:
        d = bs[a][:, metric] - bs[b][:, metric]
        out["diffs"][f"{a} - {b}"] = {"diff": float(point[a][metric] - point[b][metric]),
                                      "ci": np.percentile(d, [2.5, 97.5]).tolist(), "p_le0": float(np.mean(d <= 0))}
    return out


def op_row(y, P, thrs):
    """F1 / MCC at the validation F1-optimal threshold, recall at the 10% budget, Brier (raw scores)."""
    r = {"f1": [], "mcc": [], "recall_fpr10": [], "brier": []}
    for p, th in zip(P, thrs):
        yh = p >= th["f1opt"]
        tp = int((yh & (y == 1)).sum()); fp = int((yh & (y == 0)).sum())
        fn = int((~yh & (y == 1)).sum()); tn = int((~yh & (y == 0)).sum())
        prec, rec = tp / max(1, tp + fp), tp / max(1, tp + fn)
        r["f1"].append(2 * prec * rec / max(1e-12, prec + rec))
        den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        r["mcc"].append((tp * tn - fp * fn) / den if den else 0.0)
        r["recall_fpr10"].append(float((p[y == 1] >= th["fpr10"]).mean()))
        r["brier"].append(float(np.mean((p - y) ** 2)))
    return {k: float(np.mean(v)) for k, v in r.items()}


def main():
    import stage5
    t0 = time.time()
    res = {}
    # 1. lock-box
    lb = {k: v for k, v in np.load(CACHE / "s4_lockbox.npz").items()}
    y = lb["y"]
    y_ai, _ = stage5.labels(lb)
    runs = {m: np.load(OD / "main" / f"{m}.npz") for m in ("edgefeatraw__HGB", "mpnn__MPNN", "gat__GAT", "edgefeat__HGB")}
    P = {m: r["lockbox"] for m, r in runs.items()}
    pairs = [("mpnn__MPNN", m) for m in P if m != "mpnn__MPNN"]
    res["lockbox_unsafe"] = boot(y, P, pairs)
    res["lockbox_attacker"] = boot(y_ai, P, pairs)
    res["lockbox_unsafe_roc"] = boot(y, P, pairs, metric=0)["diffs"]
    res["lockbox_ops"] = {m: op_row(y, P[m], json.loads(str(runs[m]["thr"]))) for m in P}
    print("lockbox", json.dumps({k: res[k]["diffs"] for k in ("lockbox_unsafe", "lockbox_attacker")}, indent=0)[:1500], flush=True)

    # 2. Stage-3 variants
    from stage7 import model_specs
    fam, builder, src = model_specs()["edgefeatraw__HGB"]
    res["stage3"] = {}
    for v in ["g2", "g3", "g4", "g5", "g6", "max", "dose", "spikes", "moving"]:
        dev = {k: x for k, x in np.load(CACHE / f"s3_{v}_dev.npz").items()}
        test = {k: x for k, x in np.load(CACHE / f"s3_{v}_test.npz").items()}
        rs = train(fam, builder, src, dev, {"test": test})
        Pv = {"edgefeatraw__HGB": np.array([r["pred"]["test"] for r in rs], np.float32),
              "mpnn__MPNN": np.load(OUT / "stage3" / v / "mpnn__MPNN_preds.npy")}
        res["stage3"][v] = boot(test["y"], Pv, [("mpnn__MPNN", "edgefeatraw__HGB")])
        d = res["stage3"][v]["diffs"]["mpnn__MPNN - edgefeatraw__HGB"]
        print(f"stage3 {v}: raw {res['stage3'][v]['models']['edgefeatraw__HGB']['pr_auc'][0]:.3f} "
              f"mpnn {res['stage3'][v]['models']['mpnn__MPNN']['pr_auc'][0]:.3f} diff {d['diff']:+.3f} {np.round(d['ci'], 3)} "
              f"({time.time() - t0:.0f}s)", flush=True)

    # 3. Stage-5 attacks (deterministic ones)
    std = stage5.base("standard", {})
    tests = {"standard": std, "oov_audio": stage5.base("oov_audio", {"oov": True}),
             "oov_audio_d1": stage5.base("oov_audio_d1", {"oov": True, "oov_min_d": 1.0}),
             "switching": stage5.switching(std), "split": stage5.split(std)}
    dev = {k: x for k, x in np.load(CACHE / "s4_main.npz").items()}
    rs = train(fam, builder, src, dev, tests)
    res["stage5"] = {}
    for c, t in tests.items():
        ev = [stage5.evaluate(t, r["pred"][c], r["thr"]) for r in rs]
        res["stage5"][c] = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]}
        print("stage5", c, {k: round(x, 3) for k, x in res["stage5"][c].items()}, flush=True)
    jdump(res, OD / "extra.json")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
