"""
rigorous_rerun.stage4
=====================
Stage 4 (PREREGISTRATION_v5.md): attacker-induced label.
Run with XRP_GEN=v2.

The v2 main corpus (development) and lock-box (test) are regenerated with the Stage-3
generator, which also returns the attacker's term g*c_{a->i} and the noisy score; both are
asserted to reproduce the stored v2 labels exactly. Each model is trained on the unsafe
label and on the attacker-induced label (fixed v2 configurations, 5 seeds, 45/15 cohort
split) and scored once on the lock-box against both targets.
"""
from __future__ import annotations

import json
import time

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CACHE, OUT, SEED_RANGES, SEEDS, generated_tensors, jdump, main_tensors
from stage3 import V2_GAIN, generate
from stage3_ci import B, prep, wmetrics

OD = OUT / "stage4"
MODELS = ["rule__RULE", "proto__MLPpub", "edgefeat__HGB", "gat__GAT", "mpnn__MPNN"]
ROWS_PER_COHORT = 8 * 4 * 48


def corpus(name, n, ref):
    f = CACHE / f"s4_{name}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    t = generate(SEED_RANGES[name][0], n, {}, V2_GAIN)
    assert np.array_equal(t["y"], ref["y"]) and np.array_equal(t["sev"], ref["sev"]), f"{name}: labels differ from v2"
    assert np.allclose(t["E_chan"], ref["E_chan"]) and np.allclose(t["V_raw"], ref["V_raw"]), f"{name}: inputs differ"
    np.savez_compressed(f, **t)
    return t


def targets(t):
    s, a = t["s_noisy"], t["atk_term"]
    y_ai = ((s > 0.30) & (s - a <= 0.30)).astype(np.int64)
    assert np.array_equal((s > 0.30).astype(np.int64), t["y"])
    assert not (y_ai & (t["y"] == 0)).any()
    sc = t["oracle_score"]
    ceil = {"unsafe": norm.cdf((sc - 0.30) / 0.05),
            "attacker": norm.cdf((0.30 - sc + a) / 0.05) - norm.cdf((0.30 - sc) / 0.05)}
    return {"unsafe": t["y"], "attacker": y_ai}, ceil


def composition(t, y_ai):
    y = t["y"]
    atk_track = t["atk_id"] >= 0
    n = int(y.sum())
    return {"n_frames": int(len(y)), "unsafe": n, "prevalence_unsafe": float(y.mean()),
            "prevalence_attacker": float(y_ai.mean()),
            "attacker_induced": int(y_ai.sum()),
            "unsafe_attacked_track_not_attacker_induced": int((y & atk_track & (y_ai == 0)).sum()),
            "unsafe_benign_track": int((y & ~atk_track).sum())}


def train_and_score(dev, test, ytr_full, name):
    import baselines  # noqa: F401
    from models import FAMILIES
    from run_detection import SPECS, fixed_cfg
    from run_final import train_val_split
    fam = name.split("__")[1]
    builder = SPECS[name][1]
    cfg = {} if fam in ("MLPpub", "RULE") else fixed_cfg(fam, name)
    cfg.pop("n_iter", None)
    kd, kt = builder(dev), builder(test)
    preds = []
    for s in SEEDS:
        tr, va = train_val_split(dev["ctx"], ytr_full, s)
        if kd[0] == "tab":
            d = {"Xtr": kd[1][tr], "ytr": ytr_full[tr], "Xva": kd[1][va], "yva": ytr_full[va], "Xte": {"test": kt[1]}}
        else:
            d = {"Vtr": kd[1][tr], "Etr": kd[2][tr], "ytr": ytr_full[tr], "Vva": kd[1][va], "Eva": kd[2][va],
                 "yva": ytr_full[va], "Gte": {"test": (kt[1], kt[2])}}
        _, pt = FAMILIES[fam](d, cfg, s)
        preds.append(pt["test"])
    return np.array(preds, np.float32)


def bootstrap(P, Y, ceil, pairs):
    """P: {key: (seeds, rows)}; Y: {target: y}. Same cohort draws for every key."""
    g = np.arange(len(next(iter(Y.values())))) // ROWS_PER_COHORT
    nc = g.max() + 1
    keys = [(k, tgt) for k in P for tgt in Y] + [(f"ceiling_{tgt}", tgt) for tgt in Y]
    srt = {}
    for k, tgt in keys:
        ps = [ceil[tgt]] if k.startswith("ceiling") else P[k]
        srt[(k, tgt)] = [prep(p, Y[tgt], g) for p in ps]
    point = {kk: np.mean([wmetrics(ys, np.ones_like(ys)) for ys, _ in v], 0) for kk, v in srt.items()}
    rng = np.random.default_rng(0)
    boot = {kk: [] for kk in srt}
    for _ in range(B):
        cnt = np.bincount(rng.integers(0, nc, nc), minlength=nc).astype(float)
        for kk, v in srt.items():
            boot[kk].append(np.mean([wmetrics(ys, cnt[gs]) for ys, gs in v], 0))
    boot = {kk: np.array(b) for kk, b in boot.items()}
    res = {f"{k} | test={tgt}": {m: [float(point[(k, tgt)][i]), *np.percentile(boot[(k, tgt)][:, i], [2.5, 97.5]).tolist()]
                                 for i, m in enumerate(["roc_auc", "pr_auc"])} for k, tgt in srt}
    diffs = {}
    for a, b, tgt in pairs:
        d = boot[(a, tgt)][:, 1] - boot[(b, tgt)][:, 1]
        diffs[f"{a} - {b} | test={tgt}"] = {"pr_auc_diff": float(point[(a, tgt)][1] - point[(b, tgt)][1]),
                                            "ci": np.percentile(d, [2.5, 97.5]).tolist(), "p_le0": float(np.mean(d <= 0))}
    return res, diffs


def main():
    t0 = time.time()
    OD.mkdir(parents=True, exist_ok=True)
    dev = corpus("main", 60, main_tensors())
    test = corpus("lockbox", 120, generated_tensors("lockbox", 120))
    Yd, _ = targets(dev)
    Yt, ceil = targets(test)
    res = {"composition": {"dev": composition(dev, Yd["attacker"]), "lockbox": composition(test, Yt["attacker"])}}
    print(json.dumps(res["composition"], indent=1), flush=True)
    P = {}
    for m in MODELS:
        for tr in ("unsafe", "attacker"):
            f = OD / f"{m}__train-{tr}.npy"
            if not f.exists():
                np.save(f, train_and_score(dev, test, Yd[tr], m))
            P[f"{m} | train={tr}"] = np.load(f)
            print(f"{m} train={tr} done ({time.time() - t0:.0f}s)", flush=True)
    mp = "mpnn__MPNN | train=attacker"
    pairs = [(mp, f"{b} | train=attacker", "attacker") for b in ("rule__RULE", "gat__GAT", "edgefeat__HGB", "proto__MLPpub")]
    pairs += [(f"{m} | train=attacker", f"{m} | train=unsafe", "attacker") for m in MODELS if m != "rule__RULE"]
    pairs += [(mp, "mpnn__MPNN | train=unsafe", "unsafe")]
    res["metrics"], res["diffs"] = bootstrap(P, Yt, ceil, pairs)
    jdump(res, OD / "summary.json")
    for k, v in res["metrics"].items():
        print(f"{k:50s} ROC {v['roc_auc'][0]:.3f} PR {v['pr_auc'][0]:.3f} [{v['pr_auc'][1]:.3f}, {v['pr_auc'][2]:.3f}]")
    for k, d in res["diffs"].items():
        print(f"{k:80s} {d['pr_auc_diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] p<=0 {d['p_le0']:.3f}")
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
