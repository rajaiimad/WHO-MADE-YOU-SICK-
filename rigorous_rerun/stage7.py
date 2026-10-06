"""
rigorous_rerun.stage7
=====================
Stage 7 (PREREGISTRATION_v7.md): from one generator to a generator family. Run with XRP_GEN=v2.

  python stage7.py variants   A (structural), B (12 random), D (dose) variants: train on each
                              development corpus, score once on its test corpus
  python stage7.py main       models trained on the v2 main corpus: lock-box fairness check (E),
                              transfer to every A/B test corpus (C), within-band attacker (F)
  python stage7.py ci         cohort-bootstrap CIs and paired differences for everything above
Each step resumes: finished variants (summary.json present) are skipped.
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CACHE, OUT, PEER_HI, SEEDS, jdump, thresholds_from_val
from stage3 import S3, V2_GAIN, calibrate_gain, causal_mean, generate

OD = OUT / "stage7"
T = 48
STRUCT = {"d2": {"dpow": 2.0}, "cutoff": {"dcut": 2.5}, "fovfree": {"fovfree": (0, 3)},
          "and": {"and": (0, 3, 0.3)}, "sat": {"sat": 0.15}}
DOSE = {"dose_ramp": {"dose": 0.9}, "dose_intermittent": {"dose": 0.9, "profile": "intermittent"}}
LAMS = (0.5, 0.75, 0.875, 0.95)


def sample_prior(k):
    """Generator k of the pre-registered random family."""
    rng = np.random.default_rng(7000 + k)
    v = {"w": [float(x) for x in rng.dirichlet(np.ones(5))]}
    if rng.random() < 0.5:
        v["dpow"] = float(rng.uniform(0.5, 2.0))
    else:
        v["dcut"] = float(rng.uniform(1.5, 4.0))
    v["fovq"] = float(rng.uniform(0.0, 2.0))
    v["fovfree"] = tuple(int(c) for c in range(4) if rng.random() < 0.2)
    if rng.random() >= 0.5:
        a, b = (int(x) for x in rng.choice(5, 2, replace=False))
        v["and"] = (a, b, float(rng.uniform(0.0, 0.3)))
    if rng.random() >= 0.5:
        v["sat"] = float(rng.uniform(0.1, 0.5))
    u = rng.random()
    if u < 0.5:
        v["agg"] = "sum"
    elif u < 0.75:
        v["agg"] = "max"
    else:
        v["agg"], v["lse_t"] = "lse", float(rng.uniform(0.02, 0.1))
    v["sigma"] = float(rng.uniform(0.03, 0.08))
    return v


RANDOM = {f"rand{k:02d}": sample_prior(k) for k in range(12)}
ALL = {**STRUCT, **RANDOM, **DOSE}


def corpus7(name, which, v, gain):
    f = CACHE / f"s7_{name}_{which}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    s0, s1 = (70_000, 70_060) if which == "near" else S3[which]
    t = generate(s0, s1 - s0, v, gain)
    np.savez_compressed(f, **t)
    return t


# ---------------------------------------------------------------- features
def ema(A, lam):
    n = len(A) // T
    S = A.reshape(n, T, *A.shape[1:]).astype(np.float32)
    out = np.empty_like(S)
    out[:, 0] = S[:, 0]
    for t in range(1, T):
        out[:, t] = lam * out[:, t - 1] + (1 - lam) * S[:, t]
    return out.reshape(A.shape)


def edge_raw_features(t):
    """E (pre-registered): edge-interaction features + the same terms on the raw normalised channels."""
    from baselines import edge_interaction_features
    x = t["E_chan"] / PEER_HI
    d, v = t["E_geo"][..., 0:1], t["E_geo"][..., 1:2]
    a, b = x * v / (d + 0.1), x / (d + 0.1)
    return np.concatenate([edge_interaction_features(t), a.sum(1), a.max(1), b.sum(1), b.max(1)], 1).astype(np.float32)


def model_specs(temporal=False):
    import baselines  # noqa: F401
    from baselines import edge_interaction_features
    from run_detection import SPECS, graph, tab
    m = {"rule__RULE": ("RULE", SPECS["rule__RULE"][1], None),
         "proto__MLPpub": ("MLPpub", SPECS["proto__MLPpub"][1], None),
         "edgefeat__HGB": ("HGB", SPECS["edgefeat__HGB"][1], "edgefeat__HGB"),
         "edgefeatraw__HGB": ("HGB", tab(edge_raw_features), "edgefeat__HGB"),
         "gat__GAT": ("GAT", SPECS["gat__GAT"][1], "gat__GAT"),
         "mpnn__MPNN": ("MPNN", SPECS["mpnn__MPNN"][1], "mpnn__MPNN")}
    if temporal:
        g = graph(["bb"], ["chan", "int", "bb", "geo"])

        def ef_cm(t):
            X = edge_interaction_features(t)
            return np.concatenate([X, causal_mean(X, T)], 1)

        def ef_ema(t):
            X = edge_interaction_features(t)
            return np.concatenate([X] + [ema(X, l) for l in LAMS], 1)

        def mp_cm(t):
            _, V, E = g(t)
            return ("graph", V, np.concatenate([E, causal_mean(E[..., :10], T)], 2).astype(np.float32))

        def mp_ema(t):
            _, V, E = g(t)
            return ("graph", V, np.concatenate([E] + [ema(E[..., :10], l) for l in LAMS], 2).astype(np.float32))
        m.update({"edgefeat_cm__HGB": ("HGB", tab(ef_cm), "edgefeat__HGB"),
                  "edgefeat_ema__HGB": ("HGB", tab(ef_ema), "edgefeat__HGB"),
                  "mpnn_cm__MPNN": ("MPNN", mp_cm, "mpnn__MPNN"),
                  "mpnn_ema__MPNN": ("MPNN", mp_ema, "mpnn__MPNN")})
    return m


def train(fam, builder, cfg_src, dev, tests):
    """5 seeds, 45/15 cohort split of `dev`; returns [{seed, thr, pred: {name: p}}]."""
    from models import FAMILIES
    from run_detection import fixed_cfg
    from run_final import train_val_split
    cfg = {} if cfg_src is None else fixed_cfg(fam, cfg_src)
    cfg.pop("n_iter", None)
    kd = builder(dev)
    kts = {c: builder(t) for c, t in tests.items()}
    y = dev["y"]
    out = []
    for s in SEEDS:
        tr, va = train_val_split(dev["ctx"], y, s)
        if kd[0] == "tab":
            d = {"Xtr": kd[1][tr], "ytr": y[tr], "Xva": kd[1][va], "yva": y[va], "Xte": {c: k[1] for c, k in kts.items()}}
        else:
            d = {"Vtr": kd[1][tr], "Etr": kd[2][tr], "ytr": y[tr], "Vva": kd[1][va], "Eva": kd[2][va], "yva": y[va],
                 "Gte": {c: (k[1], k[2]) for c, k in kts.items()}}
        pv, pt = FAMILIES[fam](d, cfg, s)
        out.append({"seed": s, "thr": thresholds_from_val(y[va], pv), "pred": pt})
    return out


def ceiling(t, v):
    return norm.cdf((t["oracle_score"] - 0.30) / v.get("sigma", 0.05))


# ---------------------------------------------------------------- steps
def run_variants():
    t0 = time.time()
    for name, v in ALL.items():
        od = OD / name
        if (od / "summary.json").exists():
            continue
        od.mkdir(parents=True, exist_ok=True)
        gain = calibrate_gain(v)
        dev, test = corpus7(name, "dev", v, gain), corpus7(name, "test", v, gain)
        yt = test["y"]
        res = {"variant": name, "spec": {k: (list(x) if isinstance(x, tuple) else x) for k, x in v.items()},
               "gain": gain, "prevalence_dev": float(dev["y"].mean()), "prevalence_test": float(yt.mean()), "models": {}}
        pc = ceiling(test, v)
        res["ceiling"] = {"roc_auc": float(roc_auc_score(yt, pc)), "pr_auc": float(average_precision_score(yt, pc))}
        for m, (fam, builder, src) in model_specs(temporal=name in DOSE).items():
            rs = train(fam, builder, src, dev, {"test": test})
            P = np.array([r["pred"]["test"] for r in rs], np.float32)
            np.save(od / f"{m}_preds.npy", P)
            ev = [(roc_auc_score(yt, p), average_precision_score(yt, p)) for p in P]
            res["models"][m] = {"roc_auc": [float(np.mean([e[0] for e in ev])), float(np.std([e[0] for e in ev], ddof=1))],
                                "pr_auc": [float(np.mean([e[1] for e in ev])), float(np.std([e[1] for e in ev], ddof=1))]}
            print(f"[{name}] {m:22s} PR {res['models'][m]['pr_auc'][0]:.3f} ({time.time() - t0:.0f}s)", flush=True)
        jdump(res, od / "summary.json")
        best = max((k for k in res["models"] if k != "mpnn__MPNN" and "_cm" not in k and "_ema" not in k),
                   key=lambda k: res["models"][k]["pr_auc"][0])
        print(f"[{name}] gain {gain:.3f} prev {res['prevalence_test']:.4f} ceiling {res['ceiling']['pr_auc']:.3f} | "
              f"MPNN {res['models']['mpnn__MPNN']['pr_auc'][0]:.3f} best other {best} {res['models'][best]['pr_auc'][0]:.3f}",
              flush=True)


def run_main():
    """Train once on the v2 main corpus: lock-box (E), transfer to A/B test corpora (C), within-band attacker (F)."""
    import stage5
    t0 = time.time()
    od = OD / "main"
    od.mkdir(parents=True, exist_ok=True)
    load = lambda f: {k: v for k, v in np.load(CACHE / f).items()}
    dev = load("s4_main.npz")
    tests = {"lockbox": load("s4_lockbox.npz"), "standard70k": load("s5_standard.npz"),
             "near70k": corpus7("near", "near", {"near": True}, V2_GAIN)}
    for name in list(STRUCT) + list(RANDOM):
        tests[name] = load(f"s7_{name}_test.npz")
    runs = {}
    for m, (fam, builder, src) in model_specs().items():
        f = od / f"{m}.npz"
        if not f.exists():
            rs = train(fam, builder, src, dev, tests)
            np.savez_compressed(f, **{c: np.array([r["pred"][c] for r in rs], np.float32) for c in tests},
                                thr=json.dumps([r["thr"] for r in rs]))
        runs[m] = np.load(f)
        print(f"main {m} done ({time.time() - t0:.0f}s)", flush=True)
    res = {"transfer": {}, "near": {}}
    for c, t in tests.items():
        res["transfer"][c] = {m: float(np.mean([average_precision_score(t["y"], p) for p in r[c]])) for m, r in runs.items()}
    for c in ("standard70k", "near70k"):
        t = tests[c]
        y_ai, p_ai = stage5.labels(t)
        res["near"][c] = {"harm": int(y_ai.sum()), "expected_harm": float(p_ai.sum())}
        for m, r in runs.items():
            thr = json.loads(str(r["thr"]))
            ev = [stage5.evaluate(t, p, th) for p, th in zip(r[c], thr)]
            res["near"][c][m] = {k: float(np.mean([e[k] for e in ev])) for k in ev[0]}
    jdump(res, od / "summary.json")
    print(json.dumps({c: {m: round(v, 3) for m, v in d.items()} for c, d in res["transfer"].items()}, indent=0))
    print(json.dumps(res["near"], indent=1)[:3000])


def run_ci():
    from stage3_ci import B, prep, wmetrics
    rows_per_cohort = 8 * 4 * T
    out = {}

    def boot(Y, P, pairs):
        g = np.arange(len(Y)) // rows_per_cohort
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
        r = {"models": {m: {"pr_auc": [float(point[m][1]), *np.percentile(bs[m][:, 1], [2.5, 97.5]).tolist()]} for m in P},
             "diffs": {}}
        for a, b in pairs:
            d = bs[a][:, 1] - bs[b][:, 1]
            r["diffs"][f"{a} - {b}"] = {"diff": float(point[a][1] - point[b][1]),
                                        "ci": np.percentile(d, [2.5, 97.5]).tolist(), "p_le0": float(np.mean(d <= 0))}
        return r

    for name, v in ALL.items():
        od = OD / name
        if not (od / "summary.json").exists():
            continue
        test = np.load(CACHE / f"s7_{name}_test.npz")
        P = {f.name[:-10]: np.load(f) for f in od.glob("*_preds.npy")}
        P["ceiling"] = ceiling(test, v)[None]
        base = [m for m in P if m not in ("mpnn__MPNN", "ceiling") and "_cm" not in m and "_ema" not in m]
        pairs = [("mpnn__MPNN", m) for m in base]
        if name in DOSE:
            pairs += [("mpnn_ema__MPNN", "mpnn__MPNN"), ("mpnn_cm__MPNN", "mpnn__MPNN"),
                      ("edgefeat_ema__HGB", "edgefeat__HGB"), ("edgefeat_cm__HGB", "edgefeat__HGB")]
        out[name] = boot(test["y"], P, pairs)
        print(name, {k: (round(d["diff"], 3), [round(x, 3) for x in d["ci"]]) for k, d in out[name]["diffs"].items()}, flush=True)
    f = OD / "main" / "mpnn__MPNN.npz"
    if f.exists():
        lb = np.load(CACHE / "s4_lockbox.npz")["y"]
        P = {m: np.load(OD / "main" / f"{m}.npz")["lockbox"] for m in model_specs()}
        out["main_lockbox"] = boot(lb, P, [("mpnn__MPNN", m) for m in P if m != "mpnn__MPNN"])
        print("lockbox", {k: (round(d["diff"], 3), [round(x, 3) for x in d["ci"]]) for k, d in out["main_lockbox"]["diffs"].items()})
    jdump(out, OD / "ci.json")


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "variants"
    {"variants": run_variants, "main": run_main, "ci": run_ci}[step]()
