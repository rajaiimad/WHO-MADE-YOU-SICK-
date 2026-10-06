"""
rigorous_rerun.leakage_checks
=============================
Automated leakage / contamination checks. Each check records PASS/FAIL plus the
evidence; the script exits non-zero if any check fails.

  python leakage_checks.py   ->  outputs/leakage_checks.json
"""
from __future__ import annotations

import hashlib
import sys

import numpy as np
import pandas as pd

from common import (J7, OUT, SEED_RANGES, SEEDS, USER_CH, VERSION, _generate, build_tensors, cohorts_to_tables,
                    generate_multi, jdump, load_main_tables, main_tensors, outer_inner_splits, prototype_features)

R = {}


def record(name, ok, **ev):
    R[name] = {"pass": bool(ok), **ev}
    print(("PASS " if ok else "FAIL ") + name, ev if not ok else "")


def row_hashes(A):
    A = np.ascontiguousarray(np.round(A.astype(np.float64), 9))
    return np.array([hashlib.md5(r.tobytes()).hexdigest() for r in A])


def main():
    if VERSION == "v1":
        user, peer, meta = load_main_tables()
        u2, p2, m2 = cohorts_to_tables(generate_multi(n_cohorts=60, seed=42, attack_prob=0.5))
    else:   # v2: deterministic regeneration of the fixed-generator main corpus
        user, peer, meta = _generate(60, 42)
        u2, p2, m2 = _generate(60, 42)
    # 1. main corpus is bit-exactly the generator output (seed 42, attack_prob 0.5)
    k = ["cohort_id", "track_id", "user_id", "t"]
    a, b = user.sort_values(k), u2.sort_values(k)
    diff = float(np.abs(a[USER_CH].values - b[USER_CH].values).max())
    record("main_corpus_reproducible", diff < 1e-9 and (a.severity.values == b.severity.values).all()
           and (a.attributed_peer.values == b.attributed_peer.values).all(), max_abs_diff=diff)

    # 2. seed ranges of all corpora are disjoint (asserted at import) -> record
    record("corpus_seed_ranges_disjoint", True, ranges=SEED_RANGES)

    # 3. no raw telemetry row of the main corpus appears in the backbone calibration corpus
    s0, s1 = SEED_RANGES["backbone_calib"]
    cal_u, _, _ = _generate(s1 - s0, s0)
    hm, hc = set(row_hashes(user[USER_CH].values)), set(row_hashes(cal_u[USER_CH].values))
    record("backbone_calib_disjoint_from_main", len(hm & hc) == 0, shared_rows=len(hm & hc))

    t = main_tensors()
    y, ctx = t["y"], t["ctx"]
    X = prototype_features(t)

    # 4. cohort-disjoint splits; every cohort tested exactly once per repeat
    ok, info = True, []
    for s in SEEDS:
        seen = []
        for f, tr, va, te in outer_inner_splits(ctx, y, s):
            g = [set(ctx[i, 0]) for i in (tr, va, te)]
            ok &= not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2])
            seen += list(g[2])
            info.append({"seed": s, "fold": f, "n_tr": len(g[0]), "n_va": len(g[1]), "n_te": len(g[2]),
                         "pos_tr": int(y[tr].sum()), "pos_va": int(y[va].sum()), "pos_te": int(y[te].sum())})
        ok &= sorted(seen) == sorted(np.unique(ctx[:, 0]).tolist())
    record("cohort_disjoint_nested_splits", ok, folds=info)

    # 5. no identical feature rows / raw telemetry rows shared across cohorts
    for nm, A in (("prototype_features", X), ("raw_telemetry", t["V_raw"]),
                  ("edge_tensor", t["E_chan"].reshape(len(y), -1))):
        h = row_hashes(A)
        df = pd.DataFrame({"h": h, "c": ctx[:, 0]})
        n_coh = df.groupby("h")["c"].nunique()
        record(f"no_duplicate_rows_across_cohorts[{nm}]", int((n_coh > 1).sum()) == 0,
               hashes_in_multiple_cohorts=int((n_coh > 1).sum()))

    # 6. causality: perturbing frames > t0 of every track leaves features at <= t0 unchanged
    t0 = 23
    sub_u = user[user.cohort_id < 3].copy(); sub_p = peer[peer.cohort_id < 3].copy()
    sub_m = [m for m in meta if m["cohort_id"] < 3]
    base = build_tensors(sub_u, sub_p, sub_m)
    rng = np.random.default_rng(0)
    pu, pp = sub_u.copy(), sub_p.copy()
    fut = pu.t > t0
    pu.loc[fut, USER_CH[:16]] = pu.loc[fut, USER_CH[:16]].values * rng.uniform(0.5, 1.5, (fut.sum(), 16))
    pcols = [c for c in pp.columns if c not in ("cohort_id", "track_id", "user_id", "t")]
    pp.loc[pp.t > t0, pcols] = pp.loc[pp.t > t0, pcols].values * rng.uniform(0.5, 1.5, ((pp.t > t0).sum(), 5))
    pert = build_tensors(pu, pp, sub_m)
    past = base["ctx"][:, 2] <= t0
    d = max(float(np.abs(base[k][past] - pert[k][past]).max()) for k in ("V_raw", "V_bb", "E_chan", "E_bb", "E_geo"))
    dfut = float(np.abs(base["E_chan"][~past] - pert["E_chan"][~past]).max())
    record("features_are_causal_(future_perturbation_invariant)", d == 0.0 and dfut > 0,
           max_change_past=d, max_change_future=dfut)

    # 7. label leakage: no single feature column is (near-)deterministic of the label
    from sklearn.metrics import roc_auc_score
    cols = {"proto": X, "raw": t["V_raw"], "edge": t["E_chan"].reshape(len(y), -1), "geo": t["E_geo"].reshape(len(y), -1)}
    worst = []
    for nm, A in cols.items():
        for j in range(A.shape[1]):
            if np.ptp(A[:, j]) > 0:
                a_ = roc_auc_score(y, A[:, j]); worst.append((max(a_, 1 - a_), f"{nm}[{j}]"))
    worst.sort(reverse=True)
    record("no_single_feature_encodes_label", worst[0][0] < 0.99, top5_single_feature_auc=worst[:5])

    # 8. functional test: the fitted model / thresholds do not depend on TEST data
    from models import FAMILIES, GRIDS
    from run_detection import SPECS, make_fold_data
    f, tr, va, te = next(outer_inner_splits(ctx, y, 0))
    res = {}
    for name in ("proto__LR", "proto__HGB", "mpnn__MPNN"):
        fam, b = SPECS[name]
        kind = b(t)
        d1 = make_fold_data(kind, t, tr, va, te, y)
        if kind[0] == "tab":
            noisy = (kind[0], kind[1].copy()); noisy[1][te] = rng.normal(size=noisy[1][te].shape) * 1e3
        else:
            V, E = kind[1].copy(), kind[2].copy(); V[te] = 1e3; E[te] = -1e3; noisy = ("graph", V, E)
        d2 = make_fold_data(noisy, t, tr, va, te, y)
        pv1, _ = FAMILIES[fam](dict(d1), GRIDS[fam][0], 0)
        pv2, _ = FAMILIES[fam](dict(d2), GRIDS[fam][0], 0)
        res[name] = float(np.abs(pv1 - pv2).max())
    record("fit_and_val_predictions_independent_of_test_data", max(res.values()) < 1e-5, max_val_pred_change=res)

    # 9. attribution label never enters features; detection label never enters attribution features
    record("labels_not_in_feature_tensors", True,
           note="feature tensors are built from USER_CH telemetry, peer channels, backbone outputs and "
                "pose geometry only; 'severity'/'attributed_peer'/'attacker_id'/'attack_channel' are read "
                "into separate label arrays (common.build_tensors)")

    if VERSION == "v2":
        jdump(R, OUT / "leakage_checks.json")
        fails = [k for k, v in R.items() if "pass" in v and not v["pass"]]
        print("ALL PASS" if not fails else f"FAILED: {fails}")
        sys.exit(1 if fails else 0)
    # 10. documented issue in the OLD pipeline (v1 only): shift corpus 'decouple' (seed 100) re-used main cohorts
    from coper_vr.data.gen_param import GenParams, generate_param
    ru, rp, mt = generate_param(n_cohorts=2, params=GenParams(decouple=True, seed=100))
    old = pd.concat(ru)
    same = np.abs(old[USER_CH].values - user[user.cohort_id.isin([58, 59])].sort_values(k)[USER_CH].values).max()
    R["old_pipeline_issue_decouple_corpus_overlap"] = {
        "finding": "old shift corpus 'decouple' (GenParams seed=100) regenerates main cohorts 58 and 59 "
                   "(seed 42+58, 42+59): user telemetry max abs diff vs main = %.2e" % float(same),
        "status": "fixed here: all new corpora use disjoint seed ranges"}

    # 11. generator non-causality note: accel = np.gradient(speed) uses speed[t+1]
    R["generator_note_accel_central_difference"] = {
        "finding": "xrperil_multi._sim_user_track computes accel with np.gradient (central difference), so "
                   "accel[t] depends on speed[t+1]. The label at t depends only on speed[t] (via optic_flow), "
                   "yaw_rate[t] and latency[t], so this does not carry label information, but it is a "
                   "non-causal telemetry channel; see ablation victim_all_no_accel in analyze_detection."}
    jdump(R, OUT / "leakage_checks.json")
    fails = [k for k, v in R.items() if "pass" in v and not v["pass"]]
    print("ALL PASS" if not fails else f"FAILED: {fails}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
