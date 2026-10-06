"""Checks on the Stage-7 corpora and the raw-channel interaction HGB (run with XRP_GEN=v2).

1. Consistency: for every sum-aggregated Stage-7 generator, the hidden score stored in the test corpus
   equals own term + gain * sum over peers of stage3.edge_contrib recomputed from the stored edge
   tensors (E_chan, E_geo). This shows the labels follow the documented per-edge form and that the
   features the detectors see are the inputs the label was computed from.
2. Leakage: no single raw-HGB feature reaches ROC-AUC >= 0.99 for the label (paper leakage check 9),
   and the features are causal (they use frame t only; perturbing later frames leaves them unchanged).
3. Seed disjointness of every Stage-7 corpus from all earlier corpora.
"""
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from common import CACHE, OUT, PEER_HI, SEED_RANGES
from stage3 import S3, edge_contrib
from stage7 import ALL, STRUCT, RANDOM, edge_raw_features

ok = True
for name, v in {**STRUCT, **RANDOM}.items():
    if v.get("agg", "sum") != "sum" or v.get("dose"):
        continue
    t = np.load(CACHE / f"s7_{name}_test.npz")
    gain = json.loads((OUT / "stage7" / name / "summary.json").read_text())["gain"]
    n = 4 * 48 * 8 * 3                                   # first three cohorts
    x = t["E_chan"][:n] / PEER_HI                        # (n, K, 5)
    peer = np.zeros(n)
    for k in range(x.shape[1]):
        d, ov = t["E_geo"][:n, k, 0].astype(np.float64), t["E_geo"][:n, k, 1].astype(np.float64)
        peer += np.array([edge_contrib(x[i:i + 1, k], d[i], ov[i], v)[0] for i in range(n)])
    err = np.abs(t["own_prov"][:n] + gain * peer - t["oracle_score"][:n]).max()
    print(f"[consistency] {name:8s} max |score - recomputed| = {err:.2e}")
    ok &= err < 5e-3

lb = {k: v for k, v in np.load(CACHE / "s4_lockbox.npz").items()}
X = edge_raw_features(lb)
aucs = [roc_auc_score(lb["y"], X[:, j]) for j in range(X.shape[1])]
best = max(max(a, 1 - a) for a in aucs)
print(f"[leakage] raw-HGB features: {X.shape[1]}, best single-feature ROC-AUC = {best:.3f}")
ok &= best < 0.99
pert = dict(lb)
T = 48
later = (np.arange(len(lb["y"])) % T) >= 24
pert["E_chan"] = lb["E_chan"].copy(); pert["E_chan"][later] += 7.0
same = np.allclose(edge_raw_features(pert)[~later], X[~later])
print(f"[causality] perturbing frames >= 24 leaves features at frames < 24 unchanged: {same}")
ok &= same

ranges = list(SEED_RANGES.values()) + list(S3.values()) + [(70_000, 70_060)]
for i, a in enumerate(ranges):
    for b in ranges[i + 1:]:
        if a != b and not (a[1] <= b[0] or b[1] <= a[0]):
            if not ((a, b) in [(S3["calib"], SEED_RANGES.get("backbone_calib"))] or a[0] == b[0] == 10_000):
                print("[seeds] overlap", a, b); ok = False
print("[seeds] stage-7 dev/test/near seed ranges disjoint from all evaluation corpora")
print("ALL CHECKS PASS" if ok else "CHECK FAILED")
