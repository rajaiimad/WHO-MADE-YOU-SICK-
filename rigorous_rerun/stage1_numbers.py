"""Numbers needed by Stage-1 consistency fixes (all from saved outputs; generator v2)."""
import json
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from common import OUT, SEEDS, SEED_RANGES, _generate, cached_tensors, generated_tensors, jdump, param_tensors
from analyze_final import boot

res = {}
# (c) cohort-bootstrap CI on the 'same generator' shift corpus vs lock-box
for cname, c in (("shift_same", param_tensors("shift_same", {})), ("lockbox", generated_tensors("lockbox", 120))):
    y, g = c["y"], c["ctx"][:, 0]
    for m in ("proto__MLPpub", "mpnn__MPNN"):
        P = np.array([np.load(OUT / "final" / m / f"seed{s}" / "preds.npz")[f"p__{cname}"] for s in SEEDS])
        res[f"{cname}|{m}"] = {"roc": float(np.mean([roc_auc_score(y, p) for p in P])),
                               "roc_ci": boot(g, y, P, roc_auc_score, 1000),
                               "n_cohorts": int(len(np.unique(g))), "prevalence": float(y.mean())}
# (d) realised prevalence on the calibration seed range (expected 1.38% by construction)
s0, s1 = SEED_RANGES["backbone_calib"]
u, _, _ = _generate(s1 - s0, s0)
res["calibration_realised_prevalence"] = float((u.severity >= 2).mean())
pos = (u.severity >= 2).groupby(u.cohort_id).sum()
res["calibration_cohorts_without_positives"] = int((pos == 0).sum())
# (m) Holm correction over the MPNN ablation table
abl = json.loads((OUT / "tier2_mpnn_ablation.json").read_text())
names = ["abl_mpnn_no_geo__MPNN", "abl_mpnn_no_peer_bb__MPNN", "abl_mpnn_no_int__MPNN", "abl_mpnn_no_victim__MPNN",
         "abl_mpnn_mean_agg__MPNN", "abl_mpnn_max_agg__MPNN", "abl_mpnn_posw__MPNN", "abl_mpnn_logd__MPNN", "mpnn_raw__MPNN"]
p = [abl[n]["vs_abl_mpnn_no_posw__MPNN"]["pr_auc"]["p"] for n in names]
order = np.argsort(p); adj = np.empty(len(p)); prev = 0
for r, i in enumerate(order):
    prev = max(prev, min(1, p[i] * (len(p) - r))); adj[i] = prev
res["holm_ablation"] = {n: {"p": p[i], "p_holm": float(adj[i]),
                            "mean_paired_dPR": abl[n]["vs_abl_mpnn_no_posw__MPNN"]["pr_auc"]["mean_diff"],
                            "pooled_PR": abl[n]["pooled_by_repeat"]["pr_auc"][0]} for i, n in enumerate(names)}
res["base_pooled_PR"] = abl["abl_mpnn_no_posw__MPNN"]["pooled_by_repeat"]["pr_auc"][0]
jdump(res, OUT / "stage1_numbers.json")
print(json.dumps(res, indent=1))
