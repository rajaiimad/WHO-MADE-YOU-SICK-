"""Centralised MPNN on the federated test sets (outer fold 0 of each repeat), from saved nested-CV predictions.
Writes outputs_v2/federated/centralised_fold0.json. Run with XRP_GEN=v2 from rigorous_rerun."""
import json, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from common import OUT, main_tensors
z = np.load(OUT / "detection/mpnn__MPNN/preds.npz")
y = None
try:
    t = main_tensors(); y = t["y"] if isinstance(t, dict) else t[-1]
except Exception as e:
    print("main_tensors failed", e)
res = {}
for s in np.unique(z["seed"]):
    m = (z["seed"] == s) & (z["fold"] == 0)
    rows = z["row"][m]; p = z["p"][m]
    res[int(s)] = {"roc": float(roc_auc_score(y[rows], p)), "pr": float(average_precision_score(y[rows], p)), "n": int(m.sum())}
res["mean_roc"] = float(np.mean([v["roc"] for k, v in res.items()]))
res["mean_pr"] = float(np.mean([v["pr"] for k, v in res.items() if k != "mean_roc"]))
print(json.dumps(res, indent=1))
(OUT / "federated/centralised_fold0.json").write_text(json.dumps(res, indent=1))
