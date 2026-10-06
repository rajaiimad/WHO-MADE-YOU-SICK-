"""Stage-7 summary (adds the raw-channel interaction HGB): lock-box CIs and paired nested-CV tests of every baseline vs the MPNN."""
import json
from analyze_detection import nb_ttest
from common import OUT, jdump

S = json.loads((OUT / "stage7_detection.json").read_text())
key = lambda r: (r["seed"], r["fold"])
mp = {key(r): r for r in S["mpnn__MPNN"]["_fold_rows"]}
out = {}
for m in ("edgefeatraw__HGB", "edgefeat__HGB", "gat__GAT", "rule__RULE", "proto__MLPpub"):
    a = {key(r): r for r in S[m]["_fold_rows"]}
    F = json.loads((OUT / "final_external" / f"{m}.json").read_text())["lockbox"]
    b = F["by_seed"]
    out[m] = {"cv_roc": S[m]["pooled_by_repeat"]["roc_auc"], "cv_pr": S[m]["pooled_by_repeat"]["pr_auc"],
              "cv_roc_ci": S[m]["ci95_roc_auc"], "cv_pr_ci": S[m]["ci95_pr_auc"],
              "cv_f1": S[m]["pooled_by_repeat"]["f1opt_f1"][0], "cv_mcc": S[m]["pooled_by_repeat"]["f1opt_mcc"][0],
              "lb_roc": b["roc_auc"], "lb_roc_ci": F.get("ci95_roc_auc"), "lb_pr": b["pr_auc"], "lb_pr_ci": F.get("ci95_pr_auc"),
              "lb_f1": b["f1opt_f1"][0], "lb_mcc": b["f1opt_mcc"][0], "lb_recall_fpr10": b["fpr10_recall"][0],
              "lb_brier": b["brier"][0],
              "mpnn_minus_model_pr": nb_ttest([mp[k]["pr_auc"] - a[k]["pr_auc"] for k in mp]),
              "mpnn_minus_model_roc": nb_ttest([mp[k]["roc_auc"] - a[k]["roc_auc"] for k in mp])}
jdump(out, OUT / "stage7_summary.json")
for m, v in out.items():
    print(f"{m:16s} CV ROC {v['cv_roc'][0]:.3f} PR {v['cv_pr'][0]:.3f}±{v['cv_pr'][1]:.3f} | LB ROC {v['lb_roc'][0]:.3f} {v['lb_roc_ci']} PR {v['lb_pr'][0]:.3f} {v['lb_pr_ci']} F1 {v['lb_f1']:.3f} | MPNN-model dPR {v['mpnn_minus_model_pr']['mean_diff']:+.3f} p={v['mpnn_minus_model_pr']['p']:.2g} wins {v['mpnn_minus_model_pr']['wins']}/25")
