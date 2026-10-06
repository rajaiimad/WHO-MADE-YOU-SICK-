"""Builds outputs/TABLES.md from the saved result JSONs (no hand-typed numbers)."""
from __future__ import annotations

import json

import pandas as pd

from common import OUT, PEER_NAMES

J = lambda p: json.loads((OUT / p).read_text())
f3 = lambda v: f"{v:.3f}"
ms = lambda v: f"{v[0]:.3f} ± {v[1]:.3f}"
ci = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"
L = []


def table(header, rows):
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "---|" * len(header))
    for r in rows:
        L.append("| " + " | ".join(r) + " |")
    L.append("")


def det_rows(res, names):
    rows = []
    for n in names:
        r = res[n]; p = r["pooled_by_repeat"]; f = r["per_fold"]
        rows.append([n, ms(p["roc_auc"]), ci(r["ci95_roc_auc"]) if "ci95_roc_auc" in r else "-",
                     ms(p["pr_auc"]), ci(r["ci95_pr_auc"]) if "ci95_pr_auc" in r else "-",
                     ms(f["roc_auc"]), ms(f["pr_auc"])])
    return rows


def op_rows(res, names, op):
    rows = []
    for n in names:
        p = res[n]["pooled_by_repeat"]
        rows.append([n] + [f3(p[f"{op}_{k}"][0]) for k in ("precision", "recall", "f1", "mcc", "bal_acc", "specificity", "fpr")])
    return rows


def main():
    t1 = J("tier1_detection.json")
    names = ["PREVALENCE", "victim_bb__MLPpub", "victim_all__HGB", "proto__MLPpub", "proto__LR", "proto__HGB",
             "proto__MLP", "mpnn__MPNN", "mpnn_raw__MPNN", "ORACLE"]
    L.append("# Generated result tables (rigorous rerun)\n")
    L.append("## T1. Detection, main corpus, nested repeated cohort-disjoint CV (5 repeats x 5 folds)\n")
    L.append("Pooled = out-of-fold ROC/PR per repeat, mean ± SD over 5 repeats; CI = hierarchical cohort bootstrap (B=1000); "
             "per-fold = mean ± SD over 25 folds.\n")
    table(["model", "ROC-AUC pooled", "ROC 95% CI", "PR-AUC pooled", "PR 95% CI", "ROC per-fold", "PR per-fold"],
          det_rows(t1, names))
    for op, lab in (("f1opt", "validation F1-optimal threshold"), ("fpr10", "validation 10% FPR budget")):
        L.append(f"## T2{'a' if op == 'f1opt' else 'b'}. Operating point at {lab} (pooled confusion, mean of 5 repeats)\n")
        table(["model", "precision", "recall", "F1", "MCC", "bal.acc", "specificity", "achieved FPR"], op_rows(t1, names[1:], op))
    L.append("## T3. Calibration (per-fold mean): raw score vs Platt-calibrated on validation cohorts\n")
    rows = []
    for n in names[1:]:
        f = t1[n]["per_fold"]
        rows.append([n, f3(f["brier"][0]), f3(f["cal_brier"][0]), f3(f["ece15"][0]), f3(f["cal_ece15"][0]),
                     f3(f["log_loss"][0]), f3(f["cal_log_loss"][0])])
    table(["model", "Brier raw", "Brier Platt", "ECE15 raw", "ECE15 Platt", "logloss raw", "logloss Platt"], rows)
    L.append("## T4. Paired comparison vs published prototype (proto__MLPpub): Nadeau-Bengio corrected t-test over 25 folds; "
             "cohort-bootstrap CI of pooled difference\n")
    rows = []
    for n in names[1:-1]:
        c = t1[n].get("vs_proto__MLPpub")
        if not c:
            continue
        rows.append([n, f"{c['roc_auc']['mean_diff']:+.3f} (p={c['roc_auc']['p']:.3g})", ci(c["boot_ci95_delta_roc_auc"]),
                     f"{c['pr_auc']['mean_diff']:+.3f} (p={c['pr_auc']['p']:.3g})", ci(c["boot_ci95_delta_pr_auc"]),
                     f"{c['f1opt_f1']['mean_diff']:+.3f} (p={c['f1opt_f1']['p']:.3g})", f"{c['pr_auc']['wins']}/25"])
    table(["model", "ΔROC (p)", "ΔROC CI", "ΔPR (p)", "ΔPR CI", "ΔF1 (p)", "PR wins"], rows)

    t2 = J("tier2_isolation_temporal.json")
    L.append("## T5. Feature-source / aggregation isolation (HGB, fixed Tier-1 config, 25 folds)\n")
    iso = ["iso_victim_raw__HGB", "iso_victim_bb__HGB", "iso_peer_bb__HGB", "iso_peer_chan_mean__HGB", "iso_peer_max_int__HGB",
           "iso_peer_channels__HGB", "iso_uniform_w__HGB", "iso_random_w__HGB", "iso_permuted_peers__HGB",
           "iso_geometry_only__HGB", "temporal_K1__HGB"]
    table(["condition", "ROC pooled", "-", "PR pooled", "-", "ROC per-fold", "PR per-fold"], det_rows(t2, iso))
    L.append("(temporal_K1__HGB = full 20-D prototype with edge weights, same fixed config.)\n")
    L.append("## T6. Imbalance handling (HGB) and temporal context (RQ3)\n")
    names6 = ["abl_hgb_no_cw__HGB", "abl_hgb_cw__HGB", "temporal_K1__HGB", "temporal_K4__HGB", "temporal_K8__HGB", "temporal_K16__HGB"]
    rows = det_rows(t2, names6)
    for r, n in zip(rows, names6):
        c = t2[n].get("vs_temporal_K1__HGB")
        r.append(f"{c['pr_auc']['mean_diff']:+.3f} (p={c['pr_auc']['p']:.3g})" if c else "ref")
    table(["model", "ROC pooled", "-", "PR pooled", "-", "ROC per-fold", "PR per-fold", "ΔPR vs K1 (p)"], rows)

    t3 = J("tier2_mpnn_ablation.json")
    L.append("## T7. MPNN ablations (fixed Tier-1 modal config; reference = abl_mpnn_no_posw = base config)\n")
    names7 = ["abl_mpnn_no_posw__MPNN", "abl_mpnn_no_geo__MPNN", "abl_mpnn_no_peer_bb__MPNN", "abl_mpnn_no_int__MPNN",
              "abl_mpnn_no_victim__MPNN", "abl_mpnn_mean_agg__MPNN", "abl_mpnn_max_agg__MPNN", "abl_mpnn_posw__MPNN",
              "abl_mpnn_logd__MPNN", "mpnn_raw__MPNN", "mpnn__MPNN"]
    rows = det_rows(t3, names7)
    for r, n in zip(rows, names7):
        c = t3[n].get("vs_abl_mpnn_no_posw__MPNN")
        r.append(f"{c['pr_auc']['mean_diff']:+.3f} (p={c['pr_auc']['p']:.3g})" if c and n != "abl_mpnn_no_posw__MPNN" else "ref")
    table(["variant", "ROC pooled", "-", "PR pooled", "-", "ROC per-fold", "PR per-fold", "ΔPR vs base (p)"], rows)

    A = J("attribution_summary.json")
    L.append("## T8. Attribution, in-distribution (HGB per edge, 25 folds, mean ± SD)\n")
    rows = []
    for k, v in A["in_distribution"].items():
        rows.append([k, ms(v["roc_auc"]), ms(v["pr_auc"]), f3(v["top1"][0]), f3(v["top2"][0]), f3(v["precision"][0]),
                     f3(v["recall"][0]), f3(v["macro_f1"][0]), f3(v["spurious_rate_no_source"][0])])
    table(["features", "edge ROC", "edge PR", "top-1", "top-2", "precision", "recall", "macro-F1", "spurious (no-source frames)"], rows)
    L.append("## T9. Attribution, leave-one-attack-channel-out (mean ± SD over folds)\n")
    rows = []
    for c in PEER_NAMES:
        rows.append([c] + [f"{A['loco'][f'{fs}|{c}']['roc_auc'][0]:.3f} / {A['loco'][f'{fs}|{c}']['top1'][0]:.3f}"
                           for fs in ("paper", "relative", "agnostic")])
    rows.append(["**mean**"] + [f"{A['loco_mean'][fs]['roc_auc']:.3f} / {A['loco_mean'][fs]['top1']:.3f}"
                                for fs in ("paper", "relative", "agnostic")])
    table(["held-out channel", "paper (ROC / top-1)", "relative", "agnostic"], rows)

    L.append("## T10. Stage B: trained once per seed on main corpus, scored on external corpora (mean ± SD over 5 seeds)\n")
    models = ["victim_bb__MLPpub", "victim_all__HGB", "proto__MLPpub", "proto__LR", "proto__HGB", "proto__MLP",
              "mpnn__MPNN", "mpnn_raw__MPNN", "ORACLE"]
    F = {m: J(f"final_external/{m}.json") for m in models}
    L.append("### T10a. Lock-box (120 fresh cohorts, scored once); CI = cohort bootstrap of seed-averaged metric (B=1000)\n")
    rows = []
    for m in models:
        r = F[m]["lockbox"]; b = r["by_seed"]
        rows.append([m, ms(b["roc_auc"]), ci(r["ci95_roc_auc"]), ms(b["pr_auc"]), ci(r["ci95_pr_auc"]),
                     f3(b["f1opt_precision"][0]), f3(b["f1opt_recall"][0]), f3(b["f1opt_f1"][0]), f3(b["f1opt_mcc"][0]),
                     f3(b["f1opt_bal_acc"][0]), f3(b["f1opt_specificity"][0]), f3(b["fpr10_recall"][0]),
                     f3(b["fpr10_precision"][0]), f3(b["fpr10_fpr"][0]), f3(b["brier"][0]), f3(b["cal_ece15"][0])])
    table(["model", "ROC", "ROC CI", "PR", "PR CI", "P@F1opt", "R@F1opt", "F1", "MCC", "BalAcc", "Spec",
           "R@10%FPR", "P@10%FPR", "achieved FPR", "Brier", "ECE(Platt)"], rows)
    for title, keys in (("T10b. Cohort-size transfer (train N=4)", ["size_N2", "size_N3", "lockbox", "size_N6", "size_N8"]),
                        ("T10c. Generator shift / decoupling (labels re-simulated)",
                         ["shift_same", "shift_coeff", "shift_thresh", "shift_pulse", "shift_intensity", "shift_decouple",
                          "lockbox_peerperm"]),
                        ("T10d. Adaptive attacker intensity sweep (labels re-simulated)",
                         ["adapt_a1.0", "adapt_a0.8", "adapt_a0.6", "adapt_a0.4", "adapt_a0.2"])):
        L.append(f"### {title}: ROC / PR (prevalence)\n")
        rows = []
        for m in models:
            rows.append([m] + [(f"{F[m][k]['by_seed']['roc_auc'][0]:.3f} / {F[m][k]['by_seed']['pr_auc'][0]:.3f}"
                                if k in F[m] else "n/a") for k in keys])
        prev = [f"{F['proto__LR'][k]['by_seed']['prevalence'][0]:.4f}" for k in keys]
        table(["model"] + keys, rows + [["prevalence"] + prev])
    L.append("### T10e. Adaptive attacker: harm vs detection at the 10%-FPR operating point\n")
    rows = []
    for k in ["adapt_a1.0", "adapt_a0.8", "adapt_a0.6", "adapt_a0.4", "adapt_a0.2"]:
        row = [k, f3(F["proto__MLPpub"][k]["by_seed"]["unsafe_per_attacked_track"][0])]
        for m in ("proto__MLPpub", "mpnn__MPNN", "ORACLE"):
            b = F[m][k]["by_seed"]
            row += [f3(b["fpr10_recall"][0]), f3(b["missed_unsafe_per_attacked_track_fpr10"][0])]
        rows.append(row)
    table(["alpha", "unsafe frames / attacked track", "proto recall", "proto missed/track", "MPNN recall",
           "MPNN missed/track", "oracle recall", "oracle missed/track"], rows)

    fed = pd.read_csv(OUT / "federated" / "federated_summary.csv")
    L.append("## T11. Federated MPNN (48 clients, 30 rounds, 5 seeds): ROC / PR mean (SD)\n")
    rows = []
    for (a, fr), g in fed.groupby(["attack", "frac"], sort=False):
        row = [a, f"{fr:.1f}"]
        for agg in ("fedavg", "median", "trimmed"):
            x = g[g["agg"] == agg].iloc[0]
            row.append(f"{x.roc_auc_mean:.3f} / {x.pr_auc_mean:.3f} ({x.pr_auc_std:.3f})")
        rows.append(row)
    table(["attack", "byz frac", "FedAvg", "median", "trimmed mean"], rows)

    lat = J("latency.json")
    L.append(f"## T12. Latency (CPU {lat['cpu']}, 1 thread, ms per cohort-frame incl. feature assembly, median / p95)\n")
    rows = []
    for N, v in lat["by_size"].items():
        rows.append([N] + [f"{v[m]['ms_per_cohort_frame_median']:.3f} / {v[m]['ms_per_cohort_frame_p95']:.3f}"
                           for m in ("proto__MLPpub", "proto__HGB", "mpnn__MPNN")])
    table(["cohort size", "proto__MLPpub", "proto__HGB", "mpnn__MPNN"], rows)

    lk = J("leakage_checks.json")
    L.append("## T13. Leakage checks\n")
    table(["check", "result"], [[k, ("PASS" if v["pass"] else "FAIL") if "pass" in v else "note"] for k, v in lk.items()])
    (OUT / "TABLES.md").write_text("\n".join(L), encoding="utf-8")
    print("wrote", OUT / "TABLES.md")


if __name__ == "__main__":
    main()
