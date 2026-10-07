"""Round-5 review additions (no new training of detectors). Run with XRP_GEN=v2.

1. Attacker-target ceiling per attack corpus (PR-AUC of the exact probability that a frame is
   attacker-induced, given the generator's hidden scores), for Table XVI.
2. Lock-box 2: per-training-seed paired differences MPNN - raw-channel HGB on both targets
   (training-seed variance of the confirmatory comparison).
3. Geometry-only attribution rule on the within-band and proximity-only attackers: score each
   incoming edge by (0.5 + 0.5 v) / (d + 0.5), ignoring channels.
-> outputs_v2/stage10/extra.json
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from common import CACHE, OUT, jdump


def main():
    import stage5
    from run_attribution import ranking_metrics
    load = lambda p: {k: v for k, v in np.load(CACHE / p).items()}
    res = {"ceiling": {}, "lb2_seed": {}, "geo_rule": {}}
    std = load("s5_standard.npz")
    corp = {"standard": std, "oov_audio": stage5.base("oov_audio", {"oov": True}),
            "oov_audio_d1": stage5.base("oov_audio_d1", {"oov": True, "oov_min_d": 1.0}),
            "switching": stage5.switching(std), "split": stage5.split(std),
            "near_top": load("s7_near_near.npz"), "near_benign": load("s9_near_benign.npz")}
    for c, t in corp.items():
        y_ai, p_ai = stage5.labels(t)
        res["ceiling"][c] = float(average_precision_score(y_ai, p_ai))
    print("ceilings", {k: round(v, 3) for k, v in res["ceiling"].items()}, flush=True)
    lb2 = load("s8_lockbox2.npz")
    y = lb2["y"]
    y_ai, _ = stage5.labels(lb2)
    P = {m: np.load(OUT / "stage8" / f"{m}_lb2.npy") for m in ("mpnn__MPNN", "edgefeatraw__HGB")}
    for tgt, yy in (("unsafe", y), ("attacker", y_ai)):
        d = [average_precision_score(yy, a) - average_precision_score(yy, b) for a, b in zip(P["mpnn__MPNN"], P["edgefeatraw__HGB"])]
        res["lb2_seed"][tgt] = {"per_seed": [float(x) for x in d], "mean": float(np.mean(d)), "sd": float(np.std(d, ddof=1)),
                                "same_sign": bool(all(x < 0 for x in d) or all(x > 0 for x in d))}
    print("lb2 per-seed", res["lb2_seed"], flush=True)
    for c in ("standard", "near_top", "near_benign"):
        t = corp[c]
        Kp = t["y_att"].shape[1]
        yat = t["y_att"].ravel()
        g = ((0.5 + 0.5 * t["E_geo"][..., 1]) / (t["E_geo"][..., 0] + 0.5)).ravel()
        r = {"roc_auc": float(roc_auc_score(yat, g)), "pr_auc": float(average_precision_score(yat, g))}
        r.update({k: v for k, v in ranking_metrics(yat, g, Kp, np.inf).items() if k in ("top1", "top2")})
        res["geo_rule"][c] = r
    print("geometry rule", {c: round(v["top1"], 3) for c, v in res["geo_rule"].items()})
    (OUT / "stage10").mkdir(parents=True, exist_ok=True)
    jdump(res, OUT / "stage10" / "extra.json")


if __name__ == "__main__":
    main()
