"""
rigorous_rerun.run_counterfactual
=================================
Counterfactual source masking on the LOCK-BOX corpus (replaces the old
_counterfactual.py, which had no saved analysis script).

For every lock-box victim-frame whose generator ground truth names a source
peer (y_att has a positive edge), mask one incoming peer -- set all five of
its channels to the top of the public benign band -- and measure
    dP = P(unsafe | original) - P(unsafe | masked)
for four choices of masked peer: TRUE source, PREDICTED source (top-1 of the
per-edge attribution model), a RANDOM non-source peer, and the non-source peer
with the HIGHEST attribution score ("wrong" source).

Detectors: the final models trained once per seed on the main corpus
(mpnn__MPNN, proto__MLPpub), 5 seeds. Attribution model: per-edge HistGB
(paper features), trained per seed on the same main-corpus train cohorts.

Because the corpus is simulated, the same intervention can be applied to the
generator's hidden score: the ORACLE dP (P_oracle(orig) - P_oracle(masked)) is
the ground-truth effect of the intervention, against which detector dP is
compared. Statistics: mean dP per condition, paired difference
(predicted - random) with a cohort bootstrap 95% CI (B = 2000) and Cohen's d_z.
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import torch
from scipy.stats import norm

from common import GEN_COEFFS, gen_kwargs, OUT, PEER_BEN_HI, PEER_HI, SEEDS, generated_tensors, jdump, main_tensors, prototype_features
from models import MPNN
from run_attribution import edge_tables, hgb
from run_detection import SPECS, fixed_cfg


def contrib(chan, geo):
    """Generator per-peer contribution (oracle only). chan (n,5) raw, geo (n,3)."""
    gk = gen_kwargs()
    g = gk.get("peer_gain", 1.0)
    aw = GEN_COEFFS[4] if gk.get("audio_oof_weight", 1.0) is None else gk.get("audio_oof_weight", 1.0)
    d = geo[:, 0] + 1e-3
    ov = geo[:, 1]
    nrm = chan / PEER_HI
    full = g * (nrm @ np.asarray(GEN_COEFFS, np.float32)) * ov / d
    audio = g * aw * nrm[:, 4] / d
    return np.where(ov == 0.0, audio, full)


def mpnn_predict(seed, t):
    cfg = fixed_cfg("MPNN", "mpnn__MPNN")
    d = OUT / "final" / "mpnn__MPNN" / f"seed{seed}"
    sv, se = pickle.load(open(d / "scalers.pkl", "rb"))
    _, V, E = SPECS["mpnn__MPNN"][1](t)
    if cfg.get("logd"):
        E = np.concatenate([E, np.log(E[..., :1] + 0.05)], -1)
    net = MPNN(V.shape[1], E.shape[2], cfg["h"])
    net.load_state_dict(torch.load(d / "net.pt", map_location="cpu"))
    net.eval()
    with torch.no_grad():
        return np.concatenate([torch.sigmoid(net(torch.tensor(sv(V[s:s + 8192])), torch.tensor(se(E[s:s + 8192])))).numpy()
                               for s in range(0, len(V), 8192)])


def proto_predict(seed, t):
    m = pickle.load(open(OUT / "final" / "proto__MLPpub" / f"seed{seed}" / "model.pkl", "rb"))
    return m.predict_proba(prototype_features(t))[:, 1]


def masked(t, rows, slot):
    tt = {k: (v[rows] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(t["y"]) else v) for k, v in t.items()}
    ch = tt["E_chan"].copy()
    ch[np.arange(len(rows)), slot] = PEER_BEN_HI
    tt["E_chan"] = ch
    return tt


def boot_paired(groups, diff, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    cs = np.unique(groups)
    by = {c: np.where(groups == c)[0] for c in cs}
    v = [diff[np.concatenate([by[c] for c in rng.choice(cs, len(cs))])].mean() for _ in range(n)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def main():
    lb = generated_tensors("lockbox", 120)
    mt = main_tensors()
    K = lb["y_att"].shape[1]
    rows = np.where(lb["y_att"].max(1) == 1)[0]          # victim-frames with a true source
    true_slot = lb["y_att"][rows].argmax(1)
    groups = lb["ctx"][rows, 0]
    Fm = edge_tables(mt)["paper"]
    Fl = edge_tables(lb)["paper"]
    # oracle effect of each intervention (ground truth under simulator semantics)
    base_score = lb["oracle_score"][rows]
    p_or = lambda s: 1 - norm.cdf((0.30 - s) / 0.05)

    def oracle_dp(slot):
        i = np.arange(len(rows))
        ch, geo = lb["E_chan"][rows, slot], lb["E_geo"][rows, slot]
        ch_m = np.broadcast_to(PEER_BEN_HI, ch.shape)
        s_m = base_score - contrib(ch, geo) + contrib(ch_m, geo)
        return p_or(base_score) - p_or(s_m)

    res = {"n_victim_frames": int(len(rows)), "n_cohorts": int(len(np.unique(groups))), "by_seed": []}
    acc = {}
    for s in SEEDS:
        meta = json.loads((OUT / "final" / "mpnn__MPNN" / f"seed{s}" / "meta.json").read_text())
        tr = np.isin(mt["ctx"][:, 0], meta["train_cohorts"])
        etr = (np.where(tr)[0][:, None] * K + np.arange(K)).ravel()
        att = hgb(s).fit(Fm[etr], mt["y_att"].ravel()[etr])
        sc = att.predict_proba(Fl)[:, 1].reshape(-1, K)[rows]
        pred_slot = sc.argmax(1)
        rng = np.random.default_rng(s)
        non = np.array([[k for k in range(K) if k != ts] for ts in true_slot])
        rand_slot = non[np.arange(len(rows)), rng.integers(0, K - 1, len(rows))]
        sc_non = np.where(np.arange(K)[None] == true_slot[:, None], -np.inf, sc)
        wrong_slot = sc_non.argmax(1)
        slots = {"true": true_slot, "predicted": pred_slot, "random_nonsource": rand_slot, "wrong_top_nonsource": wrong_slot}
        r = {"seed": s, "attribution_top1": float((pred_slot == true_slot).mean())}
        for det, fn in (("mpnn__MPNN", mpnn_predict), ("proto__MLPpub", proto_predict)):
            sub = {k: (v[rows] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(lb["y"]) else v) for k, v in lb.items()}
            p0 = fn(s, sub)
            for name, sl in slots.items():
                dp = p0 - fn(s, masked(lb, rows, sl))
                acc.setdefault((det, name), []).append(dp)
                r[f"{det}|{name}"] = float(dp.mean())
        for name, sl in slots.items():
            dpo = oracle_dp(sl)
            acc.setdefault(("ORACLE", name), []).append(dpo)
            r[f"ORACLE|{name}"] = float(dpo.mean())
        res["by_seed"].append(r)
        print(r, flush=True)
    summ = {}
    for det in ("mpnn__MPNN", "proto__MLPpub", "ORACLE"):
        dps = {k: np.mean(acc[(det, k)], 0) for k in ("true", "predicted", "random_nonsource", "wrong_top_nonsource")}
        diff = dps["predicted"] - dps["random_nonsource"]
        summ[det] = {f"mean_dP_{k}": float(v.mean()) for k, v in dps.items()}
        summ[det]["paired_pred_minus_random"] = {"mean": float(diff.mean()), "ci95_cohort_boot": boot_paired(groups, diff),
                                                 "cohens_dz": float(diff.mean() / (diff.std(ddof=1) + 1e-12))}
        if det != "ORACLE":
            o = np.mean(acc[("ORACLE", "true")], 0)
            summ[det]["corr_with_oracle_dP_true"] = float(np.corrcoef(dps["true"], o)[0, 1])
        summ[det]["mean_dP_true_by_channel"] = {
            c: float(dps["true"][lb["atk_ch"][rows] == i + 1].mean()) for i, c in
            enumerate(["avatar_ang_vel", "teleport_rate", "particle_intensity", "shader_flicker_hz", "spatial_audio_jit"])
            if (lb["atk_ch"][rows] == i + 1).any()}
    res["summary"] = summ
    res["attribution_top1_mean"] = float(np.mean([r["attribution_top1"] for r in res["by_seed"]]))
    jdump(res, OUT / "counterfactual.json")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
