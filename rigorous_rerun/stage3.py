"""
rigorous_rerun.stage3
=====================
Stage 3 (PREREGISTRATION_v4.md): generator sensitivity and harder variants.
Run with XRP_GEN=v2 (uses the v2 backbone and v2 nested-CV configurations).

Each variant generates a 60-cohort development corpus (seeds 60000-) and a 60-cohort
test corpus (seeds 61000-) directly as tensors (same keys as common.build_tensors, but
with per-frame geometry and the generator's own hidden score as the ceiling input).
Models are trained once per seed on the development corpus (45/15 cohort split) with
their fixed v2 configurations and scored once on the test corpus.

usage: python stage3.py [variant ...]      (default: all, in pre-registered order)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from common import (CACHE, OUT, PEER_HI, PEER_NAMES, SEEDS, SEED_RANGES, USER_CH, get_backbone, jdump,
                    peer_intensity, gen_kwargs)
from coper_vr.config import PEER_CHANNELS
from coper_vr.data.gen_param import GenParams, _peer_state_param
from coper_vr.data.xrperil_multi import _fov_overlap, _sample_avatar_pose, _sim_user_track

S3 = {"dev": (60_000, 60_060), "test": (61_000, 61_060), "calib": (10_000, 10_030)}
for a, b in ((S3["dev"], S3["test"]),):
    for r in SEED_RANGES.values():
        assert a[1] <= r[0] or a[0] >= r[1], "stage-3 dev seeds overlap"
        assert b[1] <= r[0] or b[0] >= r[1], "stage-3 test seeds overlap"
W = np.array([0.35, 0.20, 0.15, 0.20, 0.10], np.float32)
BEN_HI = np.array([c.benign_hi for c in PEER_CHANNELS], np.float32)
V2_GAIN = gen_kwargs()["peer_gain"]

VARIANTS = {
    "g2": dict(gain=2.0), "g3": dict(gain=3.0), "g4": dict(gain=4.0), "g5": dict(gain=5.0), "g6": dict(gain=6.0),
    "max": dict(gain=None, agg="max"),
    "dose": dict(gain=None, dose=0.9),
    "spikes": dict(gain=V2_GAIN, spike_p=0.05),
    "moving": dict(gain=V2_GAIN, moving=True),
}


# ---------------------------------------------------------------- generator
def _rot(ori, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * ori[0] - s * ori[1], s * ori[0] + c * ori[1], ori[2]])


def generate(seed0, n_cohorts, v, gain, N=4, tracks=8, T=48, with_backbone=True):
    """Returns tensors dict; `gain` overrides v['gain'] (used by calibration)."""
    agg, dose, spike_p, moving = v.get("agg", "sum"), v.get("dose"), v.get("spike_p", 0.0), v.get("moving", False)
    params = GenParams()
    rows = n_cohorts * tracks * N * T
    K = N - 1
    out = dict(ctx=np.zeros((rows, 4), np.int64), V_raw=np.zeros((rows, 20), np.float32),
               E_src=np.zeros((rows, K), np.int64), E_chan=np.zeros((rows, K, 5), np.float32),
               E_geo=np.zeros((rows, K, 3), np.float32), E_w=np.zeros((rows, K), np.float32),
               y_att=np.zeros((rows, K), np.int64), sev=np.zeros(rows, np.int64),
               oracle_score=np.zeros(rows, np.float32), peer_term=np.zeros(rows, np.float32),
               own_prov=np.zeros(rows, np.float32), atk_id=np.zeros(rows, np.int64), atk_ch=np.zeros(rows, np.int64))
    user_rows = np.zeros((rows, 20), np.float32)
    r = 0
    for cid in range(n_cohorts):
        rng = np.random.default_rng(seed0 + cid)
        hmds = rng.integers(0, 4, N)
        for tk in range(tracks):
            poses = [_sample_avatar_pose(rng, 8.0) for _ in range(N)]
            udf = [_sim_user_track(rng, T, int(hmds[u])) for u in range(N)]
            is_atk = rng.random() < 0.5
            aid = int(rng.integers(0, N)) if is_atk else -1
            ach = str(rng.choice(PEER_NAMES)) if is_atk else ""
            pdf = [_peer_state_param(rng, T, u == aid, ach if u == aid else "", params) for u in range(N)]
            PC = np.stack([p[PEER_NAMES].values.astype(np.float32) for p in pdf])     # (N,T,5)
            if spike_p > 0:
                for u in range(N):
                    if u == aid:
                        continue
                    m = rng.random(T) < spike_p
                    ks = rng.integers(0, 5, T)
                    for t in np.where(m)[0]:
                        k = ks[t]
                        PC[u, t, k] = rng.uniform(BEN_HI[k], BEN_HI[k] + 0.5 * (PEER_HI[k] - BEN_HI[k]))
            # per-frame poses
            pos = np.zeros((N, T, 3)); ori = np.zeros((N, T, 3)); fov = np.array([p[2] for p in poses])
            for u in range(N):
                vel = np.r_[rng.normal(0, 0.04, 2), 0.0] if moving else np.zeros(3)
                yaw = rng.normal(0, 1.5) if moving else 0.0
                for t in range(T):
                    pos[u, t] = np.asarray(poses[u][0]) + vel * t
                    ori[u, t] = _rot(np.asarray(poses[u][1]), yaw * t)
            nrm = PC / PEER_HI
            D = np.zeros((N, N, T)); OV = np.zeros((N, N, T))
            for i in range(N):
                for j in range(N):
                    if i == j:
                        continue
                    for t in range(T):
                        D[i, j, t] = np.linalg.norm(pos[j, t] - pos[i, t])
                        OV[i, j, t] = _fov_overlap(pos[i, t], ori[i, t], pos[j, t], fov[i])
            for i in range(N):
                js = [j for j in range(N) if j != i]
                own = (udf[i]["optic_flow"].values / 50.0 + np.abs(udf[i]["yaw_rate"].values) / 300.0
                       + udf[i]["latency_ms"].values / 200.0) / 3.0
                C = np.zeros((N, T), np.float32)          # contribution at gain 1
                for j in js:
                    dd = D[i, j] + 1e-3
                    full = (nrm[j] @ W) * OV[i, j] / dd
                    C[j] = np.where(OV[i, j] == 0, W[4] * nrm[j, :, 4] / dd, full)
                peer = C.max(0) if agg == "max" else C.sum(0)
                if dose:
                    P = np.zeros(T)
                    acc = 0.0
                    for t in range(T):
                        acc = dose * acc + (1 - dose) * peer[t]
                        P[t] = acc
                    peer = P
                g = gain
                score = own + g * peer
                s = score + rng.normal(0, 0.05, T)
                sev = np.zeros(T, np.int64); sev[s > 0.15] = 1; sev[s > 0.30] = 2; sev[s > 0.55] = 3
                attributed = is_atk and i != aid and (g * C[aid]).mean() > 0.02
                sl = slice(r, r + T)
                o = out
                o["ctx"][sl] = np.c_[np.full(T, cid), np.full(T, tk), np.arange(T), np.full(T, i)]
                user_rows[sl] = udf[i][USER_CH].values
                o["E_src"][sl] = js
                o["E_chan"][sl] = PC[js].transpose(1, 0, 2)
                d3 = np.stack([D[i, js].T, OV[i, js].T, np.clip(1 - D[i, js].T / 6.0, 0, 1)], 2)
                o["E_geo"][sl] = d3
                o["E_w"][sl] = 1 / (1 + np.exp(-(1.5 * np.exp(-d3[..., 0] / 3) + d3[..., 1] + 0.75 * d3[..., 2])))
                o["y_att"][sl] = np.array([[1 if (attributed and j == aid) else 0 for j in js]] * T)
                o["sev"][sl] = sev; o["oracle_score"][sl] = score; o["peer_term"][sl] = peer
                o["own_prov"][sl] = own; o["atk_id"][sl] = aid
                o["atk_ch"][sl] = 0 if not ach else PEER_NAMES.index(ach) + 1
                r += T
    out["V_raw"] = user_rows
    out["y"] = (out["sev"] >= 2).astype(np.int64)
    if with_backbone:
        proba = get_backbone().predict_proba(user_rows)
        bb = np.concatenate([np.log(np.clip(proba, 1e-9, 1)), (1 - proba.max(1))[:, None]], 1).astype(np.float32)
        out["V_bb"] = bb
        # peer backbone outputs: rows are (cohort, track, victim, t); peer j of the same (cohort, track, t)
        idx = {tuple(c): n for n, c in enumerate(out["ctx"][:, [0, 1, 2, 3]].tolist())}
        E_bb = np.zeros((rows, K, 5), np.float32)
        for n, (c, tk, t, i) in enumerate(out["ctx"].tolist()):
            for kk, j in enumerate(out["E_src"][n]):
                E_bb[n, kk] = bb[idx[(c, tk, t, int(j))]]
        out["E_bb"] = E_bb
    out["N"], out["T"] = np.array(N), np.array(T)
    return out


def calibrate_gain(v, target=0.0138):
    s0, s1 = S3["calib"]
    t = generate(s0, s1 - s0, v, gain=1.0, with_backbone=False)
    own, peer = t["own_prov"], t["peer_term"]
    return float(brentq(lambda g: (1 - norm.cdf((0.30 - own - g * peer) / 0.05)).mean() - target, 0.1, 100))


def corpus(name, which, v, gain):
    f = CACHE / f"s3_{name}_{which}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    t = generate(S3[which][0], S3[which][1] - S3[which][0], v, gain)
    np.savez_compressed(f, **t)
    return t


# ---------------------------------------------------------------- models
def causal_mean(A, T, K=8):
    n = len(A) // T
    S = A.reshape(n, T, *A.shape[1:])
    out = np.zeros_like(S)
    c = np.cumsum(S, 1)
    for t in range(T):
        lo = t - K
        out[:, t] = (c[:, t] - (c[:, lo] if lo >= 0 else 0)) / min(t + 1, K)
    return out.reshape(A.shape)


def run_variant(name):
    import baselines  # noqa: F401
    from baselines import edge_interaction_features
    from models import FAMILIES, platt
    from run_attribution import edge_tables, hgb, ranking_metrics
    from run_detection import SPECS, fixed_cfg, graph, tab
    from run_final import train_val_split
    from common import prototype_features, thresholds_from_val

    t0 = time.time()
    v = dict(VARIANTS[name])
    gain = v["gain"] if v.get("gain") is not None else calibrate_gain(v)
    dev, test = corpus(name, "dev", v, gain), corpus(name, "test", v, gain)
    T = int(dev["T"])
    models = {
        "rule__RULE": ("RULE", SPECS["rule__RULE"][1]),
        "proto__MLPpub": ("MLPpub", SPECS["proto__MLPpub"][1]),
        "edgefeat__HGB": ("HGB", SPECS["edgefeat__HGB"][1]),
        "gat__GAT": ("GAT", SPECS["gat__GAT"][1]),
        "mpnn__MPNN": ("MPNN", SPECS["mpnn__MPNN"][1]),
    }
    if name == "dose":
        def edgefeat_temporal(t):
            X = edge_interaction_features(t)
            return np.concatenate([X, causal_mean(X, T)], 1)
        def mpnn_temporal(t):
            _, V, E = graph(["bb"], ["chan", "int", "bb", "geo"])(t)
            return ("graph", V, np.concatenate([E, causal_mean(E[..., :10], T)], 2).astype(np.float32))
        models["edgefeat_temporal__HGB"] = ("HGB", tab(edgefeat_temporal))
        models["mpnn_temporal__MPNN"] = ("MPNN", mpnn_temporal)
    y, yt = dev["y"], test["y"]
    res = {"variant": name, "spec": v, "gain": gain, "prevalence_dev": float(y.mean()), "prevalence_test": float(yt.mean()),
           "ceiling": {}, "models": {}, "attribution": {}}
    pc = 1 - norm.cdf((0.30 - test["oracle_score"]) / 0.05)
    res["ceiling"] = {"roc_auc": float(roc_auc_score(yt, pc)), "pr_auc": float(average_precision_score(yt, pc))}
    od = OUT / "stage3" / name
    od.mkdir(parents=True, exist_ok=True)
    for mname, (fam, builder) in models.items():
        cfg_src = mname.replace("_temporal", "")
        cfg = {} if fam in ("MLPpub", "RULE") else fixed_cfg(fam, cfg_src)
        cfg.pop("n_iter", None)
        kd, kt = builder(dev), builder(test)
        rows, preds = [], []
        for s in SEEDS:
            tr, va = train_val_split(dev["ctx"], y, s)
            if kd[0] == "tab":
                d = {"Xtr": kd[1][tr], "ytr": y[tr], "Xva": kd[1][va], "yva": y[va], "Xte": {"test": kt[1]}}
            else:
                d = {"Vtr": kd[1][tr], "Etr": kd[2][tr], "ytr": y[tr], "Vva": kd[1][va], "Eva": kd[2][va],
                     "yva": y[va], "Gte": {"test": (kt[1], kt[2])}}
            pv, pt = FAMILIES[fam](d, cfg, s)
            p = pt["test"]
            thr = thresholds_from_val(y[va], pv)
            yh = p >= thr["f1opt"]
            tp = int((yh & (yt == 1)).sum()); fp = int((yh & (yt == 0)).sum()); fn = int((~yh & (yt == 1)).sum())
            prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
            rows.append({"roc_auc": float(roc_auc_score(yt, p)), "pr_auc": float(average_precision_score(yt, p)),
                         "f1": 2 * prec * rec / max(1e-12, prec + rec)})
            preds.append(p)
        np.save(od / f"{mname}_preds.npy", np.array(preds, np.float32))
        res["models"][mname] = {k: [float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows], ddof=1))]
                                for k in rows[0]}
        print(f"[{name}] {mname:24s} ROC {res['models'][mname]['roc_auc'][0]:.3f} PR {res['models'][mname]['pr_auc'][0]:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    # attribution: learned (channel-specific HGB) vs rule
    Kp = dev["y_att"].shape[1]
    Fd, Ft = edge_tables(dev)["paper"], edge_tables(test)["paper"]
    yad, yat = dev["y_att"].ravel(), test["y_att"].ravel()
    ex = peer_intensity(test["E_chan"]).max(-1)
    rule = (ex * (0.5 + 0.5 * test["E_geo"][..., 1]) / (test["E_geo"][..., 0] + 0.5)).ravel()
    att = []
    for s in SEEDS:
        tr, va = train_val_split(dev["ctx"], y, s)
        etr = (tr[:, None] * Kp + np.arange(Kp)).ravel(); eva = (va[:, None] * Kp + np.arange(Kp)).ravel()
        m = hgb(s).fit(Fd[etr], yad[etr])
        pt = m.predict_proba(Ft)[:, 1]
        thr = thresholds_from_val(yad[eva], m.predict_proba(Fd[eva])[:, 1])["f1opt"]
        r = {"roc_auc": float(roc_auc_score(yat, pt)), "pr_auc": float(average_precision_score(yat, pt))}
        r.update({k: v_ for k, v_ in ranking_metrics(yat, pt, Kp, thr).items() if k in ("top1", "top2")})
        att.append(r)
    res["attribution"]["learned_hgb"] = {k: [float(np.mean([a[k] for a in att])), float(np.std([a[k] for a in att], ddof=1))] for k in att[0]}
    rr = {"roc_auc": float(roc_auc_score(yat, rule)), "pr_auc": float(average_precision_score(yat, rule))}
    rr.update({k: v_ for k, v_ in ranking_metrics(yat, rule, Kp, np.inf).items() if k in ("top1", "top2")})
    res["attribution"]["rule_excess_geo"] = rr
    jdump(res, od / "summary.json")
    print(f"[{name}] gain {gain:.3f} prev dev {res['prevalence_dev']:.4f} test {res['prevalence_test']:.4f} "
          f"ceiling PR {res['ceiling']['pr_auc']:.3f} | attr learned top1 {res['attribution']['learned_hgb']['top1'][0]:.3f} "
          f"rule top1 {rr['top1']:.3f} | done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(VARIANTS):
        run_variant(n)
