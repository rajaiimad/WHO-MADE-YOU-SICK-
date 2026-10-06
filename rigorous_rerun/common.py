"""
rigorous_rerun.common
=====================
Shared data, feature, split and metric code for the Oct-2026 rigorous rerun of
the XR-PERIL-Multi study ("Who Made You Sick?").

Design rules (enforced by asserts and by rigorous_rerun/leakage_checks.py):
  * the unit of independence is the COHORT; every split is cohort-disjoint;
  * every feature of victim-frame (cohort, track, t, victim) is computed from
    frame t only (or frames <= t for the causal temporal variants);
  * scalers / models / calibrators / thresholds are fitted on training or
    validation cohorts only -- never on test cohorts;
  * every corpus is generated from a disjoint seed range (see SEED_RANGES).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent            # .../rigorous_rerun
J7 = ROOT.parent
if str(J7) not in sys.path:
    sys.path.insert(0, str(J7))

from coper_vr.config import PEER_CHANNELS                                     # noqa: E402
from coper_vr.data.xrperil_multi import _USER_CHANNELS, generate_multi       # noqa: E402

# Generator version. v1 = published XR-PERIL-Multi generator (out-of-FOV audio weight 1.0).
# v2 = fixed generator: audio weight coeffs[4] applied in and out of FOV, plus one global
#      peer-coupling gain calibrated label-free (calibrate_gain_v2) to keep prevalence ~1.4%.
VERSION = os.environ.get("XRP_GEN", "v1")
assert VERSION in ("v1", "v2")
OUT = ROOT / ("outputs" if VERSION == "v1" else "outputs_v2")
V2_GAIN = None          # filled in below from outputs_v2/gain.json (written by calibrate_gain_v2)
CACHE = OUT / "cache"
for _p in (OUT, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

USER_CH = list(_USER_CHANNELS)
PEER_NAMES = [c.name for c in PEER_CHANNELS]
PEER_HI = np.array([c.hi for c in PEER_CHANNELS], np.float32)
PEER_BEN_HI = np.array([c.benign_hi for c in PEER_CHANNELS], np.float32)
GEN_COEFFS = (0.35, 0.20, 0.15, 0.20, 0.10)       # generator, used ONLY for the oracle
SEEDS = (0, 1, 2, 42, 2024)

# Per-cohort generator seed = corpus seed + cohort index. Ranges must not overlap.
SEED_RANGES = {
    "main": (42, 42 + 60),               # the published corpus (verified bit-exact)
    "backbone_calib": (10_000, 10_060),  # frozen per-user backbone training corpus
    "lockbox": (20_000, 20_120),         # fresh confirmation corpus, evaluated ONCE
    "size_N2": (30_000, 30_060), "size_N3": (31_000, 31_060),
    "size_N6": (32_000, 32_060), "size_N8": (33_000, 33_060),
    "shift_same": (40_000, 40_040), "shift_coeff": (41_000, 41_040),
    "shift_thresh": (42_000, 42_040), "shift_pulse": (43_000, 43_040),
    "shift_intensity": (44_000, 44_040), "shift_decouple": (45_000, 45_040),
    "adapt_a1.0": (50_000, 50_040), "adapt_a0.8": (51_000, 51_040),
    "adapt_a0.6": (52_000, 52_040), "adapt_a0.4": (53_000, 53_040),
    "adapt_a0.2": (54_000, 54_040),
}


def check_seed_ranges():
    iv = sorted(SEED_RANGES.values())
    for (a0, a1), (b0, b1) in zip(iv, iv[1:]):
        assert a1 <= b0, f"seed ranges overlap: {(a0, a1)} {(b0, b1)}"


check_seed_ranges()


def gen_kwargs():
    """Generator keyword overrides for the active version (passed to GenParams)."""
    if VERSION == "v1":
        return {}
    g = json.loads((J7 / "data" / "v2_generator" / "gain.json").read_text())["peer_gain"]
    return {"audio_oof_weight": None, "peer_gain": float(g)}


def sha(path: Path, n=16) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]


# ---------------------------------------------------------------------------
# Corpus -> long tables (same schema as save_multi)
# ---------------------------------------------------------------------------
def cohorts_to_tables(cohorts):
    """generate_multi output -> (user_df, peer_df, meta) exactly like save_multi."""
    ru, rp, meta = [], [], []
    for coh in cohorts:
        cid = coh["cohort_id"]
        for tk in coh["tracks"]:
            for u in range(len(tk["user_dfs"])):
                du = tk["user_dfs"][u].copy()
                du["cohort_id"], du["track_id"], du["user_id"] = cid, tk["track_id"], u
                du["t"] = np.arange(len(du))
                du["severity"] = tk["severities"][u]
                du["attributed_peer"] = tk["peer_attrib"][u]
                ru.append(du)
                dp = tk["peer_dfs"][u].copy()
                dp["cohort_id"], dp["track_id"], dp["user_id"] = cid, tk["track_id"], u
                dp["t"] = np.arange(len(dp))
                rp.append(dp)
            meta.append({"cohort_id": cid, "track_id": tk["track_id"], "attacker_id": tk["attacker_id"],
                         "attack_channel": str(tk["attack_channel"]), "poses": tk["poses"]})
    return pd.concat(ru, ignore_index=True), pd.concat(rp, ignore_index=True), meta


def param_tables(rows_u, rows_p, meta):
    return pd.concat(rows_u, ignore_index=True), pd.concat(rows_p, ignore_index=True), meta


def load_main_tables():
    d = J7 / "data" / "v1_published_corpus"
    user = pd.read_csv(d / "user_telemetry.csv.gz")
    peer = pd.read_csv(d / "peer_state.csv.gz")
    meta = [json.loads(l) for l in (d / "tracks_meta.jsonl").read_text().splitlines()]
    return user, peer, meta


# ---------------------------------------------------------------------------
# Frozen per-user backbone (independent calibration corpus, seeds 10000+)
# ---------------------------------------------------------------------------
BACKBONE_PKL = OUT / "models" / "backbone_calib_v3.pkl"


def _generate(n_cohorts, seed, cohort_size=4, attack_prob=0.5, **gp):
    """Version-aware corpus generation -> (user, peer, meta). v1 uses generate_multi
    (the published code path); v2 uses gen_param with the fix applied."""
    if VERSION == "v1" and not gp:
        return cohorts_to_tables(generate_multi(n_cohorts=n_cohorts, cohort_size=cohort_size,
                                                attack_prob=attack_prob, seed=seed))
    from coper_vr.data.gen_param import GenParams, generate_param
    params = GenParams(seed=seed, **{**gen_kwargs(), **gp})
    return param_tables(*generate_param(n_cohorts=n_cohorts, cohort_size=cohort_size,
                                        attack_prob=attack_prob, params=params))


def get_backbone():
    """Re-trains the J7 calibration backbone deterministically in THIS
    environment (the shipped pickle was written by sklearn 1.8 and loads with
    a version warning under 1.6). Same recipe as coper_vr.models.backbone_bridge."""
    if BACKBONE_PKL.exists():
        return pickle.load(open(BACKBONE_PKL, "rb"))["model"]
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    s0, s1 = SEED_RANGES["backbone_calib"]
    cu, _, _ = _generate(s1 - s0, s0)
    X = cu[USER_CH].values.astype(np.float32)
    y = cu["severity"].values
    model = make_pipeline(StandardScaler(), MLPClassifier(
        (64, 32), max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=0))
    model.fit(X, y)
    BACKBONE_PKL.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump({"model": model, "seed_range": SEED_RANGES["backbone_calib"], "n_frames": int(len(y)),
                 "class_counts": np.bincount(y, minlength=4).tolist()}, open(BACKBONE_PKL, "wb"))
    return model


# ---------------------------------------------------------------------------
# Geometry (identical formulas to xrperil_multi / copresence_graph)
# ---------------------------------------------------------------------------
def _fov(pos_i, ori_i, pos_j, fov_i):
    v = np.asarray(pos_j) - np.asarray(pos_i)
    d = np.linalg.norm(v) + 1e-9
    ori = np.asarray(ori_i) / (np.linalg.norm(ori_i) + 1e-9)
    th = np.degrees(np.arccos(np.clip(np.dot(ori, v / d), -1, 1)))
    half = fov_i / 2.0
    return 0.0 if th > half else float(1.0 - th / half)


def track_geometry(poses, audio_range=6.0):
    """D[i,j], OV[i,j], AP[i,j], W[i,j] for victim i, peer j."""
    n = len(poses)
    D = np.zeros((n, n), np.float32); OV = np.zeros_like(D); AP = np.zeros_like(D); W = np.zeros_like(D)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = float(np.linalg.norm(np.asarray(poses[j][0]) - np.asarray(poses[i][0])))
            ov = _fov(poses[i][0], poses[i][1], poses[j][0], poses[i][2])
            ap = 0.0 if d >= audio_range else 1.0 - d / audio_range
            D[i, j], OV[i, j], AP[i, j] = d, ov, ap
            W[i, j] = 1.0 / (1.0 + np.exp(-(1.5 * np.exp(-d / 3.0) + 1.0 * ov + 0.75 * ap)))
    return D, OV, AP, W


# ---------------------------------------------------------------------------
# Tensor build: one row per victim-frame, ordered (cohort, track, victim, t)
# ---------------------------------------------------------------------------
def build_tensors(user, peer, meta, coeffs=GEN_COEFFS, name="corpus", audio_oof_weight=1.0, gain=1.0,
                  with_backbone=True):
    """Returns dict of arrays. Peer slots per victim are the other users in
    increasing id order (a canonical order with no label information)."""
    keys = ["cohort_id", "track_id", "user_id", "t"]
    user = user.sort_values(keys).reset_index(drop=True)
    peer = peer.sort_values(keys).reset_index(drop=True)
    assert (user[keys].values == peer[keys].values).all(), "user/peer misaligned"
    N = int(user.user_id.max()) + 1
    T = int(user.t.max()) + 1
    M = len(meta)
    assert len(user) == M * N * T, "ragged corpus"
    mlk = {(m["cohort_id"], m["track_id"]): m for m in meta}
    tracks = user[["cohort_id", "track_id"]].values[:: N * T]
    U = user[USER_CH].values.astype(np.float32).reshape(M, N, T, 20)
    PC = peer[PEER_NAMES].values.astype(np.float32).reshape(M, N, T, 5)
    SEV = user["severity"].values.reshape(M, N, T)
    AP_ = user["attributed_peer"].values.reshape(M, N, T)

    if with_backbone:
        proba = get_backbone().predict_proba(user[USER_CH].values.astype(np.float32))
    else:
        proba = np.tile([[0.7, 0.2, 0.06, 0.04]], (len(user), 1)) + np.linspace(0, 1e-2, len(user))[:, None]
    assert np.ptp(proba, axis=0).max() > 1e-3, "backbone constant"
    BB = np.concatenate([np.log(np.clip(proba, 1e-9, 1)), (1 - proba.max(1))[:, None]], 1)
    BB = BB.astype(np.float32).reshape(M, N, T, 5)

    rows = M * N * T
    ctx = np.zeros((rows, 4), np.int64)
    V_raw = np.zeros((rows, 20), np.float32); V_bb = np.zeros((rows, 5), np.float32)
    E_src = np.zeros((rows, N - 1), np.int64)
    E_chan = np.zeros((rows, N - 1, 5), np.float32); E_bb = np.zeros((rows, N - 1, 5), np.float32)
    E_geo = np.zeros((rows, N - 1, 3), np.float32); E_w = np.zeros((rows, N - 1), np.float32)
    y_att = np.zeros((rows, N - 1), np.int64)
    sev = np.zeros(rows, np.int64); oracle = np.zeros(rows, np.float32); own = np.zeros(rows, np.float32)
    atk_id = np.zeros(rows, np.int64); atk_ch = np.zeros(rows, np.int64)   # 0 = none, 1..5 channel
    cw = np.asarray(coeffs, np.float32)
    r = 0
    for k in range(M):
        cid, tid = map(int, tracks[k])
        m = mlk[(cid, tid)]
        D, OV, APX, W = track_geometry(m["poses"])
        ach = 0 if not m["attack_channel"] else PEER_NAMES.index(m["attack_channel"]) + 1
        norm = PC[k] / PEER_HI                                      # (N,T,5)
        for i in range(N):
            js = [j for j in range(N) if j != i]
            sl = slice(r, r + T)
            ctx[sl, 0], ctx[sl, 1], ctx[sl, 2], ctx[sl, 3] = cid, tid, np.arange(T), i
            V_raw[sl] = U[k, i]; V_bb[sl] = BB[k, i]
            E_src[sl] = js
            E_chan[sl] = PC[k, js].transpose(1, 0, 2)
            E_bb[sl] = BB[k, js].transpose(1, 0, 2)
            E_geo[sl] = np.stack([D[i, js], OV[i, js], APX[i, js]], 1)[None]
            E_w[sl] = W[i, js][None]
            y_att[sl] = (AP_[k, i][:, None] == np.array(js)[None]).astype(np.int64)
            sev[sl] = SEV[k, i]
            # oracle provocation score (generator formula; used ONLY as a ceiling reference)
            own_prov = (U[k, i, :, 5] / 50.0 + np.abs(U[k, i, :, 2]) / 300.0 + U[k, i, :, 9] / 200.0) / 3.0
            tot = own_prov.copy()
            for j in js:
                dd = float(np.linalg.norm(np.asarray(m["poses"][j][0]) - np.asarray(m["poses"][i][0]))) + 1e-3
                aw = cw[4] if audio_oof_weight is None else audio_oof_weight
                if OV[i, j] == 0.0:
                    tot += gain * aw * norm[j, :, 4] / dd
                else:
                    tot += gain * (norm[j] @ cw) * OV[i, j] / dd
            oracle[sl] = tot; own[sl] = own_prov
            atk_id[sl] = m["attacker_id"]; atk_ch[sl] = ach
            r += T
    out = dict(ctx=ctx, V_raw=V_raw, V_bb=V_bb, E_src=E_src, E_chan=E_chan, E_bb=E_bb, E_geo=E_geo,
               E_w=E_w, y_att=y_att, sev=sev, y=(sev >= 2).astype(np.int64), oracle_score=oracle,
               own_prov=own, atk_id=atk_id, atk_ch=atk_ch, N=np.array(N), T=np.array(T))
    return out


def cached_tensors(name, make_tables, coeffs=GEN_COEFFS):
    f = CACHE / f"{name}.npz"
    if f.exists():
        z = np.load(f)
        return {k: z[k] for k in z.files}
    user, peer, meta = make_tables()
    gk = gen_kwargs()
    t = build_tensors(user, peer, meta, coeffs=coeffs, name=name,
                      audio_oof_weight=gk.get("audio_oof_weight", 1.0), gain=gk.get("peer_gain", 1.0))
    np.savez_compressed(f, **t)
    return t


def main_tensors():
    if VERSION == "v1":
        return cached_tensors("main", load_main_tables)
    return cached_tensors("main", lambda: _generate(60, SEED_RANGES["main"][0]))   # same seeds/telemetry as v1 main


def generated_tensors(name, n_cohorts, cohort_size=4, attack_prob=0.5):
    s0, s1 = SEED_RANGES[name]
    assert n_cohorts <= s1 - s0
    return cached_tensors(name, lambda: _generate(n_cohorts, s0, cohort_size=cohort_size, attack_prob=attack_prob))


def param_tensors(name, gp_kwargs, n_cohorts=40):
    from coper_vr.data.gen_param import GenParams
    s0, s1 = SEED_RANGES[name]
    return cached_tensors(name, lambda: _generate(n_cohorts, s0, **gp_kwargs),
                          coeffs=GenParams(**gp_kwargs).coeffs)


def calibrate_gain_v2(target=0.0138):
    """Label-free: choose the peer gain so that the EXPECTED unsafe prevalence
    E[P(score + N(0,.05) > .30)] on the backbone-calibration seed range equals the
    published corpus prevalence. Uses telemetry/geometry only (no labels, no
    evaluation corpus)."""
    from scipy.optimize import brentq
    from scipy.stats import norm as _n
    from coper_vr.data.gen_param import GenParams, generate_param
    s0, s1 = SEED_RANGES["backbone_calib"]
    u, p, m = param_tables(*generate_param(n_cohorts=s1 - s0, attack_prob=0.5,
                                           params=GenParams(seed=s0, audio_oof_weight=None)))
    t0 = build_tensors(u, p, m, audio_oof_weight=None, gain=1.0, with_backbone=False)
    t1 = build_tensors(u, p, m, audio_oof_weight=None, gain=2.0, with_backbone=False)
    own = t0["own_prov"]; peer = t1["oracle_score"] - t0["oracle_score"]     # = sum of peer terms at gain 1
    f = lambda g: (1 - _n.cdf((0.30 - own - g * peer) / 0.05)).mean() - target
    g = float(brentq(f, 1.0, 50.0))
    (J7 / "data" / "v2_generator").mkdir(parents=True, exist_ok=True)
    (J7 / "data" / "v2_generator" / "gain.json").write_text(json.dumps(
        {"peer_gain": g, "target_expected_prevalence": target, "calibration_seeds": [s0, s1],
         "method": "label-free expected prevalence on backbone-calibration seeds"}, indent=2))
    return g


def subset(t, mask):
    return {k: (v[mask] if (isinstance(v, np.ndarray) and v.ndim > 0 and len(v) == len(mask)) else v)
            for k, v in t.items()}


# ---------------------------------------------------------------------------
# Feature sets (per victim-frame; frame t only)
# ---------------------------------------------------------------------------
def peer_intensity(chan):
    """Normalised excess above the PUBLIC benign band, in [0,1] (per channel)."""
    return np.clip((chan - PEER_BEN_HI) / np.maximum(1e-6, PEER_HI - PEER_BEN_HI), 0, 1)


def prototype_features(t, weights="edge", agg="paper"):
    """The manuscript's 20-D prototype: victim bb(5) | w-mean peer bb(5) |
    w-mean peer channels(5) | max_j w_j*intensity_j (5).
    weights: 'edge' (sigmoid geometry), 'uniform', 'random' (fixed seed, not label-aware)."""
    W = t["E_w"]
    if weights == "uniform":
        W = np.ones_like(W)
    elif weights == "random":
        W = np.random.default_rng(12345).uniform(0.5, 1.0, W.shape).astype(np.float32)
    ws = W / W.sum(1, keepdims=True)
    mean_bb = (ws[..., None] * t["E_bb"]).sum(1)
    mean_ch = (ws[..., None] * t["E_chan"]).sum(1)
    max_int = (W[..., None] * peer_intensity(t["E_chan"])).max(1)
    return np.concatenate([t["V_bb"], mean_bb, mean_ch, max_int], 1).astype(np.float32)


FEATURE_BLOCKS = {"victim_bb": slice(0, 5), "peer_bb": slice(5, 10), "peer_chan_mean": slice(10, 15),
                  "peer_max_int": slice(15, 20)}


# ---------------------------------------------------------------------------
# Cohort-disjoint nested splits
# ---------------------------------------------------------------------------
def outer_inner_splits(ctx, y, seed, n_outer=5, n_inner=4):
    """Yields (fold, tr_idx, va_idx, te_idx) frame-index arrays.
    Outer: StratifiedGroupKFold over cohorts (each cohort tested once per repeat).
    Inner: first StratifiedGroupKFold split of the outer-train cohorts -> validation
    cohorts (model selection, early stopping, calibration, thresholds)."""
    from sklearn.model_selection import StratifiedGroupKFold
    g = ctx[:, 0]
    outer = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed)
    for f, (trv, te) in enumerate(outer.split(np.zeros(len(y)), y, g)):
        inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed + 1000 + f)
        a, b = next(inner.split(np.zeros(len(trv)), y[trv], g[trv]))
        tr, va = trv[a], trv[b]
        assert not (set(g[tr]) & set(g[va])) and not (set(g[tr]) & set(g[te])) and not (set(g[va]) & set(g[te]))
        yield f, tr, va, te


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ece(y, p, n_bins=15, adaptive=False):
    y = np.asarray(y); p = np.asarray(p)
    if adaptive:
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, len(edges) - 2)
    e = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def thresholds_from_val(yv, pv, budgets=(0.01, 0.05, 0.10, 0.20)):
    from sklearn.metrics import precision_recall_curve
    pr, rc, th = precision_recall_curve(yv, pv)
    f1 = 2 * pr[:-1] * rc[:-1] / np.maximum(1e-12, pr[:-1] + rc[:-1])
    out = {"f1opt": float(th[int(np.nanargmax(f1))]) if len(th) else 0.5}
    neg = pv[yv == 0]
    for b in budgets:
        out[f"fpr{int(b * 100)}"] = float(np.quantile(neg, 1 - b))
    return out


def op_metrics(y, p, thr):
    yh = (p >= thr).astype(int)
    tp = int(((yh == 1) & (y == 1)).sum()); fp = int(((yh == 1) & (y == 0)).sum())
    fn = int(((yh == 0) & (y == 1)).sum()); tn = int(((yh == 0) & (y == 0)).sum())
    return confusion_metrics(tp, fp, fn, tn)


def confusion_metrics(tp, fp, fn, tn):
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "mcc": float(mcc), "specificity": spec,
            "bal_acc": (rec + spec) / 2, "fpr": 1 - spec, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def score_metrics(y, p):
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    pc = np.clip(p, 1e-7, 1 - 1e-7)
    return {"roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, pc)), "log_loss": float(log_loss(y, pc, labels=[0, 1])),
            "ece15": ece(y, pc), "ece15_adaptive": ece(y, pc, adaptive=True), "prevalence": float(np.mean(y))}


def cohort_bootstrap(groups, y, p, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    cs = np.unique(groups)
    by = {c: np.where(groups == c)[0] for c in cs}
    v = []
    for _ in range(n):
        idx = np.concatenate([by[c] for c in rng.choice(cs, len(cs))])
        if 0 < y[idx].sum() < len(idx):
            v.append(fn(y[idx], p[idx]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def jdump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=conv))
