"""
rigorous_rerun.run_detection
============================
Nested, repeated, cohort-disjoint evaluation of per-victim detection models on
the main XR-PERIL-Multi corpus.

  repeats : SEEDS = (0, 1, 2, 42, 2024)   (fold assignment + model init)
  outer   : 5-fold StratifiedGroupKFold over the 60 cohorts (test, scored once)
  inner   : 1 split of the outer-train cohorts into train (3/4) / validation (1/4)
            -> hyper-parameter selection, early stopping, Platt calibration,
               operating thresholds. TEST cohorts are never seen before scoring.

Writes outputs/detection/<model>/preds.npz (+ selection.json); metrics are
computed afterwards by analyze_detection.py from these saved predictions.

usage: python run_detection.py MODEL [MODEL ...]      (see SPECS)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np

from common import (OUT, SEEDS, PEER_HI, jdump, main_tensors, outer_inner_splits, peer_intensity,
                    prototype_features, thresholds_from_val)
from models import GRIDS, platt, select_and_predict


# ---------------------------------------------------------------------------
# Input builders. Each returns ('tab', X) or ('graph', V, E) for a tensor dict.
# ---------------------------------------------------------------------------
def tab(fn):
    return lambda t: ("tab", fn(t))


def graph(v_blocks, e_blocks, perm_peers=False):
    def f(t):
        V = np.concatenate([{"bb": t["V_bb"], "raw": t["V_raw"]}[b] for b in v_blocks], 1) if v_blocks else \
            np.zeros((len(t["y"]), 1), np.float32)
        parts = {"chan": lambda: t["E_chan"] / PEER_HI, "int": lambda: peer_intensity(t["E_chan"]),
                 "bb": lambda: t["E_bb"], "geo": lambda: t["E_geo"]}
        E = np.concatenate([parts[b]() for b in e_blocks], 2).astype(np.float32)
        return ("graph", V.astype(np.float32), E)
    return f


def _geo_flat(t):
    return t["E_geo"].reshape(len(t["y"]), -1)


SPECS = {
    # --- Tier 1: nested HPO --------------------------------------------------
    "victim_bb__MLPpub": ("MLPpub", tab(lambda t: t["V_bb"])),
    "victim_all__HGB": ("HGB", tab(lambda t: np.concatenate([t["V_bb"], t["V_raw"]], 1))),
    "proto__MLPpub": ("MLPpub", tab(prototype_features)),
    "proto__LR": ("LR", tab(prototype_features)),
    "proto__HGB": ("HGB", tab(prototype_features)),
    "proto__MLP": ("MLP", tab(prototype_features)),
    "mpnn__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "bb", "geo"])),
    "mpnn_raw__MPNN": ("MPNN", graph(["bb", "raw"], ["chan", "int", "bb", "geo"])),
}

# Tier-2 isolation / ablation specs use a FIXED configuration (no per-fold search)
# chosen by a pre-declared rule: the configuration selected most often for that
# family in Tier 1 (see fixed_cfg()).
TIER2 = {
    # feature-source isolation (HGB)
    "iso_victim_bb__HGB": ("HGB", tab(lambda t: t["V_bb"])),
    "iso_victim_raw__HGB": ("HGB", tab(lambda t: t["V_raw"])),
    "iso_peer_bb__HGB": ("HGB", tab(lambda t: prototype_features(t)[:, 5:10])),
    "iso_peer_chan_mean__HGB": ("HGB", tab(lambda t: prototype_features(t)[:, 10:15])),
    "iso_peer_max_int__HGB": ("HGB", tab(lambda t: prototype_features(t)[:, 15:20])),
    "iso_peer_channels__HGB": ("HGB", tab(lambda t: prototype_features(t)[:, 10:20])),
    "iso_uniform_w__HGB": ("HGB", tab(lambda t: prototype_features(t, weights="uniform"))),
    "iso_random_w__HGB": ("HGB", tab(lambda t: prototype_features(t, weights="random"))),
    "iso_geometry_only__HGB": ("HGB", tab(_geo_flat)),
    "iso_permuted_peers__HGB": ("HGB", "PERMUTED"),
    # MPNN ablations
    "abl_mpnn_no_geo__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "bb"])),
    "abl_mpnn_no_peer_bb__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "geo"])),
    "abl_mpnn_no_int__MPNN": ("MPNN", graph(["bb"], ["chan", "bb", "geo"])),
    "abl_mpnn_no_victim__MPNN": ("MPNN", graph([], ["chan", "int", "bb", "geo"])),
    "abl_mpnn_mean_agg__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "bb", "geo"])),
    "abl_mpnn_max_agg__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "bb", "geo"])),
    "abl_mpnn_no_posw__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "bb", "geo"])),
    "abl_mpnn_posw__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "bb", "geo"])),
    "abl_mpnn_no_logd__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "bb", "geo"])),
    "abl_mpnn_logd__MPNN": ("MPNN", graph(["bb"], ["chan", "int", "bb", "geo"])),
    "abl_hgb_no_cw__HGB": ("HGB", tab(prototype_features)),
    "abl_hgb_cw__HGB": ("HGB", tab(prototype_features)),
}
TIER2_OVERRIDES = {"abl_mpnn_mean_agg__MPNN": {"agg": "mean"}, "abl_mpnn_max_agg__MPNN": {"agg": "max"},
                   "abl_mpnn_no_posw__MPNN": {"pw": "none"}, "abl_mpnn_posw__MPNN": {"pw": "sqrt"},
                   "abl_mpnn_no_logd__MPNN": {"logd": False}, "abl_mpnn_logd__MPNN": {"logd": True},
                   "abl_hgb_no_cw__HGB": {"cw": None}, "abl_hgb_cw__HGB": {"cw": "balanced"}}


def fixed_cfg(family, tier1_model):
    """Most frequently selected Tier-1 configuration for this family (ties -> first)."""
    sel = json.loads((OUT / "detection" / tier1_model / "selection.json").read_text())
    cfgs = [json.dumps({k: v for k, v in s["cfg"].items() if k != "n_iter"}, sort_keys=True) for s in sel]
    best = max(set(cfgs), key=lambda c: (cfgs.count(c), -cfgs.index(c)))
    return json.loads(best)


def permuted_peer_features(t, rng):
    """Break the peer->victim correspondence: each victim-frame receives the
    peer aggregates of a random other victim-frame (victim block untouched)."""
    X = prototype_features(t)
    X[:, 5:] = X[rng.permutation(len(X)), 5:]
    return X


def make_fold_data(kind, built, tr, va, te, y):
    if kind[0] == "tab":
        X = kind[1]
        return {"Xtr": X[tr], "ytr": y[tr], "Xva": X[va], "yva": y[va], "Xte": {"test": X[te]}}
    _, V, E = kind
    return {"Vtr": V[tr], "Etr": E[tr], "ytr": y[tr], "Vva": V[va], "Eva": E[va], "yva": y[va],
            "Gte": {"test": (V[te], E[te])}}


def run(name):
    t0 = time.time()
    t = main_tensors()
    y, ctx = t["y"], t["ctx"]
    spec = SPECS.get(name) or TIER2[name]
    fam, builder = spec
    grid = None
    if name in TIER2:
        ref = {"HGB": "proto__HGB", "MPNN": "mpnn__MPNN"}[fam]
        cfg = fixed_cfg(fam, ref)
        cfg.update(TIER2_OVERRIDES.get(name, {}))
        grid = [cfg]
    out_dir = OUT / "detection" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {k: [] for k in ("seed", "fold", "row", "p", "p_cal")}
    sel = []
    for s in SEEDS:
        if builder == "PERMUTED":
            rng = np.random.default_rng(s)
            kind = ("tab", permuted_peer_features(t, rng))
        else:
            kind = builder(t)
        for f, tr, va, te in outer_inner_splits(ctx, y, s):
            d = make_fold_data(kind, t, tr, va, te, y)
            pv, pt, cfg, vap, log = select_and_predict(fam, d, s, grid)
            cal = platt(pv, y[va])
            thr = thresholds_from_val(y[va], pv)
            sel.append({"seed": s, "fold": f, "cfg": cfg, "val_ap": vap, "thresholds": thr,
                        "n_train_cohorts": int(len(np.unique(ctx[tr, 0]))),
                        "n_val_cohorts": int(len(np.unique(ctx[va, 0]))),
                        "n_test_cohorts": int(len(np.unique(ctx[te, 0]))),
                        "val_pos": int(y[va].sum()), "search": log})
            for k, v in (("seed", np.full(len(te), s)), ("fold", np.full(len(te), f)), ("row", te),
                         ("p", pt["test"]), ("p_cal", cal(pt["test"]))):
                preds[k].append(v)
            print(f"[{name}] seed {s} fold {f}  val_AP {vap:.3f}  cfg {cfg}  ({time.time() - t0:.0f}s)",
                  flush=True)
    np.savez_compressed(out_dir / "preds.npz", **{k: np.concatenate(v) for k, v in preds.items()})
    jdump(sel, out_dir / "selection.json")
    print(f"[{name}] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        run(n)
