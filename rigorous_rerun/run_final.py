"""
rigorous_rerun.run_final
========================
Stage B: train each pre-declared final model ONCE per seed on the main corpus
(45 train / 15 validation cohorts; validation -> early stopping, Platt, thresholds)
and score every EXTERNAL corpus, none of which is used for any fitting or choice:

  lockbox          120 fresh cohorts, same generator (confirmation, scored once)
  size_N2..N8      cohort-size transfer (train N=4 -> test N in {2,3,6,8})
  shift_*          generator shifts (coeff / thresholds / pulse / low intensity /
                   time-decoupled channels; labels re-simulated by the generator)
  adapt_a*         attacker intensity sweep alpha in {1,.8,.6,.4,.2}; labels
                   RE-SIMULATED (fixes the old E3 flaw)

Configurations: for each family the configuration most often selected by the
nested CV on the main corpus (run_detection.fixed_cfg) -- declared in
PREREGISTRATION.md before this script was run.

Writes outputs/final/<model>/seed<s>/{model.pkl|net.pt, scalers.pkl, preds.npz, meta.json}
"""
from __future__ import annotations

import pickle
import sys
import time

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from common import (OUT, SEEDS, generated_tensors, jdump, main_tensors, param_tensors, thresholds_from_val)
from models import FAMILIES, platt
from run_detection import SPECS, fixed_cfg

FINAL_MODELS = ["victim_bb__MLPpub", "victim_all__HGB", "proto__MLPpub", "proto__LR", "proto__HGB",
                "proto__MLP", "mpnn__MPNN", "mpnn_raw__MPNN"]


def external_corpora():
    C = {"lockbox": generated_tensors("lockbox", 120)}
    for n in (2, 3, 6, 8):
        C[f"size_N{n}"] = generated_tensors(f"size_N{n}", 60, cohort_size=n)
    C["shift_same"] = param_tensors("shift_same", {})
    C["shift_coeff"] = param_tensors("shift_coeff", {"coeffs": (0.10, 0.10, 0.35, 0.15, 0.30)})
    C["shift_thresh"] = param_tensors("shift_thresh", {"thresholds": (0.20, 0.40, 0.65)})
    C["shift_pulse"] = param_tensors("shift_pulse", {"temporal_profile": "pulse"})
    C["shift_intensity"] = param_tensors("shift_intensity", {"intensity_scale": 0.6})
    C["shift_decouple"] = param_tensors("shift_decouple", {"decouple": True})
    for a in (1.0, 0.8, 0.6, 0.4, 0.2):
        C[f"adapt_a{a}"] = param_tensors(f"adapt_a{a}", {"intensity_scale": a})
    # feature-level decoupling control: every lock-box victim-frame receives the incoming
    # edges (peer channels, peer backbone outputs, geometry) of a random other victim-frame
    lb = C["lockbox"]
    perm = np.random.default_rng(99).permutation(len(lb["y"]))
    C["lockbox_peerperm"] = dict(lb, **{k: lb[k][perm] for k in ("E_src", "E_chan", "E_bb", "E_geo", "E_w")})
    return C


def train_val_split(ctx, y, seed):
    sgk = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
    tr, va = next(sgk.split(np.zeros(len(y)), y, ctx[:, 0]))
    assert not set(ctx[tr, 0]) & set(ctx[va, 0])
    return tr, va


def run(name, corpora):
    import torch
    fam, builder = SPECS[name]
    cfg = {} if fam == "MLPpub" else fixed_cfg(fam, name)
    cfg.pop("n_iter", None)
    t = main_tensors()
    y, ctx = t["y"], t["ctx"]
    kind = builder(t)
    ext = {k: builder(c) for k, c in corpora.items()}
    for s in SEEDS:
        t0 = time.time()
        tr, va = train_val_split(ctx, y, s)
        if kind[0] == "tab":
            d = {"Xtr": kind[1][tr], "ytr": y[tr], "Xva": kind[1][va], "yva": y[va],
                 "Xte": {k: v[1] for k, v in ext.items()}}
        else:
            d = {"Vtr": kind[1][tr], "Etr": kind[2][tr], "ytr": y[tr], "Vva": kind[1][va], "Eva": kind[2][va],
                 "yva": y[va], "Gte": {k: (v[1], v[2]) for k, v in ext.items()}}
        pv, pt = FAMILIES[fam](d, cfg, s)
        cal = platt(pv, y[va])
        thr = thresholds_from_val(y[va], pv)
        od = OUT / "final" / name / f"seed{s}"
        od.mkdir(parents=True, exist_ok=True)
        meta = d.get("_meta", {})
        if "net" in meta:
            torch.save(meta["net"].state_dict(), od / "net.pt")
            pickle.dump(meta["scalers"], open(od / "scalers.pkl", "wb"))
        if "model" in meta:
            pickle.dump(meta["model"], open(od / "model.pkl", "wb"))
        np.savez_compressed(od / "preds.npz", p_val=pv, val_rows=va,
                            **{f"p__{k}": v for k, v in pt.items()},
                            **{f"pcal__{k}": cal(v) for k, v in pt.items()})
        jdump({"model": name, "family": fam, "cfg": cfg, "n_iter": meta.get("n_iter"), "seed": s,
               "thresholds": thr, "train_cohorts": sorted(set(ctx[tr, 0].tolist())),
               "val_cohorts": sorted(set(ctx[va, 0].tolist()))}, od / "meta.json")
        print(f"[{name}] seed {s} done ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    corpora = external_corpora()
    if sys.argv[1:] == ["--build-only"]:
        print({k: len(v["y"]) for k, v in corpora.items()})
        sys.exit(0)
    for n in sys.argv[1:] or FINAL_MODELS:
        run(n, corpora)
