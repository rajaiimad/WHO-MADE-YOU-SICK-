"""RQ7 latency: per-frame inference cost of the final detectors on CPU, one
thread, batch = all victims of one cohort-frame (feature assembly included).
1 warm-up + 500 timed runs per cohort size. -> outputs/latency.json"""
from __future__ import annotations

import pickle
import platform
import time

import numpy as np
import torch

from common import CACHE, OUT, PEER_HI, jdump, peer_intensity, prototype_features
from models import MPNN
from run_detection import fixed_cfg


def main(runs=500):
    torch.set_num_threads(1)
    sk = pickle.load(open(OUT / "final" / "proto__MLPpub" / "seed0" / "model.pkl", "rb"))
    hgb = pickle.load(open(OUT / "final" / "proto__HGB" / "seed0" / "model.pkl", "rb"))
    cfg = fixed_cfg("MPNN", "mpnn__MPNN")
    sv, se = pickle.load(open(OUT / "final" / "mpnn__MPNN" / "seed0" / "scalers.pkl", "rb"))
    res = {"cpu": platform.processor(), "python": platform.python_version(), "torch": torch.__version__,
           "threads": 1, "runs": runs, "by_size": {}}
    for N, name in ((2, "size_N2"), (3, "size_N3"), (4, "lockbox"), (6, "size_N6"), (8, "size_N8")):
        z = dict(np.load(CACHE / f"{name}.npz"))
        T = int(z["T"]); rows = np.arange(N) * T            # all victims of track 0, frame 0
        one = {k: (v[rows] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(z["y"]) else v) for k, v in z.items()}
        de = 5 + 5 + 5 + 3 + (1 if cfg.get("logd") else 0)
        net = MPNN(5, de, cfg["h"])
        net.load_state_dict(torch.load(OUT / "final" / "mpnn__MPNN" / "seed0" / "net.pt", map_location="cpu"))
        net.eval()

        def mpnn_call():
            E = np.concatenate([one["E_chan"] / PEER_HI, peer_intensity(one["E_chan"]), one["E_bb"], one["E_geo"]], 2)
            if cfg.get("logd"):
                E = np.concatenate([E, np.log(E[..., :1] + 0.05)], -1)
            with torch.no_grad():
                return torch.sigmoid(net(torch.tensor(sv(one["V_bb"])), torch.tensor(se(E)))).numpy()

        def proto_call(m):
            return m.predict_proba(prototype_features(one))[:, 1]

        out = {}
        for k, fn in (("proto__MLPpub", lambda: proto_call(sk)), ("proto__HGB", lambda: proto_call(hgb)),
                      ("mpnn__MPNN", mpnn_call)):
            fn()
            ts = []
            for _ in range(runs):
                t0 = time.perf_counter(); fn(); ts.append((time.perf_counter() - t0) * 1e3)
            ts = np.array(ts)
            out[k] = {"ms_per_cohort_frame_median": float(np.median(ts)), "ms_per_cohort_frame_p95": float(np.percentile(ts, 95)),
                      "ms_per_victim_median": float(np.median(ts) / N), "meets_60Hz_budget_16.7ms_p95": bool(np.percentile(ts, 95) < 16.7)}
        res["by_size"][N] = out
        print(N, {k: round(v["ms_per_cohort_frame_median"], 3) for k, v in out.items()})
    jdump(res, OUT / "latency.json")


if __name__ == "__main__":
    main()
