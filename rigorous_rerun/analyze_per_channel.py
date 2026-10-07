"""Lock-box detection per attack channel (tracks attacked through that channel +
all benign tracks), seed-averaged scores of the final models and the generator
ceiling. -> outputs*/per_channel_detection_lockbox.json"""
import numpy as np
from scipy.stats import norm
from sklearn.metrics import average_precision_score, roc_auc_score

from common import OUT, PEER_NAMES, SEEDS, generated_tensors, jdump


def main():
    lb = generated_tensors("lockbox", 120)
    y, ch = lb["y"], lb["atk_ch"]
    P = {m: np.mean([np.load(OUT / "final" / m / f"seed{s}" / "preds.npz")["p__lockbox"] for s in SEEDS], 0)
         for m in ("proto__MLPpub", "proto__LR", "proto__HGB", "mpnn__MPNN", "rule__RULE", "edgefeat__HGB", "gat__GAT",
                   "edgefeatraw__HGB")}
    P["ORACLE"] = 1 - norm.cdf((0.30 - lb["oracle_score"]) / 0.05)
    out = {}
    for c in range(1, 6):
        m = (ch == 0) | (ch == c)
        r = {"unsafe_frames": int(y[ch == c].sum()), "unsafe_share": round(float(y[ch == c].sum() / y.sum()), 3)}
        for k, p in P.items():
            r[k] = (round(float(roc_auc_score(y[m], p[m])), 3), round(float(average_precision_score(y[m], p[m])), 3))
        out[PEER_NAMES[c - 1]] = r
        print(PEER_NAMES[c - 1], r)
    jdump(out, OUT / "per_channel_detection_lockbox.json")


if __name__ == "__main__":
    main()
