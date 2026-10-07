# Pre-registration v9 — proximity-only attacker (round-4 review)
Written 2026-10-07, before the corpus was generated or scored.

## Why
The within-band attacker of v7 holds its channel exactly at the top of the benign band. Training attacks also start near that value, so detectors and attribution models may be keying on the constant top-of-band value rather than on proximity (round-4 methods reviewer).

## Variant
`near_benign`: the same pose rule as the v7 within-band attacker (at least 1.0 m from every user; among 64 candidates, the one maximising sum_i v_i / d_i). All of the attacker's channels are drawn from the benign band exactly like a benign peer's (no clamping), so only its position is adversarial. Seeds 70000–70059 and the v2 gain, as in Stage 5/7.

## Measured (detectors and attribution trained once on the v2 main corpus; no retraining)
* Harm (attacker-induced unsafe victim-frames) relative to the standard attack.
* For MPNN, raw-channel HGB, excess-only HGB, GAT, rule, prototype: PR-AUC on the attacker-induced target, and recall at the 10% and 1% budgets.
* Attribution top-1 and edge PR-AUC of the rule and of the learned channel-specific model.

## Reporting rule
Reported whatever it shows. If learned attribution top-1 falls below 0.5, the paper must state that the v7 within-band attribution result depended on the top-of-band value.
