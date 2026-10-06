# Pre-registration v4 — Stage 3 generator sensitivity and harder variants
Written 2026-10-06 before any Stage-3 corpus was generated or any model was trained on one.

Purpose: test whether the conclusions of the v2 study survive (a) a different coupling gain,
(b) a label that is NOT a sum over peers (the circularity test), (c) a label that accumulates
over time (makes RQ3 a real question), (d) benign peers that occasionally leave the benign band
(makes attribution non-trivial), and (e) avatars that move within a track.

## Variants (all other generator settings = v2)
| id | change |
|---|---|
| g2, g3, g4, g5, g6 | peer gain g in {2, 3, 4, 5, 6}; nothing else changed (g4 ~ v2) |
| max | score = own + max_j c_{j->i} instead of the sum; g recalibrated label-free |
| dose | score = own + P(t), P(t) = 0.9 P(t-1) + 0.1 sum_j c_{j->i}(t); g recalibrated label-free |
| spikes | each benign peer, each frame, with prob. 0.05 sets one random channel to U(top of benign band, top + 0.5 (max - top)); g = v2 gain |
| moving | each avatar moves linearly (velocity N(0, 0.04 m/frame) per axis) and turns (N(0, 1.5) deg/frame) within a track; geometry recomputed every frame; g = v2 gain |

Label-free recalibration (max, dose): g chosen so that the expected unsafe prevalence on 30
calibration cohorts (seeds 10000-10029) is 1.38%, exactly as for v2.

## Corpora
Per variant: development corpus 60 cohorts (seeds 60000-60059) and test corpus 60 cohorts
(seeds 61000-61059). The same seeds are used for every variant, so differences between variants
come from the generator change, not from different samples. These seed ranges are disjoint from
all earlier corpora.

## Models (no re-tuning; configurations fixed to the v2 nested-CV modal configurations)
rule detector, prototype (original MLP), edge-interaction HGB, GAT, MPNN, and the generator
ceiling. For the `dose` variant only, two temporal models are added: edge-interaction HGB and MPNN
with the causal mean over the previous 8 frames of each edge's channels and excess features
appended. Training: development corpus, 45 train / 15 validation cohorts (seed s), 5 seeds;
evaluation once on the test corpus.

Attribution per variant: per-edge HGB (channel-specific features) and the rule
`max excess x geometry`; top-1, edge ROC-AUC, edge PR-AUC.

## Reporting rules fixed in advance
* Every variant and every model is reported, whatever it shows.
* Primary question (circularity): in the `max` variant, does the MPNN still exceed the prototype
  and the GAT in PR-AUC? A "no" is reported as a limitation of the MPNN claim.
* RQ3 is answered by the `dose` variant: do temporal models exceed the per-frame models?
* Attribution: in `spikes`, does the learned model exceed the rule in top-1?
