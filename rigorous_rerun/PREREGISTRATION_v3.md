# Pre-registration v3 — additional baselines (Stage 2), written 2026-10-06 before any of them was run

Context: the committee review (`review_committee/`) found that the only graph baseline was the
authors' own averaging prototype. This file fixes, before running anything, which baselines are
added, how they are evaluated, and how the primary model will be chosen. Everything else
(generator v2, corpora, seeds, nested CV, lock-box, metrics, analysis scripts) is unchanged from
`PREREGISTRATION_v2.md`.

## New detection models (generator v2, same nested CV and same lock-box)

Notation per incoming edge j of victim i at frame t: `x_j` = peer channels / physical maximum
(5), `e_j` = excess over the public benign band (5), `b_j` = peer backbone output (5),
`d_j, v_j, a_j` = distance, view (FOV) overlap, audio proximity.

1. `rule__RULE` (no learning). score_i = sum_j max_k(e_jk) * (0.5 + 0.5 v_j) / (d_j + 0.5).
   Equal channel weights; the geometry term is a generic proximity/visibility prior, not the
   generator's coefficients. Only the operating thresholds are chosen on validation cohorts.
2. `edgefeat__HGB`. HistGB on explicit per-edge interaction features, aggregated over peers:
   for each channel k, sum_j and max_j of e_jk * v_j / (d_j + 0.1) and of e_jk / (d_j + 0.1)
   (20 values), plus the victim backbone output (5). Same 16-configuration HGB grid.
3. `gcn__GCN`. Graph convolution on the star graph of victim i and its peers; peer node
   features [x_j, e_j, b_j]; no edge features (standard GCN); symmetric normalisation with
   self-loops; one layer then an MLP readout with b_i.
4. `gcngeo__GCN`. As 3, with the edge geometry (d_j, v_j, a_j) appended to each peer's node
   features (linear message, then aggregation).
5. `gat__GAT`. GATv2-style attention with edge features: attention logits from
   [b_i, x_j, e_j, b_j, geometry]; messages are linear in the peer features; 4 heads.

Neural grids (3-5): width {32, 64} x learning rate {1e-3, 3e-3} x positive-class weight
{1, sqrt(n-/n+)} = 8 configurations, selected by validation PR-AUC, same trainer as the MPNN.

## New attribution baselines (no learning; same 25 folds)

6. `rule_excess`: edge score = max_k e_jk.
7. `rule_excess_geo`: edge score = max_k e_jk * (0.5 + 0.5 v_j) / (d_j + 0.5).

## Decision rules fixed in advance

* Primary detector = the model with the highest mean pooled nested-CV PR-AUC. If a simpler model
  (rule < HGB < GCN < GAT/MPNN in complexity) is within one between-repeat SD of the best, the
  simpler model is named primary.
* All results are reported whatever they show. No model or grid is changed after its test
  results are seen.
* Lock-box: each new model is trained once per seed (fixed modal configuration) and scored once
  on the same 120-cohort lock-box.
