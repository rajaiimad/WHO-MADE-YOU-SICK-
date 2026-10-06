# Pre-registration v6 — Stage 5: stronger adaptive attackers
Written 2026-10-06 before any Stage-5 corpus was generated or any detector was scored on one.

Purpose: the paper's adaptive-attacker test only lowers the attack intensity. This stage tests
attackers that change where, how, and with whom they attack, and one that queries a detector.

## Detectors
rule detector, prototype (original MLP), edge-interaction HGB, GAT, MPNN; trained once per seed
(5 seeds, 45/15 cohort split) on the v2 main corpus with the unsafe label and the fixed v2
configurations. Thresholds for the 10% and 1% false-alarm budgets are set on the validation
cohorts of each seed, as in the paper. No detector is retrained on any attack.

## Test corpora
60 new cohorts (seeds 70000-70059), v2 generator. All attacks use the same seeds, so benign users,
benign peers, poses and noise are identical; only the attacker changes.
* **standard**: the v2 attack (reference).
* **oov_audio**: the attacker uses spatial audio and places itself outside the FOV of every
  co-present user, choosing among 64 random positions the one with the largest sum of
  1/distance to the other users that is outside every FOV.
* **switching**: the attacker changes channel every 8 frames (random order over the five
  channels), keeping the ramped intensity profile; the other channels stay benign.
* **split**: two colluding peers (the attacker and one random other peer) attack on the same
  channel, each with half the excess over the benign band (outside the original threat model).
* **blackbox**: per attacked track the attacker draws 24 candidates (channel, intensity scale
  U(0.3, 1.0), temporal profile in {ramp, constant, pulse, burst, intermittent}) plus the
  standard attack, and picks the candidate with the fewest surrogate alarms (MPNN seed 0, its
  10%-budget threshold), subject to an expected harm of at least 50% of the standard attack's
  expected harm on that track. Ties: lower mean surrogate score. The other four MPNN seeds and
  all other detectors are not queried (transfer).

Harm = attacker-induced unsafe victim-frames (Stage-4 definition, with all colluders' terms
removed for `split`), realised and expected.

## Reporting rules fixed in advance
* For every attack: harm (absolute and relative to standard), and for every detector PR-AUC on
  the attacker-induced and unsafe targets, and recall of attacker-induced frames at the 10% and
  1% budgets. Black-box results are reported separately for the queried seed and the other seeds.
* Decision rule for the paper's "robust to adaptive attackers" sentence: it stays only if the
  MPNN's recall of attacker-induced frames at the 10% budget is >= 0.80 under every attack whose
  realised harm is >= 25% of the standard attack's. Otherwise the sentence is removed and the
  failing attack is reported as a limitation.

## Amendment (2026-10-06, written after generating the attack corpora and computing harm, before any detector was trained or scored for Stage 5)
The `oov_audio` attacker as specified places itself as close as 0.16 m to a victim (1st
percentile; benign pairs: 0.69 m), because nothing in the generator stops avatars from overlapping
and the out-of-view audio term grows as 1/distance. Its harm is 10x that of the standard attack.
`oov_audio` is still reported exactly as specified. An added variant, `oov_audio_d1`, is the
same attacker restricted to positions at least 1.0 m from every other user. It is labelled as an
amendment wherever it is reported, and the decision rule applies to both variants.
