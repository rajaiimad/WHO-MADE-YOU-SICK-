# Pre-registration v5 — Stage 4: attacker-induced label
Written 2026-10-06 before any attacker-induced label was computed or any model was trained on one.

Purpose: the v2 "unsafe" label mixes harm caused by the attacker with harm that arises without
any attacker (19.4% of unsafe frames come from benign tracks). The paper's threat model concerns
the attacker. This stage adds a second target that measures exactly that.

## Targets
* **Unsafe** (safety view, unchanged): severity >= 2, i.e. s > 0.30, with
  s = own + g * sum_j c_{j->i} + noise.
* **Attacker-induced** (security view, new): the frame is unsafe AND would not be unsafe if the
  attacker's contribution were removed with the same noise draw:
  s > 0.30 and s - g * c_{a->i} <= 0.30, where a is the track's attacker.
  On benign tracks, and for the attacker's own frames, the label is 0.
* Ceiling for the new target: P(0.30 - score < n <= 0.30 - score + g c_{a->i}), n ~ N(0, 0.05^2).

## Corpora
The v2 main corpus (60 cohorts, the main seeds) for development and the v2 lock-box (120 cohorts,
seeds 20000-20119) for testing, regenerated with the Stage-3 generator, which must reproduce the
stored v2 labels exactly (asserted). The lock-box has already been scored once for the "unsafe"
target; for the attacker-induced target it is scored once, here.

## Models (no re-tuning; v2 nested-CV modal configurations)
rule detector, prototype (original MLP), edge-interaction HGB, GAT, MPNN. Each is trained on
the development corpus (45 train / 15 validation cohorts, 5 seeds) in two ways:
(a) on the unsafe label, (b) on the attacker-induced label. Each trained model is scored on the
lock-box against both targets.

## Reporting rules fixed in advance
* Composition of unsafe frames: attacker-induced, unsafe on attacked tracks but not
  attacker-induced, unsafe on benign tracks.
* Every model x training target x test target is reported (ROC-AUC, PR-AUC, 95% cohort-bootstrap
  CI, prevalence, ceiling).
* Primary question: on the attacker-induced target, does the MPNN trained on that target exceed
  the rule, the GAT and the edge-interaction HGB in lock-box PR-AUC (paired bootstrap CI excluding 0)?
* Secondary: does training on the attacker-induced label improve over training on the unsafe
  label, when both are tested on the attacker-induced target?
