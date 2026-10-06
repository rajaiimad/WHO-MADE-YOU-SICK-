# Pre-registration of the confirmation (lock-box) analysis

Written on 2026-10-05, while the Tier-1 nested-CV runs were still in progress
and before any model scored the lock-box corpus. Disclosure: a one-fold
timing smoke test (seed 0, outer fold 0, first grid configuration of each
family) had been run and its test-fold scores were seen.

## Lock-box corpus
* 120 fresh cohorts (4 users, 8 tracks x 48 frames, attack_prob 0.5) from the
  unchanged XR-PERIL-Multi generator (`generate_multi`), per-cohort seeds
  20000-20119. These seeds are disjoint from every other corpus
  (`common.SEED_RANGES`, asserted at import).
* No lock-box row is used for fitting, model selection, early stopping,
  calibration or threshold choice. The lock-box is scored exactly once by
  `run_final.py`; its metrics are computed by `analyze_final.py`.

## Models (all evaluated; none is dropped after seeing lock-box results)
`victim_bb__MLPpub, victim_all__HGB, proto__MLPpub, proto__LR, proto__HGB,
proto__MLP, mpnn__MPNN, mpnn_raw__MPNN` (definitions in `run_detection.SPECS`).

Configuration of each family = the configuration selected most often by the
inner validation in the 25 nested-CV folds on the main corpus
(`run_detection.fixed_cfg`). Training: main corpus, 45 train / 15 validation
cohorts (StratifiedGroupKFold, seed s), 5 seeds {0, 1, 2, 42, 2024}.

## Primary model (decision rule fixed in advance)
The "proposed" detector is the Tier-1 model with the highest mean pooled
out-of-fold PR-AUC in the main-corpus nested CV. If a simpler model
(LR or HGB on the prototype features) is within one between-repeat SD of it,
that is reported as such.

## Metrics
Primary: PR-AUC and ROC-AUC (mean +- SD over 5 seeds; cohort-bootstrap 95% CI
over lock-box cohorts, B = 2000). Secondary: precision, recall, F1, MCC,
balanced accuracy, specificity at the validation-selected F1-optimal threshold
and at the validation 10%-FPR budget (achieved test FPR reported); Brier,
log-loss and ECE of raw and Platt-calibrated scores. Reference ceiling: the
generator oracle P(score + N(0, 0.05) > 0.30).
