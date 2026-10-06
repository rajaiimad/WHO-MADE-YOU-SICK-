# Pre-registration — generator v2 rerun (written 2026-10-05, before any v2 model was trained)

Generator v2 = XR-PERIL-Multi with the out-of-FOV audio weight fixed (coeffs[4] = 0.10 applied
in and out of FOV) and one global peer-coupling gain = 4.0056, calibrated label-free on the
backbone-calibration seeds (10000-10059) so the expected unsafe prevalence is 1.38%
(`outputs_v2/gain.json`). Thresholds, channel coefficients, seeds, cohort/track sizes unchanged.

Everything else is frozen from the v1 analysis and is NOT re-tuned by the analyst:
model families, hyper-parameter grids, nested-CV protocol, Tier-2 rule (modal Tier-1 config),
attribution feature sets, external corpora and their seed ranges, lock-box (120 cohorts, seeds
20000-20119, scored once), metrics, primary-model rule (highest nested-CV pooled PR-AUC),
and analysis scripts. The per-user backbone is re-trained with the same recipe on the v2
calibration corpus (its labels come from the generator).
