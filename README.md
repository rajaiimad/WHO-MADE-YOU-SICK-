# XR-PERIL-Multi: detecting and attributing peer-induced cybersickness risk in multi-user VR

Code and data for the paper *"Who Made You Sick? Detecting and Attributing Peer-Induced
Cybersickness Risks in Multi-User Virtual Reality with Learned Co-Presence Graph Aggregation."*

All data are **synthetic**. Severity labels come from a known generator; they are not
human-reported cybersickness.

## Contents

```
coper_vr/                     benchmark generator
  config.py                   peer channels, benign bands, corpus defaults
  data/xrperil_multi.py       original generator (v1)
  data/gen_param.py           parametrised generator; v2 fix = audio_oof_weight=None + peer_gain
rigorous_rerun/               experiment pipeline (all paper numbers come from here)
  common.py                   corpora, tensors, features, nested splits, metrics
  models.py                   LR, HGB, MLP, MPNN; nested model selection; Platt scaling
  leakage_checks.py           11 automated leakage checks
  run_detection.py            nested CV detection (Tier-1 models, Tier-2 ablations)
  run_temporal.py             RQ3 causal temporal windows
  run_attribution.py          per-edge attribution + leave-one-channel-out
  run_final.py                final models; lock-box and stress corpora (scored once)
  run_counterfactual.py       counterfactual source masking
  run_federated.py            federated MPNN with poisoning attacks
  run_latency.py              CPU inference latency
  analyze_*.py, make_tables.py, make_figures.py   metrics, tables, figures from saved predictions
  analyze_per_channel.py      lock-box detection per attack channel
  run_v2_pipeline.sh          runs everything after Tier-1 for generator v2
  PREREGISTRATION*.md         lock-box pre-registration (v1 and v2)
data/
  v1_published_corpus/        original corpus (60 cohorts x 4 users x 8 tracks x 48 frames)
  v2_main_corpus/             development corpus, corrected generator (v2)
  v2_lockbox_corpus/          lock-box, 120 unseen cohorts (v2)
  v2_generator/gain.json      label-free calibrated peer-coupling gain (4.006)
results/v1, results/v2        summary results (JSON) and TABLES.md for each generator version
```

Each corpus folder holds `user_telemetry.csv.gz` (20 telemetry channels, `severity` 0-3,
`attributed_peer`), `peer_state.csv.gz` (5 peer-controllable channels) and
`tracks_meta.jsonl` (attacker id, attack channel, avatar poses). Rows are keyed by
`cohort_id, track_id, user_id, t`. A frame is unsafe when `severity >= 2`.

Every other corpus (cohort-size, generator-shift, adaptive-attacker, calibration) is
regenerated deterministically from disjoint seed ranges listed in
`rigorous_rerun/common.py` (`SEED_RANGES`).

## Generator versions

* **v1** (original): an out-of-view peer's spatial audio is weighted 1.0 instead of 0.10,
  so 93% of unsafe frames come from audio attacks.
* **v2** (used in the paper): audio weight 0.10 in and out of view, plus one global
  peer-coupling gain calibrated without labels to keep prevalence near 1.4%.

Select the version with the environment variable `XRP_GEN` (`v1` or `v2`, default `v1`).
Outputs go to `rigorous_rerun/outputs/` (v1) or `rigorous_rerun/outputs_v2/` (v2).

## Setup

```
pip install -r requirements.txt
```

Tested with Python 3.13, PyTorch 2.11, scikit-learn 1.6, NumPy 2.2, pandas 2.2.
A CUDA GPU is optional (the neural models run on CPU, more slowly).

## Reproduce (generator v2)

```
cd rigorous_rerun
export XRP_GEN=v2                      # PowerShell: $env:XRP_GEN="v2"
python -c "from common import calibrate_gain_v2; print(calibrate_gain_v2())"   # optional: re-derives data/v2_generator/gain.json
python leakage_checks.py
mkdir -p outputs_v2/logs
# Tier-1 nested CV (run_v2_pipeline.sh waits for these three logs)
python run_detection.py proto__LR proto__MLPpub victim_bb__MLPpub proto__HGB victim_all__HGB > outputs_v2/logs/tier1_cpu.log 2>&1
python run_detection.py mpnn__MPNN mpnn_raw__MPNN > outputs_v2/logs/tier1_mpnn.log 2>&1
python run_detection.py proto__MLP > outputs_v2/logs/tier1_mlp.log 2>&1
python run_attribution.py
bash run_v2_pipeline.sh                # Tier-2, temporal, final/lock-box, federated, counterfactual, latency, analysis
python make_tables.py                  # -> outputs_v2/TABLES.md
python make_figures.py                 # -> ../figures/
```

The first run retrains the frozen per-user backbone on the calibration seed range and
builds tensor caches. A full v2 run took about 9 hours on one laptop (most of it the
federated experiments).

## Evaluation protocol (summary)

* Cohort-disjoint splits only; the cohort is the unit of independence.
* Development: 5 repeats x 5-fold stratified group cross-validation; inside each fold,
  36 training / 12 validation cohorts. Hyper-parameters, early stopping, Platt
  calibration and thresholds use validation cohorts only.
* Lock-box: 120 fresh cohorts, scored once by pre-registered models.
* Reference ceiling: P(score + N(0, 0.05) > 0.30) from the generator's hidden score.

## License

Code, generated corpora, and results are released under the MIT License (see `LICENSE`). All data in this repository are synthetic and were produced by the generator in this repository.
