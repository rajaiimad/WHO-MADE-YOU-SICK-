#!/bin/bash
# Stage 2+ of the generator-v2 rerun; waits for Tier-1, then runs everything else.
export XRP_GEN=v2 PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
L=outputs_v2/logs
until [ "$(grep -c 'done in' $L/tier1_cpu.log)" -ge 5 ] && grep -q 'mpnn_raw__MPNN\] done' $L/tier1_mpnn.log && grep -q 'done in' $L/tier1_mlp.log; do sleep 60; done
echo "tier1 done $(date)"
python run_detection.py iso_victim_bb__HGB iso_victim_raw__HGB iso_peer_bb__HGB iso_peer_chan_mean__HGB iso_peer_max_int__HGB iso_peer_channels__HGB iso_uniform_w__HGB iso_random_w__HGB iso_geometry_only__HGB iso_permuted_peers__HGB abl_hgb_no_cw__HGB abl_hgb_cw__HGB > $L/tier2_hgb.log 2>&1 &
P1=$!
python run_temporal.py > $L/temporal.log 2>&1 &
P2=$!
python run_final.py > $L/final.log 2>&1
python run_detection.py abl_mpnn_no_geo__MPNN abl_mpnn_no_peer_bb__MPNN abl_mpnn_no_int__MPNN abl_mpnn_no_victim__MPNN abl_mpnn_mean_agg__MPNN abl_mpnn_max_agg__MPNN abl_mpnn_no_posw__MPNN abl_mpnn_posw__MPNN abl_mpnn_no_logd__MPNN abl_mpnn_logd__MPNN > $L/tier2_mpnn.log 2>&1 &
P3=$!
python run_federated.py MPNN > $L/federated.log 2>&1 &
P4=$!
python run_counterfactual.py > $L/counterfactual.log 2>&1
python run_latency.py > $L/latency.log 2>&1
python analyze_detection.py tier1_detection PREVALENCE ORACLE victim_bb__MLPpub victim_all__HGB proto__MLPpub proto__LR proto__HGB proto__MLP mpnn__MPNN mpnn_raw__MPNN > $L/an_tier1.log 2>&1 &
for m in victim_bb__MLPpub victim_all__HGB proto__MLPpub proto__LR proto__HGB proto__MLP mpnn__MPNN mpnn_raw__MPNN ORACLE; do python analyze_final.py $m > $L/anf_$m.log 2>&1 & done
wait $P1 $P2 $P3
python analyze_attribution.py > $L/an_attr.log 2>&1
python analyze_per_channel.py > $L/an_per_channel.log 2>&1
python - > $L/an_tier2.log 2>&1 <<'PY'
from analyze_detection import analyse
from common import jdump, OUT
iso=['proto__HGB','iso_victim_bb__HGB','iso_victim_raw__HGB','iso_peer_bb__HGB','iso_peer_chan_mean__HGB','iso_peer_max_int__HGB','iso_peer_channels__HGB','iso_uniform_w__HGB','iso_random_w__HGB','iso_geometry_only__HGB','iso_permuted_peers__HGB','abl_hgb_no_cw__HGB','abl_hgb_cw__HGB','temporal_K1__HGB','temporal_K4__HGB','temporal_K8__HGB','temporal_K16__HGB']
jdump(analyse(iso, ref='temporal_K1__HGB', boot=False), OUT/'tier2_isolation_temporal.json')
abl=['mpnn__MPNN','abl_mpnn_no_geo__MPNN','abl_mpnn_no_peer_bb__MPNN','abl_mpnn_no_int__MPNN','abl_mpnn_no_victim__MPNN','abl_mpnn_mean_agg__MPNN','abl_mpnn_max_agg__MPNN','abl_mpnn_no_posw__MPNN','abl_mpnn_posw__MPNN','abl_mpnn_no_logd__MPNN','abl_mpnn_logd__MPNN','mpnn_raw__MPNN']
jdump(analyse(abl, ref='abl_mpnn_no_posw__MPNN', boot=False), OUT/'tier2_mpnn_ablation.json')
PY
wait
python - > $L/fed_summary.log 2>&1 <<'PY'
import json, pandas as pd
from common import OUT
d=json.load(open(OUT/'federated'/'federated_MPNN.json')); r=pd.DataFrame(d['runs'])
s=r.groupby(['attack','frac','agg'])[['roc_auc','pr_auc']].agg(['mean','std']); s.columns=['_'.join(c) for c in s.columns]
s.reset_index().to_csv(OUT/'federated'/'federated_summary.csv',index=False); print(s)
PY
echo "ALL V2 DONE $(date)" > $L/PIPELINE_DONE
