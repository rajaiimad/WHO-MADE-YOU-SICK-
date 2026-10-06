#!/bin/bash
# Stage 2 (PREREGISTRATION_v3): new baselines, nested CV then lock-box, generator v2.
export XRP_GEN=v2 PYTHONWARNINGS=ignore OMP_NUM_THREADS=4
L=outputs_v2/logs
python run_detection.py rule__RULE edgefeat__HGB > $L/s2_cpu.log 2>&1 &
python run_detection.py gcn__GCN gcngeo__GCNGEO > $L/s2_gcn.log 2>&1 &
python run_detection.py gat__GAT > $L/s2_gat.log 2>&1 &
wait
python run_final.py rule__RULE edgefeat__HGB gcn__GCN gcngeo__GCNGEO gat__GAT > $L/s2_final.log 2>&1
python run_attribution_rules.py > $L/s2_attr_rules.log 2>&1
for m in rule__RULE edgefeat__HGB gcn__GCN gcngeo__GCNGEO gat__GAT; do python analyze_final.py $m > $L/s2_anf_$m.log 2>&1 & done
python analyze_detection.py stage2_detection ORACLE proto__MLPpub proto__LR mpnn__MPNN rule__RULE edgefeat__HGB gcn__GCN gcngeo__GCNGEO gat__GAT > $L/s2_an.log 2>&1
wait
echo "STAGE2 DONE $(date)" > $L/STAGE2_DONE
