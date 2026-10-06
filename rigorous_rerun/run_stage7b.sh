#!/bin/bash
# Stage 7b: raw-channel interaction HGB through the same nested CV, lock-box and stress corpora as every model.
export XRP_GEN=v2 PYTHONWARNINGS=ignore OMP_NUM_THREADS=6
L=outputs_v2/logs
python run_detection.py edgefeatraw__HGB > $L/s7b_cv.log 2>&1 || exit 1
python run_final.py edgefeatraw__HGB > $L/s7b_final.log 2>&1 || exit 1
python analyze_final.py edgefeatraw__HGB > $L/s7b_anf.log 2>&1 || exit 1
python analyze_detection.py stage7_detection ORACLE proto__MLPpub mpnn__MPNN rule__RULE edgefeat__HGB gat__GAT edgefeatraw__HGB > $L/s7b_an.log 2>&1 || exit 1
python stage7_summary.py > $L/s7b_summary.log 2>&1 || exit 1
echo "STAGE7B DONE $(date)" > $L/STAGE7B_DONE
