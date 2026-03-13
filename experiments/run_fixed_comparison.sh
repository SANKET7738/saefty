#!/bin/bash
# Phase 1 re-run with fixed L1 loss and dead feature tracking.
# Run from repo root: nohup bash experiments/run_fixed_comparison.sh > fixed_comparison.log 2>&1 &

echo "============================================================"
echo "RUN 1/3: Vanilla SAE (L1=5e-4, FIXED L1 loss)"
echo "============================================================"
python experiments/train_vanilla_sae.py \
    --l1-coefficient 5e-4 \
    --output-dir results/train_sae/vanilla_fixed

echo ""
echo "============================================================"
echo "RUN 2/3: Gated SAE (L1=5e-4, FIXED L1 loss)"
echo "============================================================"
python experiments/train_gated_sae.py \
    --l1-coefficient 5e-4 \
    --output-dir results/train_sae/gated_fixed

echo ""
echo "============================================================"
echo "RUN 3/3: TopK SAE (k=64, auxk_alpha=1/16)"
echo "============================================================"
python experiments/train_topk_sae.py \
    --k 64 \
    --auxk-alpha 0.0625 \
    --output-dir results/train_sae/topk_k64_auxk

echo ""
echo "============================================================"
echo "COMPARISON ANALYSIS"
echo "============================================================"
python experiments/analyze_comparison.py

echo ""
echo "============================================================"
echo "ALL DONE!"
echo "============================================================"