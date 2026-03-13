#!/bin/bash
# Phase 1 fair comparison: 3 SAE architectures at matched sparsity.
# Run from repo root: bash experiments/run_fair_comparison.sh
set -e

echo "============================================================"
echo "RUN 1/3: Vanilla SAE (L1=5e-3)"
echo "============================================================"
python experiments/train_vanilla_sae.py \
    --l1-coefficient 5e-3 \
    --output-dir results/train_sae/vanilla_l1-5e-3

echo ""
echo "============================================================"
echo "RUN 2/3: TopK SAE (k=32, auxk_alpha=1/16)"
echo "============================================================"
python experiments/train_topk_sae.py \
    --auxk-alpha 0.0625 \
    --output-dir results/train_sae/topk_auxk

echo ""
echo "============================================================"
echo "RUN 3/3: Gated SAE (L1=5e-3)"
echo "============================================================"
python experiments/train_gated_sae.py \
    --l1-coefficient 5e-3 \
    --output-dir results/train_sae/gated_l1-5e-3

echo ""
echo "============================================================"
echo "COMPARISON ANALYSIS"
echo "============================================================"
python experiments/analyze_comparison.py

echo ""
echo "============================================================"
echo "ALL DONE!"
echo "============================================================"