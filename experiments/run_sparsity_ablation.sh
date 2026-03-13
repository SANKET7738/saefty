#!/bin/bash
# Sparsity ablation: Gated SAE at L1=5e-4 (already done), 5e-3, 5e-2
# Baseline: gated_fixed (8x, L1=5e-4) — not re-run here

echo "============================================================"
echo "RUN 1/2: Gated SAE — L1=5e-3"
echo "============================================================"
python experiments/train_gated_sae.py \
    --l1-coefficient 5e-3 \
    --output-dir results/train_sae/gated_l1-5e-3-fixed

echo ""
echo "============================================================"
echo "RUN 2/2: Gated SAE — L1=5e-2"
echo "============================================================"
python experiments/train_gated_sae.py \
    --l1-coefficient 5e-2 \
    --output-dir results/train_sae/gated_l1-5e-2

echo ""
echo "============================================================"
echo "DONE — run analyze_comparison.py to see results"
echo "============================================================"
python experiments/analyze_comparison.py