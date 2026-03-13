#!/bin/bash
# Expansion factor ablation: Gated SAE at 4x, 8x (already done), 16x
# Baseline: gated_fixed (8x, L1=5e-4) — not re-run here

echo "============================================================"
echo "RUN 1/2: Gated SAE — 4x expansion (d_sae=8192)"
echo "============================================================"
python experiments/train_gated_sae.py \
    --expansion-factor 4 \
    --l1-coefficient 5e-4 \
    --output-dir results/train_sae/gated_4x

echo ""
echo "============================================================"
echo "RUN 2/2: Gated SAE — 16x expansion (d_sae=32768)"
echo "============================================================"
python experiments/train_gated_sae.py \
    --expansion-factor 16 \
    --l1-coefficient 5e-4 \
    --output-dir results/train_sae/gated_16x

echo ""
echo "============================================================"
echo "DONE — run analyze_comparison.py to see results"
echo "============================================================"
python experiments/analyze_comparison.py