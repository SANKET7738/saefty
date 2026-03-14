#!/bin/bash
# Dry run: small-scale test of the full training pipeline.
# Validates: model loading, activation store balance, SAE training,
#            evaluation, plots, checkpoints, and (optionally) HF upload.
#
# Uses 100K tokens and 3 languages — runs in ~2-3 minutes on GPU.
# Run from repo root: bash experiments/run_dry_run.sh

# load .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

LANGUAGES="english,standard_arabic,hindi"
OUTPUT_DIR="results/train_sae/dry_run"

echo "============================================================"
echo "DRY RUN — Gated SAE (100K tokens, 3 languages)"
echo "============================================================"
echo "Languages: $LANGUAGES"
echo "Output:    $OUTPUT_DIR"
echo "============================================================"

python experiments/train_gated_sae.py \
    --training-tokens 100000 \
    --lang "$LANGUAGES" \
    --batch-size 4096 \
    --buffer-size 16384 \
    --checkpoint-every 10 \
    --output-dir "$OUTPUT_DIR" \
    --eval-texts 5 \
    "$@"

echo ""
echo "============================================================"
echo "VALIDATING OUTPUTS"
echo "============================================================"

PASS=true

check_file() {
    if [ -f "$1" ]; then
        echo "  ✓ $1"
    else
        echo "  ✗ MISSING: $1"
        PASS=false
    fi
}

check_file "$OUTPUT_DIR/config.json"
check_file "$OUTPUT_DIR/metrics.json"
check_file "$OUTPUT_DIR/training_log.json"
check_file "$OUTPUT_DIR/training_log.csv"
check_file "$OUTPUT_DIR/language_balance.json"
check_file "$OUTPUT_DIR/checkpoints/sae_final.pt"
check_file "$OUTPUT_DIR/plots/training_loss.png"
check_file "$OUTPUT_DIR/plots/l0_curve.png"
check_file "$OUTPUT_DIR/plots/dead_features.png"
check_file "$OUTPUT_DIR/plots/language_balance.png"

echo ""
if [ "$PASS" = true ]; then
    echo "ALL CHECKS PASSED — pipeline is working correctly."
    echo "You can now run the full 50M training:"
    echo "  bash experiments/run_50m_multilingual.sh"
else
    echo "SOME CHECKS FAILED — review output above."
fi

echo ""
echo "============================================================"
echo "Language balance:"
cat "$OUTPUT_DIR/language_balance.json" 2>/dev/null
echo ""
echo "============================================================"
