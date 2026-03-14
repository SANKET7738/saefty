#!/bin/bash
# 50M token Gated SAE training across 9 languages.
# Run from repo root: bash experiments/run_50m_multilingual.sh
#
# Languages: English, Arabic, German, Spanish, French, Hindi, Japanese, Bengali, Chinese
# All 9 are in both aya_collection (training) and XSafety (eval).
#
# NOTE: Set HF_TOKEN in .env or environment for HuggingFace upload.
#       If not set, upload is skipped gracefully.

# load .env if present
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

LANGUAGES="english,standard_arabic,german,spanish,french,hindi,japanese,bengali,simplified_chinese"

echo "============================================================"
echo "Gated SAE — 50M tokens, 9 languages"
echo "============================================================"
echo "Languages: $LANGUAGES"
echo "HF upload: ${HF_REPO:-disabled (set --hf-repo or HF_REPO)}"
echo "============================================================"

python experiments/train_gated_sae.py \
    --training-tokens 50000000 \
    --lang "$LANGUAGES" \
    --buffer-size 524288 \
    --checkpoint-every 2000 \
    --output-dir results/train_sae/gated_50m_9lang \
    ${HF_REPO:+--hf-repo "$HF_REPO"} \
    ${HF_MODEL_NAME:+--hf-model-name "$HF_MODEL_NAME"} \
    "$@"

echo ""
echo "============================================================"
echo "DONE!"
echo "============================================================"
echo "Results: results/train_sae/gated_50m_9lang/"
echo "Plots:   results/train_sae/gated_50m_9lang/plots/"
echo "Weights: results/train_sae/gated_50m_9lang/checkpoints/"
