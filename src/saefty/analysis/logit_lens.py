#!/usr/bin/env python3
"""
Logit lens feature interpretation: project SAE decoder weights through the
model's token embedding matrix to identify which tokens each safety feature
promotes or suppresses.

Usage:
    python -m saefty.analysis.logit_lens
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── load .env if present (so HF_TOKEN is available) ────────────────────────
_env_path = Path(__file__).resolve().parents[3] / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from saefty.analysis.feature_identification import (
    CHECKPOINT_PATH,
    MODEL_NAME,
    load_sae,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
TOP_K_TOKENS = 20
PER_LANG_PATH = "results/per_lang_combined.json"
OUTPUT_PATH = "results/logit_lens.json"
DEVICE = "cpu"


def load_feature_ids(path: str) -> list[int]:
    """Extract all unique feature IDs from per_lang_combined.json."""
    with open(path) as f:
        data = json.load(f)
    ids = sorted(set(feat["feature_id"] for feat in data["features"]))
    print(f"Loaded {len(ids)} unique feature IDs from {path}")
    return ids


def load_embedding_matrix(model_name: str) -> torch.Tensor:
    """Load model, extract embedding matrix, free model memory."""
    print(f"Loading model: {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
    embed = model.model.embed_tokens.weight.detach().clone()
    del model
    print(f"Embedding matrix shape: {embed.shape}")  # (vocab_size, d_model)
    return embed


def load_decoder_weights(checkpoint_path: str) -> torch.Tensor:
    """Load SAE and extract decoder weight matrix."""
    sae = load_sae(checkpoint_path, DEVICE)
    W_dec = sae.W_dec.data.detach().clone()  # (d_sae, d_model)
    print(f"Decoder weight shape:   {W_dec.shape}")
    del sae
    return W_dec


def compute_logit_lens(
    feature_ids: list[int],
    W_dec: torch.Tensor,
    embed: torch.Tensor,
    tokenizer,
    top_k: int,
) -> list[dict]:
    """Compute top promoted/suppressed tokens for each feature."""
    special_ids = set(tokenizer.all_special_ids)
    results = []

    for feat_id in feature_ids:
        feature_vec = W_dec[feat_id]  # (d_model,)
        logits = embed @ feature_vec  # (vocab_size,)

        top_pos_idx = logits.argsort()[-top_k:].flip(0)
        top_neg_idx = logits.argsort()[:top_k]

        pos_tokens_raw = tokenizer.convert_ids_to_tokens(top_pos_idx.tolist())
        neg_tokens_raw = tokenizer.convert_ids_to_tokens(top_neg_idx.tolist())

        pos_tokens = [
            f"[special]{t}" if top_pos_idx[i].item() in special_ids else t
            for i, t in enumerate(pos_tokens_raw)
        ]
        neg_tokens = [
            f"[special]{t}" if top_neg_idx[i].item() in special_ids else t
            for i, t in enumerate(neg_tokens_raw)
        ]

        pos_scores = [round(logits[idx].item(), 3) for idx in top_pos_idx]
        neg_scores = [round(logits[idx].item(), 3) for idx in top_neg_idx]

        results.append({
            "feature_id": feat_id,
            "top_positive_tokens": pos_tokens,
            "top_positive_scores": pos_scores,
            "top_negative_tokens": neg_tokens,
            "top_negative_scores": neg_scores,
        })

    return results


def print_summary(features: list[dict], n_top: int = 5) -> None:
    """Print one-line summary per feature, sorted by feature_id."""
    for feat in sorted(features, key=lambda f: f["feature_id"]):
        pos = " · ".join(feat["top_positive_tokens"][:n_top])
        neg = " · ".join(feat["top_negative_tokens"][:n_top])
        print(f"feat {feat['feature_id']:>5} | promotes: {pos} | suppresses: {neg}")


def main():
    # Step 1 — load feature IDs
    feature_ids = load_feature_ids(PER_LANG_PATH)

    # Step 2 — load weights (CPU only, no forward pass)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    with torch.no_grad():
        embed = load_embedding_matrix(MODEL_NAME)
        W_dec = load_decoder_weights(CHECKPOINT_PATH)

    print(f"\nEmbed:  {embed.shape}  (vocab_size, d_model)")
    print(f"W_dec:  {W_dec.shape}  (d_sae, d_model)")

    # Step 3 — compute logit lens
    print(f"\nComputing logit lens for {len(feature_ids)} features (top {TOP_K_TOKENS} tokens) ...")
    with torch.no_grad():
        features = compute_logit_lens(feature_ids, W_dec, embed, tokenizer, TOP_K_TOKENS)

    # Step 4 — save output
    output = {
        "metadata": {
            "model": MODEL_NAME,
            "checkpoint": CHECKPOINT_PATH,
            "n_features": len(features),
            "top_k_tokens": TOP_K_TOKENS,
        },
        "features": features,
    }

    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")

    # Step 5 — human-readable summary
    print(f"\n{'─' * 100}")
    print_summary(features)
    print(f"{'─' * 100}")
    print(f"\nDone — {len(features)} features analyzed")


if __name__ == "__main__":
    main()
