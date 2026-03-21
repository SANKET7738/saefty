#!/usr/bin/env python3
"""
Per-language max activating examples.

For each language, takes its top 50 features from per_lang_top50.json and
collects max-activating examples using ONLY text from that language
(FLORES+ and XSafety tiers only).

Outputs per language:
  results/max_act_per_lang/{language}/top_activations.json
  results/max_act_per_lang/{language}/activation_stats.json
  results/max_act_per_lang/{language}/dashboards/feat_{id}.txt

Usage:
  python -m saefty.analysis.max_activating_per_lang
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from saefty.analysis.feature_identification import (
    MODEL_NAME, CHECKPOINT_PATH, HOOK_LAYER, LANGUAGES, DEVICE, load_sae,
)
from saefty.analysis.max_activating_examples import (
    ActivatingExample,
    FeatureTracker,
    get_token_sae_activations,
    process_batches,
    stream_flores,
    stream_xsafety,
    format_dashboard,
    BATCH_SIZE, SEQ_LEN, CONTEXT_WINDOW, TOP_K,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────
PER_LANG_TOP50_PATH = "results/per_lang_top50.json"
OUTPUT_BASE = "results/max_act_per_lang"


def load_per_lang_features(path: str) -> Dict[str, List[int]]:
    """Load per-language top 50 feature IDs."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for lang, feats in data["per_language"].items():
        result[lang] = [f["feature_id"] for f in feats]

    return result


def main():
    t0 = time.time()

    # ── load feature lists ──
    per_lang_features = load_per_lang_features(PER_LANG_TOP50_PATH)
    print(f"Loaded features for {len(per_lang_features)} languages from {PER_LANG_TOP50_PATH}")
    for lang, fids in per_lang_features.items():
        print(f"  {lang}: {len(fids)} features")

    # ── load model + SAE once ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading tokenizer + model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
    ).to(DEVICE).eval()

    print(f"Loading SAE: {CHECKPOINT_PATH}")
    sae = load_sae(CHECKPOINT_PATH, DEVICE)

    # ── process each language ──
    for lang in LANGUAGES:
        if lang not in per_lang_features:
            print(f"\n{'═' * 60}")
            print(f"  {lang} — skipping (no features)")
            continue

        feature_ids = per_lang_features[lang]
        feat_arr = np.array(feature_ids)
        trackers = {fid: FeatureTracker(fid, TOP_K) for fid in feature_ids}

        print(f"\n{'═' * 60}")
        print(f"  {lang} — {len(feature_ids)} features")
        print(f"{'═' * 60}")

        lang_t0 = time.time()

        # ── FLORES+ ──
        print(f"  ── flores ──")
        flores_stream = stream_flores(lang, tokenizer, SEQ_LEN, BATCH_SIZE)
        flores_tokens = process_batches(
            flores_stream, "flores", lang, feature_ids, feat_arr,
            trackers, model, sae, tokenizer, CONTEXT_WINDOW,
        )
        print(f"  flores/{lang}: {flores_tokens:,} tokens")

        # ── XSafety ──
        print(f"  ── xsafety ──")
        xsafety_stream = stream_xsafety(lang, tokenizer, SEQ_LEN, BATCH_SIZE)
        xsafety_tokens = process_batches(
            xsafety_stream, "xsafety", lang, feature_ids, feat_arr,
            trackers, model, sae, tokenizer, CONTEXT_WINDOW,
        )
        print(f"  xsafety/{lang}: {xsafety_tokens:,} tokens")

        total_tokens = flores_tokens + xsafety_tokens
        lang_time = time.time() - lang_t0
        print(f"  {lang} total: {total_tokens:,} tokens in {lang_time:.0f}s")

        # ── save ──
        out_dir = Path(OUTPUT_BASE) / lang
        dash_dir = out_dir / "dashboards"
        out_dir.mkdir(parents=True, exist_ok=True)
        dash_dir.mkdir(parents=True, exist_ok=True)

        top_out = {
            "metadata": {
                "model": MODEL_NAME,
                "sae_checkpoint": CHECKPOINT_PATH,
                "hook_layer": HOOK_LAYER,
                "language": lang,
                "top_k": TOP_K,
                "context_window": CONTEXT_WINDOW,
                "tokens_processed": total_tokens,
                "time_seconds": round(lang_time, 1),
            },
            "features": [],
        }
        stats_out = {
            "metadata": {"model": MODEL_NAME, "language": lang},
            "features": [],
        }

        for fid in feature_ids:
            t = trackers[fid]
            examples = t.get_sorted_examples()
            stats = t.get_stats()
            top_out["features"].append({
                "feature_id": fid,
                "top_examples": examples,
            })
            stats_out["features"].append(stats)
            dashboard = format_dashboard(fid, examples, stats, tokenizer)
            (dash_dir / f"feat_{fid}.txt").write_text(dashboard, encoding="utf-8")

        top_path = out_dir / "top_activations.json"
        stats_path = out_dir / "activation_stats.json"

        with open(top_path, "w", encoding="utf-8") as f:
            json.dump(top_out, f, indent=2, ensure_ascii=False)
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats_out, f, indent=2, ensure_ascii=False)

        print(f"  Saved: {top_path}")

        # ── per-language summary ──
        print(f"\n  Top 5 features for {lang}:")
        sorted_feats = sorted(
            top_out["features"],
            key=lambda x: x["top_examples"][0]["activation"] if x["top_examples"] else 0,
            reverse=True,
        )
        for feat in sorted_feats[:5]:
            fid = feat["feature_id"]
            if feat["top_examples"]:
                ex = feat["top_examples"][0]
                tok = ex["token"]
                act = ex["activation"]
                src = ex.get("source", "?")
                print(f"    feat {fid:>5}: act={act:.1f} tok='{tok}' src={src}")
            else:
                print(f"    feat {fid:>5}: no examples")

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  Done — {len(per_lang_features)} languages, {elapsed:.0f}s total")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
