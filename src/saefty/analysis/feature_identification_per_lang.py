#!/usr/bin/env python3
"""
Extract top 50 most selective features per language independently.

Reuses data loading, forward pass, and SAE encoding from feature_identification.py.
Caches intermediate selectivity arrays to results/selectivity_cache.npz to avoid
re-running the expensive forward pass on subsequent runs.

Usage:
    python src/saefty/analysis/feature_identification_per_lang.py
"""

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from saefty.analysis.feature_identification import (
    MODEL_NAME, CHECKPOINT_PATH, HOOK_LAYER, LANGUAGES, BATCH_SIZE,
    XSAFETY_DATA_DIR, DEVICE, LANG_MAP,
    load_xsafety_data, get_activations, load_sae, encode_with_sae,
)

# ── CONFIG ──────────────────────────────────────────────────────────────────
TOP_K = 50
CACHE_PATH = "results/selectivity_cache.npz"
OUTPUT_PER_LANG = "results/per_lang_top50.json"
OUTPUT_COMBINED = "results/per_lang_combined.json"


# ── COMPUTE OR LOAD SELECTIVITY ────────────────────────────────────────────

def compute_all_selectivity(cache_path: str) -> Dict[str, np.ndarray]:
    """
    Compute selectivity[lang] = mean_harmful - mean_benign for all 16,384 features.
    Caches result to disk. Returns dict of {lang: ndarray(16384,)}.
    """
    cache = Path(cache_path)

    if cache.exists():
        print(f"Loading cached selectivity from {cache_path}")
        data = np.load(cache_path)
        selectivity = {lang: data[lang] for lang in LANGUAGES if lang in data}
        if len(selectivity) == len(LANGUAGES):
            print(f"  {len(LANGUAGES)} languages loaded from cache")
            return selectivity
        print("  Cache incomplete, recomputing...")

    # ── full forward pass ──
    harmful, benign = load_xsafety_data(XSAFETY_DATA_DIR, LANGUAGES)

    print(f"Loading model: {MODEL_NAME} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    sae = load_sae(CHECKPOINT_PATH, DEVICE)

    selectivity = {}
    for lang in LANGUAGES:
        print(f"\n── {lang} ──")

        if lang in harmful and harmful[lang]:
            h_act = get_activations(
                harmful[lang], model, tokenizer, HOOK_LAYER, DEVICE, BATCH_SIZE,
                label=f"{lang}/harmful",
            )
            h_feat = encode_with_sae(h_act, sae)
            mean_h = h_feat.mean(dim=0).numpy()
        else:
            mean_h = np.zeros(sae.d_sae)

        if lang in benign and benign[lang]:
            b_act = get_activations(
                benign[lang], model, tokenizer, HOOK_LAYER, DEVICE, BATCH_SIZE,
                label=f"{lang}/benign",
            )
            b_feat = encode_with_sae(b_act, sae)
            mean_b = b_feat.mean(dim=0).numpy()
        else:
            mean_b = np.zeros_like(mean_h)

        selectivity[lang] = mean_h - mean_b

    # cache to disk
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, **selectivity)
    print(f"\nSelectivity cached: {cache_path}")

    return selectivity


# ── PER-LANGUAGE TOP-K ──────────────────────────────────────────────────────

def extract_per_lang_top_k(
    selectivity: Dict[str, np.ndarray], top_k: int
) -> Dict[str, List[Dict]]:
    """For each language, get top_k features by selectivity descending."""
    per_lang = {}
    for lang in LANGUAGES:
        sel = selectivity[lang]
        top_idx = np.argsort(sel)[::-1][:top_k]
        entries = []
        for rank, idx in enumerate(top_idx, 1):
            entries.append({
                "rank": rank,
                "feature_id": int(idx),
                "selectivity": round(float(sel[idx]), 6),
            })
        per_lang[lang] = entries
    return per_lang


def build_combined(
    selectivity: Dict[str, np.ndarray],
    per_lang: Dict[str, List[Dict]],
) -> Dict:
    """Build combined feature list with cross-language info."""
    # collect all unique feature ids
    all_ids = set()
    lang_sets = {}
    lang_ranks = {}
    for lang in LANGUAGES:
        ids = {e["feature_id"] for e in per_lang[lang]}
        lang_sets[lang] = ids
        lang_ranks[lang] = {e["feature_id"]: e["rank"] for e in per_lang[lang]}
        all_ids |= ids

    features = []
    for fid in sorted(all_ids):
        appears_in = [lang for lang in LANGUAGES if fid in lang_sets[lang]]
        entry = {
            "feature_id": fid,
            "appears_in": appears_in,
            "n_languages": len(appears_in),
        }
        for lang in LANGUAGES:
            entry[f"selectivity_{lang}"] = round(float(selectivity[lang][fid]), 6)
            entry[f"rank_{lang}"] = lang_ranks[lang].get(fid, None)
        features.append(entry)

    # sort by n_languages descending, then by selectivity_english descending
    features.sort(key=lambda x: (-x["n_languages"], -x["selectivity_english"]))

    # summary stats
    in_all_9 = sum(1 for f in features if f["n_languages"] == 9)
    in_7_plus = sum(1 for f in features if f["n_languages"] >= 7)
    hi_bn_ids = lang_sets.get("hindi", set()) | lang_sets.get("bengali", set())

    summary = {
        "total_unique_features": len(all_ids),
        "features_in_all_9_langs": in_all_9,
        "features_in_7_plus_langs": in_7_plus,
        "features_in_hi_or_bn_top50": len(hi_bn_ids),
    }

    return {"features": features, "summary": summary}


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # step 1: compute or load selectivity
    print("── Computing selectivity for all 16,384 features ──")
    selectivity = compute_all_selectivity(CACHE_PATH)

    # step 2: per-language top 50
    print("\n── Extracting per-language top 50 ──")
    per_lang = extract_per_lang_top_k(selectivity, TOP_K)

    # step 3: combined output
    combined = build_combined(selectivity, per_lang)

    # step 4: save
    Path(OUTPUT_PER_LANG).parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PER_LANG, "w") as f:
        json.dump({"per_language": per_lang}, f, indent=2)
    print(f"Saved: {OUTPUT_PER_LANG}")

    with open(OUTPUT_COMBINED, "w") as f:
        json.dump(combined, f, indent=2)
    print(f"Saved: {OUTPUT_COMBINED}")

    # ── stdout summary ──
    print("\n── per-language top feature ──")
    print(f"{'language':<25} | {'top feature id':>15} | {'selectivity':>12}")
    print("-" * 58)
    for lang in LANGUAGES:
        top = per_lang[lang][0]
        print(f"{lang:<25} | {top['feature_id']:>15} | {top['selectivity']:>12.3f}")

    summary = combined["summary"]
    print(f"\n── overlap analysis ──")
    print(f"features appearing in all 9 languages:  {summary['features_in_all_9_langs']}")
    print(f"features appearing in 7+ languages:     {summary['features_in_7_plus_langs']}")
    print(f"features in hi or bn top-50:            {summary['features_in_hi_or_bn_top50']}")
    print(f"unique features total:                  {summary['total_unique_features']}")

    # hindi top 10
    print(f"\n── hindi top 10 ──")
    print(f"{'rank':>4} | {'feature_id':>10} | {'sel_hi':>8} | {'sel_en':>8} | {'sel_ar':>8}")
    print("-" * 50)
    for entry in per_lang["hindi"][:10]:
        fid = entry["feature_id"]
        sel_hi = entry["selectivity"]
        sel_en = float(selectivity["english"][fid])
        sel_ar = float(selectivity["standard_arabic"][fid])
        print(f"{entry['rank']:>4} | {fid:>10} | {sel_hi:>8.3f} | {sel_en:>8.3f} | {sel_ar:>8.3f}")

    # bengali top 10
    print(f"\n── bengali top 10 ──")
    print(f"{'rank':>4} | {'feature_id':>10} | {'sel_bn':>8} | {'sel_en':>8} | {'sel_ar':>8}")
    print("-" * 50)
    for entry in per_lang["bengali"][:10]:
        fid = entry["feature_id"]
        sel_bn = entry["selectivity"]
        sel_en = float(selectivity["english"][fid])
        sel_ar = float(selectivity["standard_arabic"][fid])
        print(f"{entry['rank']:>4} | {fid:>10} | {sel_bn:>8.3f} | {sel_en:>8.3f} | {sel_ar:>8.3f}")

    # overlap check: are hindi top features the same as english?
    en_ids = {e["feature_id"] for e in per_lang["english"]}
    hi_ids = {e["feature_id"] for e in per_lang["hindi"]}
    bn_ids = {e["feature_id"] for e in per_lang["bengali"]}
    en_hi_overlap = len(en_ids & hi_ids)
    en_bn_overlap = len(en_ids & bn_ids)
    hi_bn_overlap = len(hi_ids & bn_ids)

    print(f"\n── hindi/english overlap ──")
    print(f"english ∩ hindi top-50:   {en_hi_overlap}/50 features shared")
    print(f"english ∩ bengali top-50: {en_bn_overlap}/50 features shared")
    print(f"hindi ∩ bengali top-50:   {hi_bn_overlap}/50 features shared")
    if en_hi_overlap < 25:
        print("→ Hindi top features are DIFFERENT from English — safety is encoded differently!")
    else:
        print("→ Hindi top features OVERLAP with English — shared safety representation.")

    elapsed = time.time() - t0
    print(f"\n── Done ── wall time: {elapsed:.0f}s ({elapsed/60:.1f}m)")


if __name__ == "__main__":
    main()
