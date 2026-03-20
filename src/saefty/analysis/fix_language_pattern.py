#!/usr/bin/env python3
"""
Fix language_pattern in feature_labels.json using deterministic selectivity scores
instead of LLM judgments.

Usage:
    python -m saefty.analysis.fix_language_pattern
"""

import json
from pathlib import Path

FEATURE_LABELS_PATH = "results/feature_labels.json"
PER_LANG_PATH = "results/per_lang_combined.json"


def compute_language_pattern(feature_combined: dict) -> str:
    hi = feature_combined.get("selectivity_hindi", 0) or 0
    bn = feature_combined.get("selectivity_bengali", 0) or 0
    en = feature_combined.get("selectivity_english", 0) or 1
    low_resource_avg = (hi + bn) / 2
    ratio = low_resource_avg / en
    if ratio < 0.05:
        return "high_resource_only"
    elif ratio < 0.25:
        return "partial_cross_lingual"
    else:
        return "cross_lingual"


def main():
    with open(FEATURE_LABELS_PATH, encoding="utf-8") as f:
        labels = json.load(f)

    with open(PER_LANG_PATH, encoding="utf-8") as f:
        per_lang = json.load(f)

    lookup = {feat["feature_id"]: feat for feat in per_lang["features"]}

    # Count before
    before = {}
    for feat in labels["features"]:
        p = feat.get("language_pattern", "unknown")
        before[p] = before.get(p, 0) + 1

    # Fix
    for feat in labels["features"]:
        fid = feat["feature_id"]
        combined = lookup.get(fid, {})
        feat["language_pattern"] = compute_language_pattern(combined)

    # Count after
    after = {}
    for feat in labels["features"]:
        p = feat["language_pattern"]
        after[p] = after.get(p, 0) + 1

    # Save
    with open(FEATURE_LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)

    # Print
    print("language_pattern distribution:")
    print(f"  {'pattern':<25} {'before':>6} → {'after':>6}")
    print(f"  {'─' * 42}")
    all_keys = sorted(set(list(before.keys()) + list(after.keys())))
    for k in all_keys:
        print(f"  {k:<25} {before.get(k, 0):>6} → {after.get(k, 0):>6}")
    print(f"\nSaved to {FEATURE_LABELS_PATH}")


if __name__ == "__main__":
    main()
