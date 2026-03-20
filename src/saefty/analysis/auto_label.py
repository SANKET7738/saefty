#!/usr/bin/env python3
"""
Auto-label SAE safety features by feeding top activating prompts to an LLM.

Usage:
    python -m saefty.analysis.auto_label
"""

import json
import time
from datetime import datetime
from pathlib import Path

from saefty.llm_utils.azure_client import AzureOpenAIClient, DEPLOYMENT_NAME

# ── CONFIG ──────────────────────────────────────────────────────────────────
TOP_ACTIVATIONS_PATH = "results/interpretability/top_activations.json"
PER_LANG_PATH = "results/per_lang_combined.json"
OUTPUT_PATH = "results/feature_labels.json"
MODEL = DEPLOYMENT_NAME
MODEL_DISPLAY = "gpt-5.2"  # for metadata only, no endpoint info
TOP_N_PROMPTS = 10
ALL_LANGUAGES = [
    "english", "standard_arabic", "german", "spanish",
    "french", "hindi", "japanese", "bengali", "simplified_chinese",
]

SYSTEM_PROMPT = (
    "You are analyzing features of a sparse autoencoder trained on a "
    "multilingual language model to study safety representations. "
    "Respond only with valid JSON, no preamble."
)


def build_label_prompt(feature_id: int, prompts: list[dict], appears_in: list[str]) -> str:
    """Build the labeling prompt for a single feature."""
    missing = [lang for lang in ALL_LANGUAGES if lang not in appears_in]

    prompt_lines = []
    for p in prompts[:TOP_N_PROMPTS]:
        lang = p["language"]
        act = p["activation"]
        text = p["text"].strip()
        prompt_lines.append(f"  [{lang}, act={act}] {text}")

    prompts_block = "\n".join(prompt_lines)
    appears_str = ", ".join(appears_in) if appears_in else "none"
    missing_str = ", ".join(missing) if missing else "none"

    return f"""Below are the top {len(prompts[:TOP_N_PROMPTS])} prompts that most strongly activate feature #{feature_id}.
Each prompt is shown with its language and activation strength.

{prompts_block}

This feature has high selectivity scores in these languages: {appears_str}
And near-zero selectivity in: {missing_str}

Based on the prompts above, answer in JSON:
{{
  "label": "one short phrase describing the concept (max 6 words)",
  "category": one of ["harmful_request", "social_harm", "pii_extraction", "harassment", "cross_lingual_safety", "prompt_injection", "noise", "other"],
  "language_pattern": one of ["high_resource_only", "cross_lingual", "language_specific", "unclear"],
  "confidence": 0.0 to 1.0,
  "reasoning": "one sentence explaining your label"
}}"""


def label_feature(client: AzureOpenAIClient, feature_id: int, prompts: list[dict], appears_in: list[str]) -> dict:
    """Call LLM to label a single feature. Returns parsed label dict."""
    prompt = build_label_prompt(feature_id, prompts, appears_in)

    try:
        result = client.prompt_llm(
            model=MODEL,
            message_history=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=500,
            temperature=0,
            retries=3,
            retry_delay=2,
        )

        raw = result.get("response", "")
        # Try to parse JSON from response
        raw = raw.strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1]).strip()

        label = json.loads(raw)
        return label

    except Exception as e:
        print(f"  ERROR labeling feature {feature_id}: {e}")
        return {
            "label": "api_error",
            "category": "other",
            "language_pattern": "unclear",
            "confidence": 0.0,
            "reasoning": f"API call failed: {str(e)[:100]}",
        }


def save_results(output: dict, path: str):
    """Write results to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def main():
    t0 = time.time()

    # ── load data ──
    with open(TOP_ACTIVATIONS_PATH, encoding="utf-8") as f:
        top_act_data = json.load(f)
    features_list = top_act_data["features"]
    print(f"Loaded {len(features_list)} features from {TOP_ACTIVATIONS_PATH}")

    with open(PER_LANG_PATH, encoding="utf-8") as f:
        per_lang_data = json.load(f)
    lang_lookup = {
        feat["feature_id"]: feat
        for feat in per_lang_data["features"]
    }
    print(f"Loaded {len(lang_lookup)} features from {PER_LANG_PATH}")

    # ── init client ──
    client = AzureOpenAIClient()
    print(f"Using model: {MODEL}\n")

    # ── label each feature ──
    output = {
        "metadata": {
            "model": MODEL_DISPLAY,
            "n_features": len(features_list),
            "timestamp": datetime.now().isoformat(),
        },
        "features": [],
    }

    errors = 0
    for i, feat in enumerate(features_list):
        feature_id = feat["feature_id"]
        rank = feat.get("rank", i + 1)
        prompts = feat["top_prompts"]

        lang_info = lang_lookup.get(feature_id, {})
        appears_in = lang_info.get("appears_in", [])

        print(f"[{i+1}/{len(features_list)}] Feature {feature_id} (rank {rank}) ...", end=" ")

        label = label_feature(client, feature_id, prompts, appears_in)

        if label.get("label") == "api_error":
            errors += 1

        entry = {
            "feature_id": feature_id,
            "rank": rank,
            "selectivity_english": feat.get("selectivity_english"),
            "appears_in_n_languages": lang_info.get("n_languages"),
            **label,
        }
        output["features"].append(entry)

        print(f"{label.get('label', '?')} ({label.get('category', '?')}, conf={label.get('confidence', '?')})")

        # Incremental save every 10 features
        if (i + 1) % 10 == 0:
            save_results(output, OUTPUT_PATH)
            print(f"  [saved {i+1} features to {OUTPUT_PATH}]")

        # Rate limit
        time.sleep(1)

    # ── final save ──
    save_results(output, OUTPUT_PATH)
    elapsed = time.time() - t0

    # ── summary table ──
    print(f"\n{'─' * 100}")
    print(f"{'rank':>4} | {'feat_id':>8} | {'label':<35} | {'category':<22} | {'lang_pattern':<18} | {'conf':>4}")
    print(f"{'─' * 100}")
    for entry in sorted(output["features"], key=lambda x: x.get("rank", 999)):
        print(
            f"{entry.get('rank', '?'):>4} | "
            f"{entry['feature_id']:>8} | "
            f"{entry.get('label', '?'):<35.35} | "
            f"{entry.get('category', '?'):<22.22} | "
            f"{entry.get('language_pattern', '?'):<18.18} | "
            f"{entry.get('confidence', '?'):>4}"
        )
    print(f"{'─' * 100}")

    print(f"\nDone — {len(output['features'])} features labeled, {errors} errors, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
