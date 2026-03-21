#!/usr/bin/env python3
"""
Per-language auto-interpretation.

For each language, loads that language's max-activating examples and runs
the explain + detection + fuzzing pipeline independently.

Inputs:
  results/max_act_per_lang/{language}/top_activations.json
  results/per_lang_top50.json

Outputs:
  results/auto_interp_per_lang/{language}/explanations.json
  results/auto_interp_per_lang/summary.json

Usage:
  python -m saefty.analysis.auto_interp_per_lang
  python -m saefty.analysis.auto_interp_per_lang --languages hindi english
  python -m saefty.analysis.auto_interp_per_lang --explain-only
"""

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from saefty.analysis.feature_identification import LANGUAGES

from saefty.analysis.auto_interp import (
    LLMClient,
    format_example_for_explain,
    build_alignment_note,
    parse_json_response,
    _diverse_select,
    score_detection,
    score_fuzzing,
    N_EXAMPLES_FOR_EXPLAIN,
    DETECTION_SYSTEM,
    DETECTION_PROMPT,
    FUZZING_SYSTEM,
    FUZZING_PROMPT,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────
PER_LANG_TOP50_PATH = "results/per_lang_top50.json"
MAX_ACT_BASE = "results/max_act_per_lang"
OUTPUT_BASE = "results/auto_interp_per_lang"

# ── per-language explain prompt ────────────────────────────────────────────
EXPLAIN_SYSTEM_PER_LANG = """\
You are an expert in mechanistic interpretability of neural networks.
You are analyzing features learned by a Sparse Autoencoder (SAE) trained
on the residual stream of a multilingual language model (CohereForAI tiny-aya-global, layer 20).
Each feature potentially corresponds to a single interpretable concept.

You are currently analyzing features specifically for {language}.
All examples below are from {language} text only."""

EXPLAIN_PROMPT_PER_LANG = """\
Below are the top {n_examples} contexts that most strongly activate SAE feature #{feature_id} in {language} text.

In each context, the peak activating token is marked with >>>TOKEN<<< and its activation is in brackets.
Other tokens with nonzero activation are shown as token[activation].
The source corpus is noted for each example.

{examples_block}
{alignment_note}

This feature has selectivity score {selectivity:.3f} in {language} (rank {rank}).

Based on ALL the patterns above, respond with ONLY valid JSON (no preamble, no markdown fences):
{{
    "explanation": "one clear sentence describing what concept this feature fires on in {language}",
    "short_label": "2-5 word label",
    "token_pattern": "what the peak activating tokens have in common",
    "context_pattern": "what the surrounding contexts have in common",
    "is_token_level": true or false,
    "safety_relevance": "how this relates to safety/harmful content, or 'none'",
    "corpus_consistency": "does the feature fire on the same concept across corpus types (flores/xsafety)?",
    "confidence": 0.0 to 1.0
}}"""


def explain_feature_per_lang(
    llm: LLMClient,
    fid: int,
    examples: List[dict],
    language: str,
    selectivity: float,
    rank: int,
    tokenizer=None,
) -> dict:
    selected = _diverse_select(examples, N_EXAMPLES_FOR_EXPLAIN)

    examples_block = "\n\n".join(
        format_example_for_explain(ex, i + 1, tokenizer)
        for i, ex in enumerate(selected)
    )
    alignment_note = build_alignment_note(selected)

    system = EXPLAIN_SYSTEM_PER_LANG.format(language=language)
    prompt = EXPLAIN_PROMPT_PER_LANG.format(
        n_examples=len(selected),
        feature_id=fid,
        language=language,
        examples_block=examples_block,
        alignment_note=alignment_note,
        selectivity=selectivity,
        rank=rank,
    )

    raw = llm.complete(system, prompt)
    result = parse_json_response(raw)
    result["feature_id"] = fid
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=None,
                        help="Languages to process (default: all 9)")
    parser.add_argument("--explain-only", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-features", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()
    languages = args.languages or list(LANGUAGES)

    # Load selectivity data
    with open(PER_LANG_TOP50_PATH, encoding="utf-8") as f:
        per_lang_data = json.load(f)
    sel_lookup = {}
    for lang, feats in per_lang_data["per_language"].items():
        for feat in feats:
            sel_lookup[(lang, feat["feature_id"])] = {
                "selectivity": feat["selectivity"],
                "rank": feat["rank"],
            }

    # Load tokenizer for readable context
    from transformers import AutoTokenizer
    from saefty.analysis.feature_identification import MODEL_NAME
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    llm = LLMClient(model=args.model)
    all_summaries = {}

    for lang in languages:
        top_act_path = Path(MAX_ACT_BASE) / lang / "top_activations.json"
        if not top_act_path.exists():
            print(f"\n{'═' * 60}")
            print(f"  {lang} — skipping (no data at {top_act_path})")
            continue

        print(f"\n{'═' * 60}")
        print(f"  {lang}")
        print(f"{'═' * 60}")

        with open(top_act_path, encoding="utf-8") as f:
            top_data = json.load(f)

        features_data = {
            feat["feature_id"]: feat["top_examples"]
            for feat in top_data["features"]
        }
        fids = sorted(features_data.keys())
        if args.max_features:
            fids = fids[:args.max_features]

        print(f"  {len(fids)} features to process")

        out_dir = Path(OUTPUT_BASE) / lang
        out_dir.mkdir(parents=True, exist_ok=True)
        expl_path = out_dir / "explanations.json"

        output = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "language": lang,
                "n_features": len(fids),
                "explain_only": args.explain_only,
            },
            "features": [],
        }
        errors = 0

        for i, fid in enumerate(fids):
            examples = features_data[fid]
            sel_info = sel_lookup.get((lang, fid), {})
            selectivity = sel_info.get("selectivity", 0.0)
            rank = sel_info.get("rank", 0)

            print(f"  [{i+1}/{len(fids)}] feat {fid} ...", end=" ")

            explanation = explain_feature_per_lang(
                llm, fid, examples, lang, selectivity, rank, tokenizer,
            )
            entry = {
                "feature_id": fid,
                "language": lang,
                "selectivity": selectivity,
                "rank": rank,
                **explanation,
            }

            if explanation.get("error"):
                errors += 1
                print("ERROR")
            else:
                label = explanation.get("short_label", "?")
                conf = explanation.get("confidence", "?")
                print(f"{label} (conf={conf})", end="")

                if not args.explain_only:
                    expl_text = explanation.get("explanation", "")
                    short = explanation.get("short_label", "unknown")

                    det = score_detection(
                        llm, expl_text, examples, features_data, fid, tokenizer,
                    )
                    fuz = score_fuzzing(
                        llm, expl_text, short, examples, tokenizer,
                    )
                    entry.update(det)
                    entry.update(fuz)
                    print(f"  det={det.get('detection_f1', '—')}  "
                          f"fuz={fuz.get('fuzzing_exact', '—')}", end="")

                print()

            output["features"].append(entry)

            if (i + 1) % 10 == 0:
                with open(expl_path, "w", encoding="utf-8") as f:
                    json.dump(output, f, indent=2, ensure_ascii=False)
                print(f"  [checkpoint: {i+1} features saved]")

            time.sleep(0.5)

        # Final save for this language
        with open(expl_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"  Saved: {expl_path}")

        # Per-language summary
        valid = [f for f in output["features"] if not f.get("error")]
        lang_summary = {
            "total": len(output["features"]),
            "successful": len(valid),
            "errors": errors,
        }
        if valid:
            confs = [f["confidence"] for f in valid
                     if isinstance(f.get("confidence"), (int, float))]
            if confs:
                lang_summary["mean_confidence"] = round(sum(confs) / len(confs), 3)

            if not args.explain_only:
                dets = [f["detection_f1"] for f in valid
                        if isinstance(f.get("detection_f1"), (int, float))]
                fuzs = [f["fuzzing_exact"] for f in valid
                        if isinstance(f.get("fuzzing_exact"), (int, float))]
                if dets:
                    lang_summary["mean_detection_f1"] = round(sum(dets) / len(dets), 3)
                if fuzs:
                    lang_summary["mean_fuzzing_exact"] = round(sum(fuzs) / len(fuzs), 3)

            safety = [f for f in valid
                       if str(f.get("safety_relevance", "none")) != "none"]
            lang_summary["safety_relevant"] = len(safety)

            lang_summary["top_labels"] = dict(
                Counter(f.get("short_label", "?") for f in valid).most_common(10)
            )

        all_summaries[lang] = lang_summary

        print(f"\n  {lang}: {lang_summary['successful']} explained, "
              f"{errors} errors, "
              f"conf={lang_summary.get('mean_confidence', '—')}")

    # Combined summary
    summary_path = Path(OUTPUT_BASE) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    combined = {
        "timestamp": datetime.now().isoformat(),
        "languages_processed": languages,
        "explain_only": args.explain_only,
        "per_language": all_summaries,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {summary_path}")

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  Done — {len(all_summaries)} languages, {elapsed:.0f}s")
    for lang, s in all_summaries.items():
        det_str = f"det={s.get('mean_detection_f1', '—')}" if not args.explain_only else ""
        print(f"  {lang:<20} {s['successful']:>3} ok  {s['errors']:>2} err  "
              f"conf={s.get('mean_confidence', '—')}  {det_str}  "
              f"safety={s.get('safety_relevant', '?')}")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
