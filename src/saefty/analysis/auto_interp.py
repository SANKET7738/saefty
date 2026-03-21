#!/usr/bin/env python3
"""
Method 2: Auto-Interpretation with Detection & Fuzzing Scoring

Takes the max-activating examples from Method 1 (any corpus tier) and:
  1. EXPLAIN: Shows an LLM the top activating contexts with highlighted
     peak tokens, asks it to identify the common pattern.
  2. SCORE (detection): LLM distinguishes activating from non-activating
     contexts given only the explanation.
  3. SCORE (fuzzing): LLM identifies the peak token given explanation + context.

Adaptations for multi-tier corpus:
  - When examples come from multiple corpus tiers (mixed mode), we show the
    explainer examples from ALL tiers, tagged by source. This lets the LLM
    see that a feature fires on the same concept in general text AND safety text.
  - When FLORES+ examples are available with sentence_ids, we include a
    cross-lingual alignment note: "this feature fires on sentence X in both
    english and hindi", which is powerful evidence for the LLM to identify
    the concept.

Inputs:
  results/max_act/{corpus}/top_activations.json  (from Method 1)
  results/per_lang_combined.json
  results/logit_lens.json                        (optional)

Outputs:
  results/auto_interp_v2/explanations.json
  results/auto_interp_v2/summary.json

Usage:
  python -m saefty.analysis.auto_interp --corpus mixed
  python -m saefty.analysis.auto_interp --corpus flores --explain-only
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# ── CONFIG ─────────────────────────────────────────────────────────────────
N_EXAMPLES_FOR_EXPLAIN = 12    # contexts shown to the explainer
N_EXAMPLES_FOR_DETECTION = 5   # held-out activating contexts for scoring
N_RANDOM_FOR_DETECTION = 20    # non-activating contexts for scoring
N_EXAMPLES_FOR_FUZZING = 5     # contexts for fuzzing scoring

ALL_LANGUAGES = [
    "english", "standard_arabic", "german", "spanish",
    "french", "hindi", "japanese", "bengali", "simplified_chinese",
]


# ── PROMPT TEMPLATES ───────────────────────────────────────────────────────
EXPLAIN_SYSTEM = """\
You are an expert in mechanistic interpretability of neural networks.
You are analyzing features learned by a Sparse Autoencoder (SAE) trained
on the residual stream of a multilingual language model (CohereForAI tiny-aya-global, layer 20).
Each feature potentially corresponds to a single interpretable concept.
Your job is to identify what concept each feature encodes by examining
which tokens activate it most strongly across multiple languages and corpus types."""

EXPLAIN_PROMPT = """\
Below are the top {n_examples} contexts that most strongly activate SAE feature #{feature_id}.

In each context, the peak activating token is marked with >>>TOKEN<<< and its activation is in brackets.
Other tokens with nonzero activation are shown as token[activation].
The source corpus and language are noted for each example.

{examples_block}
{alignment_note}
{logit_info}
Cross-language selectivity info:
  High selectivity in: {appears_in}
  Low selectivity in: {missing_in}

Based on ALL the patterns above, respond with ONLY valid JSON (no preamble, no markdown fences):
{{
    "explanation": "one clear sentence describing what concept this feature fires on",
    "short_label": "2-5 word label",
    "token_pattern": "what the peak activating tokens have in common",
    "context_pattern": "what the surrounding contexts have in common",
    "is_token_level": true or false,
    "is_cross_lingual": true or false,
    "safety_relevance": "how this relates to safety/harmful content, or 'none'",
    "corpus_consistency": "does the feature fire on the same concept across corpus types (flores/culturax/xsafety)?",
    "confidence": 0.0 to 1.0
}}"""


DETECTION_SYSTEM = """\
You are evaluating how well a feature explanation matches actual activations.
Given a feature explanation and text contexts, predict which ones activate the feature.
Respond with ONLY a JSON list of context indices."""

DETECTION_PROMPT = """\
Feature explanation: "{explanation}"

Below are {n_total} text contexts (0 to {n_max}). Some activate the feature, some do not.

{contexts_block}

Which contexts activate a feature described as "{explanation}"?
Respond with ONLY a JSON list of indices, e.g. [0, 3, 7]."""


FUZZING_SYSTEM = """\
Given a feature explanation and a text context, predict which token activates the feature most.
Respond with ONLY a JSON object: {{"token_index": <int>}}"""

FUZZING_PROMPT = """\
Feature: "{short_label}" — {explanation}

Tokens (numbered):
{numbered_tokens}

Which token index does this feature most likely activate on?
Respond with ONLY: {{"token_index": <int>}}"""


# ── helpers ────────────────────────────────────────────────────────────────
def format_example_for_explain(ex: dict, rank: int, tokenizer=None) -> str:
    ctx = ex["context_tokens"]
    acts = ex["context_activations"]
    peak = ex["peak_idx_in_context"]
    ctx_ids = ex.get("context_token_ids", [])
    parts = []
    for j, act in enumerate(acts):
        if tokenizer is not None and j < len(ctx_ids):
            tok = tokenizer.decode([ctx_ids[j]])
        else:
            tok = ctx[j] if j < len(ctx) else "?"
        if j == peak:
            parts.append(f">>>{tok}<<<[{act:.2f}]")
        elif act > 0.01:
            parts.append(f"{tok}[{act:.2f}]")
        else:
            parts.append(tok)
    src = ex.get("source", "?")
    lang = ex["language"]
    act_val = ex["activation"]
    sid = ex.get("sentence_id", "")
    sid_str = f" (sentence #{sid})" if sid else ""
    return (
        f"Example {rank} [{lang}, {src}{sid_str}, activation={act_val:.3f}]:\n"
        f"  {' '.join(parts)}"
    )


def build_alignment_note(examples: List[dict]) -> str:
    """
    If FLORES+ examples share sentence_ids across languages, build a note.
    This is the key cross-lingual evidence.
    """
    by_sid = {}
    for ex in examples:
        sid = ex.get("sentence_id")
        if sid:
            by_sid.setdefault(sid, []).append(ex)

    cross = []
    for sid, exs in by_sid.items():
        langs = list(set(e["language"] for e in exs))
        if len(langs) > 1:
            tokens = [f"'{e['token']}' ({e['language']})" for e in exs]
            cross.append(
                f"  Sentence #{sid} fires in {', '.join(langs)}: "
                f"tokens {', '.join(tokens)}"
            )

    if not cross:
        return ""
    return (
        "CROSS-LINGUAL ALIGNMENT (same source sentence, different languages):\n"
        + "\n".join(cross[:5])
    )


def parse_json_response(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
        m = re.search(r'\[[^\[\]]*\]', raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    return {"error": "parse_failed", "raw": raw[:200]}


# ── LLM client ────────────────────────────────────────────────────────────
from saefty.llm_utils.azure_client import AzureOpenAIClient, DEPLOYMENT_NAME


class LLMClient:
    """Thin wrapper around AzureOpenAIClient for convenience."""

    def __init__(self, model: str = None):
        self.model = model or DEPLOYMENT_NAME
        self.client = AzureOpenAIClient()

    def complete(self, system: str, user: str, max_tokens=800, temperature=0.0) -> str:
        try:
            r = self.client.prompt_llm(
                model=self.model,
                message_history=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens, temperature=temperature,
                retries=2, retry_delay=2,
            )
            return r.get("response", "")
        except Exception as e:
            err = str(e)
            if "content_filter" in err or "ResponsibleAI" in err:
                return '{"error": "content_filter"}'
            return f'{{"error": "{err[:100]}"}}'


# ── STEP 1: explain ───────────────────────────────────────────────────────
def explain_feature(
    llm: LLMClient,
    fid: int,
    examples: List[dict],
    appears_in: List[str],
    logit_info: str = "",
    tokenizer=None,
) -> dict:
    # select top N examples, but try to maximise language + source diversity
    selected = _diverse_select(examples, N_EXAMPLES_FOR_EXPLAIN)

    examples_block = "\n\n".join(
        format_example_for_explain(ex, i + 1, tokenizer)
        for i, ex in enumerate(selected)
    )
    alignment_note = build_alignment_note(selected)
    missing = [l for l in ALL_LANGUAGES if l not in appears_in]

    prompt = EXPLAIN_PROMPT.format(
        n_examples=len(selected),
        feature_id=fid,
        examples_block=examples_block,
        alignment_note=alignment_note,
        logit_info=logit_info,
        appears_in=", ".join(appears_in) if appears_in else "unknown",
        missing_in=", ".join(missing) if missing else "none",
    )
    raw = llm.complete(EXPLAIN_SYSTEM, prompt)
    result = parse_json_response(raw)
    result["feature_id"] = fid
    return result


def _diverse_select(examples: List[dict], n: int) -> List[dict]:
    """
    Select n examples maximising diversity of language and source,
    while still preferring higher activations.
    """
    if len(examples) <= n:
        return examples

    # bucket by (language, source)
    buckets: Dict[tuple, List[dict]] = {}
    for ex in examples:
        key = (ex.get("language", "?"), ex.get("source", "?"))
        buckets.setdefault(key, []).append(ex)

    selected = []
    # round-robin from buckets, taking highest activation first
    keys = sorted(buckets.keys())
    idx = {k: 0 for k in keys}

    while len(selected) < n:
        added = False
        for k in keys:
            if idx[k] < len(buckets[k]) and len(selected) < n:
                selected.append(buckets[k][idx[k]])
                idx[k] += 1
                added = True
        if not added:
            break

    # if still not enough, fill from top by activation
    if len(selected) < n:
        used = {id(e) for e in selected}
        for ex in examples:
            if id(ex) not in used and len(selected) < n:
                selected.append(ex)

    return selected


# ── STEP 2: detection scoring ─────────────────────────────────────────────
def score_detection(
    llm: LLMClient,
    explanation: str,
    all_examples: List[dict],
    other_features_examples: Dict[int, List[dict]],
    fid: int,
    tokenizer=None,
) -> dict:
    act_ex = all_examples[N_EXAMPLES_FOR_EXPLAIN:][:N_EXAMPLES_FOR_DETECTION]
    if len(act_ex) < 2:
        return {"detection_f1": None, "reason": "not enough held-out examples"}

    # non-activating from other features
    non_act = []
    other_fids = [f for f in other_features_examples if f != fid]
    random.shuffle(other_fids)
    for ofid in other_fids:
        if len(non_act) >= N_RANDOM_FOR_DETECTION:
            break
        oe = other_features_examples[ofid]
        if oe:
            non_act.append(random.choice(oe))

    if len(non_act) < 5:
        return {"detection_f1": None, "reason": "not enough negatives"}

    all_ctx = []
    gt = set()
    for ex in act_ex:
        gt.add(len(all_ctx))
        if tokenizer and ex.get("context_token_ids"):
            all_ctx.append(tokenizer.decode(ex["context_token_ids"]))
        else:
            all_ctx.append(" ".join(ex["context_tokens"]))
    for ex in non_act:
        if tokenizer and ex.get("context_token_ids"):
            all_ctx.append(tokenizer.decode(ex["context_token_ids"]))
        else:
            all_ctx.append(" ".join(ex["context_tokens"]))

    order = list(range(len(all_ctx)))
    random.shuffle(order)
    shuffled = [all_ctx[i] for i in order]
    shuffled_gt = {ni for ni, oi in enumerate(order) if oi in gt}

    block = "\n\n".join(f"Context {i}:\n  {c}" for i, c in enumerate(shuffled))
    prompt = DETECTION_PROMPT.format(
        explanation=explanation, n_total=len(shuffled),
        n_max=len(shuffled) - 1, contexts_block=block,
    )

    raw = llm.complete(DETECTION_SYSTEM, prompt)
    parsed = parse_json_response(raw)
    predicted = set(parsed) if isinstance(parsed, list) else set()

    tp = len(predicted & shuffled_gt)
    fp = len(predicted - shuffled_gt)
    fn = len(shuffled_gt - predicted)
    tn = len(set(range(len(shuffled))) - predicted - shuffled_gt)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    bal_acc = ((tp / max(tp + fn, 1)) + (tn / max(tn + fp, 1))) / 2

    return {
        "detection_precision": round(prec, 4),
        "detection_recall": round(rec, 4),
        "detection_f1": round(f1, 4),
        "detection_balanced_acc": round(bal_acc, 4),
    }


# ── STEP 3: fuzzing scoring ──────────────────────────────────────────────
def score_fuzzing(
    llm: LLMClient,
    explanation: str,
    short_label: str,
    examples: List[dict],
    tokenizer=None,
) -> dict:
    test = examples[N_EXAMPLES_FOR_EXPLAIN:][:N_EXAMPLES_FOR_FUZZING]
    if len(test) < 2:
        return {"fuzzing_exact": None, "reason": "not enough examples"}

    correct = close = total = 0
    for ex in test:
        ctx_ids = ex.get("context_token_ids", [])
        if tokenizer and ctx_ids:
            toks = [tokenizer.decode([tid]) for tid in ctx_ids]
        else:
            toks = ex["context_tokens"]
        numbered = "\n".join(f"  [{i}] {t}" for i, t in enumerate(toks))
        prompt = FUZZING_PROMPT.format(
            short_label=short_label, explanation=explanation,
            numbered_tokens=numbered,
        )
        raw = llm.complete(FUZZING_SYSTEM, prompt, max_tokens=100)
        parsed = parse_json_response(raw)
        if isinstance(parsed, dict) and "token_index" in parsed:
            pred = parsed["token_index"]
            true = ex["peak_idx_in_context"]
            if pred == true:
                correct += 1
            if abs(pred - true) <= 2:
                close += 1
            total += 1

    if total == 0:
        return {"fuzzing_exact": None, "reason": "no valid predictions"}
    return {
        "fuzzing_exact": round(correct / total, 4),
        "fuzzing_close": round(close / total, 4),
        "fuzzing_n": total,
    }


# ── main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="mixed",
                        choices=["flores", "culturax", "xsafety", "mixed"])
    parser.add_argument("--explain-only", action="store_true")
    parser.add_argument("--model", default=None,
                        help="Azure deployment name (default: DEPLOYMENT_NAME from .env)")
    parser.add_argument("--max-features", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()

    top_act_path = f"results/max_act/{args.corpus}/top_activations.json"
    per_lang_path = "results/per_lang_combined.json"
    logit_lens_path = "results/logit_lens.json"

    print(f"Loading {top_act_path} ...")
    with open(top_act_path, encoding="utf-8") as f:
        top_data = json.load(f)
    features_data = {
        feat["feature_id"]: feat["top_examples"]
        for feat in top_data["features"]
    }
    print(f"  {len(features_data)} features")

    print(f"Loading {per_lang_path} ...")
    with open(per_lang_path, encoding="utf-8") as f:
        per_lang = json.load(f)
    lang_lookup = {f["feature_id"]: f for f in per_lang["features"]}

    logit_lookup = {}
    if Path(logit_lens_path).exists():
        with open(logit_lens_path, encoding="utf-8") as f:
            ll = json.load(f)
        for feat in ll.get("features", []):
            toks = feat.get("top_positive_tokens", [])[:10]
            logit_lookup[feat["feature_id"]] = (
                f"Top positive logit tokens: {', '.join(toks)}"
            )
        print(f"  {len(logit_lookup)} features have logit data")

    llm = LLMClient(model=args.model)

    from transformers import AutoTokenizer
    from saefty.analysis.feature_identification import MODEL_NAME
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    fids = sorted(features_data.keys())
    if args.max_features:
        fids = fids[:args.max_features]

    out_dir = Path("results/auto_interp_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    expl_path = out_dir / "explanations.json"
    summary_path = out_dir / "summary.json"

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "corpus": args.corpus,
            "n_features": len(fids),
            "explain_only": args.explain_only,
        },
        "features": [],
    }
    errors = 0

    for i, fid in enumerate(fids):
        examples = features_data[fid]
        li = lang_lookup.get(fid, {})
        appears_in = li.get("appears_in", [])
        logit_info = logit_lookup.get(fid, "")

        print(f"[{i+1}/{len(fids)}] feat {fid} ...", end=" ")

        explanation = explain_feature(llm, fid, examples, appears_in, logit_info, tokenizer)
        entry = {
            "feature_id": fid,
            "appears_in": appears_in,
            "n_languages": li.get("n_languages", 0),
            **explanation,
        }

        if explanation.get("error"):
            errors += 1
            print(f"ERROR")
        else:
            label = explanation.get("short_label", "?")
            conf = explanation.get("confidence", "?")
            print(f"{label} (conf={conf})", end="")

            if not args.explain_only:
                expl_text = explanation.get("explanation", "")
                short = explanation.get("short_label", "unknown")

                det = score_detection(llm, expl_text, examples, features_data, fid, tokenizer)
                fuz = score_fuzzing(llm, expl_text, short, examples, tokenizer)
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

    # final save
    with open(expl_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {expl_path}")

    # ── summary ──
    valid = [f for f in output["features"] if not f.get("error")]
    summary = {
        "total": len(output["features"]),
        "successful": len(valid),
        "errors": errors,
        "corpus": args.corpus,
    }
    if valid:
        confs = [f["confidence"] for f in valid
                 if isinstance(f.get("confidence"), (int, float))]
        if confs:
            summary["mean_confidence"] = round(sum(confs) / len(confs), 3)

        if not args.explain_only:
            dets = [f["detection_f1"] for f in valid
                    if isinstance(f.get("detection_f1"), (int, float))]
            fuzs = [f["fuzzing_exact"] for f in valid
                    if isinstance(f.get("fuzzing_exact"), (int, float))]
            if dets:
                summary["mean_detection_f1"] = round(sum(dets) / len(dets), 3)
            if fuzs:
                summary["mean_fuzzing_exact"] = round(sum(fuzs) / len(fuzs), 3)

        xl = [f for f in valid if f.get("is_cross_lingual")]
        ls = [f for f in valid if f.get("is_cross_lingual") is False]
        sf = [f for f in valid
              if f.get("safety_relevance", "none") != "none"]

        summary["cross_lingual_features"] = len(xl)
        summary["language_specific_features"] = len(ls)
        summary["safety_relevant_features"] = len(sf)

        # corpus consistency breakdown
        consist = Counter(
            str(f.get("corpus_consistency", "?"))[:30] for f in valid
        )
        summary["corpus_consistency_counts"] = dict(consist.most_common(10))

        summary["top_labels"] = dict(
            Counter(f.get("short_label", "?") for f in valid).most_common(20)
        )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved: {summary_path}")

    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print(f"  {len(valid)} explained, {errors} errors, {elapsed:.0f}s")
    for k in ["mean_confidence", "mean_detection_f1", "mean_fuzzing_exact",
              "cross_lingual_features", "safety_relevant_features"]:
        if k in summary:
            print(f"  {k}: {summary[k]}")
    print(f"{'═' * 70}")

    # top results table
    print(f"\n  {'feat':>6} | {'label':<28} | {'conf':>4} | {'det':>5} | {'fuz':>5} | {'x-ling':>6} | {'safety'}")
    print(f"  {'─' * 95}")
    for f in sorted(valid, key=lambda x: -(x.get("confidence") or 0))[:30]:
        sf_str = str(f.get("safety_relevance", "none"))[:20]
        print(
            f"  {f['feature_id']:>6} | "
            f"{str(f.get('short_label', '?')):<28.28} | "
            f"{f.get('confidence', '?'):>4} | "
            f"{str(f.get('detection_f1', '—')):>5} | "
            f"{str(f.get('fuzzing_exact', '—')):>5} | "
            f"{str(f.get('is_cross_lingual', '?')):>6} | "
            f"{sf_str}"
        )


if __name__ == "__main__":
    main()