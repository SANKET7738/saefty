# Auto-Interpretation

## Goal

Automatically label what each of the 118 candidate features encodes, then score those labels to measure how well they actually describe the feature's behavior.

## Method

Three-step pipeline per feature, all using GPT-5-Chat:

### Step 1 — Explain

Show the LLM 12 diverse max-activating contexts (selected to maximise language + corpus diversity) with highlighted peak tokens, per-token activations, cross-lingual FLORES+ alignment notes, and selectivity scores. Ask for a structured label:

- `explanation` — one sentence describing the concept
- `short_label` — 2-5 word label
- `token_pattern` — what peak tokens have in common
- `context_pattern` — what surrounding contexts have in common
- `is_cross_lingual` — true/false
- `safety_relevance` — how it relates to safety, or "none"
- `confidence` — 0.0 to 1.0

<details>
<summary>Sample explain prompt (feature 5158)</summary>

```
System: You are an expert in mechanistic interpretability of neural networks.
You are analyzing features learned by a Sparse Autoencoder (SAE) trained
on the residual stream of a multilingual language model (CohereForAI tiny-aya-global, layer 20).
Each feature potentially corresponds to a single interpretable concept.
Your job is to identify what concept each feature encodes by examining
which tokens activate it most strongly across multiple languages and corpus types.

User: Below are the top 12 contexts that most strongly activate SAE feature #5158.

In each context, the peak activating token is marked with >>>TOKEN<<< and its activation is in brackets.
Other tokens with nonzero activation are shown as token[activation].
The source corpus and language are noted for each example.

Example 1 [hindi, xsafety, activation=6.808]:
  à¥Ģ[1.32] à¤ľ[1.69] à¤¾ à¤¨à¤ķ[5.73] >>>à¤¾<<<[6.81] à¤°[3.25] à¥Ģ[3.36] à¤ķ[1.31] à¤¾[0.45]

Example 2 [french, flores, activation=6.308]:
  ,[2.54] croyances[3.92] ,[3.25] savo[2.90] >>>irs<<<[6.31] et[3.21] désirs[0.84] des[1.10] autres[1.42]

Example 3 [english, flores, activation=6.204]:
  us with a lot of >>>information<<<[6.20] and material to build simulation ...

...  (12 examples total, diverse across languages and corpus tiers)

Cross-language selectivity info:
  High selectivity in: hindi
  Low selectivity in: english, standard_arabic, german, spanish, french, japanese, bengali, simplified_chinese

Based on ALL the patterns above, respond with ONLY valid JSON:
{ "explanation": "...", "short_label": "...", ... }
```

</details>

<details>
<summary>Actual LLM response (feature 5158)</summary>

```json
{
  "explanation": "This feature activates on words that denote knowledge, information, data, or theories across multiple languages.",
  "short_label": "Knowledge / Information",
  "token_pattern": "Nouns meaning knowledge, information, data, theories, or secrets (e.g., knowledge, information, conocimiento, savoirs, ज्ञान, معلومات, তথ্য)",
  "context_pattern": "Discussions about acquiring, possessing, generating, or using knowledge or information in academic, scientific, or personal data contexts.",
  "is_token_level": true,
  "is_cross_lingual": true,
  "safety_relevance": "Often appears in contexts about personal or secret information, which can relate to privacy or data misuse.",
  "corpus_consistency": "Yes, it consistently fires on the concept of knowledge/information across flores, culturax, and xsafety corpora.",
  "confidence": 0.98
}
```

</details>

### Step 2 — Detection scoring

Hold out 5 activating contexts (not shown during explanation). Mix with 20 non-activating contexts from other features. Shuffle order. Ask the LLM to identify which contexts activate the feature using only the explanation. Compute precision, recall, F1, balanced accuracy.

This measures: **can the explanation distinguish activating from non-activating text?**

<details>
<summary>Sample detection prompt (feature 5158)</summary>

```
System: You are evaluating how well a feature explanation matches actual activations.
Given a feature explanation and text contexts, predict which ones activate the feature.
Respond with ONLY a JSON list of context indices.

User: Feature explanation: "This feature activates on words that denote knowledge,
information, data, or theories across multiple languages."

Below are 25 text contexts (0 to 24). Some activate the feature, some do not.

Context 0:
  ... (shuffled mix of 5 activating + 20 non-activating contexts) ...

Which contexts activate a feature described as "This feature activates on words
that denote knowledge, information, data, or theories across multiple languages."?
Respond with ONLY a JSON list of indices, e.g. [0, 3, 7].
```

</details>

<details>
<summary>Feature 5158 detection result</summary>

```
precision: 0.625  (some false positives — LLM guessed too broadly)
recall:    1.000  (found all 5 activating contexts)
F1:        0.769
```

</details>

### Step 3 — Fuzzing scoring

Hold out 5 activating contexts. Show the LLM the explanation + numbered tokens in each context. Ask it to predict which token is the peak. Score exact match and close match (within 2 positions).

This measures: **can the explanation predict the exact token position?**

<details>
<summary>Sample fuzzing prompt (feature 5158)</summary>

```
System: Given a feature explanation and a text context, predict which token
activates the feature most. Respond with ONLY a JSON object: {"token_index": <int>}

User: Feature: "Knowledge / Information" — This feature activates on words that
denote knowledge, information, data, or theories across multiple languages.

Tokens (numbered):
  [0] Ġus
  [1] Ġwith
  [2] Ġa
  [3] Ġlot
  [4] Ġof
  [5] Ġinformation    ← (ground truth peak)
  [6] Ġand
  [7] Ġmaterial
  ...

Which token index does this feature most likely activate on?
Respond with ONLY: {"token_index": <int>}
```

</details>

<details>
<summary>Feature 5158 fuzzing result</summary>

```
exact match: 0.2   (1/5 — hard to pinpoint exact subword in non-Latin scripts)
close match: 0.6   (3/5 — within 2 positions)
```

</details>

### Parameters

| Parameter | Value |
|-----------|-------|
| Model | GPT-5-Chat |
| Examples for explain | 12 (diverse selection) |
| Examples for detection | 5 activating + 20 non-activating |
| Examples for fuzzing | 5 |
| Temperature | 0 |
| API calls per feature | ~8 (1 explain + 1 detection + ~5 fuzzing + retries) |
| Total API calls | ~826 |

Script: `src/saefty/analysis/auto_interp.py`

## Results

### Overall

| Metric | Value |
|--------|-------|
| Features explained | 118 / 118 |
| Errors | 0 |
| Mean confidence | 0.872 |
| Mean detection F1 | 0.458 |
| Mean fuzzing exact | 0.553 |
| Cross-lingual features | 101 |
| Language-specific features | 17 |

### Detection F1 distribution

| F1 range | Features |
|----------|----------|
| 0.8 – 1.0 | 16 |
| 0.6 – 0.8 | 11 |
| 0.4 – 0.6 | 51 |
| 0.2 – 0.4 | 22 |
| 0.0 – 0.2 | 18 |

Mean detection F1 of 0.458 — most features land in the 0.4-0.6 range. This is typical for auto-interp on SAE features; many features encode subtle subword patterns that are hard for LLMs to distinguish from context alone.

### What most features actually encode

The dominant labels tell the story:

| Label | Count |
|-------|-------|
| Single-character tokens | 14 |
| Leading number digit | 14 |
| Numeric digit tokens | 8 |
| Word-final suffixes | 3 |
| Various subword patterns | ~79 |

Most of the 118 features are **subword/character-level patterns** — digits, single characters, suffixes. These have high selectivity because XSafety harmful prompts contain numbers, punctuation, and structure that differ from benign prompts. They are not safety features in any meaningful sense.

### Caveats

- **Detection scoring uses raw BPE contexts** — for non-Latin scripts (Hindi, Bengali, Arabic) the LLM sees garbled subword tokens, which likely depresses detection F1 for those languages. Fixed in the script but results were generated before the fix.
- **Safety relevance is LLM-judged** — the LLM decides what counts as "safety-relevant" based on the activating examples. Since most examples come from XSafety, there's a risk of circular reasoning: features that activate on XSafety content get labeled safety-relevant because the content is from a safety benchmark.
- **No causal validation yet** — high detection F1 means the label is descriptive, not that the feature causally controls safety behavior. Feature steering experiments are needed to make that claim.
- **Content filter** — Azure content filtering blocked some detection/fuzzing prompts for features with the most harmful activating contexts. These features got `det=0.0` on those specific scoring rounds.

## Output

- `results/auto_interp_v2/explanations.json` — full results for all 118 features (labels, scores, explanations)
- `results/auto_interp_v2/summary.json` — aggregate statistics
