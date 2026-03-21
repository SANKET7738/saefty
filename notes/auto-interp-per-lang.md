# Auto-Interpretation (Per-Language)

## Goal

For each language independently, automatically label what each of its top 50 features encodes, then score those labels. This is the per-language version of the auto-interpretation pipeline — each language's features are explained using only that language's activating examples, and scored against only that language's text.

## Method

Three-step pipeline per feature, using gpt-5.2-chat via Azure:

### Step 1 — Explain

Show gpt-5.2-chat the top 12 diverse max-activating contexts for a feature (from that language's `top_activations.json`). The system prompt tells the LLM it's analyzing features specifically for that language. The LLM returns:

- `explanation` — one sentence describing the concept
- `short_label` — 2-5 word label
- `token_pattern` / `context_pattern` — what peak tokens and contexts have in common
- `safety_relevance` — how it relates to safety, or "none"
- `confidence` — 0.0 to 1.0

### Step 2 — Detection scoring

Hold out 5 activating contexts not shown during explanation. Mix with 20 non-activating contexts from **other features in the same language** (not from other languages). Ask the LLM to identify which contexts activate the feature using only the explanation.

Metric: **detection F1** — can the explanation distinguish activating from non-activating text?

### Step 3 — Fuzzing scoring

Hold out 5 activating contexts. Show the LLM the explanation + numbered tokens. Ask it to predict which token is the peak activation.

Metric: **fuzzing exact match** — can the explanation predict the exact peak token?

### Key difference from cross-lingual auto-interp

In the cross-lingual version, all 9 languages' examples are mixed together. Here, each language is processed independently — Hindi features are explained using only Hindi text and scored against only Hindi negatives. This avoids high-resource languages dominating the examples.

### Parameters

| Parameter | Value |
|-----------|-------|
| Model | gpt-5.2-chat |
| Features per language | 50 |
| Languages | 9 |
| Examples for explain | 12 (diverse selection within language) |
| Detection negatives | 20 (from other features, same language) |
| Fuzzing examples | 5 |
| Parallel workers | 4 |
| Temperature | 0 |

Script: `src/saefty/analysis/auto_interp_per_lang.py`

## Sample Prompts and Responses

### Step 1 — Explain (Hindi, feature 5169)

<details>
<summary>Sample explain prompt</summary>

```
System: You are an expert in mechanistic interpretability of neural networks.
You are analyzing features learned by a Sparse Autoencoder (SAE) trained
on the residual stream of a multilingual language model (CohereForAI tiny-aya-global, layer 20).
Each feature potentially corresponds to a single interpretable concept.

You are currently analyzing features specifically for hindi.
All examples below are from hindi text only.

User: Below are the top 12 contexts that most strongly activate SAE feature #5169 in hindi text.

In each context, the peak activating token is marked with >>>TOKEN<<< and its activation is in brackets.
...

Example 1 [hindi, xsafety, activation=6.408]:
  ... >>>ैं<<<[6.41] ...  (within context about fraud/deception)

Example 2 [hindi, flores, activation=6.370]:
  ... >>>र<<<[6.37] ...  (within context about corruption)

...  (12 examples total, all Hindi)

This feature has selectivity score 0.119 in hindi (rank 38).

Based on ALL the patterns above, respond with ONLY valid JSON:
{ "explanation": "...", "short_label": "...", ... }
```

</details>

<details>
<summary>Actual LLM response (Hindi, feature 5169)</summary>

```json
{
  "explanation": "This feature primarily fires on specific Devanagari characters within words related to fraud, deception, and corruption (e.g., धोखा, झूठ, भ्रष्टाचार).",
  "short_label": "Deception word tokens",
  "token_pattern": "individual Devanagari characters (e.g., ख, ठ, ैं, ी) occurring inside words about fraud or lying",
  "context_pattern": "contexts discussing cheating, fraud, lying, corruption, or deceptive behavior",
  "is_token_level": true,
  "safety_relevance": "associated with discussions of fraud, deception, and corrupt behavior in harmful or unethical scenarios",
  "corpus_consistency": "mostly consistent across xsafety and flores, centering on deception/corruption themes",
  "confidence": 0.67
}
```

</details>

<details>
<summary>Detection result (Hindi, feature 5169)</summary>

```
precision: 0.833  (few false positives)
recall:    1.000  (found all 5 activating contexts)
F1:        0.909
```

</details>

### Step 2 — Detection (Arabic, feature 13845)

<details>
<summary>Sample detection prompt</summary>

```
System: You are evaluating how well a feature explanation matches actual activations.
Given a feature explanation and text contexts, predict which ones activate the feature.
Respond with ONLY a JSON list of context indices.

User: Feature explanation: "This feature activates on Arabic interrogative and
help-seeking question phrases, especially components like كيف, ماذا أفعل,
كيف يمكنني, and لماذا."

Below are 25 text contexts (0 to 24). Some activate the feature, some do not.

Context 0:
  ... (shuffled mix of 5 activating + 20 non-activating Arabic contexts) ...

Which contexts activate a feature described as "..."?
Respond with ONLY a JSON list of indices, e.g. [0, 3, 7].
```

</details>

<details>
<summary>Actual detection result (Arabic, feature 13845)</summary>

```
precision: 0.714  (some false positives — other question-like contexts)
recall:    1.000  (found all 5 activating contexts)
F1:        0.833
```

</details>

## Per-language summary

| Language | Valid | Errors | Safety features | Mean confidence | Mean det F1 | Mean fuz exact |
|----------|-------|--------|-----------------|-----------------|-------------|----------------|
| English | 49 | 1 | 3 | 0.704 | 0.291 | 0.090 |
| Arabic | 50 | 0 | 3 | 0.820 | 0.154 | 0.052 |
| German | 49 | 1 | 3 | 0.775 | 0.296 | 0.314 |
| Spanish | 50 | 0 | 4 | 0.808 | 0.284 | 0.164 |
| French | 50 | 0 | 4 | 0.780 | 0.317 | 0.134 |
| Hindi | 50 | 0 | **6** | 0.827 | **0.408** | **0.329** |
| Japanese | 50 | 0 | 1 | 0.890 | 0.283 | 0.249 |
| Bengali | 50 | 0 | 1 | 0.889 | 0.261 | 0.264 |
| Chinese | 50 | 0 | **0** | 0.815 | 0.176 | 0.264 |
| **Total** | **448** | **2** | **25** | | | |

448 features explained across 9 languages, 25 flagged as safety-relevant. 2 parse errors (EN feat 10222, DE feat 4349).

### What most features encode

The dominant labels across all 450 feature-language pairs:

| Label | Count |
|-------|-------|
| word-initial tokens | 85 |
| Japanese comma / punctuation | 38 |
| single-character tokens | 23 |
| short subword fragments | 22 |
| high-frequency tokens | 15 |
| common function words | 12 |
| ... | ... |

Same pattern as the cross-lingual results — the vast majority of features encode subword/character-level patterns, not safety concepts. Out of 450 feature-language pairs, only 25 (5.6%) were flagged as safety-relevant.

### Safety features across languages

9 unique features were flagged as safety-relevant in at least one language:

| Feature | Languages | Label pattern |
|---------|-----------|---------------|
| 6988 | EN, AR, DE, ES, FR (5) | First-person pronouns / intent |
| 13845 | EN, AR, DE, ES, FR (5) | Interrogative questions |
| 15161 | EN, AR, ES, FR (4) | Negative evaluation / copula |
| 5169 | DE, ES, FR, HI (4) | Deception / wrongdoing |
| 4436 | HI, JA, BN (3) | Desire / willingness / consent |
| 875 | HI (1) | Fraud/deception terms |
| 5158 | HI (1) | Knowledge/information terms |
| 12002 | HI (1) | Abstract adjectival morphology |
| 12892 | HI (1) | Emotional clause coordination |

Two clear groupings emerge:

**High-resource cluster (6988, 13845, 15161)** — appear in 4-5 European + Arabic languages but not in Hindi/Bengali/Japanese/Chinese. These are grammatical features (first-person pronouns, question syntax, copular constructions) that are overrepresented in safety prompts because harmful requests are phrased as first-person questions with evaluative language.

**Low-resource / non-European cluster (4436, 875, 5158, 12002, 12892)** — appear only in Hindi, Bengali, or Japanese. Feature 4436 (desire/consent) appears across HI/JA/BN. The Hindi-only features (875, 5158, 12002, 12892) encode language-specific patterns — Hindi morphology creates distinct subword features for fraud terms, knowledge terms, adjectival forms, and emotional discourse markers.

### Hindi has the richest safety landscape

Hindi stands out with 6 safety features and the highest mean detection F1 (0.408) and fuzzing accuracy (0.329). This may be because:
- Hindi's Devanagari morphology creates more distinctive subword patterns for safety concepts
- XSafety's Hindi prompts produce activation patterns that are more separable from benign text at the subword level
- Or: with lower selectivity (mean 0.29), the features that do get selected are more semantically meaningful (they had to clear a lower bar but the ones that did are genuinely safety-related)

### Chinese has zero safety features

Chinese had 0 out of 50 features flagged as safety-relevant despite having the highest selectivity (5.02). This connects to the observation that Chinese features fire heavily on FLORES+ (general text) — the high selectivity comes from surface-level statistical differences between harmful and benign Chinese text, not from features encoding safety concepts.

### Caveats

- **Safety relevance is LLM-judged** — gpt-5.2-chat decides what counts as safety-relevant based on activating examples. Since examples come from XSafety, there's a risk of circular reasoning.
- **Detection F1 is generally low** — mean 0.15-0.41 across languages. Many features encode subtle subword patterns that are hard for LLMs to distinguish from context alone.
- **Content filter** — Azure content filtering blocked some prompts (2 parse errors). Features with the most harmful activating contexts may have gotten lower scores due to filtered scoring rounds.
- **No causal validation** — these labels describe activation patterns, not causal effects. Feature steering is needed to confirm whether manipulating these features actually changes model safety behavior.

## Output

- `results/auto_interp_per_lang/{language}/explanations.json` — per-language feature explanations with detection and fuzzing scores
- `results/auto_interp_per_lang/summary.json` — combined per-language statistics
