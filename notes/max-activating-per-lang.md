# Max-Activating Examples (Per-Language)

## Goal

For each language independently, find the text contexts that most strongly activate that language's top safety features. This tells us what each feature responds to within a single language — a feature's top examples in Hindi come only from Hindi text, not English or Japanese.

## Method

**Input**: `per_lang_top50.json` — top 50 features per language, ranked by selectivity (how much more a feature activates on harmful vs benign prompts in that language).

**Process**: For each of the 9 languages:
1. Load that language's 50 features
2. Stream that language's text through the model (tiny-aya-global, layer 20)
3. Encode through the Gated SAE (d_sae=16,384)
4. For each feature, keep the 30 highest-activation tokens with a 40-token context window around them
5. Record per-token activations, source corpus, language, and FLORES+ sentence IDs

**Text sources** (per language):

| Tier | Source | What it gives us |
|------|--------|-----------------|
| **FLORES+** | Parallel sentences (~2,000 per language) | Controlled, high-quality text. Same content across languages for alignment comparison |
| **XSafety** | Harmful + benign prompts (~2,600 + 200 per language) | Safety-relevant text. Features with high activations here are candidates for safety features |

**Output per language**: 
- `top_activations.json` — 50 features × 30 examples with full context, per-token activations
- `activation_stats.json` — firing rates, mean/std/max per feature
- `dashboards/` — 50 human-readable files showing each feature's top examples with highlighted peak tokens

### Parameters

| Parameter | Value |
|-----------|-------|
| Features per language | 50 |
| Top-k examples per feature | 30 |
| Context window | 40 tokens |
| Batch size | 8 |
| Hook layer | 20 |

Script: `src/saefty/analysis/max_activating_per_lang.py`

## Results

### Per-language summary

| Language | Tokens | Time | XSafety examples | FLORES+ examples | Mean firing rate* |
|----------|--------|------|------------------|------------------|-------------------|
| English | 131,840 | 45s | 804 | 696 | 0.160 |
| Arabic | 161,536 | 35s | 760 | 740 | 0.173 |
| German | 168,960 | 35s | 821 | 679 | 0.200 |
| Spanish | 157,952 | 35s | 782 | 718 | 0.171 |
| French | 181,504 | 38s | 1,024 | 476 | 0.199 |
| Hindi | 438,784 | 67s | 927 | 573 | 0.347 |
| Japanese | 142,336 | 34s | 810 | 690 | 0.179 |
| Bengali | 467,712 | 71s | 1,059 | 441 | 0.330 |
| Chinese | 130,048 | 32s | 546 | 954 | 0.190 |

_*Mean firing rate = average fraction of tokens that activate each feature, averaged across the 50 features for that language._

Each language produces 50 features × 30 examples = 1,500 total examples, plus 50 dashboards.

### Observations

**Token count varies 3.5x across languages** — Hindi (438K) and Bengali (467K) produce ~3.5x more tokens than Chinese (130K) or English (131K) for the same text. This is because Devanagari and Bangla scripts are tokenized into many more subword pieces per word.

**Firing rates are higher for Hindi/Bengali** — mean firing rate is 0.35 for Hindi and 0.33 for Bengali vs ~0.17-0.20 for other languages. This is partly because more tokens means more chances for features to fire, and partly because the features selected for Hindi/Bengali have lower selectivity thresholds (mean selectivity 0.29 for Hindi vs 3.72 for English).

**XSafety dominates for most languages** — French (1,024/476) and Bengali (1,059/441) have ~2x more XSafety examples than FLORES+. Chinese is the exception (546/954) — more FLORES+ examples than XSafety. This means Chinese features fire heavily on general text too, not just safety prompts. Chinese has the highest selectivity (5.02) but that doesn't mean its features are safety-specific — it means harmful Chinese text has very different surface statistics from benign Chinese text (enough to create high selectivity), while the features themselves respond to general patterns.

**Total runtime was ~6.5 minutes** across all 9 languages, fast enough to iterate.

### Selectivity gap

> **Selectivity** = mean feature activation on harmful prompts − mean feature activation on benign prompts, for a single language.
>
> A selectivity of 3.72 (English) means that on average, English's top-50 features activate 3.72 units higher on harmful text than on benign text. A selectivity of 0.29 (Hindi) means the difference is barely detectable.
>
> Selectivity measures how well the model's internal features separate harmful from benign content — it does not measure whether the model refuses harmful requests.

| Language | Mean selectivity | Max | Min |
|----------|-----------------|-----|-----|
| Chinese | 5.017 | 13.319 | 0.438 |
| Japanese | 3.982 | 10.447 | 0.360 |
| English | 3.719 | 9.662 | 0.334 |
| Spanish | 2.696 | 7.219 | 0.268 |
| Arabic | 2.612 | 6.585 | 0.243 |
| German | 2.257 | 5.681 | 0.246 |
| French | 1.648 | 4.967 | 0.205 |
| Hindi | 0.290 | 1.701 | 0.095 |
| Bengali | 0.227 | 1.728 | 0.068 |

Hindi and Bengali features have ~15x lower selectivity than English — the model barely distinguishes harmful from benign at the feature level in these languages. Chinese and Japanese have the highest selectivity, higher than English.

## Output

- `results/max_act_per_lang/{language}/top_activations.json` — 50 features × 30 examples each with full token context, activations, source tags
- `results/max_act_per_lang/{language}/activation_stats.json` — per-feature firing rates, mean/std/max activation
- `results/max_act_per_lang/{language}/dashboards/feat_{id}.txt` — human-readable dashboards with highlighted peak tokens and per-token activations

9 languages × (1 JSON + 1 stats + 50 dashboards) = 468 files total.
