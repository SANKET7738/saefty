# Max-Activating Examples

## Goal

Find the concrete text contexts that most strongly activate each of the 118 candidate safety features. This tells us *what* each feature actually responds to — is it a real safety concept or just noise?

## Method

For each feature, collect the top 30 highest-activation examples across three corpus tiers:

| Tier | Source | Purpose |
|------|--------|---------|
| **FLORES+** | Parallel sentences (same content in all 9 languages) | Cross-lingual alignment — does the feature fire on the same sentence in different languages? |
| **Wikipedia** | General encyclopedic text (streamed via wikimedia/wikipedia) | Benign baseline — what does the feature fire on in normal text? |
| **XSafety** | Harmful + benign prompt pairs (~2,600 harmful + 200 benign per language) | Safety signal — does it fire on harmful content? |

For each activating example we store:
- The peak token and its activation value
- A 40-token context window around the peak
- Per-token activations within the context
- Language, source corpus, FLORES+ sentence ID (if applicable)

### Parameters

| Parameter | Value |
|-----------|-------|
| Top-k examples per feature | 30 |
| Context window | 40 tokens |
| Tokens per language (Wikipedia) | 200,000 |
| Languages | 9 |
| Batch size | 8 |
| Hook layer | 20 |

Script: `src/saefty/analysis/max_activating_examples.py`

## Results

### Corpus breakdown

| Source | Examples | % |
|--------|----------|---|
| Wikipedia | 1,850 | 52% |
| XSafety | 856 | 24% |
| FLORES+ | 834 | 24% |
| **Total** | **3,540** | |

118 features × 30 examples = 3,540 total. Wikipedia dominates because it has the most tokens (200k per language), but XSafety provides the safety-relevant activations.

### Timing

| Language | Time |
|----------|------|
| English | 111s |
| Arabic | 102s |
| German | 103s |
| Spanish | 101s |
| French | 107s |
| Hindi | 166s |
| Bengali | 173s |
| Japanese | 97s |
| Chinese | 94s |
| **Total** | **~17 min** |

Hindi and Bengali take ~1.6x longer because their tokenizers produce more tokens per sentence (higher byte-to-token ratio for Devanagari/Bangla scripts).

### Firing rates

| Stat | Value |
|------|-------|
| Mean firing rate | 20.4% |
| Min firing rate | 0.4% |
| Max firing rate | 99.5% |

Most features are reasonably sparse. A few high-firing-rate features (~99%) are probably general-purpose (e.g., punctuation, common function words) rather than safety-specific.

### Cross-lingual alignment

9 out of 118 features show cross-lingual alignment on FLORES+ — the same parallel sentence activates the feature in multiple languages. This is strong evidence that these features encode a language-independent concept rather than language-specific tokens.

### Example — Feature 5169

Top 5 activating contexts all come from XSafety harmful prompts across 5 languages (French, Chinese, English, Spanish). The contexts share a theme of deception and fraud (falsifying data, deceiving voters, false statements).

| # | Language | Source | Activation | Peak token | Context |
|---|----------|--------|------------|------------|---------|
| 1 | French | xsafety | 10.5 | `-` | "Pensez-vous que..." (about discovering fraud) |
| 2 | Chinese | xsafety | 9.6 | `，` | 经常虚报经济数据 (falsifying economic data) |
| 3 | English | xsafety | 9.5 | `and` | "false statements, deceive voters and..." |
| 4 | Spanish | xsafety | 9.1 | `,` | "sobre los datos económicos" (economic data fraud) |
| 5 | French | xsafety | 9.0 | `et` | "trompez les électeurs et..." (deceive voters and...) |

However, the peak tokens themselves are function words (commas, conjunctions, hyphens), not content words. This means either:

1. the feature encodes a **contextual pattern** associated with deception-related discourse, or
2. the feature encodes a **general linguistic pattern** (e.g., list-like sentence structure) that happens to be overrepresented in XSafety harmful prompts compared to Wikipedia/FLORES+

The auto-interp detection score (F1=0.91) suggests the pattern is distinctive, but causal validation (feature steering) would be needed to confirm genuine safety relevance.

## Dashboards

Each feature gets a human-readable dashboard at `results/max_act/mixed/dashboards/feat_{id}.txt` showing all 30 examples with highlighted peak tokens and per-token activations.

## Output

- `results/max_act/mixed/top_activations.json` — all 118 features × 30 examples with full context
- `results/max_act/mixed/activation_stats.json` — per-feature firing rates, mean/std/max activation
- `results/max_act/mixed/dashboards/` — 118 human-readable dashboard files

## Takeaways

- the three-tier corpus design works well — Wikipedia provides baseline, XSafety provides signal, FLORES+ provides cross-lingual evidence
- features that activate primarily on XSafety (not Wikipedia) are likely safety-relevant, features that activate on both are probably general linguistic patterns
- 9 features with cross-lingual FLORES+ alignment are the strongest candidates for language-independent safety concepts
- the dashboards are essential for manual verification — you can't trust selectivity scores alone, you need to see what the feature actually fires on
