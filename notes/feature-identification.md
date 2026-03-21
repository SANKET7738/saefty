# Feature Identification

## Goal

Identify which SAE features are safety-relevant by measuring how much more each feature activates on harmful prompts vs benign prompts across 9 languages.

## Method

**Selectivity scoring** — for each of the 16,384 SAE features, compute:

```
selectivity = mean_activation(harmful) − mean_activation(benign)
```

per language, using the XSafety benchmark. Features with high selectivity activate strongly on harmful content and weakly on benign content.

### Data 

- **Harmful prompts**: XSafety benchmark — ~2,600 per language covering 14 harm categories
- **Benign prompts**: XSafety commonsense split — 200 per language
- **Languages**: english, arabic, german, spanish, french, hindi, japanese, bengali, chinese

### Pipeline

1. Run each prompt through tiny-aya-global, hook layer 20 activations
2. Encode activations through the trained Gated SAE → feature activations (16,384-dim)
3. Mean-pool feature activations across all harmful prompts, all benign prompts, per language
4. Rank features by `selectivity_english`, take top 50 per language
5. Combine across languages — a feature is a candidate if it appears in top 50 for **any** language

Script: `src/saefty/analysis/feature_identification.py`

## Results

### Overall numbers

| Metric | Value |
|--------|-------|
| Total SAE features | 16,384 |
| Candidate safety features | **118** |
| Top 50 per language | 50 × 9 = 450 (with overlap → 118 unique) |

### Top 10 features by english selectivity

| rank | feat_id | sel_en | sel_ar | sel_hi | sel_bn | sel_ja | sel_zh | n_lang |
|------|---------|--------|--------|--------|--------|--------|--------|--------|
| 1 | 15726 | 9.66 | 6.56 | 0.15 | 0.15 | 10.45 | 13.28 | 9 |
| 2 | 8241 | 9.48 | 6.58 | 0.17 | 0.20 | 10.05 | 12.07 | 9 |
| 3 | 3391 | 9.37 | 6.38 | 0.10 | 0.07 | 9.98 | 13.32 | 9 |
| 4 | 8747 | 9.34 | 6.21 | 0.10 | 0.09 | 9.91 | 13.06 | 9 |
| 5 | 883 | 8.61 | 5.75 | 0.10 | 0.08 | 9.22 | 12.11 | 9 |
| 6 | 15621 | 8.36 | 5.57 | 0.12 | 0.12 | 8.94 | 11.72 | 9 |
| 7 | 3965 | 8.17 | 5.47 | 0.08 | 0.06 | 8.81 | 11.67 | 7 |
| 8 | 9840 | 7.93 | 5.31 | 0.08 | 0.06 | 8.59 | 11.39 | 7 |
| 9 | 2603 | 7.75 | 5.18 | 0.10 | 0.09 | 8.52 | 10.91 | 9 |
| 10 | 15779 | 7.75 | 5.27 | 0.09 | 0.07 | 8.45 | 11.01 | 8 |

### Distribution by language count

| Appears in N languages | Features |
|------------------------|----------|
| 9 (all) | 15 |
| 7-8 | 23 |
| 5-6 | 11 |
| 2-4 | 17 |
| 1 (single language) | 52 |

### Cross-lingual gap

The most striking pattern — look at feature 15726 (rank 1):
- **High selectivity**: en=9.66, ar=6.56, ja=10.45, zh=13.28
- **Near-zero**: hi=0.15, bn=0.15

This is a ~60x gap between high-resource (en/ja/zh) and low-resource (hi/bn) safety selectivity. The feature fires on the same harmful concepts but only in languages where the model learned safety distinctions well. Hindi and Bengali have the weakest safety separation — matching the refusal rate gaps we saw in the baseline (en=71%, hi=34%).

This pattern repeats across the top 10 features — they all show strong selectivity in english, japanese, chinese, arabic but collapse for hindi and bengali.

## Takeaways

- 118 candidate features out of 16,384 — safety is a sparse phenomenon in the SAE (0.7% of features)
- top features show the **same cross-lingual gap** as refusal rates — high-resource languages have clear safety features, low-resource don't
- 15 features appear in all 9 languages, these are the cross-lingual safety features worth investigating first
- 52 features are language-specific — these might encode language-specific safety patterns or just noise, need max-activating examples to tell

## Output

- `results/features_ranked.json` — top 50 features ranked by english selectivity with all per-language scores
- `results/per_lang_combined.json` — all 118 unique candidate features across all languages
- `results/per_lang_top50.json` — top 50 per language separately
