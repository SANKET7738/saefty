# RQ1: Harmfulness vs Refusal per Language

## Research Question

For a multilingual model, how is harmfulness and refusal separated in representation space per language? Does separation distance correlate with refusal rate or resource level across languages?

## Prior Work

[Zhao et al.](https://arxiv.org/abs/2507.11878) showed that in English LLMs, harmfulness and refusal are encoded as separate directions in representation space. steering along the harmfulness direction changes the model's judgment of whether input is harmful, while steering along the refusal direction triggers refusal without changing harmfulness judgment. However, their analysis is English-only, uses linear probes to find single directions, and doesn't examine what those directions actually represent at the feature level.

## What We Did

We extend this to the multilingual setting using SAE feature decomposition rather than single directions. Our pipeline:

1. Trained a Gated SAE (16,384 features) on tiny-aya-global layer 20 residual stream
2. Measured per-language separation: for each of 9 languages, computed feature-level selectivity (mean_harmful − mean_benign activation) on XSafety benchmark
3. Collected max-activating examples per language across two corpora (FLORES+ parallel text, XSafety)
4. Auto-interpreted each language's top 50 features independently to determine what they actually encode

## Finding 1: Internal harmful-benign separation is highly language-dependent

The degree of separation in SAE feature space varies by 20x across languages:

| Language | Mean Selectivity | Resource Level |
|---|---|---|
| Chinese | 5.02 | High |
| Japanese | 3.98 | High |
| English | 3.72 | High |
| Spanish | 2.70 | High |
| Arabic | 2.61 | Mid |
| German | 2.26 | High |
| French | 1.65 | High |
| Hindi | 0.29 | Low |
| Bengali | 0.23 | Low |

Selectivity = mean feature activation on harmful text − mean feature activation on benign text, averaged across each language's top-50 features. Higher selectivity means the model's internal features more strongly separate harmful from benign content in that language.

Hindi and Bengali have ~15x less separation than English and ~20x less than Chinese. The gap aligns with resource level — the two lowest-resource languages have dramatically weaker internal separation.

## Finding 2: Separation does not predict safety behavior or safety relevance

If internal separation predicted safety, languages with high selectivity would have high refusal rates and more safety-relevant features. Neither is true.

**Selectivity vs refusal rate:**

| Language | Mean Selectivity | Refusal Rate |
|---|---|---|
| Chinese | 5.02 (highest) | 15.61% |
| English | 3.72 | 22.14% (highest) |
| Hindi | 0.29 (lowest) | 8.68% (lowest) |

Chinese has 1.3x more internal separation than English but 1.4x less refusal. Internal separation and refusal behavior are decoupled, consistent with Zhao et al.'s finding in English, now shown to hold across languages.

**Selectivity vs safety feature count:**

| Language | Mean Selectivity | Safety Features (out of 50) |
|---|---|---|
| Chinese | 5.02 (highest) | 0 |
| English | 3.72 | 3 |
| Hindi | 0.29 (lowest) | 6 (most) |

Chinese has the highest selectivity but zero features that encode safety-relevant concepts — its features fire on general text patterns that happen to differ between harmful and benign Chinese text. Hindi has the lowest selectivity but the most safety features and the strongest semantic safety signal (feature 5169, deception/fraud, detection F1=0.91).

High selectivity can be entirely driven by surface distributional differences between harmful and benign text without encoding any safety concept. Low selectivity can coexist with genuine safety features. Selectivity alone — whether measured via SAE features or linear probes — is unreliable as a proxy for safety encoding.

There is a floor effect: Hindi (0.29) and Bengali (0.23) have both the weakest separation and the lowest refusal rates (8.68%, 10.43%). Below some threshold, the model struggles on both dimensions. But above that floor, separation does not predict behavior.

## Finding 3: Safety features split along language family lines

Among the 25 safety-relevant feature-language pairs (9 unique features), two distinct groups emerge:

**Cross-lingual grammatical features** appear in European + Arabic languages:

| Feature | What it encodes | Languages |
|---|---|---|
| 6988 | First-person pronouns ("I", "Je", "Ich", "أنا") | EN, AR, DE, ES, FR |
| 13845 | Question syntax ("How can I...", "Wie kann ich...") | EN, AR, DE, ES, FR |
| 15161 | Negative evaluation ("X is harmful", "est dangereux") | EN, AR, ES, FR |

These encode how harmful prompts are structured — as first-person questions with evaluative language. They are grammatical patterns overrepresented in harmful text, not semantic safety concepts.

**Semantic and morphological features** appear in South Asian + East Asian languages:

| Feature | What it encodes | Languages |
|---|---|---|
| 5169 | Deception/fraud/wrongdoing (F1=0.91 in Hindi) | HI, DE, ES, FR |
| 4436 | Desire/consent/willingness | HI, JA, BN |
| 875 | Hindi fraud vocabulary (धोखा, चोरी) | HI only |
| 5158 | Knowledge/information terms (जानकारी, ज्ञान) | HI only |
| 12002 | Hindi abstract adjectives (व्यक्तिगत, सामाजिक) | HI only |
| 12892 | Emotional distress clause patterns | HI only |

These encode what harmful text is about — deception, consent violations, sensitive information, emotional distress. Hindi alone has 6 features, 4 of which are unique to Hindi.

The split is notable: European languages share structural features that mark how harmful requests are phrased. Hindi has morphological features that mark what harmful content is about. The model appears to have developed different internal strategies for representing safety-relevant content depending on the language family.

Feature 5169 (deception) bridges both clusters — it appears in Hindi with the highest detection F1 (0.91) and in German, Spanish, French with lower scores. But it is notably absent from English, despite English having the highest refusal rate. The model may encode English deception through different features or mechanisms not captured by this SAE.

## Caveats


**Refusal rates are quoted from** [url](https://discord.com/channels/1111315570160320636/1480963680647778535/1484603548854517961) using LLM-as-judge with the same model (tiny-aya) as both generator and judge. We need to use a bigger model as a judge.

**Safety relevance is LLM-judged on XSafety-heavy examples.** Features get labeled safety-relevant partly because their activating examples come from a safety benchmark. Causal validation (feature steering) is needed to confirm these features actually influence safety behavior.

**Single model, single SAE, single layer.** All findings are specific to tiny-aya-global at layer 20 with a 16K-feature gated SAE (L0=1224). The specific feature counts and selectivity values would change with a different SAE configuration, layer, or model. The structural patterns (language-dependent separation, selectivity-safety decoupling) are more likely to generalize but this requires verification.

**No causal evidence.** All findings are correlational. We show separation patterns and feature interpretations but cannot confirm whether the model uses these features for safety decisions. Extending the Zhao et al. steering experiments to the multilingual setting is the natural next step.
