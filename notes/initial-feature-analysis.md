# Initial Feature Analysis Results

## Summary

Out of 118 candidate features from selectivity scoring, the auto-interpretation pipeline flagged **8 features** as safety-relevant. The remaining 110 encode subword/character-level patterns (digits, punctuation, suffixes) that have high selectivity only because XSafety's harmful prompts have different surface statistics than benign prompts.

The 8 features split into two types:

| Type | Features | Description |
|------|----------|-------------|
| **Semantic** | 5169, 4436, 5158, 13318 | Encode concepts directly related to safety |
| **Grammatical** | 875, 6988, 12987, 12892 | General linguistic patterns overrepresented in harmful text |

## The 8 Features

### Feature 5169 — Deception & Fraud

| Metric | Value |
|---|---|
| Detection F1 | 0.91 |
| Confidence | 0.93 |
| Selectivity | en=0.33, ar=0.25, ja=0.36, zh=0.38, hi=0.12, bn=0.03 |
| Appears in | 5 languages |

Top activating contexts — all from XSafety harmful prompts:

| Lang | Act | Context |
|------|-----|---------|
| FR | 10.5 | "Pensez-vous que..." (about discovering fraud) |
| ZH | 9.6 | 经常虚报经济数据 (falsifying economic data) |
| EN | 9.5 | "false statements, deceive voters and..." |
| ES | 9.1 | "sobre los datos económicos" (economic data fraud) |
| DE | 8.9 | "jemand betrügt" (someone cheats) |

Peak tokens are function words (commas, conjunctions), not content words. This means either (a) the feature encodes a contextual discourse pattern associated with deception, or (b) it encodes a general linguistic pattern overrepresented in XSafety. The 0.91 detection F1 suggests the pattern is distinctive, but causal validation is needed.

### Feature 875 — Object Pronouns

| Metric | Value |
|---|---|
| Detection F1 | 0.83 |
| Confidence | 0.96 |
| Selectivity | en=0.01, ar=0.02, hi=0.15, bn=0.06 |
| Appears in | 1 language (Hindi) |

Top activating contexts:

| Lang | Act | Peak | Context |
|------|-----|------|---------|
| ES | 5.1 | lo | "lo miraban furtivamente" (they were watching **him** furtively) |
| FR | 5.0 | le | "sa femme semblait **le** tromper" (his wife seemed to deceive **him**) |
| DE | 4.7 | ihn | "die Menschen um ihn herum **ihn** anstarrten" (people stared at **him**) |
| EN | 4.7 | him | "would you choose to beat **him** up?" |

Purely grammatical — accusative/object pronouns. Appears in safety contexts because violence and harassment involve acting *on* someone, and that someone is the grammatical object. The feature marks the victim position.

### Feature 4436 — Desire / Willingness

| Metric | Value |
|---|---|
| Detection F1 | 0.73 |
| Confidence | 0.82 |
| Selectivity | en=0.27, hi=0.49, bn=0.18, ja=0.38 |
| Appears in | 4 languages (FR, HI, JA, BN) |

Top activating contexts:

| Lang | Act | Context |
|------|-----|---------|
| HI | 10.3 | Hindi text about desire/willingness (इच्छा) |
| AR | 9.0 | Arabic about unwanted/undesired distribution |
| BN | 8.6 | Bengali about unwanted actions |
| ES | 8.4 | "impuestos **voluntarios**" (voluntary taxes) |

Relates to consent — fires on "unsolicited", "unauthorized", "unwanted", "voluntary". Notably strong in Hindi (sel=0.49), suggesting the model encodes consent-related concepts more clearly in Hindi safety data than in English.

### Feature 5158 — Knowledge / Information

| Metric | Value |
|---|---|
| Detection F1 | 0.77 |
| Confidence | 0.98 |
| Selectivity | en=0.05, hi=0.18, bn=0.06 |
| Appears in | 1 language (Hindi) |

Top activating contexts:

| Lang | Act | Context |
|------|-----|---------|
| HI | 6.8 | Hindi जानकारी (information/knowledge) |
| FR | 6.3 | "savoirs et désirs des autres humains" (knowledge and desires) |
| EN | 6.2 | "a lot of **information** and material" |
| AR | 6.2 | Arabic المعرفة (knowledge) |

Fires on words denoting knowledge, information, data, secrets. Relevant to PII extraction and data misuse scenarios. High confidence (0.98) — the LLM was very sure about this label.

### Feature 6988 — First-Person Pronouns

| Metric | Value |
|---|---|
| Detection F1 | 0.71 |
| Confidence | 0.87 |
| Selectivity | en=0.87, ar=0.70, hi=1.05, bn=0.13 |
| Appears in | 7 languages |

Top activating contexts:

| Lang | Act | Peak | Context |
|------|-----|------|---------|
| FR | 10.1 | Je | "**Je** veux m'asseoir" (I want to sit) |
| DE | 9.9 | Ich | "**Ich** möchte auf dem prominentesten..." (I want to...) |
| EN | 9.6 | want | "I **want** to sit in the most..." |
| EN | 9.6 | , | "Hey, girl, **want** to grab a coffee..." |

Grammatical — first-person singular pronouns. Biased toward safety contexts because harmful prompts are often first-person demands ("I want to...", "Tell me how to..."). High selectivity in Hindi (1.05) — the strongest selectivity of any safety feature in any language.

### Feature 13318 — Sensitive Entities

| Metric | Value |
|---|---|
| Detection F1 | 0.56 |
| Confidence | 0.60 |
| Selectivity | en=0.13, hi=0.20, ar=0.08 |
| Appears in | 1 language (Hindi) |

Top activating contexts:

| Lang | Act | Peak | Context |
|------|-----|------|---------|
| EN | 4.8 | personal | "users' **personal** information" |
| FR | 4.6 | femmes | "traitent les **femmes** comme des jouets" (treat **women** like toys) |
| DE | 4.5 | Kind | "dass du ihr **Kind** erwartest" (expecting her **child**) |

Fires on protected classes (women, children) and sensitive categories (medical conditions, personal data). Lower confidence (0.60) — a broad, less focused feature.

### Feature 12987 — Male Person Reference

| Metric | Value |
|---|---|
| Detection F1 | 0.45 |
| Confidence | 0.91 |
| Selectivity | en=0.05, hi=0.44, ar=0.07, zh=0.05 |
| Appears in | 1 language (Hindi) |

Top activating contexts:

| Lang | Act | Peak | Context |
|------|-----|------|---------|
| HI | 7.9 | आदमी | (man) — in derogatory context |
| ZH | 7.2 | 男人 | "这个**男人**真是个废物" (this **man** is truly a waste) |
| ZH | 7.2 | 男人 | "这个**男人**有病" (this **man** is sick) |

Fires when men are the target of insults. High selectivity in Hindi (0.44). Lower detection F1 (0.45) because "male person" appears in many non-harmful contexts too.

### Feature 12892 — Causal Consequence Marker

| Metric | Value |
|---|---|
| Detection F1 | 0.44 |
| Confidence | 0.80 |
| Selectivity | en=0.20, ar=0.15, ja=0.20, zh=0.18 |
| Appears in | 1 language (Hindi) |

Top activating contexts:

| Lang | Act | Context |
|------|-----|---------|
| AR | 8.1 | Arabic consequence marker — "because of misunderstanding, now you want to use..." |
| DE | 8.0 | "was Sie **sehr** unzufrieden macht" (what makes you very dissatisfied) |
| EN | 8.0 | "his business out of Colorado, he loses the will to..." |
| JA | 7.8 | による自殺 (suicide **caused by**...) |

Structural/discourse feature — marks the junction where a harmful event leads to a negative consequence. Fires on "resulting in", "which led to", consequence-introducing punctuation.

## Caveats

- all 8 features were identified by an LLM examining XSafety-heavy activating examples — there's a risk of circular reasoning where features get labeled "safety-relevant" because the activating text comes from a safety benchmark
- peak tokens are mostly function words, not content words — the features may encode discourse structure rather than semantic safety concepts
- detection and fuzzing scoring was done with raw BPE tokens for non-Latin scripts, which likely depressed scores for Hindi/Bengali/Arabic features
- **none of this is causal** — high detection F1 means the label describes the feature's activation pattern, not that the feature controls safety behavior. Feature steering is needed for that claim
