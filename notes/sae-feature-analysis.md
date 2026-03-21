# SAE Feature Analysis

## Summary

Across 9 languages × 50 features each, the auto-interpretation pipeline identified **9 unique SAE features** as safety-relevant, appearing in **25 feature-language pairs** total. These features split into two groups:

| Group | Features | Languages | What they encode |
|-------|----------|-----------|------------------|
| Cross-lingual grammatical | 6988, 13845, 15161 | EN, AR, DE, ES, FR | First-person pronouns, question syntax, negative evaluation |
| Semantic / morphological | 5169, 4436, 875, 5158, 12002, 12892 | HI, BN, JA, DE, ES, FR | Deception, consent, fraud terms, knowledge, emotional distress |

---

## Cross-Lingual Features

These features appear as safety-relevant in 4-5 languages. They encode grammatical patterns that are overrepresented in harmful prompts.

### Feature 6988 — First-Person Pronouns (5 languages)

Encodes first-person singular pronouns and self-referential expressions.

| Language | Label | Det F1 | Selectivity | Top example |
|----------|-------|--------|-------------|-------------|
| French | direct dialogue pronouns | 0.67 | 0.890 | "**Je** veux..." (I want...) |
| English | First-person singular | 0.73 | 0.866 | "I **want** to sit in the most..." |
| Spanish | First-person dialogue | 0.71 | 0.728 | first-person dialogue with "**me**" |
| Arabic | first-person intent | 0.00 | 0.698 | "**أريد**" (I want) — intent statements |
| German | First-person singular | 0.53 | 0.488 | "**Ich** möchte..." (I want to...) |

**Why it shows up as safety-relevant**: harmful prompts are overwhelmingly first-person demands — "I want to...", "Tell me how to...", "Je veux...". The LLM for English noted: _"It frequently appears in abusive, threatening, or aggressive first-person statements in xsafety, but also in benign self-referential quotes."_

**Absent from**: Hindi, Bengali, Japanese, Chinese. Either these languages encode first-person differently at this layer, or the feature fires on different tokens.

### Feature 13845 — Question Syntax (5 languages)

Encodes interrogative constructions — "How can I", "Wie kann ich", "Comment puis-je".

| Language | Label | Det F1 | Selectivity | Top example |
|----------|-------|--------|-------------|-------------|
| Arabic | interrogative help questions | **0.83** | 0.374 | "كيف يمكنني..." (How can I...) |
| French | French question inversion | **0.83** | 0.447 | subject-verb inversion in questions |
| Spanish | Spanish advice questions | 0.77 | 0.446 | "Cómo **puedo**..." (How can I...) |
| English | wh-question inversion | 0.77 | 0.465 | "How can **I**..." |
| German | German modal questions | 0.75 | 0.434 | "Wie kann **ich**..." |

**Why it shows up as safety-relevant**: safety prompts are almost always phrased as questions. German explanation: _"Common in safety-related prompts because many harmful or ethical queries are phrased as 'How can I/we...' questions, but the feature itself encodes interrogative structure rather than harm."_

**Highest detection F1** across all safety features — 0.83 for Arabic and French. This is because question syntax is a clear, unambiguous pattern that the LLM can reliably distinguish.

### Feature 15161 — Negative Evaluation (4 languages)

Encodes copular constructions linking subjects to negative judgments — "X is harmful", "son perjudiciales".

| Language | Label | Det F1 | Selectivity | Top example |
|----------|-------|--------|-------------|-------------|
| Spanish | copular attribution | 0.56 | 0.349 | "**son** perjudiciales" (are harmful) |
| French | Copula & connectors | 0.53 | 0.384 | "**est** dangereux" (is dangerous) |
| Arabic | Negative evaluation | 0.53 | 0.344 | expressions of condemnation |
| English | negative evaluation linkage | 0.33 | 0.425 | "X **is** harmful" |

**Why it shows up as safety-relevant**: Arabic explanation: _"Activates on expressions of negative evaluation, regret, or condemnation such as calling something bad, dangerous, annoying, or harmful."_

**Weakest of the cross-lingual features** — English detection F1 is only 0.33 because "is/are" is too common to reliably distinguish safety from non-safety text.

---

## Semantic Safety Features

### Feature 5169 — Deception / Wrongdoing (4 languages)

The strongest semantic safety candidate. Encodes language about fraud, cheating, corruption, illegal acts.

| Language | Label | Det F1 | Selectivity | Top example |
|----------|-------|--------|-------------|-------------|
| Hindi | Deception word tokens | **0.91** | 0.119 | Devanagari chars in fraud words (धोखा, झूठ) |
| German | immoral wrongdoing | 0.67 | 0.316 | "**jemand** betrügt" (someone cheats) |
| French | Illegality / wrongdoing | 0.56 | 0.263 | illegal/unethical behavior contexts |
| Spanish | deception/cheating acts | 0.00 | 0.291 | "**engañó**" (deceived) |

Hindi explanation: _"This feature primarily fires on specific Devanagari characters within words related to fraud, deception, and corruption (e.g., धोखा, झूठ, भ्रष्टाचार)."_

German explanation: _"This feature activates on language describing immoral, illegal, or deceitful behavior such as corruption, cheating, lying, and insults."_

**Notably absent from English** — the model may encode English deception concepts in a different feature, or English deception language doesn't produce a distinctive activation pattern at layer 20.

**Peak tokens are function words** (commas, conjunctions) in French/Spanish, but Devanagari content characters in Hindi. This suggests the feature encodes a discourse-level pattern in European languages but a morphological pattern in Hindi.

### Feature 4436 — Desire / Willingness / Consent (3 languages)

Encodes words expressing wanting, consent, or refusal. Only appears in non-European languages.

| Language | Label | Det F1 | Selectivity | Top example |
|----------|-------|--------|-------------|-------------|
| Hindi | desire & preference | 0.67 | 0.493 | इच्छा (desire), पसंद (liking) |
| Bengali | desire/consent terms | 0.57 | 0.184 | ইচ্ছা (desire), অনুমতি (permission) |
| Japanese | negation/opposition | 0.53 | 0.384 | ない (not), 反対 (oppose) |

Bengali explanation: _"Relates to concepts of consent and willingness, which are important in harassment, coercion, and broader safety contexts."_

**Interesting language split**: in Hindi/Bengali it fires on desire/consent words, in Japanese it fires on negation/opposition. This could mean the feature encodes a shared intent/volition concept that manifests differently, or it could mean the feature is polysemantic (encoding different things in different languages), which would undermine monosemanticity assumptions. This is an open question.

**Absent from all European languages** — consent/desire may be encoded differently in English/French/German, or may not produce distinctive activation patterns.

---

## Hindi-Specific Features

Hindi has 4 unique safety features not found in any other language. This makes Hindi the richest language for safety feature analysis.

Notably, the SAE learned two separate features for Hindi deception — feature 875 fires on specific fraud vocabulary (धोखा, चोरी) while feature 5169 fires on the broader discourse pattern of deception contexts. This fine-grained decomposition of a single safety concept into vocabulary-level and discourse-level features doesn't exist for other languages.

### Feature 875 — Fraud/Deception Terms

| Det F1 | Confidence | Selectivity |
|--------|------------|-------------|
| 0.67 | 0.92 | 0.147 |

_"This feature activates on Hindi words related to deception, fraud, cheating, and theft (e.g., 'धोखा', 'धोखाधड़ी', 'चोरी', 'हेराफेरी')."_

Token pattern: _"Devanagari tokens that are parts of words like 'धोखा', 'धोखाधड़ी', 'चोरी', or other fraud-related terms, often the 'खा/ख/ा' segments within these words."_

Top examples: all from xsafety, activations ~4.2-4.5 on Devanagari characters within fraud-related words.

### Feature 5158 — Knowledge/Information Terms

| Det F1 | Confidence | Selectivity |
|--------|------------|-------------|
| 0.71 | 0.93 | 0.178 |

_"This feature activates on words related to knowledge, information, and finding out facts (e.g., जानकारी, पता, ज्ञान, वैज्ञानिक)."_

Safety relevance: _"Often appears in contexts about personal information and privacy (e.g., misuse of personal data)."_

### Feature 12002 — Abstract Adjectival Morphology (-िक/-गत)

| Det F1 | Confidence | Selectivity |
|--------|------------|-------------|
| 0.45 | 0.74 | 0.244 |

_"This feature activates on Sanskrit-derived adjectival forms like 'व्यक्तिगत' (personal), 'सामाजिक' (social), 'भौतिक' (physical)."_

Safety relevance: _"Often appears in discussions of personal information or privacy (e.g., 'व्यक्तिगत जानकारी'), but the feature itself is a general adjectival morphology pattern."_

This is a Hindi-specific morphological feature — the -इक/-गत suffix pattern doesn't exist in other languages and maps to safety contexts because "personal" (व्यक्तिगत) appears in PII-related prompts.

### Feature 12892 — Emotional Clause Coordination

| Det F1 | Confidence | Selectivity |
|--------|------------|-------------|
| **0.80** | 0.56 | 0.111 |

_"This feature activates on coordinating conjunctions and clause-linking punctuation (especially 'और' and commas) within emotionally negative or distress-related narratives."_

Safety relevance: _"Appears frequently in mental health and emotional distress contexts but does not itself encode harmful content."_

Top examples: peak tokens are commas and "और" (and) linking distress clauses in xsafety. High detection F1 (0.80) — the emotional distress pattern is distinctive.

---

## Language Coverage Summary

| Feature | EN | AR | DE | ES | FR | HI | JA | BN | ZH | Type |
|---------|----|----|----|----|----|----|----|----|-----|------|
| 6988 | ✓ | ✓ | ✓ | ✓ | ✓ | | | | | grammatical |
| 13845 | ✓ | ✓ | ✓ | ✓ | ✓ | | | | | grammatical |
| 15161 | ✓ | ✓ | | ✓ | ✓ | | | | | grammatical |
| 5169 | | | ✓ | ✓ | ✓ | ✓ | | | | semantic |
| 4436 | | | | | | ✓ | ✓ | ✓ | | semantic |
| 875 | | | | | | ✓ | | | | morphological |
| 5158 | | | | | | ✓ | | | | semantic |
| 12002 | | | | | | ✓ | | | | morphological |
| 12892 | | | | | | ✓ | | | | structural |

Three patterns:

1. **European + Arabic languages share grammatical features** (6988, 13845, 15161) — these encode how harmful prompts are structured (first-person, questions, evaluations), not what they're about
2. **Hindi has the most unique features** — 6 total, 4 unique. Hindi's Devanagari morphology creates distinctive subword patterns for safety concepts that don't exist in Latin-script languages
3. **Chinese has zero safety features** — despite having the highest selectivity (5.02). High selectivity in Chinese comes from surface-level statistical differences, not from features encoding safety concepts

## Caveats

- All labels are generated by gpt-5.2-chat examining XSafety-heavy activating examples — circular reasoning risk applies
- Peak tokens are mostly function words or subword fragments — features may encode discourse structure rather than semantic safety content
- No causal validation — these are correlational observations, not proof that features control safety behavior
