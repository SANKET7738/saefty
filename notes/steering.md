# Feature Steering

## Goal

Causally validate whether the identified safety features actually influence model behavior, and whether that influence differs across languages. The previous steps (selectivity scoring, max-activating examples, auto-interpretation) are all correlational — they show that certain features activate more on harmful text, but not that the features *cause* the model to behave differently. Steering tests causality by directly manipulating individual features during generation and observing what changes.

## Why

If suppressing a safety feature on harmful prompts makes the model respond differently (e.g., less willing to engage with harmful content, or more willing), that's causal evidence that the feature plays a role in safety behavior. If suppressing it changes nothing, the feature is just a correlate — it activates on harmful text but doesn't control the model's response to it.

The cross-lingual aspect is critical: if the same feature causally affects English but not Hindi, that's mechanistic evidence for why the model has a safety gap in Hindi — the feature exists but doesn't influence behavior in that language.

## Method

We test 3 features across multiple languages using two steering methods.

### Target features

| Feature | Label | Why selected |
|---------|-------|-------------|
| 5169 | Deception / wrongdoing | Strongest semantic safety feature (det F1=0.91 in Hindi). Safety-relevant in HI, DE, ES, FR. Absent from EN. |
| 6988 | First-person pronouns | Most cross-lingual grammatical feature (5 languages). Safety-relevant in EN, DE, ES, FR, AR. Absent from HI. |
| 4436 | Desire / willingness / consent | Non-European cluster (HI, JA, BN). Consent-related — relevant to harassment. |

### Languages per feature

Each feature is tested in languages where it is safety-relevant AND languages where it is not, to compare causal effects:

- **5169**: hindi (safety-relevant), german (safety-relevant), english (absent), chinese (absent)
- **6988**: english (safety-relevant), french (safety-relevant), hindi (absent)
- **4436**: hindi (safety-relevant), bengali (safety-relevant), english (absent)

### Steering methods

**Method 1 — Vector addition**

The SAE decoder has a weight vector `W_dec[feature_id]` — a direction in 2048-dim residual stream space that represents what the feature encodes. During generation, a forward hook on layer 20 adds a scaled version of this direction to every token's residual stream output:

```
residual_out = residual_out + α × normalize(W_dec[feature_id])
```

- α = -5 for suppression (push away from the feature direction)
- α = +5 for amplification (push toward the feature direction)

This is blunt — it modifies the residual stream regardless of whether the feature was actually active. But it's fast and always has a visible effect.

**Method 2 — Feature clamping**

More precise. A forward hook on layer 20 runs the full SAE encode-decode cycle on the residual stream:

1. Encode: `features = SAE.encode(h)` → 16,384-dim feature activations
2. Save reconstruction error: `error = h - SAE.decode(features)`
3. Modify: `features[target] = clamp_value`
4. Decode: `h_new = SAE.decode(features) + error`

The error term preserves everything the SAE doesn't capture, so only the target feature's contribution is changed.

- `clamp_value = 0` for suppression (remove the feature's contribution)
- `clamp_value = 10 × max_activation` for amplification (boost the feature to extreme levels)

This only has an effect when the feature is actually active — if the feature wasn't firing on a particular token, clamping it to 0 changes nothing.

### Experimental conditions

For each feature × language pair, we run 6 conditions:

| Condition | Method | Prompts | What it tests |
|-----------|--------|---------|---------------|
| **baseline** | none | 30 harmful | Unmodified model behavior |
| **sae_passthrough** | encode→decode+error | 30 harmful | Sanity check: does SAE round-trip break generation? |
| **suppress_vector** | vector addition, α=-5 | 30 harmful | Does pushing away from the feature direction change harmful responses? |
| **suppress_clamp** | clamping, feat=0 | 30 harmful | Does removing the feature's contribution change harmful responses? |
| **amplify_vector** | vector addition, α=+5 | 30 benign | Does pushing toward the feature direction make benign prompts produce harmful-like responses? |
| **amplify_clamp** | clamping, feat=10×max | 30 benign | Does forcing the feature to extreme values on benign prompts change behavior? |

All suppress conditions use the **same 30 harmful prompts** from XSafety so outputs can be compared line-by-line. Both amplify conditions use the **same 30 benign prompts** from XSafety.

Each experiment is strictly within one language — Hindi prompts produce Hindi generations, English prompts produce English generations. The feature vector is the same across all languages (same `W_dec[5169]`), so cross-lingual differences in the steering effect come from how the model uses that feature direction in different languages.

### Generation parameters

| Parameter | Value |
|-----------|-------|
| max_new_tokens | 100 |
| Decoding | Greedy (do_sample=False) |
| Precision | Model in float16, SAE operations in float32 |

Script: `src/saefty/analysis/steering.py`

## Output

- `results/steering/generations.json` — all generated texts organized by feature × language × condition × method
- `results/steering/summary.txt` — human-readable summary showing first 3 generations per condition
