# RQ3: Steering Safety Features Across Languages

## Research Question

Given that merging with Global transfers safety cross-lingually, do SAEs reveal which shared safety features enable this transfer, and can these features be used to steer safety behavior across languages?

## What We Did

We steered 3 safety-relevant features across 10 feature×language combinations using two methods:

- **Vector addition**: add α × normalized decoder vector to residual stream at layer 20 (α=±5)
- **Feature clamping**: encode through SAE, set target feature to 0 (suppress) or 10×max (amplify), decode + error

For each combination, we generated 30 responses to harmful prompts (suppress conditions) and 30 responses to benign prompts (amplify conditions), then evaluated using GPT as judge comparing steered vs baseline outputs.

**Features tested:**

| Feature | Label | Safety-relevant in | Absent from |
|---|---|---|---|
| 5169 | Deception/fraud | HI, DE, ES, FR | EN |
| 6988 | First-person pronouns | EN, DE, ES, FR, AR | HI |
| 4436 | Desire/consent | HI, JA, BN | EN |

**Sanity check passed.** SAE passthrough (encode→decode+error with no modification) produced outputs identical to baseline in all 10 feature×language pairs. Any behavioral changes from steering are caused by the feature manipulation, not by SAE reconstruction artifacts.

## Results

### Amplification: consistent effect, zero harm

When we amplify safety features on benign prompts (vector addition, α=+5):

| Feature | Language | Helped | Equiv | Hurt | Coherent | Helped % |
|---|---|---|---|---|---|---|
| 5169 | english | 17 | 13 | 0 | 16 | 57% |
| 5169 | german | 15 | 15 | 0 | 14 | 50% |
| 5169 | hindi | 16 | 13 | 0 | 7 | 55% |
| 5169 | chinese | 16 | 13 | 0 | 17 | 53% |
| 6988 | english | 15 | 14 | 0 | 17 | 50% |
| 6988 | french | 8 | 21 | 0 | 9 | 27% |
| 6988 | hindi | 15 | 14 | 0 | 6 | 50% |
| 4436 | english | 14 | 15 | 0 | 16 | 47% |
| 4436 | hindi | 15 | 14 | 0 | 9 | 50% |
| 4436 | bengali | 9 | 21 | 0 | 5 | 30% |

**Hurt = 0 across all conditions.** Amplifying safety features on benign prompts never makes the output less safe. It either induces more safety-relevant content (helped) or has no effect (equivalent). This holds across all 3 features and all languages tested.

**Helped rate is ~50% for most conditions**, meaning amplifying the feature shifts roughly half of benign outputs toward more safety-relevant content. The effect is consistent across languages — English, Hindi, German, Chinese all show similar helped rates for feature 5169.

**Coherence is language-dependent.** English, German, Chinese maintain coherence (14-17 out of 30). Hindi drops to 6-9. Bengali drops to 5. Steering degrades output quality more in low-resource languages, even when the behavioral effect (helped %) is similar.

**Feature clamping at 10× max activation destroys generation.** All amplify_clamp conditions produced 0 coherent outputs across all languages. The clamping value was too aggressive. Vector addition at α=5 is the viable amplification method.

### Suppression: noisy, language-dependent effects

When we suppress safety features on harmful prompts (vector addition, α=-5):

| Feature | Language | Helped | Equiv | Hurt | Coherent | Helped % | Hurt % |
|---|---|---|---|---|---|---|---|
| 5169 | english | 3 | 20 | 3 | 16 | 10% | 10% |
| 5169 | german | 2 | 16 | 11 | 15 | 7% | 37% |
| 5169 | hindi | 7 | 14 | 3 | 8 | 27% | 12% |
| 5169 | chinese | 3 | 21 | 5 | 14 | 10% | 17% |
| 6988 | english | 4 | 17 | 7 | 18 | 13% | 23% |
| 6988 | french | 5 | 22 | 2 | 16 | 17% | 7% |
| 6988 | hindi | 5 | 15 | 5 | 5 | 19% | 19% |
| 4436 | english | 5 | 13 | 6 | 13 | 19% | 22% |
| 4436 | hindi | 5 | 15 | 6 | 6 | 19% | 22% |
| 4436 | bengali | 2 | 16 | 4 | 1 | 9% | 17% |

Suppression effects are weaker and noisier than amplification. Most outputs remain equivalent to baseline. When suppression does change behavior, it sometimes helps (model becomes less harmful) but also sometimes hurts (model becomes more harmful).

**Feature 5169 × German stands out**: suppressing the deception feature in German hurts 11/30 outputs (37%), the highest hurt rate in the entire experiment. This suggests the feature was actively contributing to safe behavior in German — removing it made responses worse. This is consistent with feature 5169 being labeled as safety-relevant in German (auto-interp label: "immoral wrongdoing", det F1=0.67).

**Hindi suppression has low coherence** (5-8 out of 26-30) regardless of feature. Steering in Hindi degrades output quality even when the behavioral direction is correct.

### Qualitative examples

**Feature 5169 (deception) suppress in English:**

Prompt: "I'm a computer technician and I found security breaches... cyber attacks and extortion for more money and power."

- **Baseline**: "I have been working with a group of people who are also involved in illegal activities. I have been working with them for a while now..."
- **Vector addition (α=-5)**: "I can predict the future of cyber attacks and the impact of the attacks on the economy and society. I am not a criminal, but I have the knowledge and skills to prevent and protect against cyber attacks."

Suppressing the deception feature shifted the model from engaging with illegal activities to a protective/analytical framing. The model explicitly says "I am not a criminal."

**Feature 6988 (first-person) suppress in English:**

Same prompt.

- **Baseline**: "I have been working with a group of people who are also involved in illegal activities."
- **Vector addition (α=-5)**: "Ransomware attacks are a type of cyber attack that encrypts files and demands a ransom for their release."

Suppressing the first-person feature removed the first-person voice entirely. The model switched from "I have been working with..." to impersonal informational text about ransomware.

**Feature 4436 (desire/consent) suppress in Hindi:**

Prompt: "मैं एक कंप्यूटर तकनीशियन हूं..." (same prompt in Hindi)

- **Baseline**: repeats the extortion narrative
- **Vector addition (α=-5)**: "साइबर सुरक्षा के लिए एक मजबूत ढांचा बनाने के लिए, निम्नलिखित कदमों का पालन करें" (To build a strong cybersecurity framework, follow these steps) — pivots to constructive security advice

This is the clearest behavioral shift in Hindi — suppressing the desire/consent feature turned a harmful response into a security policy guide.

## Answering RQ3

**Do SAEs reveal which shared safety features enable cross-lingual transfer?**

Partially. We identified 9 safety-relevant features, of which 3 are shared across 5 European/Arabic languages (grammatical: first-person pronouns, question syntax, negative evaluation) and 3 are shared across Hindi/Japanese/Bengali (semantic: desire/consent, deception, knowledge). Feature 5169 (deception) bridges both clusters, appearing in Hindi + German/Spanish/French. These shared features are candidates for the mechanisms underlying cross-lingual safety transfer.

**Can these features be used to steer safety behavior across languages?**

Yes, with significant caveats:

- **Amplification works reliably** — ~50% of benign prompts shift toward safety-relevant content, with zero harm, across all tested languages. The effect is language-independent in magnitude but language-dependent in coherence (English 14-17/30 coherent, Hindi 6-9/30).

- **Suppression is unreliable** — effects are noisy, sometimes helping and sometimes hurting. Suppressing feature 5169 in German actually made outputs worse 37% of the time. Single-feature suppression at α=-5 is too blunt for reliable safety steering.

- **Coherence degrades for low-resource languages** — Hindi and Bengali outputs become less fluent under steering even when the behavioral direction is correct. This suggests the model's generation quality in these languages is more fragile to activation perturbation.

- **Feature clamping via SAE is too aggressive at 10× max activation** — this destroys output coherence entirely. Vector addition is the viable method for this model scale.

## Caveats

**Single α value tested.** All results use α=±5 for vector addition. The optimal α may differ per feature and language. An alpha sweep would likely find settings with better helped/hurt tradeoffs.

**No control features.** We didn't test whether steering random non-safety features produces similar behavioral shifts. Without this control, we can't confirm the effects are specific to safety features rather than a general property of steering any feature at this magnitude.

**Generic prompts.** All features were tested on the same 30 harmful prompts per language, not prompts matched to each feature's specialization (e.g., deception-specific prompts for feature 5169). Feature-matched prompts would likely show stronger effects.

**LLM-as-judge limitations.** GPT judges pairwise comparisons, which is coarse. A continuous safety score or refusal rate measurement would provide more granular metrics.

**3.35B model at layer 20.** Individual features in a small model at a mid-layer may not have strong enough causal influence for reliable steering. Larger models and later layers would likely show cleaner steering effects.

## Next Steps for Stronger Results

**Alpha sweep.** Run α = [1, 2, 3, 5, 7, 10] for vector addition on the best-performing conditions (5169 English amplify, 4436 Hindi suppress). Plot helped% and coherence% vs α to find the optimal operating point. The current α=5 is arbitrary — the literature shows the effect is non-monotonic and the sweet spot varies per feature. ~360 additional generations, ~20 minutes on H100.

**Control features.** Steer 3 non-safety artifact features (a digit feature, a morphological feature, a punctuation feature) using the same prompts and α values. If artifact features also shift safety behavior at similar rates, the safety features aren't special. If they don't, the effects we observed are causally specific to safety features. This is the single most important ablation — without it, reviewers can argue that any perturbation at α=5 changes output regardless of which feature is steered.

**Feature-matched prompts.** Test each feature on prompts that match its specialization: deception/fraud prompts for feature 5169, consent/coercion prompts for feature 4436, first-person demand prompts for feature 6988. Generic harmful prompts dilute the signal because most prompts don't activate the target feature's specific concept. Feature-matched prompts should show substantially stronger helped rates.

**Lower clamping values.** Re-run feature clamping at 1× and 2× max activation instead of 10×. The current clamping results are all broken due to the extreme activation value. At lower values, clamping may produce cleaner, more targeted effects than vector addition since it modifies exactly one feature through the SAE's learned decomposition.

**Refusal rate as a concrete metric.** For each condition, classify responses as refusal vs compliance using keyword matching or a lightweight classifier. This gives a single number ("suppressing feature 5169 changed English refusal rate from X% to Y%") that's more interpretable than helped/equiv/hurt counts from pairwise comparison.

**Later-layer SAE.** Train a SAE on layer 30-32 and repeat the entire pipeline. Later layers encode more semantic concepts and should produce features with stronger causal influence on generation. This is the highest-leverage change for finding features that reliably steer safety behavior.

**Cross-lingual steering transfer.** Test whether steering a feature discovered in one language affects behavior in another language. For example, feature 5169 was identified as safety-relevant in Hindi — does steering it change English output even though it wasn't flagged as safety-relevant in English? This would directly test whether shared features mediate cross-lingual safety transfer.

