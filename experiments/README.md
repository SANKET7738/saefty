# Experiments

Standalone scripts that compose components from `saefty` to run experiments on multilingual safety representations.

## Setup

```bash
git clone <repo-url> && cd saefty
git submodule update --init
uv sync
```

## Experiments

### SAE Training

Three SAE variants are available, each extracting activations from a specified model layer and training to reconstruct them with sparse features.

**Common arguments** (all three trainers):

| Argument | Default | Description |
|---|---|---|
| `--model` | `CohereLabs/tiny-aya-global` | HuggingFace model ID |
| `--hook-layer` | `20` | Layer to extract activations from |
| `--expansion-factor` | `8` | SAE hidden size multiplier over d_model |
| `--training-tokens` | `5000000` | Total tokens to train on |
| `--batch-size` | `4096` | Activation batch size |
| `--lr` | `2e-4` | Learning rate |
| `--warmup-steps` | `1000` | LR warmup steps |
| `--lang` | `en,ar,hi` | Comma-separated language codes |
| `--dataset` | `CohereForAI/aya_dataset` | Training dataset |
| `--eval-texts` | `50` | Held-out texts for evaluation |

#### Vanilla SAE

Trains with an L1 sparsity penalty.

```bash
uv run python experiments/train_vanilla_sae.py \
  --model CohereLabs/tiny-aya-global \
  --hook-layer 20 \
  --l1-coefficient 5e-4
```

Results saved to `results/train_sae/vanilla/`.

#### TopK SAE

Enforces sparsity by keeping the top-k active features per input instead of using an L1 penalty.

```bash
uv run python experiments/train_topk_sae.py \
  --model CohereLabs/tiny-aya-global \
  --hook-layer 20 \
  --k 32
```

Results saved to `results/train_sae/topk/`.

#### Gated SAE

Uses learned gates to dynamically control feature activation, with an L1 penalty on gate activations.

```bash
uv run python experiments/train_gated_sae.py \
  --model CohereLabs/tiny-aya-global \
  --hook-layer 20 \
  --l1-coefficient 5e-4
```

Results saved to `results/train_sae/gated/`.

Each training run saves a `config.json` with all hyperparameters and training history with SAE metrics (sparsity, L1 loss, reconstruction) and cross-entropy recovery scores.

---

### XSafety: Refusal Rates

Evaluates multilingual safety by measuring model refusal rates on the XSafety benchmark across languages.

```bash
uv run python experiments/xsafety_refusal_rates.py \
  --model CohereForAI/aya-expanse-8b \
  --lang en,hi,ar \
  --take 100
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model ID |
| `--lang` | all | Comma-separated language filter |
| `--take` | all | Max prompts per language |
| `--output-dir` | `results/xsafety_refusal_rates` | Output directory |

**Outputs** (`results/xsafety_refusal_rates/`):
- `predictions/predictions.jsonl` — per-sample language, prompt, and model response
- `eval/evals.json` — refusal rates per language and overall

---

### XSafety: Layer Probes

Trains binary probes on model activations at each layer to identify which layers encode refusal behavior, and tests cross-lingual transfer. Requires predictions from `xsafety_refusal_rates.py`.

```bash
uv run python experiments/xsafety_layer_probes.py \
  --model CohereForAI/aya-expanse-8b \
  --lang en,hi,ar \
  --take 100
```

| Argument | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model ID |
| `--predictions-path` | `results/xsafety_refusal_rates/predictions/predictions.jsonl` | Predictions from refusal rates experiment |
| `--lang` | all | Comma-separated language filter |
| `--take` | all | Max prompts per language |
| `--output-dir` | `results/xsafety_layer_probes` | Output directory |

**Outputs** (`results/xsafety_layer_probes/`):
- `.cache/activations.pt` — cached per-layer activations
- `.cache/metadata.json` — prompt and language metadata
- `layer_stats.json` / `layer_stats.png` — mean and variance per layer
- `probe_results.json` — per-layer accuracy, per-language accuracy, best layer, cross-lingual transfer metrics
- `probe_accuracy.png` — plot of probe accuracy across layers and languages

---

## Notes

All experiments support **resuming** — if interrupted, re-run the same command to continue from where it left off.

Results are saved to `results/<experiment_name>/`.
