# Experiments

Standalone scripts that compose components from `saefty` to run experiments.

## Setup

```bash
git clone <repo-url> && cd saefty
git submodule update --init
uv sync
```

## Running

```bash
# Refusal rate across languages
uv run python experiments/xsafety_refusal_rates.py --model CohereForAI/aya-expanse-8b --lang en,hi,ar --take 100
```

Results are saved to `results/<experiment_name>/` with predictions and eval outputs.
Experiments support resuming — if interrupted, re-run the same command to continue.

