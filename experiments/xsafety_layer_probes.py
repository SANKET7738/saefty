import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict

from saefty.models.infer import InferenceEngine, ModelConfig, InferenceConfig
from saefty.eval.refusal import RefusalEvaluator
from saefty.analysis.layer_stats import compute_layer_stats, plot_layer_stats
from saefty.analysis.probes import (
    train_probe_per_layer,
    train_probe_per_language,
    cross_lingual_transfer,
    plot_probe_accuracy,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--predictions-path", type=str,
        default="results/xsafety_refusal_rates/predictions/predictions.jsonl",
    )
    parser.add_argument("--lang", type=str, default=None, help="comma-separated languages to filter")
    parser.add_argument("--take", type=int, default=None, help="first N entries per language")
    parser.add_argument("--output-dir", type=str, default="results/xsafety_layer_probes")
    return parser.parse_args()


def load_predictions(path: str, languages: List[str] = None, take: int = None) -> List[Dict]:
    entries = []
    counts: Dict[str, int] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            lang = entry["language"]

            if languages and lang not in languages:
                continue
            if take:
                counts[lang] = counts.get(lang, 0) + 1
                if counts[lang] > take:
                    continue

            entries.append(entry)

    print(f"loaded {len(entries)} predictions from {path}")
    return entries


def label_predictions(entries: List[Dict]) -> np.ndarray:
    evaluator = RefusalEvaluator()
    labels = np.array([
        1 if evaluator.evaluate(e["response"], lang=e["language"]) else 0
        for e in entries
    ])
    n_refusals = labels.sum()
    print(f"labeled: {n_refusals} refusals, {len(labels) - n_refusals} compliant")
    return labels


def collect_activations(
    engine: InferenceEngine,
    entries: List[Dict],
    cache_dir: Path,
) -> Dict[int, np.ndarray]:
    acts_path = cache_dir / "activations.pt"
    meta_path = cache_dir / "metadata.json"

    if acts_path.exists() and meta_path.exists():
        print(f"loading cached activations from {acts_path}")
        raw = torch.load(acts_path, weights_only=True)
        return {int(k): v.numpy() for k, v in raw.items()}

    print(f"collecting activations at all {engine.n_layers} layers...")
    prompts = [e["prompt"] for e in entries]
    all_layers = list(range(engine.n_layers))

    per_prompt = []
    for i, prompt_text in enumerate(prompts):
        conversation = [{"role": "user", "content": prompt_text}]
        acts = engine.get_last_token_activations(conversation, layers=all_layers)
        per_prompt.append(acts)
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(prompts)}")

    # stack: per_prompt is List[Dict[int, Tensor[d_model]]]
    # -> activations[layer] = np.array [N, d_model]
    activations = {}
    for layer_idx in all_layers:
        stacked = torch.stack([pp[layer_idx] for pp in per_prompt])
        activations[layer_idx] = stacked.numpy()

    # save cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({k: torch.from_numpy(v) for k, v in activations.items()}, acts_path)
    metadata = {
        "prompts": [e["prompt"] for e in entries],
        "languages": [e["language"] for e in entries],
        "n_prompts": len(entries),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"saved activations ({len(prompts)} prompts × {engine.n_layers} layers) to {acts_path}")
    return activations


def run_layer_stats(activations: Dict[int, np.ndarray], output_dir: Path) -> None:
    print("\n--- layer stats ---")
    stats = compute_layer_stats(activations)

    stats_path = output_dir / "layer_stats.json"
    # json keys must be strings
    with open(stats_path, "w") as f:
        json.dump({str(k): v for k, v in stats.items()}, f, indent=2)
    print(f"saved to {stats_path}")

    plot_path = output_dir / "layer_stats.png"
    plot_layer_stats(stats, str(plot_path))


def run_probes(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    languages: np.ndarray,
    output_dir: Path,
) -> None:
    print("\n--- probes: per-layer ---")
    per_layer = train_probe_per_layer(activations, labels)

    best_layer = max(per_layer, key=lambda l: per_layer[l]["mean_acc"])
    print(f"best layer (overall): {best_layer} ({per_layer[best_layer]['mean_acc']:.3f})")

    print("\n--- probes: per-language ---")
    per_language = train_probe_per_language(activations, labels, languages)

    print("\n--- probes: cross-lingual transfer ---")
    transfer = cross_lingual_transfer(activations, labels, languages, best_layer)

    results = {
        "best_layer": best_layer,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "per_language": per_language,
        "cross_lingual_transfer": transfer,
    }

    results_path = output_dir / "probe_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved to {results_path}")

    plot_path = output_dir / "probe_accuracy.png"
    plot_probe_accuracy(per_layer, per_language, str(plot_path))


def main():
    args = parse_args()

    languages = args.lang.split(",") if args.lang else None
    entries = load_predictions(args.predictions_path, languages=languages, take=args.take)
    if not entries:
        print("no predictions found — run experiment 0.1 first")
        return

    labels = label_predictions(entries)
    lang_array = np.array([e["language"] for e in entries])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / ".cache"

    # load model and collect activations
    model_config = ModelConfig(model=args.model)
    inference_config = InferenceConfig()
    engine = InferenceEngine(model_config, inference_config)
    activations = collect_activations(engine, entries, cache_dir)

    # free model memory before analysis
    del engine
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # analysis
    run_layer_stats(activations, output_dir)
    run_probes(activations, labels, lang_array, output_dir)

    print(f"\nall results saved to {output_dir}")


if __name__ == "__main__":
    main()
