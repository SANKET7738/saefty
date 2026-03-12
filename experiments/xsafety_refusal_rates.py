import argparse
import json
from pathlib import Path
from typing import List, Dict

from saefty.models.infer import InferenceEngine, ModelConfig, InferenceConfig
from saefty.data.xsafety import XSafetyLoader, XSafetyConfig
from saefty.data.base import Benchmark
from saefty.eval.refusal import RefusalEvaluator
from saefty.eval.metrics import Metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lang", type=str, default=None, help="comma-separated languages, e.g. en,hi,ar")
    parser.add_argument("--take", type=int, default=None, help="first N prompts per language")
    parser.add_argument("--output-dir", type=str, default="results/xsafety_refusal_rates")
    return parser.parse_args()


def load_existing_predictions(predictions_path: Path) -> Dict[str, Dict[int, str]]:
    existing: Dict[str, Dict[int, str]] = {}
    if not predictions_path.exists():
        return existing
    
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            lang = entry["language"]
            idx = entry["index"]
            if lang not in existing:
                existing[lang] = {}
            existing[lang][idx] = entry["response"]
    
    return existing


def append_prediction(predictions_path: Path, entry: Dict) -> None:
    with open(predictions_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_predictions(
    engine: InferenceEngine,
    benchmark: Benchmark,
    predictions_path: Path,
) -> Dict[str, List[str]]:
    existing = load_existing_predictions(predictions_path)
    
    total_skipped = sum(len(v) for v in existing.values())
    if total_skipped > 0:
        print(f"resuming: found {total_skipped} existing predictions")
    
    predictions: Dict[str, List[str]] = {}
    
    for lang, split in benchmark.splits.items():
        lang_existing = existing.get(lang, {})
        lang_preds: List[str] = []
        
        for i, prompt in enumerate(split.prompts):
            if i in lang_existing:
                lang_preds.append(lang_existing[i])
                continue
            
            response = engine.infer(prompt)[0]
            lang_preds.append(response)
            
            entry = {
                "language": lang,
                "index": i,
                "prompt": prompt[0]["content"],
                "response": response,
            }
            append_prediction(predictions_path, entry)
            
            if (i + 1) % 10 == 0:
                print(f"  [{lang}] {i + 1}/{len(split.prompts)}")
        
        predictions[lang] = lang_preds
        print(f"  [{lang}] done — {len(lang_preds)} predictions")
    
    return predictions


def run_eval(
    benchmark: Benchmark,
    predictions: Dict[str, List[str]],
    eval_path: Path,
) -> None:
    evaluator = RefusalEvaluator()
    metrics = Metrics(evaluator)
    result = metrics.compute_benchmark(benchmark, predictions)
    
    output = {
        "per_split": result.per_split,
        "overall": result.overall,
    }
    
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*50}")
    print(f"overall refusal rate: {result.overall['rate']:.2%}")
    print(f"{'='*50}")
    for lang, m in result.per_split.items():
        print(f"  {lang}: {m['rate']:.2%} ({m['count']}/{m['total']})")
    print()


def main():
    args = parse_args()
    
    languages = args.lang.split(",") if args.lang else None
    
    # load benchmark
    xsafety_config = XSafetyConfig(
        languages=languages,
        take_per_language=args.take,
    )
    benchmark = XSafetyLoader(xsafety_config).load()
    
    # load model
    model_config = ModelConfig(model=args.model)
    inference_config = InferenceConfig()
    engine = InferenceEngine(model_config, inference_config)
    
    # setup output dirs
    output_dir = Path(args.output_dir)
    predictions_dir = output_dir / "predictions"
    eval_dir = output_dir / "eval"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    predictions_path = predictions_dir / "predictions.jsonl"
    eval_path = eval_dir / "evals.json"
    
    # run predictions (with resume)
    print(f"\n--- running predictions ---")
    predictions = run_predictions(engine, benchmark, predictions_path)
    
    # run eval
    print(f"\n--- running evaluation ---")
    run_eval(benchmark, predictions, eval_path)
    
    print(f"results saved to {output_dir}")


if __name__ == "__main__":
    main()
