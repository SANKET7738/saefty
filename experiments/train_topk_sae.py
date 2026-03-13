import argparse
import json
import time
from pathlib import Path

from saefty.models.infer import InferenceEngine, ModelConfig, InferenceConfig
from saefty.models.sae.topk import TopKSAE, TopKConfig
from saefty.models.sae.activation_store import ActivationStore, ActivationStoreConfig
from saefty.models.sae.trainer import SAETrainer, TrainerConfig
from saefty.models.sae.evaluate import evaluate_sae, evaluate_ce_recovered


def parse_args():
    parser = argparse.ArgumentParser(description="Train TopK SAE on model activations")
    parser.add_argument("--model", type=str, default="CohereLabs/tiny-aya-global")
    parser.add_argument("--hook-layer", type=int, default=20)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--training-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--buffer-size", type=int, default=262144)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--k", type=int, default=32, help="number of active features per input")
    parser.add_argument("--auxk-alpha", type=float, default=0.0, help="auxiliary dead latent loss weight (0=off, 1/16=standard)")
    parser.add_argument("--auxk", type=int, default=256, help="number of dead features to activate in aux pass")
    parser.add_argument("--dead-steps-threshold", type=int, default=10, help="steps without firing = dead")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", type=str, default="english,standard_arabic,hindi")
    parser.add_argument("--dataset", type=str, default="CohereLabs/aya_collection_language_split")
    parser.add_argument("--output-dir", type=str, default="results/train_sae/topk")
    parser.add_argument("--eval-texts", type=int, default=50, help="number of texts for CE evaluation")
    return parser.parse_args()


def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    elif m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def main():
    args = parse_args()
    languages = [l.strip() for l in args.lang.split(",")]
    t_start = time.time()

    # 1. load model
    engine = InferenceEngine(
        ModelConfig(model=args.model),
        InferenceConfig(),
    )

    # 2. create activation store
    store_config = ActivationStoreConfig(
        dataset=args.dataset,
        languages=languages,
        hook_layer=args.hook_layer,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        total_tokens=args.training_tokens,
        seed=args.seed,
    )
    store = ActivationStore(engine, store_config)

    # 3. create SAE
    sae_config = TopKConfig(
        d_model=engine.d_model,
        expansion_factor=args.expansion_factor,
        k=args.k,
        auxk_alpha=args.auxk_alpha,
        auxk=args.auxk,
        dead_steps_threshold=args.dead_steps_threshold,
        seed=args.seed,
    )
    sae = TopKSAE(sae_config)
    print(f"SAE: {sae.d_model} → {sae.d_sae} features (k={args.k}), "
          f"params={sum(p.numel() for p in sae.parameters()):,}")

    # 4. train
    trainer_config = TrainerConfig(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
    )
    trainer = SAETrainer(sae, store, trainer_config)
    history = trainer.train()

    # 5. evaluate on held-out activations
    print("collecting eval activations...")
    eval_store = ActivationStore(engine, ActivationStoreConfig(
        dataset=args.dataset,
        languages=languages,
        hook_layer=args.hook_layer,
        batch_size=args.batch_size,
        buffer_size=args.batch_size,
        total_tokens=args.batch_size,
        seed=args.seed + 1,
    ))
    eval_batch = next(iter(eval_store))
    sae_metrics = evaluate_sae(sae, eval_batch)
    print("SAE metrics:")
    for k, v in sae_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 6. CE recovered
    print("evaluating CE recovered...")
    eval_texts = []
    text_iter = eval_store._make_text_iterator()
    for text in text_iter:
        eval_texts.append(text)
        if len(eval_texts) >= args.eval_texts:
            break
    ce_metrics = evaluate_ce_recovered(sae, engine, eval_texts, args.hook_layer)
    print("CE metrics:")
    for k, v in ce_metrics.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # 7. save everything
    all_metrics = {**sae_metrics, **ce_metrics}
    trainer.save_results(extra_metrics=all_metrics)

    # save config for reproducibility
    config_dump = {
        "model": args.model,
        "hook_layer": args.hook_layer,
        "expansion_factor": args.expansion_factor,
        "training_tokens": args.training_tokens,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "k": args.k,
        "auxk_alpha": args.auxk_alpha,
        "auxk": args.auxk,
        "dead_steps_threshold": args.dead_steps_threshold,
        "seed": args.seed,
        "languages": languages,
        "dataset": args.dataset,
    }
    elapsed = time.time() - t_start
    config_dump["elapsed_seconds"] = round(elapsed, 1)
    config_dump["elapsed_human"] = format_duration(elapsed)

    config_path = Path(args.output_dir) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config_dump, f, indent=2)
    print(f"config saved: {config_path}")
    print(f"done! total time: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
