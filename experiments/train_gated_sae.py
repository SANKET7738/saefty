import argparse
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from saefty.models.infer import InferenceEngine, ModelConfig, InferenceConfig
from saefty.models.sae.gated import GatedSAE, GatedConfig
from saefty.models.sae.activation_store import ActivationStore, ActivationStoreConfig
from saefty.models.sae.trainer import SAETrainer, TrainerConfig
from saefty.models.sae.evaluate import evaluate_sae, evaluate_ce_recovered


def parse_args():
    parser = argparse.ArgumentParser(description="Train Gated SAE on model activations")
    parser.add_argument("--model", type=str, default="CohereLabs/tiny-aya-global")
    parser.add_argument("--hook-layer", type=int, default=20)
    parser.add_argument("--expansion-factor", type=int, default=8)
    parser.add_argument("--training-tokens", type=int, default=5_000_000)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--buffer-size", type=int, default=262144)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--l1-coefficient", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lang", type=str, default="english,standard_arabic,hindi")
    parser.add_argument("--dataset", type=str, default="CohereLabs/aya_collection_language_split")
    parser.add_argument("--output-dir", type=str, default="results/train_sae/gated")
    parser.add_argument("--eval-texts", type=int, default=50, help="number of texts for CE evaluation")
    parser.add_argument("--checkpoint-every", type=int, default=5000, help="save checkpoint every N steps")
    parser.add_argument("--hf-repo", type=str, default=None, help="HuggingFace repo id (e.g. username/saefty)")
    parser.add_argument("--hf-model-name", type=str, default=None, help="subdirectory name in HF repo (e.g. gated-50m-9lang)")
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


def save_training_csv(history, output_dir):
    """Save training history as CSV for easy inspection."""
    if not history:
        return
    csv_path = Path(output_dir) / "training_log.csv"
    keys = list(history[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    print(f"training CSV saved: {csv_path}")


def plot_training_metrics(history, output_dir):
    """Generate training metric plots as PNGs."""
    if not history:
        return

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    steps = [h["step"] for h in history]

    # loss curves
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, [h["loss"] for h in history], label="total loss", alpha=0.8)
    if "reconstruction_loss" in history[0]:
        ax.plot(steps, [h["reconstruction_loss"] for h in history],
                label="reconstruction", alpha=0.8)
    if "l1_loss" in history[0]:
        ax.plot(steps, [h["l1_loss"] for h in history], label="L1", alpha=0.8)
    if "aux_loss" in history[0]:
        ax.plot(steps, [h["aux_loss"] for h in history], label="aux", alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # L0 curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, [h["l0"] for h in history], color="tab:blue", alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("L0 (avg active features)")
    ax.set_title("L0 Over Training")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "l0_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # dead features curve
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(steps, [h["dead_fraction"] for h in history], color="tab:red", alpha=0.8)
    ax.set_xlabel("step")
    ax.set_ylabel("dead feature fraction")
    ax.set_title("Dead Features Over Training")
    ax.grid(True, alpha=0.3)
    fig.savefig(plots_dir / "dead_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"training plots saved: {plots_dir}/")


def plot_language_balance(lang_counts, output_dir):
    """Generate per-language token distribution bar chart."""
    if not lang_counts:
        return

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    langs = sorted(lang_counts.keys(), key=lambda k: -lang_counts[k])
    counts = [lang_counts[l] for l in langs]
    total = sum(counts)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(langs, counts, color="tab:green", alpha=0.8)
    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{count:,}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("language")
    ax.set_ylabel("tokens")
    ax.set_title("Per-Language Token Distribution")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, alpha=0.3, axis="y")
    fig.savefig(plots_dir / "language_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"language balance plot saved: {plots_dir / 'language_balance.png'}")


def generate_model_card(config_dump, metrics, lang_counts):
    """Generate a HuggingFace model card README."""
    langs = ", ".join(config_dump.get("languages", []))
    card = f"""---
tags:
  - sparse-autoencoder
  - safety
  - multilingual
  - mechanistic-interpretability
license: mit
---

# Gated SAE — {config_dump.get('model', 'unknown')}

Sparse Autoencoder trained on multilingual activations for cross-lingual safety analysis.

## Architecture
- **Type**: Gated SAE (separate gate + magnitude pathways)
- **Base model**: `{config_dump.get('model', 'unknown')}`
- **Hook layer**: {config_dump.get('hook_layer', 'N/A')}
- **d_model**: {config_dump.get('d_model', 'N/A')}
- **Expansion factor**: {config_dump.get('expansion_factor', 'N/A')}x
- **d_sae**: {config_dump.get('d_sae', 'N/A')}
- **L1 coefficient**: {config_dump.get('l1_coefficient', 'N/A')}

## Training
- **Training tokens**: {config_dump.get('training_tokens', 0):,}
- **Languages**: {langs}
- **Dataset**: `{config_dump.get('dataset', 'unknown')}`
- **Batch size**: {config_dump.get('batch_size', 'N/A')}
- **Learning rate**: {config_dump.get('lr', 'N/A')}
- **Seed**: {config_dump.get('seed', 'N/A')}
- **Training time**: {config_dump.get('elapsed_human', 'N/A')}

## Metrics
| Metric | Value |
|--------|-------|
"""
    for k, v in metrics.items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        card += f"| {k} | {val} |\n"

    if lang_counts:
        total = sum(lang_counts.values())
        card += "\n## Language Balance\n| Language | Tokens | % |\n|----------|--------|---|\n"
        for lang in sorted(lang_counts, key=lambda k: -lang_counts[k]):
            count = lang_counts[lang]
            pct = count / total * 100
            card += f"| {lang} | {count:,} | {pct:.1f}% |\n"

    card += """
## Usage

```python
import torch
from saefty.models.sae.gated import GatedSAE, GatedConfig

config = GatedConfig(d_model=..., expansion_factor=..., l1_coefficient=...)
sae = GatedSAE(config)
sae.load_state_dict(torch.load("sae_final.pt", map_location="cpu"))
sae.eval()

# encode activations
features = sae.encode(activations)
# decode back
x_hat = sae.decode(features)
```
"""
    return card


def upload_to_huggingface(args, config_dump, metrics, lang_counts, output_dir):
    """Upload model artifacts to HuggingFace Hub."""
    if not args.hf_repo:
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: --hf-repo set but HF_TOKEN not found in environment. Skipping upload.")
        print("  Set HF_TOKEN in your .env file or environment to enable upload.")
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("WARNING: huggingface_hub not installed. Skipping upload.")
        print("  Install with: pip install huggingface_hub")
        return

    api = HfApi(token=token)
    output_path = Path(output_dir)
    model_name = args.hf_model_name or output_path.name

    # create repo if needed
    api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True)
    print(f"HF repo ready: {args.hf_repo}")

    # generate model card
    card = generate_model_card(config_dump, metrics, lang_counts)
    card_path = output_path / "model_card.md"
    with open(card_path, "w") as f:
        f.write(card)

    # upload files
    files_to_upload = [
        ("checkpoints/sae_final.pt", f"{model_name}/sae_final.pt"),
        ("config.json", f"{model_name}/config.json"),
        ("metrics.json", f"{model_name}/metrics.json"),
        ("model_card.md", f"{model_name}/README.md"),
    ]

    # add plots if they exist
    plots_dir = output_path / "plots"
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*.png"):
            files_to_upload.append(
                (f"plots/{plot_file.name}", f"{model_name}/plots/{plot_file.name}")
            )

    for local_name, remote_name in files_to_upload:
        local_path = output_path / local_name
        if local_path.exists():
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=args.hf_repo,
                repo_type="model",
            )
            print(f"  uploaded: {remote_name}")
        else:
            print(f"  skipped (not found): {local_name}")

    print(f"HF upload complete: https://huggingface.co/{args.hf_repo}/tree/main/{model_name}")


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
    sae_config = GatedConfig(
        d_model=engine.d_model,
        expansion_factor=args.expansion_factor,
        l1_coefficient=args.l1_coefficient,
        seed=args.seed,
    )
    sae = GatedSAE(sae_config)
    print(f"SAE: {sae.d_model} → {sae.d_sae} features (gated), "
          f"params={sum(p.numel() for p in sae.parameters()):,}")

    # 4. train
    trainer_config = TrainerConfig(
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_every=args.checkpoint_every,
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
    for text, _lang in text_iter:
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

    # save training log as CSV
    save_training_csv(history, args.output_dir)

    # save language balance
    lang_counts = store.language_token_counts
    lang_balance_path = Path(args.output_dir) / "language_balance.json"
    with open(lang_balance_path, "w") as f:
        json.dump(lang_counts, f, indent=2)
    print(f"language balance saved: {lang_balance_path}")

    # generate plots
    plot_training_metrics(history, args.output_dir)
    plot_language_balance(lang_counts, args.output_dir)

    # save config for reproducibility
    config_dump = {
        "model": args.model,
        "hook_layer": args.hook_layer,
        "expansion_factor": args.expansion_factor,
        "d_model": engine.d_model,
        "d_sae": sae.d_sae,
        "training_tokens": args.training_tokens,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "l1_coefficient": args.l1_coefficient,
        "checkpoint_every": args.checkpoint_every,
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

    # 8. upload to HuggingFace
    upload_to_huggingface(args, config_dump, all_metrics, lang_counts, args.output_dir)

    print(f"done! total time: {format_duration(elapsed)}")


if __name__ == "__main__":
    main()
