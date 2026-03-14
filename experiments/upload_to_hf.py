#!/usr/bin/env python3
"""Standalone script to upload trained SAE artifacts to HuggingFace Hub.

Reads HF_TOKEN, HF_REPO, and HF_MODEL_NAME from .env file automatically.

Usage:
  python experiments/upload_to_hf.py \
    --output-dir results/train_sae/gated_50m_9lang

  # Override .env values via CLI:
  python experiments/upload_to_hf.py \
    --output-dir results/train_sae/gated_50m_9lang \
    --hf-repo YOUR_USERNAME/saefty \
    --hf-model-name gated-50m-9lang
"""
import argparse
import json
import os
from pathlib import Path


def load_dotenv(path=".env"):
    """Load key=value pairs from a .env file into os.environ."""
    env_path = Path(path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        os.environ.setdefault(key, value)


def generate_model_card(config, metrics, lang_counts):
    langs = ", ".join(sorted(lang_counts.keys()))
    total_tokens = sum(lang_counts.values())
    return f"""---
tags:
  - sparse-autoencoder
  - multilingual
  - safety
  - interpretability
language: [{langs}]
---

# Gated SAE — Multilingual Safety Features

Sparse autoencoder trained on **CohereLabs/tiny-aya-global** (layer 20) for cross-lingual safety representation analysis.

## Training Config

| Parameter | Value |
|-----------|-------|
| Architecture | Gated SAE |
| d_model | {config.get('d_model', 2048)} |
| d_sae | {config.get('d_sae', 16384)} |
| Expansion | {config.get('expansion_factor', 8)}x |
| L1 coefficient | {config.get('l1_coefficient', 5e-4)} |
| Training tokens | {total_tokens:,} |
| Languages | {len(lang_counts)} ({langs}) |

## Metrics

| Metric | Value |
|--------|-------|
| CE recovered | {metrics.get('ce_recovered', 0):.4f} |
| Cosine similarity | {metrics.get('cosine_similarity', 0):.4f} |
| Explained variance | {metrics.get('explained_variance', 0):.4f} |
| L0 (eval) | {metrics.get('l0', 0):.1f} |
| Dead features | {metrics.get('dead_fraction', 0):.4%} |
| Total steps | {metrics.get('total_steps', 'N/A')} |

## Language Balance

| Language | Tokens |
|----------|--------|
""" + "\n".join(f"| {lang} | {count:,} |" for lang, count in sorted(lang_counts.items())) + """

## Usage

```python
import torch
from saefty.models.sae.gated import GatedSAE

state = torch.load("sae_final.pt", map_location="cpu")
sae = GatedSAE(d_model=2048, d_sae=16384)
sae.load_state_dict(state["sae_state_dict"])
```
"""


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Upload SAE artifacts to HuggingFace Hub")
    parser.add_argument("--output-dir", required=True, help="Path to training output directory")
    parser.add_argument("--hf-repo", default=os.environ.get("HF_REPO"),
                        help="HuggingFace repo ID (default: HF_REPO from .env)")
    parser.add_argument("--hf-model-name", default=os.environ.get("HF_MODEL_NAME"),
                        help="Subdirectory name in repo (default: HF_MODEL_NAME from .env, or output dir name)")
    args = parser.parse_args()

    if not args.hf_repo:
        print("ERROR: --hf-repo not set and HF_REPO not found in .env")
        return 1

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not found in .env or environment.")
        return 1

    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: huggingface_hub not installed. Run: pip install huggingface_hub")
        return 1

    output_path = Path(args.output_dir)
    if not output_path.exists():
        print(f"ERROR: output directory not found: {output_path}")
        return 1

    model_name = args.hf_model_name or output_path.name

    # load existing artifacts
    config = json.loads((output_path / "config.json").read_text())
    metrics = json.loads((output_path / "metrics.json").read_text())
    lang_counts = json.loads((output_path / "language_balance.json").read_text())

    print(f"Uploading to {args.hf_repo}/{model_name}")
    print(f"  Config: {output_path / 'config.json'}")
    print(f"  Metrics: CE recovered={metrics.get('ce_recovered', 0):.4f}")
    print(f"  Languages: {len(lang_counts)}")

    api = HfApi(token=token)
    api.create_repo(repo_id=args.hf_repo, repo_type="model", exist_ok=True)
    print(f"HF repo ready: {args.hf_repo}")

    # generate model card
    card = generate_model_card(config, metrics, lang_counts)
    card_path = output_path / "model_card.md"
    card_path.write_text(card)

    # collect files to upload
    files_to_upload = [
        ("checkpoints/sae_final.pt", f"{model_name}/sae_final.pt"),
        ("config.json", f"{model_name}/config.json"),
        ("metrics.json", f"{model_name}/metrics.json"),
        ("training_log.json", f"{model_name}/training_log.json"),
        ("training_log.csv", f"{model_name}/training_log.csv"),
        ("language_balance.json", f"{model_name}/language_balance.json"),
        ("model_card.md", f"{model_name}/README.md"),
    ]

    # add plots
    plots_dir = output_path / "plots"
    if plots_dir.exists():
        for plot_file in plots_dir.glob("*.png"):
            files_to_upload.append(
                (f"plots/{plot_file.name}", f"{model_name}/plots/{plot_file.name}")
            )

    # upload
    uploaded = 0
    for local_name, remote_name in files_to_upload:
        local_path = output_path / local_name
        if local_path.exists():
            size_mb = local_path.stat().st_size / (1024 * 1024)
            print(f"  uploading: {remote_name} ({size_mb:.1f} MB)")
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_name,
                repo_id=args.hf_repo,
                repo_type="model",
            )
            uploaded += 1
        else:
            print(f"  skipped (not found): {local_name}")

    print(f"\nDone! Uploaded {uploaded} files.")
    print(f"View at: https://huggingface.co/{args.hf_repo}/tree/main/{model_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
