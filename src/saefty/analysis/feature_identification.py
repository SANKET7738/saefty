#!/usr/bin/env python3
"""
Identify candidate safety features by computing per-feature selectivity scores
(harmful − benign mean activation) over the XSafety benchmark.

Usage:
    python -m saefty.analysis.feature_identification
    # or
    python src/saefty/analysis/feature_identification.py
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_NAME = "CohereLabs/tiny-aya-global"
CHECKPOINT_PATH = "results/train_sae/gated_50m_9lang/checkpoints/sae_final.pt"
HOOK_LAYER = 20
LANGUAGES = [
    "english", "standard_arabic", "german", "spanish",
    "french", "hindi", "japanese", "bengali", "simplified_chinese",
]
TOP_K = 50
BATCH_SIZE = 8
XSAFETY_DATA_DIR = "data/xsafety/"
OUTPUT_PATH = "results/features_ranked.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Map training language names → xsafety 2-letter directory codes
LANG_MAP = {
    "english": "en",
    "standard_arabic": "ar",
    "german": "de",
    "spanish": "sp",
    "french": "fr",
    "hindi": "hi",
    "japanese": "ja",
    "bengali": "bn",
    "simplified_chinese": "zh",
}


# ── DATA LOADING ────────────────────────────────────────────────────────────

def _read_csv_prompts(path: Path) -> List[str]:
    """Read a plain-text CSV (one prompt per line, no header)."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def _read_xlsx_prompts(path: Path) -> List[str]:
    """Read prompts from first column of an xlsx file."""
    import openpyxl
    wb = openpyxl.load_workbook(str(path), read_only=True)
    ws = wb.active
    prompts = []
    for row in ws.iter_rows(values_only=True):
        val = row[0]
        if val and str(val).strip():
            prompts.append(str(val).strip())
    wb.close()
    return prompts


def load_xsafety_data(
    data_dir: str, languages: List[str]
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Load XSafety prompts per language.

    Returns:
        harmful: {lang: [prompt, ...]}
        benign:  {lang: [prompt, ...]}
    """
    base = Path(data_dir)
    harmful: Dict[str, List[str]] = {}
    benign: Dict[str, List[str]] = {}

    for lang in languages:
        code = LANG_MAP[lang]
        lang_dir = base / code
        if not lang_dir.exists():
            print(f"WARNING: xsafety dir not found for {lang} ({code}), skipping")
            continue

        # ── benign: commonsense.csv (fallback commen_sense.csv) ──
        benign_path = lang_dir / "commonsense.csv"
        if not benign_path.exists():
            benign_path = lang_dir / "commen_sense.csv"
        if benign_path.exists():
            benign[lang] = _read_csv_prompts(benign_path)
        else:
            print(f"WARNING: no commonsense file for {lang}, using empty benign set")
            benign[lang] = []

        # ── harmful: all other files (csv + xlsx), excluding commonsense ──
        harmful_prompts = []
        for f in sorted(lang_dir.iterdir()):
            if f.name.lower().startswith("commonsense") or f.name.lower().startswith("commen_sense"):
                continue
            if f.suffix == ".csv":
                harmful_prompts.extend(_read_csv_prompts(f))
            elif f.suffix == ".xlsx":
                harmful_prompts.extend(_read_xlsx_prompts(f))

        # deduplicate while preserving order
        seen = set()
        deduped = []
        for p in harmful_prompts:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        harmful[lang] = deduped

    # log counts
    print("\n── XSafety data loaded ──")
    print(f"{'language':<25} {'harmful':>8} {'benign':>8}")
    print("-" * 45)
    for lang in languages:
        nh = len(harmful.get(lang, []))
        nb = len(benign.get(lang, []))
        print(f"{lang:<25} {nh:>8} {nb:>8}")
    print()

    return harmful, benign


# ── FORWARD PASS ────────────────────────────────────────────────────────────

def get_activations(
    prompts: List[str],
    model,
    tokenizer,
    hook_layer: int,
    device: str,
    batch_size: int,
    label: str = "",
) -> torch.Tensor:
    """
    Run prompts through model and capture mean-pooled residual stream at hook_layer.

    Returns: (N, d_model) tensor
    """
    all_activations = []
    captured = {}
    skipped = 0

    def hook_fn(module, input, output):
        # some layers return tuples
        act = output[0] if isinstance(output, tuple) else output
        captured["act"] = act.detach()

    handle = model.model.layers[hook_layer].register_forward_hook(hook_fn)

    try:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            try:
                tokens = tokenizer(
                    batch,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    model(**tokens)

                act = captured["act"]  # (batch, seq_len, d_model)
                mask = tokens["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)

                # mean-pool over seq_len, excluding padding
                pooled = (act * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                all_activations.append(pooled.cpu())

            except Exception as e:
                skipped += len(batch)
                warnings.warn(f"Batch {i} failed ({label}): {e}")

            if (i + batch_size) % 50 < batch_size or i + batch_size >= len(prompts):
                done = min(i + batch_size, len(prompts))
                print(f"  [{label}] {done}/{len(prompts)} prompts processed", end="\r")

        print(f"  [{label}] {len(prompts)}/{len(prompts)} done, {skipped} skipped")
    finally:
        handle.remove()

    if not all_activations:
        return torch.zeros(0, model.config.hidden_size)
    return torch.cat(all_activations, dim=0)


# ── SAE LOADING ─────────────────────────────────────────────────────────────

def load_sae(checkpoint_path: str, device: str):
    """Load GatedSAE from checkpoint."""
    from saefty.models.sae.gated import GatedSAE, GatedConfig

    config = GatedConfig(d_model=2048, expansion_factor=8)
    sae = GatedSAE(config)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # handle wrapped format
    if isinstance(state, dict) and "sae_state_dict" in state:
        state = state["sae_state_dict"]

    sae.load_state_dict(state)
    sae.to(device)
    sae.eval()
    print(f"SAE loaded: {checkpoint_path} (d_sae={sae.d_sae}, device={device})")
    return sae


# ── ENCODE WITH SAE ─────────────────────────────────────────────────────────

def encode_with_sae(activations: torch.Tensor, sae, batch_size: int = 512) -> torch.Tensor:
    """Encode activations through SAE in batches. Returns (N, d_sae)."""
    if activations.shape[0] == 0:
        return torch.zeros(0, sae.d_sae)

    device = sae.device
    all_features = []

    with torch.no_grad():
        for i in range(0, activations.shape[0], batch_size):
            batch = activations[i : i + batch_size].to(device)
            features = sae.encode(batch)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


# ── SELECTIVITY ─────────────────────────────────────────────────────────────

def compute_selectivity(
    harmful_features: Dict[str, torch.Tensor],
    benign_features: Dict[str, torch.Tensor],
    languages: List[str],
    top_k: int,
) -> Tuple[List[Dict], Dict[str, np.ndarray]]:
    """
    Compute per-feature selectivity = mean_harmful - mean_benign for each language.
    Rank by English selectivity, return top_k features.
    """
    selectivity = {}
    mean_harmful = {}
    mean_benign = {}

    for lang in languages:
        if lang in harmful_features and harmful_features[lang].shape[0] > 0:
            mean_harmful[lang] = harmful_features[lang].mean(dim=0).numpy()
        else:
            mean_harmful[lang] = np.zeros(harmful_features[languages[0]].shape[1])

        if lang in benign_features and benign_features[lang].shape[0] > 0:
            mean_benign[lang] = benign_features[lang].mean(dim=0).numpy()
        else:
            mean_benign[lang] = np.zeros_like(mean_harmful[lang])

        selectivity[lang] = mean_harmful[lang] - mean_benign[lang]

    # rank by English selectivity descending
    en_sel = selectivity["english"]
    top_indices = np.argsort(en_sel)[::-1][:top_k]

    features_list = []
    for rank, idx in enumerate(top_indices, 1):
        entry = {
            "rank": rank,
            "feature_id": int(idx),
        }
        for lang in languages:
            key = f"selectivity_{lang}"
            entry[key] = round(float(selectivity[lang][idx]), 6)
        entry["mean_activation_harmful_english"] = round(float(mean_harmful["english"][idx]), 6)
        entry["mean_activation_benign_english"] = round(float(mean_benign["english"][idx]), 6)
        features_list.append(entry)

    return features_list, selectivity


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # ── load data ──
    harmful, benign = load_xsafety_data(XSAFETY_DATA_DIR, LANGUAGES)

    # ── load model ──
    print(f"Loading model: {MODEL_NAME} ...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    # ── load SAE ──
    sae = load_sae(CHECKPOINT_PATH, DEVICE)

    # ── collect activations per language per split ──
    harmful_features: Dict[str, torch.Tensor] = {}
    benign_features: Dict[str, torch.Tensor] = {}

    for lang in LANGUAGES:
        print(f"\n── {lang} ──")

        if lang in harmful and harmful[lang]:
            print(f"  harmful: {len(harmful[lang])} prompts")
            h_act = get_activations(
                harmful[lang], model, tokenizer, HOOK_LAYER, DEVICE, BATCH_SIZE,
                label=f"{lang}/harmful",
            )
            harmful_features[lang] = encode_with_sae(h_act, sae)
        else:
            print(f"  harmful: 0 prompts (skipping)")
            harmful_features[lang] = torch.zeros(0, sae.d_sae)

        if lang in benign and benign[lang]:
            print(f"  benign: {len(benign[lang])} prompts")
            b_act = get_activations(
                benign[lang], model, tokenizer, HOOK_LAYER, DEVICE, BATCH_SIZE,
                label=f"{lang}/benign",
            )
            benign_features[lang] = encode_with_sae(b_act, sae)
        else:
            print(f"  benign: 0 prompts (skipping)")
            benign_features[lang] = torch.zeros(0, sae.d_sae)

    # ── compute selectivity ──
    print("\n── Computing selectivity ──")
    features_list, selectivity = compute_selectivity(
        harmful_features, benign_features, LANGUAGES, TOP_K
    )

    # ── save results ──
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_harmful = {lang: len(harmful.get(lang, [])) for lang in LANGUAGES}
    n_benign = {lang: len(benign.get(lang, [])) for lang in LANGUAGES}

    result = {
        "metadata": {
            "model": MODEL_NAME,
            "hook_layer": HOOK_LAYER,
            "checkpoint": CHECKPOINT_PATH,
            "n_harmful_per_lang": n_harmful,
            "n_benign_per_lang": n_benign,
            "top_k": TOP_K,
        },
        "features": features_list,
    }

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # ── summary table ──
    abbrev = {
        "english": "en", "standard_arabic": "ar", "hindi": "hi",
        "german": "de", "spanish": "es", "french": "fr",
        "japanese": "ja", "bengali": "bn", "simplified_chinese": "zh",
    }
    cols = ["en", "ar", "hi", "de", "es", "fr", "ja", "bn", "zh"]
    lang_for_col = {v: k for k, v in abbrev.items()}

    header = f"{'rank':>4} | {'feat_id':>8} | " + " | ".join(f"sel_{c:>2}" for c in cols)
    print(f"\n{'─' * len(header)}")
    print(header)
    print(f"{'─' * len(header)}")

    for feat in features_list[:20]:
        vals = []
        for c in cols:
            lang = lang_for_col[c]
            val = feat.get(f"selectivity_{lang}", 0.0)
            vals.append(f"{val:>6.3f}")
        print(f"{feat['rank']:>4} | {feat['feature_id']:>8} | " + " | ".join(vals))

    elapsed = time.time() - t0
    print(f"\n── Done ──")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Features ranked: {len(features_list)}")
    print(f"  Top feature selectivity_english: {features_list[0]['selectivity_english']:.4f}")
    for lang in LANGUAGES:
        nh = harmful_features[lang].shape[0]
        nb = benign_features[lang].shape[0]
        print(f"  {lang}: {nh} harmful + {nb} benign encoded")


if __name__ == "__main__":
    main()
