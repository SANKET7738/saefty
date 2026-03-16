#!/usr/bin/env python3
"""
Find top activating prompts for each candidate safety feature and save
a human-readable interpretability report.

Single forward pass over all ~23K harmful prompts → store activations
for only the 50 candidate features → rank → save.

Usage:
    python src/saefty/analysis/interpret_features.py
"""

import json
import time
import warnings
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_NAME = "CohereLabs/tiny-aya-global"
CHECKPOINT_PATH = "results/train_sae/gated_50m_9lang/checkpoints/sae_final.pt"
HOOK_LAYER = 20
FEATURES_RANKED_PATH = "results/features_ranked.json"
XSAFETY_DATA_DIR = "data/xsafety/"
TOP_N_PROMPTS = 20
BATCH_SIZE = 8
OUTPUT_JSON = "results/interpretability/top_activations.json"
OUTPUT_TXT = "results/interpretability/interpretability_report.txt"
LANGUAGES = [
    "english", "standard_arabic", "german", "spanish",
    "french", "hindi", "japanese", "bengali", "simplified_chinese",
]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip()]


def _read_xlsx_prompts(path: Path) -> List[str]:
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


def load_harmful_prompts(data_dir: str, languages: List[str]) -> List[Dict]:
    """Load all harmful prompts across all languages. Returns list of {text, language}."""
    base = Path(data_dir)
    all_prompts = []

    for lang in languages:
        code = LANG_MAP[lang]
        lang_dir = base / code
        if not lang_dir.exists():
            print(f"WARNING: xsafety dir not found for {lang} ({code}), skipping")
            continue

        harmful = []
        for f in sorted(lang_dir.iterdir()):
            if f.name.lower().startswith("commonsense") or f.name.lower().startswith("commen_sense"):
                continue
            if f.suffix == ".csv":
                harmful.extend(_read_csv_prompts(f))
            elif f.suffix == ".xlsx":
                harmful.extend(_read_xlsx_prompts(f))

        # deduplicate
        seen = set()
        for p in harmful:
            if p not in seen:
                seen.add(p)
                all_prompts.append({"text": p, "language": lang})

    print(f"Loaded {len(all_prompts)} harmful prompts across {len(languages)} languages")
    per_lang = {}
    for p in all_prompts:
        per_lang[p["language"]] = per_lang.get(p["language"], 0) + 1
    for lang in languages:
        print(f"  {lang}: {per_lang.get(lang, 0)}")

    return all_prompts


# ── SAE LOADING ─────────────────────────────────────────────────────────────

def load_sae(checkpoint_path: str, device: str):
    from saefty.models.sae.gated import GatedSAE, GatedConfig

    config = GatedConfig(d_model=2048, expansion_factor=8)
    sae = GatedSAE(config)

    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(state, dict) and "sae_state_dict" in state:
        state = state["sae_state_dict"]

    sae.load_state_dict(state)
    sae.to(device)
    sae.eval()
    print(f"SAE loaded: {checkpoint_path} (d_sae={sae.d_sae})")
    return sae


# ── FORWARD PASS ────────────────────────────────────────────────────────────

def collect_feature_activations(
    prompts: List[Dict],
    feature_ids: List[int],
    model,
    tokenizer,
    sae,
    hook_layer: int,
    device: str,
    batch_size: int,
) -> np.ndarray:
    """
    Single forward pass over all prompts. For each prompt, store activations
    for only the candidate feature_ids.

    Returns: (n_prompts, n_features) numpy array
    """
    n_prompts = len(prompts)
    n_features = len(feature_ids)
    feature_idx = torch.tensor(feature_ids, dtype=torch.long, device=device)

    # pre-allocate result array
    result = np.zeros((n_prompts, n_features), dtype=np.float32)
    captured = {}
    skipped = 0

    def hook_fn(module, input, output):
        act = output[0] if isinstance(output, tuple) else output
        captured["act"] = act.detach()

    handle = model.model.layers[hook_layer].register_forward_hook(hook_fn)

    try:
        for i in range(0, n_prompts, batch_size):
            batch_texts = [p["text"] for p in prompts[i : i + batch_size]]
            try:
                tokens = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    max_length=256,
                    truncation=True,
                    padding=True,
                ).to(device)

                with torch.no_grad():
                    model(**tokens)

                act = captured["act"]  # (batch, seq_len, d_model)
                mask = tokens["attention_mask"].unsqueeze(-1).float()
                pooled = (act * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, d_model)

                # encode through SAE
                with torch.no_grad():
                    features = sae.encode(pooled)  # (batch, d_sae)

                # extract only candidate features
                selected = features[:, feature_idx].cpu().numpy()  # (batch, n_features)
                actual_batch = selected.shape[0]
                result[i : i + actual_batch] = selected

            except Exception as e:
                actual_batch = len(batch_texts)
                skipped += actual_batch
                warnings.warn(f"Batch {i} failed: {e}")

            done = min(i + batch_size, n_prompts)
            if done % 200 < batch_size or done >= n_prompts:
                print(f"  {done}/{n_prompts} prompts processed ({skipped} skipped)", end="\r")

        print(f"  {n_prompts}/{n_prompts} done, {skipped} skipped              ")
    finally:
        handle.remove()

    return result


# ── TOP PROMPTS PER FEATURE ─────────────────────────────────────────────────

def find_top_prompts(
    prompts: List[Dict],
    activations: np.ndarray,
    feature_ids: List[int],
    top_n: int,
) -> Dict[int, List[Dict]]:
    """For each feature, find top_n highest-activating prompts."""
    top_prompts = {}
    n_features = len(feature_ids)

    for fi in range(n_features):
        fid = feature_ids[fi]
        col = activations[:, fi]
        top_idx = np.argsort(col)[::-1][:top_n]

        entries = []
        for pos, idx in enumerate(top_idx, 1):
            entries.append({
                "position": pos,
                "language": prompts[idx]["language"],
                "activation": round(float(col[idx]), 4),
                "text": prompts[idx]["text"],
            })
        top_prompts[fid] = entries

        if (fi + 1) % 10 == 0:
            print(f"  {fi + 1}/{n_features} features ranked")

    return top_prompts


# ── OUTPUT ──────────────────────────────────────────────────────────────────

def save_json(features_ranked: List[Dict], top_prompts: Dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output = {"features": []}
    for feat in features_ranked:
        fid = feat["feature_id"]
        entry = {
            "rank": feat["rank"],
            "feature_id": fid,
            "selectivity_english": feat.get("selectivity_english", 0),
            "top_prompts": top_prompts.get(fid, []),
        }
        output["features"].append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"JSON saved: {output_path}")


def save_txt(features_ranked: List[Dict], top_prompts: Dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    sel_keys = [f"selectivity_{lang}" for lang in LANGUAGES]
    abbrev = {
        "english": "en", "standard_arabic": "ar", "hindi": "hi",
        "german": "de", "spanish": "es", "french": "fr",
        "japanese": "ja", "bengali": "bn", "simplified_chinese": "zh",
    }

    lines = []
    for feat in features_ranked:
        fid = feat["feature_id"]
        rank = feat["rank"]

        # header line
        sel_en = feat.get("selectivity_english", 0)
        sel_ar = feat.get("selectivity_standard_arabic", 0)
        sel_hi = feat.get("selectivity_hindi", 0)
        lines.append("=" * 80)
        lines.append(
            f"FEATURE {fid} | rank {rank} | "
            f"sel_en={sel_en:.3f} | sel_ar={sel_ar:.3f} | sel_hi={sel_hi:.3f}"
        )

        # second line with remaining languages
        remaining = []
        for lang in ["german", "spanish", "french", "japanese", "bengali", "simplified_chinese"]:
            ab = abbrev[lang]
            val = feat.get(f"selectivity_{lang}", 0)
            remaining.append(f"sel_{ab}={val:.3f}")
        lines.append(" | ".join(remaining))
        lines.append("-" * 80)

        # top prompts
        entries = top_prompts.get(fid, [])
        for entry in entries:
            pos = entry["position"]
            lang = entry["language"]
            act = entry["activation"]
            text = entry["text"][:200]  # truncate very long prompts for readability
            lines.append(f"#{pos:02d} [{lang}, act={act:.2f}] {text}")

        lines.append("")

    lines.append("=" * 80)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"TXT report saved: {output_path}")


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # step 1: load candidate features
    print("── Loading candidate features ──")
    with open(FEATURES_RANKED_PATH) as f:
        ranked_data = json.load(f)
    features_ranked = ranked_data["features"]
    feature_ids = [feat["feature_id"] for feat in features_ranked]
    print(f"Loaded {len(feature_ids)} candidate features")

    # step 2: load all harmful prompts
    print("\n── Loading harmful prompts ──")
    prompts = load_harmful_prompts(XSAFETY_DATA_DIR, LANGUAGES)

    # step 3: load model + SAE
    print(f"\n── Loading model: {MODEL_NAME} ──")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16
    ).to(DEVICE)
    model.eval()
    print(f"Model loaded on {DEVICE}")

    sae = load_sae(CHECKPOINT_PATH, DEVICE)

    # step 4: single forward pass
    print("\n── Forward pass (single pass over all prompts) ──")
    activations = collect_feature_activations(
        prompts, feature_ids, model, tokenizer, sae,
        HOOK_LAYER, DEVICE, BATCH_SIZE,
    )
    print(f"Activations collected: {activations.shape}")

    # step 5: find top prompts per feature
    print("\n── Finding top prompts per feature ──")
    top_prompts = find_top_prompts(prompts, activations, feature_ids, TOP_N_PROMPTS)

    # step 6: save outputs
    print("\n── Saving outputs ──")
    save_json(features_ranked, top_prompts, OUTPUT_JSON)
    save_txt(features_ranked, top_prompts, OUTPUT_TXT)

    elapsed = time.time() - t0
    print(f"\n── Done ──")
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Total prompts processed: {len(prompts)}")
    print(f"  Features analyzed: {len(feature_ids)}")
    print(f"  Top prompts per feature: {TOP_N_PROMPTS}")
    print(f"  JSON: {OUTPUT_JSON}")
    print(f"  TXT:  {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
