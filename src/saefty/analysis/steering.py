#!/usr/bin/env python3
"""
Feature steering experiments — causal validation of safety features.

Two steering methods:
  1. Vector addition: add α × normalized decoder vector to residual stream
  2. Feature clamping: encode → modify target feature → decode + error

For each target feature × language × method:
  - Baseline: no steering
  - SAE passthrough: encode→decode+error, no modification (sanity check)
  - Suppress: zero out feature on harmful prompts
  - Amplify: boost feature on benign prompts

Usage:
  python -m saefty.analysis.steering
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from saefty.analysis.feature_identification import (
    MODEL_NAME, CHECKPOINT_PATH, HOOK_LAYER, DEVICE, LANGUAGES, load_sae,
    load_xsafety_data, XSAFETY_DATA_DIR,
)

# ── CONFIG ─────────────────────────────────────────────────────────────────
N_PROMPTS = 30
MAX_NEW_TOKENS = 100
OUTPUT_DIR = "results/steering"

EXPERIMENTS = [
    # feature 5169 (deception) — safety-relevant in HI, DE, ES, FR, absent from EN
    {"feature_id": 5169, "languages": ["hindi", "english", "german", "simplified_chinese"]},
    # feature 6988 (first-person) — safety-relevant in EN, DE, ES, FR, absent from HI
    {"feature_id": 6988, "languages": ["english", "hindi", "french"]},
    # feature 4436 (desire/consent) — safety-relevant in HI, JA, BN
    {"feature_id": 4436, "languages": ["hindi", "english", "bengali"]},
]

ALPHA_VALUES = [0, 1, 3, 5, 10]


# ── steering hooks ────────────────────────────────────────────────────────
class VectorAdditionHook:
    """Add α × normalized decoder vector to layer output."""

    def __init__(self, direction: torch.Tensor, alpha: float):
        self.direction = F.normalize(direction, dim=-1)
        self.alpha = alpha

    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        h = h + self.alpha * self.direction.to(h.device, h.dtype)
        if isinstance(output, tuple):
            return (h,) + output[1:]
        return h


class FeatureClampHook:
    """Encode through SAE, modify target feature, decode + error."""

    def __init__(self, sae, feature_idx: int, clamp_value: float):
        self.sae = sae
        self.feature_idx = feature_idx
        self.clamp_value = clamp_value

    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        B, S, D = h.shape
        h_flat = h.reshape(-1, D).float()

        with torch.no_grad():
            features = self.sae.encode(h_flat)
            x_hat = self.sae.decode(features)
            error = h_flat - x_hat

            features[:, self.feature_idx] = self.clamp_value
            x_hat_modified = self.sae.decode(features)
            h_new = x_hat_modified + error

        h_new = h_new.reshape(B, S, D).to(h.dtype)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


# ── generation ────────────────────────────────────────────────────────────
def generate_with_hook(model, tokenizer, prompt: str, hook_fn=None) -> str:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    handle = None
    if hook_fn is not None:
        handle = model.model.layers[HOOK_LAYER].register_forward_hook(hook_fn)

    try:
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response
    finally:
        if handle is not None:
            handle.remove()


def get_max_activation(feature_id: int, language: str) -> float:
    """Get max activation for a feature from per-language results."""
    path = Path(f"results/max_act_per_lang/{language}/top_activations.json")
    if not path.exists():
        return 10.0  # fallback
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for feat in data["features"]:
        if feat["feature_id"] == feature_id and feat["top_examples"]:
            return feat["top_examples"][0]["activation"]
    return 10.0


# ── main ──────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    # Load model + SAE
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16,
    ).to(DEVICE).eval()

    print(f"Loading SAE: {CHECKPOINT_PATH}")
    sae = load_sae(CHECKPOINT_PATH, DEVICE)

    # Load xsafety prompts
    print("Loading XSafety prompts...")
    harmful, benign = load_xsafety_data(XSAFETY_DATA_DIR, list(LANGUAGES))

    # Output
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = {"experiments": []}

    for exp in EXPERIMENTS:
        fid = exp["feature_id"]
        decoder_vec = sae.W_dec[fid].detach().clone()

        for lang in exp["languages"]:
            harmful_prompts = harmful.get(lang, [])[:N_PROMPTS]
            benign_prompts = benign.get(lang, [])[:N_PROMPTS]

            if not harmful_prompts:
                print(f"\n  {lang}: no harmful prompts, skipping")
                continue

            max_act = get_max_activation(fid, lang)
            clamp_amplify = max_act * 10

            print(f"\n{'═' * 70}")
            print(f"  Feature {fid} × {lang}")
            print(f"  max_act={max_act:.1f}, clamp_amplify={clamp_amplify:.1f}")
            print(f"  {len(harmful_prompts)} harmful, {len(benign_prompts)} benign prompts")
            print(f"{'═' * 70}")

            conditions = [
                # (label, method, prompts, prompt_type, hook_factory)
                ("baseline", "none", harmful_prompts, "harmful", lambda: None),
                ("sae_passthrough", "clamping", harmful_prompts, "harmful",
                 lambda: FeatureClampHook(sae, fid, None)),  # passthrough marker
                ("suppress_vector", "vector_addition", harmful_prompts, "harmful",
                 lambda: VectorAdditionHook(decoder_vec, -5.0)),
                ("suppress_clamp", "clamping", harmful_prompts, "harmful",
                 lambda: FeatureClampHook(sae, fid, 0.0)),
                ("amplify_vector", "vector_addition", benign_prompts, "benign",
                 lambda: VectorAdditionHook(decoder_vec, 5.0)),
                ("amplify_clamp", "clamping", benign_prompts, "benign",
                 lambda: FeatureClampHook(sae, fid, clamp_amplify)),
            ]

            for cond_name, method, prompts, prompt_type, hook_factory in conditions:
                print(f"\n  {cond_name} ({method}, {prompt_type})...")

                generations = []
                for i, prompt in enumerate(prompts):
                    hook = hook_factory()

                    # SAE passthrough: encode→decode+error without modification
                    if cond_name == "sae_passthrough":
                        # Use clamping hook but read the original activation value
                        hook = _PassthroughHook(sae)

                    response = generate_with_hook(model, tokenizer, prompt, hook)
                    generations.append({
                        "prompt": prompt,
                        "response": response,
                        "prompt_type": prompt_type,
                    })

                    if (i + 1) % 10 == 0:
                        print(f"    {i+1}/{len(prompts)}")

                result_entry = {
                    "feature_id": fid,
                    "language": lang,
                    "method": method,
                    "condition": cond_name,
                    "prompt_type": prompt_type,
                    "n_prompts": len(prompts),
                    "generations": generations,
                }
                if method == "vector_addition":
                    result_entry["alpha"] = -5.0 if "suppress" in cond_name else 5.0
                if method == "clamping":
                    result_entry["clamp_value"] = 0.0 if "suppress" in cond_name else clamp_amplify

                all_results["experiments"].append(result_entry)

                # Print first 3 for inspection
                print(f"    Sample outputs ({cond_name}):")
                for g in generations[:3]:
                    print(f"      P: {g['prompt'][:60]}...")
                    print(f"      R: {g['response'][:80]}...")
                    print()

            # Incremental save after each feature×language
            with open(out_dir / "generations.json", "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Final save
    gen_path = out_dir / "generations.json"
    with open(gen_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    summary_path = out_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        for exp_result in all_results["experiments"]:
            fid = exp_result["feature_id"]
            lang = exp_result["language"]
            cond = exp_result["condition"]
            f.write(f"\n{'═' * 70}\n")
            f.write(f"Feature {fid} × {lang} — {cond}\n")
            f.write(f"{'═' * 70}\n")
            for g in exp_result["generations"][:3]:
                f.write(f"\nPrompt: {g['prompt'][:100]}\n")
                f.write(f"Response: {g['response'][:200]}\n")

    elapsed = time.time() - t0
    print(f"\n{'═' * 70}")
    print(f"Done — {len(all_results['experiments'])} conditions, {elapsed:.0f}s")
    print(f"Saved: {gen_path}")
    print(f"Saved: {summary_path}")
    print(f"{'═' * 70}")


class _PassthroughHook:
    """SAE encode→decode+error with no modification (sanity check)."""

    def __init__(self, sae):
        self.sae = sae

    def __call__(self, module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        B, S, D = h.shape
        h_flat = h.reshape(-1, D).float()

        with torch.no_grad():
            features = self.sae.encode(h_flat)
            x_hat = self.sae.decode(features)
            error = h_flat - x_hat
            h_new = x_hat + error  # should be ≈ h_flat

        h_new = h_new.reshape(B, S, D).to(h.dtype)
        if isinstance(output, tuple):
            return (h_new,) + output[1:]
        return h_new


if __name__ == "__main__":
    main()
