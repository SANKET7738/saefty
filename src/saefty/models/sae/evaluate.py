import torch
import json
from pathlib import Path
from typing import Dict, List, Optional

from saefty.models.sae.base import BaseSAE


def evaluate_sae(sae: BaseSAE, eval_activations: torch.Tensor) -> Dict[str, float]:
    sae.eval()
    device = sae.device

    with torch.no_grad():
        x = eval_activations.to(device)
        x_hat, features = sae(x)

        # reconstruction MSE
        mse = (x - x_hat).pow(2).mean().item()

        # cosine similarity between x and x_hat
        cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()

        # L0: avg number of active features per sample
        l0 = (features > 0).float().sum(dim=-1).mean().item()

        # dead features: fraction that never activate across all eval samples
        ever_active = (features > 0).any(dim=0)
        dead_fraction = 1.0 - ever_active.float().mean().item()

        # explained variance: 1 - Var(x - x_hat) / Var(x)
        residual_var = (x - x_hat).var().item()
        input_var = x.var().item()
        explained_variance = 1.0 - residual_var / input_var if input_var > 0 else 0.0

    return {
        "reconstruction_mse": mse,
        "cosine_similarity": cos_sim,
        "l0": l0,
        "dead_fraction": dead_fraction,
        "explained_variance": explained_variance,
        "num_eval_samples": len(eval_activations),
    }


def evaluate_ce_recovered(
    sae: BaseSAE,
    engine,
    eval_texts: List[str],
    hook_layer: int,
) -> Dict[str, float]:
    sae.eval()
    device = sae.device

    normal_losses = []
    sae_losses = []

    for text in eval_texts:
        tokens = engine.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(engine.model.device)

        labels = tokens.input_ids.clone()

        # normal CE loss
        with torch.no_grad():
            out = engine.model(**tokens, labels=labels)
            normal_losses.append(out.loss.item())

        # CE loss with SAE patching layer output
        def make_sae_hook(sae_model):
            def hook_fn(module, input, output):
                act = output[0] if isinstance(output, tuple) else output
                original_shape = act.shape
                original_dtype = act.dtype
                flat = act.float().reshape(-1, sae_model.d_model)
                x_hat, _ = sae_model(flat.to(device))
                patched = x_hat.to(act.device).to(original_dtype).reshape(original_shape)
                if isinstance(output, tuple):
                    return (patched,) + output[1:]
                return patched
            return hook_fn

        handle = engine.model.model.layers[hook_layer].register_forward_hook(
            make_sae_hook(sae)
        )

        with torch.no_grad():
            out = engine.model(**tokens, labels=labels)
            sae_losses.append(out.loss.item())

        handle.remove()

    normal_ce = sum(normal_losses) / len(normal_losses)
    sae_ce = sum(sae_losses) / len(sae_losses)
    ce_recovered = 1.0 - (sae_ce - normal_ce) / normal_ce if normal_ce > 0 else 0.0

    return {
        "normal_ce": normal_ce,
        "sae_ce": sae_ce,
        "ce_recovered": ce_recovered,
        "num_eval_texts": len(eval_texts),
    }
