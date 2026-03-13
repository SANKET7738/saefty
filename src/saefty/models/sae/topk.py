"""
TopK Sparse Autoencoder (hard top-k sparsity, no L1 penalty).

Reference:
    Gao et al., "Scaling and Evaluating Sparse Autoencoders"
    OpenAI, ICLR 2025. arXiv:2406.04093.
    https://arxiv.org/abs/2406.04093
    Code: https://github.com/openai/sparse_autoencoder
"""

import torch
import torch.nn as nn
from typing import Dict

from saefty.models.sae.base import BaseSAE, SAEConfig


class TopKConfig(SAEConfig):
    k: int = 32
    auxk_alpha: float = 0.0      # 0 = disabled, 1/16 = standard (Gao et al. §3.3)
    auxk: int = 256              # how many dead features to activate in aux pass
    dead_steps_threshold: int = 10  # steps without firing → dead


class TopKSAE(BaseSAE):
    def __init__(self, config: TopKConfig):
        super().__init__(config)
        self.k = config.k
        self.auxk_alpha = config.auxk_alpha
        self.auxk = config.auxk
        self.dead_steps_threshold = config.dead_steps_threshold

        self.W_enc = nn.Parameter(torch.empty(self.d_model, self.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        self.W_dec.data = self.W_enc.data.T.clone()
        self.normalize_decoder()

        # dead feature tracking for auxiliary loss
        if self.auxk_alpha > 0:
            self.register_buffer(
                "num_steps_since_fired",
                torch.zeros(self.d_sae, dtype=torch.long),
            )

        self._pre_acts = None


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc  # (batch, d_sae)

        # cache for auxiliary loss
        self._pre_acts = pre_acts

        # keep only top-k, zero out the rest
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        features = torch.zeros_like(pre_acts)
        features.scatter_(-1, topk_indices, torch.relu(topk_values))

        # update dead feature tracking during training
        if self.training and self.auxk_alpha > 0:
            fired = (features > 0).any(dim=0)  # (d_sae,)
            self.num_steps_since_fired += 1
            self.num_steps_since_fired[fired] = 0

        return features


    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec


    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction_loss = (x - x_hat).pow(2).mean()

        total_loss = reconstruction_loss
        losses = {"reconstruction_loss": reconstruction_loss}

        # auxiliary dead latent loss (Gao et al. §3.3)
        # encourages dead features to become useful by reconstructing what live features missed
        if self.auxk_alpha > 0 and self._pre_acts is not None and self.training:
            dead_mask = self.num_steps_since_fired > self.dead_steps_threshold  # (d_sae,)
            num_dead = dead_mask.sum().item()

            if num_dead > 0:
                # mask out live features, keep only dead pre-activations
                dead_pre_acts = self._pre_acts.clone()
                dead_pre_acts[:, ~dead_mask] = -float("inf")

                # select top-auxk among dead features
                k_aux = min(self.auxk, int(num_dead))
                auxk_values, auxk_indices = torch.topk(dead_pre_acts, k_aux, dim=-1)

                auxk_features = torch.zeros_like(self._pre_acts)
                auxk_features.scatter_(-1, auxk_indices, torch.relu(auxk_values))

                # reconstruct from dead features using FROZEN decoder (paper spec)
                auxk_recon = auxk_features @ self.W_dec.detach() + self.b_dec.detach()

                # dead features try to explain what live features missed
                residual = (x - x_hat).detach()
                aux_loss = (residual - auxk_recon).pow(2).mean()

                total_loss = total_loss + self.auxk_alpha * aux_loss
                losses["aux_loss"] = aux_loss
            else:
                losses["aux_loss"] = torch.tensor(0.0, device=x.device)

        losses["loss"] = total_loss
        return losses
