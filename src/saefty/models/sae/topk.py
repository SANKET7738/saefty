import torch
import torch.nn as nn
from typing import Dict

from saefty.models.sae.base import BaseSAE, SAEConfig


class TopKConfig(SAEConfig):
    k: int = 32


class TopKSAE(BaseSAE):
    def __init__(self, config: TopKConfig):
        super().__init__(config)
        self.k = config.k

        self.W_enc = nn.Parameter(torch.empty(self.d_model, self.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        self.W_dec.data = self.W_enc.data.T.clone()
        self.normalize_decoder()


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc

        # keep only top-k, zero out the rest
        topk_values, topk_indices = torch.topk(pre_acts, self.k, dim=-1)
        features = torch.zeros_like(pre_acts)
        features.scatter_(-1, topk_indices, torch.relu(topk_values))
        return features


    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec


    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction_loss = (x - x_hat).pow(2).mean()
        # no L1 penalty — sparsity is structural (exactly k features active)
        return {
            "loss": reconstruction_loss,
            "reconstruction_loss": reconstruction_loss,
        }
