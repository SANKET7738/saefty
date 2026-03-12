import torch
import torch.nn as nn
from typing import Dict

from saefty.models.sae.base import BaseSAE, SAEConfig


class VanillaConfig(SAEConfig):
    l1_coefficient: float = 5e-4


class VanillaSAE(BaseSAE):
    def __init__(self, config: VanillaConfig):
        super().__init__(config)
        self.l1_coefficient = config.l1_coefficient

        self.W_enc = nn.Parameter(torch.empty(self.d_model, self.d_sae))
        self.b_enc = nn.Parameter(torch.zeros(self.d_sae))
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        nn.init.kaiming_uniform_(self.W_enc)
        self.W_dec.data = self.W_enc.data.T.clone()
        self.normalize_decoder()


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        return torch.relu(x_centered @ self.W_enc + self.b_enc)


    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec


    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction_loss = (x - x_hat).pow(2).mean()
        l1_loss = features.abs().mean()
        total_loss = reconstruction_loss + self.l1_coefficient * l1_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "l1_loss": l1_loss,
        }
