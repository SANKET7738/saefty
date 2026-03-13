"""
Gated Sparse Autoencoder (separate gate + magnitude pathways, weight-tied).

Reference:
    Rajamanoharan et al., "Improving Dictionary Learning with Gated Sparse Autoencoders"
    DeepMind, 2024. arXiv:2404.16014.
    https://arxiv.org/abs/2404.16014
"""

import torch
import torch.nn as nn
from typing import Dict

from saefty.models.sae.base import BaseSAE, SAEConfig


class GatedConfig(SAEConfig):
    l1_coefficient: float = 5e-4


class GatedSAE(BaseSAE):
    def __init__(self, config: GatedConfig):
        super().__init__(config)
        self.l1_coefficient = config.l1_coefficient

        # gate weights — shared with magnitude via r_mag rescaling
        self.W_gate = nn.Parameter(torch.empty(self.d_model, self.d_sae))
        self.b_gate = nn.Parameter(torch.zeros(self.d_sae))

        # magnitude rescaling: W_mag = W_gate * exp(r_mag)
        self.r_mag = nn.Parameter(torch.zeros(self.d_sae))
        self.b_mag = nn.Parameter(torch.zeros(self.d_sae))

        # decoder
        self.W_dec = nn.Parameter(torch.empty(self.d_sae, self.d_model))
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        nn.init.kaiming_uniform_(self.W_gate)
        self.W_dec.data = self.W_gate.data.T.clone()
        self.normalize_decoder()

        self._gate_pre_acts = None


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec

        # gate: pure Heaviside (no gradient through gate — intentional per paper)
        # gate gradients come from L_sparsity and L_aux, not L_reconstruct
        gate_pre_acts = x_centered @ self.W_gate + self.b_gate
        self._gate_pre_acts = gate_pre_acts
        gate = (gate_pre_acts > 0).float()

        # magnitude: W_gate rescaled by exp(r_mag), with ReLU (Eq. 6)
        W_mag = self.W_gate * self.r_mag.exp()
        magnitude = torch.relu(x_centered @ W_mag + self.b_mag)

        return gate * magnitude


    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec


    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction_loss = (x - x_hat).pow(2).mean()

        # sparsity: L1 on ReLU of gate pre-activations (Eq. 8)
        gate_features = torch.relu(self._gate_pre_acts)
        l1_loss = gate_features.sum(dim=-1).mean()

        # auxiliary loss: frozen decoder reconstructs from ReLU(π_gate) (Eq. 8)
        # .detach() stops gradients flowing to W_dec/b_dec from this term
        aux_reconstruction = gate_features @ self.W_dec.detach() + self.b_dec.detach()
        aux_loss = (x - aux_reconstruction).pow(2).mean()

        total_loss = reconstruction_loss + self.l1_coefficient * l1_loss + aux_loss
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "l1_loss": l1_loss,
            "aux_loss": aux_loss,
        }
