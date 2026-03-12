import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Tuple, Dict


class SAEConfig(BaseModel):
    d_model: int
    expansion_factor: int = 8
    seed: int = 42


class BaseSAE(nn.Module, ABC):
    def __init__(self, config: SAEConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_sae = config.d_model * config.expansion_factor

        torch.manual_seed(config.seed)


    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...


    @abstractmethod
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        ...


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        x_hat = self.decode(features)
        return x_hat, features


    @abstractmethod
    def loss(self, x: torch.Tensor, x_hat: torch.Tensor, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        ...


    def normalize_decoder(self):
        if hasattr(self, "W_dec"):
            with torch.no_grad():
                self.W_dec.data = F.normalize(self.W_dec.data, dim=-1)


    @property
    def num_features(self) -> int:
        return self.d_sae


    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
