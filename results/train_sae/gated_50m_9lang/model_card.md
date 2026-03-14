---
tags:
  - sparse-autoencoder
  - multilingual
  - safety
  - interpretability
language: [arb, ben, deu, eng, fra, hin, jpn, spa, zho]
---

# Gated SAE — Multilingual Safety Features

Sparse autoencoder trained on **CohereLabs/tiny-aya-global** (layer 20) for cross-lingual safety representation analysis.

## Training Config

| Parameter | Value |
|-----------|-------|
| Architecture | Gated SAE |
| d_model | 2048 |
| d_sae | 16384 |
| Expansion | 8x |
| L1 coefficient | 0.0005 |
| Training tokens | 50,000,008 |
| Languages | 9 (arb, ben, deu, eng, fra, hin, jpn, spa, zho) |

## Metrics

| Metric | Value |
|--------|-------|
| CE recovered | 0.9989 |
| Cosine similarity | 0.9508 |
| Explained variance | 0.9861 |
| L0 (eval) | 1223.8 |
| Dead features | 0.0183% |
| Total steps | 12161 |

## Language Balance

| Language | Tokens |
|----------|--------|
| arb | 5,555,597 |
| ben | 5,555,660 |
| deu | 5,555,565 |
| eng | 5,555,259 |
| fra | 5,555,561 |
| hin | 5,555,584 |
| jpn | 5,555,595 |
| spa | 5,555,600 |
| zho | 5,555,587 |

## Usage

```python
import torch
from saefty.models.sae.gated import GatedSAE

state = torch.load("sae_final.pt", map_location="cpu")
sae = GatedSAE(d_model=2048, d_sae=16384)
sae.load_state_dict(state["sae_state_dict"])
```
