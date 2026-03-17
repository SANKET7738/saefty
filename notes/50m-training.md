# 50M Multilingual Gated SAE Training

## Model
- **Base model**: [CohereLabs/tiny-aya-global](https://huggingface.co/CohereLabs/tiny-aya-global)
- **Hook point**: Layer 20 residual stream output

## Architecture
- **SAE type**: Gated SAE (Rajamanoharan et al., 2024)
- **d_model**: 2048
- **d_sae**: 16,384 (8x expansion)

## Training Parameters
| Parameter | Value |
|-----------|-------|
| Training tokens | 50,000,000 |
| Batch size | 4,096 |
| Learning rate | 2e-4 |
| Warmup steps | 1,000 |
| L1 coefficient | 5e-4 |
| Total steps | 12,161 |
| Seed | 42 |

## Data
- **Dataset**: [CohereLabs/aya_collection_language_split](https://huggingface.co/datasets/CohereLabs/aya_collection_language_split)
- **Languages** (9): English, Arabic, German, Spanish, French, Hindi, Japanese, Bengali, Chinese
- **Balance**: ~5.55M tokens per language (token-level budgeting)

## Eval Metrics
| Metric | Value |
|--------|-------|
| CE recovered | 99.89% |
| Cosine similarity | 0.951 |
| Explained variance | 98.6% |
| L0 (eval) | 1,224 |
| Dead features | 0.018% |

## Links
- **Weights**: [huggingface.co/sanket-mhatre/saefty/tree/main/gated-50m-9lang](https://huggingface.co/sanket-mhatre/saefty/tree/main/gated-50m-9lang)
