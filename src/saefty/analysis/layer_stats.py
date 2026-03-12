import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.decomposition import PCA
from typing import Dict


def compute_layer_stats(
    activations: Dict[int, np.ndarray],
) -> Dict[int, Dict[str, float]]:
    stats = {}
    for layer_idx in sorted(activations.keys()):
        acts = activations[layer_idx]  # [N, d_model]

        pca = PCA(n_components=min(10, acts.shape[0], acts.shape[1]))
        pca.fit(acts)

        stats[layer_idx] = {
            "mean": float(acts.mean()),
            "std": float(acts.std()),
            "kurtosis": float(kurtosis(acts.flatten())),
            "mean_norm": float(np.linalg.norm(acts, axis=-1).mean()),
            "pca_var_10": float(pca.explained_variance_ratio_.sum()),
        }
        print(f"  layer {layer_idx}: std={stats[layer_idx]['std']:.4f}, kurtosis={stats[layer_idx]['kurtosis']:.4f}")

    return stats


def plot_layer_stats(
    stats: Dict[int, Dict[str, float]],
    save_path: str,
) -> None:
    layers = sorted(stats.keys())
    stat_names = ["mean", "std", "kurtosis", "mean_norm", "pca_var_10"]

    fig, axes = plt.subplots(len(stat_names), 1, figsize=(12, 4 * len(stat_names)))

    for ax, name in zip(axes, stat_names):
        values = [stats[l][name] for l in layers]
        ax.plot(layers, values, marker="o", markersize=3)
        ax.set_xlabel("layer")
        ax.set_ylabel(name)
        ax.set_title(f"{name} across layers")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved layer stats plot to {save_path}")
