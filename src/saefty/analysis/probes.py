import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from typing import Dict


def train_probe_per_layer(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
) -> Dict[int, Dict[str, float]]:
    results = {}

    for layer_idx in sorted(activations.keys()):
        X = activations[layer_idx]  # [N, d_model]
        y = labels                  # [N]

        probe = LogisticRegression(max_iter=1000, C=0.1)
        scores = cross_val_score(probe, X, y, cv=5, scoring="accuracy")

        results[layer_idx] = {
            "mean_acc": float(scores.mean()),
            "std_acc": float(scores.std()),
        }
        print(f"  layer {layer_idx}: {scores.mean():.3f} ± {scores.std():.3f}")

    return results


def train_probe_per_language(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    languages: np.ndarray,
) -> Dict[str, Dict[int, float]]:
    results = {}

    for lang in np.unique(languages):
        mask = (languages == lang)
        n_samples = mask.sum()

        if n_samples < 20:
            print(f"  skipping {lang}: only {n_samples} samples")
            continue

        n_folds = min(5, min(np.sum(labels[mask] == 0), np.sum(labels[mask] == 1)))
        if n_folds < 2:
            print(f"  skipping {lang}: not enough class balance for CV")
            continue

        results[lang] = {}

        for layer_idx in sorted(activations.keys()):
            X = activations[layer_idx][mask]
            y = labels[mask]

            probe = LogisticRegression(max_iter=1000, C=0.1)
            scores = cross_val_score(probe, X, y, cv=n_folds, scoring="accuracy")
            results[lang][layer_idx] = float(scores.mean())

        best_layer = max(results[lang], key=results[lang].get)
        print(f"  {lang}: best layer {best_layer} ({results[lang][best_layer]:.3f}), {n_samples} samples")

    return results


def cross_lingual_transfer(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    languages: np.ndarray,
    best_layer: int,
) -> Dict[str, float]:
    en_mask = (languages == "en")
    X_en = activations[best_layer][en_mask]
    y_en = labels[en_mask]

    probe = LogisticRegression(max_iter=1000, C=0.1)
    probe.fit(X_en, y_en)

    transfer = {}
    for lang in np.unique(languages):
        if lang == "en":
            continue
        lang_mask = (languages == lang)
        score = float(probe.score(activations[best_layer][lang_mask], labels[lang_mask]))
        transfer[lang] = score
        print(f"  en → {lang}: {score:.3f}")

    return transfer


def plot_probe_accuracy(
    per_layer: Dict[int, Dict[str, float]],
    per_language: Dict[str, Dict[int, float]],
    save_path: str,
) -> None:
    layers = sorted(per_layer.keys())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # overall accuracy with error bars
    means = [per_layer[l]["mean_acc"] for l in layers]
    stds = [per_layer[l]["std_acc"] for l in layers]
    ax1.errorbar(layers, means, yerr=stds, marker="o", markersize=4, capsize=3)
    best = layers[np.argmax(means)]
    ax1.axvline(x=best, color="red", linestyle="--", alpha=0.5, label=f"best: layer {best}")
    ax1.set_xlabel("layer")
    ax1.set_ylabel("accuracy")
    ax1.set_title("probe accuracy across layers (overall)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # per-language accuracy
    for lang in sorted(per_language.keys()):
        lang_layers = sorted(per_language[lang].keys())
        lang_accs = [per_language[lang][l] for l in lang_layers]
        ax2.plot(lang_layers, lang_accs, marker="o", markersize=3, label=lang)
    ax2.set_xlabel("layer")
    ax2.set_ylabel("accuracy")
    ax2.set_title("probe accuracy across layers (per language)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"saved probe accuracy plot to {save_path}")
