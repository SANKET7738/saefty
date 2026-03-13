# Compare SAE training results across multiple runs.

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Compare SAE training results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/train_sae",
        help="Root directory containing per-run result subdirectories",
    )
    return parser.parse_args()


def load_results(results_dir: Path) -> list[dict]:
    results = []
    for subdir in sorted(results_dir.iterdir()):
        if not subdir.is_dir():
            continue
        metrics_path = subdir / "metrics.json"
        if not metrics_path.exists():
            print(f"  Skipping {subdir.name}/ (no metrics.json)")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        results.append({"name": subdir.name, **metrics})
    results.sort(key=lambda r: r.get("ce_recovered", 0), reverse=True)
    return results


def print_table(results: list[dict]):
    header = f"{'Name':<25} {'L0':>10} {'Dead%':>10} {'CosSim':>10} {'CE Recov':>10} {'ExplVar':>10} {'Recon MSE':>12}"
    print(header)
    print("-" * len(header))
    for r in results:
        l0 = r.get("l0")
        dead = r.get("dead_fraction")
        cos = r.get("cosine_similarity")
        ce = r.get("ce_recovered")
        ev = r.get("explained_variance")
        mse = r.get("reconstruction_mse")

        l0_s = f"{l0:>10.1f}" if l0 is not None else f"{'N/A':>10}"
        dead_s = f"{dead * 100:>9.2f}%" if dead is not None else f"{'N/A':>10}"
        cos_s = f"{cos:>10.4f}" if cos is not None else f"{'N/A':>10}"
        ce_s = f"{ce:>10.4f}" if ce is not None else f"{'N/A':>10}"
        ev_s = f"{ev:>10.4f}" if ev is not None else f"{'N/A':>10}"
        mse_s = f"{mse:>12.6f}" if mse is not None else f"{'N/A':>12}"

        print(f"{r['name']:<25} {l0_s} {dead_s} {cos_s} {ce_s} {ev_s} {mse_s}")


def make_charts(results: list[dict], output_path: Path):
    names = [r["name"] for r in results]
    ce_recovered = [r.get("ce_recovered", 0) for r in results]
    l0 = [r.get("l0", 0) for r in results]
    dead_pct = [r.get("dead_fraction", 0) * 100 for r in results]
    cos_sim = [r.get("cosine_similarity", 0) for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("SAE Training Comparison", fontsize=16, fontweight="bold")

    charts = [
        (axes[0, 0], ce_recovered, "CE Recovered", "tab:green"),
        (axes[0, 1], l0, "L0", "tab:blue"),
        (axes[1, 0], dead_pct, "Dead Features (%)", "tab:red"),
        (axes[1, 1], cos_sim, "Cosine Similarity", "tab:purple"),
    ]

    for ax, values, title, color in charts:
        bars = ax.bar(names, values, color=color, alpha=0.8)
        ax.set_title(title, fontsize=13)
        ax.set_ylabel(title)
        ax.tick_params(axis="x", rotation=30)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.3f}" if val < 100 else f"{val:.0f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nChart saved to {output_path}")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: results directory not found: {results_dir}")
        return

    print(f"Scanning {results_dir}/\n")
    results = load_results(results_dir)

    if not results:
        print("No results found.")
        return

    print(f"\nFound {len(results)} run(s), sorted by CE recovered:\n")
    print_table(results)

    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    json_path = comparison_dir / "comparison.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTable saved to {json_path}")

    make_charts(results, comparison_dir / "comparison.png")


if __name__ == "__main__":
    main()
