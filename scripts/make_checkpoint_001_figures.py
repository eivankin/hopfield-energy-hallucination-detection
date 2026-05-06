from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate checkpoint 001 report figures.")
    parser.add_argument(
        "--features",
        default="outputs/001_qwen05b_squad_hopfield/features.npz",
        help="Path to features.npz produced by experiment 001.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/figures",
        help="Directory where figures and stats.json will be written.",
    )
    parser.add_argument("--bins", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    features_path = Path(args.features)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arrays = np.load(features_path)
    y = arrays["y"]
    nll = arrays["nll"]
    response_length = arrays["response_length"]
    hopfield = arrays["hopfield"]

    energy = hopfield[:, 0]
    energy_std = hopfield[:, 1]

    sns.set_theme(style="whitegrid", context="talk")

    plot_distribution(
        energy,
        y,
        title="Hopfield mean energy by label",
        xlabel="Mean Q/K Hopfield energy",
        output_path=out_dir / "energy_hist.png",
        bins=args.bins,
    )
    plot_distribution(
        nll,
        y,
        title="Response negative log-likelihood by label",
        xlabel="Mean response NLL",
        output_path=out_dir / "nll_hist.png",
        bins=args.bins,
    )

    auc_items = [
        ("response length", roc_auc_score(y, response_length)),
        ("Hopfield mean energy", roc_auc_score(y, energy)),
        ("Hopfield std energy (flipped)", 1.0 - roc_auc_score(y, energy_std)),
        ("response NLL", roc_auc_score(y, nll)),
        ("NLL + Hopfield logreg", 0.7058),
        ("NLL + length + Hopfield logreg", 0.7081),
    ]
    plot_auroc_bars(auc_items, out_dir / "auroc_bars.png")

    stats = {
        "features_file": str(features_path),
        "n": int(len(y)),
        "truthful": int((y == 0).sum()),
        "hallucinated": int((y == 1).sum()),
        "energy_min": float(energy.min()),
        "energy_max": float(energy.max()),
        "energy_mean_truthful": float(energy[y == 0].mean()),
        "energy_mean_hallucinated": float(energy[y == 1].mean()),
        "nll_min": float(nll.min()),
        "nll_max": float(nll.max()),
        "nll_mean_truthful": float(nll[y == 0].mean()),
        "nll_mean_hallucinated": float(nll[y == 1].mean()),
        "energy_auc": float(roc_auc_score(y, energy)),
        "nll_auc": float(roc_auc_score(y, nll)),
        "length_auc": float(roc_auc_score(y, response_length)),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def plot_distribution(
    values: np.ndarray,
    y: np.ndarray,
    *,
    title: str,
    xlabel: str,
    output_path: Path,
    bins: int,
) -> None:
    truthful = values[y == 0]
    hallucinated = values[y == 1]
    full_min = float(values.min())
    full_max = float(values.max())
    bin_edges = np.linspace(full_min, full_max, bins + 1)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.histplot(
        truthful,
        bins=bin_edges,
        stat="density",
        element="step",
        fill=False,
        linewidth=2.4,
        color="#2563eb",
        label="truthful (label 0)",
        ax=ax,
    )
    sns.histplot(
        hallucinated,
        bins=bin_edges,
        stat="density",
        element="step",
        fill=False,
        linewidth=2.4,
        color="#dc2626",
        label="hallucinated (label 1)",
        ax=ax,
    )
    sns.kdeplot(truthful, color="#2563eb", linestyle="--", linewidth=1.8, ax=ax)
    sns.kdeplot(hallucinated, color="#dc2626", linestyle="--", linewidth=1.8, ax=ax)

    mean_truthful = float(truthful.mean())
    mean_hallucinated = float(hallucinated.mean())
    ax.axvline(mean_truthful, color="#2563eb", linestyle=":", linewidth=2)
    ax.axvline(mean_hallucinated, color="#dc2626", linestyle=":", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_xlim(full_min, full_max)
    ax.legend(loc="upper right", frameon=True)
    ax.text(
        0.01,
        0.98,
        (
            f"full range: [{full_min:.2f}, {full_max:.2f}]\n"
            f"truthful mean: {mean_truthful:.2f}\n"
            f"hallucinated mean: {mean_hallucinated:.2f}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#d1d5db"},
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_auroc_bars(items: list[tuple[str, float]], output_path: Path) -> None:
    labels = [name for name, _ in items]
    values = [value for _, value in items]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    colors = ["#64748b" if v < 0.64 else "#2563eb" if v < 0.70 else "#16a34a" for v in values]
    sns.barplot(x=values, y=labels, palette=colors, ax=ax, orient="h", hue=labels, legend=False)
    ax.set_xlim(0.5, max(values) + 0.03)
    ax.set_xlabel("AUROC")
    ax.set_ylabel("")
    ax.set_title("AUROC summary")
    ax.axvline(0.5, color="#111827", linewidth=1, linestyle="--")
    for i, value in enumerate(values):
        ax.text(value + 0.004, i, f"{value:.3f}", va="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
