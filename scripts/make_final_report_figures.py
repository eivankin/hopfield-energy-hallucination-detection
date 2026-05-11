from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate final report figures from cached eval CSV.")
    parser.add_argument(
        "--cached-summary",
        default="outputs/cached_eval/cached_eval_summary.csv",
        help="Path produced by scripts/evaluate_cached_features.py.",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/final_report/figures",
        help="Figure output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.cached_summary)

    sns.set_theme(style="whitegrid", context="paper")
    make_feature_family_auroc(summary, out_dir / "feature_family_auroc.pdf")
    make_best_auroc(summary, out_dir / "best_auroc_by_run.pdf")
    make_scalar_energy_direction(summary, out_dir / "scalar_energy_direction.pdf")
    print(f"Wrote figures to {out_dir}")


def make_feature_family_auroc(summary: pd.DataFrame, output: Path) -> None:
    method_labels = {
        "response_nll_scalar": "NLL",
        "nll_length_logreg": "NLL+len",
        "hopfield_energy_per_layer_logreg": "Energy",
        "attention_per_layer_logreg": "Attention",
        "nll_length_hopfield_per_layer_logreg": "Combined",
    }
    runs = [
        "002_qwen05b_attention_energy",
        "003_qwen05b_ragtruth",
        "004_smollm2_360m_ragtruth",
        "005_gemma3_1b_ragtruth",
        "005_qwen05b_truthfulqa",
    ]
    data = summary[summary["run"].isin(runs) & summary["name"].isin(method_labels)].copy()
    data["setting"] = data.apply(_setting_label, axis=1)
    data["method"] = data["name"].map(method_labels)

    plt.figure(figsize=(7.2, 3.6))
    ax = sns.barplot(
        data=data,
        x="setting",
        y="auroc_mean",
        hue="method",
        hue_order=["NLL", "NLL+len", "Energy", "Attention", "Combined"],
        palette=["#BAB0AC", "#9D755D", "#4C78A8", "#72B7B2", "#F58518"],
    )
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.6)
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.35, 0.76)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="", ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def make_best_auroc(summary: pd.DataFrame, output: Path) -> None:
    data = summary[summary["name"] == "nll_length_hopfield_per_layer_logreg"].copy()
    data = data[~data["run"].str.contains("001_qwen05b_squad_hopfield")]
    data["setting"] = data.apply(_setting_label, axis=1)
    order = data.sort_values("auroc_mean", ascending=False)["setting"].tolist()

    plt.figure(figsize=(7.0, 3.2))
    ax = sns.barplot(data=data, x="setting", y="auroc_mean", order=order, color="#4C78A8")
    for patch, (_, row) in zip(ax.patches, data.set_index("setting").loc[order].iterrows(), strict=True):
        x = patch.get_x() + patch.get_width() / 2
        ax.errorbar(
            x,
            row["auroc_mean"],
            yerr=row["auroc_std"],
            color="black",
            capsize=3,
            linewidth=1,
        )
        ax.text(
            x,
            row["auroc_mean"] - 0.012,
            f"{row['auroc_mean']:.3f}",
            ha="center",
            va="top",
            fontsize=8,
            fontweight="bold",
            color="white",
        )
    ax.set_xlabel("")
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.60, 0.75)
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def make_scalar_energy_direction(summary: pd.DataFrame, output: Path) -> None:
    names = ["hopfield_energy_mean_scalar", "hopfield_energy_mean_neg_scalar"]
    data = summary[summary["name"].isin(names)].copy()
    data = data[
        data["run"].isin(
            [
                "003_qwen05b_ragtruth",
                "004_smollm2_360m_ragtruth",
                "005_gemma3_1b_ragtruth",
                "005_qwen05b_truthfulqa",
            ]
        )
    ]
    data["setting"] = data.apply(_setting_label, axis=1)
    data["score"] = data["name"].map(
        {
            "hopfield_energy_mean_scalar": "raw mean E",
            "hopfield_energy_mean_neg_scalar": "negated mean E",
        }
    )

    plt.figure(figsize=(7.0, 3.2))
    ax = sns.barplot(data=data, x="setting", y="auroc_mean", hue="score", palette=["#4C78A8", "#F58518"])
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("Scalar AUROC")
    ax.set_ylim(0.30, 0.72)
    ax.tick_params(axis="x", rotation=20)
    ax.legend(title="")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def _setting_label(row: pd.Series) -> str:
    if row["run"] == "002_qwen05b_attention_energy":
        return "SMILES/SQuAD\nQwen"
    if row["run"] == "003_qwen05b_ragtruth":
        return "RAGTruth\nQwen"
    if row["run"] == "004_smollm2_360m_ragtruth":
        return "RAGTruth\nSmolLM2"
    if row["run"] == "005_gemma3_1b_ragtruth":
        return "RAGTruth\nGemma"
    if row["run"] == "005_qwen05b_truthfulqa":
        return "TruthfulQA\nQwen"
    return str(row["run"])


if __name__ == "__main__":
    main()
