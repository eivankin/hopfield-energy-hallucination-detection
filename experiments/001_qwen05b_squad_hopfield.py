from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hopfield_study.data import labels_array, load_examples
from hopfield_study.evaluate import (
    cv_logreg_grid_metrics,
    cv_logreg_metrics,
    majority_metrics,
    scalar_score_metrics,
)
from hopfield_study.extract import (
    DEFAULT_LAYERS,
    DEFAULT_MODEL,
    extract_features,
    load_model_and_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="First Qwen2.5-0.5B Hopfield-energy experiment on SMILES/SQuAD data."
    )
    parser.add_argument("--data", required=True, help="Path to dataset.csv")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", default="outputs/001_qwen05b_squad_hopfield")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means full dataset")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-response-tokens", type=int, default=32)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_LAYERS),
        help="Qwen decoder layer indices for Hopfield energy",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(args.data, max_samples=args.max_samples)
    if not examples:
        raise RuntimeError("No labeled examples loaded")
    y = labels_array(examples)

    model, tokenizer, device = load_model_and_tokenizer(args.model, args.device)
    features = extract_features(
        examples,
        model,
        tokenizer,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        layers=tuple(args.layers),
        max_response_tokens=args.max_response_tokens,
    )

    layer_feature_names = [
        "energy_mean_layermean",
        "energy_std_layermean",
        "energy_max_layermean",
        "energy_final_layermean",
        "prompt_attention_mass_mean_layermean",
        "prompt_attention_mass_final_layermean",
        "prompt_attention_max_mean_layermean",
        "prompt_attention_entropy_mean_layermean",
        "energy_mean_layerstd",
        "energy_std_layerstd",
        "energy_max_layerstd",
        "energy_final_layerstd",
        "prompt_attention_mass_mean_layerstd",
        "prompt_attention_mass_final_layerstd",
        "prompt_attention_max_mean_layerstd",
        "prompt_attention_entropy_mean_layerstd",
    ]
    per_layer_names = []
    base_layer_names = [
        "energy_mean",
        "energy_std",
        "energy_max",
        "energy_final",
        "prompt_attention_mass_mean",
        "prompt_attention_mass_final",
        "prompt_attention_max_mean",
        "prompt_attention_entropy_mean",
    ]
    for layer in args.layers:
        per_layer_names.extend([f"layer{layer}_{name}" for name in base_layer_names])

    X_nll = features.nll.reshape(-1, 1)
    X_nll_length = np.column_stack([features.nll, features.response_length])
    X_summary = features.hopfield
    X_per_layer = features.layer_features
    energy_cols = [0, 1, 2, 3, 8, 9, 10, 11]
    attention_cols = [4, 5, 6, 7, 12, 13, 14, 15]
    per_layer_energy_cols = [
        idx
        for idx, name in enumerate(per_layer_names)
        if name.endswith(("energy_mean", "energy_std", "energy_max", "energy_final"))
    ]
    per_layer_attention_cols = [
        idx for idx, name in enumerate(per_layer_names) if "prompt_attention" in name
    ]
    X_energy = X_summary[:, energy_cols]
    X_attention = X_summary[:, attention_cols]
    X_combined_summary = np.column_stack([features.nll, features.response_length, X_summary])
    X_combined_per_layer = np.column_stack([features.nll, features.response_length, X_per_layer])
    groups = np.asarray([ex.context for ex in examples])

    metrics = [
        majority_metrics(y),
        scalar_score_metrics("response_length_scalar", features.response_length, y),
        scalar_score_metrics("response_nll_scalar", features.nll, y),
        scalar_score_metrics("hopfield_energy_mean_scalar", X_summary[:, 0], y),
        scalar_score_metrics("hopfield_energy_mean_neg_scalar", -X_summary[:, 0], y),
        scalar_score_metrics("prompt_attention_mass_scalar", X_summary[:, 4], y),
        scalar_score_metrics("prompt_attention_entropy_scalar", X_summary[:, 7], y),
        cv_logreg_metrics("response_nll_logreg", X_nll, y),
        cv_logreg_metrics("nll_length_logreg", X_nll_length, y),
        cv_logreg_metrics("hopfield_energy_summary_logreg", X_energy, y),
        cv_logreg_metrics("attention_summary_logreg", X_attention, y),
        cv_logreg_metrics(
            "hopfield_energy_per_layer_logreg",
            X_per_layer[:, per_layer_energy_cols],
            y,
        ),
        cv_logreg_metrics(
            "attention_per_layer_logreg",
            X_per_layer[:, per_layer_attention_cols],
            y,
        ),
        cv_logreg_metrics("hopfield_all_summary_logreg", X_summary, y),
        cv_logreg_metrics("hopfield_all_per_layer_logreg", X_per_layer, y),
        cv_logreg_metrics("nll_length_hopfield_summary_logreg", X_combined_summary, y),
        cv_logreg_metrics("nll_length_hopfield_per_layer_logreg", X_combined_per_layer, y),
        cv_logreg_grid_metrics(
            "nll_length_hopfield_per_layer_logreg_grid",
            X_combined_per_layer,
            y,
        ),
        cv_logreg_metrics(
            "context_group_nll_length_hopfield_summary_logreg",
            X_combined_summary,
            y,
            groups=groups,
        ),
    ]

    metrics_payload = {
        "model": args.model,
        "data": str(Path(args.data).resolve()),
        "n_samples": len(examples),
        "labels": {
            "truthful": int((y == 0).sum()),
            "hallucinated": int((y == 1).sum()),
        },
        "layers": list(args.layers),
        "max_response_tokens": args.max_response_tokens,
        "feature_names": {
            "hopfield_summary": layer_feature_names,
            "per_layer": per_layer_names,
            "combined_summary": ["response_nll", "response_length", *layer_feature_names],
            "combined_per_layer": ["response_nll", "response_length", *per_layer_names],
        },
        "metrics": [row.__dict__ for row in metrics],
    }

    np.savez_compressed(
        output_dir / "features.npz",
        y=y,
        nll=features.nll,
        response_length=features.response_length,
        hopfield=features.hopfield,
        layer_features=features.layer_features,
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2),
        encoding="utf-8",
    )
    (output_dir / "summary.md").write_text(
        _summary_md(metrics_payload),
        encoding="utf-8",
    )
    print(_summary_md(metrics_payload))


def _summary_md(payload: dict) -> str:
    lines = [
        "# Experiment 001: Qwen2.5-0.5B SQuAD-Hopfield",
        "",
        f"- model: `{payload['model']}`",
        f"- samples: {payload['n_samples']}",
        f"- labels: {payload['labels']}",
        f"- layers: {payload['layers']}",
        "",
        "| method | accuracy | f1 | auroc |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in payload["metrics"]:
        auroc = row["auroc"]
        auroc_text = "nan" if auroc != auroc else f"{auroc:.4f}"
        lines.append(
            f"| {row['name']} | {row['accuracy']:.4f} | {row['f1']:.4f} | {auroc_text} |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
