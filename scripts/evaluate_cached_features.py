from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hopfield_study.data import load_examples
from hopfield_study.evaluate import (
    MetricRow,
    cv_logreg_grid_metrics,
    cv_logreg_metrics,
    majority_metrics,
    scalar_score_metrics,
)


DEFAULT_SEEDS = (13, 21, 42, 87, 101)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-evaluate cached Hopfield-study feature files over multiple CV seeds."
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Directory containing experiment output subdirectories.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Optional explicit output directories. If unset, discover directories under outputs-root.",
    )
    parser.add_argument(
        "--exclude-smoke",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip output directories with fewer than 100 samples.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="CV seeds for learned probes.",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/cached_eval",
        help="Where to write aggregate CSV/Markdown files.",
    )
    parser.add_argument(
        "--grid",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also run the slower C-grid logistic-regression variant.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _select_runs(args)
    rows: list[dict[str, Any]] = []
    for run_dir in runs:
        print(f"Evaluating cached run: {run_dir}")
        rows.extend(_evaluate_run(run_dir, seeds=tuple(args.seeds), include_grid=args.grid))

    long_df = pd.DataFrame(rows)
    if long_df.empty:
        raise RuntimeError("No cached runs evaluated")

    summary_df = _summarize(long_df)
    comparison_df = _best_comparison(summary_df)

    long_df.to_csv(out_dir / "cached_eval_long.csv", index=False)
    summary_df.to_csv(out_dir / "cached_eval_summary.csv", index=False)
    comparison_df.to_csv(out_dir / "cached_eval_best.csv", index=False)
    (out_dir / "cached_eval_summary.md").write_text(
        _summary_markdown(summary_df, comparison_df, seeds=tuple(args.seeds)),
        encoding="utf-8",
    )
    print((out_dir / "cached_eval_summary.md").read_text(encoding="utf-8"))


def _select_runs(args: argparse.Namespace) -> list[Path]:
    if args.include:
        candidates = [Path(path) for path in args.include]
    else:
        candidates = sorted(
            path
            for path in Path(args.outputs_root).iterdir()
            if path.is_dir() and (path / "features.npz").exists() and (path / "metrics.json").exists()
        )
    runs = []
    for run_dir in candidates:
        metadata = _read_json(run_dir / "metrics.json")
        if args.exclude_smoke and int(metadata.get("n_samples", 0)) < 100:
            continue
        runs.append(run_dir)
    return runs


def _evaluate_run(run_dir: Path, *, seeds: tuple[int, ...], include_grid: bool) -> list[dict[str, Any]]:
    metadata = _read_json(run_dir / "metrics.json")
    feature_file = np.load(run_dir / "features.npz")
    y = np.asarray(feature_file["y"], dtype=np.int64)
    groups = _load_groups(metadata, expected_len=len(y))
    feature_sets = _build_feature_sets(feature_file, metadata)

    rows: list[dict[str, Any]] = []
    run_info = _run_info(run_dir, metadata)
    for metric in _scalar_metrics(feature_file, metadata, y):
        rows.append({**run_info, "seed": "scalar", **asdict(metric)})

    rows.append({**run_info, "seed": "scalar", **asdict(majority_metrics(y))})

    for seed in seeds:
        for name, X in feature_sets.items():
            rows.append({**run_info, "seed": seed, **asdict(cv_logreg_metrics(name, X, y, seed=seed))})
        if groups is not None and "nll_length_hopfield_summary_logreg" in feature_sets:
            metric = cv_logreg_metrics(
                "context_group_nll_length_hopfield_summary_logreg",
                feature_sets["nll_length_hopfield_summary_logreg"],
                y,
                seed=seed,
                groups=groups,
            )
            rows.append({**run_info, "seed": seed, **asdict(metric)})
        if include_grid and "nll_length_hopfield_per_layer_logreg" in feature_sets:
            metric = cv_logreg_grid_metrics(
                "nll_length_hopfield_per_layer_logreg_grid",
                feature_sets["nll_length_hopfield_per_layer_logreg"],
                y,
                seed=seed,
            )
            rows.append({**run_info, "seed": seed, **asdict(metric)})
    return rows


def _scalar_metrics(
    feature_file: np.lib.npyio.NpzFile,
    metadata: dict[str, Any],
    y: np.ndarray,
) -> list[MetricRow]:
    hopfield = np.asarray(feature_file["hopfield"], dtype=np.float32)
    summary_names = _feature_names(metadata, "hopfield_summary") or _feature_names(metadata, "hopfield")
    metrics = [
        scalar_score_metrics("response_length_scalar", feature_file["response_length"], y),
        scalar_score_metrics("response_nll_scalar", feature_file["nll"], y),
        scalar_score_metrics("hopfield_energy_mean_scalar", hopfield[:, 0], y),
        scalar_score_metrics("hopfield_energy_mean_neg_scalar", -hopfield[:, 0], y),
    ]
    if any("prompt_attention" in name for name in summary_names):
        metrics.extend(
            [
                scalar_score_metrics("prompt_attention_mass_scalar", hopfield[:, 4], y),
                scalar_score_metrics("prompt_attention_entropy_scalar", hopfield[:, 7], y),
            ]
        )
    return metrics


def _build_feature_sets(
    feature_file: np.lib.npyio.NpzFile,
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    nll = np.asarray(feature_file["nll"], dtype=np.float32)
    response_length = np.asarray(feature_file["response_length"], dtype=np.float32)
    hopfield = np.asarray(feature_file["hopfield"], dtype=np.float32)
    feature_sets: dict[str, np.ndarray] = {
        "response_nll_logreg": nll.reshape(-1, 1),
        "nll_length_logreg": np.column_stack([nll, response_length]),
    }

    summary_names = _feature_names(metadata, "hopfield_summary") or _feature_names(metadata, "hopfield")
    if summary_names:
        energy_cols = _matching_columns(
            summary_names,
            include_any=("energy_",),
            exclude_any=("prompt_attention",),
        )
        attention_cols = _matching_columns(summary_names, include_any=("prompt_attention",))
        if energy_cols:
            feature_sets["hopfield_energy_summary_logreg"] = hopfield[:, energy_cols]
        if attention_cols:
            feature_sets["attention_summary_logreg"] = hopfield[:, attention_cols]
        feature_sets["hopfield_all_summary_logreg"] = hopfield
        feature_sets["nll_length_hopfield_summary_logreg"] = np.column_stack(
            [nll, response_length, hopfield]
        )

    if "layer_features" in feature_file:
        layer_features = np.asarray(feature_file["layer_features"], dtype=np.float32)
        per_layer_names = _feature_names(metadata, "per_layer")
        energy_cols = _matching_columns(
            per_layer_names,
            include_any=("energy_",),
            exclude_any=("prompt_attention",),
        )
        attention_cols = _matching_columns(per_layer_names, include_any=("prompt_attention",))
        if energy_cols:
            feature_sets["hopfield_energy_per_layer_logreg"] = layer_features[:, energy_cols]
        if attention_cols:
            feature_sets["attention_per_layer_logreg"] = layer_features[:, attention_cols]
        feature_sets["hopfield_all_per_layer_logreg"] = layer_features
        feature_sets["nll_length_hopfield_per_layer_logreg"] = np.column_stack(
            [nll, response_length, layer_features]
        )
    return feature_sets


def _matching_columns(
    names: list[str],
    *,
    include_any: tuple[str, ...],
    exclude_any: tuple[str, ...] = (),
) -> list[int]:
    return [
        idx
        for idx, name in enumerate(names)
        if any(marker in name for marker in include_any)
        and not any(marker in name for marker in exclude_any)
    ]


def _feature_names(metadata: dict[str, Any], key: str) -> list[str]:
    names = metadata.get("feature_names", {}).get(key, [])
    return list(names) if isinstance(names, list) else []


def _load_groups(metadata: dict[str, Any], *, expected_len: int) -> np.ndarray | None:
    data_path = metadata.get("data")
    if not data_path:
        return None
    path = Path(data_path)
    if not path.exists():
        return None
    examples = load_examples(path)
    if len(examples) != expected_len:
        return None
    groups = np.asarray([example.context for example in examples])
    if len(np.unique(groups)) < 2:
        return None
    return groups


def _run_info(run_dir: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    data_path = Path(str(metadata.get("data", "")))
    dataset = data_path.stem if data_path.name else "unknown"
    return {
        "run": run_dir.name,
        "dataset": dataset,
        "model": metadata.get("model", "unknown"),
        "layers": " ".join(str(layer) for layer in metadata.get("layers", [])),
        "n_samples": metadata.get("n_samples"),
        "truthful": metadata.get("labels", {}).get("truthful"),
        "hallucinated": metadata.get("labels", {}).get("hallucinated"),
    }


def _summarize(long_df: pd.DataFrame) -> pd.DataFrame:
    grouped = long_df.groupby(
        ["run", "dataset", "model", "layers", "n_samples", "truthful", "hallucinated", "name"],
        dropna=False,
    )
    rows = []
    for keys, group in grouped:
        record = dict(
            zip(
                ["run", "dataset", "model", "layers", "n_samples", "truthful", "hallucinated", "name"],
                keys,
                strict=True,
            )
        )
        for metric in ("accuracy", "f1", "auroc"):
            values = pd.to_numeric(group[metric], errors="coerce")
            record[f"{metric}_mean"] = values.mean()
            record[f"{metric}_std"] = values.std(ddof=0)
        record["n_repeats"] = int(group["seed"].nunique())
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["run", "name"]).reset_index(drop=True)


def _best_comparison(summary_df: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "response_nll_scalar",
        "nll_length_logreg",
        "hopfield_energy_mean_scalar",
        "prompt_attention_mass_scalar",
        "hopfield_energy_per_layer_logreg",
        "attention_per_layer_logreg",
        "nll_length_hopfield_summary_logreg",
        "nll_length_hopfield_per_layer_logreg",
        "context_group_nll_length_hopfield_summary_logreg",
    ]
    return (
        summary_df[summary_df["name"].isin(preferred)]
        .sort_values(["run", "auroc_mean"], ascending=[True, False])
        .reset_index(drop=True)
    )


def _summary_markdown(
    summary_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    *,
    seeds: tuple[int, ...],
) -> str:
    lines = [
        "# Cached Feature Evaluation",
        "",
        f"- CV seeds: `{list(seeds)}`",
        "- Learned probes report mean/std over CV seeds.",
        "- Scalar metrics are deterministic, so their std is zero.",
        "",
        "## Best/Key Methods",
        "",
        "| run | model | dataset | method | AUROC mean | AUROC std | Acc mean | F1 mean |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison_df.itertuples(index=False):
        lines.append(
            "| "
            f"{row.run} | "
            f"`{row.model}` | "
            f"{row.dataset} | "
            f"{row.name} | "
            f"{_fmt(row.auroc_mean)} | "
            f"{_fmt(row.auroc_std)} | "
            f"{_fmt(row.accuracy_mean)} | "
            f"{_fmt(row.f1_mean)} |"
        )

    lines.extend(
        [
            "",
            "## Best AUROC Per Run",
            "",
            "| run | model | dataset | best method | AUROC mean | AUROC std |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    learned_or_scalar = summary_df[summary_df["name"] != "majority"].copy()
    learned_or_scalar = learned_or_scalar.dropna(subset=["auroc_mean"])
    best_rows = learned_or_scalar.loc[learned_or_scalar.groupby("run")["auroc_mean"].idxmax()]
    for row in best_rows.sort_values("run").itertuples(index=False):
        lines.append(
            "| "
            f"{row.run} | "
            f"`{row.model}` | "
            f"{row.dataset} | "
            f"{row.name} | "
            f"{_fmt(row.auroc_mean)} | "
            f"{_fmt(row.auroc_std)} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float) -> str:
    if value != value:
        return "nan"
    return f"{value:.4f}"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
