from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert public hallucination/factuality datasets to prompt,response,label CSV."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["ragtruth", "truthfulqa"],
        help="Public dataset adapter to use.",
    )
    parser.add_argument("--split", default=None, help="Dataset split. Defaults depend on adapter.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 means keep all converted rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ragtruth-task",
        default=None,
        choices=["QA", "Summary", "Data-to-text"],
        help="Optional RAGTruth task_type filter.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.dataset == "ragtruth":
        rows = build_ragtruth_rows(
            split=args.split or "test",
            max_samples=args.max_samples,
            task_type=args.ragtruth_task,
        )
    elif args.dataset == "truthfulqa":
        rows = build_truthfulqa_rows(
            split=args.split or "validation",
            max_samples=args.max_samples,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["prompt", "response", "label", "source_id", "source_dataset"]).to_csv(
        output,
        index=False,
    )
    labels = pd.Series([row["label"] for row in rows]).value_counts().sort_index().to_dict()
    print(f"Wrote {len(rows)} rows to {output}")
    print(f"Labels: {labels}")


def build_ragtruth_rows(
    *,
    split: str,
    max_samples: int,
    task_type: str | None,
) -> list[dict[str, Any]]:
    datasets = _import_datasets()
    dataset = datasets.load_dataset("wandb/RAGTruth-processed", split=split)
    rows: list[dict[str, Any]] = []

    for item in dataset:
        if task_type is not None and item.get("task_type") != task_type:
            continue
        labels = item.get("hallucination_labels_processed") or {}
        label = int(
            int(labels.get("evident_conflict", 0) or 0) > 0
            or int(labels.get("baseless_info", 0) or 0) > 0
        )
        query = str(item.get("query") or "").strip()
        context = str(item.get("context") or "").strip()
        output = str(item.get("output") or "").strip()
        if not query or not output:
            continue
        prompt = _context_prompt(context=context, question=query)
        rows.append(
            {
                "prompt": prompt,
                "response": output,
                "label": label,
                "source_id": str(item.get("id") or len(rows)),
                "source_dataset": "wandb/RAGTruth-processed",
            }
        )
        if max_samples and len(rows) >= max_samples:
            break
    return rows


def build_truthfulqa_rows(*, split: str, max_samples: int) -> list[dict[str, Any]]:
    datasets = _import_datasets()
    dataset = datasets.load_dataset("truthfulqa/truthful_qa", "generation", split=split)
    rows: list[dict[str, Any]] = []

    for item_idx, item in enumerate(dataset):
        question = str(item["question"]).strip()
        correct_answers = [str(answer).strip() for answer in item["correct_answers"] if answer]
        incorrect_answers = [str(answer).strip() for answer in item["incorrect_answers"] if answer]
        if not question or not correct_answers or not incorrect_answers:
            continue

        rows.append(
            {
                "prompt": _open_domain_prompt(question),
                "response": correct_answers[0],
                "label": 0,
                "source_id": f"{item_idx}:correct",
                "source_dataset": "truthfulqa/truthful_qa:generation",
            }
        )
        rows.append(
            {
                "prompt": _open_domain_prompt(question),
                "response": incorrect_answers[0],
                "label": 1,
                "source_id": f"{item_idx}:incorrect",
                "source_dataset": "truthfulqa/truthful_qa:generation",
            }
        )
        if max_samples and len(rows) >= max_samples:
            rows = rows[:max_samples]
            break
    return rows


def _context_prompt(*, context: str, question: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Given the context, answer the question in a single brief but complete sentence.\n\n"
        f"{context}\n\n"
        f"Here is the question: {question}\n\n"
        "Your answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _open_domain_prompt(question: str) -> str:
    return (
        "<|im_start|>system\n"
        "You are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n"
        "Answer the question in a single brief but complete sentence.\n\n"
        f"Here is the question: {question}\n\n"
        "Your answer:<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _import_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: install the public-data group with `uv sync --group public-data`."
        ) from exc
    return datasets


if __name__ == "__main__":
    main()
