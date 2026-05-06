from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


QUESTION_RE = re.compile(r"Here is the question:\s*(.*?)\n\nYour answer:", re.DOTALL)
CONTEXT_RE = re.compile(
    r"Given the context, answer the question in a single brief but complete sentence\.\n\n"
    r"(.*?)\n\nNote that your answer",
    re.DOTALL,
)


@dataclass(frozen=True)
class Example:
    idx: int
    prompt: str
    response: str
    label: int
    context: str
    question: str

    @property
    def full_text(self) -> str:
        return f"{self.prompt}{self.response}"


def _extract(pattern: re.Pattern[str], text: str) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def load_examples(path: str | Path, max_samples: int = 0) -> list[Example]:
    df = pd.read_csv(path)
    if max_samples and max_samples > 0:
        df = df.head(max_samples).copy()

    examples: list[Example] = []
    for idx, row in df.iterrows():
        label_value = row.get("label")
        if pd.isna(label_value):
            continue
        prompt = str(row["prompt"])
        response = str(row["response"])
        examples.append(
            Example(
                idx=int(idx),
                prompt=prompt,
                response=response,
                label=int(float(label_value)),
                context=_extract(CONTEXT_RE, prompt),
                question=_extract(QUESTION_RE, prompt),
            )
        )
    return examples


def labels_array(examples: list[Example]) -> np.ndarray:
    return np.asarray([ex.label for ex in examples], dtype=np.int64)
