# Hallucination Hopfield Study

Separate study-project repository for testing whether attention-derived
modern-Hopfield energy contains signal for hallucination detection.

The first experiment is intentionally small and controlled:

- dataset: the SMILES/SQuAD-derived `data/dataset.csv`
- model: `Qwen/Qwen2.5-0.5B`
- hallucination type: context-grounded unsupported/contradictory answers
- baseline: response negative log-likelihood
- proposed signal: Q/K log-sum-exp Hopfield energy from selected attention layers

## Setup

```bash
uv sync
```

## First Experiment

From this repository:

```bash
uv run python experiments/001_qwen05b_squad_hopfield.py \
  --data ./data/smiles_squad/dataset.csv \
  --device cuda
```

Use `--max-samples 16` for a smoke run.

Outputs are written to `outputs/001_qwen05b_squad_hopfield/`.

## Public Dataset Checks

Install the extra dataset loader dependency:

```bash
uv sync --group public-data
```

Convert RAGTruth test examples to the local CSV format:

```bash
uv run python scripts/prepare_public_dataset.py \
  --dataset ragtruth \
  --split test \
  --output data/public/ragtruth_test.csv
```

Convert TruthfulQA generation examples to paired correct/incorrect answer rows:

```bash
uv run python scripts/prepare_public_dataset.py \
  --dataset truthfulqa \
  --output data/public/truthfulqa_generation_pairs.csv
```

Then run the same feature extraction and evaluation script:

```bash
uv run python experiments/001_qwen05b_squad_hopfield.py \
  --data data/public/ragtruth_test.csv \
  --device cuda \
  --output-dir outputs/003_qwen05b_ragtruth
```

For a small non-Qwen check, try TinyLlama with explicit valid layer indices:

```bash
uv run python experiments/001_qwen05b_squad_hopfield.py \
  --data data/public/ragtruth_test.csv \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --layers 10 15 20 \
  --device cuda \
  --output-dir outputs/004_tinyllama_ragtruth
```

## Cached Evaluation

After feature extraction has finished, rerun all probe evaluations over several
CV seeds without loading the language model again:

```bash
uv run python scripts/evaluate_cached_features.py \
  --outputs-root outputs \
  --out-dir outputs/cached_eval
```

The script writes:

- `outputs/cached_eval/cached_eval_long.csv`
- `outputs/cached_eval/cached_eval_summary.csv`
- `outputs/cached_eval/cached_eval_best.csv`
- `outputs/cached_eval/cached_eval_summary.md`
