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
