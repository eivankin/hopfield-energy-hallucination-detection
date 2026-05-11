# Hallucination Hopfield Study

This repository contains a study project on whether attention-derived
modern-Hopfield energy and prompt-attention summaries contain signal for
hallucination or factuality detection in frozen language models.

The project is separate from the SMILES application submission. Here we allow
direct access to attention internals and evaluate the idea scientifically across
datasets and model families.

## Main Result

The strongest feature set combines response NLL, response length, per-layer Q/K
Hopfield-energy summaries, and per-layer prompt-attention summaries. Cached
multi-seed cross-validation gives:

| Setting | Best AUROC mean +/- std |
| --- | ---: |
| SMILES/SQuAD + Qwen2.5-0.5B | 0.723 +/- 0.010 |
| RAGTruth + Qwen2.5-0.5B | 0.707 +/- 0.005 |
| RAGTruth + SmolLM2-360M-Instruct | 0.705 +/- 0.003 |
| RAGTruth + Gemma-3-1B-it | 0.679 +/- 0.003 |
| TruthfulQA + Qwen2.5-0.5B | 0.706 +/- 0.003 |

The main conclusion is limited but positive: Q/K energy and attention-derived
summaries contain transferable hallucination/factuality signal, but scalar
energy is not a universal monotonic hallucination score.

## Setup

```bash
uv sync --group public-data
```

For development utilities:

```bash
uv sync --group dev --group public-data
```

## Data Preparation

The SMILES/SQuAD-derived dataset is expected at:

```text
data/smiles_squad/dataset.csv
```

Convert RAGTruth:

```bash
uv run python scripts/prepare_public_dataset.py \
  --dataset ragtruth \
  --split test \
  --output data/public/ragtruth_test.csv
```

Convert TruthfulQA:

```bash
uv run python scripts/prepare_public_dataset.py \
  --dataset truthfulqa \
  --output data/public/truthfulqa_generation_pairs.csv
```

## Feature Extraction Experiments

SMILES/SQuAD with Qwen:

```bash
uv run python experiments/001_qwen05b_squad_hopfield.py \
  --data data/smiles_squad/dataset.csv \
  --device cuda \
  --output-dir outputs/002_qwen05b_attention_energy
```

RAGTruth with Qwen:

```bash
uv run python experiments/001_qwen05b_squad_hopfield.py \
  --data data/public/ragtruth_test.csv \
  --device cuda \
  --output-dir outputs/003_qwen05b_ragtruth
```

RAGTruth with SmolLM2:

```bash
uv run python experiments/001_qwen05b_squad_hopfield.py \
  --data data/public/ragtruth_test.csv \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --layers 6 12 20 \
  --device cuda \
  --output-dir outputs/004_smollm2_360m_ragtruth
```

RAGTruth with Gemma:

```bash
uv run python experiments/001_qwen05b_squad_hopfield.py \
  --data data/public/ragtruth_test.csv \
  --model google/gemma-3-1b-it \
  --layers 6 12 20 \
  --device cuda \
  --output-dir outputs/005_gemma3_1b_ragtruth
```

TruthfulQA with Qwen:

```bash
uv run python experiments/001_qwen05b_squad_hopfield.py \
  --data data/public/truthfulqa_generation_pairs.csv \
  --device cuda \
  --output-dir outputs/005_qwen05b_truthfulqa
```

Use `--max-samples 16` for a smoke run.

## Cached Evaluation

After feature extraction, rerun probe evaluation over several CV seeds without
loading the language model:

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

## Final Report

Generate report figures:

```bash
uv run python scripts/make_final_report_figures.py \
  --cached-summary outputs/cached_eval/cached_eval_summary.csv \
  --out-dir reports/final_report/figures
```

Compile the report:

```bash
cd reports/final_report
latexmk -pdf main.tex
```

If `latexmk` is unavailable:

```bash
cd reports/final_report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Presentation

The final presentation sources are in `reports/presentation/`.
