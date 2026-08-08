# GitHub Action Model

`GitHub Action Model` is a Python research project for flaky-job prediction on GitHub Actions. It combines structured job features, log similarity retrieval, and linear score fusion to estimate whether a workflow job is likely to be flaky.

## Overview

The project supports three main workflows:

- `tune`: search over predefined `feature_k`, `log_k`, and fusion weights on rolling chronological splits
- `train`: run the rolling evaluation pipeline with fixed hyperparameters
- `single`: run one chronological train/test split and export detailed test predictions

The structured branch currently supports these models:

- `random_forest`
- `svm`
- `mlp`
- `xgboost`

## Method Summary

The pipeline is organized as:

1. Load repository-specific CSV files from `data/csv/job_features` and `data/csv/run_features`
2. Select structured features with mutual information
3. Train a structured classifier
4. Build a log-similarity branch from run-level text features
5. Fuse the structured and log probabilities with a linear weight
6. Report validation and future-window test metrics

For `single` mode, the pipeline runs one chronological split and additionally writes row-level prediction outputs for the test set.

## Repository Layout

```text
Github_Action_Model/
  data/
    csv/
      job_features/
      run_features/
  logs/
  results/
  scripts/
  src/
    branches/
    config/
    data/
    experiment/
    features/
  pyproject.toml
  requirements.txt
  run.py
  README.md
```

## Data Preparation

The CLI expects repository files to follow this layout:

```text
data/csv/job_features/<owner>@<repo>.csv
data/csv/run_features/<owner>@<repo>.csv
```

For example, `--repository apache/accumulo` maps to:

- `data/csv/job_features/apache@accumulo.csv`
- `data/csv/run_features/apache@accumulo.csv`

Important note for the published GitHub snapshot:

- raw files under `data/csv/job_features/*.csv` may be omitted from repository history to keep pushes GitHub-safe
- if you clone the published repository and those files are missing, extract `data/csv/job_features/job_features.zip` into `data/csv/job_features/`
- `data/csv/run_features/run_features.zip` is also included as an archive copy of the run-feature dataset

After extraction, the directory should contain the per-repository CSV files that `run.py` reads directly.

## Installation

Use Python 3.9 or newer.

Install in editable mode:

```bash
pip install -e .
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Install development tools:

```bash
pip install -e .[dev]
```

## Quick Start

Tune hyperparameters for one repository:

```bash
python run.py tune --repository apache/accumulo --model random_forest
```

Run rolling evaluation with fixed parameters:

```bash
python run.py train --repository apache/accumulo --model random_forest --feature-k 10 --log-k 10 --fusion-alpha 0.5
```

Run a single chronological split:

```bash
python run.py single --repository apache/accumulo --model random_forest --feature-k 10 --log-k 10 --fusion-alpha 0.5 --train-size 0.8 --test-size 0.2
```

Run the structured branch only:

```bash
python run.py train --repository apache/accumulo --model random_forest --feature-k 10 --structured-only
```

Set a custom decision threshold:

```bash
python run.py single --repository apache/accumulo --model random_forest --feature-k 10 --log-k 10 --fusion-alpha 0.5 --train-size 0.8 --test-size 0.2 --threshold 0.4
```

## CLI Reference

### `tune`

Use `tune` to search the fixed candidate grids defined in `src/model_registry.py`.

Common arguments:

- `--repository`: repository name like `apache/accumulo`
- `--model`: one of `random_forest`, `svm`, `mlp`, `xgboost`
- `--data-dir`: root directory containing `job_features/` and `run_features/`
- `--output`: optional JSON output path
- `--threshold`: probability threshold for positive prediction
- `--disable-oversample`: turn off train-set oversampling
- `--structured-only`: skip log branch and fusion
- `--train-start`, `--train-end`, `--train-step`: rolling training window controls
- `--valid-size`, `--test-size`: validation and future test window ratios

### `train`

Use `train` when you already know the hyperparameters to evaluate.

Additional required arguments:

- `--feature-k`
- `--log-k`
- `--fusion-alpha`

When `--structured-only` is enabled, only `--feature-k` is required.

### `single`

Use `single` for one chronological train/test split without rolling validation.

Additional required arguments:

- `--feature-k`
- `--train-size`
- `--test-size`

If fusion is enabled, `--log-k` and `--fusion-alpha` are also required.

## Outputs

The program writes:

- console logs
- a timestamped log file in `logs/`, unless `--log-file` is provided
- optional JSON summaries when `--output` is set
- row-level prediction CSV files in `results/` for `single` mode

Typical `single` outputs include:

- `results/*.predictions.csv`
- `results/*.minimal.csv`
- `results/*.rerun.csv`

## Reproducibility Notes

- evaluation is chronological rather than random-split based
- hyperparameter search ranges are fixed in code
- model parameter presets are fixed in code
- validation metrics and rolling future-window metrics are reported separately
- the default repository slug format replaces `/` with `@`

## Useful Files

- `run.py`: CLI entry point
- `src/config/schema.py`: experiment configuration models
- `src/experiment/runner.py`: training and evaluation pipeline
- `src/model_registry.py`: model presets and tuning grids

## License

This repository is released under the MIT License.
