from __future__ import annotations

import itertools
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Union

import pandas as pd

from branches.fusion import fuse_branch_probabilities
from branches.log_branch import run_log_branch
from branches.structured_branch import MODEL_RUNNERS, compute_binary_classification_metrics
from config.schema import SingleRunConfig, TrainConfig, TuneConfig
from data.csv_repository import CSVRepositoryDataSource, ExperimentDataSource
from experiment.splitter import split_dataset, split_single_dataset
from features.selection import select_structured_features
from features.structural_preprocessing import (
    DROP_COLUMNS,
    SplitAwarePreprocessor,
    enhance_with_rerun_features,
    keep_failures_only,
    oversample_by_status_and_flaky,
    prepare_base_dataframe,
)
from model_registry import (
    DEFAULT_SINGLE_PARAMS,
    TUNE_FEATURE_K_VALUES,
    TUNE_FUSION_ALPHA_VALUES,
    TUNE_LOG_K_VALUES,
)


class _ProgressBar:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = max(total_steps, 1)
        self.current_step = 0

    def update(self, candidate_index: int, candidate_total: int, group_name: str) -> None:
        self.current_step += 1
        width = 24
        filled = int(width * self.current_step / self.total_steps)
        bar = "#" * filled + "-" * (width - filled)
        message = (
            f"\rProgress [{bar}] {self.current_step}/{self.total_steps} "
            f"| candidate {candidate_index}/{candidate_total} | group {group_name}"
        )
        sys.stdout.write(message)
        sys.stdout.flush()

    def finish(self) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def build_structured_branch_sets(
    raw_sets: Dict[str, pd.DataFrame],
    feature_k: int,
    oversample_train: bool,
) -> Dict[str, pd.DataFrame]:
    enhanced = {
        split_name: enhance_with_rerun_features(split_df.copy())
        for split_name, split_df in raw_sets.items()
    }
    preprocessor = SplitAwarePreprocessor().fit(enhanced["train"])
    processed = {split_name: preprocessor.transform(split_df) for split_name, split_df in enhanced.items()}
    if oversample_train:
        processed["train"] = oversample_by_status_and_flaky(processed["train"])
    dropped = {
        split_name: split_df.drop(columns=DROP_COLUMNS, errors="ignore")
        for split_name, split_df in processed.items()
    }
    selected_features = select_structured_features(dropped["train"], feature_k)
    selected_columns = selected_features + ["flaky"]
    return {split_name: split_df[selected_columns].copy() for split_name, split_df in dropped.items()}


def _candidate_param_sets(config: Union[TuneConfig, TrainConfig]) -> Iterable[Dict[str, Any]]:
    if isinstance(config, TrainConfig):
        yield {
            "model_params": DEFAULT_SINGLE_PARAMS[config.model],
            "feature_k": config.feature_k,
            "log_k": config.log_k,
        }
        return

    if config.structured_only:
        for feature_k in TUNE_FEATURE_K_VALUES:
            yield {
                "model_params": DEFAULT_SINGLE_PARAMS[config.model],
                "feature_k": feature_k,
                "log_k": None,
            }
        return

    for feature_k, log_k in itertools.product(TUNE_FEATURE_K_VALUES, TUNE_LOG_K_VALUES):
        yield {
            "model_params": DEFAULT_SINGLE_PARAMS[config.model],
            "feature_k": feature_k,
            "log_k": log_k,
        }


def _fusion_alpha_values(config: Union[TuneConfig, TrainConfig]) -> List[float]:
    if config.structured_only:
        return [1.0]
    if isinstance(config, TrainConfig):
        return [config.fusion_alpha]
    return TUNE_FUSION_ALPHA_VALUES.copy()


def _aggregate_metric(group_rows: List[Dict[str, Any]], metric_group: str, metric_name: str) -> float:
    return sum(group[metric_group][metric_name] for group in group_rows) / len(group_rows)


def _repository_slug(repository: str) -> str:
    return repository.replace("/", "@").replace("\\", "@")


def _single_predictions_output_path(result: Dict[str, Any], output_path: str | None) -> Path:
    config = result["config"]
    if output_path:
        return Path(output_path).with_suffix(".predictions.csv")
    log_path = config.get("log_path")
    if log_path:
        return Path("results") / f"{Path(log_path).stem}.predictions.csv"
    return Path("results") / f"single.{_repository_slug(config['repository'])}.predictions.csv"


def run_experiment(
    config: Union[TuneConfig, TrainConfig],
    data_source: ExperimentDataSource | None = None,
) -> Dict[str, Any]:
    data_source = data_source or CSVRepositoryDataSource(config.data_dir)
    base_df = prepare_base_dataframe(data_source.load_repository(config.repository))
    base_df = keep_failures_only(base_df)
    raw_split_sets = split_dataset(base_df, config.split)
    if not raw_split_sets:
        raise ValueError(f"No valid temporal splits were generated for repository {config.repository}.")

    candidate_sets = list(_candidate_param_sets(config))
    progress_bar = _ProgressBar(len(candidate_sets) * len(raw_split_sets))
    all_results = []
    for candidate_index, candidate in enumerate(candidate_sets, start=1):
        logging.debug(
            "Evaluating repo=%s model=%s feature_k=%s log_k=%s params=%s",
            config.repository,
            config.model,
            candidate["feature_k"],
            candidate["log_k"],
            candidate["model_params"],
        )
        alpha_values = _fusion_alpha_values(config)
        alpha_group_rows: Dict[float, List[Dict[str, Any]]] = {alpha: [] for alpha in alpha_values}
        alpha_validation_scores: Dict[float, List[float]] = {alpha: [] for alpha in alpha_values}
        alpha_validation_aucs: Dict[float, List[float]] = {alpha: [] for alpha in alpha_values}
        for group_name, raw_sets in raw_split_sets.items():
            structured_sets = build_structured_branch_sets(
                raw_sets,
                int(candidate["feature_k"]),
                config.oversample_train,
            )
            structured_runner = MODEL_RUNNERS[config.model]
            structured_result = structured_runner(structured_sets, candidate["model_params"])
            structured_validation_metrics = compute_binary_classification_metrics(
                structured_result.y_valid,
                structured_result.validation_proba,
                threshold=config.threshold,
            )
            structured_test_metrics = compute_binary_classification_metrics(
                structured_result.y_test,
                structured_result.test_proba,
                threshold=config.threshold,
            )
            if config.structured_only:
                for fusion_alpha in alpha_values:
                    alpha_validation_scores[fusion_alpha].append(structured_validation_metrics["f1"])
                    alpha_validation_aucs[fusion_alpha].append(structured_validation_metrics["auc"])
                    alpha_group_rows[fusion_alpha].append(
                        {
                            "group": group_name,
                            "feature_k": candidate["feature_k"],
                            "log_k": None,
                            "fusion_alpha": fusion_alpha,
                            "fusion_beta": 0.0,
                            "model_params": candidate["model_params"],
                            "fused_validation_metrics": structured_validation_metrics,
                            "fused_test_metrics": structured_test_metrics,
                            "structured_validation_metrics": structured_validation_metrics,
                            "structured_test_metrics": structured_test_metrics,
                        }
                    )
                progress_bar.update(candidate_index, len(candidate_sets), group_name)
                continue

            log_result = run_log_branch(
                raw_sets["train"],
                raw_sets["valid"],
                raw_sets["test"],
                int(candidate["log_k"]),
            )
            for fusion_alpha in alpha_values:
                fused_valid = fuse_branch_probabilities(
                    structured_result.validation_proba,
                    log_result.validation_proba,
                    float(fusion_alpha),
                )
                fused_test = fuse_branch_probabilities(
                    structured_result.test_proba,
                    log_result.test_proba,
                    float(fusion_alpha),
                )

                fused_valid_metrics = compute_binary_classification_metrics(
                    log_result.y_valid,
                    fused_valid,
                    threshold=config.threshold,
                )
                fused_test_metrics = compute_binary_classification_metrics(
                    log_result.y_test,
                    fused_test,
                    threshold=config.threshold,
                )
                alpha_validation_scores[fusion_alpha].append(fused_valid_metrics["f1"])
                alpha_validation_aucs[fusion_alpha].append(fused_valid_metrics["auc"])
                alpha_group_rows[fusion_alpha].append(
                    {
                        "group": group_name,
                        "feature_k": candidate["feature_k"],
                        "log_k": candidate["log_k"],
                        "fusion_alpha": fusion_alpha,
                        "fusion_beta": 1.0 - float(fusion_alpha),
                        "model_params": candidate["model_params"],
                        "fused_validation_metrics": fused_valid_metrics,
                        "fused_test_metrics": fused_test_metrics,
                        "structured_validation_metrics": structured_validation_metrics,
                        "structured_test_metrics": structured_test_metrics,
                    }
                )
            progress_bar.update(candidate_index, len(candidate_sets), group_name)

        for fusion_alpha in alpha_values:
            group_rows = alpha_group_rows[fusion_alpha]
            average_validation_f1 = sum(alpha_validation_scores[fusion_alpha]) / len(alpha_validation_scores[fusion_alpha])
            average_validation_auc = sum(alpha_validation_aucs[fusion_alpha]) / len(alpha_validation_aucs[fusion_alpha])
            average_rolling_test_f1 = _aggregate_metric(group_rows, "fused_test_metrics", "f1")
            average_rolling_test_auc = _aggregate_metric(group_rows, "fused_test_metrics", "auc")
            all_results.append(
                {
                    "repository": config.repository,
                    "model": config.model,
                    "candidate": {
                        **candidate,
                        "fusion_alpha": fusion_alpha,
                    },
                    "average_validation_f1": average_validation_f1,
                    "average_validation_auc": average_validation_auc,
                    "average_rolling_test_f1": average_rolling_test_f1,
                    "average_rolling_test_auc": average_rolling_test_auc,
                    "average_test_f1": average_rolling_test_f1,
                    "average_test_auc": average_rolling_test_auc,
                    "groups": group_rows,
                }
            )
    progress_bar.finish()

    # Select the configuration using only validation-side signals so the
    # rolling future windows remain purely evaluative within each round.
    best_result = max(all_results, key=lambda row: (row["average_validation_f1"], row["average_validation_auc"]))
    return {
        "config": asdict(config),
        "best_result": best_result,
        "all_results": all_results if isinstance(config, TuneConfig) else None,
    }


def run_single_experiment(
    config: SingleRunConfig,
    data_source: ExperimentDataSource | None = None,
) -> Dict[str, Any]:
    data_source = data_source or CSVRepositoryDataSource(config.data_dir)
    base_df = prepare_base_dataframe(data_source.load_repository(config.repository))
    base_df = keep_failures_only(base_df)
    single_split = split_single_dataset(base_df, config.train_size, config.test_size)

    raw_sets = {
        "train": single_split["train"],
        # Branch interfaces expect three splits. Single mode does not use a
        # validation loop or validation-based selection; the test split is
        # reused as a placeholder eval split so we can keep the branch code simple.
        "valid": single_split["test"],
        "test": single_split["test"],
    }
    candidate = {
        "model_params": DEFAULT_SINGLE_PARAMS[config.model],
        "feature_k": config.feature_k,
        "log_k": config.log_k,
        "fusion_alpha": config.fusion_alpha,
    }
    structured_sets = build_structured_branch_sets(
        raw_sets,
        int(candidate["feature_k"]),
        config.oversample_train,
    )
    structured_runner = MODEL_RUNNERS[config.model]
    structured_result = structured_runner(structured_sets, candidate["model_params"])
    selected_structured_features = [column for column in structured_sets["test"].columns if column != "flaky"]
    structured_test_metrics = compute_binary_classification_metrics(
        structured_result.y_test,
        structured_result.test_proba,
        threshold=config.threshold,
    )
    prediction_rows = single_split["test"].reset_index(drop=True).copy()
    prediction_rows["structured_probability"] = structured_result.test_proba
    if config.structured_only:
        fused_test_metrics = structured_test_metrics
        prediction_rows["prediction_proba"] = structured_result.test_proba
        prediction_rows["prediction"] = (structured_result.test_proba >= config.threshold).astype(int)
    else:
        log_result = run_log_branch(
            raw_sets["train"],
            raw_sets["valid"],
            raw_sets["test"],
            int(candidate["log_k"]),
        )
        fused_test = fuse_branch_probabilities(
            structured_result.test_proba,
            log_result.test_proba,
            float(candidate["fusion_alpha"]),
        )
        fused_test_metrics = compute_binary_classification_metrics(
            log_result.y_test,
            fused_test,
            threshold=config.threshold,
        )
        prediction_rows["log_probability"] = log_result.test_proba
        prediction_rows["prediction_proba"] = fused_test
        prediction_rows["prediction"] = (fused_test >= config.threshold).astype(int)
    prediction_rows["threshold"] = config.threshold
    prediction_rows["model"] = config.model
    prediction_rows["feature_k"] = config.feature_k
    if config.log_k is not None:
        prediction_rows["log_k"] = config.log_k
    if config.fusion_alpha is not None:
        prediction_rows["fusion_alpha"] = config.fusion_alpha
    return {
        "config": asdict(config),
        "single_result": {
            "repository": config.repository,
            "model": config.model,
            "candidate": candidate,
            "selected_structured_features": selected_structured_features,
            "split": {
                "train_size": config.train_size,
                "test_size": config.test_size,
                "train_rows": int(len(single_split["train"])),
                "test_rows": int(len(single_split["test"])),
            },
            "structured_test_metrics": structured_test_metrics,
            "fused_test_metrics": fused_test_metrics,
            "prediction_rows": prediction_rows.to_dict(orient="records"),
        },
    }


def save_result(result: Dict[str, Any], output_path: str | None) -> None:
    if "single_result" in result and "prediction_rows" in result["single_result"]:
        prediction_path = _single_predictions_output_path(result, output_path)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_rows = result["single_result"]["prediction_rows"]
        pd.DataFrame(prediction_rows).to_csv(prediction_path, index=False, encoding="utf-8")
        result["single_result"]["predictions_path"] = str(prediction_path)
        logging.info("Saved single predictions to %s", prediction_path)

    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    json_result = result
    if "single_result" in result and "prediction_rows" in result["single_result"]:
        json_result = dict(result)
        single_result = dict(result["single_result"])
        single_result.pop("prediction_rows", None)
        json_result["single_result"] = single_result
    with open(path, "w", encoding="utf-8") as file:
        json.dump(json_result, file, ensure_ascii=False, indent=2)
    logging.info("Saved result to %s", path)
