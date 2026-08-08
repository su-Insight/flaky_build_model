from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config.schema import SingleRunConfig, SplitConfig, TrainConfig, TuneConfig
from model_registry import SUPPORTED_MODELS


def _slugify_repository(repository: str) -> str:
    return repository.replace("/", "@").replace("\\", "@")


def _default_log_path(command: str, repository: str, output_path: Optional[str]) -> str:
    if output_path:
        output = Path(output_path)
        return str(output.with_suffix(".log"))

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{command}.{_slugify_repository(repository)}.{timestamp}.log"
    return str(Path("logs") / filename)


def _resolve_log_path(
    command: str,
    repository: str,
    cli_log_file: Optional[str],
    output_path: Optional[str],
) -> str:
    if cli_log_file:
        return cli_log_file
    return _default_log_path(command, repository, output_path)


def _configure_logging(log_level_name: str, log_path: str) -> None:
    level = getattr(logging, log_level_name.upper(), logging.INFO)
    log_file = Path(log_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


def _resolve_oversample_flag(disable_oversample: bool) -> bool:
    return not disable_oversample


def _build_split_config(args: argparse.Namespace) -> SplitConfig:
    return SplitConfig(
        train_start=args.train_start if args.train_start is not None else 0.5,
        train_end=args.train_end if args.train_end is not None else 0.9,
        step=args.train_step if args.train_step is not None else 0.1,
        valid_size=args.valid_size if args.valid_size is not None else 0.05,
        test_size=args.test_size if args.test_size is not None else 0.05,
    )


def _build_tune_config(args: argparse.Namespace) -> TuneConfig:
    if not args.repository:
        raise ValueError("tune mode requires --repository.")
    split = _build_split_config(args)
    output_path = args.output
    return TuneConfig(
        repository=args.repository,
        model=args.model or "random_forest",
        oversample_train=_resolve_oversample_flag(args.disable_oversample),
        threshold=args.threshold if args.threshold is not None else 0.5,
        structured_only=args.structured_only,
        split=split,
        data_dir=args.data_dir or "data/csv",
        output_path=output_path,
        log_path=_resolve_log_path("tune", args.repository, args.log_file, output_path),
    )


def _build_train_config(args: argparse.Namespace) -> TrainConfig:
    if not args.repository:
        raise ValueError("train mode requires --repository.")
    split = _build_split_config(args)
    output_path = args.output
    feature_k = args.feature_k
    log_k = args.log_k
    fusion_alpha = args.fusion_alpha
    if feature_k is None:
        raise ValueError("train mode requires --feature-k.")
    if not args.structured_only and (log_k is None or fusion_alpha is None):
        raise ValueError("train mode requires --feature-k, --log-k, and --fusion-alpha.")
    return TrainConfig(
        repository=args.repository,
        model=args.model or "random_forest",
        feature_k=feature_k,
        log_k=log_k,
        fusion_alpha=fusion_alpha,
        oversample_train=_resolve_oversample_flag(args.disable_oversample),
        threshold=args.threshold if args.threshold is not None else 0.5,
        structured_only=args.structured_only,
        split=split,
        data_dir=args.data_dir or "data/csv",
        output_path=output_path,
        log_path=_resolve_log_path("train", args.repository, args.log_file, output_path),
    )


def _build_single_config(args: argparse.Namespace) -> SingleRunConfig:
    if not args.repository:
        raise ValueError("single mode requires --repository.")
    feature_k = args.feature_k
    log_k = args.log_k
    fusion_alpha = args.fusion_alpha
    train_size = args.train_size
    test_size = args.test_size
    if feature_k is None:
        raise ValueError("single mode requires --feature-k.")
    if not args.structured_only and (log_k is None or fusion_alpha is None):
        raise ValueError("single mode requires --feature-k, --log-k, and --fusion-alpha.")
    if train_size is None or test_size is None:
        raise ValueError("single mode requires --train-size and --test-size.")
    output_path = args.output
    return SingleRunConfig(
        repository=args.repository,
        model=args.model or "random_forest",
        feature_k=feature_k,
        log_k=log_k,
        fusion_alpha=fusion_alpha,
        train_size=train_size,
        test_size=test_size,
        oversample_train=_resolve_oversample_flag(args.disable_oversample),
        threshold=args.threshold if args.threshold is not None else 0.5,
        structured_only=args.structured_only,
        data_dir=args.data_dir or "data/csv",
        output_path=output_path,
        log_path=_resolve_log_path("single", args.repository, args.log_file, output_path),
    )


def _log_summary(result: Dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    best = result["best_result"]
    candidate = best["candidate"]
    metrics_key = "structured_test_metrics" if result["config"].get("structured_only") else "fused_test_metrics"
    logger.info("")
    logger.info("=== Best Validation Configuration ===")
    logger.info("repository=%s", best["repository"])
    logger.info("model=%s", best["model"])
    logger.info("feature_k=%s", candidate["feature_k"])
    if candidate.get("log_k") is not None:
        logger.info("log_k=%s", candidate["log_k"])
    if candidate.get("fusion_alpha") is not None:
        logger.info("fusion_alpha=%s", candidate["fusion_alpha"])
        logger.info("fusion_beta=%s", 1.0 - float(candidate["fusion_alpha"]))
    logger.info("threshold=%s", result["config"]["threshold"])
    logger.info("fixed_model_params=%s", candidate["model_params"])
    logger.info("avg_validation_f1=%.4f", best["average_validation_f1"])
    logger.info("avg_validation_auc=%.4f", best["average_validation_auc"])
    logger.info("avg_rolling_test_f1=%.4f", best["average_rolling_test_f1"])
    logger.info("avg_rolling_test_auc=%.4f", best["average_rolling_test_auc"])

    logger.info("")
    logger.info("=== Rolling Group Results ===")
    for group_row in best["groups"]:
        fused = group_row[metrics_key]
        logger.info(
            "%s | rolling_test_f1=%.4f | precision=%.4f | recall=%.4f | auc=%.4f | label1_cm(tp=%s, fp=%s, fn=%s, tn=%s)",
            group_row["group"],
            fused["f1"],
            fused["precision"],
            fused["recall"],
            fused["auc"],
            fused["tp"],
            fused["fp"],
            fused["fn"],
            fused["tn"],
        )


def _log_single_summary(result: Dict[str, Any]) -> None:
    logger = logging.getLogger(__name__)
    single_result = result["single_result"]
    candidate = single_result["candidate"]
    metrics_key = "structured_test_metrics" if result["config"].get("structured_only") else "fused_test_metrics"
    test_metrics = single_result[metrics_key]
    split = single_result["split"]
    logger.info("")
    logger.info("=== Single Test Result ===")
    logger.info("repository=%s", single_result["repository"])
    logger.info("model=%s", single_result["model"])
    logger.info("train_size=%.4f", split["train_size"])
    logger.info("test_size=%.4f", split["test_size"])
    logger.info("train_rows=%s", split["train_rows"])
    logger.info("test_rows=%s", split["test_rows"])
    logger.info("selected_structured_feature_count=%s", len(single_result["selected_structured_features"]))
    logger.info("feature_k=%s", candidate["feature_k"])
    if candidate.get("log_k") is not None:
        logger.info("log_k=%s", candidate["log_k"])
    if candidate.get("fusion_alpha") is not None:
        logger.info("fusion_alpha=%s", candidate["fusion_alpha"])
        logger.info("fusion_beta=%s", 1.0 - float(candidate["fusion_alpha"]))
    logger.info("threshold=%s", result["config"]["threshold"])
    logger.info("fixed_model_params=%s", candidate["model_params"])
    if single_result.get("predictions_path"):
        logger.info("predictions_path=%s", single_result["predictions_path"])
    logger.info(
        "test_f1=%.4f | precision=%.4f | recall=%.4f | auc=%.4f | label1_cm(tp=%s, fp=%s, fn=%s, tn=%s)",
        test_metrics["f1"],
        test_metrics["precision"],
        test_metrics["recall"],
        test_metrics["auc"],
        test_metrics["tp"],
        test_metrics["fp"],
        test_metrics["fn"],
        test_metrics["tn"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="Publication-grade CLI for structured + log + linear-fusion flaky-job prediction.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune_parser = subparsers.add_parser("tune", help="Tune hyperparameters for a specified repository.")
    tune_parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    tune_parser.add_argument("--log-file", help="Optional log file path. Defaults to logs/ or the output path stem.")
    tune_parser.add_argument("--repository", help="Repository name like apache/accumulo.")
    tune_parser.add_argument("--model", choices=SUPPORTED_MODELS, help="Structured model type.")
    tune_parser.add_argument("--data-dir", help="Root directory containing job_features/ and run_features/ CSV files.")
    tune_parser.add_argument("--output", help="Optional JSON result output path.")
    tune_parser.add_argument("--disable-oversample", action="store_true", help="Disable train-set oversampling.")
    tune_parser.add_argument("--structured-only", action="store_true", help="Run only the structured branch without log modeling or fusion.")
    tune_parser.add_argument("--threshold", type=float, help="Classification threshold for converting probability to label.")
    tune_parser.add_argument("--train-start", type=float, help="Rolling train window start ratio.")
    tune_parser.add_argument("--train-end", type=float, help="Rolling train window end ratio.")
    tune_parser.add_argument("--train-step", type=float, help="Rolling train window step size.")
    tune_parser.add_argument("--valid-size", type=float, help="Validation window ratio.")
    tune_parser.add_argument("--test-size", type=float, help="Test window ratio.")

    train_parser = subparsers.add_parser("train", help="Run fixed-parameter training for a specified repository.")
    train_parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    train_parser.add_argument("--log-file", help="Optional log file path. Defaults to logs/ or the output path stem.")
    train_parser.add_argument("--repository", help="Repository name like apache/accumulo.")
    train_parser.add_argument("--model", choices=SUPPORTED_MODELS, help="Structured model type.")
    train_parser.add_argument("--feature-k", type=int, help="Structured feature count selected by mutual information.")
    train_parser.add_argument("--log-k", type=int, help="Log similarity K value.")
    train_parser.add_argument("--fusion-alpha", type=float, help="Structured branch weight in linear fusion.")
    train_parser.add_argument("--data-dir", help="Root directory containing job_features/ and run_features/ CSV files.")
    train_parser.add_argument("--output", help="Optional JSON result output path.")
    train_parser.add_argument("--disable-oversample", action="store_true", help="Disable train-set oversampling.")
    train_parser.add_argument("--structured-only", action="store_true", help="Run only the structured branch without log modeling or fusion.")
    train_parser.add_argument("--threshold", type=float, help="Classification threshold for converting probability to label.")
    train_parser.add_argument("--train-start", type=float, help="Rolling train window start ratio.")
    train_parser.add_argument("--train-end", type=float, help="Rolling train window end ratio.")
    train_parser.add_argument("--train-step", type=float, help="Rolling train window step size.")
    train_parser.add_argument("--valid-size", type=float, help="Validation window ratio.")
    train_parser.add_argument("--test-size", type=float, help="Test window ratio.")

    single_parser = subparsers.add_parser("single", help="Run one fixed train/test split without tuning or rolling validation.")
    single_parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    single_parser.add_argument("--log-file", help="Optional log file path. Defaults to logs/ or the output path stem.")
    single_parser.add_argument("--repository", help="Repository name like apache/accumulo.")
    single_parser.add_argument("--model", choices=SUPPORTED_MODELS, help="Structured model type.")
    single_parser.add_argument("--feature-k", type=int, help="Structured feature count selected by mutual information.")
    single_parser.add_argument("--log-k", type=int, help="Log similarity K value.")
    single_parser.add_argument("--fusion-alpha", type=float, help="Structured branch weight in linear fusion.")
    single_parser.add_argument("--train-size", type=float, help="Single-run train split ratio.")
    single_parser.add_argument("--test-size", type=float, help="Single-run test split ratio.")
    single_parser.add_argument("--data-dir", help="Root directory containing job_features/ and run_features/ CSV files.")
    single_parser.add_argument("--output", help="Optional JSON result output path.")
    single_parser.add_argument("--disable-oversample", action="store_true", help="Disable train-set oversampling.")
    single_parser.add_argument("--structured-only", action="store_true", help="Run only the structured branch without log modeling or fusion.")
    single_parser.add_argument("--threshold", type=float, help="Classification threshold for converting probability to label.")

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        from experiment.runner import run_experiment, run_single_experiment, save_result

        if args.command == "tune":
            config = _build_tune_config(args)
            runner = run_experiment
            summary_logger = _log_summary
        elif args.command == "train":
            config = _build_train_config(args)
            runner = run_experiment
            summary_logger = _log_summary
        else:
            config = _build_single_config(args)
            runner = run_single_experiment
            summary_logger = _log_single_summary
        _configure_logging(
            args.log_level,
            config.log_path or _default_log_path(args.command, config.repository, config.output_path),
        )
        logging.getLogger(__name__).info(
            "Starting %s for repository=%s model=%s",
            args.command,
            config.repository,
            config.model,
        )
        logging.getLogger(__name__).info("Log file: %s", config.log_path)
        if config.output_path:
            logging.getLogger(__name__).info("Result file: %s", config.output_path)
        logging.getLogger(__name__).debug("Resolved config: %s", config)
        result = runner(config)
        save_result(result, config.output_path)
        summary_logger(result)
        return 0
    except Exception as exc:
        logging.error("%s", exc)
        logging.debug("CLI execution failed", exc_info=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
