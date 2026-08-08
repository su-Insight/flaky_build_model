from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


DEFAULT_CUTOFF = pd.Timestamp("2024-12-01")


@dataclass(frozen=True)
class ChainInfo:
    waiting_time_seconds: int
    first_started_at: pd.Timestamp
    last_started_at: pd.Timestamp
    attempt_count: int


def _normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            return str(int(float(text)))
        except ValueError:
            return text[:-2]
    try:
        if "." in text:
            number = float(text)
            if number.is_integer():
                return str(int(number))
    except ValueError:
        return text
    return text


def _load_prediction_index(predictions: pd.DataFrame) -> pd.DataFrame:
    if "job_id" not in predictions.columns:
        raise ValueError("Predictions file must include a job_id column.")
    if "run_id" not in predictions.columns or "name" not in predictions.columns:
        raise ValueError("Predictions file must include run_id and name columns.")
    if "started_at" not in predictions.columns:
        raise ValueError("Predictions file must include started_at.")

    indexed = predictions.copy()
    indexed["job_id_key"] = indexed["job_id"].map(_normalize_id)
    indexed["run_id_key"] = indexed["run_id"].map(_normalize_id)
    indexed["started_at_ts"] = pd.to_datetime(indexed["started_at"], errors="coerce")
    indexed["completed_at_ts"] = pd.to_datetime(indexed["completed_at"], errors="coerce")
    return indexed


def _build_chain_map(predictions: pd.DataFrame, cutoff: pd.Timestamp) -> Dict[Tuple[str, str], ChainInfo]:
    indexed = _load_prediction_index(predictions)
    eligible = indexed[indexed["started_at_ts"].notna() & (indexed["started_at_ts"] < cutoff)].copy()
    chain_map: Dict[Tuple[str, str], ChainInfo] = {}

    for (run_id_key, job_name), group in eligible.groupby(["run_id_key", "name"], dropna=False):
        ordered = group.sort_values("started_at_ts")
        first_started_at = ordered["started_at_ts"].iloc[0]
        last_started_at = ordered["started_at_ts"].iloc[-1]
        waiting_time_seconds = 0
        for previous_row, next_row in zip(ordered.itertuples(index=False), ordered.iloc[1:].itertuples(index=False)):
            if pd.isna(previous_row.completed_at_ts) or pd.isna(next_row.started_at_ts):
                continue
            gap_seconds = int((next_row.started_at_ts - previous_row.completed_at_ts).total_seconds())
            if gap_seconds > 0:
                waiting_time_seconds += gap_seconds
        chain_map[(run_id_key, str(job_name))] = ChainInfo(
            waiting_time_seconds=waiting_time_seconds,
            first_started_at=first_started_at,
            last_started_at=last_started_at,
            attempt_count=len(ordered),
        )
    return chain_map


def _augment_rerun_file(
    predictions_path: Path,
    rerun_path: Path,
    output_path: Path,
    cutoff: pd.Timestamp,
) -> None:
    predictions = pd.read_csv(predictions_path)
    rerun = pd.read_csv(rerun_path)

    chain_map = _build_chain_map(predictions, cutoff)
    prediction_index = _load_prediction_index(predictions).set_index("job_id_key", drop=False)

    waiting_seconds = []
    waiting_first_started_at = []
    waiting_last_started_at = []
    waiting_attempt_count = []

    for _, row in rerun.iterrows():
        job_id_key = _normalize_id(row.get("job_id"))
        if not job_id_key or job_id_key not in prediction_index.index:
            waiting_seconds.append(pd.NA)
            waiting_first_started_at.append(pd.NA)
            waiting_last_started_at.append(pd.NA)
            waiting_attempt_count.append(pd.NA)
            continue

        prediction_row = prediction_index.loc[job_id_key]
        if isinstance(prediction_row, pd.DataFrame):
            prediction_row = prediction_row.iloc[0]

        started_at = prediction_row["started_at_ts"]
        if pd.isna(started_at) or started_at >= cutoff:
            waiting_seconds.append(pd.NA)
            waiting_first_started_at.append(pd.NA)
            waiting_last_started_at.append(pd.NA)
            waiting_attempt_count.append(pd.NA)
            continue

        chain_key = (_normalize_id(prediction_row["run_id"]), str(prediction_row["name"]))
        chain_info = chain_map.get(chain_key)
        if chain_info is None:
            waiting_seconds.append(0)
            waiting_first_started_at.append(started_at)
            waiting_last_started_at.append(started_at)
            waiting_attempt_count.append(1)
            continue

        waiting_seconds.append(chain_info.waiting_time_seconds)
        waiting_first_started_at.append(chain_info.first_started_at.isoformat(sep=" "))
        waiting_last_started_at.append(chain_info.last_started_at.isoformat(sep=" "))
        waiting_attempt_count.append(chain_info.attempt_count)

    augmented = rerun.copy()
    augmented["waiting_time_seconds"] = waiting_seconds
    augmented["waiting_first_started_at"] = waiting_first_started_at
    augmented["waiting_last_started_at"] = waiting_last_started_at
    augmented["waiting_attempt_count"] = waiting_attempt_count
    output_path.parent.mkdir(parents=True, exist_ok=True)
    augmented.to_csv(output_path, index=False, encoding="utf-8")


def _iter_result_pairs(results_dir: Path) -> Iterable[Tuple[Path, Path]]:
    for predictions_path in sorted(results_dir.glob("single.*.predictions.csv")):
        rerun_path = predictions_path.with_name(predictions_path.name.replace(".predictions.csv", ".rerun.csv"))
        if rerun_path.exists():
            yield predictions_path, rerun_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Add waiting time columns to rerun result files.")
    parser.add_argument("--results-dir", default="results", help="Directory containing prediction/rerun CSV files.")
    parser.add_argument(
        "--cutoff",
        default="2024-12-01",
        help="Only use attempts started before this date when computing waiting time.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the original rerun CSV files instead of creating *.waiting.csv copies.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cutoff = pd.Timestamp(args.cutoff)

    updated = 0
    for predictions_path, rerun_path in _iter_result_pairs(results_dir):
        output_path = rerun_path if args.in_place else rerun_path.with_name(rerun_path.name.replace(".csv", ".waiting.csv"))
        _augment_rerun_file(predictions_path, rerun_path, output_path, cutoff)
        updated += 1

    print(f"Updated {updated} rerun file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
