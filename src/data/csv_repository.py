from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd


DEFAULT_DATA_DIR = Path("data/csv")


class ExperimentDataSource(Protocol):
    def load_repository(self, repository: str) -> pd.DataFrame:
        ...


@dataclass
class DataFrameDataSource:
    dataframe: pd.DataFrame

    def load_repository(self, repository: str) -> pd.DataFrame:
        del repository
        return self.dataframe.copy()


@dataclass
class CSVRepositoryDataSource:
    data_dir: Path | str = DEFAULT_DATA_DIR

    def load_repository(self, repository: str) -> pd.DataFrame:
        if not repository or not repository.strip():
            raise ValueError("repository must be provided when loading data.")

        data_dir = Path(self.data_dir)
        slug = repository.replace("/", "@")
        job_path = data_dir / "job_features" / f"{slug}.csv"
        run_path = data_dir / "run_features" / f"{slug}.csv"

        if not job_path.exists():
            raise FileNotFoundError(f"job_features CSV not found for repository '{repository}': {job_path}")
        if not run_path.exists():
            raise FileNotFoundError(f"run_features CSV not found for repository '{repository}': {run_path}")

        job_df = pd.read_csv(job_path)
        run_df = pd.read_csv(run_path)
        if job_df.empty:
            raise ValueError(f"job_features CSV is empty for repository '{repository}'.")
        if run_df.empty:
            raise ValueError(f"run_features CSV is empty for repository '{repository}'.")

        return self._merge_frames(job_df, run_df, repository)

    @staticmethod
    def _merge_frames(job_df: pd.DataFrame, run_df: pd.DataFrame, repository: str) -> pd.DataFrame:
        if "run_id" not in job_df.columns or "run_id" not in run_df.columns:
            raise ValueError(f"Both CSV tables must include run_id for repository '{repository}'.")

        run_feature_columns = ["run_id"]
        run_feature_columns.extend(
            column for column in run_df.columns if column != "run_id" and column not in job_df.columns
        )
        merged = job_df.merge(run_df[run_feature_columns], on="run_id", how="inner")
        if merged.empty:
            raise ValueError(f"Merging job_features and run_features yielded no rows for '{repository}'.")
        return merged
