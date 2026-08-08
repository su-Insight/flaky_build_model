from __future__ import annotations

import math
import logging
from typing import Dict, List

import pandas as pd

from features.structural_preprocessing import has_label_diversity


LOGGER = logging.getLogger(__name__)


def _flatten_index_windows(index_windows: List[List[int]], selected_windows: List[int]) -> List[int]:
    return [row_index for window_index in selected_windows for row_index in index_windows[window_index]]


def split_dataset(df: pd.DataFrame, split_config) -> Dict[str, Dict[str, pd.DataFrame]]:
    res = df.reset_index(drop=True)
    grouped_info = (
        res.groupby("run_id")
        .agg(row_indices=("run_id", lambda run_ids: list(run_ids.index)), min_time=("created_at", "min"))
        .sort_values("min_time")
    )
    commit_windows = grouped_info["row_indices"].tolist()
    window_count = len(commit_windows)
    valid_window_count = math.ceil(window_count * split_config.valid_size)
    test_window_count = math.ceil(window_count * split_config.test_size)
    if window_count < (valid_window_count + test_window_count + 1):
        raise ValueError("Not enough run windows to create train/valid/test splits.")

    train_max_percentages = [
        round(split_config.train_start + index * split_config.step, 2)
        for index in range(int((split_config.train_end - split_config.train_start) / split_config.step) + 1)
    ]

    datasets: Dict[str, Dict[str, pd.DataFrame]] = {}
    for max_percentage in train_max_percentages:
        train_window_end = math.ceil(window_count * max_percentage)
        valid_start = train_window_end
        valid_end = min(valid_start + valid_window_count, window_count)
        test_start = valid_end
        test_end = min(test_start + test_window_count, window_count)
        if test_end <= test_start:
            continue

        train_windows = list(range(0, train_window_end))
        valid_windows = list(range(valid_start, valid_end))
        test_windows = list(range(test_start, test_end))
        candidate_splits = {
            f"{max_percentage}-groupA": {
                "train": res.iloc[_flatten_index_windows(commit_windows, train_windows)].reset_index(drop=True),
                "valid": res.iloc[_flatten_index_windows(commit_windows, valid_windows)].reset_index(drop=True),
                "test": res.iloc[_flatten_index_windows(commit_windows, test_windows)].reset_index(drop=True),
            },
            f"{max_percentage}-groupB": {
                "train": res.iloc[_flatten_index_windows(commit_windows, train_windows)].reset_index(drop=True),
                "valid": res.iloc[_flatten_index_windows(commit_windows, test_windows)].reset_index(drop=True),
                "test": res.iloc[_flatten_index_windows(commit_windows, valid_windows)].reset_index(drop=True),
            },
        }
        for group_name, sets in candidate_splits.items():
            if all(has_label_diversity(sets[split_name], "flaky") for split_name in ("train", "valid", "test")):
                datasets[group_name] = sets
    return datasets


def split_single_dataset(df: pd.DataFrame, train_size: float, test_size: float) -> Dict[str, pd.DataFrame]:
    res = df.reset_index(drop=True)
    grouped_info = (
        res.groupby("run_id")
        .agg(row_indices=("run_id", lambda run_ids: list(run_ids.index)), min_time=("created_at", "min"))
        .sort_values("min_time")
    )
    commit_windows = grouped_info["row_indices"].tolist()
    window_count = len(commit_windows)
    train_window_count = round(window_count * train_size)
    test_window_count = window_count - train_window_count
    LOGGER.info(
        "Single split windows: total=%s | train_size=%.4f -> train_windows=%s | test_size=%.4f -> test_windows=%s | allocated_total=%s",
        window_count,
        train_size,
        train_window_count,
        test_size,
        test_window_count,
        train_window_count + test_window_count,
    )
    if train_window_count <= 0 or test_window_count <= 0:
        raise ValueError("train_size and test_size must each produce at least one run window.")

    train_windows = list(range(0, train_window_count))
    test_windows = list(range(train_window_count, train_window_count + test_window_count))
    LOGGER.info(
        "Single split rows: train_windows=%s | test_windows=%s",
        len(train_windows),
        len(test_windows),
    )
    datasets = {
        "train": res.iloc[_flatten_index_windows(commit_windows, train_windows)].reset_index(drop=True),
        "test": res.iloc[_flatten_index_windows(commit_windows, test_windows)].reset_index(drop=True),
    }
    if datasets["train"].empty or datasets["test"].empty:
        raise ValueError("Single split produced an empty train or test set.")
    if not has_label_diversity(datasets["train"], "flaky"):
        raise ValueError("Single split training set does not contain both flaky classes.")
    return datasets
