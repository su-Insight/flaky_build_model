from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler


BINARY_FEATURES = [
    "is_runner_changed",
    "is_action_changed",
    "use_unverified_action",
    "is_artifact_share",
    "is_action_cache",
    "commit_message_issue_ref",
    "git_same_committer",
    "gh_hotspot_files_touched",
    "src_ast_diff",
    "test_ast_diff",
    "dockerfile_changed",
    "docker_compose_changed",
    "tests_ran",
    "is_core_member",
    "committer_cross_project_exp",
    "gh_committer_first_build",
    "prev_build_same_files_touched",
    "is_master",
]

NUMERIC_FEATURES = [
    "duration",
    "log_warn_nums",
    "git_num_committers",
    "repo_team_size_last_3_month",
    "gh_commits_on_files_touched",
    "gh_num_pr_comments",
    "gh_pr_description_complexity",
    "git_commits",
    "gh_files_modified",
    "gh_lines_added",
    "gh_lines_deleted",
    "gh_other_files",
    "gh_src_files",
    "gh_src_churn",
    "gh_files_type_modified",
    "gh_test_churn",
    "gh_files_entropy",
    "gh_cross_module_changes",
    "ast_class_added",
    "ast_class_deleted",
    "ast_class_modified",
    "ast_class_changed",
    "ast_met_added",
    "ast_met_deleted",
    "ast_met_changed",
    "ast_met_sig_modified",
    "ast_field_added",
    "ast_field_deleted",
    "ast_field_changed",
    "ast_import_added",
    "ast_import_deleted",
    "ast_import_changed",
    "gh_dependencies_churn",
    "dependencies_count",
    "gh_files_added",
    "gh_files_deleted",
    "gh_tests_added",
    "gh_tests_deleted",
    "gh_doc_files",
    "gh_config_files",
    "ast_met_body_modified",
    "workflow_size",
    "sloc_initial",
    "test_lines_initial",
    "test_lines_per_1000_sloc",
    "tests_passed",
    "tests_failed",
    "tests_skipped",
    "tests_total",
    "repo_fail_rate_history",
    "gh_committer_bayesian_trust_score_history",
    "repo_fail_rate_recent",
    "gh_committer_bayesian_trust_score_recent",
    "git_committer_repo_exp",
    "concurrent_jobs",
    "repo_ci_config_churn_nums",
]

NOMINAL_FEATURES = [
    "sub_reason",
    "gh_first_error_step",
    "operation_system",
    "runner_type",
    "trigger_event",
]
SET_FEATURE = "git_commit_attention"

DROP_COLUMNS = [
    "index",
    "id",
    "job_id",
    "workflow_name",
    "test_frameworks",
    "build_language",
    "head_branch",
    "gh_previous_build_result",
    "run_id",
    "conclusion",
    "created_at",
    "started_at",
    "completed_at",
    "name",
    "repository_id",
    "main_reason",
    "error_msg",
    "rf.id",
    "rf.run_id",
    "rf.repository_id",
    "rf.name",
    "rf.head_branch",
    "head_sha",
    "path",
    "rf.conclusion",
    "workflow_id",
    "actor_name",
    "pr_number",
    "base_sha",
    "sub_reason",
    "success",
]

REQUIRED_BASE_COLUMNS = [
    "run_id",
    "job_id",
    "conclusion",
    "created_at",
    "gh_previous_build_result",
    "name",
]

REQUIRED_STRUCTURED_COLUMNS = (
    BINARY_FEATURES
    + NUMERIC_FEATURES
    + NOMINAL_FEATURES
    + [SET_FEATURE, "status", "flaky", "run_id", "job_id", "created_at", "name"]
)


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], context: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def prepare_base_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(df, REQUIRED_BASE_COLUMNS, "Base dataframe")
    prepared = df.copy()
    run_summary = prepared[["run_id", "conclusion"]].copy()
    run_summary["fail_rate"] = run_summary["conclusion"].ne("success").astype(int)
    run_failure_rate = run_summary.groupby("run_id")["fail_rate"].mean().reset_index()
    run_failure_rate["flaky"] = run_failure_rate["fail_rate"].apply(lambda value: 1 if 0 < value < 1 else 0)
    prepared = prepared.merge(run_failure_rate[["run_id", "flaky"]], on="run_id", how="left")
    prepared["status"] = prepared["conclusion"].ne("success").astype(int)
    prepared["previous_status"] = prepared["gh_previous_build_result"].ne("success").astype(int)
    prepared["created_at"] = pd.to_datetime(prepared["created_at"], errors="coerce")
    if prepared["created_at"].isna().any():
        raise ValueError("created_at contains invalid timestamps after parsing.")
    prepared["created_at"] = prepared["created_at"].astype("int64", copy=False) // 10**9
    return prepared


def keep_failures_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["conclusion"] == "failure"].reset_index(drop=True)


def has_label_diversity(df: pd.DataFrame, target_col: str) -> bool:
    return not df.empty and target_col in df.columns and df[target_col].nunique() > 1


def parse_set_string(set_str: Any) -> set:
    if pd.isna(set_str):
        return set()
    if not isinstance(set_str, str):
        if isinstance(set_str, (set, list, tuple)):
            return set(set_str)
        return {str(set_str)}
    try:
        return set(ast.literal_eval(set_str))
    except (ValueError, SyntaxError):
        return set()


def compute_since_flaky(df: pd.DataFrame) -> List[int]:
    grouped = (
        df.groupby("run_id")
        .agg(
            has_flaky=("flaky", lambda values: bool((values == 1).any())),
            created_at=("created_at", "min"),
            row_indices=("run_id", lambda run_ids: list(run_ids.index)),
        )
        .sort_values("created_at")
    )
    distances: List[int] = []
    distance = 0
    for has_flaky in grouped["has_flaky"].tolist():
        distances.append(distance)
        distance += 1
        if has_flaky:
            distance = 0
    grouped["since_flaky"] = distances
    flat = [0] * df.shape[0]
    for row_indices, value in zip(grouped["row_indices"], grouped["since_flaky"]):
        for row_index in row_indices:
            flat[row_index] = value
    return flat


def enhance_with_rerun_features(df: pd.DataFrame) -> pd.DataFrame:
    ensure_columns(df, ["run_id", "name", "created_at", "status", "flaky"], "Rerun feature input")
    enhanced = df.copy()
    info = [{} for _ in range(enhanced.shape[0])]
    current_commit_job = ""
    current_stats: Dict[str, int] = {}
    since_flaky = compute_since_flaky(enhanced)

    for row_index, row in enhanced.sort_values(by=["run_id", "name", "created_at"]).iterrows():
        commit_job = f"{row['run_id']}::{row['name']}"
        if current_commit_job != commit_job:
            current_commit_job = commit_job
            current_stats = {"rerun": 0, "fail": 0, "success": 0, "commit_since_flaky": 0}
        current_stats["commit_since_flaky"] = since_flaky[row_index]
        current_stats["rerun"] += 1
        current_stats["success"] += 1 - row["status"]
        current_stats["fail"] += row["status"]
        info[row_index] = current_stats.copy()

    enhanced["info"] = info
    info_df = pd.json_normalize(enhanced["info"])
    enhanced = enhanced.drop(columns=["info"]).join(info_df)
    enhanced["rerun*fail"] = enhanced["rerun"] * enhanced["fail"]
    return enhanced


def oversample_by_status_and_flaky(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ensure_columns(df, ["status", "flaky"], "Oversampling input")
    groups = [
        df[(df["status"] == status) & (df["flaky"] == flaky)].index.tolist()
        for status in sorted(df["status"].unique())
        for flaky in sorted(df["flaky"].unique())
    ]
    groups = [group for group in groups if group]
    if len(groups) <= 1:
        return df.copy().reset_index(drop=True)

    max_length = max(len(group) for group in groups)
    sampled_indices: List[int] = []
    for group in groups:
        expanded_group = group * math.ceil(max_length / len(group))
        sampled_indices.extend(expanded_group[:max_length])
    sampled_indices.sort()
    return df.iloc[sampled_indices].reset_index(drop=True)


@dataclass
class SplitAwarePreprocessor:
    scaler: StandardScaler = field(default_factory=StandardScaler)
    nominal_encoder: OneHotEncoder = field(
        default_factory=lambda: OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    )
    set_binarizer: MultiLabelBinarizer = field(default_factory=MultiLabelBinarizer)
    nominal_feature_names: List[str] = field(default_factory=list)
    set_feature_names: List[str] = field(default_factory=list)

    @staticmethod
    def normalize_binary_column(series: pd.Series) -> pd.Series:
        mapped = series.astype(str).replace({"nan": "unknown", "0.0": "No", "1.0": "Yes"})
        return mapped.where(mapped.isin({"Yes", "No", "unknown"}), "unknown")

    def fit(self, train_df: pd.DataFrame) -> "SplitAwarePreprocessor":
        ensure_columns(train_df, REQUIRED_STRUCTURED_COLUMNS, "Structured training dataframe")
        self.scaler.fit(train_df[NUMERIC_FEATURES])
        train_nominal = train_df[NOMINAL_FEATURES].fillna("__nan__").astype(str)
        self.nominal_encoder.fit(train_nominal)
        self.nominal_feature_names = self.nominal_encoder.get_feature_names_out(NOMINAL_FEATURES).tolist()
        parsed_sets = train_df[SET_FEATURE].apply(parse_set_string)
        self.set_binarizer.fit(parsed_sets)
        self.set_feature_names = [f"{SET_FEATURE}_{label}" for label in self.set_binarizer.classes_]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        ensure_columns(df, REQUIRED_STRUCTURED_COLUMNS, "Structured dataframe")
        transformed = df.copy().reset_index(drop=True)
        untouched = transformed.drop(
            columns=BINARY_FEATURES + NUMERIC_FEATURES + NOMINAL_FEATURES + [SET_FEATURE],
            errors="ignore",
        )

        numeric_df = pd.DataFrame(
            self.scaler.transform(transformed[NUMERIC_FEATURES]),
            columns=NUMERIC_FEATURES,
            index=transformed.index,
        )

        binary_frames = []
        for column in BINARY_FEATURES:
            normalized = self.normalize_binary_column(transformed[column])
            binary_df = pd.get_dummies(normalized, prefix=column)
            expected_columns = [f"{column}_No", f"{column}_Yes", f"{column}_unknown"]
            binary_df = binary_df.reindex(columns=expected_columns, fill_value=0)
            binary_frames.append(binary_df)
        binary_df = pd.concat(binary_frames, axis=1)

        nominal_values = transformed[NOMINAL_FEATURES].fillna("__nan__").astype(str)
        nominal_df = pd.DataFrame(
            self.nominal_encoder.transform(nominal_values),
            columns=self.nominal_feature_names,
            index=transformed.index,
        )

        parsed_sets = transformed[SET_FEATURE].apply(parse_set_string)
        set_df = pd.DataFrame(
            self.set_binarizer.transform(parsed_sets),
            columns=self.set_feature_names,
            index=transformed.index,
        )

        return pd.concat([untouched, numeric_df, binary_df, nominal_df, set_df], axis=1)
