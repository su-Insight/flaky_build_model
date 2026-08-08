from __future__ import annotations

import contextlib
import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_CACHED_SENTENCE_MODEL = None
_THIRD_PARTY_LOGGING_CONFIGURED = False


@dataclass
class LogBranchResult:
    validation_proba: np.ndarray
    test_proba: np.ndarray
    y_valid: pd.Series
    y_test: pd.Series


@dataclass
class _LogRecord:
    text: str
    metadata: Dict[str, Any]
    token_set: set[str]


class _SentenceTransformerEncoder:
    def __init__(self) -> None:
        self._configure_third_party_logging()
        from sentence_transformers import SentenceTransformer

        model_name_or_path = self._resolve_model_source()
        global _CACHED_SENTENCE_MODEL
        if _CACHED_SENTENCE_MODEL is None:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _CACHED_SENTENCE_MODEL = SentenceTransformer(model_name_or_path)
        self._model = _CACHED_SENTENCE_MODEL

    @staticmethod
    def _configure_third_party_logging() -> None:
        global _THIRD_PARTY_LOGGING_CONFIGURED
        if _THIRD_PARTY_LOGGING_CONFIGURED:
            return

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARNING)
        logging.getLogger("faiss").setLevel(logging.WARNING)
        logging.getLogger("faiss.loader").setLevel(logging.WARNING)
        try:  # pragma: no cover
            from transformers.utils import logging as transformers_logging

            transformers_logging.set_verbosity_error()
            if hasattr(transformers_logging, "disable_progress_bar"):
                transformers_logging.disable_progress_bar()
        except Exception:
            pass

        _THIRD_PARTY_LOGGING_CONFIGURED = True

    @staticmethod
    def _resolve_model_source() -> str:
        candidate_paths = [
            Path(__file__).resolve().parents[2] / "log_similarity" / "local_models",
            Path(__file__).resolve().parents[1] / "local_models",
        ]
        for candidate in candidate_paths:
            if candidate.is_dir():
                return str(candidate)
        return DEFAULT_SENTENCE_MODEL

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, convert_to_numpy=True, show_progress_bar=False))


class LogVectorDatabase:
    def __init__(self) -> None:
        self._records: List[_LogRecord] = []
        self._encoder: Optional[_SentenceTransformerEncoder] = None
        self._embedding_index = None
        self._embedding_rows: List[np.ndarray] = []
        self._try_enable_dense_backend()

    def _try_enable_dense_backend(self) -> None:
        try:
            self._encoder = _SentenceTransformerEncoder()
        except Exception as exc:  # pragma: no cover
            LOGGER.warning(
                "Sentence-transformer backend unavailable (%s). Falling back to lexical similarity.",
                exc,
            )
            self._encoder = None
            return

        try:  # pragma: no cover
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                import faiss

            sample_vector = self._normalize_embeddings(self._encoder.encode(["bootstrap"]))[0]
            self._embedding_index = faiss.IndexFlatIP(sample_vector.shape[0])
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("FAISS unavailable (%s). Falling back to in-memory dense search.", exc)
            self._embedding_index = None

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token for token in re.findall(r"[A-Za-z0-9_]+", text.lower()) if token}

    @staticmethod
    def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return embeddings / norms

    def add_log_entry(self, log_text: str, metadata: Dict[str, Any]) -> None:
        normalized_text = str(log_text).strip()
        if not normalized_text:
            return

        record = _LogRecord(
            text=normalized_text,
            metadata=metadata,
            token_set=self._tokenize(normalized_text),
        )
        self._records.append(record)

        if self._encoder is None:
            return

        embedding = self._normalize_embeddings(self._encoder.encode([normalized_text]))[0].astype("float32")
        if self._embedding_index is not None:
            self._embedding_index.add(np.array([embedding], dtype="float32"))
        else:
            self._embedding_rows.append(embedding)

    def find_similar_logs(self, query_log: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self._records or not str(query_log).strip():
            return []

        top_k = min(k, len(self._records))
        if self._encoder is not None:
            return self._dense_search(str(query_log), top_k)
        return self._lexical_search(str(query_log), top_k)

    def _dense_search(self, query_log: str, k: int) -> List[Dict[str, Any]]:
        query_vector = self._normalize_embeddings(self._encoder.encode([query_log]))[0].astype("float32")
        if self._embedding_index is not None:
            similarities, indices = self._embedding_index.search(np.array([query_vector]), k)
            ranked = zip(indices[0].tolist(), similarities[0].tolist())
        else:
            matrix = np.vstack(self._embedding_rows)
            similarities = matrix @ query_vector
            indices = np.argsort(similarities)[::-1][:k]
            ranked = ((int(index), float(similarities[index])) for index in indices)

        return [
            {
                "similarity": float(similarity),
                "metadata": self._records[index].metadata,
                "original_log": self._records[index].text,
            }
            for index, similarity in ranked
        ]

    def _lexical_search(self, query_log: str, k: int) -> List[Dict[str, Any]]:
        query_tokens = self._tokenize(query_log)
        scored = []
        for record in self._records:
            union = query_tokens | record.token_set
            similarity = 0.0 if not union else len(query_tokens & record.token_set) / len(union)
            scored.append((similarity, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "similarity": float(similarity),
                "metadata": record.metadata,
                "original_log": record.text,
            }
            for similarity, record in scored[:k]
        ]


def _build_vector_db(train_df: pd.DataFrame) -> LogVectorDatabase:
    vector_db = LogVectorDatabase()
    for _, row in train_df[train_df["error_msg"].astype(bool)].iterrows():
        vector_db.add_log_entry(
            row["error_msg"],
            {"job_id": row["job_id"], "flaky": row["flaky"], "error_type": row["sub_reason"]},
        )
    return vector_db


def _similarity_features(vector_db: LogVectorDatabase, df: pd.DataFrame, k: int, train: bool) -> pd.DataFrame:
    features: Dict[str, List[float]] = {"job_id": [], "flaky": []}
    for index in range(k):
        features[f"case{index}_similarity"] = []
        features[f"case{index}_flaky"] = []

    for _, row in df.iterrows():
        features["job_id"].append(row["job_id"])
        features["flaky"].append(row["flaky"])
        if pd.isna(row["error_msg"]) or not str(row["error_msg"]).strip():
            similar_logs = []
        else:
            similar_logs = vector_db.find_similar_logs(row["error_msg"], k=k + 1 if train else k)
        if train and similar_logs:
            similar_logs = similar_logs[1:]

        for index in range(k):
            if index < len(similar_logs):
                features[f"case{index}_similarity"].append(similar_logs[index]["similarity"])
                features[f"case{index}_flaky"].append(similar_logs[index]["metadata"]["flaky"])
            else:
                features[f"case{index}_similarity"].append(0.0)
                features[f"case{index}_flaky"].append(0)
    return pd.DataFrame(features)


def run_log_branch(train_df: pd.DataFrame, valid_df: pd.DataFrame, test_df: pd.DataFrame, log_k: int) -> LogBranchResult:
    if log_k <= 0:
        raise ValueError("log_k must be a positive integer.")

    from sklearn.linear_model import LogisticRegression

    vector_db = _build_vector_db(train_df)
    train_features = _similarity_features(vector_db, train_df, log_k, train=True)
    valid_features = _similarity_features(vector_db, valid_df, log_k, train=False)
    test_features = _similarity_features(vector_db, test_df, log_k, train=False)

    feature_columns = []
    for index in range(log_k):
        feature_columns.extend([f"case{index}_similarity", f"case{index}_flaky"])

    if train_features["flaky"].nunique() < 2:
        raise ValueError("The log branch training set does not contain both flaky classes.")

    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    model.fit(train_features[feature_columns], train_features["flaky"])
    validation_proba = model.predict_proba(valid_features[feature_columns])[:, 1]
    test_proba = model.predict_proba(test_features[feature_columns])[:, 1]
    return LogBranchResult(
        validation_proba=validation_proba,
        test_proba=test_proba,
        y_valid=valid_features["flaky"].reset_index(drop=True),
        y_test=test_features["flaky"].reset_index(drop=True),
    )
