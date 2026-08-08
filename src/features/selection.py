from __future__ import annotations

from typing import List

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def select_structured_features(df_processed: pd.DataFrame, k_best: int) -> List[str]:
    if k_best <= 0:
        raise ValueError("k_best must be a positive integer.")

    X = df_processed.drop(columns=["flaky", "job_id"], errors="ignore").fillna(0)
    y = df_processed["flaky"]
    if X.empty:
        raise ValueError("Structured feature selection received no candidate columns.")

    scores = mutual_info_classif(X, y, random_state=42)
    feature_scores = pd.Series(scores, index=X.columns)
    return feature_scores.nlargest(min(k_best, len(feature_scores))).index.tolist()
