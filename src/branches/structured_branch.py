from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover
    xgb = None


@dataclass
class StructuredBranchResult:
    model_name: str
    params: Dict[str, Any]
    y_valid: pd.Series
    y_test: pd.Series
    validation_proba: np.ndarray
    test_proba: np.ndarray


def compute_binary_classification_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if y_true.nunique() > 1 else float("nan")
    return {
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "auc": float(auc),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _split_xy(sets: Dict[str, pd.DataFrame]):
    y_train = sets["train"]["flaky"]
    X_train = sets["train"].drop(columns=["flaky"])
    y_valid = sets["valid"]["flaky"]
    X_valid = sets["valid"].drop(columns=["flaky"])
    y_test = sets["test"]["flaky"]
    X_test = sets["test"].drop(columns=["flaky"])
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def fit_random_forest(sets: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> StructuredBranchResult:
    X_train, y_train, X_valid, y_valid, X_test, y_test = _split_xy(sets)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_valid = imputer.transform(X_valid)
    X_test = imputer.transform(X_test)
    model = RandomForestClassifier(random_state=42, **params)
    model.fit(X_train, y_train)
    return StructuredBranchResult(
        "random_forest",
        params,
        y_valid.reset_index(drop=True),
        y_test.reset_index(drop=True),
        model.predict_proba(X_valid)[:, 1],
        model.predict_proba(X_test)[:, 1],
    )


def fit_svm(sets: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> StructuredBranchResult:
    X_train, y_train, X_valid, y_valid, X_test, y_test = _split_xy(sets)
    model = Pipeline(
        [("imputer", SimpleImputer()), ("scaler", StandardScaler()), ("svc", SVC(probability=True, random_state=42))]
    )
    model.set_params(**params)
    model.fit(X_train, y_train)
    return StructuredBranchResult(
        "svm",
        params,
        y_valid.reset_index(drop=True),
        y_test.reset_index(drop=True),
        model.predict_proba(X_valid)[:, 1],
        model.predict_proba(X_test)[:, 1],
    )


def fit_mlp(sets: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> StructuredBranchResult:
    X_train, y_train, X_valid, y_valid, X_test, y_test = _split_xy(sets)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_valid = imputer.transform(X_valid)
    X_test = imputer.transform(X_test)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    model = MLPClassifier(max_iter=1000, random_state=42, **params)
    model.fit(X_train, y_train)
    return StructuredBranchResult(
        "mlp",
        params,
        y_valid.reset_index(drop=True),
        y_test.reset_index(drop=True),
        model.predict_proba(X_valid)[:, 1],
        model.predict_proba(X_test)[:, 1],
    )


def fit_xgboost(sets: Dict[str, pd.DataFrame], params: Dict[str, Any]) -> StructuredBranchResult:
    if xgb is None:
        raise RuntimeError("xgboost is not installed, but the xgboost model was requested.")
    X_train, y_train, X_valid, y_valid, X_test, y_test = _split_xy(sets)
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_valid = imputer.transform(X_valid)
    X_test = imputer.transform(X_test)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test, label=y_test)
    fixed_params = {"objective": "binary:logistic", "eval_metric": "auc", "verbosity": 0, "seed": 42}
    model = xgb.train(
        {**fixed_params, **params},
        dtrain,
        num_boost_round=200,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )
    return StructuredBranchResult(
        "xgboost",
        params,
        y_valid.reset_index(drop=True),
        y_test.reset_index(drop=True),
        model.predict(dvalid),
        model.predict(dtest),
    )


MODEL_RUNNERS: Dict[str, Callable[[Dict[str, pd.DataFrame], Dict[str, Any]], StructuredBranchResult]] = {
    "random_forest": fit_random_forest,
    "svm": fit_svm,
    "mlp": fit_mlp,
    "xgboost": fit_xgboost,
}
