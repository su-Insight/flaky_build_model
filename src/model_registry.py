from __future__ import annotations

from typing import Any, Dict, Tuple


DEFAULT_SINGLE_PARAMS: Dict[str, Dict[str, Any]] = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 5,
        "min_samples_split": 2,
        "class_weight": "balanced",
    },
    "svm": {
        "imputer__strategy": "median",
        "svc__class_weight": "balanced",
        "svc__C": 10,
        "svc__kernel": "rbf",
        "svc__gamma": "scale",
    },
    "mlp": {
        "activation": "relu",
        "solver": "adam",
        "alpha": 0.0001,
        "batch_size": 32,
        "learning_rate_init": 0.01,
        "early_stopping": True,
        "validation_fraction": 0.1,
    },
    "xgboost": {
        "max_depth": 3,
        "eta": 0.01,
        "subsample": 0.6,
        "colsample_bytree": 0.6,
        "min_child_weight": 1,
        "gamma": 0,
    },
}


SUPPORTED_MODELS: Tuple[str, ...] = tuple(DEFAULT_SINGLE_PARAMS.keys())


TUNE_FEATURE_K_VALUES = [5, 10, 20, 30, 40, 50, 60]
TUNE_LOG_K_VALUES = [5, 10, 15, 20, 25, 30]
TUNE_FUSION_ALPHA_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
