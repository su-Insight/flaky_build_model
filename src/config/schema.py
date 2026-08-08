from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from model_registry import SUPPORTED_MODELS


class RunMode(str, Enum):
    TUNE = "tune"
    TRAIN = "train"
    SINGLE = "single"


@dataclass(frozen=True)
class SplitConfig:
    train_start: float = 0.5
    train_end: float = 0.9
    step: float = 0.1
    valid_size: float = 0.05
    test_size: float = 0.05

    def __post_init__(self) -> None:
        ratio_fields = {
            "train_start": self.train_start,
            "train_end": self.train_end,
            "step": self.step,
            "valid_size": self.valid_size,
            "test_size": self.test_size,
        }
        for field_name, value in ratio_fields.items():
            if value <= 0 or value >= 1:
                raise ValueError(f"{field_name} must be in the open interval (0, 1).")
        if self.train_start >= self.train_end:
            raise ValueError("train_start must be smaller than train_end.")
        if self.train_end + self.valid_size + self.test_size > 1:
            raise ValueError("train_end + valid_size + test_size must not exceed 1.")


@dataclass(frozen=True)
class BaseConfig:
    repository: str
    model: str
    oversample_train: bool = True
    threshold: float = 0.5
    structured_only: bool = False
    split: SplitConfig = field(default_factory=SplitConfig)
    data_dir: str = "data/csv"
    output_path: Optional[str] = None
    log_path: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.repository or not self.repository.strip():
            raise ValueError("repository must be a non-empty string.")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{self.model}'. Expected one of {SUPPORTED_MODELS}.")
        if self.threshold <= 0 or self.threshold >= 1:
            raise ValueError("threshold must be in the open interval (0, 1).")
        if not self.data_dir or not self.data_dir.strip():
            raise ValueError("data_dir must be a non-empty string.")


@dataclass(frozen=True)
class TuneConfig(BaseConfig):
    mode: RunMode = RunMode.TUNE


@dataclass(frozen=True)
class TrainConfig(BaseConfig):
    mode: RunMode = RunMode.TRAIN
    feature_k: int = 10
    log_k: Optional[int] = 10
    fusion_alpha: Optional[float] = 0.5

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.feature_k <= 0:
            raise ValueError("feature_k must be a positive integer.")
        if not self.structured_only:
            if self.log_k is None or self.log_k <= 0:
                raise ValueError("log_k must be a positive integer.")
            if self.fusion_alpha is None or self.fusion_alpha <= 0 or self.fusion_alpha >= 1:
                raise ValueError("fusion_alpha must be in the open interval (0, 1).")


@dataclass(frozen=True)
class SingleRunConfig:
    repository: str
    model: str
    feature_k: int
    log_k: Optional[int]
    fusion_alpha: Optional[float]
    train_size: float
    test_size: float
    oversample_train: bool = True
    threshold: float = 0.5
    structured_only: bool = False
    data_dir: str = "data/csv"
    output_path: Optional[str] = None
    log_path: Optional[str] = None
    mode: RunMode = RunMode.SINGLE

    def __post_init__(self) -> None:
        if not self.repository or not self.repository.strip():
            raise ValueError("repository must be a non-empty string.")
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model '{self.model}'. Expected one of {SUPPORTED_MODELS}.")
        if self.feature_k <= 0:
            raise ValueError("feature_k must be a positive integer.")
        if not self.structured_only:
            if self.log_k is None or self.log_k <= 0:
                raise ValueError("log_k must be a positive integer.")
            if self.fusion_alpha is None or self.fusion_alpha <= 0 or self.fusion_alpha >= 1:
                raise ValueError("fusion_alpha must be in the open interval (0, 1).")
        if self.threshold <= 0 or self.threshold >= 1:
            raise ValueError("threshold must be in the open interval (0, 1).")
        if self.train_size <= 0 or self.train_size >= 1:
            raise ValueError("train_size must be in the open interval (0, 1).")
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be in the open interval (0, 1).")
        if self.train_size + self.test_size > 1:
            raise ValueError("train_size + test_size must not exceed 1.")
        if not self.data_dir or not self.data_dir.strip():
            raise ValueError("data_dir must be a non-empty string.")
