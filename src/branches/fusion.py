from __future__ import annotations

import numpy as np


def fuse_branch_probabilities(structured_proba, log_proba, fusion_alpha: float):
    structured = np.asarray(structured_proba, dtype=float)
    log_values = np.asarray(log_proba, dtype=float)
    if structured.shape != log_values.shape:
        raise ValueError("Structured and log probabilities must share the same shape for fusion.")
    return fusion_alpha * structured + (1.0 - fusion_alpha) * log_values
