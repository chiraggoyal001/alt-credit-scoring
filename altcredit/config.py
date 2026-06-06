from __future__ import annotations

from dataclasses import dataclass


RANDOM_SEED = 42
DEFAULT_ROWS_PER_MODEL = 5000

MODEL_TYPES = (
    "retail_active",
    "retail_cold_start",
    "msme_active",
    "msme_cold_start",
)

MODEL_LABELS = {
    "retail_active": "Model 1: Retail Active",
    "retail_cold_start": "Model 2: Retail Cold-Start",
    "msme_active": "Model 3: MSME Active",
    "msme_cold_start": "Model 4: MSME Cold-Start",
}

BASE_EXPOSURE_LIMITS_INR = {
    "retail_active": 75000,
    "retail_cold_start": 12000,
    "msme_active": 750000,
    "msme_cold_start": 175000,
}

SYNTHETIC_CIBIL_APPROVAL_RATES = {
    "retail_active": 0.46,
    "retail_cold_start": 0.27,
    "msme_active": 0.42,
    "msme_cold_start": 0.25,
}

MAX_ALT_APPROVAL_RATES = {
    "retail_active": 0.82,
    "retail_cold_start": 0.68,
    "msme_active": 0.78,
    "msme_cold_start": 0.62,
}


@dataclass(frozen=True)
class OperatingPoint:
    score_threshold: float
    pd_cutoff: float
    approval_rate: float
    approved_default_rate: float
    cibil_like_approval_rate: float
    cibil_like_default_rate: float
