from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import numpy as np


def calculate_routine_entropy(location_cells: Iterable[str]) -> float:
    cells = [cell for cell in location_cells if cell]
    if not cells:
        return 1.0
    counts = Counter(cells)
    total = sum(counts.values())
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    max_entropy = math.log2(max(len(counts), 2))
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def calculate_cash_velocity_stability(values: Iterable[float]) -> float:
    amounts = np.array([float(value) for value in values], dtype=float)
    if len(amounts) < 2:
        return 1.0
    mean = float(np.mean(amounts))
    if mean <= 0:
        return 1.0
    return float(np.clip(np.std(amounts) / mean, 0.0, 1.0))


def calculate_spend_rationality(
    essential_spend: float,
    discretionary_spend: float,
    speculative_spend: float,
    total_inflow: float,
) -> dict[str, float]:
    inflow = max(float(total_inflow), 1.0)
    essential_ratio = float(np.clip(essential_spend / inflow, 0.0, 1.0))
    discretionary_ratio = float(np.clip((discretionary_spend + speculative_spend) / inflow, 0.0, 1.0))
    rationality = float(np.clip(essential_ratio + 0.5 * (1.0 - discretionary_ratio), 0.0, 1.0))
    return {
        "essential_spend_ratio": essential_ratio,
        "discretionary_spend_ratio": discretionary_ratio,
        "spend_rationality_score": rationality,
    }


def calculate_msme_financial_ratios(
    revenue: float,
    gross_profit: float,
    debt_service: float,
    current_assets: float,
    current_liabilities: float,
) -> dict[str, float]:
    safe_revenue = max(float(revenue), 1.0)
    safe_debt_service = max(float(debt_service), 1.0)
    gross_margin = float(np.clip(float(gross_profit) / safe_revenue, -1.0, 1.0))
    dscr = float(np.clip(float(gross_profit) / safe_debt_service, 0.0, 3.0) / 3.0)
    working_capital_gap = float(
        np.clip((float(current_liabilities) - float(current_assets)) / safe_revenue, 0.0, 1.0)
    )
    return {
        "gross_margin_ratio": gross_margin,
        "debt_service_coverage_proxy": dscr,
        "working_capital_gap_ratio": working_capital_gap,
    }

