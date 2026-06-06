from __future__ import annotations

import pandas as pd


def selection_rate_report(
    df: pd.DataFrame,
    decision_column: str,
    audit_columns: tuple[str, ...],
    target_column: str = "default_90d",
) -> dict[str, dict[str, object]]:
    report: dict[str, dict[str, object]] = {}
    for column in audit_columns:
        if column not in df.columns:
            continue
        grouped = df.groupby(column, dropna=False)
        selection_rates = grouped[decision_column].mean().sort_index()
        default_rates = grouped[target_column].mean().sort_index()
        counts = grouped.size().sort_index()
        max_rate = float(selection_rates.max()) if len(selection_rates) else 0.0
        min_rate = float(selection_rates.min()) if len(selection_rates) else 0.0
        disparate_impact = 1.0 if max_rate == 0 else min_rate / max_rate

        report[column] = {
            "disparate_impact_ratio": float(disparate_impact),
            "passes_80_percent_rule": bool(disparate_impact >= 0.80),
            "selection_rates": {str(k): float(v) for k, v in selection_rates.items()},
            "observed_default_rates": {str(k): float(v) for k, v in default_rates.items()},
            "counts": {str(k): int(v) for k, v in counts.items()},
        }
    return report


def label_balance_report(
    df: pd.DataFrame,
    audit_columns: tuple[str, ...],
    target_column: str = "default_90d",
) -> dict[str, dict[str, object]]:
    """Checks whether synthetic defaults were generated directly from protected groups."""
    report: dict[str, dict[str, object]] = {}
    for column in audit_columns:
        if column not in df.columns:
            continue
        rates = df.groupby(column, dropna=False)[target_column].mean().sort_index()
        max_rate = float(rates.max()) if len(rates) else 0.0
        min_rate = float(rates.min()) if len(rates) else 0.0
        report[column] = {
            "target_rate_gap": float(max_rate - min_rate),
            "target_rates": {str(k): float(v) for k, v in rates.items()},
        }
    return report

