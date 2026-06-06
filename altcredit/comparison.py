from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from altcredit.backtest import cibil_like_decision, cibil_like_risk, score_from_pd
from altcredit.config import MODEL_LABELS
from altcredit.fairness import selection_rate_report
from altcredit.features import get_model_spec
from altcredit.json_io import load_json_file
from altcredit.scoring import score_dataframe


COMPARISON_METRIC_LABELS = {
    "records": "Records evaluated",
    "portfolio_default_rate": "Observed portfolio default rate",
    "approval_rate": "Approval rate",
    "decline_rate": "Decline rate",
    "approved_default_rate": "Default rate among approved",
    "declined_default_rate": "Default rate among declined",
    "good_rejection_rate": "Good borrowers rejected",
    "bad_approval_rate": "Bad borrowers approved",
    "roc_auc": "ROC AUC",
    "gini": "Gini",
    "average_precision": "Average precision",
    "brier_loss": "Brier loss",
    "default_precision": "Default precision",
    "default_recall": "Default recall",
    "accuracy": "Classification accuracy",
    "average_score": "Average score",
    "median_score": "Median score",
    "score_threshold": "Score threshold",
    "risk_cutoff": "Risk cutoff",
    "average_limit_approved": "Average approved limit",
    "total_exposure_approved": "Total approved exposure",
    "expected_loss_proxy": "Expected loss proxy",
}


def evaluate_uploaded_testing_data(
    bundle: dict[str, Any],
    records: list[dict[str, Any]],
    reports_dir: str | Path = "artifacts/reports",
    persist: bool = True,
) -> dict[str, Any]:
    spec = bundle["spec"]
    df = pd.DataFrame(records)
    _validate_testing_dataframe(df, spec)

    scored = score_dataframe(bundle, df)
    cibil_risk = cibil_like_risk(scored, spec.model_type)
    cibil_cutoff = float(bundle.get("cibil_like_risk_cutoff", np.quantile(cibil_risk, 0.5)))
    cibil_approved = cibil_like_decision(scored, spec.model_type, cutoff=cibil_cutoff)
    cibil_score = score_from_pd(cibil_risk)

    scored["cibil_like_risk"] = cibil_risk
    scored["cibil_like_score"] = cibil_score
    scored["cibil_like_approved"] = cibil_approved

    metrics = comparison_metrics(scored, spec.model_type, cibil_cutoff)
    fairness = {
        "alt_credit": selection_rate_report(scored, "approved", spec.audit_columns, spec.target_column),
        "cibil_like": selection_rate_report(scored, "cibil_like_approved", spec.audit_columns, spec.target_column),
    }

    output = {
        "model_type": spec.model_type,
        "model_label": MODEL_LABELS[spec.model_type],
        "records": int(len(scored)),
        "metrics": metrics,
        "comparison_table": comparison_table(metrics),
        "fairness": fairness,
    }

    if persist:
        report_path = Path(reports_dir)
        report_path.mkdir(parents=True, exist_ok=True)
        scored.to_csv(report_path / f"{spec.model_type}_uploaded_scored_testing_data.csv", index=False)
        with (report_path / f"{spec.model_type}_uploaded_comparison.json").open("w", encoding="utf-8") as handle:
            json.dump(_json_ready(output), handle, indent=2)

    return output


def comparison_metrics(scored: pd.DataFrame, model_type: str, cibil_cutoff: float) -> dict[str, dict[str, float | int | None]]:
    y_true = scored["default_90d"].astype(int).to_numpy()
    alt_pd = scored["predicted_pd_90d"].astype(float).to_numpy()
    cibil_pd = scored["cibil_like_risk"].astype(float).to_numpy()
    alt_approved = scored["approved"].astype(bool).to_numpy()
    cibil_approved = scored["cibil_like_approved"].astype(bool).to_numpy()

    alt_metrics = _single_model_metrics(
        y_true=y_true,
        pd_values=alt_pd,
        score_values=scored["alt_score"].astype(float).to_numpy(),
        approved=alt_approved,
        score_threshold=float(scored.attrs.get("score_threshold", np.nan)),
        risk_cutoff=float(scored.attrs.get("pd_cutoff", np.nan)),
        exposure=scored["dynamic_exposure_limit_inr"].astype(float).to_numpy(),
    )

    cibil_exposure = _cibil_static_exposure(scored, model_type, cibil_approved)
    cibil_metrics = _single_model_metrics(
        y_true=y_true,
        pd_values=cibil_pd,
        score_values=scored["cibil_like_score"].astype(float).to_numpy(),
        approved=cibil_approved,
        score_threshold=float(score_from_pd(np.array([cibil_cutoff]))[0]),
        risk_cutoff=cibil_cutoff,
        exposure=cibil_exposure,
    )

    lift = {
        "approval_rate_lift": _delta(alt_metrics["approval_rate"], cibil_metrics["approval_rate"]),
        "approved_default_rate_delta": _delta(
            alt_metrics["approved_default_rate"],
            cibil_metrics["approved_default_rate"],
        ),
        "gini_lift": _delta(alt_metrics["gini"], cibil_metrics["gini"]),
        "expected_loss_delta": _delta(
            alt_metrics["expected_loss_proxy"],
            cibil_metrics["expected_loss_proxy"],
        ),
        "total_exposure_lift": _delta(
            alt_metrics["total_exposure_approved"],
            cibil_metrics["total_exposure_approved"],
        ),
    }
    return {"alt_credit": alt_metrics, "cibil_like": cibil_metrics, "delta": lift}


def comparison_table(metrics: dict[str, dict[str, float | int | None]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    alt = metrics["alt_credit"]
    cibil = metrics["cibil_like"]
    delta = metrics["delta"]
    for key, label in COMPARISON_METRIC_LABELS.items():
        rows.append(
            {
                "parameter": label,
                "metric_key": key,
                "alt_credit": alt.get(key),
                "cibil_like": cibil.get(key),
                "delta": _delta(alt.get(key), cibil.get(key)),
            }
        )
    rows.extend(
        [
            {"parameter": "Approval rate lift", "metric_key": "approval_rate_lift", "alt_credit": None, "cibil_like": None, "delta": delta["approval_rate_lift"]},
            {"parameter": "Approved default rate delta", "metric_key": "approved_default_rate_delta", "alt_credit": None, "cibil_like": None, "delta": delta["approved_default_rate_delta"]},
            {"parameter": "Gini lift", "metric_key": "gini_lift", "alt_credit": None, "cibil_like": None, "delta": delta["gini_lift"]},
            {"parameter": "Expected loss delta", "metric_key": "expected_loss_delta", "alt_credit": None, "cibil_like": None, "delta": delta["expected_loss_delta"]},
            {"parameter": "Total exposure lift", "metric_key": "total_exposure_lift", "alt_credit": None, "cibil_like": None, "delta": delta["total_exposure_lift"]},
        ]
    )
    return rows


def load_report_summary(reports_dir: str | Path = "artifacts/reports") -> pd.DataFrame:
    path = Path(reports_dir) / "summary.json"
    if not path.exists():
        return pd.DataFrame()
    data = load_json_file(path)
    rows = []
    for model_type, metrics in data.items():
        rows.append({"model_type": model_type, "model_label": MODEL_LABELS.get(model_type, model_type), **metrics})
    return pd.DataFrame(rows)


def _single_model_metrics(
    y_true: np.ndarray,
    pd_values: np.ndarray,
    score_values: np.ndarray,
    approved: np.ndarray,
    score_threshold: float | None,
    risk_cutoff: float | None,
    exposure: np.ndarray,
) -> dict[str, float | int | None]:
    predicted_default = ~approved
    both_classes = len(np.unique(y_true)) == 2
    approved_default_rate = _rate(y_true[approved]) if approved.any() else 0.0
    declined_default_rate = _rate(y_true[~approved]) if (~approved).any() else 0.0
    nondefaults = y_true == 0
    defaults = y_true == 1

    return {
        "records": int(len(y_true)),
        "portfolio_default_rate": _rate(y_true),
        "approval_rate": float(approved.mean()) if len(approved) else 0.0,
        "decline_rate": float((~approved).mean()) if len(approved) else 0.0,
        "approved_default_rate": approved_default_rate,
        "declined_default_rate": declined_default_rate,
        "good_rejection_rate": float((~approved & nondefaults).sum() / max(nondefaults.sum(), 1)),
        "bad_approval_rate": float((approved & defaults).sum() / max(defaults.sum(), 1)),
        "roc_auc": _metric_or_none(roc_auc_score, y_true, pd_values, both_classes),
        "gini": _gini(y_true, pd_values, both_classes),
        "average_precision": _metric_or_none(average_precision_score, y_true, pd_values, both_classes),
        "brier_loss": float(brier_score_loss(y_true, pd_values)),
        "default_precision": float(precision_score(y_true, predicted_default.astype(int), zero_division=0)),
        "default_recall": float(recall_score(y_true, predicted_default.astype(int), zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predicted_default.astype(int))),
        "average_score": float(np.mean(score_values)),
        "median_score": float(np.median(score_values)),
        "score_threshold": score_threshold,
        "risk_cutoff": risk_cutoff,
        "average_limit_approved": float(np.mean(exposure[approved])) if approved.any() else 0.0,
        "total_exposure_approved": float(np.sum(exposure[approved])),
        "expected_loss_proxy": float(np.sum(pd_values * exposure * 0.65)),
    }


def _cibil_static_exposure(scored: pd.DataFrame, model_type: str, approved: np.ndarray) -> np.ndarray:
    spec = get_model_spec(model_type)
    score_factor = np.clip((scored["cibil_like_score"].astype(float).to_numpy() - 300.0) / 600.0, 0.0, 1.0)
    return np.where(approved, spec.base_exposure_limit_inr * score_factor * 0.75, 0.0)


def _validate_testing_dataframe(df: pd.DataFrame, spec: Any) -> None:
    missing = [column for column in spec.feature_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Uploaded testing data is missing feature columns: {', '.join(missing)}")
    if spec.target_column not in df.columns:
        raise ValueError(f"Uploaded testing data must include `{spec.target_column}` for evaluation.")


def _rate(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _metric_or_none(func: Any, y_true: np.ndarray, values: np.ndarray, both_classes: bool) -> float | None:
    if not both_classes:
        return None
    return float(func(y_true, values))


def _gini(y_true: np.ndarray, values: np.ndarray, both_classes: bool) -> float | None:
    auc = _metric_or_none(roc_auc_score, y_true, values, both_classes)
    return None if auc is None else float(2.0 * auc - 1.0)


def _delta(a: Any, b: Any) -> float | None:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def _json_ready(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value
