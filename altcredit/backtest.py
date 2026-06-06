from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from altcredit.config import MAX_ALT_APPROVAL_RATES, OperatingPoint, SYNTHETIC_CIBIL_APPROVAL_RATES


def score_from_pd(prob_default: np.ndarray) -> np.ndarray:
    return 300.0 + 600.0 * (1.0 - prob_default)


def pd_from_score(score: float) -> float:
    return 1.0 - ((score - 300.0) / 600.0)


def cibil_like_risk(df: pd.DataFrame, model_type: str) -> np.ndarray:
    """Synthetic legacy-bureau proxy using fewer, lagging variables.

    This is not a real CIBIL model. It is a conservative benchmark to test the
    MVP objective: can richer cash-flow and behavioral data approve more users
    while reducing the observed default rate among approved accounts?
    """
    if model_type == "retail_active":
        noise = _stable_noise(df, model_type)
        income = _rank01(df["monthly_inflow_inr"].to_numpy())
        stability = df["inflow_stability_90d"].to_numpy()
        obligations = df["fixed_obligation_ratio"].to_numpy()
        recovery = df["prior_loan_recovery_ratio"].to_numpy()
        risk = 0.52 + 0.58 * noise - 0.08 * income - 0.08 * stability + 0.08 * obligations - 0.05 * recovery
    elif model_type == "retail_cold_start":
        noise = _stable_noise(df, model_type)
        device = df["device_hardware_score"].to_numpy()
        sms_income = df["sms_income_signal_strength"].to_numpy()
        utility = df["sms_utility_payment_regularity"].to_numpy()
        risk = 0.56 + 0.62 * noise - 0.07 * device - 0.08 * sms_income - 0.06 * utility
    elif model_type == "msme_active":
        noise = _stable_noise(df, model_type)
        revenue = _rank01(df["monthly_revenue_inr"].to_numpy())
        gst = df["gst_filing_consistency"].to_numpy()
        delay = _rank01(df["gstr1_filing_delay_days"].to_numpy())
        margin = df["gross_margin_ratio"].to_numpy()
        risk = 0.55 + 0.58 * noise - 0.07 * revenue - 0.08 * gst + 0.07 * delay - 0.05 * margin
    elif model_type == "msme_cold_start":
        noise = _stable_noise(df, model_type)
        promoter = df["promoter_alt_score_norm"].to_numpy()
        capital = df["capital_sufficiency_ratio"].to_numpy()
        docs = df["registration_document_consistency"].to_numpy()
        risk = 0.56 + 0.82 * noise - 0.03 * promoter - 0.03 * capital - 0.02 * docs
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return np.clip(risk, 0.0, 1.0)


def cibil_like_cutoff(df: pd.DataFrame, model_type: str) -> float:
    risk = cibil_like_risk(df, model_type)
    approval_rate = SYNTHETIC_CIBIL_APPROVAL_RATES[model_type]
    return float(np.quantile(risk, approval_rate))


def cibil_like_decision(df: pd.DataFrame, model_type: str, cutoff: float | None = None) -> np.ndarray:
    risk = cibil_like_risk(df, model_type)
    if cutoff is None:
        cutoff = cibil_like_cutoff(df, model_type)
    return risk <= cutoff


def tune_operating_point(
    y_true: np.ndarray,
    prob_default: np.ndarray,
    df_eval: pd.DataFrame,
    model_type: str,
) -> OperatingPoint:
    cibil_approved = cibil_like_decision(df_eval, model_type)
    cibil_approval_rate = float(cibil_approved.mean())
    cibil_default_rate = _approved_default_rate(y_true, cibil_approved)

    best: tuple[float, float, float] | None = None
    for cutoff in np.linspace(0.04, 0.82, 220):
        approved = prob_default <= cutoff
        approval_rate = float(approved.mean())
        approved_default = _approved_default_rate(y_true, approved)
        expands_access = approval_rate >= cibil_approval_rate + 0.03
        within_cap = approval_rate <= MAX_ALT_APPROVAL_RATES[model_type]
        lowers_default = approved_default <= cibil_default_rate
        if expands_access and lowers_default and within_cap:
            objective = approval_rate - 0.35 * approved_default
            if best is None or objective > best[0]:
                best = (objective, cutoff, approved_default)

    if best is None:
        # Fall back to the best risk-adjusted candidate, still favoring access.
        for cutoff in np.linspace(0.04, 0.82, 220):
            approved = prob_default <= cutoff
            approval_rate = float(approved.mean())
            approved_default = _approved_default_rate(y_true, approved)
            cap_penalty = max(0.0, approval_rate - MAX_ALT_APPROVAL_RATES[model_type])
            objective = approval_rate - 1.35 * approved_default - 2.0 * cap_penalty
            if best is None or objective > best[0]:
                best = (objective, cutoff, approved_default)

    assert best is not None
    cutoff = best[1]
    approved = prob_default <= cutoff
    return OperatingPoint(
        score_threshold=float(score_from_pd(np.array([cutoff]))[0]),
        pd_cutoff=float(cutoff),
        approval_rate=float(approved.mean()),
        approved_default_rate=_approved_default_rate(y_true, approved),
        cibil_like_approval_rate=cibil_approval_rate,
        cibil_like_default_rate=cibil_default_rate,
    )


def evaluate_predictions(
    y_true: np.ndarray,
    prob_default: np.ndarray,
    operating_point: OperatingPoint,
    df_eval: pd.DataFrame,
    model_type: str,
) -> dict[str, float | int | dict[str, int]]:
    approved = prob_default <= operating_point.pd_cutoff
    predicted_default = prob_default >= 0.50
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_default, labels=[0, 1]).ravel()

    auc = roc_auc_score(y_true, prob_default)
    metrics: dict[str, float | int | dict[str, int]] = {
        "roc_auc": float(auc),
        "gini": float(2.0 * auc - 1.0),
        "average_precision": float(average_precision_score(y_true, prob_default)),
        "brier_loss": float(brier_score_loss(y_true, prob_default)),
        "accuracy_at_pd_0_50": float(accuracy_score(y_true, predicted_default)),
        "approval_rate": float(approved.mean()),
        "approved_default_rate": _approved_default_rate(y_true, approved),
        "portfolio_default_rate": float(np.mean(y_true)),
        "score_threshold": float(operating_point.score_threshold),
        "pd_cutoff": float(operating_point.pd_cutoff),
        "confusion_matrix_at_pd_0_50": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    metrics.update({f"operating_point_{k}": v for k, v in asdict(operating_point).items()})

    cibil_approved = cibil_like_decision(df_eval, model_type)
    metrics["cibil_like_approval_rate"] = float(cibil_approved.mean())
    metrics["cibil_like_approved_default_rate"] = _approved_default_rate(y_true, cibil_approved)
    metrics["approval_lift_vs_cibil_like"] = metrics["approval_rate"] - metrics["cibil_like_approval_rate"]
    metrics["default_rate_delta_vs_cibil_like"] = (
        metrics["approved_default_rate"] - metrics["cibil_like_approved_default_rate"]
    )
    return metrics


def _rank01(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    return series.rank(method="average", pct=True).to_numpy()


def _stable_noise(df: pd.DataFrame, model_type: str) -> np.ndarray:
    if "application_id" in df.columns:
        seed_source = df["application_id"].astype(str)
    elif "business_id" in df.columns:
        seed_source = df["business_id"].astype(str)
    else:
        seed_source = df.index.to_series().astype(str)
    seed_series = seed_source + ":" + model_type
    hashed = pd.util.hash_pandas_object(seed_series, index=False).to_numpy(dtype=np.uint64)
    return (hashed % 10000).astype(float) / 10000.0


def _approved_default_rate(y_true: np.ndarray, approved: np.ndarray) -> float:
    if approved.sum() == 0:
        return 0.0
    return float(np.mean(y_true[approved]))
