from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd

from altcredit.backtest import score_from_pd
from altcredit.config import MODEL_LABELS, OperatingPoint
from altcredit.features import ModelSpec, get_model_spec

warnings.filterwarnings("ignore", message="is_sparse is deprecated.*", category=FutureWarning)


HIGH_RISK_FEATURES = {
    "upi_debit_velocity_variance",
    "fixed_obligation_ratio",
    "discretionary_spend_ratio",
    "routine_entropy",
    "bot_financial_stress_index",
    "high_risk_app_ratio",
    "onboarding_hesitation_score",
    "copy_paste_flag",
    "gstr1_filing_delay_days",
    "daily_qr_upi_volume_variance_7d",
    "bounced_mandates_30d",
    "supplier_concentration_risk",
    "working_capital_gap_ratio",
    "industry_sector_beta",
    "district_infrastructure_anomaly_index",
    "peer_default_rate",
}

LOW_RISK_FEATURES = {
    "inflow_stability_90d",
    "essential_payment_timeliness",
    "commute_consistency",
    "telco_recharge_regularity",
    "prior_loan_recovery_ratio",
    "network_working_ratio",
    "bot_repayment_intent_score",
    "financial_literacy_score",
    "blog_completion_rate",
    "device_hardware_score",
    "device_integrity_score",
    "utility_finance_app_ratio",
    "typing_cadence_stability",
    "sms_utility_payment_regularity",
    "sms_income_signal_strength",
    "telco_network_stability",
    "consent_completeness",
    "network_mobility_consistency",
    "revenue_growth_90d",
    "gross_margin_ratio",
    "debt_service_coverage_proxy",
    "gst_filing_consistency",
    "itc_reconciliation_ratio",
    "district_news_sentiment_score",
    "sector_growth_index",
    "promoter_alt_score_norm",
    "promoter_prior_recovery_ratio",
    "sector_survival_rate",
    "pincode_cluster_survival_rate",
    "capital_sufficiency_ratio",
    "registration_document_consistency",
    "licensing_completeness",
    "startup_expense_coverage_ratio",
}


def build_reference_stats(df: pd.DataFrame, spec: ModelSpec) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for column in spec.numeric_columns:
        series = pd.to_numeric(df[column], errors="coerce")
        stats[column] = {
            "q05": float(series.quantile(0.05)),
            "q25": float(series.quantile(0.25)),
            "median": float(series.quantile(0.50)),
            "q75": float(series.quantile(0.75)),
            "q95": float(series.quantile(0.95)),
        }
    return stats


def load_model_bundle(model_type: str, models_dir: str | Path = "artifacts/models") -> dict[str, Any]:
    path = Path(models_dir) / f"{model_type}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}. Run `python -m altcredit.cli run-all` first.")
    return joblib.load(path)


def load_all_bundles(models_dir: str | Path = "artifacts/models") -> dict[str, dict[str, Any]]:
    return {
        model_type: load_model_bundle(model_type, models_dir)
        for model_type in MODEL_LABELS
    }


def score_features(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    spec: ModelSpec = bundle["spec"]
    operating_point: OperatingPoint = bundle["operating_point"]
    model = bundle["model"]

    complete_features = _complete_features(bundle, features)
    row = _coerce_row(complete_features, spec)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        prob_default = float(model.predict_proba(row[list(spec.feature_columns)])[0][1])
    score = float(score_from_pd(np.array([prob_default]))[0])
    status = _decision_status(score, operating_point.score_threshold)
    exposure = _dynamic_exposure_limit(spec, row.iloc[0].to_dict(), score, status)
    interest_rate = _risk_based_interest_rate(prob_default, spec.model_type, status)
    cibil = cibil_like_single_score(bundle, row.iloc[0].to_dict())

    return {
        "model_type": spec.model_type,
        "model_label": MODEL_LABELS[spec.model_type],
        "score": round(score, 1),
        "probability_default_90d": round(prob_default, 4),
        "decision": status,
        "score_threshold": round(float(operating_point.score_threshold), 1),
        "dynamic_exposure_limit_inr": int(round(exposure)),
        "risk_based_interest_rate_apr": round(interest_rate, 2),
        "reason_codes": reason_codes(row.iloc[0].to_dict(), bundle.get("reference_stats", {}), spec),
        "cibil_like_benchmark": cibil,
    }


def score_dataframe(bundle: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    spec: ModelSpec = bundle["spec"]
    operating_point: OperatingPoint = bundle["operating_point"]
    out = df.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        prob_default = bundle["model"].predict_proba(out[list(spec.feature_columns)])[:, 1]
    out["predicted_pd_90d"] = prob_default
    out["alt_score"] = score_from_pd(prob_default)
    out["decision"] = np.where(
        out["alt_score"] >= operating_point.score_threshold,
        "APPROVED",
        np.where(out["alt_score"] >= operating_point.score_threshold - 35.0, "MANUAL_REVIEW", "DECLINED"),
    )
    out["approved"] = out["decision"] == "APPROVED"
    out["dynamic_exposure_limit_inr"] = [
        int(round(_dynamic_exposure_limit(spec, row.to_dict(), float(row["alt_score"]), str(row["decision"]))))
        for _, row in out.iterrows()
    ]
    out["risk_based_interest_rate_apr"] = [
        round(_risk_based_interest_rate(float(pd_value), spec.model_type, str(decision)), 2)
        for pd_value, decision in zip(out["predicted_pd_90d"], out["decision"])
    ]
    out.attrs["score_threshold"] = float(operating_point.score_threshold)
    out.attrs["pd_cutoff"] = float(operating_point.pd_cutoff)
    return out


def score_trace(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    spec: ModelSpec = bundle["spec"]
    operating_point: OperatingPoint = bundle["operating_point"]
    complete_features = _complete_features(bundle, features)
    row = _coerce_row(complete_features, spec)
    row_dict = row.iloc[0].to_dict()
    result = score_features(bundle, row_dict)
    components = component_model_probabilities(bundle, row)

    stages = [
        {
            "stage": "1. Route",
            "calculation": "Select scoring engine from model_type.",
            "output": {
                "model_type": spec.model_type,
                "model_label": MODEL_LABELS[spec.model_type],
                "base_exposure_limit_inr": spec.base_exposure_limit_inr,
                "segment_context": _segment_context(row_dict, spec),
            },
        },
        {
            "stage": "2. Validate Input",
            "calculation": "Coerce uploaded/manual JSON into the exact feature schema.",
            "output": {
                "required_features": len(spec.feature_columns),
                "received_features": len([c for c in spec.feature_columns if c in features]),
                "missing_features_filled": [c for c in spec.feature_columns if c not in features],
                "fill_strategy": "Missing score-test fields use training medians or first known category.",
            },
        },
        {
            "stage": "3. Feature Diagnostics",
            "calculation": "Compare each numeric input with training reference quantiles.",
            "output": feature_diagnostics(row_dict, bundle.get("reference_stats", {}), spec),
        },
        {
            "stage": "4. Ensemble PD",
            "calculation": "Preprocess features, collect base learner probabilities, then soft-vote.",
            "output": {
                "base_learners": components,
                "final_probability_default_90d": result["probability_default_90d"],
            },
        },
        {
            "stage": "5. Score Scaling",
            "calculation": "Alt score = 300 + 600 * (1 - PD).",
            "output": {
                "probability_default_90d": result["probability_default_90d"],
                "score": result["score"],
            },
        },
        {
            "stage": "6. Operating Point",
            "calculation": "Compare score and PD with the tuned approval threshold.",
            "output": {
                "score_threshold": round(float(operating_point.score_threshold), 1),
                "pd_cutoff": round(float(operating_point.pd_cutoff), 4),
                "decision": result["decision"],
            },
        },
        {
            "stage": "7. Exposure And Pricing",
            "calculation": "Apply score factor plus model-specific live risk modifiers.",
            "output": {
                "dynamic_exposure_limit_inr": result["dynamic_exposure_limit_inr"],
                "risk_based_interest_rate_apr": result["risk_based_interest_rate_apr"],
                "reason_codes": result["reason_codes"],
            },
        },
        {
            "stage": "8. CIBIL-Like Benchmark",
            "calculation": "Score the same payload with the synthetic lagging-data benchmark.",
            "output": result["cibil_like_benchmark"],
        },
    ]
    return {"result": result, "stages": stages}


def reason_codes(features: dict[str, Any], reference_stats: dict[str, dict[str, float]], spec: ModelSpec) -> list[str]:
    reasons: list[tuple[float, str]] = []
    for column in spec.numeric_columns:
        if column not in reference_stats:
            continue
        value = _safe_float(features.get(column))
        stats = reference_stats[column]
        iqr = max(stats["q75"] - stats["q25"], 1e-6)
        if column in HIGH_RISK_FEATURES and value > stats["q75"]:
            severity = (value - stats["median"]) / iqr
            reasons.append((severity, f"High {column.replace('_', ' ')}"))
        elif column in LOW_RISK_FEATURES and value < stats["q25"]:
            severity = (stats["median"] - value) / iqr
            reasons.append((severity, f"Low {column.replace('_', ' ')}"))

    sector = features.get("sector")
    if sector in {"food_service", "textiles", "light_manufacturing"}:
        reasons.append((0.35, f"Elevated sector volatility: {sector}"))
    capital = features.get("capital_tier")
    if capital == "micro":
        reasons.append((0.25, "Micro capital tier requires tighter exposure ramp"))

    reasons.sort(key=lambda item: item[0], reverse=True)
    return [text for _, text in reasons[:3]] or ["No dominant adverse factor"]


def default_feature_payload(bundle: dict[str, Any]) -> dict[str, Any]:
    spec: ModelSpec = bundle["spec"]
    reference_stats = bundle.get("reference_stats", {})
    payload: dict[str, Any] = {}
    for column in spec.feature_columns:
        if column in spec.categorical_columns:
            payload[column] = _first_category(bundle, column)
        else:
            payload[column] = reference_stats.get(column, {}).get("median", 0.5)
    return payload


def _complete_features(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    defaults = default_feature_payload(bundle)
    complete = dict(defaults)
    complete.update(features)
    return complete


def feature_diagnostics(
    features: dict[str, Any],
    reference_stats: dict[str, dict[str, float]],
    spec: ModelSpec,
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for column in spec.feature_columns:
        value = features.get(column)
        if column in spec.categorical_columns:
            diagnostics.append(
                {
                    "feature": column,
                    "value": value,
                    "type": "categorical",
                    "position": "category",
                    "risk_direction": "category-specific",
                }
            )
            continue

        stats = reference_stats.get(column, {})
        numeric_value = _safe_float(value)
        percentile_band = _percentile_band(numeric_value, stats)
        if column in HIGH_RISK_FEATURES:
            direction = "higher increases risk"
            flag = percentile_band in {"high", "very_high"}
        elif column in LOW_RISK_FEATURES:
            direction = "higher lowers risk"
            flag = percentile_band in {"low", "very_low"}
        else:
            direction = "non-monotonic or context-dependent"
            flag = False
        diagnostics.append(
            {
                "feature": column,
                "value": round(numeric_value, 4),
                "type": "numeric",
                "position": percentile_band,
                "risk_direction": direction,
                "adverse_signal": bool(flag),
            }
        )
    return diagnostics


def component_model_probabilities(bundle: dict[str, Any], row: pd.DataFrame) -> dict[str, float]:
    model = bundle["model"]
    spec: ModelSpec = bundle["spec"]
    if "preprocessor" not in model.named_steps or "classifier" not in model.named_steps:
        return {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        transformed = model.named_steps["preprocessor"].transform(row[list(spec.feature_columns)])

    classifier = model.named_steps["classifier"]
    output: dict[str, float] = {}
    for name, estimator in classifier.named_estimators_.items():
        if hasattr(estimator, "predict_proba"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                output[name] = round(float(estimator.predict_proba(transformed)[0][1]), 4)
    return output


def cibil_like_single_score(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    spec: ModelSpec = bundle["spec"]
    stats = bundle.get("reference_stats", {})
    risk = _cibil_like_single_risk(spec.model_type, features, stats)
    score = float(score_from_pd(np.array([risk]))[0])
    cutoff = float(bundle.get("cibil_like_risk_cutoff", 0.50))
    decision = "APPROVED" if risk <= cutoff else "DECLINED"
    return {
        "score": round(score, 1),
        "risk": round(float(risk), 4),
        "risk_cutoff": round(cutoff, 4),
        "decision": decision,
        "inputs_used": _cibil_input_columns(spec.model_type),
    }


def _coerce_row(features: dict[str, Any], spec: ModelSpec) -> pd.DataFrame:
    row: dict[str, Any] = {}
    for column in spec.feature_columns:
        if column in features:
            row[column] = features[column]
        else:
            row[column] = "" if column in spec.categorical_columns else 0.0
    return pd.DataFrame([row])


def _segment_context(features: dict[str, Any], spec: ModelSpec) -> dict[str, Any]:
    if not spec.model_type.startswith("msme"):
        return {"customer_family": "individual", "route": "active" if "active" in spec.model_type else "cold_start"}
    return {
        "customer_family": "msme",
        "route": "active" if spec.model_type == "msme_active" else "cold_start",
        "sector": features.get("sector"),
        "capital_tier": features.get("capital_tier"),
        "business_location_tier": features.get("business_location_tier"),
    }


def _percentile_band(value: float, stats: dict[str, float]) -> str:
    if not stats:
        return "unknown"
    if value <= stats["q05"]:
        return "very_low"
    if value <= stats["q25"]:
        return "low"
    if value <= stats["median"]:
        return "below_median"
    if value <= stats["q75"]:
        return "above_median"
    if value <= stats["q95"]:
        return "high"
    return "very_high"


def _rank_from_stats(value: Any, stats: dict[str, float]) -> float:
    numeric = _safe_float(value)
    low = stats.get("q05", 0.0)
    high = stats.get("q95", 1.0)
    if high <= low:
        return 0.5
    return float(np.clip((numeric - low) / (high - low), 0.0, 1.0))


def _cibil_like_single_risk(model_type: str, features: dict[str, Any], stats: dict[str, dict[str, float]]) -> float:
    # Single-payload benchmark uses the same lagging variables as the holdout
    # benchmark, with the stochastic bureau-noise term fixed at its midpoint.
    noise = 0.50
    if model_type == "retail_active":
        income = _rank_from_stats(features.get("monthly_inflow_inr"), stats.get("monthly_inflow_inr", {}))
        stability = _safe_float(features.get("inflow_stability_90d"))
        obligations = _safe_float(features.get("fixed_obligation_ratio"))
        recovery = _safe_float(features.get("prior_loan_recovery_ratio"))
        risk = 0.52 + 0.58 * noise - 0.08 * income - 0.08 * stability + 0.08 * obligations - 0.05 * recovery
    elif model_type == "retail_cold_start":
        device = _safe_float(features.get("device_hardware_score"))
        sms_income = _safe_float(features.get("sms_income_signal_strength"))
        utility = _safe_float(features.get("sms_utility_payment_regularity"))
        risk = 0.56 + 0.62 * noise - 0.07 * device - 0.08 * sms_income - 0.06 * utility
    elif model_type == "msme_active":
        revenue = _rank_from_stats(features.get("monthly_revenue_inr"), stats.get("monthly_revenue_inr", {}))
        gst = _safe_float(features.get("gst_filing_consistency"))
        delay = _rank_from_stats(features.get("gstr1_filing_delay_days"), stats.get("gstr1_filing_delay_days", {}))
        margin = _safe_float(features.get("gross_margin_ratio"))
        risk = 0.55 + 0.58 * noise - 0.07 * revenue - 0.08 * gst + 0.07 * delay - 0.05 * margin
    elif model_type == "msme_cold_start":
        promoter = _safe_float(features.get("promoter_alt_score_norm"))
        capital = _safe_float(features.get("capital_sufficiency_ratio"))
        docs = _safe_float(features.get("registration_document_consistency"))
        risk = 0.56 + 0.82 * noise - 0.03 * promoter - 0.03 * capital - 0.02 * docs
    else:
        risk = 1.0
    return float(np.clip(risk, 0.0, 1.0))


def _cibil_input_columns(model_type: str) -> list[str]:
    return {
        "retail_active": [
            "monthly_inflow_inr",
            "inflow_stability_90d",
            "fixed_obligation_ratio",
            "prior_loan_recovery_ratio",
        ],
        "retail_cold_start": [
            "device_hardware_score",
            "sms_income_signal_strength",
            "sms_utility_payment_regularity",
        ],
        "msme_active": [
            "monthly_revenue_inr",
            "gst_filing_consistency",
            "gstr1_filing_delay_days",
            "gross_margin_ratio",
        ],
        "msme_cold_start": [
            "promoter_alt_score_norm",
            "capital_sufficiency_ratio",
            "registration_document_consistency",
        ],
    }.get(model_type, [])


def _decision_status(score: float, threshold: float) -> str:
    if score >= threshold:
        return "APPROVED"
    if score >= threshold - 35:
        return "MANUAL_REVIEW"
    return "DECLINED"


def _dynamic_exposure_limit(spec: ModelSpec, features: dict[str, Any], score: float, status: str) -> float:
    if status == "DECLINED":
        return 0.0

    score_factor = float(np.clip((score - 300.0) / 600.0, 0.0, 1.0))
    exposure = spec.base_exposure_limit_inr * score_factor

    if spec.model_type == "retail_active":
        exposure *= 1.0 - 0.20 * _safe_float(features.get("upi_debit_velocity_variance"))
        exposure *= 0.90 + 0.20 * _safe_float(features.get("bot_repayment_intent_score"))
    elif spec.model_type == "retail_cold_start":
        exposure *= 0.70 + 0.30 * _safe_float(features.get("consent_completeness"))
        exposure *= 1.0 - 0.18 * _safe_float(features.get("high_risk_app_ratio"))
    elif spec.model_type == "msme_active":
        exposure *= 0.80 + 0.35 * _safe_float(features.get("district_news_sentiment_score"))
        exposure *= 1.0 - 0.25 * _safe_float(features.get("working_capital_gap_ratio"))
    elif spec.model_type == "msme_cold_start":
        exposure *= 0.70 + 0.30 * _safe_float(features.get("promoter_alt_score_norm"))
        exposure *= 1.0 - 0.30 * _safe_float(features.get("peer_default_rate"))

    if status == "MANUAL_REVIEW":
        exposure *= 0.35
    return max(0.0, exposure)


def _risk_based_interest_rate(prob_default: float, model_type: str, status: str) -> float:
    base = 10.0 if model_type.startswith("retail") else 12.0
    stress_addon = 3.0 if status == "MANUAL_REVIEW" else 0.0
    return float(np.clip(base + 28.0 * prob_default + stress_addon, 10.0, 36.0))


def _first_category(bundle: dict[str, Any], column: str) -> str:
    categories = bundle.get("categorical_values", {}).get(column, [])
    return categories[0] if categories else ""


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
