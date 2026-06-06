from __future__ import annotations

from typing import Any

from altcredit.scoring import HIGH_RISK_FEATURES, LOW_RISK_FEATURES, default_feature_payload


def score_json_template(bundle: dict[str, Any]) -> dict[str, Any]:
    spec = bundle["spec"]
    return {
        "model_type": spec.model_type,
        "features": default_feature_payload(bundle),
    }


def testing_json_template(bundle: dict[str, Any]) -> dict[str, Any]:
    spec = bundle["spec"]
    base = default_feature_payload(bundle)
    benign = dict(base)
    stressed = dict(base)

    for column, stats in bundle.get("reference_stats", {}).items():
        if column in benign:
            benign[column] = stats.get("median", benign[column])
        if column not in stressed:
            continue
        if column in HIGH_RISK_FEATURES:
            stressed[column] = stats.get("q95", stressed[column])
        elif column in LOW_RISK_FEATURES:
            stressed[column] = stats.get("q05", stressed[column])
        else:
            stressed[column] = stats.get("median", stressed[column])

    if "default_90d" not in benign:
        benign["default_90d"] = 0
    stressed["default_90d"] = 1

    return {
        "model_type": spec.model_type,
        "records": [benign, stressed],
    }


def normalize_score_json(payload: Any, selected_model_type: str | None = None) -> tuple[str, dict[str, Any]]:
    if isinstance(payload, dict) and "features" in payload:
        model_type = str(payload.get("model_type") or selected_model_type or "")
        features = payload["features"]
    elif isinstance(payload, dict):
        model_type = str(payload.get("model_type") or selected_model_type or "")
        features = {k: v for k, v in payload.items() if k != "model_type"}
    else:
        raise ValueError("Score JSON must be an object with `model_type` and `features`.")

    if not model_type:
        raise ValueError("Score JSON is missing `model_type`.")
    if not isinstance(features, dict):
        raise ValueError("Score JSON `features` must be an object.")
    return model_type, features


def normalize_testing_json(payload: Any, selected_model_type: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(payload, dict) and "records" in payload:
        model_type = str(payload.get("model_type") or selected_model_type or "")
        records = payload["records"]
    elif isinstance(payload, list):
        model_type = str(selected_model_type or "")
        records = payload
    else:
        raise ValueError("Testing JSON must be `{ model_type, records: [...] }` or a list of records.")

    if not model_type:
        raise ValueError("Testing JSON is missing `model_type`.")
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise ValueError("Testing JSON `records` must be a list of objects.")
    return model_type, records
