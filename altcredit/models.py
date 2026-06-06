from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier

from altcredit.backtest import cibil_like_cutoff, evaluate_predictions, tune_operating_point
from altcredit.config import MODEL_TYPES, RANDOM_SEED
from altcredit.fairness import label_balance_report, selection_rate_report
from altcredit.features import ModelSpec, get_model_spec
from altcredit.scoring import build_reference_stats, score_dataframe

warnings.filterwarnings("ignore", message="is_sparse is deprecated.*", category=FutureWarning)


def build_estimator(spec: ModelSpec, fast: bool = False) -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, list(spec.numeric_columns)),
            ("cat", categorical_transformer, list(spec.categorical_columns)),
        ],
        remainder="drop",
    )

    n_rf = 60 if fast else 180
    n_gb = 45 if fast else 140
    rf_depth = 7 if fast else 10
    tree_depth = 4 if fast else 5

    gradient_boosting = GradientBoostingClassifier(
        n_estimators=n_gb,
        learning_rate=0.045 if spec.model_type.endswith("active") else 0.055,
        max_depth=3,
        random_state=RANDOM_SEED,
    )
    random_forest = RandomForestClassifier(
        n_estimators=n_rf,
        max_depth=rf_depth,
        min_samples_leaf=8,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    svm = CalibratedClassifierCV(
        LinearSVC(class_weight="balanced", C=0.55, max_iter=5000, dual=False, random_state=RANDOM_SEED),
        cv=3,
    )
    decision_tree = DecisionTreeClassifier(
        max_depth=tree_depth,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_SEED,
    )

    weights_by_model = {
        "retail_active": [0.54, 0.28, 0.10, 0.08],
        "retail_cold_start": [0.26, 0.44, 0.10, 0.20],
        "msme_active": [0.58, 0.26, 0.06, 0.10],
        "msme_cold_start": [0.30, 0.35, 0.07, 0.28],
    }
    classifier = VotingClassifier(
        estimators=[
            ("gradient_boosting", gradient_boosting),
            ("random_forest", random_forest),
            ("svm_calibrated", svm),
            ("shallow_decision_tree", decision_tree),
        ],
        voting="soft",
        weights=weights_by_model[spec.model_type],
        n_jobs=None,
    )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_one_model(
    df: pd.DataFrame,
    model_type: str,
    models_dir: str | Path = "artifacts/models",
    reports_dir: str | Path = "artifacts/reports",
    fast: bool = False,
) -> dict[str, Any]:
    spec = get_model_spec(model_type)
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=df[spec.target_column],
    )

    model = build_estimator(spec, fast=fast)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(train_df[list(spec.feature_columns)], train_df[spec.target_column])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        prob_default = model.predict_proba(test_df[list(spec.feature_columns)])[:, 1]
    operating_point = tune_operating_point(
        test_df[spec.target_column].to_numpy(),
        prob_default,
        test_df,
        model_type,
    )
    metrics = evaluate_predictions(
        test_df[spec.target_column].to_numpy(),
        prob_default,
        operating_point,
        test_df,
        model_type,
    )

    bundle = {
        "model": model,
        "spec": spec,
        "operating_point": operating_point,
        "cibil_like_risk_cutoff": cibil_like_cutoff(test_df, model_type),
        "reference_stats": build_reference_stats(train_df, spec),
        "categorical_values": {
            column: sorted(str(v) for v in train_df[column].dropna().unique())
            for column in spec.categorical_columns
        },
        "metrics": metrics,
    }

    scored_test = score_dataframe(bundle, test_df)
    fair_report = selection_rate_report(scored_test, "approved", spec.audit_columns, spec.target_column)
    label_report = label_balance_report(df, spec.audit_columns, spec.target_column)

    model_path = Path(models_dir)
    report_path = Path(reports_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(bundle, model_path / f"{model_type}.joblib")
    _write_json(report_path / f"{model_type}_metrics.json", metrics)
    _write_json(report_path / f"{model_type}_fairness.json", fair_report)
    _write_json(report_path / f"{model_type}_label_balance.json", label_report)
    scored_test.to_csv(report_path / f"{model_type}_scored_holdout.csv", index=False)

    return {
        "model_type": model_type,
        "metrics": metrics,
        "fairness": fair_report,
        "label_balance": label_report,
        "model_path": str(model_path / f"{model_type}.joblib"),
    }


def train_all_models(
    datasets: dict[str, pd.DataFrame],
    models_dir: str | Path = "artifacts/models",
    reports_dir: str | Path = "artifacts/reports",
    fast: bool = False,
) -> dict[str, dict[str, Any]]:
    results = {}
    for model_type in MODEL_TYPES:
        results[model_type] = train_one_model(
            datasets[model_type],
            model_type,
            models_dir=models_dir,
            reports_dir=reports_dir,
            fast=fast,
        )
    _write_json(Path(reports_dir) / "summary.json", _summary(results))
    return results


def _summary(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        model_type: {
            "roc_auc": result["metrics"]["roc_auc"],
            "gini": result["metrics"]["gini"],
            "approval_rate": result["metrics"]["approval_rate"],
            "approved_default_rate": result["metrics"]["approved_default_rate"],
            "cibil_like_approval_rate": result["metrics"]["cibil_like_approval_rate"],
            "cibil_like_approved_default_rate": result["metrics"]["cibil_like_approved_default_rate"],
            "score_threshold": result["metrics"]["score_threshold"],
        }
        for model_type, result in results.items()
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2)


def _json_ready(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if hasattr(value, "__dataclass_fields__"):
        return _json_ready(asdict(value))
    return value
