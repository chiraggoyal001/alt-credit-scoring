from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from altcredit.comparison import evaluate_uploaded_testing_data, load_report_summary
from altcredit.config import MODEL_LABELS, MODEL_TYPES
from altcredit.coverage import CLARIFYING_QUESTIONS, coverage_rows, pasted_text_audit_rows
from altcredit.data import generate_all_datasets, load_generated_datasets
from altcredit.json_io import loads_json_bytes
from altcredit.models import train_all_models
from altcredit.scoring import default_feature_payload, load_model_bundle, score_features, score_trace
from altcredit.templates import (
    normalize_score_json,
    normalize_testing_json,
    score_json_template,
    testing_json_template,
)


st.set_page_config(page_title="AltCredit MVP", layout="wide")

MODELS_DIR = Path("artifacts/models")
REPORTS_DIR = Path("artifacts/reports")
SYNTHETIC_DATA_DIR = Path("data/synthetic")
TESTING_DATA_DIR = Path("data/testing")


def _is_trained() -> bool:
    """Check if model artifacts exist for all models."""
    if not MODELS_DIR.exists():
        return False
    return all((MODELS_DIR / f"{mt}.joblib").exists() for mt in MODEL_TYPES)


def _inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1440px; }
        div[data-testid="stSidebar"] { border-right: 1px solid #e6e8ec; }
        .hero {
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            padding: 18px 20px;
            background: #ffffff;
            margin-bottom: 14px;
        }
        .hero h1 { margin: 0 0 4px 0; font-size: 1.8rem; line-height: 1.2; letter-spacing: 0; color: #152033; }
        .hero p { margin: 0; color: #56616f; font-size: 0.96rem; }
        .panel {
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            padding: 14px;
            background: #ffffff;
        }
        .badge {
            display: inline-block;
            border-radius: 999px;
            padding: 4px 10px;
            font-weight: 700;
            font-size: 0.78rem;
            border: 1px solid transparent;
        }
        .approved { color: #0f6848; background: #e8f6ef; border-color: #bfe8d3; }
        .review { color: #86540a; background: #fff5df; border-color: #f1d39b; }
        .declined { color: #9a2e2e; background: #fdecec; border-color: #f4baba; }
        .small-muted { color: #657282; font-size: 0.86rem; }
        .section-rule { border-top: 1px solid #edf0f3; margin: 0.8rem 0 1rem 0; }
        .metric-strip {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin: 0.4rem 0 1rem 0;
        }
        .metric-card {
            border: 1px solid #e6e8ec;
            border-radius: 8px;
            padding: 12px 14px;
            background: #ffffff;
        }
        .metric-card .label { color: #657282; font-size: 0.82rem; margin-bottom: 4px; }
        .metric-card .value { color: #152033; font-size: 1.45rem; font-weight: 750; }
        .comparison-row {
            display: grid;
            grid-template-columns: minmax(180px, 1.3fr) minmax(180px, 2fr) minmax(180px, 2fr) minmax(96px, 0.9fr);
            gap: 12px;
            align-items: center;
            padding: 10px 0;
            border-bottom: 1px solid #edf0f3;
        }
        .comparison-label { font-weight: 700; color: #202b3c; }
        .bar-shell { height: 12px; border-radius: 999px; background: #edf0f3; overflow: hidden; }
        .bar-alt { height: 12px; background: #19735a; }
        .bar-cibil { height: 12px; background: #738091; }
        .bar-value { color: #56616f; font-size: 0.8rem; margin-top: 3px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _feature_group(column: str, model_type: str) -> str:
    if model_type.startswith("retail"):
        if any(token in column for token in ("inflow", "upi", "obligation", "spend", "payment")):
            return "Cash Flow And Spend"
        if any(token in column for token in ("routine", "commute", "network", "telco")):
            return "Routine And Telco"
        if any(token in column for token in ("bot", "literacy", "blog", "intent")):
            return "Intent And Literacy"
        return "Onboarding And Device"
    if column in {"sector", "capital_tier", "business_location_tier"}:
        return "3D Business Category"
    if any(token in column for token in ("gst", "gstr", "itc", "revenue", "margin", "coverage", "working")):
        return "GST, P&L, Cash Flow"
    if any(token in column for token in ("news", "infrastructure", "industry", "sector_growth", "survival", "peer")):
        return "Macro And Ecosystem"
    return "Promoter And Documentation"


def _feature_form(
    spec: Any,
    defaults: dict[str, Any],
    stats: dict[str, dict[str, float]],
    categories: dict[str, list[str]],
    key_prefix: str,
) -> dict[str, Any]:
    features: dict[str, Any] = {}
    groups: dict[str, list[str]] = {}
    for column in spec.feature_columns:
        groups.setdefault(_feature_group(column, spec.model_type), []).append(column)

    for group, columns in groups.items():
        with st.expander(group, expanded=group in {"Cash Flow And Spend", "3D Business Category", "GST, P&L, Cash Flow"}):
            left, right = st.columns(2)
            for index, column in enumerate(columns):
                target = left if index % 2 == 0 else right
                label = column.replace("_", " ").title()
                if column in spec.categorical_columns:
                    options = categories.get(column, [defaults[column]])
                    default_index = options.index(defaults[column]) if defaults[column] in options else 0
                    features[column] = target.selectbox(
                        label,
                        options=options,
                        index=default_index,
                        key=f"{key_prefix}_{spec.model_type}_{column}",
                    )
                    continue

                ref = stats.get(column, {})
                default = float(defaults[column])
                low = float(ref.get("q05", min(0.0, default)))
                high = float(ref.get("q95", max(1.0, default)))
                if high <= low:
                    high = low + 1.0
                help_text = f"Training q05={low:.3f}, median={float(ref.get('median', default)):.3f}, q95={high:.3f}"
                features[column] = target.slider(
                    label,
                    min_value=low,
                    max_value=high,
                    value=min(max(default, low), high),
                    step=(high - low) / 100.0,
                    help=help_text,
                    key=f"{key_prefix}_{spec.model_type}_{column}",
                )
    return features


def _decision_badge(decision: str) -> None:
    css_class = "approved" if decision == "APPROVED" else "review" if decision == "MANUAL_REVIEW" else "declined"
    st.markdown(f"<span class='badge {css_class}'>{decision}</span>", unsafe_allow_html=True)


def _render_score_result(result: dict[str, Any]) -> None:
    score_col, pd_col, decision_col, exposure_col, apr_col = st.columns([1, 1, 1, 1.2, 1])
    score_col.metric("Alt Score", result["score"])
    pd_col.metric("PD 90d", f"{result['probability_default_90d']:.2%}")
    with decision_col:
        st.caption("Decision")
        _decision_badge(result["decision"])
    exposure_col.metric("Dynamic Limit", f"INR {result['dynamic_exposure_limit_inr']:,}")
    apr_col.metric("Risk APR", f"{result['risk_based_interest_rate_apr']:.2f}%")

    cibil = result["cibil_like_benchmark"]
    st.markdown("<div class='section-rule'></div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CIBIL-Like Score", cibil["score"])
    c2.metric("CIBIL-Like Risk", f"{cibil['risk']:.2%}")
    with c3:
        st.caption("CIBIL-Like Decision")
        _decision_badge(cibil["decision"])
    c4.metric("Risk Cutoff", f"{cibil['risk_cutoff']:.2%}")

    reasons = pd.DataFrame({"Reason Code": result["reason_codes"]})
    st.dataframe(reasons, width="stretch", hide_index=True)


def _render_stage_trace(trace: dict[str, Any]) -> None:
    _render_score_result(trace["result"])
    for stage in trace["stages"]:
        with st.expander(stage["stage"], expanded=stage["stage"].startswith(("1.", "4.", "5.", "6."))):
            st.write(stage["calculation"])
            output = stage["output"]
            if isinstance(output, list):
                st.dataframe(pd.DataFrame(output), width="stretch", hide_index=True)
            elif isinstance(output, dict):
                st.json(output)
            else:
                st.write(output)


def _render_fairness_summary(fairness: dict[str, Any]) -> None:
    rows = []
    for model_name, report in fairness.items():
        for column, detail in report.items():
            rows.append(
                {
                    "Model": model_name,
                    "Audit Column": column,
                    "Disparate Impact Ratio": detail.get("disparate_impact_ratio"),
                    "Passes 80 Percent Rule": detail.get("passes_80_percent_rule"),
                }
            )
    if rows:
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def _render_summary_charts(summary: pd.DataFrame) -> None:
    if summary.empty:
        st.info("Run training to generate model comparison reports.")
        return

    avg_approval_lift = (summary["approval_rate"] - summary["cibil_like_approval_rate"]).mean()
    avg_default_delta = (summary["approved_default_rate"] - summary["cibil_like_approved_default_rate"]).mean()
    avg_gini = summary["gini"].mean()
    st.markdown(
        f"""
        <div class="metric-strip">
          <div class="metric-card"><div class="label">Average Approval Lift</div><div class="value">{avg_approval_lift:.2%}</div></div>
          <div class="metric-card"><div class="label">Average Default Delta</div><div class="value">{avg_default_delta:.2%}</div></div>
          <div class="metric-card"><div class="label">Average Gini</div><div class="value">{avg_gini:.3f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Approval Rate")
    _render_comparison_bars(summary, "approval_rate", "cibil_like_approval_rate", higher_is_better=True)
    st.markdown("#### Approved Default Rate")
    _render_comparison_bars(summary, "approved_default_rate", "cibil_like_approved_default_rate", higher_is_better=False)

    display = summary.copy()
    for column in ("approval_rate", "approved_default_rate", "cibil_like_approval_rate", "cibil_like_approved_default_rate"):
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    display["gini"] = display["gini"].map(lambda value: f"{value:.3f}")
    display["score_threshold"] = display["score_threshold"].map(lambda value: f"{value:.1f}")
    st.dataframe(display, width="stretch", hide_index=True)


def _render_comparison_bars(
    summary: pd.DataFrame,
    alt_column: str,
    cibil_column: str,
    higher_is_better: bool,
) -> None:
    rows: list[str] = []
    for _, row in summary.iterrows():
        alt_value = float(row[alt_column])
        cibil_value = float(row[cibil_column])
        alt_width = max(0.0, min(100.0, alt_value * 100.0))
        cibil_width = max(0.0, min(100.0, cibil_value * 100.0))
        delta = alt_value - cibil_value
        outcome = "better" if (delta >= 0 if higher_is_better else delta <= 0) else "watch"
        rows.append(
            f"""
            <div class="comparison-row">
              <div class="comparison-label">{row['model_label']}</div>
              <div>
                <div class="bar-shell"><div class="bar-alt" style="width: {alt_width:.1f}%"></div></div>
                <div class="bar-value">AltCredit {alt_value:.2%}</div>
              </div>
              <div>
                <div class="bar-shell"><div class="bar-cibil" style="width: {cibil_width:.1f}%"></div></div>
                <div class="bar-value">CIBIL-like {cibil_value:.2%}</div>
              </div>
              <div class="bar-value">{outcome}: {delta:+.2%}</div>
            </div>
            """
        )
    st.markdown("\n".join(rows), unsafe_allow_html=True)


def _format_comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for column in ("alt_credit", "cibil_like", "delta"):
        df[column] = df[column].map(_format_value)
    return df.rename(
        columns={
            "parameter": "Parameter",
            "alt_credit": "AltCredit",
            "cibil_like": "CIBIL-Like",
            "delta": "Delta",
            "metric_key": "Metric Key",
        }
    )


def _format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    numeric = float(value)
    if abs(numeric) <= 1.0:
        return f"{numeric:.2%}"
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}"
    return f"{numeric:.3f}"


def _load_bundle(model_type: str) -> dict[str, Any]:
    return load_model_bundle(model_type, MODELS_DIR)


def _render_header(model_type: str, bundle: dict[str, Any]) -> None:
    spec = bundle["spec"]
    st.markdown(
        f"""
        <div class="hero">
          <h1>Alternative Credit Scoring Simulator</h1>
          <p>{MODEL_LABELS[model_type]} - {len(spec.feature_columns)} features - base exposure INR {spec.base_exposure_limit_inr:,}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


_inject_css()


def _is_trained() -> bool:
    """Check if model artifacts exist for all models."""
    if not MODELS_DIR.exists():
        return False
    return all((MODELS_DIR / f"{mt}.joblib").exists() for mt in MODEL_TYPES)


if not _is_trained():
    st.warning(
        "First-time setup: preparing synthetic data and training models. "
        "This will take a few minutes on Streamlit Cloud..."
    )
    with st.spinner("Initializing models (smoke-test mode)..."):
        # Use existing data if available in the repo, otherwise generate
        data_exists = SYNTHETIC_DATA_DIR.exists() and all(
            (SYNTHETIC_DATA_DIR / f"{mt}.csv").exists() for mt in MODEL_TYPES
        )
        if data_exists:
            datasets = load_generated_datasets(SYNTHETIC_DATA_DIR)
        else:
            datasets = generate_all_datasets(rows_per_model=2000, seed=42)

        train_all_models(datasets, fast=True)
    st.success("Models trained successfully! Reloading the app...")
    st.rerun()

st.sidebar.title("AltCredit MVP")
model_type = st.sidebar.selectbox(
    "Model",
    options=list(MODEL_LABELS),
    format_func=lambda value: MODEL_LABELS[value],
)
page = st.sidebar.radio(
    "View",
    [
        "Overview",
        "Score Lab",
        "Calculation Trace",
        "Testing Data",
        "Model Comparison",
        "JSON Templates",
        "Coverage Audit",
    ],
)

bundle = _load_bundle(model_type)
spec = bundle["spec"]
defaults = default_feature_payload(bundle)
stats = bundle.get("reference_stats", {})
categories = bundle.get("categorical_values", {})
_render_header(model_type, bundle)

st.sidebar.metric("Score Threshold", f"{bundle['operating_point'].score_threshold:.1f}")
st.sidebar.metric("PD Cutoff", f"{bundle['operating_point'].pd_cutoff:.2%}")

if page == "Overview":
    summary = load_report_summary(REPORTS_DIR)
    _render_summary_charts(summary)
    st.subheader("Selected Model Snapshot")
    result = score_features(bundle, defaults)
    _render_score_result(result)

elif page == "Score Lab":
    left, right = st.columns([1.15, 0.85])
    with left:
        st.subheader("Manual Score Test")
        manual_features = _feature_form(spec, defaults, stats, categories, "score_lab")
    with right:
        st.subheader("Result")
        _render_score_result(score_features(bundle, manual_features))

    st.subheader("Upload JSON For Score Testing")
    score_upload = st.file_uploader("Score JSON", type=["json"], key="score_json_upload")
    if score_upload is not None:
        try:
            uploaded_payload = loads_json_bytes(score_upload.getvalue())
            uploaded_model_type, uploaded_features = normalize_score_json(uploaded_payload, model_type)
            uploaded_bundle = _load_bundle(uploaded_model_type)
            uploaded_result = score_features(uploaded_bundle, uploaded_features)
            st.success(f"Scored uploaded JSON with {MODEL_LABELS[uploaded_model_type]}")
            _render_score_result(uploaded_result)
            with st.expander("Raw Score Response"):
                st.code(json.dumps(uploaded_result, indent=2), language="json")
        except Exception as exc:
            st.error(str(exc))

elif page == "Calculation Trace":
    st.subheader("Stage-By-Stage Calculation")
    trace_features = _feature_form(spec, defaults, stats, categories, "trace_lab")
    _render_stage_trace(score_trace(bundle, trace_features))

    st.subheader("Trace Uploaded Score JSON")
    trace_upload = st.file_uploader("Trace JSON", type=["json"], key="trace_json_upload")
    if trace_upload is not None:
        try:
            uploaded_payload = loads_json_bytes(trace_upload.getvalue())
            uploaded_model_type, uploaded_features = normalize_score_json(uploaded_payload, model_type)
            uploaded_bundle = _load_bundle(uploaded_model_type)
            _render_stage_trace(score_trace(uploaded_bundle, uploaded_features))
        except Exception as exc:
            st.error(str(exc))

elif page == "Testing Data":
    st.subheader("Upload JSON To Replace Testing Data")
    testing_upload = st.file_uploader("Testing Data JSON", type=["json"], key="testing_json_upload")
    if testing_upload is not None:
        try:
            uploaded_payload = loads_json_bytes(testing_upload.getvalue())
            uploaded_model_type, records = normalize_testing_json(uploaded_payload, model_type)
            uploaded_bundle = _load_bundle(uploaded_model_type)
            TESTING_DATA_DIR.mkdir(parents=True, exist_ok=True)
            (TESTING_DATA_DIR / f"{uploaded_model_type}_replacement.json").write_text(
                json.dumps({"model_type": uploaded_model_type, "records": records}, indent=2),
                encoding="utf-8",
            )
            evaluation = evaluate_uploaded_testing_data(uploaded_bundle, records, reports_dir=REPORTS_DIR)
            st.success(f"Replacement testing data saved and evaluated for {MODEL_LABELS[uploaded_model_type]}.")
            st.dataframe(_format_comparison_table(evaluation["comparison_table"]), width="stretch", hide_index=True)
            st.download_button(
                "Download Evaluation JSON",
                data=json.dumps(evaluation, indent=2),
                file_name=f"{uploaded_model_type}_uploaded_evaluation.json",
                mime="application/json",
            )
            st.subheader("Fairness Summary")
            _render_fairness_summary(evaluation["fairness"])
        except Exception as exc:
            st.error(str(exc))

    st.subheader("Current Holdout Data")
    holdout_path = REPORTS_DIR / f"{model_type}_scored_holdout.csv"
    if holdout_path.exists():
        holdout = pd.read_csv(holdout_path)
        st.dataframe(holdout.head(50), width="stretch", hide_index=True)
    else:
        st.info("No holdout file found yet.")

elif page == "Model Comparison":
    st.subheader("Portfolio Comparison")
    _render_summary_charts(load_report_summary(REPORTS_DIR))

    st.subheader("Detailed Parameter Comparison")
    holdout_path = REPORTS_DIR / f"{model_type}_scored_holdout.csv"
    if holdout_path.exists():
        holdout_records = pd.read_csv(holdout_path).to_dict(orient="records")
        detailed = evaluate_uploaded_testing_data(bundle, holdout_records, reports_dir=REPORTS_DIR, persist=False)
        st.dataframe(_format_comparison_table(detailed["comparison_table"]), width="stretch", hide_index=True)
        st.subheader("Fairness Summary")
        _render_fairness_summary(detailed["fairness"])
    else:
        st.info("No scored holdout data found for this model.")

elif page == "JSON Templates":
    st.subheader("Score JSON Template")
    score_template = score_json_template(bundle)
    st.code(json.dumps(score_template, indent=2), language="json")
    st.download_button(
        "Download Score Template",
        data=json.dumps(score_template, indent=2),
        file_name=f"{model_type}_score_template.json",
        mime="application/json",
    )

    st.subheader("Replacement Testing Data JSON Template")
    testing_template = testing_json_template(bundle)
    st.code(json.dumps(testing_template, indent=2), language="json")
    st.download_button(
        "Download Testing Data Template",
        data=json.dumps(testing_template, indent=2),
        file_name=f"{model_type}_testing_data_template.json",
        mime="application/json",
    )

elif page == "Coverage Audit":
    st.subheader("Pasted Text Completeness Check")
    st.dataframe(
        pd.DataFrame(pasted_text_audit_rows()).rename(
            columns={
                "check": "Check",
                "status": "Status",
                "finding": "Finding",
                "fix_in_mvp": "Fix In MVP",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Original Brief Coverage")
    coverage = pd.DataFrame(coverage_rows())
    st.dataframe(
        coverage.rename(
            columns={
                "area": "Area",
                "status": "Status",
                "implementation": "Implementation",
                "evidence": "Evidence",
                "remaining_note": "Remaining Note",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Clarifying Questions Still Needed Before Production")
    st.dataframe(pd.DataFrame({"Question": CLARIFYING_QUESTIONS}), width="stretch", hide_index=True)
