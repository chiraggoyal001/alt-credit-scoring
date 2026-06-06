from __future__ import annotations

from dataclasses import dataclass

from altcredit.config import BASE_EXPOSURE_LIMITS_INR


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    id_column: str
    target_column: str
    feature_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    protected_columns: tuple[str, ...]
    audit_columns: tuple[str, ...]
    base_exposure_limit_inr: int

    @property
    def numeric_columns(self) -> tuple[str, ...]:
        return tuple(c for c in self.feature_columns if c not in self.categorical_columns)


COMMON_INDIVIDUAL_PROTECTED = (
    "protected_gender",
    "age_band",
    "state_group",
    "urbanicity",
)

COMMON_MSME_PROTECTED = (
    "promoter_gender",
    "promoter_age_band",
    "state_group",
    "location_tier",
)

SPECS: dict[str, ModelSpec] = {
    "retail_active": ModelSpec(
        model_type="retail_active",
        id_column="application_id",
        target_column="default_90d",
        feature_columns=(
            "monthly_inflow_inr",
            "inflow_stability_90d",
            "upi_debit_velocity_variance",
            "fixed_obligation_ratio",
            "discretionary_spend_ratio",
            "essential_payment_timeliness",
            "routine_entropy",
            "commute_consistency",
            "telco_recharge_regularity",
            "prior_loan_recovery_ratio",
            "network_working_ratio",
            "bot_financial_stress_index",
            "bot_repayment_intent_score",
            "financial_literacy_score",
            "blog_completion_rate",
        ),
        categorical_columns=(),
        protected_columns=COMMON_INDIVIDUAL_PROTECTED,
        audit_columns=COMMON_INDIVIDUAL_PROTECTED + ("income_band",),
        base_exposure_limit_inr=BASE_EXPOSURE_LIMITS_INR["retail_active"],
    ),
    "retail_cold_start": ModelSpec(
        model_type="retail_cold_start",
        id_column="application_id",
        target_column="default_90d",
        feature_columns=(
            "device_hardware_score",
            "device_integrity_score",
            "utility_finance_app_ratio",
            "high_risk_app_ratio",
            "onboarding_hesitation_score",
            "typing_cadence_stability",
            "copy_paste_flag",
            "sms_utility_payment_regularity",
            "sms_income_signal_strength",
            "telco_network_stability",
            "consent_completeness",
            "network_mobility_consistency",
        ),
        categorical_columns=(),
        protected_columns=COMMON_INDIVIDUAL_PROTECTED,
        audit_columns=COMMON_INDIVIDUAL_PROTECTED + ("device_tier",),
        base_exposure_limit_inr=BASE_EXPOSURE_LIMITS_INR["retail_cold_start"],
    ),
    "msme_active": ModelSpec(
        model_type="msme_active",
        id_column="business_id",
        target_column="default_90d",
        feature_columns=(
            "sector",
            "capital_tier",
            "business_location_tier",
            "monthly_revenue_inr",
            "revenue_growth_90d",
            "gross_margin_ratio",
            "debt_service_coverage_proxy",
            "gstr1_filing_delay_days",
            "gst_filing_consistency",
            "itc_reconciliation_ratio",
            "daily_qr_upi_volume_variance_7d",
            "bounced_mandates_30d",
            "supplier_concentration_risk",
            "working_capital_gap_ratio",
            "industry_sector_beta",
            "district_news_sentiment_score",
            "district_infrastructure_anomaly_index",
            "sector_growth_index",
            "promoter_alt_score_norm",
        ),
        categorical_columns=("sector", "capital_tier", "business_location_tier"),
        protected_columns=COMMON_MSME_PROTECTED,
        audit_columns=COMMON_MSME_PROTECTED + ("sector", "capital_tier", "business_location_tier"),
        base_exposure_limit_inr=BASE_EXPOSURE_LIMITS_INR["msme_active"],
    ),
    "msme_cold_start": ModelSpec(
        model_type="msme_cold_start",
        id_column="business_id",
        target_column="default_90d",
        feature_columns=(
            "sector",
            "capital_tier",
            "business_location_tier",
            "promoter_alt_score_norm",
            "promoter_prior_recovery_ratio",
            "sector_survival_rate",
            "pincode_cluster_survival_rate",
            "capital_sufficiency_ratio",
            "registration_document_consistency",
            "incorporation_age_months",
            "licensing_completeness",
            "peer_default_rate",
            "industry_sector_beta",
            "startup_expense_coverage_ratio",
        ),
        categorical_columns=("sector", "capital_tier", "business_location_tier"),
        protected_columns=COMMON_MSME_PROTECTED,
        audit_columns=COMMON_MSME_PROTECTED + ("sector", "capital_tier", "business_location_tier"),
        base_exposure_limit_inr=BASE_EXPOSURE_LIMITS_INR["msme_cold_start"],
    ),
}


def get_model_spec(model_type: str) -> ModelSpec:
    try:
        return SPECS[model_type]
    except KeyError as exc:
        valid = ", ".join(SPECS)
        raise ValueError(f"Unknown model_type '{model_type}'. Use one of: {valid}") from exc
