from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from altcredit.config import DEFAULT_ROWS_PER_MODEL, MODEL_TYPES, RANDOM_SEED


GENDERS = ("female", "male", "non_binary")
AGE_BANDS = ("18_25", "26_35", "36_50", "51_65")
STATE_GROUPS = ("north", "south", "east", "west", "central", "northeast")
URBANICITY = ("metro", "tier_1", "tier_2", "rural")
DEVICE_TIERS = ("entry", "value", "mid", "premium")
SECTORS = (
    "kirana_fmcg",
    "food_service",
    "textiles",
    "light_manufacturing",
    "professional_services",
    "traders_distribution",
    "agri_allied",
    "digital_services",
)
CAPITAL_TIERS = ("micro", "small", "lower_medium", "upper_medium")
LOCATION_TIERS = ("metro", "tier_1", "tier_2", "rural_cluster")

SECTOR_RISK = {
    "kirana_fmcg": 0.00,
    "food_service": 0.00,
    "textiles": 0.00,
    "light_manufacturing": 0.00,
    "professional_services": 0.00,
    "traders_distribution": 0.00,
    "agri_allied": 0.00,
    "digital_services": 0.00,
}

CAPITAL_RISK = {
    "micro": 0.14,
    "small": 0.06,
    "lower_medium": -0.04,
    "upper_medium": -0.10,
}

LOCATION_RISK = {
    "metro": -0.06,
    "tier_1": -0.02,
    "tier_2": 0.04,
    "rural_cluster": 0.10,
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def clipped(values: np.ndarray, low: float = 0.0, high: float = 1.0) -> np.ndarray:
    return np.clip(values, low, high)


def balanced_choice(rng: np.random.Generator, values: tuple[str, ...], n: int) -> np.ndarray:
    repeated = np.resize(np.array(values), n)
    rng.shuffle(repeated)
    return repeated


def income_band(values: np.ndarray) -> np.ndarray:
    return pd.cut(
        values,
        bins=[-np.inf, 18000, 35000, 65000, np.inf],
        labels=["low", "emerging", "stable", "high"],
    ).astype(str)


def _individual_audit_columns(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        "protected_gender": balanced_choice(rng, GENDERS, n),
        "age_band": balanced_choice(rng, AGE_BANDS, n),
        "state_group": balanced_choice(rng, STATE_GROUPS, n),
        "urbanicity": balanced_choice(rng, URBANICITY, n),
    }


def _msme_audit_columns(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    return {
        "promoter_gender": balanced_choice(rng, GENDERS, n),
        "promoter_age_band": balanced_choice(rng, AGE_BANDS, n),
        "state_group": balanced_choice(rng, STATE_GROUPS, n),
        "location_tier": balanced_choice(rng, LOCATION_TIERS, n),
    }


def _rebalance_audit_columns(
    df: pd.DataFrame,
    columns_to_values: dict[str, tuple[str, ...]],
    rng: np.random.Generator,
    score_column: str = "simulated_pd_90d",
    bins: int = 10,
) -> pd.DataFrame:
    """Spread audit-only groups evenly across synthetic risk deciles."""
    ordered_index = df.sort_values(score_column).index.to_numpy()
    chunks = np.array_split(ordered_index, bins)
    for column, values in columns_to_values.items():
        labels = pd.Series(index=df.index, dtype="object")
        for chunk in chunks:
            choices = np.resize(np.array(values), len(chunk))
            rng.shuffle(choices)
            labels.loc[chunk] = choices
        df[column] = labels.astype(str)
    return df


def generate_retail_active(n: int, rng: np.random.Generator) -> pd.DataFrame:
    capacity = rng.normal(0.0, 1.0, n)
    discipline = rng.normal(0.0, 1.0, n)
    stress = rng.normal(0.0, 1.0, n)
    routine = rng.normal(0.0, 1.0, n)

    monthly_inflow = np.exp(10.25 + 0.45 * capacity + rng.normal(0, 0.18, n))
    monthly_inflow = np.clip(monthly_inflow, 8000, 220000)
    inflow_stability = clipped(0.55 + 0.18 * discipline + 0.12 * capacity - 0.12 * stress + rng.normal(0, 0.10, n))
    upi_variance = clipped(0.35 + 0.18 * stress - 0.12 * discipline + rng.normal(0, 0.10, n))
    fixed_obligation = clipped(0.36 - 0.10 * capacity + 0.10 * stress + rng.normal(0, 0.08, n), 0.02, 0.92)
    discretionary = clipped(0.33 + 0.12 * stress - 0.10 * discipline + rng.normal(0, 0.10, n), 0.02, 0.95)
    timeliness = clipped(0.70 + 0.18 * discipline - 0.10 * stress + rng.normal(0, 0.08, n))
    routine_entropy = clipped(0.38 + 0.18 * stress - 0.20 * routine + rng.normal(0, 0.12, n))
    commute_consistency = clipped(0.65 + 0.22 * routine - 0.08 * stress + rng.normal(0, 0.10, n))
    telco_regularity = clipped(0.68 + 0.15 * discipline + 0.06 * capacity + rng.normal(0, 0.09, n))
    recovery_ratio = clipped(0.70 + 0.18 * discipline - 0.08 * stress + rng.normal(0, 0.10, n))
    network_working = clipped(0.68 + 0.18 * routine + 0.08 * capacity + rng.normal(0, 0.10, n))
    bot_stress = clipped(0.35 + 0.24 * stress - 0.10 * discipline + rng.normal(0, 0.10, n))
    intent = clipped(0.62 + 0.18 * discipline - 0.08 * stress + rng.normal(0, 0.11, n))
    literacy = clipped(0.54 + 0.16 * discipline + 0.08 * capacity + rng.normal(0, 0.13, n))
    blog_completion = clipped(0.42 + 0.17 * discipline + 0.05 * capacity + rng.normal(0, 0.14, n))

    risk_logit = (
        -2.05
        + 1.18 * upi_variance
        + 1.20 * fixed_obligation
        + 0.95 * discretionary
        + 1.02 * routine_entropy
        + 1.18 * bot_stress
        - 1.35 * inflow_stability
        - 1.05 * timeliness
        - 1.10 * intent
        - 0.72 * literacy
        - 0.62 * recovery_ratio
        - 0.45 * network_working
        + rng.normal(0, 0.42, n)
    )
    pd_90d = sigmoid(risk_logit)
    target = rng.binomial(1, pd_90d)

    data = {
        "application_id": [f"IND-A-{i:06d}" for i in range(n)],
        "model_type": "retail_active",
        **_individual_audit_columns(rng, n),
        "monthly_inflow_inr": monthly_inflow.round(2),
        "income_band": income_band(monthly_inflow),
        "inflow_stability_90d": inflow_stability,
        "upi_debit_velocity_variance": upi_variance,
        "fixed_obligation_ratio": fixed_obligation,
        "discretionary_spend_ratio": discretionary,
        "essential_payment_timeliness": timeliness,
        "routine_entropy": routine_entropy,
        "commute_consistency": commute_consistency,
        "telco_recharge_regularity": telco_regularity,
        "prior_loan_recovery_ratio": recovery_ratio,
        "network_working_ratio": network_working,
        "bot_financial_stress_index": bot_stress,
        "bot_repayment_intent_score": intent,
        "financial_literacy_score": literacy,
        "blog_completion_rate": blog_completion,
        "simulated_pd_90d": pd_90d,
        "default_90d": target,
    }
    df = pd.DataFrame(data)
    return _rebalance_audit_columns(
        df,
        {
            "protected_gender": GENDERS,
            "age_band": AGE_BANDS,
            "state_group": STATE_GROUPS,
            "urbanicity": URBANICITY,
        },
        rng,
    )


def generate_retail_cold_start(n: int, rng: np.random.Generator) -> pd.DataFrame:
    capacity = rng.normal(0.0, 1.0, n)
    discipline = rng.normal(0.0, 1.0, n)
    friction = rng.normal(0.0, 1.0, n)
    consent = rng.normal(0.0, 1.0, n)

    device_hardware = clipped(0.50 + rng.normal(0, 0.17, n))
    device_integrity = clipped(0.68 + 0.18 * discipline - 0.10 * friction + rng.normal(0, 0.10, n))
    utility_finance_ratio = clipped(0.38 + 0.13 * discipline + 0.05 * capacity + rng.normal(0, 0.11, n))
    high_risk_app_ratio = clipped(0.13 + 0.12 * friction - 0.10 * discipline + rng.normal(0, 0.06, n), 0.0, 0.65)
    hesitation = clipped(0.42 + 0.20 * friction - 0.07 * discipline + rng.normal(0, 0.11, n))
    typing_stability = clipped(0.60 + 0.18 * discipline - 0.14 * friction + rng.normal(0, 0.10, n))
    copy_paste = rng.binomial(1, clipped(0.08 + 0.22 * high_risk_app_ratio + 0.16 * hesitation, 0.01, 0.65))
    sms_utility = clipped(0.54 + 0.18 * discipline + 0.08 * capacity + rng.normal(0, 0.12, n))
    sms_income = clipped(0.49 + 0.18 * capacity + 0.09 * discipline + rng.normal(0, 0.13, n))
    telco_stability = clipped(0.62 + 0.14 * discipline + 0.05 * capacity - 0.06 * friction + rng.normal(0, 0.11, n))
    consent_completeness = clipped(0.66 + 0.18 * consent + 0.06 * discipline + rng.normal(0, 0.10, n))
    network_mobility = clipped(0.58 + 0.11 * capacity + 0.10 * discipline - 0.08 * friction + rng.normal(0, 0.12, n))

    risk_logit = (
        -0.45
        - 0.10 * device_hardware
        - 1.05 * device_integrity
        - 0.86 * utility_finance_ratio
        + 1.34 * high_risk_app_ratio
        + 1.00 * hesitation
        - 0.92 * typing_stability
        + 0.62 * copy_paste
        - 1.00 * sms_utility
        - 1.00 * sms_income
        - 0.72 * telco_stability
        - 0.68 * consent_completeness
        - 0.45 * network_mobility
        + rng.normal(0, 0.45, n)
    )
    pd_90d = sigmoid(risk_logit)
    target = rng.binomial(1, pd_90d)

    device_tier = pd.cut(
        device_hardware,
        bins=[-0.01, 0.32, 0.52, 0.74, 1.01],
        labels=DEVICE_TIERS,
    ).astype(str)

    df = pd.DataFrame(
        {
            "application_id": [f"IND-C-{i:06d}" for i in range(n)],
            "model_type": "retail_cold_start",
            **_individual_audit_columns(rng, n),
            "device_tier": device_tier,
            "device_hardware_score": device_hardware,
            "device_integrity_score": device_integrity,
            "utility_finance_app_ratio": utility_finance_ratio,
            "high_risk_app_ratio": high_risk_app_ratio,
            "onboarding_hesitation_score": hesitation,
            "typing_cadence_stability": typing_stability,
            "copy_paste_flag": copy_paste,
            "sms_utility_payment_regularity": sms_utility,
            "sms_income_signal_strength": sms_income,
            "telco_network_stability": telco_stability,
            "consent_completeness": consent_completeness,
            "network_mobility_consistency": network_mobility,
            "simulated_pd_90d": pd_90d,
            "default_90d": target,
        }
    )
    return _rebalance_audit_columns(
        df,
        {
            "protected_gender": GENDERS,
            "age_band": AGE_BANDS,
            "state_group": STATE_GROUPS,
            "urbanicity": URBANICITY,
        },
        rng,
    )


def generate_msme_active(n: int, rng: np.random.Generator) -> pd.DataFrame:
    management = rng.normal(0.0, 1.0, n)
    liquidity = rng.normal(0.0, 1.0, n)
    market = rng.normal(0.0, 1.0, n)
    shock = rng.normal(0.0, 1.0, n)

    audit = _msme_audit_columns(rng, n)
    sectors = balanced_choice(rng, SECTORS, n)
    capital_tier = balanced_choice(rng, CAPITAL_TIERS, n)
    business_location_tier = balanced_choice(rng, LOCATION_TIERS, n)
    sector_risk = np.array([SECTOR_RISK[v] for v in sectors])
    capital_risk = np.array([CAPITAL_RISK[v] for v in capital_tier])
    location_risk = np.array([LOCATION_RISK[v] for v in business_location_tier])

    monthly_revenue = np.exp(12.2 + 0.55 * liquidity - 0.10 * capital_risk + rng.normal(0, 0.24, n))
    monthly_revenue = np.clip(monthly_revenue, 45000, 9500000)
    revenue_growth = clipped(0.50 + 0.18 * market + 0.10 * management - 0.08 * shock + rng.normal(0, 0.10, n), -0.20, 1.0)
    gross_margin = clipped(0.24 + 0.08 * management - 0.04 * sector_risk + rng.normal(0, 0.05, n), 0.03, 0.70)
    dscr = clipped(0.55 + 0.16 * liquidity + 0.08 * management - 0.10 * shock + rng.normal(0, 0.12, n))
    gstr_delay = np.maximum(0, rng.poisson(clipped(2.5 + 3.2 * shock - 2.0 * management + 1.5 * sector_risk, 0.2, 18.0))).astype(float)
    gst_consistency = clipped(0.74 + 0.18 * management - 0.10 * shock + rng.normal(0, 0.10, n))
    itc_ratio = clipped(0.78 + 0.14 * management + 0.05 * liquidity - 0.08 * shock + rng.normal(0, 0.09, n))
    upi_variance = clipped(0.30 + 0.18 * shock - 0.12 * liquidity + rng.normal(0, 0.10, n))
    bounced_mandates = rng.poisson(clipped(0.25 + 1.20 * (1 - dscr) + 0.50 * shock, 0.02, 4.5)).astype(float)
    supplier_concentration = clipped(0.36 + 0.13 * sector_risk + 0.14 * shock - 0.07 * management + rng.normal(0, 0.10, n))
    wc_gap = clipped(0.36 + 0.20 * shock - 0.14 * liquidity + 0.10 * sector_risk + rng.normal(0, 0.11, n))
    industry_beta = clipped(1.02 + 0.20 * sector_risk + 0.12 * shock + rng.normal(0, 0.10, n), 0.55, 1.85)
    news_sentiment = clipped(0.63 + 0.13 * market - 0.16 * shock - 0.08 * location_risk + rng.normal(0, 0.11, n))
    infra_anomaly = clipped(0.16 + 0.16 * location_risk + 0.13 * shock + rng.normal(0, 0.08, n))
    sector_growth = clipped(0.52 + 0.20 * market - 0.12 * sector_risk + rng.normal(0, 0.10, n), -0.20, 1.0)
    promoter_score = clipped(0.62 + 0.12 * management + 0.08 * liquidity + rng.normal(0, 0.10, n))

    risk_logit = (
        -1.95
        + 0.70 * sector_risk
        + 0.50 * capital_risk
        + 0.40 * location_risk
        - 0.90 * revenue_growth
        - 0.72 * gross_margin
        - 1.10 * dscr
        + 0.07 * gstr_delay
        - 1.05 * gst_consistency
        - 0.92 * itc_ratio
        + 0.95 * upi_variance
        + 0.18 * bounced_mandates
        + 0.82 * supplier_concentration
        + 1.08 * wc_gap
        + 0.45 * industry_beta
        - 0.92 * news_sentiment
        + 0.82 * infra_anomaly
        - 0.76 * sector_growth
        - 0.82 * promoter_score
        + rng.normal(0, 0.42, n)
    )
    pd_90d = sigmoid(risk_logit)
    target = rng.binomial(1, pd_90d)

    df = pd.DataFrame(
        {
            "business_id": [f"BUS-A-{i:06d}" for i in range(n)],
            "model_type": "msme_active",
            **audit,
            "sector": sectors,
            "capital_tier": capital_tier,
            "business_location_tier": business_location_tier,
            "monthly_revenue_inr": monthly_revenue.round(2),
            "revenue_growth_90d": revenue_growth,
            "gross_margin_ratio": gross_margin,
            "debt_service_coverage_proxy": dscr,
            "gstr1_filing_delay_days": gstr_delay,
            "gst_filing_consistency": gst_consistency,
            "itc_reconciliation_ratio": itc_ratio,
            "daily_qr_upi_volume_variance_7d": upi_variance,
            "bounced_mandates_30d": bounced_mandates,
            "supplier_concentration_risk": supplier_concentration,
            "working_capital_gap_ratio": wc_gap,
            "industry_sector_beta": industry_beta,
            "district_news_sentiment_score": news_sentiment,
            "district_infrastructure_anomaly_index": infra_anomaly,
            "sector_growth_index": sector_growth,
            "promoter_alt_score_norm": promoter_score,
            "simulated_pd_90d": pd_90d,
            "default_90d": target,
        }
    )
    return _rebalance_audit_columns(
        df,
        {
            "promoter_gender": GENDERS,
            "promoter_age_band": AGE_BANDS,
            "state_group": STATE_GROUPS,
            "location_tier": LOCATION_TIERS,
        },
        rng,
    )


def generate_msme_cold_start(n: int, rng: np.random.Generator) -> pd.DataFrame:
    promoter = rng.normal(0.0, 1.0, n)
    preparedness = rng.normal(0.0, 1.0, n)
    ecosystem = rng.normal(0.0, 1.0, n)
    funding_gap = rng.normal(0.0, 1.0, n)

    audit = _msme_audit_columns(rng, n)
    sectors = balanced_choice(rng, SECTORS, n)
    capital_tier = balanced_choice(rng, CAPITAL_TIERS, n)
    business_location_tier = balanced_choice(rng, LOCATION_TIERS, n)
    sector_risk = np.array([SECTOR_RISK[v] for v in sectors])
    capital_risk = np.array([CAPITAL_RISK[v] for v in capital_tier])
    location_risk = np.array([LOCATION_RISK[v] for v in business_location_tier])

    promoter_score = clipped(0.58 + 0.20 * promoter + 0.07 * preparedness + rng.normal(0, 0.10, n))
    recovery = clipped(0.60 + 0.18 * promoter + rng.normal(0, 0.11, n))
    sector_survival = clipped(0.64 + 0.16 * ecosystem - 0.10 * sector_risk + rng.normal(0, 0.10, n))
    pincode_survival = clipped(0.62 + 0.14 * ecosystem - 0.10 * location_risk + rng.normal(0, 0.11, n))
    capital_sufficiency = clipped(0.50 + 0.18 * preparedness - 0.14 * funding_gap - 0.10 * capital_risk + rng.normal(0, 0.11, n))
    doc_consistency = clipped(0.70 + 0.16 * preparedness + 0.06 * promoter + rng.normal(0, 0.10, n))
    incorporation_age = np.maximum(1, rng.poisson(clipped(9.0 + 5.0 * preparedness + 2.0 * ecosystem, 1.0, 36.0))).astype(float)
    licensing = clipped(0.68 + 0.17 * preparedness + rng.normal(0, 0.11, n))
    peer_default = clipped(0.18 + 0.10 * sector_risk + 0.11 * location_risk - 0.08 * ecosystem + rng.normal(0, 0.05, n), 0.02, 0.55)
    industry_beta = clipped(1.02 + 0.20 * sector_risk + 0.06 * location_risk + rng.normal(0, 0.10, n), 0.55, 1.85)
    expense_coverage = clipped(0.48 + 0.20 * preparedness - 0.18 * funding_gap + rng.normal(0, 0.12, n))

    risk_logit = (
        -0.55
        + 0.74 * sector_risk
        + 0.52 * capital_risk
        + 0.46 * location_risk
        - 1.10 * promoter_score
        - 0.80 * recovery
        - 0.92 * sector_survival
        - 0.88 * pincode_survival
        - 1.02 * capital_sufficiency
        - 0.92 * doc_consistency
        - 0.025 * incorporation_age
        - 0.82 * licensing
        + 2.10 * peer_default
        + 0.42 * industry_beta
        - 1.00 * expense_coverage
        + rng.normal(0, 0.44, n)
    )
    pd_90d = sigmoid(risk_logit)
    target = rng.binomial(1, pd_90d)

    df = pd.DataFrame(
        {
            "business_id": [f"BUS-C-{i:06d}" for i in range(n)],
            "model_type": "msme_cold_start",
            **audit,
            "sector": sectors,
            "capital_tier": capital_tier,
            "business_location_tier": business_location_tier,
            "promoter_alt_score_norm": promoter_score,
            "promoter_prior_recovery_ratio": recovery,
            "sector_survival_rate": sector_survival,
            "pincode_cluster_survival_rate": pincode_survival,
            "capital_sufficiency_ratio": capital_sufficiency,
            "registration_document_consistency": doc_consistency,
            "incorporation_age_months": incorporation_age,
            "licensing_completeness": licensing,
            "peer_default_rate": peer_default,
            "industry_sector_beta": industry_beta,
            "startup_expense_coverage_ratio": expense_coverage,
            "simulated_pd_90d": pd_90d,
            "default_90d": target,
        }
    )
    return _rebalance_audit_columns(
        df,
        {
            "promoter_gender": GENDERS,
            "promoter_age_band": AGE_BANDS,
            "state_group": STATE_GROUPS,
            "location_tier": LOCATION_TIERS,
        },
        rng,
    )


GENERATORS = {
    "retail_active": generate_retail_active,
    "retail_cold_start": generate_retail_cold_start,
    "msme_active": generate_msme_active,
    "msme_cold_start": generate_msme_cold_start,
}


def generate_all_datasets(
    rows_per_model: int = DEFAULT_ROWS_PER_MODEL,
    seed: int = RANDOM_SEED,
    out_dir: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    datasets = {
        model_type: GENERATORS[model_type](rows_per_model, rng)
        for model_type in MODEL_TYPES
    }

    if out_dir is not None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        for model_type, df in datasets.items():
            df.to_csv(out_path / f"{model_type}.csv", index=False)

    return datasets


def load_generated_datasets(data_dir: str | Path) -> dict[str, pd.DataFrame]:
    data_path = Path(data_dir)
    return {
        model_type: pd.read_csv(data_path / f"{model_type}.csv")
        for model_type in MODEL_TYPES
    }
