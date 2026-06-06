# Data Dictionary

## Common Outputs

- `simulated_pd_90d`: synthetic ground-truth default probability used to draw the target.
- `default_90d`: simulated 90-day default outcome.
- `predicted_pd_90d`: model-estimated default probability on holdout data.
- `alt_score`: 300 to 900 score derived as `300 + 600 * (1 - PD)`.
- `approved`: whether the tuned operating point approves the account.

## Retail Active Features

- `monthly_inflow_inr`: total monthly money inflow.
- `inflow_stability_90d`: stability of inflows over 90 days.
- `upi_debit_velocity_variance`: volatility of UPI debit behavior.
- `fixed_obligation_ratio`: fixed obligations divided by total inflow.
- `discretionary_spend_ratio`: discretionary spend share.
- `essential_payment_timeliness`: utility/rent/essential payment regularity.
- `routine_entropy`: geolocation routine entropy, higher means less predictable.
- `commute_consistency`: home-work movement consistency.
- `telco_recharge_regularity`: regularity of telco recharges.
- `prior_loan_recovery_ratio`: previous recovery behavior where available.
- `network_working_ratio`: share of network activity during working contexts.
- `bot_financial_stress_index`: NLP-derived stress proxy.
- `bot_repayment_intent_score`: NLP-derived repayment intent proxy.
- `financial_literacy_score`: blog/chat literacy proxy.
- `blog_completion_rate`: completion of financial literacy content.

## Retail Cold-Start Features

- `device_hardware_score`: device capability proxy.
- `device_integrity_score`: device consistency and low-risk state proxy.
- `utility_finance_app_ratio`: useful financial/utility app density.
- `high_risk_app_ratio`: speculative or risky app density.
- `onboarding_hesitation_score`: friction in application flow.
- `typing_cadence_stability`: stable typing cadence during onboarding.
- `copy_paste_flag`: copy-paste behavior in key fields.
- `sms_utility_payment_regularity`: utility payment signal from metadata.
- `sms_income_signal_strength`: salary/remittance signal from metadata.
- `telco_network_stability`: cellular/network stability.
- `consent_completeness`: completeness of consented data bundle.
- `network_mobility_consistency`: stable economic mobility pattern.

## MSME Active Features

- `sector`: business sector bucket.
- `capital_tier`: capital scale bucket.
- `monthly_revenue_inr`: current monthly revenue.
- `revenue_growth_90d`: recent revenue growth.
- `gross_margin_ratio`: gross margin proxy.
- `debt_service_coverage_proxy`: ability to service debt.
- `gstr1_filing_delay_days`: delay in GST filing.
- `gst_filing_consistency`: consistency of GST compliance.
- `itc_reconciliation_ratio`: input tax credit reconciliation health.
- `daily_qr_upi_volume_variance_7d`: QR/UPI inflow volatility.
- `bounced_mandates_30d`: failed mandate count.
- `supplier_concentration_risk`: dependency on few suppliers.
- `working_capital_gap_ratio`: working capital gap proxy.
- `industry_sector_beta`: sector macro volatility.
- `district_news_sentiment_score`: localized news sentiment.
- `district_infrastructure_anomaly_index`: local infrastructure shock proxy.
- `sector_growth_index`: sector growth health.
- `promoter_alt_score_norm`: normalized promoter alternative score.

## MSME Cold-Start Features

- `promoter_alt_score_norm`: normalized promoter score.
- `promoter_prior_recovery_ratio`: promoter repayment/recovery history proxy.
- `sector_survival_rate`: sector survival baseline.
- `pincode_cluster_survival_rate`: location cluster survival baseline.
- `capital_sufficiency_ratio`: setup capital versus sector need.
- `registration_document_consistency`: consistency of registration docs.
- `incorporation_age_months`: business age.
- `licensing_completeness`: licensing completeness.
- `peer_default_rate`: peer-cluster default rate.
- `industry_sector_beta`: sector macro volatility.
- `startup_expense_coverage_ratio`: capital coverage of early expenses.

