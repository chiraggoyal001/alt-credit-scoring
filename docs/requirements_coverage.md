# Requirements Coverage

The MVP now covers the original brief in simulation form.

## Done

- Four-model architecture: retail active, retail cold-start, MSME active, MSME cold-start.
- Unbiased synthetic datasets with protected fields separated from model features.
- MSME categorization by sector, capital tier, and business location tier.
- Retail signals for UPI velocity, telco regularity, repayment recovery, network working ratio, routine entropy, bot stress, repayment intent, spend behavior, blog completion, and financial literacy.
- MSME signals for GST consistency, filing delay, ITC, revenue, gross margin, debt-service coverage, QR/UPI volatility, bounced mandates, supplier concentration, working capital gap, district news, infrastructure anomaly, sector growth, sector beta, promoter score, peer default, registration, licensing, and survival rates.
- Ensemble blend of gradient boosting, random forest, calibrated SVM, and shallow decision tree.
- 300-900 score, PD, dynamic exposure limit, APR, reason codes, and CIBIL-like benchmark comparison.
- JSON templates for single-score testing and replacement testing-data evaluation.
- Stage-by-stage calculation trace in the app and CLI.
- Fairness reports and 80 percent rule checks.

## Production Caveats

- The CIBIL comparison is a transparent synthetic benchmark, not the real proprietary CIBIL model.
- NLP, Account Aggregator, telco, UPI, GST, and news ingestion are represented by engineered simulation vectors.
- SHAP-style explanations are represented by reason codes and diagnostics in this MVP; production should add model-native SHAP or equivalent explainability.
- Survival/hazard modeling is represented by PD and operating-point simulation; production should add explicit time-to-default modeling when real repayment histories are available.

## Remaining Questions

- Which consent rail is primary: Account Aggregator, SDK permissions, lender-owned data, or a blended consent flow?
- Which default target should be optimized first: 15 DPD, 30 DPD, or 90 DPD?
- Should cold-start approvals always use step-up credit limits?
- Which alternative signals are legally disallowed or must be constrained for the lending jurisdiction?
- Which MSME product is first: merchant cash advance, working capital line, invoice discounting, or term loan?
- What borrower-facing adverse-action format is required?

