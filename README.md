# Alternative Credit Model Simulation MVP

This project is a runnable MVP for simulating alternative credit scoring for individuals and MSMEs. It creates synthetic, fairness-auditable data, trains four model pipelines, scores applicants, explains each scoring stage, compares against a transparent synthetic CIBIL-like benchmark, and exposes both a Streamlit workbench and FastAPI endpoints.

The project is a simulation and research scaffold. It does not use real CIBIL internals, real Account Aggregator data, real UPI data, real telco data, or production GST/news feeds. Those are represented by engineered synthetic feature vectors.

## Current Status

The MVP currently supports:

- Retail active scoring for individuals with a deeper behavioral and cash-flow footprint.
- Retail cold-start scoring for thin-file individuals.
- MSME active scoring for operating businesses.
- MSME cold-start scoring for new or informal businesses entering formal credit.
- Synthetic data generation for all four models.
- Model training with a soft-voting ensemble.
- Score testing through manual controls and JSON upload.
- Replacement testing-data upload and evaluation.
- Stage-by-stage scoring trace.
- Fairness and disparate-impact reporting.
- AltCredit versus synthetic CIBIL-like comparison.
- Coverage audit for the original problem statement and pasted text.

Latest generated simulation size:

- `10,000` records per model
- `40,000` total synthetic records

Latest summary from `artifacts/reports/summary.json`:

| Model | Alt approval | Alt approved default | CIBIL-like approval | CIBIL-like approved default | Gini |
|---|---:|---:|---:|---:|---:|
| Retail Active | 81.96% | 3.66% | 46.00% | 7.83% | 0.750 |
| Retail Cold-Start | 67.84% | 2.54% | 27.00% | 5.78% | 0.611 |
| MSME Active | 77.88% | 1.80% | 42.00% | 5.05% | 0.736 |
| MSME Cold-Start | 60.88% | 1.51% | 25.00% | 1.60% | 0.367 |

## Project Structure

```text
alt_credit_model_simulation/
  altcredit/
    backtest.py              Synthetic CIBIL-like benchmark and operating-point tuning
    cli.py                   Command-line entry points
    comparison.py            AltCredit versus CIBIL-like evaluation metrics
    config.py                Global model labels, limits, and constants
    coverage.py              Original-brief and pasted-text coverage audit
    data.py                  Synthetic data generator
    fairness.py              Disparate-impact and label-balance reports
    feature_engineering.py   Reusable feature-engineering helpers
    features.py              Model schemas and feature lists
    models.py                Training pipelines and model artifacts
    scoring.py               Score, trace, reason codes, pricing, exposure
    templates.py             JSON score and testing-data templates
  app.py                     Streamlit workbench
  api.py                     FastAPI service
  data/
    synthetic/               Generated synthetic datasets
    testing/                 Uploaded replacement testing data
  artifacts/
    models/                  Trained joblib model bundles
    reports/                 Metrics, fairness reports, scored holdouts
  docs/
    data_dictionary.md
    problem_structure.md
    requirements_coverage.md
  tests/
    test_mvp.py
  requirements.txt
```

## Installation

Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

If your Anaconda environment has SSL issues, use:

```powershell
python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

The project uses:

- Python 3.9+
- pandas
- numpy
- scikit-learn
- joblib
- Streamlit
- FastAPI
- Uvicorn

## Quick Start

Generate synthetic data, train all four models, and write reports:

```powershell
python -m altcredit.cli run-all --rows 10000
```

Run a faster smoke-test training pass:

```powershell
python -m altcredit.cli run-all --rows 1000 --fast
```

Run tests:

```powershell
python -m unittest discover -s tests
```

Run the Streamlit workbench:

```powershell
streamlit run app.py
```

Run the API:

```powershell
uvicorn api:app --reload --port 8000
```

## Streamlit Workbench

The app has seven views.

### Overview

Shows portfolio-level metrics across all four models:

- Approval rate comparison.
- Approved default rate comparison.
- Average approval lift.
- Average default-rate delta.
- Average Gini.
- Selected-model score snapshot.

The Overview page avoids Streamlit Altair charts and uses native HTML/dataframe visuals, so it does not depend on Altair or MarkupSafe.

### Score Lab

Lets you manually test an applicant or business. Inputs are grouped by underwriting concept:

- Cash Flow And Spend
- Routine And Telco
- Intent And Literacy
- Onboarding And Device
- 3D Business Category
- GST, P&L, Cash Flow
- Macro And Ecosystem
- Promoter And Documentation

It returns:

- Alt score
- Probability of default over 90 days
- Approval/manual review/decline decision
- Dynamic exposure limit
- Risk-based APR
- Reason codes
- CIBIL-like benchmark decision

You can also upload JSON for single-score testing.

### Calculation Trace

Shows the full scoring path:

1. Model route
2. Input validation and missing-field fill strategy
3. Feature diagnostics against training quantiles
4. Base learner probabilities
5. Final PD and 300-900 score scaling
6. Operating-point thresholding
7. Exposure and pricing
8. CIBIL-like benchmark comparison

Partial JSON uploads are allowed for score testing. Missing fields are filled with training medians or known categories, and the trace shows which fields were filled.

### Testing Data

Lets you upload a JSON file that replaces the testing data for a selected model. The uploaded file is saved to:

```text
data/testing/{model_type}_replacement.json
```

The app then scores it and writes:

```text
artifacts/reports/{model_type}_uploaded_comparison.json
artifacts/reports/{model_type}_uploaded_scored_testing_data.csv
```

Uploaded testing data must include:

- all feature columns for that model
- `default_90d`

### Model Comparison

Compares AltCredit and the synthetic CIBIL-like benchmark across:

- approval rate
- decline rate
- default rate among approved
- default rate among declined
- good borrowers rejected
- bad borrowers approved
- ROC AUC
- Gini
- average precision
- Brier loss
- default precision
- default recall
- accuracy
- average score
- median score
- score threshold
- risk cutoff
- average approved limit
- total approved exposure
- expected-loss proxy
- lift and delta values

### JSON Templates

Provides downloadable templates for:

- single-score JSON
- replacement testing-data JSON

### Coverage Audit

Checks whether the original problem statement and pasted text are covered. It also lists what remains as a production integration or governance question.

## CLI Commands

Generate data only:

```powershell
python -m altcredit.cli generate --rows 10000
```

Train from existing data:

```powershell
python -m altcredit.cli train --data data/synthetic
```

Generate and train:

```powershell
python -m altcredit.cli run-all --rows 10000
```

Score a median synthetic applicant:

```powershell
python -m altcredit.cli score --model-type retail_active
```

Show complete scoring trace:

```powershell
python -m altcredit.cli score --model-type retail_active --trace
```

Print a score JSON template:

```powershell
python -m altcredit.cli template --model-type retail_active --kind score
```

Print a testing-data JSON template:

```powershell
python -m altcredit.cli template --model-type retail_active --kind testing-data
```

Evaluate a replacement testing-data JSON file:

```powershell
python -m altcredit.cli evaluate-json --file .\my_testing_data.json --model-type retail_active
```

## API

Start the API:

```powershell
uvicorn api:app --reload --port 8000
```

Health:

```http
GET /health
```

Models:

```http
GET /models
```

Score template:

```http
GET /templates/score/{model_type}
```

Testing-data template:

```http
GET /templates/testing-data/{model_type}
```

Score:

```http
POST /score
```

Body:

```json
{
  "model_type": "retail_active",
  "features": {
    "monthly_inflow_inr": 30000,
    "inflow_stability_90d": 0.62
  }
}
```

Trace:

```http
POST /score/trace
```

Evaluate uploaded testing data:

```http
POST /evaluate/testing-data
```

Body:

```json
{
  "model_type": "retail_active",
  "records": [
    {
      "monthly_inflow_inr": 30000,
      "inflow_stability_90d": 0.62,
      "upi_debit_velocity_variance": 0.20,
      "fixed_obligation_ratio": 0.25,
      "discretionary_spend_ratio": 0.30,
      "essential_payment_timeliness": 0.80,
      "routine_entropy": 0.25,
      "commute_consistency": 0.75,
      "telco_recharge_regularity": 0.78,
      "prior_loan_recovery_ratio": 0.82,
      "network_working_ratio": 0.76,
      "bot_financial_stress_index": 0.20,
      "bot_repayment_intent_score": 0.80,
      "financial_literacy_score": 0.70,
      "blog_completion_rate": 0.60,
      "default_90d": 0
    }
  ]
}
```

## Model Architecture

Each model uses a soft-voting ensemble:

- Gradient Boosting
- Random Forest
- Calibrated Linear SVM
- Shallow Decision Tree

The blend is intentionally pragmatic:

- Gradient Boosting captures strong non-linear tabular patterns.
- Random Forest improves robustness under noisy alternative data.
- SVM contributes margin-based separation after preprocessing.
- Shallow Decision Tree contributes interpretable rule-like behavior.

The score is derived from predicted default probability:

```text
score = 300 + 600 * (1 - probability_default_90d)
```

The underwriting decision uses a tuned operating point selected on holdout data.

## Models

### Model 1: Retail Active

For individuals with deeper behavioral and financial footprints.

Feature themes:

- UPI debit velocity
- monthly inflow
- fixed obligations
- discretionary spend
- essential payment timeliness
- geolocation routine entropy
- commute consistency
- telco regularity
- prior loan recovery
- network working ratio
- bot stress
- repayment intent
- financial literacy
- blog completion

### Model 2: Retail Cold-Start

For thin-file individuals.

Feature themes:

- device hardware score
- device integrity
- useful app ratio
- high-risk app ratio
- onboarding hesitation
- typing cadence stability
- copy-paste flag
- SMS utility payment regularity
- SMS income signal
- telco network stability
- consent completeness
- network mobility consistency

### Model 3: MSME Active

For operating businesses.

Feature themes:

- sector
- capital tier
- business location tier
- revenue
- revenue growth
- gross margin
- debt-service coverage
- GST filing delay
- GST consistency
- ITC reconciliation
- QR/UPI volatility
- bounced mandates
- supplier concentration
- working capital gap
- industry beta
- district news sentiment
- infrastructure anomaly
- sector growth
- promoter score

### Model 4: MSME Cold-Start

For newly registered or informal businesses.

Feature themes:

- sector
- capital tier
- business location tier
- promoter score
- prior recovery proxy
- sector survival rate
- pincode cluster survival rate
- capital sufficiency
- document consistency
- incorporation age
- licensing completeness
- peer default rate
- industry beta
- startup expense coverage

## Synthetic Data

The generator creates synthetic records for all four models. It includes:

- model features
- audit fields
- simulated default probabilities
- `default_90d` target labels

Protected or sensitive audit fields are used for reporting, not for ordinary individual model features.

Examples:

- gender
- age band
- state group
- urbanicity
- promoter gender
- promoter age band
- audit location tier

Legitimate business underwriting categories are modeled:

- sector
- capital tier
- business location tier

## Fairness Reports

Fairness reports are written to:

```text
artifacts/reports/*_fairness.json
```

They include:

- selection rates by audit group
- observed default rates by group
- disparate impact ratio
- 80 percent rule pass/fail flag

Latest generated reports pass the 80 percent rule for all four models.

## CIBIL-Like Benchmark

The comparison benchmark is not the real CIBIL model. It is a synthetic legacy-bureau proxy that intentionally uses fewer, lagging variables.

Reason:

- Real CIBIL scoring logic is proprietary.
- Publicly available inputs are not enough to reproduce it.
- A transparent benchmark is better for testing than pretending to know the real model.

The benchmark is implemented in:

```text
altcredit/backtest.py
```

## Output Files

After `run-all`, expect:

```text
data/synthetic/retail_active.csv
data/synthetic/retail_cold_start.csv
data/synthetic/msme_active.csv
data/synthetic/msme_cold_start.csv

artifacts/models/retail_active.joblib
artifacts/models/retail_cold_start.joblib
artifacts/models/msme_active.joblib
artifacts/models/msme_cold_start.joblib

artifacts/reports/summary.json
artifacts/reports/*_metrics.json
artifacts/reports/*_fairness.json
artifacts/reports/*_label_balance.json
artifacts/reports/*_scored_holdout.csv
```

## Validation Checklist

Run:

```powershell
python -m compileall altcredit api.py app.py
python -m unittest discover -s tests
python -m altcredit.cli score --model-type retail_active --trace
python -m altcredit.cli template --model-type retail_active --kind testing-data
```

Start the app and API:

```powershell
streamlit run app.py
uvicorn api:app --reload --port 8000
```

Expected:

- Overview page loads without Altair or MarkupSafe errors.
- Model Comparison page loads without Altair or MarkupSafe errors.
- Score Lab can score manual inputs.
- Score JSON upload can handle partial feature JSON.
- Calculation Trace shows missing-field fill details.
- Testing Data upload requires complete feature records plus `default_90d`.
- Coverage Audit shows pasted-text and original-brief coverage.

## Troubleshooting

### ImportError: cannot import name `soft_unicode` from `markupsafe`

This happened when Streamlit tried to use Altair through `st.bar_chart`. The app no longer uses `st.bar_chart`, so Overview and Model Comparison should not hit this path.

If you still see it, restart Streamlit:

```powershell
Get-Process | Where-Object { $_.ProcessName -like '*python*' }
streamlit run app.py
```

### FastAPI typing_extensions error

Install a newer typing extensions package:

```powershell
python -m pip install --upgrade "typing_extensions>=4.8"
```

### Broken NumPy install

If NumPy fails with missing internal modules, reinstall:

```powershell
python -m pip install --force-reinstall --no-cache-dir "numpy==1.23.5"
```

### SSL errors during pip install

Use trusted hosts:

```powershell
python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

## Production Caveats

Before production, this MVP needs:

- legal consent design for UPI, telco, SDK, bot chat, blog, and GST data
- Account Aggregator integration
- UPI and telco integrations
- GST and P&L ingestion
- real news and district-level event ingestion
- explicit 15 DPD, 30 DPD, and 90 DPD target modeling
- production SHAP or equivalent explainability
- adverse-action notice templates
- monitoring for drift and fairness
- human review workflow
- security hardening
- model governance and audit logs

## Current Running URLs

When started locally:

```text
Streamlit: http://localhost:8501
FastAPI:   http://127.0.0.1:8000
```

