# Evaluation, Data, and Governance Notes

This note distinguishes what the current simulation computes from what would be required to make claims about a real underwriting product.

## 1. Data and target construction

The project generates four synthetic datasets: retail active, retail cold-start, MSME active, and MSME cold-start. Features are produced from simulated latent factors and hand-authored relationships. The generator derives a synthetic 90-day default probability (`simulated_pd_90d`) and samples `default_90d` from that probability.

This provides a controlled way to exercise training, scoring, thresholding, reporting, and UI flows. It also means the model is learning patterns from relationships designed by the project—not from observed real-world repayment histories.

The project separates model features from protected/audit columns in the feature specifications. Audit columns are then used in group-level reporting. In the synthetic generator, audit groups are deliberately spread across simulated risk deciles, so the generated data does not recreate the complex correlations and structural inequalities that may appear in a real applicant population.

## 2. Model and score

Each route uses a preprocessed soft-voting ensemble comprising:

- Gradient Boosting
- Random Forest
- Calibrated Linear SVM
- Shallow Decision Tree

Numeric fields are median-imputed and scaled. Categorical fields are filled with a most-frequent category and one-hot encoded where the route uses categorical features.

The output score is calculated from model-estimated default probability:

`score = 300 + 600 × (1 − predicted_PD_90d)`

The number is an illustrative score scale. It should not be described as calibrated to a bureau score, directly comparable to a CIBIL score, or validated for use in lending. PD calibration needs its own assessment before thresholds, limits, or pricing are used for a real product.

The application also returns exposure-limit and APR estimates from explicit simulation rules. They are product-policy demonstrations, not expected-loss pricing or recommendations for live credit decisions.

## 3. Threshold selection: current limitation

The current training workflow makes a stratified train/test split, trains on the training portion, predicts the held-out portion, tunes the operating threshold on those held-out labels, and then reports performance on that same portion.

That means the reported test metrics are not from a completely untouched final holdout. Using the same partition both to select a threshold and report performance can make the displayed operating-point results optimistic.

**Recommended correction:** split into training, validation, and final test data—or use nested cross-validation—so model fitting and threshold selection happen without using the final evaluation outcomes. Select the decision policy on validation data, freeze it, and only then evaluate on the untouched test set. Preserve a further out-of-time sample when temporal repayment data becomes available.

## 4. Synthetic baseline vs. CIBIL

The comparator is intentionally named a “CIBIL-like” or synthetic legacy-bureau proxy in the code and UI. It is a transparent formula with fewer inputs and configured approval-rate assumptions. It does not access, reproduce, or validate against the proprietary TransUnion CIBIL score model.

Therefore:

- Do not call it a direct CIBIL comparison.
- Do not report simulated approval lift as observed borrower inclusion.
- Do not claim lower real default rates or better creditworthiness prediction.
- Do not infer that CIBIL's actual methodology uses the variables in this comparator.

A credible real-world comparison would require a properly governed dataset with observed repayment outcomes, a defined population, consistent time windows and outcome labels, an agreed incumbent baseline, and pre-specified evaluation rules. Access to any bureau or lender data would require proper authorisation and compliance review.

## 5. Metrics and interpretation

| Metric | What it can help answer | Key caution |
|---|---|---|
| Approval / decline rate | How much synthetic population passes a chosen policy | Strongly depends on threshold and benchmark design |
| Default rate among approved | Observed outcome rate among accepted synthetic cases | Can be unstable for small groups and depends on label construction |
| Good-borrower rejection / bad-borrower approval | Where threshold decisions disagree with generated labels | “Good” and “bad” are defined by synthetic outcomes, not real repayment history |
| ROC AUC / Gini | Ranking discrimination | Does not establish calibration or economic value |
| Average precision | Precision/recall performance for the default class | Depends on default prevalence in the generated data |
| Brier loss | Squared probability error | Must be compared with sensible baselines and calibration plots |
| Accuracy | Overall share of labels matched | Can obscure class imbalance and unequal error costs |
| Exposure / expected-loss proxy | Illustrative book-level implications | The current formulas are proxies, not a validated loss model |
| Disparate-impact ratio | Difference in selection rates across selected groups | A screening diagnostic, not proof of fairness or legal compliance |

Where appropriate for a real pilot, add calibration curves, precision-recall curves, confidence intervals, subgroup error rates, intersectional slices, stability tests, and performance by time vintage.

## 6. Fairness and the 80% rule

The code reports selection rates by audit group and computes a disparate-impact ratio using the lowest group selection rate divided by the highest. It also flags whether the ratio is at least 0.80.

This check is only a simple diagnostic. Passing the 80% rule in a synthetic sample does not prove fairness, legal compliance, or equal customer outcomes. The current audit groups are deliberately rebalanced across synthetic risk deciles, so the report is especially unsuitable as a real-world fairness claim.

For a more serious assessment, define the groups and metrics with legal, risk, and domain experts; measure subgroup false positives/false negatives and calibration where appropriate; inspect intersectional groups; account for sample size and uncertainty; test proxies and indirect effects; and evaluate adverse impacts throughout the lifecycle. Do not optimise a single parity statistic without examining risk and borrower outcomes.

## 7. Alternative-data risk register

The feature schemas include illustrative variables relating to cash flows and business operations, alongside more intrusive or indirect proxies. The following are high scrutiny and should not move into a real product by default.

| Signal family | Main concern | Minimum follow-up before any live use |
|---|---|---|
| Cash-flow and payment regularity | Incomplete or noisy picture of ability to repay; account coverage may be uneven | Validate source quality, consent, missingness, and incremental value over simpler inputs |
| GST and business financial signals | Filing patterns can reflect business size, sector, or process capacity | Confirm applicability to the target segment; validate against real repayment outcomes |
| Device/app characteristics | May proxy for income, age, geography, accessibility, or device affordability | Demonstrate necessity and incremental value; prefer not to use without compelling evidence and legal review |
| Typing cadence, copy/paste, onboarding hesitation | Can reflect disability, language, device, literacy, or accessibility rather than credit risk | Avoid inference without strong justification, consent, validation, and a non-digital alternative |
| Geolocation / movement patterns | Highly sensitive location and lifestyle information | Default to exclusion unless lawful, necessary, proportionate, and explicitly reviewed |
| SMS/telco metadata | Sensitive communications context and uneven data availability | Data-minimised source design, meaningful consent, permitted-purpose review, and deletion controls |
| NLP-inferred stress / intent or financial-literacy engagement | Unvalidated psychological or engagement proxies; can punish people for not using educational content | Exclude from decisioning until independently validated and ethically reviewed; do not treat content engagement as repayment capacity |
| Sector, region, peer, or news context | Can create geographic/sector stereotyping or shift risk due to external shocks | Test for proxy effects and stability; consider human review or temporary context adjustments rather than automatic adverse decisions |

**Design principle:** availability of a data point is not a reason to collect it, and predictive correlation is not sufficient evidence that a signal is appropriate for a lending decision.

## 8. Explainability and recourse

The app contains a scoring trace, feature diagnostics relative to training quantiles, base learner probability outputs, and heuristic reason codes. These are helpful debugging and product-inspection affordances; they are not SHAP explanations or validated adverse-action reason codes.

For a real system, the borrower-facing explanation would need to be faithful to the model and decision policy, tested for comprehension, and aligned to applicable requirements. Applicants should have a path to correct data, ask for review, and appeal errors. Human review needs an operational policy and audit trail.

## 9. Minimum evidence gate before a pilot

Do not advance to real applicant decisioning until the team can document at least:

- A specific lending use case, target population, and incumbent baseline.
- Lawful source access, data minimisation, consent or other applicable basis, retention and deletion rules.
- Feature definitions, provenance, missingness, and quality checks.
- Leakage review and a frozen evaluation design with an untouched final holdout.
- Calibration and decision-policy validation at pre-agreed risk/access constraints.
- Subgroup and proxy-risk assessment with uncertainty analysis.
- Validated explanations, manual review, appeal/correction, and customer support processes.
- Security, access control, logging, monitoring, drift response, model versioning, and rollback plans.
- Risk, legal, compliance, and lender sign-off before use for real credit decisions.

## 10. How to describe results in a portfolio

Use: “The prototype compares an ensemble against a transparent synthetic legacy-bureau proxy on generated data.”

Avoid: “The model outperforms CIBIL,” “the model is unbiased,” “the score is as reliable as a bureau score,” or any unqualified claim about real approval lift or default reduction.
