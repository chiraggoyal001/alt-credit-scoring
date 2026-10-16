# Product Case Study: Alternative Credit Scoring Simulation

**Status:** Working independent prototype using synthetic data  
**Format:** Product concept, interactive workbench, and model-evaluation simulation  
**Demo:** [Live Streamlit app](https://alt-credit-scoring.streamlit.app/)  
**Code:** [GitHub repository](https://github.com/chiraggoyal001/alt-credit-scoring)

## 1. Summary

This project explores how a lender might assess creditworthiness when traditional credit history is limited. Instead of positioning the model as a replacement for a credit bureau, the product question is whether selected, defensible cash-flow and business signals could supplement sparse history—and how that hypothesis should be evaluated without losing sight of credit risk, privacy, and fairness.

The prototype makes the idea tangible through four scoring routes, synthetic training data, an interactive score lab, a calculation trace, a synthetic benchmark, model comparison reports, and audit-group diagnostics.

## 2. Problem framing

A thin file creates an evidence problem: a lender may have less repayment history to work with, while the applicant may still have observable business or payment activity. The product challenge is not simply to approve more applicants. It is to assess whether additional evidence improves the quality of decisions at an acceptable level of risk and with appropriate protections.

### Working hypothesis

> For applicants with limited bureau history, a carefully selected set of additional cash-flow or business signals may help distinguish repayment capacity more effectively than a deliberately sparse baseline—provided the incremental evidence is lawful, consented, reliable, explainable, and tested on real outcomes.

This remains a hypothesis. The current simulation cannot validate it against real credit performance.

## 3. Target contexts

| Route | Context | Core design question |
|---|---|---|
| Retail Active | Individual with more financial/behavioural history | Which signals help assess ability and willingness to repay? |
| Retail Cold-Start | New-to-credit or thin-file individual | What minimally invasive evidence is adequate when bureau history is sparse? |
| MSME Active | Existing business with operating signals | How should cash flow, margin, debt-service capacity, compliance patterns, and shocks be considered together? |
| MSME Cold-Start | New or informal business | How can the system avoid equating a short operating history with an inability to repay? |

The four-route design makes it possible to compare different evidence availability and risk assumptions without pretending that one feature set fits every applicant.

## 4. MVP scope

### Included

- Generate a repeatable synthetic dataset for each of the four routes.
- Train route-specific soft-voting ensembles.
- Estimate a synthetic 90-day default probability and convert it to an illustrative 300–900 score.
- Apply a configurable operating-point strategy and return approve, manual-review, or decline outcomes.
- Show indicative exposure and risk-based APR outputs from explicit simulation rules.
- Explain the scoring pipeline stage by stage and provide heuristic reason codes.
- Compare access, default outcomes, ranking/probability metrics, exposure proxies, and selection rates by audit group.
- Allow a user to score a hypothetical profile and upload replacement labelled testing data.

### Not included

- Real CIBIL score access or reverse-engineering of CIBIL internals.
- Real Account Aggregator, UPI, GST, telco, SMS, device, geolocation, news, or other alternative-data integrations.
- A real credit decision, production credit policy, or legally reviewed data-consent flow.
- A lender pilot, measured repayment outcomes, or evidence of business impact.
- Production-grade authentication, access controls, audit logging, monitoring, or model governance.

## 5. Main user journey

1. **Choose a route** based on the applicant context and data availability.
2. **Inspect the portfolio view** to understand how the simulated policy behaves across routes.
3. **Score a hypothetical applicant** by changing inputs or uploading JSON.
4. **Inspect the trace** to see how inputs are completed, transformed, scored, and turned into a decision.
5. **Compare the policy** against the transparent synthetic baseline using multiple metrics.
6. **Audit outcomes** by generated demographic or contextual groups.
7. **Challenge the assumptions** by uploading a replacement labelled dataset and reviewing what changes.

This journey is designed to make model behaviour inspectable rather than presenting a score as an unexplained answer.

## 6. Product decisions and trade-offs

### Separate active and cold-start paths

**Decision:** route applications into retail/MSME and active/cold-start variants.  
**Reasoning:** evidence availability and underwriting context differ. Separate schemas make those assumptions visible.  
**Trade-off:** multiple routes add complexity, require clear eligibility/routing rules, and can create inconsistent treatment at route boundaries. In a real product, routing should be validated and monitored.

### Evaluate access and risk together

**Decision:** report approval rate alongside approved-borrower default rate, ranking metrics, and portfolio proxies.  
**Reasoning:** approval lift alone can hide adverse selection; accuracy alone can hide who is denied.  
**Trade-off:** no single metric fully captures lender loss, customer outcomes, or access. The final operating point depends on product, segment, loss appetite, and policy constraints.

### Keep a manual-review outcome

**Decision:** include an intermediate outcome in the alternative model's decision flow.  
**Reasoning:** borderline or incomplete cases may need additional evidence or human assessment instead of a forced automated decision.  
**Trade-off:** manual review creates cost, delay, and potential reviewer inconsistency; its policy and service-level targets would need to be designed and measured.

### Make the scoring path visible

**Decision:** provide a calculation trace with feature diagnostics, model-level probabilities, score mapping, thresholding, and downstream rules.  
**Reasoning:** users should be able to understand what the system did and identify where an assumption or input affected a result.  
**Trade-off:** a trace is not the same as a model-faithful explanation. Current reason codes are heuristic; a production system would need validated explainability and borrower-appropriate adverse-action reasons.

### Use generated data for the first iteration

**Decision:** build the workbench on synthetic records instead of real financial records.  
**Reasoning:** it lets the team test product flows and reporting without exposing real borrower data.  
**Trade-off:** synthetic results can only validate software behaviour and internal logic. They cannot establish real predictive performance, fairness, data availability, or customer value.

## 7. How success should be evaluated

The first success criterion is not “beat CIBIL.” It is whether an experiment can produce credible evidence about a narrower question.

**Primary decision metrics**

- Approval rate at a fixed approved-default rate, or approved-default rate at a fixed approval rate.
- Good applicants rejected and bad applicants approved, defined against a reliable outcome label.
- Calibration and ranking quality, including Brier loss, ROC AUC, and average precision.
- Outcomes across meaningful applicant segments and relevant audit groups.
- Coverage and quality of consented input data, missingness, and time-to-decision.

**Operational and customer metrics for a future pilot**

- Manual-review rate and reviewer override rate.
- Application completion and consent withdrawal rates.
- Decision turnaround time and reason-code comprehensibility.
- Delinquency/default outcomes by vintage, amount, product, and segment.
- Complaints, appeals, and correction requests.

Targets should be agreed before a real pilot, with a lender and risk/compliance stakeholders. The current project does not have validated numeric targets or real-world outcome data.

## 8. What the project currently demonstrates

The working prototype demonstrates an implementation of the following product concepts:

- Segment-aware product flows.
- A transparent policy trade-off between access and simulated risk.
- A multi-metric model comparison rather than a single accuracy score.
- A score-to-decision trace that can be inspected during a demo.
- Testing-data upload to challenge the generated scenario.
- Initial group-level selection-rate diagnostics.
- Documentation of known limits and follow-on validation questions.

It does **not** establish that the proposed feature set is appropriate, that the model is fair on real populations, or that the system outperforms CIBIL or any lender's existing underwriting policy.

## 9. Main lessons and open questions

1. **Separate the product thesis from the evidence.** A plausible signal is not automatically predictive, lawful, fair, or useful.
2. **Make evaluation design part of the MVP.** The current threshold is tuned on the same holdout used to report metrics; a separate validation set and untouched final test set are needed.
3. **Treat alternative-data governance as a product requirement.** Consent, purpose limitation, data minimisation, signal necessity, recourse, and retention must be designed, not postponed.
4. **Measure customer outcomes, not just model outcomes.** More approvals are not a benefit if customers receive unsuitable credit, pricing, or limits.
5. **Start narrow.** The first real-world research cycle should identify a lending use case and the least invasive data needed to support it, before adding more signal types.

## 10. Suggested next discovery cycle

- Interview credit-risk/underwriting teams at potential lender partners to understand existing decision rules, thin-file decline reasons, and evidence gaps.
- Interview eligible and declined borrowers, including small business owners, to understand data-sharing comfort, application friction, and what explanations would be useful.
- Map each proposed signal to source, consent, permitted purpose, data quality, proxy risk, retention, and deletion requirements.
- Choose one narrow product and customer segment for a testable pilot hypothesis.
- Agree the evaluation population, baseline, outcomes, risk constraint, fairness slices, and stop conditions before collecting or training on real data.

These are recommendations for future discovery—not research activities claimed as completed.
