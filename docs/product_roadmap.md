# Product Roadmap

This roadmap prioritises trustworthy evidence before expanding the feature set. The stages are recommendations, not commitments or work already completed.

## Guiding principle

Prove that the evaluation is sound and the data is appropriate before adding integrations or presenting the simulation as a better underwriting model.

## Now — strengthen the experiment

### P0. Separate threshold tuning from final evaluation

**Why:** the current workflow tunes the operating threshold and reports metrics on the same held-out partition.

**Work:** introduce training, validation, and final test partitions; select model and decision threshold on training/validation only; freeze the policy; evaluate once on untouched final test data; save the split seed and dataset/model versions with each report.

**Acceptance criteria:**
- Threshold selection does not access final-test labels.
- Model, threshold, and dataset metadata are stored with evaluation outputs.
- A rerun with the same configuration is reproducible.
- Documentation labels previous figures as exploratory if they were produced under the old approach.

### P0. Make the evaluation report self-explanatory

**Work:** add metric definitions, class prevalence, sample sizes, uncertainty where feasible, calibration plots, precision-recall curves, and a clear explanation of the benchmark assumptions.

**Acceptance criteria:**
- A reviewer can identify population, sample size, label definition, threshold, and benchmark from the report alone.
- Risk and access are displayed together.
- The UI does not describe the synthetic comparator as actual CIBIL.

### P0. Add a signal inventory and governance gate

**Work:** record every feature's source assumption, rationale, sensitivity, data quality risks, potential proxies, user consent needs, retention, and whether it is included in the simulated policy.

**Acceptance criteria:**
- Every proposed live signal has a documented purpose and owner.
- High-sensitivity/indirect signals are clearly marked “research only / excluded pending review.”
- There is an explicit way to test a reduced, minimally invasive feature set.

### P0. Improve fairness diagnostics

**Work:** preserve protected attributes for auditing only; remove any synthetic construction that makes fairness results look stronger than they are; add subgroup error rates, sample sizes, uncertainty, and intersectional views where feasible.

**Acceptance criteria:**
- Reports disclose how synthetic audit groups and outcomes were generated.
- The 80% rule is presented as one diagnostic, not a pass/fail certification.
- Small groups are flagged rather than interpreted confidently.

## Next — narrow the product hypothesis

### P1. Select one initial lending use case

Run discovery with lenders, risk teams, and target borrowers before choosing whether to start with operating MSMEs, MSME cold-start, retail thin-file, or another narrow segment.

**Acceptance criteria:**
- A documented problem statement tied to a real workflow and customer segment.
- A named incumbent decision process and baseline.
- A clearly defined outcome label and time window.
- Agreed constraints for credit risk, access, fairness, turnaround time, and customer recourse.

### P1. Start with the least invasive viable evidence

Prioritise data with a clear financial or business rationale (for example, consented cash-flow or verified business records) over behavioural surveillance-style proxies. Which source is feasible depends on the lender, product, permissions, and applicable law.

**Acceptance criteria:**
- A data-source feasibility and permission review is completed before collection.
- The incremental value of each signal family is tested through ablation.
- A fallback journey exists when applicants cannot or do not consent to optional data sharing.

### P1. Turn heuristic explanations into tested explanations

**Work:** distinguish input diagnostics from feature contributions; evaluate a model-faithful explanation method; write borrower-friendly reason codes; usability-test the language.

**Acceptance criteria:**
- Reasons correspond to validated model/policy drivers and do not overstate causality.
- Users can identify incorrect information and ask for a review.
- Explanation comprehension and appeal handling are tested before pilot use.

### P1. Define a step-up credit policy for cold-start cases

Explore whether a smaller initial exposure, staged limit increases, manual review, or additional evidence is safer and more useful than a single binary decision.

**Acceptance criteria:**
- Policy options are compared on expected risk, customer access, cost-to-serve, and operational burden.
- Limit increases depend on defined evidence and monitoring, not unreviewed score movement alone.

## Later — validate with governed real outcomes

### P2. Build an authorised, privacy-conscious data path

Only after source permissions and legal review, design integrations with explicit consent, scoped access, minimisation, retention/deletion, encryption, audit logging, and failure handling. The current application does not have these integrations.

### P2. Shadow-mode evaluation with a lender partner

Run predictions without affecting decisions, compare with incumbent decisions, and wait for outcomes to mature. Avoid using the model for decisions during shadow mode.

**Acceptance criteria:**
- A documented protocol and pre-specified success/stop criteria exist.
- Evaluation includes temporal holdouts, segment cuts, calibration, and performance stability.
- Disagreements are reviewed by risk experts and investigated for data/process issues.

### P3. Controlled pilot and ongoing monitoring

Only after legal, risk, lender, and operational approvals, test a limited rollout with clear monitoring, human escalation, customer recourse, and rollback.

**Acceptance criteria:**
- Monitoring tracks calibration, risk, approval/decline, subgroup outcomes, drift, overrides, complaints, and data quality.
- Model and policy versions are traceable.
- Documented stop/rollback triggers and owners exist.

## Measures to agree before a real pilot

Do not set targets from synthetic results. Agree target levels in advance with the lender and risk/compliance teams.

| Metric family | Candidate measures |
|---|---|
| Risk | Delinquency/default by agreed outcome horizon, loss severity where available, calibration, bad-borrower approvals |
| Access | Approval at controlled risk, good applicants rejected, coverage of thin-file/MSME applications |
| Fairness | Selection and error rates across justified audit groups, intersectional analysis, uncertainty |
| Experience | Application completion, consent withdrawal, time to decision, reason comprehension, complaints and appeals |
| Operations | Manual-review rate, overrides, data failures, cost-to-serve, monitoring/rollback incidents |
| Data | Consent coverage, missingness, source uptime, correction rates, freshness, retention/deletion compliance |

## Explicit non-goals for the next iteration

- Claiming superiority over CIBIL from the synthetic comparator.
- Adding more alternative signals simply because they are available.
- Using a single fairness statistic as evidence of non-discrimination.
- Treating the prototype's indicative APR or exposure rules as a real credit policy.
- Moving directly from demo scores to automated real-world lending decisions.
