# Portfolio Demo Guide

A concise way to present the prototype in a product-management portfolio without overclaiming.

## Suggested two-to-three-minute walkthrough

### 1. Start with the product question (20–30 seconds)

“I'm exploring a product question: when an individual or MSME has limited bureau history, can additional, appropriate financial or business evidence help a lender make a better-informed decision? This app is a simulation for inspecting that question—not a claim that the model beats CIBIL.”

### 2. Show the four routes (20–30 seconds)

Point out Retail Active, Retail Cold-Start, MSME Active, and MSME Cold-Start. Explain that the evidence available and the underwriting context differ; the separate routes make those assumptions explicit.

### 3. Score a hypothetical profile (30–40 seconds)

Use **Score Lab** and change a few understandable financial/business inputs. Show the score, simulated 90-day PD, decision, indicative limit/APR, reason codes, and synthetic baseline result. Remind the viewer that all inputs and outcomes are synthetic.

### 4. Explain one decision (30–40 seconds)

Open **Calculation Trace**. Show how routing, missing-value completion, diagnostics, ensemble probabilities, score mapping, and thresholding lead to the result. Describe current reason codes as heuristic diagnostics, not SHAP explanations.

### 5. Show how the model is evaluated (30–40 seconds)

Open **Model Comparison** and explain why access metrics are shown alongside approved-default rate, ranking/probability metrics, and exposure proxies. Call out that the benchmark is an explicit synthetic formula, not actual TransUnion CIBIL.

### 6. End with critical judgement (20–30 seconds)

“The key next steps are to separate threshold tuning from final testing, validate the data and label design with real outcomes, and review each alternative signal for necessity, consent, proxy risk, and borrower recourse before any pilot.”

## Language that stays credible

**Say:**
- “I built a working simulation to make the product hypothesis testable.”
- “The synthetic benchmark helps exercise the comparison workflow; it does not model CIBIL.”
- “The project makes access-risk trade-offs visible and identifies what would need to be validated next.”
- “The feature list is exploratory, with some signals explicitly requiring a stronger governance case.”

**Avoid:**
- “This is better than CIBIL.”
- “The model is unbiased” or “the model is fair” based on generated data.
- “The score is equivalent to a real bureau score.”
- “The system is production-ready” or “lender-ready.”
- Claims about actual reduction in defaults, financial inclusion, or business impact.

## Portfolio page copy

**Title:** Alternative Credit Scoring Simulation  
**Subtitle:** A product and model-evaluation prototype for thin-file and MSME underwriting  
**Summary:** Designed a working simulation to explore how supplementary financial and business signals might help evaluate applicants with limited credit history. Built four applicant routes, an interactive scoring workbench, a scoring trace, a synthetic benchmark, and multi-metric risk/access reporting. Documented the limits of synthetic evidence and the validation, fairness, consent, and governance work required before a real pilot.

## Resume bullet options

- Built a Streamlit-based alternative-credit scoring simulation for thin-file individuals and MSMEs, translating a product hypothesis into four scoring routes, an inspectable decision trace, and risk/access evaluation workflows.
- Designed evaluation views covering approval outcomes, approved-borrower default rate, ranking/probability metrics, portfolio proxies, and audit-group selection rates; documented why synthetic results cannot establish real-world performance or CIBIL superiority.
- Defined a staged product roadmap focused on validation integrity, consent and signal necessity, subgroup diagnostics, and borrower-facing explanation/recourse before any real-data pilot.

Use only the bullets that accurately reflect your own contribution and the current repository state.

## Questions an interviewer may ask

**Why use synthetic data?**  
To exercise workflows and inspect assumptions without using real borrower data. It is useful for product prototyping, but does not validate real predictive power.

**Why not claim it beats CIBIL?**  
The comparator is a project-authored formula, not the proprietary TransUnion CIBIL model, and the prototype has no matched real outcomes. A claim of superiority would be unsupported.

**What is the most important methodological improvement?**  
Tune the operating threshold on a validation partition and evaluate it once on an untouched test set. The current workflow uses the same holdout to tune and report the operating point.

**What is the biggest product risk?**  
Assuming alternative data is inherently better. Some device, location, telco, SMS, and behavioural proxies can be intrusive or discriminatory. Each signal needs a demonstrated purpose, lawful basis, consent/privacy review where applicable, fairness testing, and a non-digital or data-limited path where feasible.

**What would you do next?**  
Choose one real lending workflow through discovery, define an incumbent baseline and outcome window, identify the least invasive viable data sources, preregister risk/access/fairness measures, and run an authorised shadow-mode evaluation before any controlled pilot.
