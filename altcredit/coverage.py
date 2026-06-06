from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RequirementCoverage:
    area: str
    status: str
    implementation: str
    evidence: str
    remaining_note: str = ""


REQUIREMENT_COVERAGE: tuple[RequirementCoverage, ...] = (
    RequirementCoverage(
        "Problem structure and clarifying questions",
        "Done",
        "Documented customer segmentation, default horizon, data access, cold-start, and MSME subtype questions.",
        "docs/problem_structure.md and app Coverage page",
    ),
    RequirementCoverage(
        "Separate individual and MSME models",
        "Done",
        "Implemented four routed models: retail active, retail cold-start, MSME active, MSME cold-start.",
        "altcredit/features.py",
    ),
    RequirementCoverage(
        "Cold-start models",
        "Done",
        "Retail and MSME cold-start pipelines have separate schemas, synthetic data, training, scoring, and templates.",
        "altcredit/data.py and altcredit/templates.py",
    ),
    RequirementCoverage(
        "Geolocation routine for business/leisure",
        "Done for MVP",
        "Retail active uses routine entropy and commute consistency; reusable entropy calculator added.",
        "altcredit/feature_engineering.py",
        "Raw H3/mobile event ingestion remains a production integration task.",
    ),
    RequirementCoverage(
        "Telco, UPI, cash payments, recovery, network working/idle",
        "Done for MVP",
        "Retail schemas include UPI velocity variance, telco regularity, prior recovery, and network working ratio.",
        "altcredit/features.py",
    ),
    RequirementCoverage(
        "Bot chat, current thought process, financial condition",
        "Done for MVP",
        "Retail active uses bot financial stress and repayment intent vectors.",
        "altcredit/features.py",
        "A production NLP embedder is represented by generated vectors in this simulation.",
    ),
    RequirementCoverage(
        "Transaction spend analysis",
        "Done",
        "Retail active includes fixed obligation, discretionary spend, essential timeliness, and spend helper functions.",
        "altcredit/features.py and altcredit/feature_engineering.py",
    ),
    RequirementCoverage(
        "GST plus P&L company analysis",
        "Done for MVP",
        "MSME active includes GST delay/consistency, ITC ratio, revenue, margin, DSCR, mandates, supplier, and working-capital signals.",
        "altcredit/features.py",
    ),
    RequirementCoverage(
        "GST city-sector analysis",
        "Done for MVP",
        "MSME models now categorize by sector, capital tier, and business location tier.",
        "altcredit/features.py",
    ),
    RequirementCoverage(
        "Blog persona and financial literacy",
        "Done for MVP",
        "Retail active includes financial literacy score and blog completion rate.",
        "altcredit/features.py",
    ),
    RequirementCoverage(
        "Beta plus industry data",
        "Done",
        "MSME active and cold-start include industry sector beta and sector growth/survival signals.",
        "altcredit/features.py",
    ),
    RequirementCoverage(
        "News state/district analysis",
        "Done for MVP",
        "MSME active includes district news sentiment and infrastructure anomaly indices.",
        "altcredit/features.py",
    ),
    RequirementCoverage(
        "Blend Random Forest, Gradient Boosting, SVM, Decision Tree",
        "Done",
        "Voting ensemble combines gradient boosting, random forest, calibrated linear SVM, and shallow decision tree.",
        "altcredit/models.py",
    ),
    RequirementCoverage(
        "Actuarial expected-loss and exposure logic",
        "Done for MVP",
        "Outputs PD, 300-900 score, risk-based APR, dynamic exposure limit, expected-loss proxy, and operating point.",
        "altcredit/scoring.py and altcredit/comparison.py",
    ),
    RequirementCoverage(
        "Unbiased synthetic data and fairness checks",
        "Done",
        "Protected/audit fields are generated for evaluation and excluded from model features except legitimate business categories.",
        "altcredit/data.py and artifacts/reports/*_fairness.json",
    ),
    RequirementCoverage(
        "CIBIL comparison",
        "Done with caveat",
        "Compares against a transparent synthetic CIBIL-like benchmark because the actual CIBIL model is proprietary.",
        "altcredit/backtest.py and altcredit/comparison.py",
    ),
)


CLARIFYING_QUESTIONS: tuple[str, ...] = (
    "Which consent path is primary: Account Aggregator, proprietary SDK permissions, lender-owned data, or all three?",
    "Which default target should drive production: 15 DPD early warning, 30 DPD technical default, or 90 DPD NPA?",
    "Should cold-start approvals always be step-up credit lines with auto-increase after repayment?",
    "Which alternative data fields are legally disallowed or sensitive for the exact lending jurisdiction?",
    "For MSMEs, should the first product prioritize kirana/micro-merchants, GST-registered small firms, or invoice-backed working capital?",
    "What adverse-action notice format is required for rejected borrowers and merchants?",
)


PASTED_TEXT_AUDIT: tuple[dict[str, str], ...] = (
    {
        "check": "Encoding integrity",
        "status": "Needs cleanup",
        "finding": "The pasted text contains corrupted/symbol-heavy characters such as em dash and lambda, so it is not clean source documentation.",
        "fix_in_mvp": "Clean project docs use plain ASCII and structured markdown.",
    },
    {
        "check": "Model count consistency",
        "status": "Resolved in MVP",
        "finding": "The text starts with two models, then correctly expands to four with cold-start variants.",
        "fix_in_mvp": "The app implements all four routed models.",
    },
    {
        "check": "CIBIL comparison",
        "status": "Caveated",
        "finding": "The pasted text implies outperforming CIBIL, but the real CIBIL model is proprietary.",
        "fix_in_mvp": "The code compares against a transparent synthetic CIBIL-like benchmark.",
    },
    {
        "check": "Algorithm claims",
        "status": "Resolved in MVP",
        "finding": "The text alternates between rejecting SVM/trees and later using them as weak learners.",
        "fix_in_mvp": "The code uses a clear soft-voting ensemble: gradient boosting, random forest, calibrated SVM, shallow tree.",
    },
    {
        "check": "SHAP and survival modeling",
        "status": "Partial by design",
        "finding": "The pasted text names SHAP and dual-horizon survival modeling, but does not provide implementable production details.",
        "fix_in_mvp": "The MVP provides reason codes, stage trace, PD, exposure logic, and a place to add SHAP/survival once real repayment histories exist.",
    },
    {
        "check": "Raw data ingestion",
        "status": "Simulation only",
        "finding": "The text describes AA, SDK, UPI, telco, GST, and news streams without actual connectors.",
        "fix_in_mvp": "Synthetic feature vectors and JSON templates simulate the data contract.",
    },
    {
        "check": "Fairness and privacy",
        "status": "Improved in MVP",
        "finding": "The text mentions guardrails but does not fully separate protected fields from model features.",
        "fix_in_mvp": "Protected/audit fields are separated and reports check disparate impact.",
    },
)


def coverage_rows() -> list[dict[str, str]]:
    return [item.__dict__ for item in REQUIREMENT_COVERAGE]


def pasted_text_audit_rows() -> list[dict[str, str]]:
    return [dict(row) for row in PASTED_TEXT_AUDIT]
