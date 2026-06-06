# Problem Structure

## Objective

Increase credit distribution while lowering the default rate among approved borrowers compared with a conservative bureau-style benchmark.

## Four-Model Router

1. Retail Active
   - For individuals with enough live cash-flow and behavioral data.
   - Main risks: income instability, spend stress, routine volatility, weak repayment intent.

2. Retail Cold-Start
   - For thin-file individuals at onboarding.
   - Main risks: sparse evidence, high-risk app profile, onboarding friction, weak utility/income signals.

3. MSME Active
   - For operating businesses with GST, P&L, UPI, and live macro signals.
   - Main risks: working capital gaps, GST delays, ITC mismatch, local shocks, sector downturns.

4. MSME Cold-Start
   - For new businesses or informal enterprises entering formal credit.
   - Main risks: weak promoter anchor, insufficient setup capital, low cluster survival, incomplete documentation.

## Data Design

The synthetic data generator separates:

- Risk features: used by the model.
- Audit features: used for fairness reporting.
- Target labels: generated from latent capacity, discipline, liquidity, shock, and market factors.

Protected fields are intentionally excluded from training feature lists.

## Operating Point

The backtest compares each alt-credit model against a synthetic CIBIL-like benchmark. The benchmark uses fewer lagging variables and applies conservative approval rates. The alt-credit model tunes a PD cutoff that tries to:

- increase approval rate by at least 3 percentage points, and
- keep approved default rate no worse than the benchmark.

If no threshold meets both goals, it falls back to the best risk-adjusted candidate and exposes that in the report.

## Remaining Product Questions

- Should first-time retail approvals become a step-up credit line rather than a binary approval?
- Which fields are legally usable in each jurisdiction and lending product?
- What is the minimum viable consent bundle for each route?
- Should district news shocks reduce exposure automatically or trigger manual review first?
- What adverse-action reason code format should be used for borrowers and regulators?

