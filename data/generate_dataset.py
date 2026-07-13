"""
DELTA — Synthetic IT Project Dataset Generator
===============================================
Generates 800-1000 realistic synthetic IT project records with correlations
grounded in "The Indian IT Services Sector at a Crossroads" research paper.

KEY RESEARCH-GROUNDED PARAMETERS:
- Employee cost ratio: 57% industry average (paper's industry-wide employee-
  cost-to-revenue ratio), with tail toward 60% (TCS-level). Paper documents
  employee costs rising 206% vs revenue growth of 185% over a decade.
- Attrition rate: ~13-14% annualized (paper's stated industry attrition rate)
- Lateral-hire cost premium: 25-30% over the replaced role's remaining cost
  (paper's stated premium for lateral hires to replace attrited staff)
- Outcome-based pricing: paper notes industry-wide shift toward outcome-based
  contracts tied to business impact, raising stakes of overruns
- Cost-saving benchmarks: 30-40% operational cost reduction, 14% labor /
  12% equipment cost savings from project optimization, 6-12 month ROI

HONESTY NOTE: This dataset is SYNTHETIC. The research paper provides
industry-aggregate calibration numbers, NOT row-level project records.
The correlations are inspired by real industry dynamics but the individual
records are generated, not observed.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path

# Reproducibility
np.random.seed(42)

# ─── Configuration ───────────────────────────────────────────────────────────
NUM_PROJECTS = 950  # Target: 800-1000 range
USD_TO_INR = 83.5   # Approximate exchange rate

INDUSTRIES = [
    "BFSI", "Healthcare", "Retail", "Telecom",
    "Manufacturing", "Government", "Energy", "EdTech"
]

CLIENT_TYPES = ["fixed_bid", "outcome_based", "time_and_material"]
# Paper: shift toward outcome-based pricing; weight distribution reflects
# current industry mix with T&M still dominant but outcome-based growing
CLIENT_TYPE_WEIGHTS = [0.30, 0.25, 0.45]

OUTCOME_LABELS = ["on_track", "at_risk", "failed"]


def generate_base_features(n: int) -> pd.DataFrame:
    """Generate base project features before correlation-driven adjustments."""

    projects = pd.DataFrame()
    projects["project_id"] = [f"DELTA-{i+1:04d}" for i in range(n)]

    # Industry type — uniform distribution across verticals
    projects["industry_type"] = np.random.choice(INDUSTRIES, n)

    # Team size: 5-80, right-skewed (most projects are small-medium)
    projects["team_size"] = np.clip(
        np.random.lognormal(mean=2.7, sigma=0.6, size=n).astype(int),
        5, 80
    )

    # Seniority mix: junior/mid/senior percentages summing to 1.0
    # Smaller teams tend to be more senior-heavy
    raw_senior = np.clip(np.random.beta(2, 5, n) + 
                         (1.0 / (projects["team_size"].values * 0.3)) * 0.1, 0.05, 0.50)
    raw_junior = np.clip(np.random.beta(3, 4, n), 0.10, 0.60)
    raw_mid = 1.0 - raw_senior - raw_junior
    raw_mid = np.clip(raw_mid, 0.10, 0.70)
    # Renormalize
    total = raw_junior + raw_mid + raw_senior
    projects["seniority_mix_junior"] = np.round(raw_junior / total, 3)
    projects["seniority_mix_mid"] = np.round(raw_mid / total, 3)
    projects["seniority_mix_senior"] = np.round(
        1.0 - projects["seniority_mix_junior"] - projects["seniority_mix_mid"], 3
    )

    # Budget planned (USD): $50K - $5M, lognormal
    projects["budget_planned_usd"] = np.clip(
        np.random.lognormal(mean=12.2, sigma=0.9, size=n),
        50_000, 5_000_000
    ).astype(int)
    # INR equivalent
    projects["budget_planned_inr"] = (
        projects["budget_planned_usd"] * USD_TO_INR
    ).astype(int)

    # Duration planned (weeks): 8-52, correlated with budget
    budget_norm = (projects["budget_planned_usd"] - 50_000) / (5_000_000 - 50_000)
    projects["duration_planned_weeks"] = np.clip(
        (8 + budget_norm * 35 + np.random.normal(0, 5, n)).astype(int),
        8, 52
    )

    # Scope change count: 0-15, Poisson distributed
    # Higher for larger/longer projects
    scope_lambda = 2.5 + budget_norm * 4.0
    projects["scope_change_count"] = np.clip(
        np.random.poisson(scope_lambda), 0, 15
    )

    # Client type — weighted random
    projects["client_type"] = np.random.choice(
        CLIENT_TYPES, n, p=CLIENT_TYPE_WEIGHTS
    )

    # ─── Employee Cost Ratio ─────────────────────────────────────────────
    # Paper: 57% industry average employee-cost-to-revenue ratio
    # Paper: employee costs grew 206% while revenue grew 185% over a decade
    # Paper: TCS-level firms have ratios pushing toward 60%
    # Distribution: centered at 0.57, std 0.04, with a right tail toward 0.65+
    projects["employee_cost_ratio"] = np.clip(
        np.random.normal(
            loc=0.57,   # 57% baseline per industry employee-cost-ratio data
            scale=0.04, # Spread to cover 0.48-0.68 range
            size=n
        ) + np.random.exponential(0.01, n),  # Right skew toward higher ratios
        0.45, 0.72
    ).round(3)

    # ─── Attrition Events ────────────────────────────────────────────────
    # Paper: ~13-14% annualized attrition rate in Indian IT
    # Convert to per-project: prob = 1 - (1 - 0.135)^(duration_weeks/52)
    # Each team member has this probability of leaving during the project
    annualized_rate = 0.135  # ~13.5% per paper
    durations_years = projects["duration_planned_weeks"].values / 52.0
    # Boost the per-project probability slightly so we get meaningful
    # attrition in shorter projects too (reality: project-level turnover
    # is often higher than company-wide average due to project stress)
    per_project_attrition_prob = np.clip(
        1.0 - (1.0 - annualized_rate) ** durations_years + 
        np.random.uniform(0.02, 0.08, len(durations_years)),
        0.03, 0.5
    )

    projects["attrition_events"] = np.array([
        np.random.binomial(
            n=team_size,
            p=min(prob, 0.5)
        )
        for team_size, prob in zip(
            projects["team_size"].values,
            per_project_attrition_prob
        )
    ])
    # Cap attrition at 40% of team size (extreme but possible)
    projects["attrition_events"] = np.minimum(
        projects["attrition_events"],
        (projects["team_size"] * 0.4).astype(int)
    )

    # Weekly burn rate variance: how much spend deviated week to week
    # Higher for projects with more scope changes and attrition
    base_variance = np.random.beta(2, 8, n) * 0.3
    scope_effect = projects["scope_change_count"].values * 0.008
    attrition_effect = projects["attrition_events"].values * 0.012
    projects["weekly_burn_rate_variance"] = np.clip(
        base_variance + scope_effect + attrition_effect +
        np.random.normal(0, 0.02, n),
        0.01, 0.45
    ).round(3)

    return projects


def compute_risk_score(row: pd.Series) -> float:
    """
    Compute a continuous risk score [0, 1] based on multiple correlated
    factors. This drives both the outcome label and the actual cost/duration.

    Research-grounded correlations are documented inline.
    Scoring is ADDITIVE — most projects accumulate risk from multiple
    factors simultaneously, which is realistic.
    """
    score = 0.0

    # ─── Factor 1: Employee Cost Ratio ───────────────────────────────────
    # Paper: 57% baseline; ratios above this indicate margin pressure
    # Paper: employee costs rising faster than revenue (206% vs 185%)
    # is the STRUCTURAL driver of margin erosion
    ecr = row["employee_cost_ratio"]
    # Continuous contribution: every point above 0.52 adds risk
    score += max(0, (ecr - 0.52) * 1.5)  # 0.57 → 0.075, 0.62 → 0.15, 0.70 → 0.27

    # ─── Factor 2: Attrition Impact ──────────────────────────────────────
    # Paper: lateral-hire cost premium of 25-30% on replaced role
    # Paper: ~13-14% annualized attrition rate
    # Each attrition event both costs more AND slows delivery
    attrition_ratio = row["attrition_events"] / max(row["team_size"], 1)
    score += attrition_ratio * 1.2  # Continuous: 10% attrition → 0.12
    # Extra penalty for absolute count (even on large teams, >3 departures hurts)
    score += min(row["attrition_events"] * 0.02, 0.12)

    # ─── Factor 3: Scope Creep × Contract Type ───────────────────────────
    # Paper: shift toward outcome-based pricing tied to business impact
    # Fixed-bid and outcome-based absorb scope creep as direct margin loss
    # T&M contracts can bill for additional scope (less risky)
    scope = row["scope_change_count"]
    if row["client_type"] in ("fixed_bid", "outcome_based"):
        score += scope * 0.035  # Each scope change = 3.5% risk on fixed contracts
    else:  # time_and_material
        score += scope * 0.015  # T&M absorbs scope changes better

    # ─── Factor 4: Team-Budget Mismatch ──────────────────────────────────
    # General PM logic: small teams on large budgets = under-resourced
    budget_per_person = row["budget_planned_usd"] / max(row["team_size"], 1)
    if budget_per_person > 150_000:
        score += 0.10 + (budget_per_person - 150_000) / 1_000_000 * 0.15
    if row["team_size"] < 8 and row["budget_planned_usd"] > 500_000:
        score += 0.08  # Very small team, substantial budget

    # ─── Factor 5: Burn Rate Instability ─────────────────────────────────
    # High week-to-week spend variance = poor project control
    brv = row["weekly_burn_rate_variance"]
    score += brv * 0.5  # Continuous: 0.20 variance → 0.10 risk

    # ─── Factor 6: Duration risk ─────────────────────────────────────────
    # Very long projects (>30 weeks) have inherently more risk
    dur = row["duration_planned_weeks"]
    if dur > 30:
        score += (dur - 30) * 0.004  # Each week over 30 → +0.4%

    # ─── Factor 7: Junior-heavy teams ────────────────────────────────────
    if row["seniority_mix_junior"] > 0.40:
        score += (row["seniority_mix_junior"] - 0.40) * 0.3

    # Add noise so the data is NOT perfectly separable
    # (real data would have unexplained variance)
    noise = np.random.normal(0, 0.06)
    score = np.clip(score + noise, 0.0, 1.0)

    return score


def assign_outcomes_and_actuals(projects: pd.DataFrame) -> pd.DataFrame:
    """
    Use the risk score to determine:
    1. Outcome label (on_track / at_risk / failed)
    2. Actual budget (with overrun correlated to risk)
    3. Actual duration (with delay correlated to risk)
    """

    # Compute risk scores
    projects["_risk_score"] = projects.apply(compute_risk_score, axis=1)

    # ─── Outcome Labels ──────────────────────────────────────────────────
    # Thresholds with some overlap (not clean boundaries)
    # Target distribution: ~45% on_track, 30% at_risk, 20-25% failed
    def label_from_score(score):
        # Add per-label noise for overlap at boundaries
        r = np.random.uniform(-0.03, 0.03)
        if score < 0.30 + r:
            return "on_track"
        elif score < 0.48 + r:
            return "at_risk"
        else:
            return "failed"

    projects["outcome_label"] = projects["_risk_score"].apply(label_from_score)

    # ─── Actual Budget ───────────────────────────────────────────────────
    # Overrun multiplier: 1.0 (on budget) to 1.6+ (60% over)
    # Driven by risk score + attrition lateral-hire premium
    def compute_actual_budget(row):
        base_overrun = 1.0 + row["_risk_score"] * 0.5  # Up to 50% from risk

        # Paper: lateral-hire cost premium of 25-30% per replaced role
        # Each attrition event adds cost: premium × (remaining_role_cost)
        # Simplified: each event adds ~2-3% to total budget
        attrition_cost_bump = row["attrition_events"] * np.random.uniform(0.02, 0.03)

        # Scope change cost: each change adds ~1-2% to budget
        scope_cost_bump = row["scope_change_count"] * np.random.uniform(0.008, 0.02)

        overrun = base_overrun + attrition_cost_bump + scope_cost_bump
        # Add noise
        overrun += np.random.normal(0, 0.04)
        overrun = max(overrun, 0.85)  # Some projects can come in under budget

        actual_usd = int(row["budget_planned_usd"] * overrun)
        return actual_usd

    projects["budget_actual_usd"] = projects.apply(compute_actual_budget, axis=1)
    projects["budget_actual_inr"] = (
        projects["budget_actual_usd"] * USD_TO_INR
    ).astype(int)

    # ─── Actual Duration ─────────────────────────────────────────────────
    def compute_actual_duration(row):
        base_delay = 1.0 + row["_risk_score"] * 0.4  # Up to 40% delay
        # Attrition slows delivery: each event adds ~1-2% delay
        attrition_delay = row["attrition_events"] * np.random.uniform(0.01, 0.02)
        delay = base_delay + attrition_delay + np.random.normal(0, 0.05)
        delay = max(delay, 0.90)  # Some finish early
        actual_weeks = max(int(row["duration_planned_weeks"] * delay), 
                          row["duration_planned_weeks"] - 2)
        return actual_weeks

    projects["duration_actual_weeks"] = projects.apply(
        compute_actual_duration, axis=1
    )

    # Drop internal risk score (not a feature for the model)
    projects.drop(columns=["_risk_score"], inplace=True)

    return projects


def validate_dataset(df: pd.DataFrame) -> None:
    """Print validation statistics to confirm realistic distributions."""
    print("\n" + "=" * 70)
    print("DELTA DATASET VALIDATION REPORT")
    print("=" * 70)

    print(f"\nTotal records: {len(df)}")

    print(f"\n── Outcome Distribution ──")
    print(df["outcome_label"].value_counts().to_string())
    print(f"  (Target: roughly 40-50% on_track, 30-35% at_risk, 15-25% failed)")

    print(f"\n── Employee Cost Ratio ──")
    print(f"  Mean:   {df['employee_cost_ratio'].mean():.3f}  (target: ~0.57)")
    print(f"  Median: {df['employee_cost_ratio'].median():.3f}")
    print(f"  Std:    {df['employee_cost_ratio'].std():.3f}")
    print(f"  Min:    {df['employee_cost_ratio'].min():.3f}")
    print(f"  Max:    {df['employee_cost_ratio'].max():.3f}")

    print(f"\n── Attrition Events ──")
    print(f"  Mean:   {df['attrition_events'].mean():.2f}")
    print(f"  Median: {df['attrition_events'].median():.1f}")
    print(f"  Max:    {df['attrition_events'].max()}")
    print(f"  Zero-attrition projects: {(df['attrition_events'] == 0).sum()}")

    print(f"\n── Budget Overrun ──")
    overrun = (df["budget_actual_usd"] / df["budget_planned_usd"])
    print(f"  Mean overrun ratio: {overrun.mean():.3f}")
    print(f"  Under-budget projects: {(overrun < 1.0).sum()}")
    print(f"  >20% overrun: {(overrun > 1.20).sum()}")
    print(f"  >50% overrun: {(overrun > 1.50).sum()}")

    print(f"\n── Correlation Checks ──")
    # Check: high employee_cost_ratio → more failures
    high_ecr = df[df["employee_cost_ratio"] > 0.60]
    low_ecr = df[df["employee_cost_ratio"] <= 0.57]
    print(f"  High ECR (>0.60) failure rate: "
          f"{(high_ecr['outcome_label'] == 'failed').mean():.1%}")
    print(f"  Low ECR (<=0.57) failure rate:  "
          f"{(low_ecr['outcome_label'] == 'failed').mean():.1%}")

    # Check: fixed_bid + high scope → more failures
    fixed_high_scope = df[
        (df["client_type"] == "fixed_bid") & (df["scope_change_count"] >= 6)
    ]
    tam_high_scope = df[
        (df["client_type"] == "time_and_material") & (df["scope_change_count"] >= 6)
    ]
    print(f"  Fixed-bid + high scope (>=6) failure rate: "
          f"{(fixed_high_scope['outcome_label'] == 'failed').mean():.1%}" 
          if len(fixed_high_scope) > 0 else "  (no data)")
    print(f"  T&M + high scope (>=6) failure rate:       "
          f"{(tam_high_scope['outcome_label'] == 'failed').mean():.1%}"
          if len(tam_high_scope) > 0 else "  (no data)")

    print(f"\n── Team Size Distribution ──")
    print(f"  Mean: {df['team_size'].mean():.1f}")
    print(f"  <10:  {(df['team_size'] < 10).sum()}")
    print(f"  >50:  {(df['team_size'] > 50).sum()}")

    print(f"\n── Sample Rows (first 5) ──")
    sample_cols = [
        "project_id", "team_size", "budget_planned_usd", "budget_actual_usd",
        "employee_cost_ratio", "attrition_events", "scope_change_count",
        "client_type", "outcome_label"
    ]
    print(df[sample_cols].head(5).to_string(index=False))

    print("\n" + "=" * 70)


def main():
    print("Generating DELTA synthetic dataset...")

    # Stage 1: Generate base features
    projects = generate_base_features(NUM_PROJECTS)

    # Stage 2: Compute outcomes and actuals based on correlated risk
    projects = assign_outcomes_and_actuals(projects)

    # Reorder columns for clarity
    column_order = [
        "project_id", "industry_type", "team_size",
        "seniority_mix_junior", "seniority_mix_mid", "seniority_mix_senior",
        "budget_planned_usd", "budget_planned_inr",
        "budget_actual_usd", "budget_actual_inr",
        "duration_planned_weeks", "duration_actual_weeks",
        "scope_change_count", "client_type",
        "employee_cost_ratio", "attrition_events",
        "weekly_burn_rate_variance", "outcome_label"
    ]
    projects = projects[column_order]

    # Save to CSV
    output_dir = Path(__file__).parent
    output_path = output_dir / "synthetic_projects.csv"
    projects.to_csv(output_path, index=False)
    print(f"\nSaved {len(projects)} records to {output_path}")

    # Validate
    validate_dataset(projects)

    return projects


if __name__ == "__main__":
    main()
