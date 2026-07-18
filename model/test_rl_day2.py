"""Day 2 — RL Verification: Test RL recommendations on 5+ sample projects."""
import json
import requests
import sys

API = "http://localhost:8000"

# 5 different project profiles covering low/medium/high risk
test_projects = [
    {
        "name": "LOW RISK: Small healthcare, on-track indicators",
        "features": {
            "industry_type": "Healthcare", "team_size": 8,
            "seniority_mix_junior": 0.20, "seniority_mix_mid": 0.40, "seniority_mix_senior": 0.40,
            "budget_planned_usd": 200000, "duration_planned_weeks": 12,
            "scope_change_count": 1, "client_type": "time_and_material",
            "employee_cost_ratio": 0.52, "attrition_events": 0,
            "weekly_burn_rate_variance": 0.05,
        },
    },
    {
        "name": "MEDIUM RISK: Mid-size BFSI, some scope creep",
        "features": {
            "industry_type": "BFSI", "team_size": 20,
            "seniority_mix_junior": 0.35, "seniority_mix_mid": 0.40, "seniority_mix_senior": 0.25,
            "budget_planned_usd": 400000, "duration_planned_weeks": 20,
            "scope_change_count": 5, "client_type": "fixed_bid",
            "employee_cost_ratio": 0.58, "attrition_events": 2,
            "weekly_burn_rate_variance": 0.12,
        },
    },
    {
        "name": "HIGH RISK: Large retail, heavy scope + fixed-bid",
        "features": {
            "industry_type": "Retail", "team_size": 40,
            "seniority_mix_junior": 0.50, "seniority_mix_mid": 0.30, "seniority_mix_senior": 0.20,
            "budget_planned_usd": 800000, "duration_planned_weeks": 30,
            "scope_change_count": 12, "client_type": "fixed_bid",
            "employee_cost_ratio": 0.65, "attrition_events": 6,
            "weekly_burn_rate_variance": 0.22,
        },
    },
    {
        "name": "MEDIUM RISK: Government T&M, high attrition",
        "features": {
            "industry_type": "Government", "team_size": 15,
            "seniority_mix_junior": 0.45, "seniority_mix_mid": 0.35, "seniority_mix_senior": 0.20,
            "budget_planned_usd": 350000, "duration_planned_weeks": 16,
            "scope_change_count": 3, "client_type": "time_and_material",
            "employee_cost_ratio": 0.60, "attrition_events": 4,
            "weekly_burn_rate_variance": 0.15,
        },
    },
    {
        "name": "HIGH RISK: Telecom outcome-based, very junior",
        "features": {
            "industry_type": "Telecom", "team_size": 30,
            "seniority_mix_junior": 0.55, "seniority_mix_mid": 0.30, "seniority_mix_senior": 0.15,
            "budget_planned_usd": 600000, "duration_planned_weeks": 26,
            "scope_change_count": 8, "client_type": "outcome_based",
            "employee_cost_ratio": 0.63, "attrition_events": 5,
            "weekly_burn_rate_variance": 0.20,
        },
    },
]

print("=" * 70)
print("DAY 2 — RL VERIFICATION: 5 Sample Projects")
print("=" * 70)

all_passed = True

for i, proj in enumerate(test_projects):
    print(f"\n{'─' * 60}")
    print(f"Project {i+1}: {proj['name']}")
    print(f"{'─' * 60}")

    try:
        resp = requests.post(f"{API}/predict", json=proj["features"], timeout=10)
        resp.raise_for_status()
        result = resp.json()
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        all_passed = False
        continue

    print(f"  Risk:       {result['risk_class']} ({result['risk_confidence']*100:.1f}%)")
    print(f"  Overrun:    {result['overrun_percentage']:+.1f}%")
    print(f"  Top factor: {result['top_factors'][0]['feature']} ({result['top_factors'][0]['impact']})")

    recs = result.get("recommendations", [])
    if not recs:
        print(f"  ✗ NO RECOMMENDATIONS returned")
        all_passed = False
    else:
        print(f"  Recommendations ({len(recs)}):")
        for rec in recs:
            reduction = rec["expected_risk_reduction"] * 100
            print(f"    → {rec['action']}: {rec['description']}")
            print(f"      Risk reduction: {reduction:.1f}%, Confidence: {rec['confidence']:.3f}")

        # Sanity checks
        checks = []

        # Check 1: At least 1 recommendation
        checks.append(("Has recommendations", len(recs) >= 1))

        # Check 2: No negative risk reductions
        checks.append(("No negative reductions", all(r["expected_risk_reduction"] >= 0 for r in recs)))

        # Check 3: Confidence values are between 0 and 1
        checks.append(("Valid confidence range", all(0 <= r["confidence"] <= 1 for r in recs)))

        # Check 4: Recommendations are sorted by usefulness (high risk should not get "No Change" first)
        if result["risk_class"] in ["at_risk", "failed"]:
            checks.append(("No 'No Change' for risky", recs[0]["action"] != "No Change"))

        for check_name, passed in checks:
            status = "✓" if passed else "✗"
            print(f"    [{status}] {check_name}")
            if not passed:
                all_passed = False

print(f"\n{'=' * 70}")
if all_passed:
    print("✓ ALL 5 PROJECTS PASSED — RL recommendations are sensible")
    print("  Decision: RL STAYS ENABLED")
else:
    print("✗ SOME CHECKS FAILED — review issues above")
print(f"{'=' * 70}")
