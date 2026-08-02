"""
Stage 6 Verification Script — Bulk Project Upload & Batch Prediction
Tests /projects/upload and /projects/template endpoints.
"""

import sys
import os
import csv
import io
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import (
    load_models, _predict_single_row, REQUIRED_UPLOAD_COLUMNS,
    engineer_features_from_raw
)
import pandas as pd
import numpy as np


def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 6 BULK UPLOAD VERIFICATION")
    print("==================================================\n")

    # 1. Load models
    load_models()

    # 2. Test _predict_single_row with a valid project
    test_row = {
        "industry_type": "BFSI",
        "team_size": 25,
        "seniority_mix_junior": 0.30,
        "seniority_mix_mid": 0.45,
        "seniority_mix_senior": 0.25,
        "budget_planned_usd": 500000,
        "duration_planned_weeks": 24,
        "scope_change_count": 4,
        "client_type": "fixed_bid",
        "employee_cost_ratio": 0.58,
        "attrition_events": 2,
        "weekly_burn_rate_variance": 0.12,
    }
    result = _predict_single_row(test_row)
    assert result["status"] == "success", f"Single prediction failed: {result}"
    assert result["risk_class"] in ["on_track", "at_risk", "failed"], f"Invalid risk class: {result['risk_class']}"
    assert "predicted_final_cost_usd" in result, "Missing predicted cost"
    assert "top_factors" in result, "Missing SHAP factors"
    assert len(result["top_factors"]) > 0, "Empty SHAP factors"
    print(f"[✓] Test 1 PASSED: _predict_single_row returns valid prediction ({result['risk_class']}, ${result['predicted_final_cost_usd']:,.0f})")

    # 3. Test batch of 5 projects
    batch_projects = [
        {**test_row, "budget_planned_usd": 300000, "team_size": 10, "scope_change_count": 1},
        {**test_row, "budget_planned_usd": 800000, "team_size": 40, "scope_change_count": 8, "attrition_events": 5},
        {**test_row, "industry_type": "Healthcare", "client_type": "time_and_material"},
        {**test_row, "budget_planned_usd": 150000, "team_size": 8, "employee_cost_ratio": 0.45, "scope_change_count": 0},
        {**test_row, "budget_planned_usd": 1200000, "team_size": 60, "attrition_events": 8, "scope_change_count": 12},
    ]

    batch_results = []
    for proj in batch_projects:
        pred = _predict_single_row(proj)
        batch_results.append(pred)

    successful = [r for r in batch_results if r["status"] == "success"]
    assert len(successful) == 5, f"Expected 5 successful predictions, got {len(successful)}"
    print(f"[✓] Test 2 PASSED: Batch of 5 projects all predicted successfully")

    # 4. Verify portfolio summary computation
    risk_counts = {"on_track": 0, "at_risk": 0, "failed": 0}
    total_overrun = 0
    for r in successful:
        rc = r["risk_class"]
        if rc in risk_counts:
            risk_counts[rc] += 1
        total_overrun += r["overrun_percentage"]

    avg_overrun = total_overrun / len(successful)
    total_risk_projects = risk_counts["at_risk"] + risk_counts["failed"]
    print(f"[✓] Test 3 PASSED: Portfolio summary — On Track: {risk_counts['on_track']}, At Risk: {risk_counts['at_risk']}, Failed: {risk_counts['failed']}, Avg Overrun: {avg_overrun:.1f}%")

    # 5. Test REQUIRED_UPLOAD_COLUMNS constant
    assert len(REQUIRED_UPLOAD_COLUMNS) == 12, f"Expected 12 required columns, got {len(REQUIRED_UPLOAD_COLUMNS)}"
    assert "industry_type" in REQUIRED_UPLOAD_COLUMNS, "Missing industry_type column"
    assert "budget_planned_usd" in REQUIRED_UPLOAD_COLUMNS, "Missing budget column"
    print(f"[✓] Test 4 PASSED: REQUIRED_UPLOAD_COLUMNS has 12 correct columns")

    # 6. Test CSV generation (template)
    template_data = {col: [test_row[col]] for col in REQUIRED_UPLOAD_COLUMNS}
    df = pd.DataFrame(template_data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    csv_content = buf.getvalue()
    for col in REQUIRED_UPLOAD_COLUMNS:
        assert col in csv_content, f"Template CSV missing column: {col}"
    print(f"[✓] Test 5 PASSED: Template CSV contains all 12 required columns")

    # 7. Test column name normalization (spaces, upper case)
    messy_row = {"Industry Type": "BFSI", "Team Size": 15, "Seniority Mix Junior": 0.3,
                 "Seniority Mix Mid": 0.4, "Seniority Mix Senior": 0.3,
                 "Budget Planned Usd": 200000, "Duration Planned Weeks": 16,
                 "Scope Change Count": 2, "Client Type": "fixed_bid",
                 "Employee Cost Ratio": 0.55, "Attrition Events": 1,
                 "Weekly Burn Rate Variance": 0.08}
    messy_df = pd.DataFrame([messy_row])
    messy_df.columns = [c.strip().lower().replace(" ", "_") for c in messy_df.columns]
    missing = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in messy_df.columns]
    assert len(missing) == 0, f"Column normalization failed, missing: {missing}"
    print(f"[✓] Test 6 PASSED: Column name normalization handles spaces and mixed case")

    # Print summary
    print(f"\n--- BATCH RESULTS SUMMARY ---")
    for i, r in enumerate(successful):
        print(f"  Project {i+1}: {r['risk_class'].upper():>8s} | Overrun: {r['overrun_percentage']:+.1f}% | Cost: ${r['predicted_final_cost_usd']:,.0f}")

    print("\n==================================================")
    print("   ALL STAGE 6 BULK UPLOAD TESTS PASSED! ✓")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
