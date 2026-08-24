"""
Stage 3 Verification Script — Executive Report Generation
Tests the /report endpoint logic directly.
"""

import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import (
    load_models,
    ReportRequest,
    generate_executive_report,
    ProjectFeatures
)

import asyncio

async def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 3 REPORT VERIFICATION")
    print("==================================================\n")
    
    # 1. Load models
    load_models()
    
    baseline = ProjectFeatures(
        industry_type="BFSI",
        team_size=15,
        seniority_mix_junior=0.4,
        seniority_mix_mid=0.35,
        seniority_mix_senior=0.25,
        budget_planned_usd=500000,
        duration_planned_weeks=24,
        scope_change_count=5,
        client_type="fixed_bid",
        employee_cost_ratio=0.62,
        attrition_events=2,
        weekly_burn_rate_variance=0.15
    )
    
    pred_res = {
        "risk_class": "at_risk",
        "risk_confidence": 0.82,
        "overrun_percentage": 28.5,
        "predicted_final_cost_usd": 642500,
        "predicted_final_cost_inr": 53648750,
        "budget_planned_usd": 500000,
        "top_factors": [
            {"feature": "employee_cost_ratio", "impact": "increases_risk", "magnitude": 0.45, "description": "Employee cost ratio 62% above 57% baseline"},
            {"feature": "attrition_events", "impact": "increases_risk", "magnitude": 0.32, "description": "2 attrition events increasing lateral hire burden"}
        ],
        "recommendations": [
            {"action": "Right-Size Seniority Mix", "description": "Rebalance junior/senior developers", "expected_risk_reduction": 0.18}
        ]
    }
    
    req = ReportRequest(
        project_features=baseline,
        prediction_result=pred_res
    )
    
    res = await generate_executive_report(req)
    
    assert res.markdown_content is not None, "Report Markdown is empty"
    assert "DELTA PMO EXECUTIVE RISK & FINANCIAL AUDIT REPORT" in res.markdown_content, "Missing header"
    assert "AT_RISK" in res.markdown_content, "Missing risk status in report"
    assert "$642,500.00 USD" in res.markdown_content, "Missing cost metric in report"
    
    print("[✓] Stage 3 Executive Report Test PASSED!")
    print(f"    Timestamp: {res.timestamp}")
    print(f"    Summary Metrics: {res.summary_metrics}")
    print("\n--- REPORT PREVIEW SNIPPET ---\n")
    print(res.markdown_content[:350])
    print("\n==================================================")
    print("   STAGE 3 EXECUTIVE REPORT GENERATOR VERIFIED!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_tests())
