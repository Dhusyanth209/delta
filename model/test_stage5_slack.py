"""
Stage 5 Verification Script — Slack Risk Alert Webhook
Tests the /alerts/slack endpoint and Block Kit payload structure.
"""

import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import load_models, _build_slack_blocks, ProjectFeatures


def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 5 SLACK ALERT VERIFICATION")
    print("==================================================\n")

    # 1. Load models
    load_models()

    # 2. Build test data
    pf = ProjectFeatures(
        industry_type="BFSI",
        team_size=18,
        seniority_mix_junior=0.40,
        seniority_mix_mid=0.35,
        seniority_mix_senior=0.25,
        budget_planned_usd=450000,
        duration_planned_weeks=22,
        scope_change_count=6,
        client_type="fixed_bid",
        employee_cost_ratio=0.63,
        attrition_events=3,
        weekly_burn_rate_variance=0.18
    )

    pred_res = {
        "risk_class": "at_risk",
        "risk_confidence": 0.85,
        "overrun_percentage": 32.0,
        "predicted_final_cost_usd": 594000,
        "predicted_final_cost_inr": 49599000,
        "budget_planned_usd": 450000,
        "top_factors": [
            {"feature": "employee_cost_ratio", "impact": "increases_risk", "magnitude": 0.48, "description": "ECR 63% above 57% baseline"},
            {"feature": "scope_change_count", "impact": "increases_risk", "magnitude": 0.38, "description": "6 scope changes driving delivery risk"},
            {"feature": "attrition_events", "impact": "increases_risk", "magnitude": 0.30, "description": "3 attrition events increasing lateral hire costs"}
        ],
        "recommendations": [
            {"action": "Scope Freeze", "description": "Freeze scope changes after 60% completion", "expected_risk_reduction": 0.22},
            {"action": "Retention Bonuses", "description": "Implement retention incentives for key staff", "expected_risk_reduction": 0.15}
        ]
    }

    # 3. Test Block Kit payload structure
    payload = _build_slack_blocks(pf, pred_res)

    assert "blocks" in payload, "Missing 'blocks' key in payload"
    assert "text" in payload, "Missing 'text' fallback in payload"
    assert len(payload["blocks"]) >= 5, f"Expected at least 5 blocks, got {len(payload['blocks'])}"

    # Check header block
    header = payload["blocks"][0]
    assert header["type"] == "header", "First block should be header"
    assert "AT_RISK" in header["text"]["text"], f"Header should contain AT_RISK, got: {header['text']['text']}"
    print(f"[✓] Test 1 PASSED: Block Kit header contains risk class AT_RISK")

    # Check section fields
    section = payload["blocks"][1]
    assert section["type"] == "section", "Second block should be section"
    assert len(section["fields"]) == 6, f"Expected 6 fields, got {len(section['fields'])}"
    field_texts = " ".join(f["text"] for f in section["fields"])
    assert "$450,000" in field_texts, "Missing budget in fields"
    assert "$594,000" in field_texts, "Missing predicted cost in fields"
    print(f"[✓] Test 2 PASSED: Section fields contain correct budget & cost metrics")

    # Check SHAP section
    shap_block = payload["blocks"][3]
    assert "Risk Drivers" in shap_block["text"]["text"], "Missing SHAP risk drivers section"
    assert "Employee Cost Ratio" in shap_block["text"]["text"], "Missing ECR factor"
    print(f"[✓] Test 3 PASSED: SHAP risk drivers block populated correctly")

    # Check recommendations section
    rec_block = payload["blocks"][4]
    assert "Recommended Actions" in rec_block["text"]["text"], "Missing recommendations section"
    assert "Scope Freeze" in rec_block["text"]["text"], "Missing Scope Freeze recommendation"
    print(f"[✓] Test 4 PASSED: RL recommendations block populated correctly")

    # Check context footer
    ctx_block = payload["blocks"][-1]
    assert ctx_block["type"] == "context", "Last block should be context"
    assert "DELTA AI v2.0" in ctx_block["elements"][0]["text"], "Missing DELTA version in context"
    print(f"[✓] Test 5 PASSED: Context footer contains DELTA version stamp")

    # Check fallback text
    assert "AT_RISK" in payload["text"], "Fallback text missing risk class"
    assert "$594,000" in payload["text"], "Fallback text missing predicted cost"
    print(f"[✓] Test 6 PASSED: Fallback text summary is correct")

    # Print preview
    print(f"\n--- SLACK PAYLOAD PREVIEW ---")
    print(f"Header: {payload['blocks'][0]['text']['text']}")
    print(f"Fields: {len(payload['blocks'][1]['fields'])} metrics")
    print(f"Blocks: {len(payload['blocks'])} total")
    print(f"Fallback: {payload['text'][:80]}...")

    print("\n==================================================")
    print("   ALL STAGE 5 SLACK ALERT TESTS PASSED! ✓")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
