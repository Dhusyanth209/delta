"""
Stage 13 Verification Script — Email Alerts & HTML Templates
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import (
    load_models,
    _build_email_html,
    EmailAlertRequest,
    send_email_alert,
    ProjectFeatures
)
import asyncio


def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 13 EMAIL ALERT VERIFICATION")
    print("==================================================\n")

    load_models()

    pf = ProjectFeatures(
        industry_type="BFSI",
        team_size=35,
        seniority_mix_junior=0.45,
        seniority_mix_mid=0.35,
        seniority_mix_senior=0.20,
        budget_planned_usd=750000,
        duration_planned_weeks=30,
        scope_change_count=7,
        client_type="fixed_bid",
        employee_cost_ratio=0.62,
        attrition_events=4,
        weekly_burn_rate_variance=0.18
    )

    pr = {
        "risk_class": "at_risk",
        "risk_confidence": 0.88,
        "budget_planned_usd": 750000.0,
        "predicted_final_cost_usd": 995000.0,
        "overrun_percentage": 32.67,
        "top_factors": [
            {"feature": "employee_cost_ratio", "impact": "increases_risk", "magnitude": 0.45, "description": "ECR is 62% vs 57% baseline"},
            {"feature": "scope_change_count", "impact": "increases_risk", "magnitude": 0.38, "description": "7 scope change events"},
            {"feature": "attrition_events", "impact": "increases_risk", "magnitude": 0.30, "description": "4 team attrition incidents"}
        ],
        "recommendations": [
            {"action": "Scope Freeze Protocol", "description": "Lock requirements past milestone 3", "expected_risk_reduction": 0.22},
            {"action": "Senior Staff Retention", "description": "Key contributor retention program", "expected_risk_reduction": 0.16}
        ]
    }

    # Test 1: HTML & Plain text email generation
    subject, plain, html = _build_email_html(pf, pr)
    assert "DELTA" in subject, "Subject missing DELTA"
    assert "AT_RISK" in subject, "Subject missing risk status"
    assert "750,000" in plain, "Plain text missing budget"
    assert "995,000" in html, "HTML missing predicted cost"
    assert "Scope Freeze Protocol" in html, "HTML missing recommendation"
    assert "employee_cost_ratio" not in html, "Raw snake_case feature name should be formatted"
    assert "Employee Cost Ratio" in html, "HTML should have formatted feature name"
    print(f"[✓] Test 1 PASSED: Email template generated ({len(html):,} bytes HTML, {len(plain):,} bytes plain)")

    # Test 2: Endpoint dry-run behavior (no SMTP credentials)
    req = EmailAlertRequest(
        recipient_email="pmo-director@enterprise.com",
        project_features=pf,
        prediction_result=pr
    )

    loop = asyncio.get_event_loop()
    resp = loop.run_until_complete(send_email_alert(req))

    assert resp.status == "dry_run", f"Expected dry_run, got {resp.status}"
    assert resp.recipient == "pmo-director@enterprise.com", "Wrong recipient"
    assert len(resp.html_preview) > 500, "HTML preview too short"
    assert "No SMTP server credentials configured" in resp.message
    print("[✓] Test 2 PASSED: Dry-run endpoint returned preview and proper status")

    # Test 3: Failed risk class styling
    pr_failed = dict(pr)
    pr_failed["risk_class"] = "failed"
    subject_f, _, html_f = _build_email_html(pf, pr_failed)
    assert "FAILED" in subject_f, "Subject missing FAILED"
    assert "#EF4444" in html_f, "HTML missing red danger color for FAILED"
    print("[✓] Test 3 PASSED: Dynamic risk coloring for FAILED status")

    # Test 4: On-track risk class styling
    pr_ok = dict(pr)
    pr_ok["risk_class"] = "on_track"
    subject_ok, _, html_ok = _build_email_html(pf, pr_ok)
    assert "ON_TRACK" in subject_ok, "Subject missing ON_TRACK"
    assert "#22C55E" in html_ok, "HTML missing green safe color for ON_TRACK"
    print("[✓] Test 4 PASSED: Dynamic risk coloring for ON_TRACK status")

    print("\n==================================================")
    print("   ALL STAGE 13 EMAIL TESTS PASSED! ✓ (4/4)")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
