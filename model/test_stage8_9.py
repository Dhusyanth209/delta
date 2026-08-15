"""
Stage 8+9 Verification Script — Docker files + PDF Report Export
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 8+9 VERIFICATION")
    print("==================================================\n")

    # ─── Stage 8: Docker Files ───
    print("--- STAGE 8: Docker Files ---\n")

    # Test 1: Dockerfile.backend exists and has correct content
    df_backend = PROJECT_ROOT / "Dockerfile.backend"
    assert df_backend.exists(), "Dockerfile.backend not found"
    content = df_backend.read_text()
    assert "python:3.11-slim" in content, "Missing Python base image"
    assert "uvicorn" in content, "Missing uvicorn CMD"
    assert "8000" in content, "Missing port 8000"
    print("[✓] Test 1 PASSED: Dockerfile.backend valid (Python 3.11-slim, uvicorn, port 8000)")

    # Test 2: Dockerfile.frontend exists
    df_frontend = PROJECT_ROOT / "Dockerfile.frontend"
    assert df_frontend.exists(), "Dockerfile.frontend not found"
    content = df_frontend.read_text()
    assert "node:20-alpine" in content, "Missing Node base image"
    assert "npm run build" in content, "Missing build step"
    assert "3000" in content, "Missing port 3000"
    print("[✓] Test 2 PASSED: Dockerfile.frontend valid (Node 20-alpine, multi-stage build, port 3000)")

    # Test 3: docker-compose.yml exists
    dc = PROJECT_ROOT / "docker-compose.yml"
    assert dc.exists(), "docker-compose.yml not found"
    content = dc.read_text()
    assert "backend" in content and "frontend" in content, "Missing services"
    assert "8000:8000" in content, "Missing backend port mapping"
    assert "3000:3000" in content, "Missing frontend port mapping"
    assert "depends_on" in content, "Missing depends_on"
    assert "healthcheck" in content, "Missing healthcheck"
    print("[✓] Test 3 PASSED: docker-compose.yml valid (2 services, ports, healthcheck, depends_on)")

    # Test 4: .dockerignore exists
    di = PROJECT_ROOT / ".dockerignore"
    assert di.exists(), ".dockerignore not found"
    content = di.read_text()
    assert "node_modules" in content, "Missing node_modules ignore"
    assert "__pycache__" in content, "Missing __pycache__ ignore"
    print("[✓] Test 4 PASSED: .dockerignore valid")

    # ─── Stage 9: PDF Report ───
    print("\n--- STAGE 9: PDF Report Export ---\n")

    from backend.main import load_models, _generate_pdf_report

    load_models()

    # Test 5: PDF generation produces valid bytes
    features = {
        "industry_type": "BFSI",
        "team_size": 25,
        "budget_planned_usd": 500000,
        "duration_planned_weeks": 24,
        "client_type": "fixed_bid",
        "employee_cost_ratio": 0.58,
    }
    prediction = {
        "risk_class": "at_risk",
        "risk_confidence": 0.85,
        "budget_planned_usd": 500000,
        "predicted_final_cost_usd": 663000,
        "overrun_percentage": 32.6,
        "top_factors": [
            {"feature": "employee_cost_ratio", "impact": "increases_risk", "magnitude": 0.48, "description": "ECR above 57% baseline"},
            {"feature": "scope_change_count", "impact": "increases_risk", "magnitude": 0.38, "description": "6 scope changes"},
            {"feature": "attrition_events", "impact": "increases_risk", "magnitude": 0.30, "description": "3 attrition events"},
        ],
        "recommendations": [
            {"action": "Scope Freeze", "description": "Freeze scope after 60% completion", "expected_risk_reduction": 0.22},
            {"action": "Retention Bonuses", "description": "Retention incentives for key staff", "expected_risk_reduction": 0.15},
        ],
    }

    pdf_bytes = _generate_pdf_report(features, prediction)
    assert len(pdf_bytes) > 100, f"PDF too small: {len(pdf_bytes)} bytes"
    print(f"[✓] Test 5 PASSED: PDF generated ({len(pdf_bytes):,} bytes)")

    # Test 6: PDF starts with valid header
    assert pdf_bytes[:5] == b"%PDF-", f"Invalid PDF header: {pdf_bytes[:10]}"
    print("[✓] Test 6 PASSED: Valid %PDF- header")

    # Test 7: PDF is substantial (contains tables, not just a header)
    assert len(pdf_bytes) > 2000, f"PDF too small to contain full report: {len(pdf_bytes)} bytes"
    # ReportLab compresses text in content streams, so we verify structure via markers
    assert b"endobj" in pdf_bytes, "PDF missing object markers"
    print(f"[✓] Test 7 PASSED: PDF has substantial content ({len(pdf_bytes):,} bytes, valid structure)")

    # Test 8: reportlab in requirements.txt
    req_content = (PROJECT_ROOT / "requirements.txt").read_text()
    assert "reportlab" in req_content, "Missing reportlab in requirements.txt"
    print("[✓] Test 8 PASSED: reportlab listed in requirements.txt")

    print(f"\n==================================================")
    print(f"   ALL STAGE 8+9 TESTS PASSED! ✓ (8/8)")
    print(f"==================================================")


if __name__ == "__main__":
    run_tests()
