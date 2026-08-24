"""Test the copilot endpoint through the running backend server."""
import urllib.request
import json
import time

payload = {
    "question": "Why is this project at risk?",
    "project_features": {
        "industry_type": "BFSI",
        "team_size": 20,
        "seniority_mix_junior": 0.4,
        "seniority_mix_mid": 0.35,
        "seniority_mix_senior": 0.25,
        "budget_planned_usd": 500000,
        "duration_planned_weeks": 24,
        "scope_change_count": 5,
        "client_type": "fixed_bid",
        "employee_cost_ratio": 0.65,
        "attrition_events": 3,
        "weekly_burn_rate_variance": 0.15,
    },
    "prediction_result": {
        "risk_class": "at_risk",
        "risk_confidence": 0.82,
        "overrun_percentage": 28.5,
        "predicted_final_cost_usd": 642500,
        "budget_planned_usd": 500000,
        "top_factors": [
            {"feature": "employee_cost_ratio", "impact": "increases_risk", "magnitude": 0.45, "description": "Employee cost ratio 65% is above the 57% industry baseline"},
            {"feature": "attrition_events", "impact": "increases_risk", "magnitude": 0.32, "description": "3 attrition events increase lateral hire costs"},
        ],
        "recommendations": [
            {"action": "Reduce Attrition", "description": "Implement retention programs", "expected_risk_reduction": 0.2},
        ],
    },
    "chat_history": [],
}

start = time.time()
req = urllib.request.Request(
    "http://localhost:8000/copilot/chat",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(req, timeout=90) as resp:
    data = json.loads(resp.read().decode("utf-8"))

elapsed = time.time() - start
print(f"TIME: {elapsed:.1f}s")
print(f"MODEL: {data['model_used']}")
print(f"FACTORS: {data['grounded_factors']}")
print(f"ANSWER:\n{data['answer']}")
