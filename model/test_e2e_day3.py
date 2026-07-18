"""Day 3 — Full end-to-end verification of all endpoints and features."""
import json
import requests
import sys

API = "http://localhost:8000"
checks_passed = 0
checks_failed = 0

def check(name, condition):
    global checks_passed, checks_failed
    if condition:
        print(f"  [✓] {name}")
        checks_passed += 1
    else:
        print(f"  [✗] {name}")
        checks_failed += 1

print("=" * 60)
print("DAY 3 — Full End-to-End Verification")
print("=" * 60)

# 1. Health endpoint
print("\n── /health ──")
r = requests.get(f"{API}/health").json()
check("status=healthy", r["status"] == "healthy")
check("models_loaded=true", r["models_loaded"] is True)
check("model_type=XGBClassifier", r["model_type"] == "XGBClassifier")
check("features=29", r["features"] == 29)
check("shap_ready=true", r["shap_ready"] is True)
check("rl_ready=true", r["rl_ready"] is True)
check("version present", "version" in r)

# 2. Metrics endpoint
print("\n── /metrics ──")
r = requests.get(f"{API}/metrics").json()
check("classifier accuracy present", "accuracy" in r["classifier"])
check("model_type=XGBClassifier", r["classifier"]["model_type"] == "XGBClassifier")
check("regressor r2 > 0.7", r["regressor"]["r2"] > 0.7)
check("rl_bandit present", "rl_bandit" in r)
check("version=2.1", r["version"] == "2.1")
check("29 features", r["dataset"]["features"] == 29)
check("8 engineered", r["dataset"]["engineered_features"] == 8)

# 3. Sample projects
print("\n── /projects/sample ──")
r = requests.get(f"{API}/projects/sample").json()
check("has projects", len(r["projects"]) > 0)
check("total >= 5", r["total"] >= 5)
proj = r["projects"][0]
check("has features", "features" in proj)
check("has prediction", "prediction" in proj)
check("has risk_class", "risk_class" in proj["prediction"])
check("has INR cost", "predicted_final_cost_inr" in proj["prediction"])

# 4. Predict with USD
print("\n── /predict (BFSI fixed_bid) ──")
features = {
    "industry_type": "BFSI", "team_size": 20,
    "seniority_mix_junior": 0.35, "seniority_mix_mid": 0.40, "seniority_mix_senior": 0.25,
    "budget_planned_usd": 400000, "duration_planned_weeks": 20,
    "scope_change_count": 5, "client_type": "fixed_bid",
    "employee_cost_ratio": 0.58, "attrition_events": 2,
    "weekly_burn_rate_variance": 0.12,
}
r = requests.post(f"{API}/predict", json=features).json()
check("risk_class valid", r["risk_class"] in ["on_track", "at_risk", "failed"])
check("confidence 0-1", 0 <= r["risk_confidence"] <= 1)
check("has USD cost", r["predicted_final_cost_usd"] > 0)
check("has INR cost", r["predicted_final_cost_inr"] > 0)
check("INR = USD * 83.5", abs(r["predicted_final_cost_inr"] - r["predicted_final_cost_usd"] * 83.5) < 1)
check("has overrun_percentage", "overrun_percentage" in r)
check("has top_factors (3)", len(r["top_factors"]) == 3)
check("has class_probabilities", len(r["class_probabilities"]) == 3)
check("probabilities sum ~1", abs(sum(r["class_probabilities"].values()) - 1.0) < 0.01)
check("has recommendations", len(r.get("recommendations", [])) > 0)

# SHAP factor quality
f0 = r["top_factors"][0]
check("factor has feature name", len(f0["feature"]) > 0)
check("factor has impact", f0["impact"] in ["increases_risk", "reduces_risk"])
check("factor has description", len(f0["description"]) > 5)

# RL recommendation quality
rec0 = r["recommendations"][0]
check("rec has action name", len(rec0["action"]) > 0)
check("rec has description", len(rec0["description"]) > 5)
check("rec reduction >= 0", rec0["expected_risk_reduction"] >= 0)

# 5. Predict with different industry
print("\n── /predict (Healthcare T&M) ──")
features2 = {
    "industry_type": "Healthcare", "team_size": 10,
    "seniority_mix_junior": 0.20, "seniority_mix_mid": 0.50, "seniority_mix_senior": 0.30,
    "budget_planned_usd": 200000, "duration_planned_weeks": 12,
    "scope_change_count": 1, "client_type": "time_and_material",
    "employee_cost_ratio": 0.50, "attrition_events": 0,
    "weekly_burn_rate_variance": 0.05,
}
r2 = requests.post(f"{API}/predict", json=features2).json()
check("different project works", r2["risk_class"] in ["on_track", "at_risk", "failed"])
check("lower risk for safe project", r2["risk_confidence"] > 0)

print(f"\n{'=' * 60}")
print(f"RESULTS: {checks_passed} passed, {checks_failed} failed")
if checks_failed == 0:
    print("✓ ALL CHECKS PASSED — System verified end-to-end")
else:
    print(f"✗ {checks_failed} CHECKS FAILED")
print(f"{'=' * 60}")
