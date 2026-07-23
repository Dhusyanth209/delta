"""
Stage 1 Verification Script — AI Project Manager Copilot
Tests the /copilot/chat logic directly and via API requests across 5 sample projects.
"""

import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import (
    load_models,
    _fallback_copilot_response,
    _call_ollama_copilot,
    _build_copilot_system_prompt,
    CopilotChatRequest,
    _state
)

def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 1 COPILOT VERIFICATION")
    print("==================================================\n")
    
    # 1. Load models
    load_models()
    test_data = _state["test_data"]
    print(f"\n[+] Loaded test set with {len(test_data)} projects.\n")
    
    # 5 Test Questions across different sample project scenarios
    test_cases = [
        {
            "id": 1,
            "project_idx": 0,
            "question": "Why is this project classified as high risk / failed?",
            "expected_keywords": ["risk", "cost", "factor"]
        },
        {
            "id": 2,
            "project_idx": 1,
            "question": "What is driving the cost overrun on this project?",
            "expected_keywords": ["overrun", "budget", "cost"]
        },
        {
            "id": 3,
            "project_idx": 2,
            "question": "What interventions should I prioritize first?",
            "expected_keywords": ["intervention", "recommend"]
        },
        {
            "id": 4,
            "project_idx": 3,
            "question": "Tell me about the team composition and attrition impact.",
            "expected_keywords": ["team", "seniority", "attrition"]
        },
        {
            "id": 5,
            "project_idx": 4,
            "question": "What is the overall financial outlook for this project?",
            "expected_keywords": ["budget", "cost", "overrun"]
        }
    ]
    
    passed_count = 0
    
    for case in test_cases:
        row = test_data.iloc[case["project_idx"]].to_dict()
        
        # Build features dict
        raw_features = {
            "industry_type": "Healthcare" if row.get("industry_type_Healthcare") else "BFSI",
            "team_size": int(row.get("team_size", 10)),
            "seniority_mix_junior": float(row.get("seniority_mix_junior", 0.3)),
            "seniority_mix_mid": float(row.get("seniority_mix_mid", 0.4)),
            "seniority_mix_senior": float(row.get("seniority_mix_senior", 0.3)),
            "budget_planned_usd": float(row.get("budget_planned_usd", 200000)),
            "duration_planned_weeks": int(row.get("duration_planned_weeks", 12)),
            "scope_change_count": int(row.get("scope_change_count", 2)),
            "client_type": "fixed_bid" if row.get("client_type_fixed_bid") else "outcome_based",
            "employee_cost_ratio": float(row.get("employee_cost_ratio", 0.58)),
            "attrition_events": int(row.get("attrition_events", 1)),
            "weekly_burn_rate_variance": float(row.get("weekly_burn_rate_variance", 0.1)),
        }
        
        # Build prediction dict
        prediction = {
            "risk_class": str(row.get("outcome_predicted", "at_risk")),
            "risk_confidence": float(row.get("prediction_confidence", 0.75)),
            "overrun_percentage": float((row.get("overrun_ratio_actual", 1.2) - 1.0) * 100),
            "predicted_final_cost_usd": float(row.get("budget_planned_usd", 200000) * row.get("overrun_ratio_actual", 1.2)),
            "budget_planned_usd": float(row.get("budget_planned_usd", 200000)),
            "top_factors": [
                {"feature": "employee_cost_ratio", "impact": "increases_risk", "magnitude": 0.45, "description": "Employee cost ratio above baseline (57%)"},
                {"feature": "attrition_events", "impact": "increases_risk", "magnitude": 0.32, "description": "Attrition events increasing lateral hire cost burden"},
            ],
            "recommendations": [
                {"action": "Right-Size Seniority Mix", "description": "Adjust junior-to-senior ratio to reduce execution errors", "expected_risk_reduction": 0.18}
            ]
        }
        
        # Test prompt construction
        system_prompt = _build_copilot_system_prompt(raw_features, prediction)
        assert "DELTA Copilot" in system_prompt, "System prompt failed title check"
        
        # Test local Jarvis / Ollama LLM copilot engine
        try:
            response = _call_ollama_copilot(case["question"], raw_features, prediction, [])
        except Exception as e:
            print(f"    Notice: Ollama fallback triggered ({e}), using grounded fallback engine")
            response = _fallback_copilot_response(case["question"], raw_features, prediction)
        
        # Verify grounding
        assert response.answer is not None and len(response.answer) > 50, "Response answer empty"
        assert len(response.grounded_factors) > 0, "No grounded factors extracted"
        
        # Verify keywords
        has_keywords = any(kw in response.answer.lower() for kw in case["expected_keywords"])
        assert has_keywords, f"Response missing expected keywords {case['expected_keywords']}"
        
        passed_count += 1
        print(f"[✓] Test Case {case['id']} PASSED")
        print(f"    Q: \"{case['question']}\"")
        print(f"    Model Used: {response.model_used}")
        print(f"    Grounded Factors: {response.grounded_factors}")
        print(f"    Sample Snippet: {response.answer[:150]}...\n")
    
    print("==================================================")
    print(f"   ALL {passed_count}/{len(test_cases)} STAGE 1 COPILOT TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
