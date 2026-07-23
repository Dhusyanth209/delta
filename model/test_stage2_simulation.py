"""
Stage 2 Verification Script — What-If Simulation
Tests the /simulate endpoint logic directly across multiple parameter adjustments.
"""

import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import (
    load_models,
    SimulationRequest,
    simulate_scenario,
    ProjectFeatures
)

import asyncio

async def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 2 SIMULATION VERIFICATION")
    print("==================================================\n")
    
    # 1. Load models
    load_models()
    
    # Baseline project setup
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
    
    test_scenarios = [
        {
            "id": 1,
            "name": "Add 2 team members",
            "req": SimulationRequest(baseline_features=baseline, team_size_delta=2)
        },
        {
            "id": 2,
            "name": "Reduce scope changes by 2",
            "req": SimulationRequest(baseline_features=baseline, scope_change_delta=-2)
        },
        {
            "id": 3,
            "name": "Switch to Outcome-Based contract",
            "req": SimulationRequest(baseline_features=baseline, client_type="outcome_based")
        },
        {
            "id": 4,
            "name": "Combined Optimization (team +2, scope -2)",
            "req": SimulationRequest(baseline_features=baseline, team_size_delta=2, scope_change_delta=-2)
        }
    ]
    
    passed_count = 0
    
    for case in test_scenarios:
        res = await simulate_scenario(case["req"])
        
        assert "baseline_prediction" in res.model_dump(), "Missing baseline prediction"
        assert "simulated_prediction" in res.model_dump(), "Missing simulated prediction"
        assert "delta" in res.model_dump(), "Missing delta summary"
        
        delta = res.delta
        print(f"[✓] Test Case {case['id']} PASSED: {case['name']}")
        print(f"    Baseline Risk: {delta['baseline_risk']} ➔ Simulated Risk: {delta['simulated_risk']}")
        print(f"    Cost Delta (USD): ${delta['cost_diff_usd']:+,.2f}")
        print(f"    Overrun Delta (%): {delta['overrun_diff_pct']:+.2f}%\n")
        
        passed_count += 1
        
    print("==================================================")
    print(f"   ALL {passed_count}/{len(test_scenarios)} STAGE 2 SIMULATION TESTS PASSED!")
    print("==================================================")

if __name__ == "__main__":
    asyncio.run(run_tests())
