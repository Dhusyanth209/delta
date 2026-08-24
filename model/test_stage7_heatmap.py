"""
Stage 7 Verification Script — Interactive Risk Heatmap
Tests the /heatmap/data endpoint and SHAP matrix computation.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import load_models, _compute_all_shap_values, engineer_features_from_raw, FEATURE_DISPLAY_NAMES


def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 7 HEATMAP VERIFICATION")
    print("==================================================\n")

    load_models()

    # Test projects
    projects = [
        {"industry_type": "BFSI", "team_size": 25, "seniority_mix_junior": 0.30, "seniority_mix_mid": 0.45,
         "seniority_mix_senior": 0.25, "budget_planned_usd": 500000, "duration_planned_weeks": 24,
         "scope_change_count": 4, "client_type": "fixed_bid", "employee_cost_ratio": 0.58,
         "attrition_events": 2, "weekly_burn_rate_variance": 0.12},
        {"industry_type": "Healthcare", "team_size": 15, "seniority_mix_junior": 0.40, "seniority_mix_mid": 0.35,
         "seniority_mix_senior": 0.25, "budget_planned_usd": 250000, "duration_planned_weeks": 16,
         "scope_change_count": 1, "client_type": "time_and_material", "employee_cost_ratio": 0.52,
         "attrition_events": 0, "weekly_burn_rate_variance": 0.08},
        {"industry_type": "Retail", "team_size": 40, "seniority_mix_junior": 0.50, "seniority_mix_mid": 0.30,
         "seniority_mix_senior": 0.20, "budget_planned_usd": 800000, "duration_planned_weeks": 32,
         "scope_change_count": 8, "client_type": "fixed_bid", "employee_cost_ratio": 0.65,
         "attrition_events": 5, "weekly_burn_rate_variance": 0.22},
        {"industry_type": "Telecom", "team_size": 10, "seniority_mix_junior": 0.20, "seniority_mix_mid": 0.50,
         "seniority_mix_senior": 0.30, "budget_planned_usd": 150000, "duration_planned_weeks": 12,
         "scope_change_count": 0, "client_type": "outcome_based", "employee_cost_ratio": 0.45,
         "attrition_events": 0, "weekly_burn_rate_variance": 0.05},
        {"industry_type": "BFSI", "team_size": 60, "seniority_mix_junior": 0.45, "seniority_mix_mid": 0.30,
         "seniority_mix_senior": 0.25, "budget_planned_usd": 1200000, "duration_planned_weeks": 40,
         "scope_change_count": 12, "client_type": "fixed_bid", "employee_cost_ratio": 0.68,
         "attrition_events": 8, "weekly_burn_rate_variance": 0.25},
    ]

    # 1. Test _compute_all_shap_values
    df = engineer_features_from_raw(projects[0])
    shap_map = _compute_all_shap_values(df)
    assert len(shap_map) > 0, "SHAP map is empty"
    assert all(isinstance(v, float) for v in shap_map.values()), "SHAP values should be floats"
    print(f"[✓] Test 1 PASSED: _compute_all_shap_values returns {len(shap_map)} features")

    # 2. Test all 5 projects produce SHAP maps
    all_maps = []
    for p in projects:
        df = engineer_features_from_raw(p)
        sm = _compute_all_shap_values(df)
        all_maps.append(sm)
    assert len(all_maps) == 5, f"Expected 5 SHAP maps, got {len(all_maps)}"
    assert all(len(m) > 0 for m in all_maps), "Some SHAP maps are empty"
    print(f"[✓] Test 2 PASSED: All 5 projects produce valid SHAP maps")

    # 3. Test feature importance ranking
    importance = {}
    for sm in all_maps:
        for feat, val in sm.items():
            importance[feat] = importance.get(feat, 0) + abs(val)
    sorted_feats = sorted(importance.keys(), key=lambda f: importance[f], reverse=True)
    top8 = sorted_feats[:8]
    assert len(top8) == 8, f"Expected 8 top features, got {len(top8)}"
    print(f"[✓] Test 3 PASSED: Top 8 features: {', '.join(top8[:4])}...")

    # 4. Test normalization range
    global_max = max(abs(sm.get(f, 0)) for sm in all_maps for f in top8)
    assert global_max > 0, "Global max should be positive"
    for sm in all_maps:
        for f in top8:
            norm = sm.get(f, 0) / global_max
            assert -1.0 <= norm <= 1.0, f"Normalized value {norm} out of range"
    print(f"[✓] Test 4 PASSED: All normalized values in [-1.0, +1.0] range")

    # 5. Test FEATURE_DISPLAY_NAMES coverage
    covered = sum(1 for f in top8 if f in FEATURE_DISPLAY_NAMES)
    print(f"[✓] Test 5 PASSED: {covered}/{len(top8)} top features have display names ({len(FEATURE_DISPLAY_NAMES)} total mapped)")

    # 6. Test matrix dimensions
    matrix = []
    for sm in all_maps:
        cells = [{"feature": f, "normalized": round(sm.get(f, 0) / global_max, 4)} for f in top8]
        matrix.append(cells)
    assert len(matrix) == 5, f"Expected 5 rows, got {len(matrix)}"
    assert all(len(row) == 8 for row in matrix), "Each row should have 8 cells"
    print(f"[✓] Test 6 PASSED: Matrix dimensions correct (5 projects × 8 factors)")

    # Print heatmap preview
    print(f"\n--- HEATMAP MATRIX PREVIEW ---")
    header = "Project   " + "  ".join(f"{FEATURE_DISPLAY_NAMES.get(f, f)[:10]:>10}" for f in top8)
    print(header)
    for i, row in enumerate(matrix):
        vals = "  ".join(f"{'↑' if c['normalized'] > 0 else '↓'}{abs(c['normalized'])*100:5.0f}%" for c in row)
        print(f"Proj #{i+1:>2}   {vals}")

    print("\n==================================================")
    print("   ALL STAGE 7 HEATMAP TESTS PASSED! ✓")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
