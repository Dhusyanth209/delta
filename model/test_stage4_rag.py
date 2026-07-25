"""
Stage 4 Verification Script — RAG Knowledge Base over PMBOK / IT Governance
Tests the /rag/query endpoint and RAG-enriched Copilot system prompt.
"""

import sys
from pathlib import Path
import json

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import load_models, _rag_retrieve, _build_copilot_system_prompt, _state


def run_tests():
    print("==================================================")
    print("   DELTA 2.0 — STAGE 4 RAG VERIFICATION")
    print("==================================================\n")

    # 1. Load models + RAG knowledge base
    load_models()

    assert _state["rag_knowledge_base"] is not None, "RAG knowledge base not loaded"
    assert len(_state["rag_knowledge_base"]) == 10, f"Expected 10 KB entries, got {len(_state['rag_knowledge_base'])}"
    assert _state["rag_tfidf_index"] is not None, "TF-IDF index not built"
    print(f"[✓] Test 1 PASSED: RAG knowledge base loaded ({len(_state['rag_knowledge_base'])} entries, TF-IDF indexed)\n")

    # 2. Test RAG retrieval — cost overrun query
    results = _rag_retrieve("How do I handle cost overrun in my project budget?", top_k=3)
    assert len(results) > 0, "No results returned for cost overrun query"
    categories = [r["category"] for r in results]
    assert any("Cost" in c for c in categories), f"Expected Cost-related results, got categories: {categories}"
    print(f"[✓] Test 2 PASSED: Cost overrun query returned {len(results)} results")
    for r in results:
        print(f"    → [{r['category']}] {r['title']} (score: {r['relevance_score']})")
    print()

    # 3. Test RAG retrieval — attrition query
    results = _rag_retrieve("team attrition and knowledge retention", top_k=3)
    assert len(results) > 0, "No results returned for attrition query"
    ids = [r["id"] for r in results]
    assert any("resource" in i or "attrition" in i for i in ids), f"Expected resource/attrition results, got: {ids}"
    print(f"[✓] Test 3 PASSED: Attrition query returned {len(results)} results")
    for r in results:
        print(f"    → [{r['category']}] {r['title']} (score: {r['relevance_score']})")
    print()

    # 4. Test RAG retrieval — scope creep query
    results = _rag_retrieve("scope change control process", top_k=2)
    assert len(results) > 0, "No results returned for scope query"
    print(f"[✓] Test 4 PASSED: Scope query returned {len(results)} results")
    for r in results:
        print(f"    → [{r['category']}] {r['title']} (score: {r['relevance_score']})")
    print()

    # 5. Test RAG retrieval — contract risk query
    results = _rag_retrieve("fixed bid vs outcome based contract risk", top_k=2)
    assert len(results) > 0, "No results returned for contract query"
    print(f"[✓] Test 5 PASSED: Contract risk query returned {len(results)} results")
    for r in results:
        print(f"    → [{r['category']}] {r['title']} (score: {r['relevance_score']})")
    print()

    # 6. Test RAG-enriched Copilot system prompt
    test_features = {
        "industry_type": "BFSI",
        "team_size": 20,
        "seniority_mix_junior": 0.35,
        "seniority_mix_mid": 0.40,
        "seniority_mix_senior": 0.25,
        "budget_planned_usd": 400000,
        "duration_planned_weeks": 20,
        "scope_change_count": 4,
        "client_type": "fixed_bid",
        "employee_cost_ratio": 0.60,
        "attrition_events": 3,
        "weekly_burn_rate_variance": 0.12,
    }
    test_prediction = {
        "risk_class": "at_risk",
        "risk_confidence": 0.78,
        "overrun_percentage": 22.5,
        "predicted_final_cost_usd": 490000,
        "budget_planned_usd": 400000,
        "top_factors": [
            {"feature": "attrition_events", "impact": "increases_risk", "magnitude": 0.4, "description": "3 attrition events"},
            {"feature": "employee_cost_ratio", "impact": "increases_risk", "magnitude": 0.35, "description": "ECR at 60%"},
        ],
        "recommendations": [
            {"action": "Retention Program", "description": "Implement retention bonuses", "expected_risk_reduction": 0.15}
        ],
    }

    prompt = _build_copilot_system_prompt(test_features, test_prediction, query_text="How should I handle attrition per PMBOK?")
    assert "PMBOK / IT GOVERNANCE REFERENCE STANDARDS" in prompt, "RAG context not injected into Copilot prompt"
    assert "PMBOK 7th Edition" in prompt, "PMBOK source citation not found in prompt"
    print(f"[✓] Test 6 PASSED: Copilot system prompt contains RAG-retrieved PMBOK citations")
    
    # Print a snippet of the injected RAG section
    rag_start = prompt.index("PMBOK / IT GOVERNANCE")
    print(f"\n--- RAG INJECTION SNIPPET ---")
    print(prompt[rag_start:rag_start + 300])
    
    print("\n==================================================")
    print("   ALL STAGE 4 RAG TESTS PASSED! ✓")
    print("==================================================")


if __name__ == "__main__":
    run_tests()
