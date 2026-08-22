"""
DELTA — FastAPI Backend v2.0
==============================
Performance-optimized backend with:
1. Pre-computed SHAP explainer cached at startup (not per-request)
2. Feature engineering pipeline matching train_model_v2
3. RL intervention recommendations endpoint
4. LRU caching for repeated predictions
5. Proper async with background SHAP computation
6. Model metrics endpoint
7. Lifespan events (replaces deprecated on_event)
8. RAG knowledge base over PMBOK / IT Governance Standards

Endpoints:
  POST /predict             → Risk class + cost + SHAP factors + RL recommendations
  GET  /projects/sample     → Sample projects with predictions
  GET  /health              → Health check + model info
  GET  /metrics             → Model training metrics
  GET  /recommend/{idx}     → RL intervention recommendations for sample project
  POST /rag/query           → Semantic retrieval from PMBOK / IT Governance knowledge base
"""

from typing import Optional, Dict, Any, List
import json
import io
import os
import time
import hashlib
from functools import lru_cache
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import re
import math
from collections import Counter
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Path Setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"

# ─── Global State ────────────────────────────────────────────────────────────
_state = {
    "classifier": None,
    "regressor": None,
    "label_encoder": None,
    "feature_columns": None,
    "test_data": None,
    "shap_explainer": None,
    "rl_bandit": None,
    "metrics": None,
    "startup_time": None,
    "prediction_count": 0,
    "rag_knowledge_base": None,
    "rag_tfidf_index": None,
}

USD_TO_INR = 83.5
DATA_DIR = PROJECT_ROOT / "data"


# ─── RAG TF-IDF Indexer ─────────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into word tokens."""
    return re.findall(r'[a-z0-9]+', text.lower())


def _build_tfidf_index(docs: list[dict]) -> dict:
    """
    Build a lightweight TF-IDF index over the knowledge base entries.
    Each doc dict must have 'content' and 'keywords' fields.
    Returns index dict with term frequencies and IDF values.
    """
    N = len(docs)
    # Combine content + keywords for each document
    doc_tokens = []
    for doc in docs:
        combined = doc.get("content", "") + " " + " ".join(doc.get("keywords", []))
        tokens = _tokenize(combined)
        doc_tokens.append(tokens)

    # Document frequency for each term
    df = Counter()
    for tokens in doc_tokens:
        unique_tokens = set(tokens)
        for t in unique_tokens:
            df[t] += 1

    # IDF = log(N / df)
    idf = {}
    for term, freq in df.items():
        idf[term] = math.log((N + 1) / (freq + 1)) + 1.0  # smoothed IDF

    # TF-IDF vectors per doc (sparse dict representation)
    doc_vectors = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        total = len(tokens) if tokens else 1
        vec = {}
        for term, count in tf.items():
            vec[term] = (count / total) * idf.get(term, 1.0)
        # Normalize
        norm = math.sqrt(sum(v ** 2 for v in vec.values())) or 1.0
        vec = {k: v / norm for k, v in vec.items()}
        doc_vectors.append(vec)

    return {"idf": idf, "doc_vectors": doc_vectors}


def _rag_retrieve(query: str, top_k: int = 3) -> list[dict]:
    """
    Retrieve top-K PMBOK knowledge base entries matching a query.
    Uses TF-IDF cosine similarity.
    """
    kb = _state["rag_knowledge_base"]
    index = _state["rag_tfidf_index"]
    if not kb or not index:
        return []

    query_tokens = _tokenize(query)
    idf = index["idf"]
    doc_vectors = index["doc_vectors"]

    # Build query vector
    tf = Counter(query_tokens)
    total = len(query_tokens) if query_tokens else 1
    q_vec = {}
    for term, count in tf.items():
        q_vec[term] = (count / total) * idf.get(term, 1.0)
    q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values())) or 1.0
    q_vec = {k: v / q_norm for k, v in q_vec.items()}

    # Cosine similarity with each doc
    scores = []
    for i, dv in enumerate(doc_vectors):
        dot = sum(q_vec.get(t, 0.0) * dv.get(t, 0.0) for t in q_vec)
        scores.append((i, dot))

    scores.sort(key=lambda x: x[1], reverse=True)

    results = []
    for idx, score in scores[:top_k]:
        if score > 0.01:  # minimum relevance threshold
            entry = kb[idx].copy()
            entry["relevance_score"] = round(score, 4)
            results.append(entry)

    return results

# ─── Feature Engineering ────────────────────────────────────────────────────

def engineer_features_from_raw(raw: dict) -> pd.DataFrame:
    """
    Apply the same feature engineering as train_model_v2.
    Takes raw feature dict, returns encoded DataFrame ready for prediction.
    """
    df = pd.DataFrame([raw])
    
    # Engineered features
    df["scope_fixed_bid_pressure"] = df["scope_change_count"] * (df["client_type"] == "fixed_bid").astype(int)
    df["attrition_cost_burden"] = df["attrition_events"] * df["employee_cost_ratio"] * 0.275
    df["budget_per_person_week"] = df["budget_planned_usd"] / (df["team_size"] * df["duration_planned_weeks"] + 1)
    df["junior_heavy"] = ((df["seniority_mix_junior"] > 0.40) & (df["seniority_mix_senior"] < 0.25)).astype(int)
    df["burn_instability"] = df["weekly_burn_rate_variance"] * df["duration_planned_weeks"]
    df["ecr_above_baseline"] = np.maximum(0, df["employee_cost_ratio"] - 0.57)
    df["scope_intensity"] = df["scope_change_count"] / (df["duration_planned_weeks"] + 1)
    df["attrition_rate"] = df["attrition_events"] / (df["team_size"] + 1)
    
    # One-hot encode
    df_encoded = pd.get_dummies(df, columns=["industry_type", "client_type"])
    
    # Align columns with training features
    for col in _state["feature_columns"]:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[_state["feature_columns"]]
    
    return df_encoded


# ─── Prediction Cache ───────────────────────────────────────────────────────

_prediction_cache = {}

def cache_key(raw: dict) -> str:
    """Create a hash key from input features."""
    return hashlib.md5(json.dumps(raw, sort_keys=True).encode()).hexdigest()


# ─── Model Loading ──────────────────────────────────────────────────────────

def load_models():
    """Load all model artifacts at startup."""
    print("Loading model artifacts...")
    t0 = time.time()
    
    _state["classifier"] = joblib.load(ARTIFACTS_DIR / "xgb_classifier.joblib")
    _state["regressor"] = joblib.load(ARTIFACTS_DIR / "cost_regressor.joblib")
    _state["label_encoder"] = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    _state["feature_columns"] = joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")
    _state["test_data"] = pd.read_csv(ARTIFACTS_DIR / "test_set_with_predictions.csv")
    
    print(f"  ✓ Classifier loaded ({type(_state['classifier']).__name__})")
    print(f"  ✓ Regressor loaded ({type(_state['regressor']).__name__})")
    print(f"  ✓ Label encoder: {list(_state['label_encoder'].classes_)}")
    print(f"  ✓ Feature columns: {len(_state['feature_columns'])}")
    print(f"  ✓ Test data: {len(_state['test_data'])} rows")
    
    # Pre-compute SHAP explainer (expensive but only done once)
    try:
        import shap
        model = _state["classifier"]
        # For ensemble models, extract XGBoost component
        if hasattr(model, 'named_estimators_'):
            for name, est in model.named_estimators_.items():
                if 'xgb' in name.lower() or 'XGB' in type(est).__name__:
                    model = est
                    break
        _state["shap_explainer"] = shap.TreeExplainer(model)
        print("  ✓ SHAP explainer pre-computed")
    except Exception as e:
        print(f"  ⚠ SHAP explainer failed: {e}")
        _state["shap_explainer"] = None
    
    # Load RL bandit if available
    bandit_path = ARTIFACTS_DIR / "rl_bandit.json"
    if bandit_path.exists():
        with open(bandit_path) as f:
            _state["rl_bandit"] = json.load(f)
        print("  ✓ RL bandit loaded")
    
    # Load metrics
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            _state["metrics"] = json.load(f)
        print("  ✓ Metrics loaded")
    
    # Load RAG knowledge base
    kb_path = DATA_DIR / "pm_knowledge_base.json"
    if kb_path.exists():
        with open(kb_path, encoding="utf-8") as f:
            _state["rag_knowledge_base"] = json.load(f)
        _state["rag_tfidf_index"] = _build_tfidf_index(_state["rag_knowledge_base"])
        print(f"  ✓ RAG knowledge base loaded ({len(_state['rag_knowledge_base'])} entries, TF-IDF indexed)")
    else:
        print("  ⚠ RAG knowledge base not found, /rag/query will return empty results")

    _state["startup_time"] = time.time()
    print(f"  ✓ All artifacts loaded in {time.time() - t0:.2f}s")


# ─── App Setup ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle using modern lifespan pattern."""
    load_models()
    yield
    # Cleanup on shutdown
    _prediction_cache.clear()

app = FastAPI(
    title="DELTA API",
    description="Project Cost-Overrun & Delivery-Risk Prediction — v2.0",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ────────────────────────────────────────────────────────

class ProjectFeatures(BaseModel):
    industry_type: str = Field(..., description="BFSI, Healthcare, Retail, etc.")
    team_size: int = Field(..., ge=1, le=200)
    seniority_mix_junior: float = Field(..., ge=0, le=1)
    seniority_mix_mid: float = Field(..., ge=0, le=1)
    seniority_mix_senior: float = Field(..., ge=0, le=1)
    budget_planned_usd: float = Field(..., gt=0)
    duration_planned_weeks: int = Field(..., ge=1, le=104)
    scope_change_count: int = Field(..., ge=0, le=50)
    client_type: str = Field(..., description="fixed_bid, outcome_based, time_and_material")
    employee_cost_ratio: float = Field(..., ge=0, le=1)
    attrition_events: int = Field(..., ge=0)
    weekly_burn_rate_variance: float = Field(..., ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "industry_type": "BFSI",
                "team_size": 25,
                "seniority_mix_junior": 0.30,
                "seniority_mix_mid": 0.45,
                "seniority_mix_senior": 0.25,
                "budget_planned_usd": 500000,
                "duration_planned_weeks": 24,
                "scope_change_count": 4,
                "client_type": "fixed_bid",
                "employee_cost_ratio": 0.58,
                "attrition_events": 2,
                "weekly_burn_rate_variance": 0.12,
            }
        }


class ShapFactor(BaseModel):
    feature: str
    impact: str
    magnitude: float
    description: str


class Recommendation(BaseModel):
    action: str
    description: str
    expected_risk_reduction: float
    confidence: float


class PredictionResponse(BaseModel):
    risk_class: str
    risk_confidence: float
    predicted_overrun_ratio: float
    predicted_final_cost_usd: float
    predicted_final_cost_inr: float
    budget_planned_usd: float
    budget_planned_inr: float
    overrun_percentage: float
    top_factors: list[ShapFactor]
    class_probabilities: dict[str, float]
    recommendations: list[Recommendation] = []


# ─── SHAP Factor Descriptions ───────────────────────────────────────────────

FACTOR_DESCRIPTIONS = {
    "scope_change_count": {
        "high": "High number of scope changes is pushing this project's risk up",
        "low": "Low scope creep is helping keep this project on track",
    },
    "employee_cost_ratio": {
        "high": "Employee cost ratio above industry baseline (57%) is squeezing margins",
        "low": "Employee costs are well-managed relative to the budget",
    },
    "attrition_events": {
        "high": "Team member departures are increasing cost (25-30% lateral-hire premium) and slowing delivery",
        "low": "Stable team composition is supporting steady delivery",
    },
    "weekly_burn_rate_variance": {
        "high": "Unstable weekly spending pattern suggests poor project control",
        "low": "Consistent burn rate indicates disciplined project execution",
    },
    "team_size": {
        "high": "Large team adds coordination overhead and communication complexity",
        "low": "Small team may be under-resourced for this project's scope",
    },
    "budget_planned_usd": {
        "high": "Large budget increases the stakes and complexity of delivery",
        "low": "Smaller budget keeps the risk exposure contained",
    },
    "duration_planned_weeks": {
        "high": "Long project timeline increases exposure to scope creep and attrition",
        "low": "Short timeline limits risk exposure but may add schedule pressure",
    },
    "seniority_mix_junior": {
        "high": "Junior-heavy team composition increases execution risk",
        "low": "Experienced team composition supports reliable delivery",
    },
    "seniority_mix_senior": {
        "high": "Senior-heavy team drives higher labor costs but better execution",
        "low": "Fewer senior members may limit technical decision-making quality",
    },
    # Engineered features
    "scope_fixed_bid_pressure": {
        "high": "Scope changes on a fixed-bid contract directly erode margins",
        "low": "Low scope-contract pressure is favorable",
    },
    "attrition_cost_burden": {
        "high": "Cumulative attrition costs (lateral-hire premiums) are significant",
        "low": "Minimal attrition-driven cost burden",
    },
    "budget_per_person_week": {
        "high": "Generous per-person budget allows quality staffing",
        "low": "Low per-person budget suggests potential under-resourcing",
    },
    "junior_heavy": {
        "high": "Team is junior-heavy with insufficient senior oversight",
        "low": "Team has adequate senior representation",
    },
    "burn_instability": {
        "high": "Burn-rate volatility amplified by project length",
        "low": "Spending stability over the project duration",
    },
    "ecr_above_baseline": {
        "high": "Employee costs exceed the 57% industry baseline",
        "low": "Employee costs are at or below industry baseline",
    },
    "scope_intensity": {
        "high": "Frequent scope changes relative to project duration",
        "low": "Low rate of scope changes per week",
    },
    "attrition_rate": {
        "high": "High attrition rate relative to team size",
        "low": "Low attrition relative to team size",
    },
    # One-hot categories
    "client_type_fixed_bid": {
        "high": "Fixed-bid contract absorbs scope creep as direct margin loss",
        "low": "Not a fixed-bid contract, reducing scope-change risk",
    },
    "client_type_outcome_based": {
        "high": "Outcome-based pricing raises the stakes of any overrun",
        "low": "Not outcome-based, reducing performance-risk pressure",
    },
    "client_type_time_and_material": {
        "high": "T&M contract provides flexibility to bill for additional scope",
        "low": "Not a T&M contract, reducing billing flexibility",
    },
    "industry_type_BFSI": {
        "high": "BFSI projects have strict regulatory requirements adding complexity",
        "low": "Not in BFSI, avoiding regulatory compliance overhead",
    },
    "industry_type_Government": {
        "high": "Government projects often face procurement and approval delays",
        "low": "Not a government project, avoiding bureaucratic overhead",
    },
    "industry_type_Healthcare": {
        "high": "Healthcare projects require compliance with data protection standards",
        "low": "Not in Healthcare, fewer compliance constraints",
    },
}


def get_factor_description(feature_name: str, shap_value: float) -> str:
    direction = "high" if shap_value > 0 else "low"
    if feature_name in FACTOR_DESCRIPTIONS:
        return FACTOR_DESCRIPTIONS[feature_name][direction]
    clean_name = feature_name.replace("_", " ")
    if shap_value > 0:
        return f"'{clean_name}' is contributing to elevated risk"
    return f"'{clean_name}' is helping reduce overall risk"


def compute_shap_factors(features_df: pd.DataFrame) -> list[ShapFactor]:
    """Compute SHAP values using pre-cached explainer."""
    explainer = _state["shap_explainer"]
    if explainer is None:
        return []
    
    try:
        shap_values = explainer.shap_values(features_df)
        
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            mean_shap = np.mean(np.abs(shap_values[0]), axis=1)
            signed_shap = np.mean(shap_values[0], axis=1)
        elif isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
            signed_shap = np.mean([sv[0] for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values[0])
            signed_shap = shap_values[0]
        
        top_indices = np.argsort(mean_shap)[-3:][::-1]
        
        factors = []
        for idx in top_indices:
            feat = _state["feature_columns"][idx]
            sv = float(signed_shap[idx])
            mag = float(mean_shap[idx])
            factors.append(ShapFactor(
                feature=feat,
                impact="increases_risk" if sv > 0 else "reduces_risk",
                magnitude=round(mag, 4),
                description=get_factor_description(feat, sv),
            ))
        return factors
    except Exception as e:
        print(f"SHAP error: {e}")
        return []


def compute_recommendations(features_df: pd.DataFrame) -> list[Recommendation]:
    """Compute RL intervention recommendations."""
    bandit = _state["rl_bandit"]
    if bandit is None:
        return []
    
    try:
        classifier = _state["classifier"]
        le = _state["label_encoder"]
        
        # Get baseline risk
        proba = classifier.predict_proba(features_df)[0]
        classes = list(le.classes_)
        on_track_idx = classes.index("on_track") if "on_track" in classes else 0
        baseline_risk = 1.0 - proba[on_track_idx]
        
        # Context bin
        ctx = 0 if baseline_risk < 0.3 else (1 if baseline_risk < 0.6 else 2)
        
        alpha = np.array(bandit["alpha"])[ctx]
        beta = np.array(bandit["beta"])[ctx]
        expected = alpha / (alpha + beta)
        
        actions = bandit["actions"]
        
        # Sort by expected reward
        order = np.argsort(expected)[::-1]
        
        recs = []
        for action_id in order[:3]:
            action_id_str = str(action_id)
            if action_id_str in actions:
                action_info = actions[action_id_str]
            else:
                continue
            
            recs.append(Recommendation(
                action=action_info["name"],
                description=action_info["description"],
                expected_risk_reduction=round(float(max(0, expected[action_id] - 0.5) * baseline_risk), 4),
                confidence=round(float(expected[action_id]), 4),
            ))
        
        return recs
    except Exception as e:
        print(f"RL recommendation error: {e}")
        return []


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    uptime = time.time() - _state["startup_time"] if _state["startup_time"] else 0
    return {
        "status": "healthy",
        "models_loaded": _state["classifier"] is not None,
        "model_type": type(_state["classifier"]).__name__ if _state["classifier"] else None,
        "features": len(_state["feature_columns"]) if _state["feature_columns"] else 0,
        "shap_ready": _state["shap_explainer"] is not None,
        "rl_ready": _state["rl_bandit"] is not None,
        "rag_ready": _state["rag_knowledge_base"] is not None,
        "rag_entries": len(_state["rag_knowledge_base"]) if _state["rag_knowledge_base"] else 0,
        "predictions_served": _state["prediction_count"],
        "uptime_seconds": round(uptime, 1),
        "version": "2.0.0",
    }


@app.get("/metrics")
async def metrics():
    """Return model training metrics."""
    if _state["metrics"] is None:
        raise HTTPException(status_code=404, detail="Metrics not available")
    return _state["metrics"]


@app.post("/predict", response_model=PredictionResponse)
async def predict(project: ProjectFeatures):
    """Predict risk class and final cost for a project."""
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    raw = {
        "industry_type": project.industry_type,
        "team_size": project.team_size,
        "seniority_mix_junior": project.seniority_mix_junior,
        "seniority_mix_mid": project.seniority_mix_mid,
        "seniority_mix_senior": project.seniority_mix_senior,
        "budget_planned_usd": project.budget_planned_usd,
        "duration_planned_weeks": project.duration_planned_weeks,
        "scope_change_count": project.scope_change_count,
        "client_type": project.client_type,
        "employee_cost_ratio": project.employee_cost_ratio,
        "attrition_events": project.attrition_events,
        "weekly_burn_rate_variance": project.weekly_burn_rate_variance,
    }
    
    # Check cache
    ck = cache_key(raw)
    if ck in _prediction_cache:
        _state["prediction_count"] += 1
        return _prediction_cache[ck]
    
    # Feature engineering + encoding
    df_encoded = engineer_features_from_raw(raw)
    
    # Predict
    risk_proba = _state["classifier"].predict_proba(df_encoded)[0]
    risk_class_idx = int(np.argmax(risk_proba))
    risk_class = _state["label_encoder"].inverse_transform([risk_class_idx])[0]
    risk_confidence = float(risk_proba[risk_class_idx])
    
    overrun_ratio = float(_state["regressor"].predict(df_encoded)[0])
    predicted_cost_usd = project.budget_planned_usd * overrun_ratio
    predicted_cost_inr = predicted_cost_usd * USD_TO_INR
    overrun_pct = (overrun_ratio - 1.0) * 100
    
    # SHAP (uses pre-cached explainer — fast)
    top_factors = compute_shap_factors(df_encoded)
    
    # RL recommendations
    recommendations = compute_recommendations(df_encoded)
    
    class_probs = {
        _state["label_encoder"].inverse_transform([i])[0]: float(risk_proba[i])
        for i in range(len(risk_proba))
    }
    
    response = PredictionResponse(
        risk_class=risk_class,
        risk_confidence=round(risk_confidence, 4),
        predicted_overrun_ratio=round(overrun_ratio, 4),
        predicted_final_cost_usd=round(predicted_cost_usd, 2),
        predicted_final_cost_inr=round(predicted_cost_inr, 2),
        budget_planned_usd=round(project.budget_planned_usd, 2),
        budget_planned_inr=round(project.budget_planned_usd * USD_TO_INR, 2),
        overrun_percentage=round(overrun_pct, 2),
        top_factors=top_factors,
        class_probabilities=class_probs,
        recommendations=recommendations,
    )
    
    # Cache (limit cache size)
    if len(_prediction_cache) < 1000:
        _prediction_cache[ck] = response
    
    _state["prediction_count"] += 1
    return response


@app.get("/projects/sample")
async def sample_projects():
    """Return 8 sample projects from the test set."""
    if _state["test_data"] is None:
        raise HTTPException(status_code=503, detail="Test data not loaded")
    
    test_data = _state["test_data"]
    samples = []
    for label in ["on_track", "at_risk", "failed"]:
        label_data = test_data[test_data["outcome_predicted"] == label]
        if len(label_data) > 0:
            n_pick = min(3 if label != "failed" else 2, len(label_data))
            picked = label_data.sample(n=n_pick, random_state=42)
            samples.append(picked)
    
    if not samples:
        raise HTTPException(status_code=404, detail="No sample data")
    
    sample_df = pd.concat(samples).head(8)
    
    projects = []
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        features = {}
        
        for col in ["team_size", "seniority_mix_junior", "seniority_mix_mid",
                     "seniority_mix_senior", "budget_planned_usd",
                     "duration_planned_weeks", "scope_change_count",
                     "employee_cost_ratio", "attrition_events",
                     "weekly_burn_rate_variance"]:
            if col in row.index:
                val = row[col]
                features[col] = float(val) if not isinstance(val, (int, np.integer)) else int(val)
        
        for col in row.index:
            if col.startswith("industry_type_") and row[col] == 1:
                features["industry_type"] = col.replace("industry_type_", "")
            if col.startswith("client_type_") and row[col] == 1:
                features["client_type"] = col.replace("client_type_", "")
        
        features.setdefault("industry_type", "Unknown")
        features.setdefault("client_type", "unknown")
        features["budget_planned_inr"] = round(features.get("budget_planned_usd", 0) * USD_TO_INR, 2)
        
        overrun = float(row.get("overrun_ratio_actual", 1.0))
        predicted_cost_usd = features.get("budget_planned_usd", 0) * overrun
        prediction = {
            "risk_class": str(row.get("outcome_predicted", "unknown")),
            "actual_outcome": str(row.get("outcome_actual", "unknown")),
            "confidence": round(float(row.get("prediction_confidence", 0)), 4),
            "overrun_ratio": round(overrun, 4),
            "predicted_final_cost_usd": round(predicted_cost_usd, 2),
            "predicted_final_cost_inr": round(predicted_cost_usd * USD_TO_INR, 2),
            "overrun_percentage": round((overrun - 1.0) * 100, 2),
        }
        
        projects.append({
            "project_index": idx,
            "features": features,
            "prediction": prediction,
        })
    
    return {"projects": projects, "total": len(projects)}


# ─── AI Copilot ──────────────────────────────────────────────────────────────

class CopilotChatRequest(BaseModel):
    question: str = Field(..., description="User's question about the project")
    project_features: dict = Field(..., description="Raw project features")
    prediction_result: dict = Field(..., description="Prediction output including risk_class, overrun, factors")
    chat_history: list[dict] = Field(default=[], description="Previous chat messages")


class CopilotChatResponse(BaseModel):
    answer: str
    grounded_factors: list[str]
    model_used: str


def _build_copilot_system_prompt(features: dict, prediction: dict, query_text: str = "") -> str:
    """Build a compact, grounded system prompt optimized for small local LLMs, enriched with RAG."""
    risk = prediction.get("risk_class", "unknown")
    confidence = prediction.get("risk_confidence", 0)
    overrun_pct = prediction.get("overrun_percentage", 0)
    cost_usd = prediction.get("predicted_final_cost_usd", 0)
    budget_usd = prediction.get("budget_planned_usd", 0)
    
    # Build compact factor list
    factors_lines = []
    for f in prediction.get("top_factors", [])[:5]:
        name = f.get("feature", "").replace("_", " ").title()
        impact = "INCREASES risk" if f.get("impact") == "increases_risk" else "REDUCES risk"
        factors_lines.append(f"- {name}: {impact}. {f.get('description', '')}")
    factors_text = "\n".join(factors_lines) if factors_lines else "- No factors available"
    
    # Build compact recommendations
    rec_lines = []
    for r in prediction.get("recommendations", [])[:3]:
        red = r.get("expected_risk_reduction", 0)
        rec_lines.append(f"- {r.get('action', '')}: {r.get('description', '')} (reduces risk by {red:.0%})")
    recs_text = "\n".join(rec_lines) if rec_lines else "- No recommendations available"
    
    base_prompt = f"""You are DELTA Copilot, an AI assistant for IT project risk analysis.
Answer ONLY using the project data below. Do NOT invent numbers. Be concise (2-3 paragraphs max).

PROJECT FACTS:
- Industry: {features.get('industry_type', 'N/A')}
- Team: {features.get('team_size', 'N/A')} people (Junior {features.get('seniority_mix_junior', 0):.0%}, Mid {features.get('seniority_mix_mid', 0):.0%}, Senior {features.get('seniority_mix_senior', 0):.0%})
- Budget: ${budget_usd:,.0f} | Duration: {features.get('duration_planned_weeks', 'N/A')} weeks
- Scope Changes: {features.get('scope_change_count', 0)} | Contract: {features.get('client_type', 'N/A')}
- Employee Cost Ratio: {features.get('employee_cost_ratio', 0):.1%} | Attrition Events: {features.get('attrition_events', 0)}
- Burn Rate Variance: {features.get('weekly_burn_rate_variance', 0):.1%}

PREDICTION:
- Risk: {risk} (confidence: {confidence:.0%})
- Cost Overrun: {overrun_pct:.1f}% | Final Cost: ${cost_usd:,.0f} vs Budget: ${budget_usd:,.0f}

RISK FACTORS:
{factors_text}

RECOMMENDED ACTIONS:
{recs_text}

RULES: Only cite numbers from above. Reference factor names when explaining risk. When citing PMBOK guidelines, always include the source in square brackets e.g. [PMBOK 7th Edition, Section X.Y]. If unsure, say so."""

    # --- RAG: Retrieve relevant PMBOK standards and inject ---
    rag_context = ""
    rag_entries = _rag_retrieve(query_text or "", top_k=2)
    if rag_entries:
        rag_lines = []
        for entry in rag_entries:
            rag_lines.append(f"[{entry['source']}] {entry['title']}: {entry['content']}")
        rag_context = "\n\nPMBOK / IT GOVERNANCE REFERENCE STANDARDS:\n" + "\n\n".join(rag_lines)

    return base_prompt + rag_context


def _fallback_copilot_response(question: str, features: dict, prediction: dict) -> CopilotChatResponse:
    """Generate a grounded response without Gemini API, using model outputs directly."""
    risk = prediction.get("risk_class", "unknown")
    confidence = prediction.get("risk_confidence", 0)
    overrun_pct = prediction.get("overrun_percentage", 0)
    budget = prediction.get("budget_planned_usd", 0)
    cost = prediction.get("predicted_final_cost_usd", 0)
    
    top_factors = prediction.get("top_factors", [])
    factor_names = [f.get("feature", "") for f in top_factors]
    
    q_lower = question.lower()
    
    # Build factor explanations
    factor_lines = []
    for f in top_factors:
        name = f.get("feature", "").replace("_", " ").title()
        desc = f.get("description", "")
        impact = "increasing" if f.get("impact") == "increases_risk" else "reducing"
        factor_lines.append(f"• **{name}** is {impact} risk: {desc}")
    factors_block = "\n".join(factor_lines) if factor_lines else "No SHAP factors available."
    
    recs = prediction.get("recommendations", [])
    rec_lines = []
    for r in recs:
        reduction = r.get("expected_risk_reduction", 0)
        rec_lines.append(f"• **{r.get('action', '')}**: {r.get('description', '')} (est. risk reduction: {reduction:.1%})")
    recs_block = "\n".join(rec_lines) if rec_lines else "No interventions available."
    
    if any(kw in q_lower for kw in ["why", "risk", "high risk", "failed", "at risk", "driver", "cause"]):
        answer = (
            f"This project is classified as **{risk}** with {confidence:.0%} confidence. "
            f"The model predicts a cost overrun of **{overrun_pct:.1f}%**, "
            f"bringing the projected final cost to **${cost:,.0f}** against a planned budget of **${budget:,.0f}**.\n\n"
            f"The top factors driving this prediction are:\n{factors_block}"
        )
    elif any(kw in q_lower for kw in ["intervention", "recommend", "action", "fix", "improve", "reduce", "mitigate"]):
        answer = (
            f"Based on the reinforcement learning analysis for this **{risk}** project, "
            f"here are the recommended interventions:\n\n{recs_block}\n\n"
            f"These are estimated via simulated counterfactual analysis against the trained model."
        )
    elif any(kw in q_lower for kw in ["cost", "overrun", "budget", "expense", "spend"]):
        answer = (
            f"The model predicts this project will overrun by **{overrun_pct:.1f}%**. "
            f"The planned budget is **${budget:,.0f}**, but the projected final cost is **${cost:,.0f}** — "
            f"an excess of **${cost - budget:,.0f}**.\n\n"
            f"Key cost factors:\n{factors_block}"
        )
    elif any(kw in q_lower for kw in ["team", "senior", "junior", "attrition", "staff"]):
        team_size = features.get("team_size", "N/A")
        jr = features.get("seniority_mix_junior", 0)
        mid = features.get("seniority_mix_mid", 0)
        sr = features.get("seniority_mix_senior", 0)
        attrition = features.get("attrition_events", 0)
        answer = (
            f"This project has a team of **{team_size}** members with a seniority mix of "
            f"**{jr:.0%} junior**, **{mid:.0%} mid-level**, and **{sr:.0%} senior**. "
            f"There have been **{attrition} attrition event(s)**.\n\n"
            f"Team-related factors in the risk model:\n{factors_block}"
        )
    else:
        answer = (
            f"This project is classified as **{risk}** (confidence: {confidence:.0%}) with a predicted "
            f"cost overrun of **{overrun_pct:.1f}%** (${cost:,.0f} vs. ${budget:,.0f} planned).\n\n"
            f"Key factors:\n{factors_block}\n\n"
            f"Recommended interventions:\n{recs_block}"
        )
    
    return CopilotChatResponse(
        answer=answer,
        grounded_factors=factor_names,
        model_used="fallback-shap-grounded"
    )


def _call_ollama_sync(messages: list) -> dict:
    """Blocking Ollama HTTP call — runs in a thread to avoid blocking async loop."""
    import urllib.request
    import json

    payload = {
        "model": "llama3.2:latest",
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 250,
            "top_p": 0.9,
        }
    }

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode("utf-8"))


async def _call_ollama_copilot(question: str, features: dict, prediction: dict, chat_history: list) -> CopilotChatResponse:
    """Query local Jarvis / Ollama LLM — runs in thread pool to avoid blocking."""
    import asyncio

    system_prompt = _build_copilot_system_prompt(features, prediction, query_text=question)

    # Keep context tight: system + last 2 exchanges + current question
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-4:]:
        role = "user" if msg.get("role") == "user" else "assistant"
        messages.append({"role": role, "content": msg.get("content", "")})
    messages.append({"role": "user", "content": question})

    # Run the blocking HTTP call in a thread so we don't block FastAPI's event loop
    data = await asyncio.to_thread(_call_ollama_sync, messages)

    text = data.get("message", {}).get("content", "")
    if text:
        top_factors = prediction.get("top_factors", [])
        grounded = [f.get("feature", "") for f in top_factors if f.get("feature", "") in text]
        return CopilotChatResponse(
            answer=text,
            grounded_factors=grounded if grounded else [f.get("feature", "") for f in top_factors],
            model_used="jarvis-ollama (llama3.2)"
        )

    raise RuntimeError("Empty response from Ollama")


@app.post("/copilot/chat", response_model=CopilotChatResponse)
async def copilot_chat(req: CopilotChatRequest):
    """AI Project Manager Copilot — answers grounded in real model outputs."""
    
    # 1. Try local Jarvis / Ollama LLM endpoint first
    try:
        return await _call_ollama_copilot(req.question, req.project_features, req.prediction_result, req.chat_history)
    except Exception as e:
        print(f"Jarvis/Ollama fallback: {e}")
    
    # 2. Try Gemini API
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    
    if api_key:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            system_prompt = _build_copilot_system_prompt(req.project_features, req.prediction_result, query_text=req.question)
            
            # Build conversation
            messages = []
            for msg in req.chat_history[-6:]:  # Last 6 messages for context window
                role = "user" if msg.get("role") == "user" else "model"
                messages.append({"role": role, "parts": [msg.get("content", "")]})
            messages.append({"role": "user", "parts": [req.question]})
            
            model = genai.GenerativeModel(
                model_name="gemini-2.0-flash",
                system_instruction=system_prompt,
            )
            response = model.generate_content(messages)
            
            # Extract grounded factor names
            top_factors = req.prediction_result.get("top_factors", [])
            grounded = [f.get("feature", "") for f in top_factors
                        if f.get("feature", "") in response.text]
            
            return CopilotChatResponse(
                answer=response.text,
                grounded_factors=grounded if grounded else [f.get("feature", "") for f in top_factors],
                model_used="gemini-2.0-flash"
            )
        except Exception as e:
            print(f"Gemini API error, falling back: {e}")
    
    # 3. Model-grounded SHAP response engine fallback
    return _fallback_copilot_response(req.question, req.project_features, req.prediction_result)


# ─── Risk Trajectory Across Milestones ────────────────────────────────────────

MILESTONE_PHASES = [
    {"id": "kickoff",  "label": "Kickoff",  "week_pct": 0.0,  "scope_mult": 0.0, "attrition_mult": 0.0, "burn_drift": 0.0},
    {"id": "planning", "label": "Planning", "week_pct": 0.15, "scope_mult": 0.2, "attrition_mult": 0.1, "burn_drift": 0.02},
    {"id": "build",    "label": "Build",    "week_pct": 0.45, "scope_mult": 0.6, "attrition_mult": 0.5, "burn_drift": 0.05},
    {"id": "testing",  "label": "Testing",  "week_pct": 0.70, "scope_mult": 0.85,"attrition_mult": 0.7, "burn_drift": 0.08},
    {"id": "uat",      "label": "UAT",      "week_pct": 0.85, "scope_mult": 1.0, "attrition_mult": 0.9, "burn_drift": 0.10},
    {"id": "golive",   "label": "Go-Live",  "week_pct": 1.0,  "scope_mult": 1.0, "attrition_mult": 1.0, "burn_drift": 0.12},
]


class TrajectoryRequest(BaseModel):
    project_features: ProjectFeatures = Field(..., description="Baseline project parameters")


class MilestonePoint(BaseModel):
    milestone_id: str
    milestone_label: str
    week_number: int
    risk_class: str
    risk_confidence: float
    overrun_percentage: float
    predicted_cost_usd: float
    budget_usd: float
    top_factor: str
    top_factor_impact: str
    health: str  # "healthy", "warning", "critical"


class TrajectoryResponse(BaseModel):
    milestones: list[MilestonePoint]
    risk_escalation_point: Optional[str] = None  # milestone where risk first escalates
    final_risk: str
    total_duration_weeks: int


@app.post("/trajectory", response_model=TrajectoryResponse)
async def compute_trajectory(req: TrajectoryRequest):
    """Simulate project risk evolution across 6 milestone phases."""
    base = req.project_features.model_dump()
    total_weeks = base["duration_planned_weeks"]
    total_scope = base["scope_change_count"]
    total_attrition = base["attrition_events"]
    base_burn_var = base["weekly_burn_rate_variance"]

    milestones: list[MilestonePoint] = []
    prev_risk = None
    escalation_point = None

    for phase in MILESTONE_PHASES:
        # Build phase-specific features: progressively apply stress
        phase_features = dict(base)
        phase_features["scope_change_count"] = max(0, round(total_scope * phase["scope_mult"]))
        phase_features["attrition_events"] = max(0, round(total_attrition * phase["attrition_mult"]))
        phase_features["weekly_burn_rate_variance"] = min(1.0, base_burn_var + phase["burn_drift"])

        # Run prediction through the model
        pf = ProjectFeatures(**phase_features)
        pred_resp = await predict(pf)
        pred = pred_resp.model_dump()

        week_num = max(1, round(total_weeks * phase["week_pct"])) if phase["week_pct"] > 0 else 0

        # Determine health status
        risk = pred["risk_class"]
        overrun = pred["overrun_percentage"]
        if risk == "on_track" and overrun < 10:
            health = "healthy"
        elif risk == "failed" or overrun > 25:
            health = "critical"
        else:
            health = "warning"

        # Track escalation point
        if prev_risk and prev_risk == "on_track" and risk != "on_track" and not escalation_point:
            escalation_point = phase["label"]
        if prev_risk and prev_risk != "failed" and risk == "failed" and not escalation_point:
            escalation_point = phase["label"]
        prev_risk = risk

        # Top SHAP factor
        top_f = pred.get("top_factors", [{}])[0] if pred.get("top_factors") else {}

        milestones.append(MilestonePoint(
            milestone_id=phase["id"],
            milestone_label=phase["label"],
            week_number=week_num,
            risk_class=risk,
            risk_confidence=pred["risk_confidence"],
            overrun_percentage=overrun,
            predicted_cost_usd=pred["predicted_final_cost_usd"],
            budget_usd=pred["budget_planned_usd"],
            top_factor=top_f.get("feature", ""),
            top_factor_impact=top_f.get("impact", ""),
            health=health,
        ))

    return TrajectoryResponse(
        milestones=milestones,
        risk_escalation_point=escalation_point,
        final_risk=milestones[-1].risk_class if milestones else "unknown",
        total_duration_weeks=total_weeks,
    )


# ─── What-If Simulation ──────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    baseline_features: ProjectFeatures = Field(..., description="Original baseline project parameters")
    team_size_delta: int = Field(default=0, description="Adjustment to team size (+/-)")
    scope_change_delta: int = Field(default=0, description="Adjustment to scope change count (+/-)")
    client_type: Optional[str] = Field(default=None, description="Override client contract type")
    seniority_mix_junior: Optional[float] = Field(default=None, description="Override junior ratio")
    seniority_mix_mid: Optional[float] = Field(default=None, description="Override mid ratio")
    seniority_mix_senior: Optional[float] = Field(default=None, description="Override senior ratio")


class SimulationResponse(BaseModel):
    baseline_prediction: dict
    simulated_prediction: dict
    delta: dict
    simulated_features: dict


@app.post("/simulate", response_model=SimulationResponse)
async def simulate_scenario(req: SimulationRequest):
    """Run a counterfactual What-If simulation comparing baseline vs modified project."""
    
    # 1. Run baseline prediction
    baseline_resp = await predict(req.baseline_features)
    baseline_result = baseline_resp.model_dump()
    
    # 2. Construct simulated feature set
    sim_features_dict = req.baseline_features.model_dump()
    
    # Apply deltas
    sim_features_dict["team_size"] = max(1, sim_features_dict["team_size"] + req.team_size_delta)
    sim_features_dict["scope_change_count"] = max(0, sim_features_dict["scope_change_count"] + req.scope_change_delta)
    
    if req.client_type:
        sim_features_dict["client_type"] = req.client_type
    if req.seniority_mix_junior is not None:
        sim_features_dict["seniority_mix_junior"] = req.seniority_mix_junior
    if req.seniority_mix_mid is not None:
        sim_features_dict["seniority_mix_mid"] = req.seniority_mix_mid
    if req.seniority_mix_senior is not None:
        sim_features_dict["seniority_mix_senior"] = req.seniority_mix_senior
        
    sim_features = ProjectFeatures(**sim_features_dict)
    
    # 3. Run simulated prediction
    simulated_resp = await predict(sim_features)
    simulated_result = simulated_resp.model_dump()
    
    # 4. Compute deltas
    cost_diff_usd = simulated_result["predicted_final_cost_usd"] - baseline_result["predicted_final_cost_usd"]
    cost_diff_inr = simulated_result["predicted_final_cost_inr"] - baseline_result["predicted_final_cost_inr"]
    overrun_diff_pct = simulated_result["overrun_percentage"] - baseline_result["overrun_percentage"]
    confidence_diff = simulated_result["risk_confidence"] - baseline_result["risk_confidence"]
    
    risk_changed = baseline_result["risk_class"] != simulated_result["risk_class"]
    
    delta_summary = {
        "cost_diff_usd": round(cost_diff_usd, 2),
        "cost_diff_inr": round(cost_diff_inr, 2),
        "overrun_diff_pct": round(overrun_diff_pct, 2),
        "confidence_diff": round(confidence_diff, 4),
        "risk_changed": risk_changed,
        "baseline_risk": baseline_result["risk_class"],
        "simulated_risk": simulated_result["risk_class"],
        "is_improvement": cost_diff_usd < 0 or (baseline_result["risk_class"] != "on_track" and simulated_result["risk_class"] == "on_track"),
    }
    
    return SimulationResponse(
        baseline_prediction=baseline_result,
        simulated_prediction=simulated_result,
        delta=delta_summary,
        simulated_features=sim_features_dict
    )


# ─── Executive Report Generator ──────────────────────────────────────────────

class ReportRequest(BaseModel):
    project_features: ProjectFeatures = Field(..., description="Project input parameters")
    prediction_result: dict = Field(..., description="Prediction output dictionary")
    simulation_result: Optional[dict] = Field(default=None, description="Optional simulation result dictionary")


class ReportResponse(BaseModel):
    markdown_content: str
    summary_metrics: dict
    timestamp: str


@app.post("/report", response_model=ReportResponse)
async def generate_executive_report(req: ReportRequest):
    """Generate a formal 1-page PMO Executive Risk & Audit Report in Markdown."""
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
    
    pf = req.project_features
    pr = req.prediction_result
    sim = req.simulation_result
    
    risk = pr.get("risk_class", "unknown").upper()
    conf = pr.get("risk_confidence", 0.0) * 100
    budget_usd = pr.get("budget_planned_usd", 0.0)
    cost_usd = pr.get("predicted_final_cost_usd", 0.0)
    cost_inr = pr.get("predicted_final_cost_inr", 0.0)
    overrun_pct = pr.get("overrun_percentage", 0.0)
    
    # SHAP Drivers
    shap_lines = ""
    for f in pr.get("top_factors", []):
        name = f.get("feature", "").replace("_", " ").title()
        impact = "↑ INCREASES RISK" if f.get("impact") == "increases_risk" else "↓ REDUCES RISK"
        shap_lines += f"- **{name}** ({impact}): {f.get('description', '')}\n"
    if not shap_lines:
        shap_lines = "- Standard execution parameters within normal range.\n"
        
    # RL Interventions
    rl_lines = ""
    for r in pr.get("recommendations", []):
        red = r.get("expected_risk_reduction", 0.0) * 100
        rl_lines += f"- **{r.get('action', '')}**: {r.get('description', '')} *(Est. Risk Reduction: -{red:.1f}%)*\n"
    if not rl_lines:
        rl_lines = "- Maintain current operational oversight.\n"
        
    # Simulation Section if available
    sim_section = ""
    if sim and "delta" in sim:
        d = sim["delta"]
        sim_section = f"""
---

### ⚡ Counterfactual Simulation Analysis
- **Simulated Scenario Risk**: `{d.get('baseline_risk', '').upper()}` ➔ `{d.get('simulated_risk', '').upper()}`
- **Net Cost Variance**: `{'-$' if d.get('cost_diff_usd', 0) <= 0 else '+$'}{abs(d.get('cost_diff_usd', 0)):,.2f} USD`
- **Overrun Shift**: `{d.get('overrun_diff_pct', 0):+.1f}%` from baseline
- **Assessment**: {'✓ Favorable scenario outcome reduces financial vulnerability.' if d.get('is_improvement') else '⚠ Scenario increases cost overrun risk; counterbalance required.'}
"""

    markdown_doc = f"""# 📄 DELTA PMO EXECUTIVE RISK & FINANCIAL AUDIT REPORT
**Generated:** {now_str} | **Track:** Open Innovation | **System:** DELTA AI v2.0

---

## 📊 Executive Overview
- **Project Classification:** `{risk}` (Model Confidence: **{conf:.1f}%**)
- **Planned Budget:** `${budget_usd:,.2f} USD`
- **Projected Final Cost:** `${cost_usd:,.2f} USD` (`₹{cost_inr:,.2f} INR`)
- **Cost Overrun Variance:** `+{overrun_pct:.1f}%` `${cost_usd - budget_usd:,.2f} USD`

---

## 🔍 Key Risk Drivers (SHAP Explainability)
{shap_lines}
---

## 💡 Recommended PMO Countermeasures (RL Agent)
{rl_lines}{sim_section}
---

## 📋 Project Parameters Audit Log
- **Industry Sector:** {pf.industry_type} | **Client Contract:** {pf.client_type}
- **Team Size:** {pf.team_size} members | **Duration:** {pf.duration_planned_weeks} weeks
- **Seniority Distribution:** Junior `{pf.seniority_mix_junior:.0%}` | Mid `{pf.seniority_mix_mid:.0%}` | Senior `{pf.seniority_mix_senior:.0%}`
- **Employee Cost Ratio:** `{pf.employee_cost_ratio:.1%}` | **Attrition Events:** {pf.attrition_events}
- **Scope Change Count:** {pf.scope_change_count} | **Burn Rate Variance:** `{pf.weekly_burn_rate_variance:.1%}`

---
*Confidential — Generated by DELTA Project Cost-Overrun & Delivery-Risk Prediction Engine*
"""

    summary_metrics = {
        "risk_class": risk,
        "confidence_pct": round(conf, 1),
        "budget_usd": budget_usd,
        "predicted_cost_usd": cost_usd,
        "overrun_pct": round(overrun_pct, 1),
    }

    return ReportResponse(
        markdown_content=markdown_doc,
        summary_metrics=summary_metrics,
        timestamp=now_str
    )


# ─── RAG Query Endpoint ──────────────────────────────────────────────────────

class RAGQueryRequest(BaseModel):
    query: str = Field(..., description="Natural language query about project governance or PMBOK standards")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of results to return")


class RAGQueryResponse(BaseModel):
    query: str
    results: list[dict]
    total_results: int


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """Retrieve relevant PMBOK / IT Governance standards matching a natural language query."""
    if not _state["rag_knowledge_base"]:
        raise HTTPException(status_code=503, detail="RAG knowledge base not loaded")

    results = _rag_retrieve(req.query, top_k=req.top_k)

    return RAGQueryResponse(
        query=req.query,
        results=results,
        total_results=len(results),
    )


# ─── Slack Risk Alert Webhook ────────────────────────────────────────────────

class SlackAlertRequest(BaseModel):
    project_features: ProjectFeatures = Field(..., description="Project input parameters")
    prediction_result: dict = Field(..., description="Prediction output dictionary")
    webhook_url: Optional[str] = Field(default=None, description="Override Slack webhook URL")


class SlackAlertResponse(BaseModel):
    status: str  # "sent" or "dry_run"
    slack_payload: dict
    message: str


def _build_slack_blocks(pf: ProjectFeatures, pr: dict) -> dict:
    """Build a Slack Block Kit message payload for risk alerts."""
    from datetime import datetime
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    risk = pr.get("risk_class", "unknown").upper()
    conf = pr.get("risk_confidence", 0.0) * 100
    budget_usd = pr.get("budget_planned_usd", 0.0)
    cost_usd = pr.get("predicted_final_cost_usd", 0.0)
    overrun_pct = pr.get("overrun_percentage", 0.0)
    variance_usd = cost_usd - budget_usd

    risk_emoji = "🔴" if risk == "FAILED" else "🟡" if risk == "AT_RISK" else "🟢"

    # SHAP factors
    shap_lines = []
    for f in pr.get("top_factors", [])[:3]:
        name = f.get("feature", "").replace("_", " ").title()
        arrow = "↑" if f.get("impact") == "increases_risk" else "↓"
        shap_lines.append(f"• {arrow} *{name}*: {f.get('description', '')}")
    shap_text = "\n".join(shap_lines) if shap_lines else "• No significant risk drivers identified."

    # RL recommendations
    rec_lines = []
    for r in pr.get("recommendations", [])[:2]:
        red = r.get("expected_risk_reduction", 0.0) * 100
        rec_lines.append(f"• *{r.get('action', '')}*: {r.get('description', '')} (est. -{red:.0f}% risk)")
    rec_text = "\n".join(rec_lines) if rec_lines else "• Maintain current oversight."

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"{risk_emoji} DELTA Risk Alert — {risk}", "emoji": True}
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Risk Classification:*\n`{risk}` ({conf:.0f}% confidence)"},
                {"type": "mrkdwn", "text": f"*Industry / Contract:*\n{pf.industry_type} / {pf.client_type.replace('_', ' ').title()}"},
                {"type": "mrkdwn", "text": f"*Planned Budget:*\n${budget_usd:,.0f} USD"},
                {"type": "mrkdwn", "text": f"*Predicted Cost:*\n${cost_usd:,.0f} USD (+{overrun_pct:.1f}%)"},
                {"type": "mrkdwn", "text": f"*Cost Variance:*\n+${variance_usd:,.0f} USD"},
                {"type": "mrkdwn", "text": f"*Team Size:*\n{pf.team_size} members"},
            ]
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*🔍 Top Risk Drivers (SHAP):*\n{shap_text}"}
        },
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": f"*💡 Recommended Actions:*\n{rec_text}"}
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {"type": "mrkdwn", "text": f"📊 _DELTA AI v2.0 — {now_str}_ | Scope Changes: {pf.scope_change_count} | Attrition: {pf.attrition_events} | ECR: {pf.employee_cost_ratio:.0%}"}
            ]
        }
    ]

    return {"blocks": blocks, "text": f"DELTA Risk Alert: {risk} — ${cost_usd:,.0f} predicted (budget ${budget_usd:,.0f})"}


@app.post("/alerts/slack", response_model=SlackAlertResponse)
async def send_slack_alert(req: SlackAlertRequest):
    """Send a formatted Slack Block Kit risk alert, or return dry-run preview."""
    payload = _build_slack_blocks(req.project_features, req.prediction_result)

    webhook_url = req.webhook_url or os.environ.get("SLACK_WEBHOOK_URL")

    if webhook_url:
        try:
            import urllib.request
            http_req = urllib.request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(http_req, timeout=10) as resp:
                resp.read()
            return SlackAlertResponse(
                status="sent",
                slack_payload=payload,
                message=f"Alert sent to Slack webhook successfully."
            )
        except Exception as e:
            return SlackAlertResponse(
                status="error",
                slack_payload=payload,
                message=f"Failed to send Slack alert: {str(e)}"
            )
    else:
        return SlackAlertResponse(
            status="dry_run",
            slack_payload=payload,
            message="No SLACK_WEBHOOK_URL configured. Showing dry-run preview of the alert payload."
        )


# ─── Email Risk Alert ────────────────────────────────────────────────────────

class EmailAlertRequest(BaseModel):
    recipient_email: str = Field(..., description="Recipient email address")
    project_features: ProjectFeatures = Field(..., description="Project input parameters")
    prediction_result: dict = Field(..., description="Prediction output dictionary")
    smtp_host: Optional[str] = Field(default=None, description="Optional SMTP server host")
    smtp_port: Optional[int] = Field(default=None, description="Optional SMTP server port")
    smtp_user: Optional[str] = Field(default=None, description="Optional SMTP username")
    smtp_password: Optional[str] = Field(default=None, description="Optional SMTP password")


class EmailAlertResponse(BaseModel):
    status: str  # "sent" or "dry_run" or "error"
    subject: str
    html_preview: str
    recipient: str
    message: str


def _build_email_html(pf: ProjectFeatures, pr: dict) -> tuple[str, str, str]:
    """Generate subject, plain text and HTML content for executive email alert."""
    from datetime import datetime
    now_str = datetime.now().strftime("%B %d, %Y %H:%M UTC")

    risk = pr.get("risk_class", "unknown").upper()
    conf = pr.get("risk_confidence", 0.0) * 100
    budget_usd = pr.get("budget_planned_usd", 0.0)
    cost_usd = pr.get("predicted_final_cost_usd", 0.0)
    overrun_pct = pr.get("overrun_percentage", 0.0)
    variance_usd = cost_usd - budget_usd

    risk_color = "#EF4444" if risk == "FAILED" else "#F59E0B" if risk == "AT_RISK" else "#22C55E"
    risk_emoji = "🚨" if risk == "FAILED" else "⚠️" if risk == "AT_RISK" else "✅"

    subject = f"[DELTA AI {risk_emoji} {risk}] Project Delivery Risk Alert — {pf.industry_type} (${budget_usd:,.0f} Budget)"

    # SHAP factors
    shap_rows = ""
    shap_plain = ""
    for f in pr.get("top_factors", [])[:4]:
        name = f.get("feature", "").replace("_", " ").title()
        impact = f.get("impact", "")
        desc = f.get("description", "")
        impact_label = "↑ Increases Risk" if impact == "increases_risk" else "↓ Reduces Risk"
        color = "#EF4444" if impact == "increases_risk" else "#22C55E"
        shap_rows += f"""<tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: #F1F5F9; font-weight: 600;">{name}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: {color}; font-weight: 600;">{impact_label}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: #94A3B8;">{desc}</td>
        </tr>"""
        shap_plain += f"- {name}: {impact_label} ({desc})\n"

    # Recommendations
    rec_rows = ""
    rec_plain = ""
    for r in pr.get("recommendations", [])[:3]:
        act = r.get("action", "")
        desc = r.get("description", "")
        red = r.get("expected_risk_reduction", 0.0) * 100
        rec_rows += f"""<tr>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: #38BDF8; font-weight: 600;">{act}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: #94A3B8;">{desc}</td>
            <td style="padding: 8px 12px; border-bottom: 1px solid #334155; color: #34D399; font-weight: 600; text-align: right;">-{red:.0f}% Risk</td>
        </tr>"""
        rec_plain += f"- {act}: {desc} (Est. -{red:.0f}% risk)\n"

    html = f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #0A0E1A; color: #E2E8F0; margin: 0; padding: 24px;">
  <div style="max-width: 640px; margin: 0 auto; background: #0F172A; border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; overflow: hidden; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
    
    <!-- Header -->
    <div style="background: linear-gradient(135deg, #2E5CFF, #7B3FE4); padding: 24px 32px; color: #FFFFFF;">
      <h1 style="margin: 0; font-size: 22px; font-weight: 800; letter-spacing: -0.02em;">Δ DELTA Delivery Risk Intelligence</h1>
      <p style="margin: 6px 0 0 0; font-size: 13px; opacity: 0.9;">Automated PMO Risk & Cost Early-Warning Alert</p>
    </div>

    <div style="padding: 32px;">
      <!-- Risk Status Banner -->
      <div style="background: rgba(255,255,255,0.03); border: 1px solid #334155; border-left: 6px solid {risk_color}; border-radius: 8px; padding: 18px 20px; margin-bottom: 24px;">
        <div style="font-size: 12px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.05em; font-weight: 700;">Prediction Outcome</div>
        <div style="font-size: 24px; font-weight: 800; color: {risk_color}; margin: 4px 0;">{risk}</div>
        <div style="font-size: 13px; color: #CBD5E1;">Model Confidence: <strong>{conf:.0f}%</strong> | Predicted Cost Overrun: <strong>+{overrun_pct:.1f}%</strong></div>
      </div>

      <!-- Financial Metrics Table -->
      <h2 style="font-size: 15px; color: #F8FAFC; margin: 0 0 12px 0; font-weight: 700; border-bottom: 1px solid #334155; padding-bottom: 6px;">Financial Impact Overview</h2>
      <table style="width: 100%; border-collapse: collapse; font-size: 13px; margin-bottom: 24px;">
        <tr>
          <td style="padding: 8px 12px; color: #94A3B8; border-bottom: 1px solid #1E293B;">Planned Budget:</td>
          <td style="padding: 8px 12px; color: #F1F5F9; font-weight: 600; text-align: right; border-bottom: 1px solid #1E293B;">${budget_usd:,.0f} USD</td>
        </tr>
        <tr>
          <td style="padding: 8px 12px; color: #94A3B8; border-bottom: 1px solid #1E293B;">Predicted Final Cost:</td>
          <td style="padding: 8px 12px; color: #F87171; font-weight: 700; text-align: right; border-bottom: 1px solid #1E293B;">${cost_usd:,.0f} USD</td>
        </tr>
        <tr>
          <td style="padding: 8px 12px; color: #94A3B8; border-bottom: 1px solid #1E293B;">Projected Cost Variance:</td>
          <td style="padding: 8px 12px; color: #F87171; font-weight: 700; text-align: right; border-bottom: 1px solid #1E293B;">+${variance_usd:,.0f} USD (+{overrun_pct:.1f}%)</td>
        </tr>
        <tr>
          <td style="padding: 8px 12px; color: #94A3B8; border-bottom: 1px solid #1E293B;">Industry / Contract:</td>
          <td style="padding: 8px 12px; color: #CBD5E1; text-align: right; border-bottom: 1px solid #1E293B;">{pf.industry_type} / {pf.client_type.replace('_', ' ').title()}</td>
        </tr>
        <tr>
          <td style="padding: 8px 12px; color: #94A3B8;">Team / Duration:</td>
          <td style="padding: 8px 12px; color: #CBD5E1; text-align: right;">{pf.team_size} members / {pf.duration_planned_weeks} weeks</td>
        </tr>
      </table>

      <!-- Top SHAP Risk Drivers -->
      <h2 style="font-size: 15px; color: #F8FAFC; margin: 0 0 12px 0; font-weight: 700; border-bottom: 1px solid #334155; padding-bottom: 6px;">Top Risk Drivers (SHAP Analysis)</h2>
      <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 24px;">
        <thead>
          <tr style="background: #1E293B; text-align: left; color: #94A3B8; font-size: 11px;">
            <th style="padding: 8px 12px;">FACTOR</th>
            <th style="padding: 8px 12px;">IMPACT</th>
            <th style="padding: 8px 12px;">DETAILS</th>
          </tr>
        </thead>
        <tbody>
          {shap_rows}
        </tbody>
      </table>

      <!-- Recommended Actions -->
      <h2 style="font-size: 15px; color: #F8FAFC; margin: 0 0 12px 0; font-weight: 700; border-bottom: 1px solid #334155; padding-bottom: 6px;">Recommended Mitigations (RL Multi-Armed Bandit)</h2>
      <table style="width: 100%; border-collapse: collapse; font-size: 12px; margin-bottom: 24px;">
        <thead>
          <tr style="background: #1E293B; text-align: left; color: #94A3B8; font-size: 11px;">
            <th style="padding: 8px 12px;">ACTION</th>
            <th style="padding: 8px 12px;">DESCRIPTION</th>
            <th style="padding: 8px 12px; text-align: right;">BENEFIT</th>
          </tr>
        </thead>
        <tbody>
          {rec_rows}
        </tbody>
      </table>

      <!-- Footer -->
      <div style="border-top: 1px solid #334155; padding-top: 16px; font-size: 11px; color: #64748B; text-align: center;">
        Sent automatically by DELTA 2.0 AI Risk Prediction Engine • Generated at {now_str}
      </div>
    </div>
  </div>
</body>
</html>"""

    plain = f"""DELTA Delivery Risk Alert: {risk} ({conf:.0f}% confidence)
Planned Budget: ${budget_usd:,.0f} USD
Predicted Cost: ${cost_usd:,.0f} USD (+{overrun_pct:.1f}%)
Cost Variance: +${variance_usd:,.0f} USD

Top Risk Factors:
{shap_plain}
Recommended Mitigations:
{rec_plain}
Generated: {now_str} by DELTA AI v2.0
"""

    return subject, plain, html


@app.post("/alerts/email", response_model=EmailAlertResponse)
async def send_email_alert(req: EmailAlertRequest):
    """Send an executive HTML email risk alert via SMTP or return dry-run preview."""
    subject, plain_text, html_content = _build_email_html(req.project_features, req.prediction_result)

    smtp_host = req.smtp_host or os.environ.get("SMTP_HOST")
    smtp_port = req.smtp_port or int(os.environ.get("SMTP_PORT", 587))
    smtp_user = req.smtp_user or os.environ.get("SMTP_USER")
    smtp_password = req.smtp_password or os.environ.get("SMTP_PASSWORD")
    sender_email = os.environ.get("SMTP_FROM", smtp_user or "delta-alerts@pm-system.ai")

    if smtp_host and smtp_user and smtp_password:
        try:
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText

            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"] = sender_email
            msg["To"] = req.recipient_email

            msg.attach(MIMEText(plain_text, "plain"))
            msg.attach(MIMEText(html_content, "html"))

            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_email, req.recipient_email, msg.as_string())

            return EmailAlertResponse(
                status="sent",
                subject=subject,
                html_preview=html_content,
                recipient=req.recipient_email,
                message=f"Executive email alert sent to {req.recipient_email} successfully."
            )
        except Exception as e:
            return EmailAlertResponse(
                status="error",
                subject=subject,
                html_preview=html_content,
                recipient=req.recipient_email,
                message=f"Failed to send email via SMTP: {str(e)}"
            )
    else:
        return EmailAlertResponse(
            status="dry_run",
            subject=subject,
            html_preview=html_content,
            recipient=req.recipient_email,
            message=f"No SMTP server credentials configured. Returning dry-run preview for {req.recipient_email}."
        )


# ─── Bulk Project Upload & Batch Prediction ──────────────────────────────────

REQUIRED_UPLOAD_COLUMNS = [
    "industry_type", "team_size", "seniority_mix_junior", "seniority_mix_mid",
    "seniority_mix_senior", "budget_planned_usd", "duration_planned_weeks",
    "scope_change_count", "client_type", "employee_cost_ratio",
    "attrition_events", "weekly_burn_rate_variance"
]


def _predict_single_row(raw: dict) -> dict:
    """Run prediction pipeline on a single project dict. Returns result dict."""
    try:
        df_encoded = engineer_features_from_raw(raw)
        
        risk_proba = _state["classifier"].predict_proba(df_encoded)[0]
        risk_class_idx = int(np.argmax(risk_proba))
        risk_class = _state["label_encoder"].inverse_transform([risk_class_idx])[0]
        risk_confidence = float(risk_proba[risk_class_idx])
        
        overrun_ratio = float(_state["regressor"].predict(df_encoded)[0])
        budget = float(raw.get("budget_planned_usd", 0))
        predicted_cost_usd = budget * overrun_ratio
        overrun_pct = (overrun_ratio - 1.0) * 100
        
        top_factors = compute_shap_factors(df_encoded)
        recommendations = compute_recommendations(df_encoded)
        
        return {
            "status": "success",
            "risk_class": risk_class,
            "risk_confidence": round(risk_confidence, 4),
            "overrun_percentage": round(overrun_pct, 2),
            "predicted_final_cost_usd": round(predicted_cost_usd, 2),
            "predicted_final_cost_inr": round(predicted_cost_usd * USD_TO_INR, 2),
            "budget_planned_usd": round(budget, 2),
            "top_factors": [f.model_dump() for f in top_factors],
            "recommendations": [r.model_dump() for r in recommendations],
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/projects/upload")
async def upload_projects(file: UploadFile = File(...)):
    """Upload CSV/XLSX file and run batch predictions on all projects."""
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    filename = file.filename or ""
    contents = await file.read()
    
    try:
        if filename.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Upload .csv or .xlsx")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
    
    # Normalize column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    
    # Validate required columns
    missing = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required columns: {', '.join(missing)}. Required: {', '.join(REQUIRED_UPLOAD_COLUMNS)}"
        )
    
    # Run predictions
    results = []
    for idx, row in df.iterrows():
        raw = {col: row[col] for col in REQUIRED_UPLOAD_COLUMNS}
        # Type coerce
        raw["team_size"] = int(raw["team_size"])
        raw["duration_planned_weeks"] = int(raw["duration_planned_weeks"])
        raw["scope_change_count"] = int(raw["scope_change_count"])
        raw["attrition_events"] = int(raw["attrition_events"])
        raw["budget_planned_usd"] = float(raw["budget_planned_usd"])
        raw["seniority_mix_junior"] = float(raw["seniority_mix_junior"])
        raw["seniority_mix_mid"] = float(raw["seniority_mix_mid"])
        raw["seniority_mix_senior"] = float(raw["seniority_mix_senior"])
        raw["employee_cost_ratio"] = float(raw["employee_cost_ratio"])
        raw["weekly_burn_rate_variance"] = float(raw["weekly_burn_rate_variance"])
        raw["industry_type"] = str(raw["industry_type"])
        raw["client_type"] = str(raw["client_type"])
        
        pred = _predict_single_row(raw)
        pred["row_index"] = int(idx)
        pred["project_features"] = raw
        results.append(pred)
    
    # Portfolio summary
    successful = [r for r in results if r.get("status") == "success"]
    risk_counts = {"on_track": 0, "at_risk": 0, "failed": 0}
    total_overrun = 0.0
    total_variance_usd = 0.0
    highest_risk_project = None
    highest_overrun = -999
    
    for r in successful:
        rc = r.get("risk_class", "unknown")
        if rc in risk_counts:
            risk_counts[rc] += 1
        op = r.get("overrun_percentage", 0)
        total_overrun += op
        total_variance_usd += r.get("predicted_final_cost_usd", 0) - r.get("budget_planned_usd", 0)
        if op > highest_overrun:
            highest_overrun = op
            highest_risk_project = r.get("row_index", 0)
    
    avg_overrun = total_overrun / len(successful) if successful else 0
    
    return {
        "total_projects": len(results),
        "successful_predictions": len(successful),
        "failed_predictions": len(results) - len(successful),
        "portfolio_summary": {
            "risk_distribution": risk_counts,
            "average_overrun_pct": round(avg_overrun, 2),
            "total_cost_variance_usd": round(total_variance_usd, 2),
            "total_cost_variance_inr": round(total_variance_usd * USD_TO_INR, 2),
            "highest_risk_project_index": highest_risk_project,
            "highest_overrun_pct": round(highest_overrun, 2),
        },
        "predictions": results,
    }


@app.get("/projects/template")
async def download_template():
    """Download a CSV template with correct column headers and example rows."""
    template_data = {
        "industry_type": ["BFSI", "Healthcare"],
        "team_size": [25, 15],
        "seniority_mix_junior": [0.30, 0.40],
        "seniority_mix_mid": [0.45, 0.35],
        "seniority_mix_senior": [0.25, 0.25],
        "budget_planned_usd": [500000, 250000],
        "duration_planned_weeks": [24, 16],
        "scope_change_count": [4, 2],
        "client_type": ["fixed_bid", "time_and_material"],
        "employee_cost_ratio": [0.58, 0.52],
        "attrition_events": [2, 0],
        "weekly_burn_rate_variance": [0.12, 0.08],
    }
    df = pd.DataFrame(template_data)
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    
    return StreamingResponse(
        io.BytesIO(buf.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=delta_project_template.csv"}
    )


# ─── PDF Report Export ───────────────────────────────────────────────────────

def _generate_pdf_report(features: dict, prediction: dict) -> bytes:
    """Generate a branded PDF risk report using ReportLab."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    from datetime import datetime

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20*mm, bottomMargin=20*mm, leftMargin=20*mm, rightMargin=20*mm)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle('DeltaTitle', parent=styles['Title'], fontSize=22, textColor=colors.HexColor('#2E5CFF'), spaceAfter=6)
    subtitle_style = ParagraphStyle('DeltaSub', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#94A3B8'), spaceAfter=12)
    heading_style = ParagraphStyle('DeltaH2', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor('#E2E8F0'), spaceBefore=16, spaceAfter=8)
    body_style = ParagraphStyle('DeltaBody', parent=styles['Normal'], fontSize=10, textColor=colors.HexColor('#CBD5E1'), leading=14)
    bold_style = ParagraphStyle('DeltaBold', parent=body_style, fontName='Helvetica-Bold', textColor=colors.HexColor('#F1F5F9'))

    elements = []
    now_str = datetime.now().strftime("%B %d, %Y at %H:%M")

    # Header
    elements.append(Paragraph("DELTA AI — Risk Analysis Report", title_style))
    elements.append(Paragraph(f"Generated: {now_str} | DELTA Copilot v2.0", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#334155'), spaceAfter=12))

    # Risk Classification
    risk = prediction.get("risk_class", "unknown").upper()
    conf = prediction.get("risk_confidence", 0) * 100
    risk_color = '#EF4444' if risk == 'FAILED' else '#F59E0B' if risk == 'AT_RISK' else '#22C55E'
    elements.append(Paragraph("Risk Classification", heading_style))
    elements.append(Paragraph(f'<font color="{risk_color}" size="16"><b>{risk}</b></font> — {conf:.0f}% confidence', body_style))
    elements.append(Spacer(1, 8))

    # Financial Summary Table
    elements.append(Paragraph("Financial Summary", heading_style))
    budget = prediction.get("budget_planned_usd", 0)
    cost = prediction.get("predicted_final_cost_usd", 0)
    overrun = prediction.get("overrun_percentage", 0)
    variance = cost - budget

    fin_data = [
        ["Metric", "Value"],
        ["Planned Budget (USD)", f"${budget:,.0f}"],
        ["Predicted Final Cost (USD)", f"${cost:,.0f}"],
        ["Cost Variance (USD)", f"${variance:+,.0f}"],
        ["Overrun Percentage", f"{overrun:+.1f}%"],
        ["Industry", str(features.get("industry_type", "N/A"))],
        ["Team Size", str(features.get("team_size", "N/A"))],
        ["Contract Type", str(features.get("client_type", "N/A")).replace("_", " ").title()],
        ["Duration (Weeks)", str(features.get("duration_planned_weeks", "N/A"))],
    ]

    fin_table = Table(fin_data, colWidths=[200, 280])
    fin_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#CBD5E1')),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0F172A')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#334155')),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    elements.append(fin_table)
    elements.append(Spacer(1, 8))

    # Top Risk Factors
    top_factors = prediction.get("top_factors", [])
    if top_factors:
        elements.append(Paragraph("Top Risk Drivers (SHAP Analysis)", heading_style))
        factor_data = [["Factor", "Impact", "Description"]]
        for f in top_factors[:5]:
            name = f.get("feature", "").replace("_", " ").title()
            impact = "↑ Increases Risk" if f.get("impact") == "increases_risk" else "↓ Reduces Risk"
            desc = f.get("description", "")
            factor_data.append([name, impact, desc[:60]])

        factor_table = Table(factor_data, colWidths=[140, 100, 240])
        factor_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0F172A')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#334155')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(factor_table)
        elements.append(Spacer(1, 8))

    # Recommendations
    recs = prediction.get("recommendations", [])
    if recs:
        elements.append(Paragraph("Recommended Actions", heading_style))
        rec_data = [["Action", "Description", "Est. Risk Reduction"]]
        for r in recs[:3]:
            red = r.get("expected_risk_reduction", 0) * 100
            rec_data.append([r.get("action", ""), r.get("description", "")[:50], f"-{red:.0f}%"])

        rec_table = Table(rec_data, colWidths=[120, 240, 120])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1E293B')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor('#CBD5E1')),
            ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#0F172A')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#334155')),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(rec_table)

    # Footer
    elements.append(Spacer(1, 20))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#334155'), spaceAfter=8))
    elements.append(Paragraph("This report was auto-generated by DELTA AI v2.0 — IT Project Risk Intelligence Platform", 
                              ParagraphStyle('Footer', parent=body_style, fontSize=8, textColor=colors.HexColor('#64748B'))))

    doc.build(elements)
    return buf.getvalue()


@app.post("/report/pdf")
async def generate_pdf_report(req: ReportRequest):
    """Generate a downloadable PDF risk analysis report."""
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    features = req.project_features.model_dump()
    prediction = req.prediction_result

    pdf_bytes = _generate_pdf_report(features, prediction)

    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=DELTA_Risk_Report.pdf"}
    )


# ─── Risk Heatmap Data ──────────────────────────────────────────────────────

FEATURE_DISPLAY_NAMES = {
    "scope_change_count": "Scope Changes",
    "employee_cost_ratio": "Employee Cost Ratio",
    "attrition_events": "Attrition Events",
    "weekly_burn_rate_variance": "Burn Rate Variance",
    "team_size": "Team Size",
    "budget_planned_usd": "Budget (USD)",
    "duration_planned_weeks": "Duration (Weeks)",
    "seniority_mix_junior": "Junior Mix",
    "seniority_mix_mid": "Mid Mix",
    "seniority_mix_senior": "Senior Mix",
    "scope_fixed_bid_pressure": "Scope-FixedBid Pressure",
    "attrition_cost_burden": "Attrition Cost Burden",
    "budget_per_person_week": "Budget/Person/Week",
    "junior_heavy": "Junior Heavy",
    "burn_instability": "Burn Instability",
    "ecr_above_baseline": "ECR Above Baseline",
    "scope_intensity": "Scope Intensity",
    "attrition_rate": "Attrition Rate",
    "client_type_fixed_bid": "Fixed Bid Contract",
    "client_type_outcome_based": "Outcome Based Contract",
    "client_type_time_and_material": "T&M Contract",
}


def _compute_all_shap_values(features_df: pd.DataFrame) -> dict:
    """Compute full SHAP vector for a single project row. Returns {feature: signed_value}."""
    explainer = _state["shap_explainer"]
    if explainer is None:
        return {}
    try:
        shap_values = explainer.shap_values(features_df)
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            signed = np.mean(shap_values[0], axis=1)
        elif isinstance(shap_values, list):
            signed = np.mean([sv[0] for sv in shap_values], axis=0)
        else:
            signed = shap_values[0]

        result = {}
        for i, feat in enumerate(_state["feature_columns"]):
            result[feat] = float(signed[i])
        return result
    except Exception:
        return {}


@app.post("/heatmap/data")
async def heatmap_data(projects: list[dict], top_n: int = 8):
    """Compute SHAP heatmap matrix for a list of projects."""
    if _state["classifier"] is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    rows = []
    all_shap_maps = []

    for idx, proj_raw in enumerate(projects):
        try:
            df_encoded = engineer_features_from_raw(proj_raw)

            # Risk prediction
            risk_proba = _state["classifier"].predict_proba(df_encoded)[0]
            risk_class_idx = int(np.argmax(risk_proba))
            risk_class = _state["label_encoder"].inverse_transform([risk_class_idx])[0]
            risk_confidence = float(risk_proba[risk_class_idx])

            overrun_ratio = float(_state["regressor"].predict(df_encoded)[0])
            overrun_pct = (overrun_ratio - 1.0) * 100

            # Full SHAP values
            shap_map = _compute_all_shap_values(df_encoded)
            all_shap_maps.append(shap_map)

            rows.append({
                "index": idx,
                "industry": str(proj_raw.get("industry_type", "Unknown")),
                "team_size": int(proj_raw.get("team_size", 0)),
                "budget_usd": float(proj_raw.get("budget_planned_usd", 0)),
                "risk_class": risk_class,
                "risk_confidence": round(risk_confidence, 4),
                "overrun_pct": round(overrun_pct, 2),
            })
        except Exception:
            continue

    if not rows:
        raise HTTPException(status_code=400, detail="No valid projects to analyze")

    # Find top N most impactful features across all projects
    feature_importance = {}
    for sm in all_shap_maps:
        for feat, val in sm.items():
            feature_importance[feat] = feature_importance.get(feat, 0) + abs(val)

    sorted_features = sorted(feature_importance.keys(), key=lambda f: feature_importance[f], reverse=True)
    top_features = sorted_features[:top_n]

    # Find global max magnitude for normalization
    global_max = 0.0
    for sm in all_shap_maps:
        for feat in top_features:
            global_max = max(global_max, abs(sm.get(feat, 0)))
    global_max = global_max if global_max > 0 else 1.0

    # Build matrix
    matrix = []
    for i, sm in enumerate(all_shap_maps):
        cells = []
        for feat in top_features:
            raw_val = sm.get(feat, 0)
            normalized = raw_val / global_max  # -1.0 to +1.0
            cells.append({
                "feature": feat,
                "raw_shap": round(raw_val, 6),
                "normalized": round(normalized, 4),
                "direction": "increases_risk" if raw_val > 0 else "reduces_risk",
            })
        matrix.append(cells)

    # Feature column metadata
    columns = []
    for feat in top_features:
        columns.append({
            "key": feat,
            "label": FEATURE_DISPLAY_NAMES.get(feat, feat.replace("_", " ").title()),
        })

    return {
        "projects": rows,
        "columns": columns,
        "matrix": matrix,
        "top_n": top_n,
        "total_features_available": len(sorted_features),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
