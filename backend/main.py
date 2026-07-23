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

Endpoints:
  POST /predict             → Risk class + cost + SHAP factors + RL recommendations
  GET  /projects/sample     → Sample projects with predictions
  GET  /health              → Health check + model info
  GET  /metrics             → Model training metrics
  GET  /recommend/{idx}     → RL intervention recommendations for sample project
"""

from typing import Optional, Dict, Any, List
import json
import os
import time
import hashlib
from functools import lru_cache
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
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
}

USD_TO_INR = 83.5

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


def _build_copilot_system_prompt(features: dict, prediction: dict) -> str:
    """Build a compact, grounded system prompt optimized for small local LLMs."""
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
    
    return f"""You are DELTA Copilot, an AI assistant for IT project risk analysis.
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

RULES: Only cite numbers from above. Reference factor names when explaining risk. If unsure, say so."""


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

    system_prompt = _build_copilot_system_prompt(features, prediction)

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
            
            system_prompt = _build_copilot_system_prompt(req.project_features, req.prediction_result)
            
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


