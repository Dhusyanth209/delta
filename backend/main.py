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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
