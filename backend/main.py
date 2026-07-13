"""
DELTA — FastAPI Backend
========================
Serves the trained XGBoost classifier and cost overrun regressor
for real-time project risk predictions.

Endpoints:
  POST /predict       → Risk class + predicted cost + SHAP factors
  GET  /projects/sample → Sample projects from test set with predictions
  GET  /health        → Health check

Loads model artifacts from /model/artifacts/ at startup.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── Path Setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"

# ─── App Setup ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="DELTA API",
    description="Project Cost-Overrun & Delivery-Risk Prediction",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for hackathon demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load Models at Startup ─────────────────────────────────────────────────
classifier = None
regressor = None
label_encoder = None
feature_columns = None
test_data = None

USD_TO_INR = 83.5  # Approximate exchange rate


def load_models():
    global classifier, regressor, label_encoder, feature_columns, test_data
    print("Loading model artifacts...")

    classifier = joblib.load(ARTIFACTS_DIR / "xgb_classifier.joblib")
    regressor = joblib.load(ARTIFACTS_DIR / "cost_regressor.joblib")
    label_encoder = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
    feature_columns = joblib.load(ARTIFACTS_DIR / "feature_columns.joblib")
    test_data = pd.read_csv(ARTIFACTS_DIR / "test_set_with_predictions.csv")

    print(f"  ✓ Classifier loaded ({type(classifier).__name__})")
    print(f"  ✓ Regressor loaded ({type(regressor).__name__})")
    print(f"  ✓ Label encoder: {list(label_encoder.classes_)}")
    print(f"  ✓ Feature columns: {len(feature_columns)}")
    print(f"  ✓ Test data: {len(test_data)} rows")


@app.on_event("startup")
async def startup():
    load_models()


# ─── Request/Response Models ────────────────────────────────────────────────

class ProjectFeatures(BaseModel):
    """Input features for prediction."""
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
    """A single SHAP-based contributing factor."""
    feature: str
    impact: str  # "positive" or "negative" (toward risk / toward safety)
    magnitude: float
    description: str  # Plain-language explanation


class PredictionResponse(BaseModel):
    """Response from /predict endpoint."""
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


class SampleProject(BaseModel):
    """A sample project with its prediction."""
    project_index: int
    features: dict
    prediction: dict


# ─── SHAP Factor Plain-Language Mapping ──────────────────────────────────────

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
    "seniority_mix_mid": {
        "high": "Mid-level staff provide good cost-effectiveness balance",
        "low": "Lack of mid-level staff may create gaps in day-to-day execution",
    },
}

# For one-hot encoded categorical features
CATEGORICAL_FACTOR_DESCRIPTIONS = {
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
    """Get a plain-language description for a SHAP factor."""
    direction = "high" if shap_value > 0 else "low"

    # Check exact match first
    if feature_name in FACTOR_DESCRIPTIONS:
        return FACTOR_DESCRIPTIONS[feature_name][direction]

    # Check categorical features
    if feature_name in CATEGORICAL_FACTOR_DESCRIPTIONS:
        return CATEGORICAL_FACTOR_DESCRIPTIONS[feature_name][direction]

    # Generic fallback
    if shap_value > 0:
        clean_name = feature_name.replace("_", " ").replace("industry type ", "").replace("client type ", "")
        return f"'{clean_name}' is contributing to elevated risk"
    else:
        clean_name = feature_name.replace("_", " ").replace("industry type ", "").replace("client type ", "")
        return f"'{clean_name}' is helping reduce overall risk"


def compute_shap_factors(features_df: pd.DataFrame) -> list[ShapFactor]:
    """Compute SHAP values for a single prediction and return top 3 factors."""
    try:
        import shap
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(features_df)

        # Handle 3D array (samples × features × classes) or list format
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Average absolute SHAP across classes for this single sample
            mean_shap = np.mean(np.abs(shap_values[0]), axis=1)  # (features,)
            signed_shap = np.mean(shap_values[0], axis=1)  # Keep sign
        elif isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv[0]) for sv in shap_values], axis=0)
            signed_shap = np.mean([sv[0] for sv in shap_values], axis=0)
        else:
            mean_shap = np.abs(shap_values[0])
            signed_shap = shap_values[0]

        # Get top 3 by absolute magnitude
        top_indices = np.argsort(mean_shap)[-3:][::-1]

        factors = []
        for idx in top_indices:
            feature_name = feature_columns[idx]
            shap_val = float(signed_shap[idx])
            magnitude = float(mean_shap[idx])

            factors.append(ShapFactor(
                feature=feature_name,
                impact="increases_risk" if shap_val > 0 else "reduces_risk",
                magnitude=round(magnitude, 4),
                description=get_factor_description(feature_name, shap_val),
            ))

        return factors

    except Exception as e:
        print(f"SHAP computation failed: {e}")
        return [ShapFactor(
            feature="unknown",
            impact="unknown",
            magnitude=0.0,
            description="SHAP analysis unavailable for this prediction",
        )]


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "models_loaded": classifier is not None and regressor is not None,
        "model_type": type(classifier).__name__ if classifier else None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(project: ProjectFeatures):
    """Predict risk class and final cost for a project."""
    if classifier is None or regressor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    # Build feature dict matching the one-hot encoded training format
    raw_features = {
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

    # One-hot encode categoricals to match training format
    df = pd.DataFrame([raw_features])
    df_encoded = pd.get_dummies(df, columns=["industry_type", "client_type"])

    # Ensure all expected columns exist (missing one-hot columns → 0)
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_columns]

    # Predict risk class
    risk_proba = classifier.predict_proba(df_encoded)[0]
    risk_class_idx = int(np.argmax(risk_proba))
    risk_class = label_encoder.inverse_transform([risk_class_idx])[0]
    risk_confidence = float(risk_proba[risk_class_idx])

    # Predict cost overrun ratio
    overrun_ratio = float(regressor.predict(df_encoded)[0])
    predicted_final_cost_usd = project.budget_planned_usd * overrun_ratio
    predicted_final_cost_inr = predicted_final_cost_usd * USD_TO_INR
    overrun_pct = (overrun_ratio - 1.0) * 100

    # SHAP factors
    top_factors = compute_shap_factors(df_encoded)

    # Class probabilities
    class_probs = {
        label_encoder.inverse_transform([i])[0]: float(risk_proba[i])
        for i in range(len(risk_proba))
    }

    return PredictionResponse(
        risk_class=risk_class,
        risk_confidence=round(risk_confidence, 4),
        predicted_overrun_ratio=round(overrun_ratio, 4),
        predicted_final_cost_usd=round(predicted_final_cost_usd, 2),
        predicted_final_cost_inr=round(predicted_final_cost_inr, 2),
        budget_planned_usd=round(project.budget_planned_usd, 2),
        budget_planned_inr=round(project.budget_planned_usd * USD_TO_INR, 2),
        overrun_percentage=round(overrun_pct, 2),
        top_factors=top_factors,
        class_probabilities=class_probs,
    )


@app.get("/projects/sample")
async def sample_projects():
    """Return 8 sample projects from the test set with their real predictions."""
    if test_data is None:
        raise HTTPException(status_code=503, detail="Test data not loaded")

    # Pick 8 diverse samples: mix of on_track, at_risk, failed
    samples = []
    for label in ["on_track", "at_risk", "failed"]:
        label_data = test_data[test_data["outcome_predicted"] == label]
        if len(label_data) > 0:
            n_pick = min(3 if label != "failed" else 2, len(label_data))
            picked = label_data.sample(n=n_pick, random_state=42)
            samples.append(picked)

    if not samples:
        raise HTTPException(status_code=404, detail="No sample data available")

    sample_df = pd.concat(samples).head(8)

    projects = []
    for idx, (_, row) in enumerate(sample_df.iterrows()):
        # Reconstruct original features from one-hot encoded
        features = {}

        # Numeric features
        for col in ["team_size", "seniority_mix_junior", "seniority_mix_mid",
                     "seniority_mix_senior", "budget_planned_usd",
                     "duration_planned_weeks", "scope_change_count",
                     "employee_cost_ratio", "attrition_events",
                     "weekly_burn_rate_variance"]:
            if col in row.index:
                val = row[col]
                features[col] = float(val) if not isinstance(val, (int, np.integer)) else int(val)

        # Decode one-hot industry_type
        for col in row.index:
            if col.startswith("industry_type_") and row[col] == 1:
                features["industry_type"] = col.replace("industry_type_", "")
            if col.startswith("client_type_") and row[col] == 1:
                features["client_type"] = col.replace("client_type_", "")

        # Defaults if not found
        features.setdefault("industry_type", "Unknown")
        features.setdefault("client_type", "unknown")

        # Add INR values
        features["budget_planned_inr"] = round(features.get("budget_planned_usd", 0) * USD_TO_INR, 2)

        # Prediction data
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

        projects.append(SampleProject(
            project_index=idx,
            features=features,
            prediction=prediction,
        ))

    return {"projects": projects, "total": len(projects)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
