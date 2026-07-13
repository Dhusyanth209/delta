"""
DELTA — Model Training Pipeline
================================
Trains two models on the synthetic IT project dataset:
1. XGBoost Classifier: predicts outcome_label (on_track / at_risk / failed)
2. Gradient Boosting Regressor: predicts budget overrun ratio (continuous)

Also generates:
- SHAP explainability analysis (summary bar plot)
- Confusion matrix visualization
- Metrics JSON for documentation

Usage:
    python model/train_model.py
"""

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_projects.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Features used for prediction (excludes IDs, actuals, and target)
FEATURE_COLUMNS = [
    "industry_type",
    "team_size",
    "seniority_mix_junior",
    "seniority_mix_mid",
    "seniority_mix_senior",
    "budget_planned_usd",
    "duration_planned_weeks",
    "scope_change_count",
    "client_type",
    "employee_cost_ratio",
    "attrition_events",
    "weekly_burn_rate_variance",
]

CATEGORICAL_FEATURES = ["industry_type", "client_type"]


def load_and_prepare_data():
    """Load dataset and prepare features/targets."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df[FEATURE_COLUMNS], columns=CATEGORICAL_FEATURES)

    # Target: outcome label
    le = LabelEncoder()
    y_class = le.fit_transform(df["outcome_label"])

    # Target: budget overrun ratio (for regression)
    y_overrun = df["budget_actual_usd"] / df["budget_planned_usd"]

    feature_names = list(df_encoded.columns)

    return df, df_encoded, y_class, y_overrun, le, feature_names


def train_classifier(X_train, X_test, y_train, y_test, le, feature_names):
    """Train XGBoost multi-class classifier for risk prediction."""
    print("\n" + "=" * 60)
    print("TRAINING XGBOOST CLASSIFIER")
    print("=" * 60)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        use_label_encoder=False,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True
    )
    report_text = classification_report(
        y_test, y_pred,
        target_names=le.classes_
    )

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"\nClassification Report:\n{report_text}")

    # Check for suspicious accuracy
    if accuracy > 0.98:
        print("\n⚠️  WARNING: Accuracy > 98% — dataset may be too deterministic!")
        print("    Consider adding more noise in generate_dataset.py")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("DELTA Risk Classifier — Confusion Matrix")
    plt.tight_layout()
    cm_path = ARTIFACTS_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nConfusion matrix saved to {cm_path}")

    return model, accuracy, report, y_pred, y_pred_proba


def train_regressor(X_train, X_test, y_train_reg, y_test_reg):
    """Train Gradient Boosting Regressor for continuous cost prediction."""
    print("\n" + "=" * 60)
    print("TRAINING COST OVERRUN REGRESSOR")
    print("=" * 60)

    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(
        X_train, y_train_reg,
        eval_set=[(X_test, y_test_reg)],
        verbose=False,
    )

    # Predictions
    y_pred_reg = model.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)

    print(f"\nRegression Metrics:")
    print(f"  MAE:  {mae:.4f} (mean absolute error in overrun ratio)")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")

    return model, {"mae": mae, "rmse": rmse, "r2": r2}


def generate_shap_analysis(model, X_test, feature_names):
    """Generate SHAP value analysis for model explainability."""
    print("\n" + "=" * 60)
    print("GENERATING SHAP ANALYSIS")
    print("=" * 60)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Handle different SHAP output formats:
        # - Newer XGBoost/SHAP: 3D ndarray (samples × features × classes)
        # - Older: list of 2D arrays [class0_array, class1_array, ...]
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # 3D array: (n_samples, n_features, n_classes)
            mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))  # Average over samples and classes
        elif isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            mean_shap = np.mean(mean_shap, axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)

        # Ensure 1D
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.flatten()[:len(feature_names)]

        # Mean importance per feature
        n_features = min(15, len(feature_names))
        sorted_idx = np.argsort(mean_shap)[-n_features:]

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(
            range(len(sorted_idx)),
            mean_shap[sorted_idx],
            color=plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
        )
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("DELTA — Feature Importance (SHAP Analysis)")
        plt.tight_layout()

        shap_path = ARTIFACTS_DIR / "shap_summary.png"
        fig.savefig(shap_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"SHAP summary plot saved to {shap_path}")

        # Also try the built-in SHAP summary plot
        try:
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                # Use mean across classes for summary plot
                shap_mean_class = np.mean(np.abs(shap_values), axis=2)
                shap.summary_plot(
                    shap_mean_class, X_test,
                    feature_names=feature_names,
                    show=False, plot_type="bar"
                )
            elif isinstance(shap_values, list):
                shap.summary_plot(
                    shap_values[0], X_test,
                    feature_names=feature_names,
                    show=False, plot_type="bar"
                )
            else:
                shap.summary_plot(
                    shap_values, X_test,
                    feature_names=feature_names,
                    show=False, plot_type="bar"
                )
            shap_detail_path = ARTIFACTS_DIR / "shap_detailed.png"
            plt.savefig(shap_detail_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"SHAP detailed plot saved to {shap_detail_path}")
        except Exception as e:
            print(f"  (Detailed SHAP plot skipped: {e})")

        return shap_values, explainer

    except ImportError:
        print("  ⚠️ SHAP not installed. Skipping SHAP analysis.")
        return None, None


def save_artifacts(classifier, regressor, le, feature_names, accuracy,
                   class_report, reg_metrics, df, X_test, y_test,
                   y_pred, y_pred_proba, y_test_reg):
    """Save all model artifacts and metrics."""
    print("\n" + "=" * 60)
    print("SAVING ARTIFACTS")
    print("=" * 60)

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save models
    joblib.dump(classifier, ARTIFACTS_DIR / "xgb_classifier.joblib")
    joblib.dump(regressor, ARTIFACTS_DIR / "cost_regressor.joblib")
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")
    joblib.dump(feature_names, ARTIFACTS_DIR / "feature_columns.joblib")
    print("  ✓ Models saved")

    # Save test data for the API's /projects/sample endpoint
    test_data = pd.DataFrame(X_test, columns=feature_names)
    test_data["outcome_actual"] = le.inverse_transform(y_test)
    test_data["outcome_predicted"] = le.inverse_transform(y_pred)
    test_data["overrun_ratio_actual"] = y_test_reg.values if hasattr(y_test_reg, 'values') else y_test_reg
    test_data["prediction_confidence"] = np.max(y_pred_proba, axis=1)
    test_data.to_csv(ARTIFACTS_DIR / "test_set_with_predictions.csv", index=False)
    print("  ✓ Test set with predictions saved")

    # Save metrics JSON
    metrics = {
        "classifier": {
            "accuracy": float(accuracy),
            "per_class": {
                cls: {
                    "precision": float(class_report[cls]["precision"]),
                    "recall": float(class_report[cls]["recall"]),
                    "f1_score": float(class_report[cls]["f1-score"]),
                    "support": int(class_report[cls]["support"]),
                }
                for cls in le.classes_
            },
            "model_type": "XGBClassifier",
            "n_estimators": 200,
            "max_depth": 6,
            "test_size": TEST_SIZE,
        },
        "regressor": {
            "mae": float(reg_metrics["mae"]),
            "rmse": float(reg_metrics["rmse"]),
            "r2": float(reg_metrics["r2"]),
            "model_type": "XGBRegressor",
            "target": "budget_overrun_ratio",
        },
        "dataset": {
            "total_records": len(df),
            "features": len(feature_names),
            "train_size": len(X_test) * 4,  # Approximate
            "test_size": len(X_test),
        },
    }

    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  ✓ Metrics JSON saved")

    print(f"\nAll artifacts saved to {ARTIFACTS_DIR}")


def main():
    # Load data
    df, X_encoded, y_class, y_overrun, le, feature_names = load_and_prepare_data()

    # Split data 80/20 (stratified for classifier)
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_class,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class
    )

    # Also split the overrun ratio for regression
    _, _, y_train_reg, y_test_reg = train_test_split(
        X_encoded, y_overrun,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_class  # Same split
    )

    print(f"Train set: {len(X_train)} | Test set: {len(X_test)}")

    # Train classifier
    classifier, accuracy, class_report, y_pred, y_pred_proba = train_classifier(
        X_train, X_test, y_train, y_test, le, feature_names
    )

    # Train regressor
    regressor, reg_metrics = train_regressor(
        X_train, X_test, y_train_reg, y_test_reg
    )

    # SHAP analysis
    shap_values, explainer = generate_shap_analysis(
        classifier, X_test, feature_names
    )

    # Save everything
    save_artifacts(
        classifier, regressor, le, feature_names, accuracy,
        class_report, reg_metrics, df, X_test, y_test,
        y_pred, y_pred_proba, y_test_reg
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Classifier accuracy: {accuracy:.4f}")
    print(f"Regressor MAE: {reg_metrics['mae']:.4f}")
    print(f"Regressor R²:  {reg_metrics['r2']:.4f}")


if __name__ == "__main__":
    main()
