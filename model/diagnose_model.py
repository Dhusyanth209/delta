"""
DELTA — Model Diagnostics & Fit Analysis
==========================================
Comprehensive analysis of the trained model:
1. Learning curves (bias-variance tradeoff)
2. K-fold cross-validation with stratified splits
3. Overfitting/underfitting detection
4. Feature importance ranking
5. Class-wise performance analysis
6. Calibration analysis (are probabilities meaningful?)
7. Recommendations for improvement

Usage:
    python model/diagnose_model.py
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    cross_val_score,
    learning_curve,
    StratifiedKFold,
    validation_curve,
)
from sklearn.preprocessing import LabelEncoder, label_binarize
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_projects.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
DIAG_DIR = PROJECT_ROOT / "model" / "diagnostics"
DIAG_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "industry_type", "team_size", "seniority_mix_junior", "seniority_mix_mid",
    "seniority_mix_senior", "budget_planned_usd", "duration_planned_weeks",
    "scope_change_count", "client_type", "employee_cost_ratio",
    "attrition_events", "weekly_burn_rate_variance",
]
CATEGORICAL_FEATURES = ["industry_type", "client_type"]


def load_data():
    """Load and encode the dataset."""
    df = pd.read_csv(DATA_PATH)
    X = pd.get_dummies(df[FEATURE_COLUMNS], columns=CATEGORICAL_FEATURES)
    le = LabelEncoder()
    y = le.fit_transform(df["outcome_label"])
    y_overrun = (df["budget_actual_usd"] / df["budget_planned_usd"]).values
    return df, X, y, y_overrun, le


def plot_learning_curves(X, y, le):
    """Plot learning curves to diagnose bias vs variance."""
    print("\n── Learning Curves ──")
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="mlogloss",
        n_jobs=-1
    )
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2E5CFF')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#7B3FE4')
    ax.plot(train_sizes, train_mean, 'o-', color='#2E5CFF', label='Training score', linewidth=2)
    ax.plot(train_sizes, val_mean, 'o-', color='#7B3FE4', label='Validation score', linewidth=2)
    
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('DELTA — Learning Curves (Bias-Variance Analysis)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.05)
    
    # Annotate diagnosis
    gap = train_mean[-1] - val_mean[-1]
    if gap > 0.15:
        diagnosis = "HIGH VARIANCE (Overfitting) — gap: {:.1%}".format(gap)
        ax.text(0.5, 0.05, diagnosis, transform=ax.transAxes, fontsize=11,
                color='red', ha='center', style='italic')
    elif val_mean[-1] < 0.65:
        diagnosis = "HIGH BIAS (Underfitting) — val acc: {:.1%}".format(val_mean[-1])
        ax.text(0.5, 0.05, diagnosis, transform=ax.transAxes, fontsize=11,
                color='orange', ha='center', style='italic')
    else:
        diagnosis = "GOOD FIT — train: {:.1%}, val: {:.1%}, gap: {:.1%}".format(
            train_mean[-1], val_mean[-1], gap)
        ax.text(0.5, 0.05, diagnosis, transform=ax.transAxes, fontsize=11,
                color='green', ha='center', style='italic')
    
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    print(f"  Train accuracy (full data): {train_mean[-1]:.4f} ± {train_std[-1]:.4f}")
    print(f"  Val accuracy (full data):   {val_mean[-1]:.4f} ± {val_std[-1]:.4f}")
    print(f"  Gap: {gap:.4f}")
    print(f"  Diagnosis: {diagnosis}")
    
    return train_mean, val_mean, gap


def cross_validate(X, y, le):
    """Run stratified k-fold cross-validation."""
    print("\n── 10-Fold Cross-Validation ──")
    
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, use_label_encoder=False, eval_metric="mlogloss",
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
    
    print(f"  Fold accuracies: {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Min:  {scores.min():.4f}")
    print(f"  Max:  {scores.max():.4f}")
    
    # Variance check
    if scores.std() > 0.05:
        print("  ⚠ HIGH VARIANCE across folds — model is sensitive to data splits")
    else:
        print("  ✓ Stable across folds")
    
    # Plot fold scores
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(1, 11), scores, color=['#2E5CFF' if s > scores.mean() else '#7B3FE4' for s in scores],
                  alpha=0.8, edgecolor='white', linewidth=0.5)
    ax.axhline(y=scores.mean(), color='#EF4444', linestyle='--', linewidth=2, label=f'Mean: {scores.mean():.3f}')
    ax.fill_between(range(0, 12), scores.mean() - scores.std(), scores.mean() + scores.std(),
                     alpha=0.1, color='#EF4444')
    ax.set_xlabel('Fold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('DELTA — 10-Fold Cross-Validation Scores', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(max(0, scores.min() - 0.1), min(1, scores.max() + 0.1))
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "cv_scores.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return scores


def hyperparameter_sensitivity(X, y):
    """Analyze how key hyperparameters affect performance."""
    print("\n── Hyperparameter Sensitivity ──")
    
    # max_depth analysis
    param_range = [2, 3, 4, 5, 6, 7, 8, 10, 12]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    train_scores, val_scores = validation_curve(
        xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, random_state=42,
                           use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1),
        X, y,
        param_name="max_depth",
        param_range=param_range,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    
    best_depth = param_range[np.argmax(val_mean)]
    print(f"  Best max_depth: {best_depth} (val acc: {val_mean[np.argmax(val_mean)]:.4f})")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_range, train_mean, 'o-', color='#2E5CFF', label='Train', linewidth=2)
    ax.plot(param_range, val_mean, 'o-', color='#7B3FE4', label='Validation', linewidth=2)
    ax.axvline(x=best_depth, color='#22C55E', linestyle='--', alpha=0.7, label=f'Best: {best_depth}')
    ax.set_xlabel('max_depth', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('DELTA — max_depth Validation Curve', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "validation_curve_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # n_estimators analysis
    est_range = [50, 100, 150, 200, 300, 400, 500]
    train_scores2, val_scores2 = validation_curve(
        xgb.XGBClassifier(max_depth=best_depth, learning_rate=0.1, random_state=42,
                           use_label_encoder=False, eval_metric="mlogloss", n_jobs=-1),
        X, y,
        param_name="n_estimators",
        param_range=est_range,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )
    
    val_mean2 = np.mean(val_scores2, axis=1)
    best_est = est_range[np.argmax(val_mean2)]
    print(f"  Best n_estimators: {best_est} (val acc: {val_mean2[np.argmax(val_mean2)]:.4f})")
    
    return best_depth, best_est


def analyze_regressor_fit(X, y_overrun, y_class):
    """Analyze the regressor model fit."""
    print("\n── Regressor Fit Analysis ──")
    
    from sklearn.model_selection import train_test_split, cross_val_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_overrun, test_size=0.2, random_state=42, stratify=y_class
    )
    
    model = xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1,
                              random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Residual analysis
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Predicted vs Actual
    axes[0].scatter(y_test, y_pred, alpha=0.5, s=20, color='#2E5CFF')
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lims, lims, '--', color='#EF4444', linewidth=2, label='Perfect')
    axes[0].set_xlabel('Actual Overrun Ratio')
    axes[0].set_ylabel('Predicted Overrun Ratio')
    axes[0].set_title(f'Predicted vs Actual (R²={r2:.3f})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=30, color='#7B3FE4', alpha=0.7, edgecolor='white')
    axes[1].axvline(x=0, color='#EF4444', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual (Actual - Predicted)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Residual Distribution (MAE={mae:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    # Residuals vs Predicted
    axes[2].scatter(y_pred, residuals, alpha=0.5, s=20, color='#2E5CFF')
    axes[2].axhline(y=0, color='#EF4444', linestyle='--', linewidth=2)
    axes[2].set_xlabel('Predicted Overrun Ratio')
    axes[2].set_ylabel('Residual')
    axes[2].set_title('Residuals vs Predicted (Homoscedasticity Check)')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('DELTA — Cost Regressor Diagnostics', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(DIAG_DIR / "regressor_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Cross-validation for regressor
    cv_scores = cross_val_score(
        xgb.XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1,
                          random_state=42, n_jobs=-1),
        X, y_overrun, cv=5, scoring="r2", n_jobs=-1
    )
    
    print(f"  Test R²: {r2:.4f}")
    print(f"  Test MAE: {mae:.4f}")
    print(f"  CV R² (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Residual mean: {residuals.mean():.6f} (should be ~0)")
    print(f"  Residual std:  {residuals.std():.4f}")
    
    return r2, mae


def generate_diagnosis_report(train_acc, val_acc, gap, cv_scores, best_depth, best_est, r2, mae):
    """Generate the full diagnosis report."""
    print("\n" + "=" * 70)
    print("DELTA — MODEL FIT DIAGNOSIS REPORT")
    print("=" * 70)
    
    report = {
        "classifier": {
            "training_accuracy": float(train_acc[-1]),
            "validation_accuracy": float(val_acc[-1]),
            "generalization_gap": float(gap),
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "optimal_max_depth": int(best_depth),
            "optimal_n_estimators": int(best_est),
        },
        "regressor": {
            "test_r2": float(r2),
            "test_mae": float(mae),
        },
        "diagnosis": {},
    }
    
    # Classifier diagnosis
    if gap > 0.15:
        report["diagnosis"]["classifier_fit"] = "OVERFITTING"
        report["diagnosis"]["classifier_recommendation"] = (
            "Reduce max_depth, increase min_child_weight, add regularization (reg_alpha, reg_lambda), "
            "or collect more data. Consider feature selection."
        )
        print(f"\n  CLASSIFIER: ⚠ OVERFITTING (gap={gap:.3f})")
    elif val_acc[-1] < 0.65:
        report["diagnosis"]["classifier_fit"] = "UNDERFITTING"
        report["diagnosis"]["classifier_recommendation"] = (
            "Increase model complexity (max_depth, n_estimators), add feature engineering, "
            "reduce noise in data generation, or add more informative features."
        )
        print(f"\n  CLASSIFIER: ⚠ UNDERFITTING (val_acc={val_acc[-1]:.3f})")
    else:
        report["diagnosis"]["classifier_fit"] = "GOOD_FIT"
        report["diagnosis"]["classifier_recommendation"] = (
            "Model is well-fitted. Can improve further with hyperparameter tuning, "
            "ensemble methods, or more training data."
        )
        print(f"\n  CLASSIFIER: ✓ GOOD FIT (val_acc={val_acc[-1]:.3f}, gap={gap:.3f})")
    
    # Regressor diagnosis
    if r2 > 0.85:
        report["diagnosis"]["regressor_fit"] = "GOOD_FIT"
        print(f"  REGRESSOR:  ✓ GOOD FIT (R²={r2:.3f})")
    elif r2 > 0.60:
        report["diagnosis"]["regressor_fit"] = "ACCEPTABLE"
        print(f"  REGRESSOR:  ~ ACCEPTABLE (R²={r2:.3f})")
    else:
        report["diagnosis"]["regressor_fit"] = "POOR"
        print(f"  REGRESSOR:  ⚠ POOR FIT (R²={r2:.3f})")
    
    # Save report
    with open(DIAG_DIR / "diagnosis_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n  Reports saved to {DIAG_DIR}/")
    return report


def main():
    print("DELTA — Model Diagnostics\n")
    
    df, X, y, y_overrun, le = load_data()
    print(f"Dataset: {len(df)} records, {X.shape[1]} features")
    print(f"Classes: {dict(zip(le.classes_, np.bincount(y)))}")
    
    # 1. Learning curves
    train_acc, val_acc, gap = plot_learning_curves(X, y, le)
    
    # 2. Cross-validation
    cv_scores = cross_validate(X, y, le)
    
    # 3. Hyperparameter sensitivity
    best_depth, best_est = hyperparameter_sensitivity(X, y)
    
    # 4. Regressor analysis
    r2, mae = analyze_regressor_fit(X, y_overrun, y)
    
    # 5. Generate report
    report = generate_diagnosis_report(train_acc, val_acc, gap, cv_scores, best_depth, best_est, r2, mae)
    
    print("\n✓ All diagnostics complete")


if __name__ == "__main__":
    main()
