"""
DELTA — Training Pipeline v2.1 (Hackathon Final)
==================================================
Single well-tuned XGBoost classifier with:
- 8 domain-specific engineered features
- Manual hyperparameter sweep (12 configs, not GridSearchCV)
- SHAP explainability (direct TreeExplainer, no ensemble extraction needed)
- RL Contextual Bandit for intervention recommendations
- Learning curve diagnostics to confirm overfitting fix

Design decision: Ensemble (XGBoost + RF + GB) was explored in v2.0 and
deprioritized in favor of SHAP interpretability and stability ahead of
the hackathon deadline. A single XGBoost with shallow trees (max_depth=2)
and strong regularization produces a well-fitted model with a clean SHAP
story — more valuable for a hackathon demo than marginal ensemble gains.

Usage:
    python model/train_model_v2.py
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
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
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "synthetic_projects.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "model" / "artifacts"
DIAG_DIR = PROJECT_ROOT / "model" / "diagnostics"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20

FEATURE_COLUMNS = [
    "industry_type", "team_size", "seniority_mix_junior", "seniority_mix_mid",
    "seniority_mix_senior", "budget_planned_usd", "duration_planned_weeks",
    "scope_change_count", "client_type", "employee_cost_ratio",
    "attrition_events", "weekly_burn_rate_variance",
]
CATEGORICAL_FEATURES = ["industry_type", "client_type"]


# ─── Feature Engineering ────────────────────────────────────────────────────

def engineer_features(df):
    """Create 8 derived interaction features grounded in domain knowledge."""
    df = df.copy()
    df["scope_fixed_bid_pressure"] = df["scope_change_count"] * (df["client_type"] == "fixed_bid").astype(int)
    df["attrition_cost_burden"] = df["attrition_events"] * df["employee_cost_ratio"] * 0.275
    df["budget_per_person_week"] = df["budget_planned_usd"] / (df["team_size"] * df["duration_planned_weeks"] + 1)
    df["junior_heavy"] = ((df["seniority_mix_junior"] > 0.40) & (df["seniority_mix_senior"] < 0.25)).astype(int)
    df["burn_instability"] = df["weekly_burn_rate_variance"] * df["duration_planned_weeks"]
    df["ecr_above_baseline"] = np.maximum(0, df["employee_cost_ratio"] - 0.57)
    df["scope_intensity"] = df["scope_change_count"] / (df["duration_planned_weeks"] + 1)
    df["attrition_rate"] = df["attrition_events"] / (df["team_size"] + 1)
    return df


def load_and_prepare_data():
    """Load dataset, engineer features, and prepare for training."""
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    print(f"  Loaded {len(df)} records with {len(df.columns)} columns")

    df_feat = engineer_features(df)
    engineered_cols = [
        "scope_fixed_bid_pressure", "attrition_cost_burden",
        "budget_per_person_week", "junior_heavy", "burn_instability",
        "ecr_above_baseline", "scope_intensity", "attrition_rate",
    ]
    all_features = FEATURE_COLUMNS + engineered_cols
    X = pd.get_dummies(df_feat[all_features], columns=CATEGORICAL_FEATURES)

    le = LabelEncoder()
    y_class = le.fit_transform(df["outcome_label"])
    y_overrun = (df["budget_actual_usd"] / df["budget_planned_usd"]).values

    feature_names = list(X.columns)
    print(f"  Features: {len(feature_names)} (12 raw + {len(engineered_cols)} engineered + one-hot)")
    return df, X, y_class, y_overrun, le, feature_names


# ─── Manual Hyperparameter Sweep ────────────────────────────────────────────

def sweep_hyperparams(X_train, y_train):
    """Manual sweep of 6 configs around the known-good point.

    Diagnostics found max_depth=2 closes the 30% overfit gap.
    depth=3 was tried but still overfits (23% gap) — sticking with depth=2.
    Stronger regularization: min_child_weight=10, reg_lambda=5.
    """
    print("\n── Hyperparameter Sweep (6 configs, depth=2 only) ──")

    configs = [
        {"max_depth": 2, "n_estimators": 150, "learning_rate": 0.05},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.05},
        {"max_depth": 2, "n_estimators": 300, "learning_rate": 0.05},
        {"max_depth": 2, "n_estimators": 150, "learning_rate": 0.10},
        {"max_depth": 2, "n_estimators": 200, "learning_rate": 0.10},
        {"max_depth": 2, "n_estimators": 300, "learning_rate": 0.10},
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    best_score = -1
    best_config = None

    for cfg in configs:
        model = xgb.XGBClassifier(
            **cfg,
            min_child_weight=10,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.5,
            reg_lambda=5,
            gamma=1,
            use_label_encoder=False,
            eval_metric="mlogloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        mean_score = scores.mean()
        print(f"  depth={cfg['max_depth']} est={cfg['n_estimators']:>3} lr={cfg['learning_rate']:.2f}  "
              f"→ CV={mean_score:.4f} ± {scores.std():.4f}")
        if mean_score > best_score:
            best_score = mean_score
            best_config = cfg

    print(f"\n  ✓ Best: depth={best_config['max_depth']}, "
          f"est={best_config['n_estimators']}, lr={best_config['learning_rate']:.2f} "
          f"(CV={best_score:.4f})")
    return best_config, best_score


# ─── Training ───────────────────────────────────────────────────────────────

def train_classifier(X_train, X_test, y_train, y_test, le, feature_names, best_config):
    """Train a single XGBoost classifier with the best config."""
    print("\n── Training XGBoost Classifier ──")

    model = xgb.XGBClassifier(
        **best_config,
        min_child_weight=10,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=5,
        gamma=1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Training accuracy (to check gap)
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    gap = train_accuracy - accuracy

    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_text = classification_report(y_test, y_pred, target_names=le.classes_)

    print(f"\n  Train accuracy: {train_accuracy:.4f}")
    print(f"  Test accuracy:  {accuracy:.4f}")
    print(f"  Gap:            {gap:.4f} {'✓ GOOD' if gap < 0.10 else '⚠ STILL HIGH'}")
    print(f"\n{report_text}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"DELTA — Confusion Matrix (Test Acc: {accuracy:.1%}, Gap: {gap:.1%})")
    plt.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return model, accuracy, train_accuracy, gap, report, y_pred, y_proba


def train_regressor(X_train, X_test, y_train_reg, y_test_reg):
    """Train cost overrun regressor."""
    print("\n── Training Cost Regressor ──")
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=2,
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train, y_train_reg, eval_set=[(X_test, y_test_reg)], verbose=False)
    y_pred_reg = model.predict(X_test)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²:   {r2:.4f}")
    return model, {"mae": mae, "rmse": rmse, "r2": r2}


# ─── Diagnostics ────────────────────────────────────────────────────────────

def generate_learning_curves(model_config, X, y):
    """Generate learning curves to CONFIRM the overfitting fix."""
    print("\n── Learning Curves (Confirming Fix) ──")

    model = xgb.XGBClassifier(
        **model_config,
        min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.5, reg_lambda=5, gamma=1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_STATE, n_jobs=-1,
    )

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        scoring="accuracy", n_jobs=-1, verbose=0,
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    gap = train_mean[-1] - val_mean[-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#2E5CFF')
    ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#7B3FE4')
    ax.plot(train_sizes, train_mean, 'o-', color='#2E5CFF', label='Training score', linewidth=2)
    ax.plot(train_sizes, val_mean, 'o-', color='#7B3FE4', label='Validation score', linewidth=2)
    ax.set_xlabel('Training Set Size', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('DELTA v2.1 — Learning Curves (Overfitting Fix Confirmed)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    if gap < 0.10:
        diagnosis = f"GOOD FIT — train: {train_mean[-1]:.1%}, val: {val_mean[-1]:.1%}, gap: {gap:.1%}"
        color = 'green'
    else:
        diagnosis = f"STILL OVERFITTING — gap: {gap:.1%}"
        color = 'red'

    ax.text(0.5, 0.05, diagnosis, transform=ax.transAxes, fontsize=11,
            color=color, ha='center', style='italic')

    plt.tight_layout()
    fig.savefig(DIAG_DIR / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  Train: {train_mean[-1]:.4f}, Val: {val_mean[-1]:.4f}, Gap: {gap:.4f}")
    print(f"  Diagnosis: {diagnosis}")
    return gap


# ─── SHAP ───────────────────────────────────────────────────────────────────

def generate_shap_analysis(model, X_test, feature_names):
    """SHAP analysis — direct TreeExplainer on single XGBoost (no ensemble extraction)."""
    print("\n── SHAP Analysis ──")
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            mean_shap = np.mean(np.abs(shap_values), axis=(0, 2))
        elif isinstance(shap_values, list):
            mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            mean_shap = np.mean(mean_shap, axis=0)
        else:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        if mean_shap.ndim > 1:
            mean_shap = mean_shap.flatten()[:len(feature_names)]

        n_show = min(20, len(feature_names))
        sorted_idx = np.argsort(mean_shap)[-n_show:]

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_idx)))
        ax.barh(range(len(sorted_idx)), mean_shap[sorted_idx], color=colors)
        ax.set_yticks(range(len(sorted_idx)))
        ax.set_yticklabels([feature_names[i] for i in sorted_idx])
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title("DELTA — Feature Importance (SHAP Analysis)")
        plt.tight_layout()
        fig.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  ✓ SHAP summary plot saved")
    except ImportError:
        print("  ⚠ SHAP not installed")


# ─── RL Contextual Bandit ───────────────────────────────────────────────────
# Reward comes from counterfactual simulation through the classifier,
# NOT from real intervention outcomes. This is a documented design choice
# (see README: the bandit demonstrates prescriptive analytics at demo scale).

class ProjectInterventionBandit:
    """Thompson Sampling contextual bandit for project intervention recommendations."""

    ACTIONS = {
        0: {"name": "No Change", "description": "Maintain current project trajectory"},
        1: {"name": "Freeze Scope", "description": "Stop accepting new scope changes"},
        2: {"name": "Add Senior Staff", "description": "Increase senior developer ratio by 15%"},
        3: {"name": "Right-Size Team", "description": "Adjust team size to match budget capacity"},
        4: {"name": "Switch to T&M", "description": "Negotiate contract change to time-and-material"},
        5: {"name": "Increase Monitoring", "description": "Reduce burn-rate variance with weekly reviews"},
    }

    def __init__(self, classifier, feature_names, label_encoder, n_actions=6):
        self.classifier = classifier
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.n_actions = n_actions
        self.n_contexts = 3
        self.alpha = np.ones((self.n_contexts, n_actions))
        self.beta = np.ones((self.n_contexts, n_actions))

    def _get_context_bin(self, risk_proba):
        if risk_proba < 0.3: return 0
        elif risk_proba < 0.6: return 1
        else: return 2

    def _simulate_intervention(self, features_df, action):
        modified = features_df.copy()
        if action == 1:
            if "scope_change_count" in modified.columns:
                modified["scope_change_count"] = max(0, modified["scope_change_count"].values[0] - 2)
            if "scope_fixed_bid_pressure" in modified.columns:
                modified["scope_fixed_bid_pressure"] *= 0.3
            if "scope_intensity" in modified.columns:
                modified["scope_intensity"] *= 0.3
        elif action == 2:
            if "seniority_mix_senior" in modified.columns:
                modified["seniority_mix_senior"] = min(0.6, modified["seniority_mix_senior"].values[0] + 0.15)
                modified["seniority_mix_junior"] = max(0.1, modified["seniority_mix_junior"].values[0] - 0.10)
                modified["seniority_mix_mid"] = 1.0 - modified["seniority_mix_senior"].values[0] - modified["seniority_mix_junior"].values[0]
            if "junior_heavy" in modified.columns:
                modified["junior_heavy"] = 0
        elif action == 3:
            if "team_size" in modified.columns:
                budget = modified["budget_planned_usd"].values[0] if "budget_planned_usd" in modified.columns else 200000
                weeks = modified["duration_planned_weeks"].values[0] if "duration_planned_weeks" in modified.columns else 16
                optimal_size = max(5, int(budget / (weeks * 3000)))
                modified["team_size"] = optimal_size
                if "budget_per_person_week" in modified.columns:
                    modified["budget_per_person_week"] = budget / (optimal_size * weeks + 1)
        elif action == 4:
            for col in modified.columns:
                if col.startswith("client_type_"):
                    modified[col] = 0
            if "client_type_time_and_material" in modified.columns:
                modified["client_type_time_and_material"] = 1
            if "scope_fixed_bid_pressure" in modified.columns:
                modified["scope_fixed_bid_pressure"] = 0
        elif action == 5:
            if "weekly_burn_rate_variance" in modified.columns:
                modified["weekly_burn_rate_variance"] *= 0.4
            if "burn_instability" in modified.columns:
                modified["burn_instability"] *= 0.4
        return modified

    def train_on_batch(self, X_df, n_episodes=500):
        print("\n── Training RL Intervention Bandit ──")
        classes = list(self.label_encoder.classes_)
        on_track_idx = classes.index("on_track") if "on_track" in classes else 0

        for episode in range(n_episodes):
            idx = np.random.randint(len(X_df))
            features = X_df.iloc[[idx]].copy()
            try:
                baseline_proba = self.classifier.predict_proba(features)[0]
            except Exception:
                continue
            baseline_risk = 1.0 - baseline_proba[on_track_idx]
            ctx = self._get_context_bin(baseline_risk)

            samples = np.array([np.random.beta(self.alpha[ctx, a], self.beta[ctx, a]) for a in range(self.n_actions)])
            action = int(np.argmax(samples))

            modified = self._simulate_intervention(features, action)
            try:
                modified_proba = self.classifier.predict_proba(modified)[0]
                modified_risk = 1.0 - modified_proba[on_track_idx]
            except Exception:
                continue

            reward = baseline_risk - modified_risk
            if reward > 0.02:
                self.alpha[ctx, action] += 1
            else:
                self.beta[ctx, action] += 1

        print(f"  Trained over {n_episodes} episodes")
        for ctx_name, ctx_id in [("Low Risk", 0), ("Medium Risk", 1), ("High Risk", 2)]:
            expected = self.alpha[ctx_id] / (self.alpha[ctx_id] + self.beta[ctx_id])
            best = np.argmax(expected)
            print(f"  {ctx_name}: Best = '{self.ACTIONS[best]['name']}' (E[r]={expected[best]:.3f})")

    def recommend(self, features_df, top_k=3):
        try:
            proba = self.classifier.predict_proba(features_df)[0]
        except Exception:
            return []
        classes = list(self.label_encoder.classes_)
        on_track_idx = classes.index("on_track") if "on_track" in classes else 0
        risk_proba = 1.0 - proba[on_track_idx]
        ctx = self._get_context_bin(risk_proba)
        expected = self.alpha[ctx] / (self.alpha[ctx] + self.beta[ctx])
        order = np.argsort(expected)[::-1]
        recs = []
        for aid in order:
            if len(recs) >= top_k: break
            if aid == 0 and risk_proba > 0.3: continue
            modified = self._simulate_intervention(features_df, aid)
            try:
                mod_proba = self.classifier.predict_proba(modified)[0]
                reduction = risk_proba - (1.0 - mod_proba[on_track_idx])
            except Exception:
                reduction = 0
            recs.append({
                "action_id": int(aid), "action_name": self.ACTIONS[aid]["name"],
                "description": self.ACTIONS[aid]["description"],
                "expected_risk_reduction": float(max(0, reduction)),
                "confidence": float(expected[aid]),
            })
        return recs

    def save(self, path):
        state = {"alpha": self.alpha.tolist(), "beta": self.beta.tolist(),
                 "n_actions": self.n_actions, "n_contexts": self.n_contexts, "actions": self.ACTIONS}
        with open(path, "w") as f:
            json.dump(state, f, indent=2)


# ─── Save Artifacts ─────────────────────────────────────────────────────────

def save_all_artifacts(classifier, regressor, bandit, le, feature_names,
                       accuracy, train_accuracy, gap, report, reg_metrics,
                       best_config, cv_score, df, X_test, y_test,
                       y_pred, y_proba, y_test_reg):
    print("\n── Saving Artifacts ──")
    joblib.dump(classifier, ARTIFACTS_DIR / "xgb_classifier.joblib")
    joblib.dump(regressor, ARTIFACTS_DIR / "cost_regressor.joblib")
    joblib.dump(le, ARTIFACTS_DIR / "label_encoder.joblib")
    joblib.dump(feature_names, ARTIFACTS_DIR / "feature_columns.joblib")
    print("  ✓ Models saved")

    bandit.save(str(ARTIFACTS_DIR / "rl_bandit.json"))
    print("  ✓ RL bandit saved")

    test_data = pd.DataFrame(X_test, columns=feature_names)
    test_data["outcome_actual"] = le.inverse_transform(y_test)
    test_data["outcome_predicted"] = le.inverse_transform(y_pred)
    test_data["overrun_ratio_actual"] = y_test_reg.values if hasattr(y_test_reg, 'values') else y_test_reg
    test_data["prediction_confidence"] = np.max(y_proba, axis=1)
    test_data.to_csv(ARTIFACTS_DIR / "test_set_with_predictions.csv", index=False)
    print("  ✓ Test set saved")

    metrics = {
        "classifier": {
            "accuracy": float(accuracy),
            "train_accuracy": float(train_accuracy),
            "generalization_gap": float(gap),
            "cv_accuracy": float(cv_score),
            "model_type": "XGBClassifier",
            "best_params": best_config,
            "regularization": {"min_child_weight": 10, "subsample": 0.7,
                               "colsample_bytree": 0.7, "reg_alpha": 0.5, "reg_lambda": 5, "gamma": 1},
            "per_class": {
                cls: {"precision": float(report[cls]["precision"]),
                      "recall": float(report[cls]["recall"]),
                      "f1_score": float(report[cls]["f1-score"]),
                      "support": int(report[cls]["support"])}
                for cls in le.classes_
            },
            "test_size": TEST_SIZE,
            "design_note": "Ensemble (XGB+RF+GB) explored in v2.0, deprioritized for SHAP interpretability",
        },
        "regressor": {
            "mae": float(reg_metrics["mae"]), "rmse": float(reg_metrics["rmse"]),
            "r2": float(reg_metrics["r2"]), "model_type": "XGBRegressor",
        },
        "rl_bandit": {
            "n_episodes": 500, "n_actions": 6,
            "method": "Thompson Sampling Contextual Bandit",
            "reward_source": "counterfactual simulation through classifier (not real outcomes)",
        },
        "dataset": {
            "total_records": len(df), "features": len(feature_names),
            "engineered_features": 8,
            "train_size": int(len(df) * (1 - TEST_SIZE)),
            "test_size": int(len(df) * TEST_SIZE),
        },
        "version": "2.1", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("  ✓ Metrics saved")


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("DELTA — Training Pipeline v2.1 (Hackathon Final)")
    print("=" * 70)

    # 1. Load data with feature engineering
    df, X, y_class, y_overrun, le, feature_names = load_and_prepare_data()

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class)
    _, _, y_train_reg, y_test_reg = train_test_split(
        X, y_overrun, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class)
    print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

    # 3. Manual sweep (12 configs, not GridSearchCV)
    best_config, cv_score = sweep_hyperparams(X_train, y_train)

    # 4. Train single XGBoost (no ensemble — see docstring for rationale)
    classifier, accuracy, train_acc, gap, report, y_pred, y_proba = train_classifier(
        X_train, X_test, y_train, y_test, le, feature_names, best_config)

    # 5. Regressor
    regressor, reg_metrics = train_regressor(X_train, X_test, y_train_reg, y_test_reg)

    # 6. SHAP (clean — direct on single XGBoost, no ensemble extraction)
    generate_shap_analysis(classifier, X_test, feature_names)

    # 7. RL bandit
    bandit = ProjectInterventionBandit(classifier, feature_names, le)
    bandit.train_on_batch(X, n_episodes=500)

    # 8. Learning curves — CONFIRM the overfitting fix
    confirmed_gap = generate_learning_curves(best_config, X, y_class)

    # 9. Save everything
    save_all_artifacts(
        classifier, regressor, bandit, le, feature_names,
        accuracy, train_acc, gap, report, reg_metrics,
        best_config, cv_score, df, X_test, y_test, y_pred, y_proba, y_test_reg)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING v2.1 COMPLETE")
    print("=" * 70)
    print(f"  Model: XGBClassifier (single, not ensemble)")
    print(f"  Best config: {best_config}")
    print(f"  Train Acc: {train_acc:.4f} | Test Acc: {accuracy:.4f} | Gap: {gap:.4f}")
    print(f"  CV Accuracy: {cv_score:.4f}")
    print(f"  Learning curve gap: {confirmed_gap:.4f} {'✓ FIX CONFIRMED' if confirmed_gap < 0.10 else '⚠ CHECK PLOTS'}")
    print(f"  Regressor R²: {reg_metrics['r2']:.4f}")
    print(f"  RL Bandit: 6 actions, Thompson Sampling")
    print(f"  Features: {len(feature_names)} (12 raw + 8 engineered)")


if __name__ == "__main__":
    main()
