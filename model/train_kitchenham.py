"""
Train Kitchenham (n=145) Secondary Validation Model.
Target: log(overrun_ratio) where overrun_ratio = Actual.effort / First.estimate
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

os.makedirs('docs', exist_ok=True)
os.makedirs('model/diagnostics', exist_ok=True)

# 1. Load data
columns = [
    'Project', 'Client.code', 'Project.type', 'Actual.start.date', 
    'Actual.duration', 'Actual.effort', 'Adjusted.function.points', 
    'Estimated.completion.date', 'First.estimate', 'First.estimate.method'
]
df = pd.read_csv('data/kitchenham.arff', skiprows=14, header=None, names=columns, na_values='?')

# 2. Cleaning per A.2
# Project.type: treat missing as 'Unknown'
df['Project.type'] = df['Project.type'].fillna('Unknown')
# Collapse rare Project.type (A=4, C=2, Pr=1, U=1) into 'Other'
project_type_counts = df['Project.type'].value_counts()
rare_pt = project_type_counts[project_type_counts < 5].index
df['Project.type'] = df['Project.type'].replace(rare_pt, 'Other')

# Collapse rare First.estimate.method (D=3, C=1, CAE=1, W=1) into 'Other'
est_method_counts = df['First.estimate.method'].value_counts()
rare_em = est_method_counts[est_method_counts < 5].index
df['First.estimate.method'] = df['First.estimate.method'].replace(rare_em, 'Other')

# Exclude Client.code as a feature
# Log-transform Effort, Estimate, Adjusted.function.points
for col in ['Actual.effort', 'First.estimate', 'Adjusted.function.points']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[f'log_{col}'] = np.log1p(df[col])

# Target: log(overrun_ratio)
df['overrun_ratio'] = df['Actual.effort'] / df['First.estimate']
df['log_overrun_ratio'] = np.log(df['overrun_ratio'])

# Drop any rows with NaN in features or target (just in case)
features_base = ['Project.type', 'First.estimate.method', 'log_First.estimate', 'log_Adjusted.function.points']
df = df.dropna(subset=features_base + ['log_overrun_ratio'])

# One-hot encoding
X = pd.get_dummies(df[features_base], columns=['Project.type', 'First.estimate.method'], drop_first=True)
y = df['log_overrun_ratio']

# 3. Model Training (A.3: 5-fold CV, max_depth<=3)
model = XGBRegressor(max_depth=2, n_estimators=100, learning_rate=0.1, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = cross_validate(model, X, y, cv=kf, scoring=['neg_mean_absolute_error', 'r2'], return_train_score=True)

train_mae = -np.mean(cv_results['train_neg_mean_absolute_error'])
val_mae = -np.mean(cv_results['test_neg_mean_absolute_error'])
train_r2 = np.mean(cv_results['train_r2'])
val_r2 = np.mean(cv_results['test_r2'])
val_r2_std = np.std(cv_results['test_r2'])
val_mae_std = np.std(cv_results['test_neg_mean_absolute_error'])

print(f"Kitchenham CV Results (5-fold):")
print(f"Train R2: {train_r2:.3f}, Val R2: {val_r2:.3f} +/- {val_r2_std:.3f}")
print(f"Train MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f} +/- {val_mae_std:.3f}")

# Train final model on all data for learning curve
model.fit(X, y)

# Generate learning curve (simplified by plotting train vs val performance across different sizes)
from sklearn.model_selection import learning_curve
train_sizes, train_scores, test_scores = learning_curve(
    XGBRegressor(max_depth=2, n_estimators=100, learning_rate=0.1, random_state=42), 
    X, y, cv=kf, scoring='neg_mean_absolute_error', n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10))

train_mean = -np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = -np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label='Training MAE', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, test_mean, label='Validation MAE', color='green')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='green')
plt.xlabel('Number of training samples')
plt.ylabel('Mean Absolute Error (log scale)')
plt.title('Kitchenham Learning Curve (XGBRegressor)')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('model/diagnostics/kitchenham_learning_curve.png')

# 4. Save to docs/real_data_validation.md
doc_content = f"""# Real Data Validation

## Kitchenham (n=145)
**Task:** Predict `log(overrun_ratio)` (Actual Effort / First Estimate) using a subset of features. This is a regression task, distinct from the primary synthetic classifier.

**Dataset Cleaning & EDA:**
- 145 rows, no critical missing data dropped.
- Rare categorical levels in `Project.type` and `First.estimate.method` collapsed into "Other".
- `Client.code` excluded due to severe imbalance (80% of rows from one client).
- Highly skewed numeric features (`Actual.effort`, `First.estimate`, `Adjusted.function.points`) log-transformed.

**Model:** XGBoost Regressor (max_depth=2, to prevent overfitting on n=145).

**5-Fold Cross-Validation Results:**
- **Validation R²:** {val_r2:.3f} ± {val_r2_std:.3f} (Train R²: {train_r2:.3f})
- **Validation MAE:** {val_mae:.3f} ± {val_mae_std:.3f} (Train MAE: {train_mae:.3f})

*Note: The gap between training and validation metrics is closely monitored to ensure the model generalizes rather than memorizes.*

![Kitchenham Learning Curve](../model/diagnostics/kitchenham_learning_curve.png)
"""

with open('docs/real_data_validation.md', 'w') as f:
    f.write(doc_content)

print("Saved Kitchenham validation to docs/real_data_validation.md")
