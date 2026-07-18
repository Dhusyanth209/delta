"""
Train NASA93 (n=93) Secondary Validation Model.
Target: log(act_effort) using size and cost drivers.
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
    'recordnumber', 'projectname', 'cat2', 'forg', 'center', 'year', 'mode',
    'rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 'acap', 'aexp',
    'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced', 'equivphyskloc', 'act_effort'
]
df = pd.read_csv('data/nasa93.arff', skiprows=314, header=None, names=columns, na_values='?')

# 2. Cleaning per A.2
# Drop cat2, projectname, center, recordnumber, forg, year (just keep mode, kloc, effort, and 15 drivers)
df = df.drop(columns=['cat2', 'projectname', 'center', 'recordnumber', 'forg', 'year'])

# Log-transform equivphyskloc and act_effort
df['log_equivphyskloc'] = np.log1p(df['equivphyskloc'])
df['log_act_effort'] = np.log(df['act_effort'])

# Ordinal encode the 15 COCOMO cost drivers
ordinal_mapping = {'vl': 1, 'l': 2, 'n': 3, 'h': 4, 'vh': 5, 'xh': 6}
cocomo_drivers = [
    'rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 
    'acap', 'aexp', 'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced'
]
for col in cocomo_drivers:
    df[col] = df[col].map(ordinal_mapping)

# One-hot encode mode
df = pd.get_dummies(df, columns=['mode'], drop_first=True)

# Prepare X and y
features = [col for col in df.columns if col not in ['equivphyskloc', 'act_effort', 'log_act_effort']]
X = df[features]
y = df['log_act_effort']

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

print(f"NASA93 CV Results (5-fold):")
print(f"Train R2: {train_r2:.3f}, Val R2: {val_r2:.3f} +/- {val_r2_std:.3f}")
print(f"Train MAE: {train_mae:.3f}, Val MAE: {val_mae:.3f} +/- {val_mae_std:.3f}")

# Generate learning curve
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
plt.title('NASA93 Learning Curve (XGBRegressor)')
plt.legend(loc='best')
plt.grid(True)
plt.savefig('model/diagnostics/nasa93_learning_curve.png')

# 4. Append to docs/real_data_validation.md
doc_content = f"""
## NASA93 (n=93, different task — size-to-effort, not overrun ratio)
**Task:** Predict `log(act_effort)` from `log(equivphyskloc)`, project `mode`, and 15 ordinal COCOMO cost drivers. 
This is a standard size-to-effort software cost estimation dataset, validating that the underlying algorithm (XGBoost) can correctly model standard parametric software metrics.

**Dataset Cleaning & EDA:**
- 93 rows, no missing data.
- COCOMO cost drivers mapped to ordinal integers (1-6).
- `cat2`, `projectname`, and `center` dropped due to high cardinality and low sample size.
- Highly skewed `equivphyskloc` and `act_effort` were log-transformed.

**Model:** XGBoost Regressor (max_depth=2).

**5-Fold Cross-Validation Results:**
- **Validation R²:** {val_r2:.3f} ± {val_r2_std:.3f} (Train R²: {train_r2:.3f})
- **Validation MAE:** {val_mae:.3f} ± {val_mae_std:.3f} (Train MAE: {train_mae:.3f})

![NASA93 Learning Curve](../model/diagnostics/nasa93_learning_curve.png)
"""

with open('docs/real_data_validation.md', 'a') as f:
    f.write(doc_content)

print("Appended NASA93 validation to docs/real_data_validation.md")
