import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure directories exist
os.makedirs("data/eda", exist_ok=True)
os.makedirs("model/diagnostics", exist_ok=True)

print("Loading Kitchenham dataset...")
# Load as CSV skipping the ARFF header
columns = [
    'Project', 'Client.code', 'Project.type', 'Actual.start.date', 
    'Actual.duration', 'Actual.effort', 'Adjusted.function.points', 
    'Estimated.completion.date', 'First.estimate', 'First.estimate.method'
]
df = pd.read_csv('data/kitchenham.arff', skiprows=14, header=None, names=columns, na_values='?')

# 1. Row count, column count, dtypes
eda_out = []
eda_out.append("# Kitchenham Dataset EDA\n")
eda_out.append(f"**Rows:** {df.shape[0]}\n**Columns:** {df.shape[1]}\n")
eda_out.append("### Data Types\n```text")
eda_out.append(str(df.dtypes))
eda_out.append("```\n")

# 2. Missing-value count per column
eda_out.append("### Missing Values\n```text")
missing = df.isnull().sum()
eda_out.append(str(missing))
eda_out.append("```\n")

# 3. Numeric columns: min, max, mean, median, histogram
numeric_cols = ['Actual.duration', 'Actual.effort', 'Adjusted.function.points', 'First.estimate']
eda_out.append("### Numeric Columns\n")
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

stats = df[numeric_cols].describe().T
stats['median'] = df[numeric_cols].median()
eda_out.append("```text\n")
eda_out.append(str(stats[['min', 'max', 'mean', '50%']]))
eda_out.append("\n```\n")

# Plot histograms
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for i, col in enumerate(numeric_cols):
    ax = axes[i//2, i%2]
    df[col].dropna().hist(ax=ax, bins=30)
    ax.set_title(col)
plt.tight_layout()
plt.savefig('data/eda/kitchenham_histograms.png')
eda_out.append("![Histograms](kitchenham_histograms.png)\n")

# Check heavy skew
eda_out.append("### Skewness Check\n")
for col in numeric_cols:
    mean = df[col].mean()
    max_val = df[col].max()
    if max_val > 10 * mean:
        eda_out.append(f"- **{col}**: Heavy skew detected (max {max_val:.2f} > 10 * mean {mean:.2f}). Needs log-transform.\n")
    else:
        eda_out.append(f"- **{col}**: Normal (max {max_val:.2f}, mean {mean:.2f}).\n")

# 4. Categorical columns
eda_out.append("\n### Categorical Columns\n")
cat_cols = ['Client.code', 'Project.type', 'First.estimate.method']
for col in cat_cols:
    eda_out.append(f"#### {col}\n```text\n")
    counts = df[col].value_counts(dropna=False)
    eda_out.append(str(counts))
    eda_out.append("\n```\n")
    rare = counts[counts < 5].index.tolist()
    if rare:
        eda_out.append(f"- **Rare categories (<5):** {rare} -> Needs collapse into 'Other'\n")
    if col == 'Client.code':
        # Check imbalance
        max_client_pct = counts.max() / len(df)
        eda_out.append(f"- **Imbalance:** Top client is {max_client_pct:.1%} of data.\n")

# 5. Target variable
eda_out.append("\n### Target Variable Analysis\n")
df['overrun_ratio'] = df['Actual.effort'] / df['First.estimate']
eda_out.append("Target: `overrun_ratio = Actual.effort / First.estimate`\n```text\n")
eda_out.append(str(df['overrun_ratio'].describe()))
eda_out.append("\n```\n")
failed_count = (df['overrun_ratio'] > 1.5).sum() # arbitrary threshold just for analysis
eda_out.append(f"Number of 'failed' projects (overrun > 1.5): {failed_count} ({failed_count/len(df):.1%})\n")
eda_out.append("Conclusion: Treat as REGRESSION on log(overrun_ratio) due to target imbalance.\n")

with open('data/eda/kitchenham_eda.md', 'w') as f:
    f.write("\n".join(eda_out))

print("EDA for Kitchenham completed. Check data/eda/kitchenham_eda.md")
