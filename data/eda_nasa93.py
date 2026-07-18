import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure directories exist
os.makedirs("data/eda", exist_ok=True)
os.makedirs("model/diagnostics", exist_ok=True)

print("Loading NASA93 dataset...")
columns = [
    'recordnumber', 'projectname', 'cat2', 'forg', 'center', 'year', 'mode',
    'rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 'acap', 'aexp',
    'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced', 'equivphyskloc', 'act_effort'
]

df = pd.read_csv('data/nasa93.arff', skiprows=314, header=None, names=columns, na_values='?')

# 1. Row count, column count, dtypes
eda_out = []
eda_out.append("# NASA93 Dataset EDA\n")
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
numeric_cols = ['year', 'equivphyskloc', 'act_effort']
eda_out.append("### Numeric Columns\n")
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

stats = df[numeric_cols].describe().T
stats['median'] = df[numeric_cols].median()
eda_out.append("```text\n")
eda_out.append(str(stats[['min', 'max', 'mean', '50%']]))
eda_out.append("\n```\n")

# Plot histograms
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, col in enumerate(numeric_cols):
    df[col].dropna().hist(ax=axes[i], bins=30)
    axes[i].set_title(col)
plt.tight_layout()
plt.savefig('data/eda/nasa93_histograms.png')
eda_out.append("![Histograms](nasa93_histograms.png)\n")

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
cat_cols = ['projectname', 'cat2', 'forg', 'center', 'mode'] + [
    'rely', 'data', 'cplx', 'time', 'stor', 'virt', 'turn', 'acap', 'aexp',
    'pcap', 'vexp', 'lexp', 'modp', 'tool', 'sced'
]
for col in cat_cols:
    eda_out.append(f"#### {col}\n```text\n")
    counts = df[col].value_counts(dropna=False)
    eda_out.append(str(counts))
    eda_out.append("\n```\n")
    rare = counts[counts < 5].index.tolist()
    if rare:
        eda_out.append(f"- **Rare categories (<5):** {rare} -> Might need collapse or drop (per instructions, cat2/projectname/center to be dropped)\n")

# 5. Target variable
eda_out.append("\n### Target Variable Analysis\n")
eda_out.append("Target: `act_effort` (predict log(act_effort) from log(equivphyskloc) + mode + ordinal drivers)\n```text\n")
eda_out.append(str(df['act_effort'].describe()))
eda_out.append("\n```\n")
eda_out.append("Conclusion: Treat as REGRESSION on log(act_effort).\n")

with open('data/eda/nasa93_eda.md', 'w') as f:
    f.write("\n".join(eda_out))

print("EDA for NASA93 completed. Check data/eda/nasa93_eda.md")
