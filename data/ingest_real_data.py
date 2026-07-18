"""
DELTA — Real Data Ingestion & Validation Pipeline
====================================================
Validates, cleans, and prepares real project data for model training.

Usage:
    python data/ingest_real_data.py --file data/real_projects.csv
    python data/ingest_real_data.py --file data/real_projects.csv --validate-only
    python data/ingest_real_data.py --file data/real_projects.csv --merge-synthetic

What it does:
1. Loads the CSV and checks all required columns exist
2. Validates data types, ranges, and logical constraints
3. Flags quality issues per-row (missing values, outliers, impossible combos)
4. Shows distribution comparison vs synthetic data
5. Generates a quality report
6. Optionally merges with synthetic data for augmented training
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).parent.parent
SYNTHETIC_PATH = PROJECT_ROOT / "data" / "synthetic_projects.csv"
REPORT_DIR = PROJECT_ROOT / "data" / "validation_report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Required Schema ────────────────────────────────────────────────────────

REQUIRED_COLUMNS = {
    "industry_type": {"type": "str", "values": ["BFSI", "Healthcare", "Retail", "Telecom", "Manufacturing", "Government", "Energy", "EdTech"]},
    "team_size": {"type": "int", "min": 1, "max": 500},
    "budget_planned_usd": {"type": "float", "min": 1000, "max": 100_000_000},
    "duration_planned_weeks": {"type": "int", "min": 1, "max": 200},
    "scope_change_count": {"type": "int", "min": 0, "max": 100},
    "client_type": {"type": "str", "values": ["fixed_bid", "outcome_based", "time_and_material"]},
    "employee_cost_ratio": {"type": "float", "min": 0.10, "max": 0.95},
}

OPTIONAL_COLUMNS = {
    "seniority_mix_junior": {"type": "float", "min": 0, "max": 1, "default": 0.35},
    "seniority_mix_mid": {"type": "float", "min": 0, "max": 1, "default": 0.40},
    "seniority_mix_senior": {"type": "float", "min": 0, "max": 1, "default": 0.25},
    "attrition_events": {"type": "int", "min": 0, "max": 50, "default": 0},
    "weekly_burn_rate_variance": {"type": "float", "min": 0, "max": 1.0, "default": 0.10},
}

TARGET_COLUMNS = {
    "budget_actual_usd": {"type": "float", "min": 1000},
    "outcome_label": {"type": "str", "values": ["on_track", "at_risk", "failed"]},
}

# Common column name mappings (fuzzy matching)
COLUMN_ALIASES = {
    "industry": "industry_type",
    "sector": "industry_type",
    "vertical": "industry_type",
    "team": "team_size",
    "headcount": "team_size",
    "team_count": "team_size",
    "budget": "budget_planned_usd",
    "planned_budget": "budget_planned_usd",
    "budget_usd": "budget_planned_usd",
    "planned_cost": "budget_planned_usd",
    "duration": "duration_planned_weeks",
    "planned_duration": "duration_planned_weeks",
    "timeline_weeks": "duration_planned_weeks",
    "scope_changes": "scope_change_count",
    "change_requests": "scope_change_count",
    "cr_count": "scope_change_count",
    "contract_type": "client_type",
    "pricing_model": "client_type",
    "ecr": "employee_cost_ratio",
    "cost_ratio": "employee_cost_ratio",
    "actual_cost": "budget_actual_usd",
    "actual_budget": "budget_actual_usd",
    "final_cost": "budget_actual_usd",
    "total_cost": "budget_actual_usd",
    "outcome": "outcome_label",
    "status": "outcome_label",
    "result": "outcome_label",
    "attrition": "attrition_events",
    "turnover": "attrition_events",
    "burn_variance": "weekly_burn_rate_variance",
    "burn_rate_var": "weekly_burn_rate_variance",
}


def auto_map_columns(df):
    """Try to automatically map column names to our schema."""
    mapped = {}
    unmapped = []
    
    for col in df.columns:
        col_lower = col.lower().strip().replace(" ", "_")
        
        # Exact match
        all_expected = list(REQUIRED_COLUMNS.keys()) + list(OPTIONAL_COLUMNS.keys()) + list(TARGET_COLUMNS.keys())
        if col_lower in all_expected:
            mapped[col] = col_lower
            continue
        
        # Alias match
        if col_lower in COLUMN_ALIASES:
            mapped[col] = COLUMN_ALIASES[col_lower]
            continue
        
        unmapped.append(col)
    
    return mapped, unmapped


def validate_row(row, row_idx, issues):
    """Validate a single row and collect issues."""
    row_issues = []
    
    # Required fields
    for col, spec in REQUIRED_COLUMNS.items():
        if col not in row.index or pd.isna(row.get(col)):
            row_issues.append(f"Missing required: {col}")
            continue
        
        val = row[col]
        if spec["type"] == "int":
            try:
                val = int(float(val))
                if "min" in spec and val < spec["min"]:
                    row_issues.append(f"{col}={val} below min {spec['min']}")
                if "max" in spec and val > spec["max"]:
                    row_issues.append(f"{col}={val} above max {spec['max']}")
            except (ValueError, TypeError):
                row_issues.append(f"{col}='{val}' not a valid integer")
        elif spec["type"] == "float":
            try:
                val = float(val)
                if "min" in spec and val < spec["min"]:
                    row_issues.append(f"{col}={val} below min {spec['min']}")
                if "max" in spec and val > spec["max"]:
                    row_issues.append(f"{col}={val} above max {spec['max']}")
            except (ValueError, TypeError):
                row_issues.append(f"{col}='{val}' not a valid float")
        elif spec["type"] == "str" and "values" in spec:
            if str(val) not in spec["values"]:
                row_issues.append(f"{col}='{val}' not in {spec['values']}")
    
    # Seniority mix check
    if all(col in row.index for col in ["seniority_mix_junior", "seniority_mix_mid", "seniority_mix_senior"]):
        try:
            total = float(row["seniority_mix_junior"]) + float(row["seniority_mix_mid"]) + float(row["seniority_mix_senior"])
            if abs(total - 1.0) > 0.1:
                row_issues.append(f"Seniority mix sums to {total:.2f} (should be ~1.0)")
        except (ValueError, TypeError):
            pass
    
    # Budget sanity
    if "budget_actual_usd" in row.index and "budget_planned_usd" in row.index:
        try:
            actual = float(row["budget_actual_usd"])
            planned = float(row["budget_planned_usd"])
            if planned > 0:
                ratio = actual / planned
                if ratio < 0.3:
                    row_issues.append(f"Actual/planned ratio={ratio:.2f} (suspiciously low)")
                if ratio > 5.0:
                    row_issues.append(f"Actual/planned ratio={ratio:.2f} (suspiciously high)")
        except (ValueError, TypeError):
            pass
    
    if row_issues:
        issues.append({"row": row_idx, "issues": row_issues})
    
    return len(row_issues) == 0


def fill_missing_columns(df):
    """Fill missing optional columns with industry-average defaults."""
    filled = []
    for col, spec in OPTIONAL_COLUMNS.items():
        if col not in df.columns:
            df[col] = spec["default"]
            filled.append(f"{col} → {spec['default']}")
    
    # Derive outcome_label from budget if missing
    if "outcome_label" not in df.columns and "budget_actual_usd" in df.columns and "budget_planned_usd" in df.columns:
        df["overrun_ratio"] = df["budget_actual_usd"] / df["budget_planned_usd"]
        df["outcome_label"] = df["overrun_ratio"].apply(
            lambda r: "on_track" if r <= 1.10 else ("at_risk" if r <= 1.30 else "failed")
        )
        filled.append("outcome_label → derived from budget overrun ratio")
    
    return df, filled


def compare_distributions(real_df, synth_df):
    """Compare real vs synthetic data distributions."""
    print("\n── Distribution Comparison: Real vs Synthetic ──")
    
    numeric_cols = ["team_size", "budget_planned_usd", "duration_planned_weeks",
                    "scope_change_count", "employee_cost_ratio"]
    
    available_cols = [c for c in numeric_cols if c in real_df.columns and c in synth_df.columns]
    
    if not available_cols:
        print("  No comparable numeric columns found")
        return
    
    n_cols = len(available_cols)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    
    for ax, col in zip(axes, available_cols):
        ax.hist(synth_df[col].dropna(), bins=20, alpha=0.5, color="#2E5CFF", label="Synthetic", density=True)
        ax.hist(real_df[col].dropna(), bins=20, alpha=0.5, color="#22C55E", label="Real", density=True)
        ax.set_title(col.replace("_", " ").title(), fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle("DELTA — Real vs Synthetic Distribution Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(REPORT_DIR / "distribution_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Comparison plot saved to {REPORT_DIR}/distribution_comparison.png")
    
    # Statistical summary
    print(f"\n  {'Column':<30} {'Synth Mean':>12} {'Real Mean':>12} {'Synth Std':>12} {'Real Std':>12}")
    print("  " + "-" * 80)
    for col in available_cols:
        sm = synth_df[col].mean()
        rm = real_df[col].mean()
        ss = synth_df[col].std()
        rs = real_df[col].std()
        print(f"  {col:<30} {sm:>12.2f} {rm:>12.2f} {ss:>12.2f} {rs:>12.2f}")


def generate_report(df, issues, filled, column_mapping, unmapped):
    """Generate validation report."""
    total = len(df)
    clean = total - len(issues)
    
    report = {
        "total_rows": total,
        "clean_rows": clean,
        "rows_with_issues": len(issues),
        "quality_score": round(clean / total * 100, 1) if total > 0 else 0,
        "columns_mapped": column_mapping,
        "columns_unmapped": unmapped,
        "columns_filled": filled,
        "issues_sample": issues[:20],  # First 20 issues
    }
    
    # Class distribution
    if "outcome_label" in df.columns:
        dist = df["outcome_label"].value_counts().to_dict()
        report["class_distribution"] = {str(k): int(v) for k, v in dist.items()}
    
    with open(REPORT_DIR / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    return report


def main():
    parser = argparse.ArgumentParser(description="DELTA — Real Data Ingestion")
    parser.add_argument("--file", required=True, help="Path to CSV/Excel file")
    parser.add_argument("--validate-only", action="store_true", help="Only validate, don't save")
    parser.add_argument("--merge-synthetic", action="store_true", help="Merge with synthetic data")
    args = parser.parse_args()
    
    file_path = Path(args.file)
    if not file_path.exists():
        print(f"ERROR: File not found: {file_path}")
        sys.exit(1)
    
    # Load data
    print(f"Loading {file_path}...")
    if file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path)
    else:
        df = pd.read_csv(file_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")
    
    # Auto-map columns
    column_mapping, unmapped = auto_map_columns(df)
    if column_mapping:
        print(f"\n  Auto-mapped columns:")
        for orig, mapped in column_mapping.items():
            if orig != mapped:
                print(f"    '{orig}' → '{mapped}'")
        df = df.rename(columns=column_mapping)
    
    if unmapped:
        print(f"\n  Unmapped columns (ignored): {unmapped}")
    
    # Check required columns
    missing_required = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_required:
        print(f"\n  ⚠ MISSING REQUIRED COLUMNS: {missing_required}")
        print("  These must be present. Check column names or add them manually.")
        if len(missing_required) > 3:
            print("  Too many missing columns — aborting.")
            sys.exit(1)
    
    # Fill missing optional columns
    df, filled = fill_missing_columns(df)
    if filled:
        print(f"\n  Filled missing columns with defaults:")
        for f in filled:
            print(f"    {f}")
    
    # Validate each row
    print(f"\n── Row-by-Row Validation ──")
    issues = []
    for idx, row in df.iterrows():
        validate_row(row, idx, issues)
    
    clean_count = len(df) - len(issues)
    quality_pct = clean_count / len(df) * 100 if len(df) > 0 else 0
    
    print(f"  Total rows:      {len(df)}")
    print(f"  Clean rows:      {clean_count}")
    print(f"  Rows w/ issues:  {len(issues)}")
    print(f"  Quality score:   {quality_pct:.1f}%")
    
    if issues:
        print(f"\n  Sample issues (first 5):")
        for issue in issues[:5]:
            print(f"    Row {issue['row']}: {', '.join(issue['issues'])}")
    
    # Class distribution
    if "outcome_label" in df.columns:
        print(f"\n── Class Distribution ──")
        dist = df["outcome_label"].value_counts()
        for label, count in dist.items():
            pct = count / len(df) * 100
            print(f"  {label:<12} {count:>5} ({pct:.1f}%)")
        
        # Warning if heavily imbalanced
        if dist.min() / dist.max() < 0.15:
            print("  ⚠ WARNING: Heavily imbalanced classes — consider resampling")
    
    # Compare with synthetic
    if SYNTHETIC_PATH.exists():
        synth_df = pd.read_csv(SYNTHETIC_PATH)
        compare_distributions(df, synth_df)
    
    # Generate report
    report = generate_report(df, issues, filled, column_mapping, unmapped)
    print(f"\n  Validation report saved to {REPORT_DIR}/validation_report.json")
    
    if args.validate_only:
        print("\n── Validate-only mode. No files saved. ──")
        return
    
    # Save cleaned data
    clean_df = df.drop(index=[i["row"] for i in issues]) if issues else df
    output_path = PROJECT_ROOT / "data" / "real_projects_clean.csv"
    clean_df.to_csv(output_path, index=False)
    print(f"\n  Clean data ({len(clean_df)} rows) saved to {output_path}")
    
    # Merge with synthetic if requested
    if args.merge_synthetic and SYNTHETIC_PATH.exists():
        synth_df = pd.read_csv(SYNTHETIC_PATH)
        
        # Align columns
        common_cols = list(set(clean_df.columns) & set(synth_df.columns))
        merged = pd.concat([synth_df[common_cols], clean_df[common_cols]], ignore_index=True)
        merged_path = PROJECT_ROOT / "data" / "merged_dataset.csv"
        merged.to_csv(merged_path, index=False)
        print(f"  Merged dataset ({len(merged)} rows: {len(synth_df)} synthetic + {len(clean_df)} real) saved to {merged_path}")
    
    print("\n✓ Ingestion complete. Ready for retraining.")
    print(f"  Next step: python model/train_model_v2.py")


if __name__ == "__main__":
    main()
