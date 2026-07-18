# DELTA — Real Project Data Requirements

> This document specifies exactly what project data you need to collect to train DELTA on real-world projects.

---

## Required Fields (Must-Have)

These 12 fields are the core inputs the model uses. **Every row must have all 12.**

| # | Field | Type | Range/Values | Description |
|---|-------|------|-------------|-------------|
| 1 | `industry_type` | String (enum) | BFSI, Healthcare, Retail, Telecom, Manufacturing, Government, Energy, EdTech | Client's industry vertical |
| 2 | `team_size` | Integer | 3–200 | Total team members assigned at project start |
| 3 | `seniority_mix_junior` | Float | 0.0–1.0 | Fraction of team that is junior (0–3 yrs experience) |
| 4 | `seniority_mix_mid` | Float | 0.0–1.0 | Fraction of team that is mid-level (3–7 yrs) |
| 5 | `seniority_mix_senior` | Float | 0.0–1.0 | Fraction of team that is senior (7+ yrs) |
| 6 | `budget_planned_usd` | Float | $10,000+ | Original approved project budget in USD |
| 7 | `duration_planned_weeks` | Integer | 2–104 | Original planned duration in weeks |
| 8 | `scope_change_count` | Integer | 0–50 | Number of formal scope changes / CRs during project |
| 9 | `client_type` | String (enum) | fixed_bid, outcome_based, time_and_material | Contract/pricing model |
| 10 | `employee_cost_ratio` | Float | 0.30–0.90 | Employee costs as fraction of project revenue/budget |
| 11 | `attrition_events` | Integer | 0–30 | Number of team members who left during the project |
| 12 | `weekly_burn_rate_variance` | Float | 0.0–0.5 | Standard deviation of weekly spend / mean weekly spend |

> **Constraint**: `seniority_mix_junior + seniority_mix_mid + seniority_mix_senior ≈ 1.0` (within ±0.05)

---

## Required Outcomes (Target Variables)

These are what the model learns to predict. You need at least one:

| Field | Type | Description |
|-------|------|-------------|
| `budget_actual_usd` | Float | **Actual** total project cost at completion (USD) |
| `outcome_label` | String | Project outcome: `on_track`, `at_risk`, or `failed` |

### How to determine `outcome_label`:
- **on_track**: Delivered within ±10% of budget and ≤2 weeks late
- **at_risk**: Overran budget by 10–30% OR was 3–8 weeks late
- **failed**: Overran budget by >30% OR was >8 weeks late OR was cancelled/descoped significantly

> If you only have `budget_actual_usd`, the model can compute the overrun ratio (`budget_actual_usd / budget_planned_usd`) and derive the label automatically.

---

## Nice-to-Have Fields (Improve Model)

These are optional but would significantly improve prediction quality:

| Field | Type | Description |
|-------|------|-------------|
| `duration_actual_weeks` | Integer | Actual project duration |
| `client_satisfaction_score` | Float (1–5) | Post-project client satisfaction |
| `technology_stack` | String | Primary technology (Java, .NET, Python, SAP, etc.) |
| `project_type` | String | new_development, enhancement, migration, support |
| `team_location` | String | onshore, offshore, hybrid |
| `pm_experience_years` | Integer | Project manager's years of PM experience |
| `rework_percentage` | Float (0–1) | Fraction of deliverables requiring rework |
| `defect_density` | Float | Defects per KLOC or per function point |
| `milestone_slip_count` | Integer | Number of missed milestones |
| `stakeholder_count` | Integer | Number of key stakeholders |

---

## Data Quality Requirements

### Minimum Dataset Sizes

| Stage | Rows | Purpose |
|-------|------|---------|
| **Initial validation** | 50–100 | Verify the model works on real data at all |
| **Usable model** | 200–500 | Enough for basic train/test with reasonable accuracy |
| **Production quality** | 1,000+ | Sufficient for cross-validation and generalization |
| **High confidence** | 5,000+ | Needed for reliable per-industry predictions |

### Quality Checklist

- [ ] All 12 required fields present for every row
- [ ] No obviously wrong values (e.g., budget $0, team size 0)
- [ ] Seniority mix sums to ~1.0
- [ ] `employee_cost_ratio` is between 0.30 and 0.90
- [ ] `budget_actual_usd` ≥ `budget_planned_usd * 0.5` (not absurdly low)
- [ ] Reasonable class distribution:
  - at least 15% of projects should be "at_risk" or "failed"
  - if ALL projects are "on_track", the data is biased or the labeling is too lenient
- [ ] No duplicate projects
- [ ] Currency is consistent (all USD, or specify a currency column)

---

## File Format

### Preferred: CSV
```csv
industry_type,team_size,seniority_mix_junior,seniority_mix_mid,seniority_mix_senior,budget_planned_usd,duration_planned_weeks,scope_change_count,client_type,employee_cost_ratio,attrition_events,weekly_burn_rate_variance,budget_actual_usd,outcome_label
BFSI,25,0.30,0.45,0.25,500000,24,4,fixed_bid,0.58,2,0.12,575000,at_risk
Healthcare,15,0.20,0.50,0.30,200000,16,1,time_and_material,0.52,0,0.06,195000,on_track
Retail,40,0.45,0.35,0.20,800000,32,8,fixed_bid,0.63,5,0.22,1120000,failed
```

### Also Acceptable
- Excel (.xlsx) with a single sheet
- JSON array of objects
- Parquet file

---

## Where to Get Real Data

### Internal Sources (Best)
1. **PMO databases** — Most IT companies track project P&L in tools like Clarity, Jira Portfolio, or custom dashboards
2. **Finance systems** — Actual vs. planned cost data from SAP, Oracle Financials
3. **HR systems** — Attrition events, team composition data from Workday, SAP SuccessFactors
4. **Timesheet systems** — Weekly burn rates from tools like Replicon, Harvest

### External/Public Sources
1. **Standish Group CHAOS Reports** — Aggregate project success/failure rates (no row-level data)
2. **ISBSG (International Software Benchmarking Standards Group)** — Paid dataset of ~9,000 IT projects with effort, duration, and cost data. This is the closest to what DELTA needs.
   - Website: https://www.isbsg.org
   - Cost: ~$300–$500 for academic license
3. **NASA Software Engineering Lab (SEL)** — Historical software project data (mostly aerospace)
4. **Promise Repository** — Software engineering datasets for research
5. **Kaggle** — Search for "software project" or "IT project" datasets
6. **Government contracts** — Public procurement data (USASpending.gov, GeM in India)

### Synthetic Enhancement
If you have a small real dataset (50–100 rows), DELTA can:
1. Use the real data to calibrate the synthetic generator's distributions
2. Generate 1,000+ synthetic rows that match real-world statistics
3. Train on the combined dataset (real + calibrated synthetic)
4. This is standard practice in small-data ML (data augmentation)

---

## Data Privacy Notes

> **Important**: If collecting data from a real company:
> - Remove all project names, client names, and employee names
> - Use anonymized project IDs (e.g., P001, P002)
> - Aggregate team data (just ratios, not individual records)
> - Do NOT include any PII, contract terms, or NDA-protected information
> - Get explicit written permission before using any company data
> - Consider differential privacy techniques for sensitive cost data

---

## How to Upload

Once you have the data:

1. Save it as `data/real_projects.csv` in the DELTA project
2. Run the ingestion script (will be created):
   ```bash
   python data/ingest_real_data.py --file data/real_projects.csv --validate
   ```
3. This will:
   - Validate all fields and flag issues
   - Show distribution statistics
   - Merge with synthetic data if needed
   - Retrain the model automatically

---

## Example Template

Save this as your starting template:

```csv
industry_type,team_size,seniority_mix_junior,seniority_mix_mid,seniority_mix_senior,budget_planned_usd,duration_planned_weeks,scope_change_count,client_type,employee_cost_ratio,attrition_events,weekly_burn_rate_variance,budget_actual_usd,outcome_label
```

Fill in one row per completed project. The more projects, the better the model.
