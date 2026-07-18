"use client";

import { useState, useCallback } from "react";
import "./globals.css";

// ─── Types ──────────────────────────────────────────────────────────────────

interface ShapFactor {
  feature: string;
  impact: string;
  magnitude: number;
  description: string;
}

interface Recommendation {
  action: string;
  description: string;
  expected_risk_reduction: number;
  confidence: number;
}

interface PredictionResult {
  risk_class: string;
  risk_confidence: number;
  predicted_overrun_ratio: number;
  predicted_final_cost_usd: number;
  predicted_final_cost_inr: number;
  budget_planned_usd: number;
  budget_planned_inr: number;
  overrun_percentage: number;
  top_factors: ShapFactor[];
  class_probabilities: Record<string, number>;
  recommendations: Recommendation[];
}

interface SampleProject {
  project_index: number;
  features: Record<string, unknown>;
  prediction: Record<string, unknown>;
}

interface FormData {
  industry_type: string;
  team_size: number;
  seniority_mix_junior: number;
  seniority_mix_mid: number;
  seniority_mix_senior: number;
  budget_planned_usd: number;
  duration_planned_weeks: number;
  scope_change_count: number;
  client_type: string;
  employee_cost_ratio: number;
  attrition_events: number;
  weekly_burn_rate_variance: number;
}

// ─── Constants ──────────────────────────────────────────────────────────────

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const DEFAULT_FORM: FormData = {
  industry_type: "BFSI",
  team_size: 25,
  seniority_mix_junior: 0.30,
  seniority_mix_mid: 0.45,
  seniority_mix_senior: 0.25,
  budget_planned_usd: 500000,
  duration_planned_weeks: 24,
  scope_change_count: 4,
  client_type: "fixed_bid",
  employee_cost_ratio: 0.58,
  attrition_events: 2,
  weekly_burn_rate_variance: 0.12,
};

const INDUSTRIES = ["BFSI", "Healthcare", "Retail", "Telecom", "Manufacturing", "Government", "Energy", "EdTech"];
const CLIENT_TYPES = [
  { value: "fixed_bid", label: "Fixed Bid" },
  { value: "outcome_based", label: "Outcome Based" },
  { value: "time_and_material", label: "Time & Material" },
];

const USD_TO_INR = 83.5;

// ─── Helpers ────────────────────────────────────────────────────────────────

function formatCurrency(amount: number, currency: "USD" | "INR"): string {
  if (currency === "INR") {
    // Indian format: ₹XX,XX,XXX
    const inr = amount * USD_TO_INR;
    if (inr >= 10000000) return `₹${(inr / 10000000).toFixed(2)} Cr`;
    if (inr >= 100000) return `₹${(inr / 100000).toFixed(2)} L`;
    return `₹${inr.toLocaleString("en-IN", { maximumFractionDigits: 0 })}`;
  }
  if (amount >= 1000000) return `$${(amount / 1000000).toFixed(2)}M`;
  if (amount >= 1000) return `$${(amount / 1000).toFixed(1)}K`;
  return `$${amount.toFixed(0)}`;
}

function riskColor(risk: string): string {
  switch (risk) {
    case "on_track": return "#22C55E";
    case "at_risk": return "#F59E0B";
    case "failed": return "#EF4444";
    default: return "#64748B";
  }
}

function riskLabel(risk: string): string {
  switch (risk) {
    case "on_track": return "On Track";
    case "at_risk": return "At Risk";
    case "failed": return "Failed";
    default: return risk;
  }
}

// ─── Main Component ─────────────────────────────────────────────────────────

export default function DeltaDashboard() {
  const [form, setForm] = useState<FormData>(DEFAULT_FORM);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [samples, setSamples] = useState<SampleProject[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currency, setCurrency] = useState<"USD" | "INR">("USD");
  const [showForm, setShowForm] = useState(false);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Normalize values if entered as raw percentages (> 1.0)
      const payload = {
        ...form,
        seniority_mix_junior: form.seniority_mix_junior > 1.0 ? form.seniority_mix_junior / 100.0 : form.seniority_mix_junior,
        seniority_mix_mid: form.seniority_mix_mid > 1.0 ? form.seniority_mix_mid / 100.0 : form.seniority_mix_mid,
        seniority_mix_senior: form.seniority_mix_senior > 1.0 ? form.seniority_mix_senior / 100.0 : form.seniority_mix_senior,
        weekly_burn_rate_variance: form.weekly_burn_rate_variance > 1.0 ? form.weekly_burn_rate_variance / 100.0 : form.weekly_burn_rate_variance,
        employee_cost_ratio: form.employee_cost_ratio > 1.0 ? form.employee_cost_ratio / 100.0 : form.employee_cost_ratio,
      };

      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, [form]);

  const handleLoadSamples = useCallback(async () => {
    setLoadingSamples(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/projects/sample`);
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setSamples(data.projects);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to load samples");
    } finally {
      setLoadingSamples(false);
    }
  }, []);

  const handleSelectSample = useCallback(async (sample: SampleProject) => {
    const f = sample.features;
    const newForm: FormData = {
      industry_type: (f.industry_type as string) || "BFSI",
      team_size: (f.team_size as number) || 25,
      seniority_mix_junior: (f.seniority_mix_junior as number) || 0.33,
      seniority_mix_mid: (f.seniority_mix_mid as number) || 0.34,
      seniority_mix_senior: (f.seniority_mix_senior as number) || 0.33,
      budget_planned_usd: (f.budget_planned_usd as number) || 500000,
      duration_planned_weeks: (f.duration_planned_weeks as number) || 24,
      scope_change_count: (f.scope_change_count as number) || 0,
      client_type: (f.client_type as string) || "fixed_bid",
      employee_cost_ratio: (f.employee_cost_ratio as number) || 0.57,
      attrition_events: (f.attrition_events as number) || 0,
      weekly_burn_rate_variance: (f.weekly_burn_rate_variance as number) || 0.1,
    };
    setForm(newForm);

    // Auto-predict with selected sample
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newForm),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  }, []);

  const updateForm = (key: keyof FormData, value: string | number) => {
    setForm(prev => ({ ...prev, [key]: value }));
  };

  return (
    <>
      {/* Header */}
      <header className="header">
        <div className="container header-content">
          <div className="logo">
            <div className="logo-icon">Δ</div>
            <div>
              <div className="logo-text">DELTA</div>
              <div className="logo-subtitle">Delivery Risk Intelligence</div>
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
            <div className="currency-toggle">
              <button
                className={`currency-btn ${currency === "USD" ? "active" : ""}`}
                onClick={() => setCurrency("USD")}
              >
                USD
              </button>
              <button
                className={`currency-btn ${currency === "INR" ? "active" : ""}`}
                onClick={() => setCurrency("INR")}
              >
                INR
              </button>
            </div>
            <div className="header-badge glass">
              <span className="pulse"></span>
              Model Active
            </div>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="main">
        <div className="container">
          <h1 className="page-title">Project Risk Prediction</h1>
          <p className="page-desc">
            Predict cost overruns and delivery risk for IT projects using ML-powered analysis
          </p>

          {/* Problem Statement & Solution Block */}
          <div className="glass" style={{
            padding: "20px",
            borderRadius: "var(--radius-md)",
            marginBottom: "20px",
            background: "rgba(255, 255, 255, 0.015)"
          }}>
            <p style={{
              fontSize: "13.5px",
              color: "var(--text-primary)",
              lineHeight: "1.6",
              marginBottom: "8px",
              fontWeight: 500
            }}>
              ⚠️ <strong>Problem:</strong> IT services firms lose significant margin every year to project cost overruns and delivery delays caught too late to prevent.
            </p>
            <p style={{
              fontSize: "13.5px",
              color: "var(--text-secondary)",
              lineHeight: "1.6",
              fontWeight: 400
            }}>
              ✅ <strong>Solution:</strong> Delta predicts risk 4–8 weeks ahead using ML trained on patterns calibrated against published industry research, providing explainable, plain-language reasoning and reinforcement learning recommendations for every prediction.
            </p>
          </div>

          {/* Real Metrics Strip */}
          <div style={{
            display: "flex",
            flexWrap: "wrap",
            gap: "10px",
            marginBottom: "24px",
            alignItems: "center"
          }}>
            <div className="glass" style={{
              padding: "6px 12px",
              borderRadius: "20px",
              fontSize: "11px",
              color: "var(--text-secondary)",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}>
              <span style={{ color: "#2E5CFF", fontWeight: "bold" }}>●</span>
              <span>Classifier Accuracy: <strong>75.5% (5-Fold CV)</strong></span>
            </div>
            <div className="glass" style={{
              padding: "6px 12px",
              borderRadius: "20px",
              fontSize: "11px",
              color: "var(--text-secondary)",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}>
              <span style={{ color: "#22C55E", fontWeight: "bold" }}>●</span>
              <span>NASA93 Validation: <strong>R² = 0.735</strong></span>
            </div>
            <div className="glass" style={{
              padding: "6px 12px",
              borderRadius: "20px",
              fontSize: "11px",
              color: "var(--text-secondary)",
              display: "flex",
              alignItems: "center",
              gap: "6px"
            }}>
              <span style={{ color: "#7B3FE4", fontWeight: "bold" }}>●</span>
              <span>Industry Baseline ECR: <strong>57% - 60%</strong></span>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="actions-row">
            <button
              className="btn btn-primary"
              onClick={handleLoadSamples}
              disabled={loadingSamples}
            >
              {loadingSamples ? <span className="loading-spinner"></span> : "📊"}
              Load Sample Projects
            </button>
            <button
              className="btn btn-secondary"
              onClick={() => setShowForm(!showForm)}
            >
              {showForm ? "✕ Hide Form" : "✎ Custom Prediction"}
            </button>
          </div>

          {/* Error */}
          {error && (
            <div style={{
              padding: "12px 20px",
              borderRadius: "var(--radius-md)",
              background: "rgba(239, 68, 68, 0.1)",
              border: "1px solid rgba(239, 68, 68, 0.2)",
              color: "#F87171",
              fontSize: 13,
              marginBottom: 24,
            }}>
              ⚠ {error}
            </div>
          )}

          {/* Custom Prediction Form */}
          {showForm && (
            <div className="form-panel glass">
              <div className="panel-header">
                <div className="panel-icon glass" style={{ background: "rgba(46, 92, 255, 0.15)" }}>
                  ⚙
                </div>
                <div className="panel-title">Project Parameters</div>
              </div>
              <div className="form-grid">
                <div className="form-group">
                  <label className="form-label">Industry</label>
                  <select
                    className="form-select"
                    value={form.industry_type}
                    onChange={e => updateForm("industry_type", e.target.value)}
                  >
                    {INDUSTRIES.map(i => <option key={i} value={i}>{i}</option>)}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Team Size</label>
                  <input
                    className="form-input"
                    type="number"
                    min={1}
                    max={200}
                    value={form.team_size}
                    onChange={e => updateForm("team_size", parseInt(e.target.value) || 1)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Budget (USD)</label>
                  <input
                    className="form-input"
                    type="number"
                    min={10000}
                    value={form.budget_planned_usd}
                    onChange={e => updateForm("budget_planned_usd", parseInt(e.target.value) || 10000)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Duration (weeks)</label>
                  <input
                    className="form-input"
                    type="number"
                    min={1}
                    max={104}
                    value={form.duration_planned_weeks}
                    onChange={e => updateForm("duration_planned_weeks", parseInt(e.target.value) || 1)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Scope Changes</label>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={50}
                    value={form.scope_change_count}
                    onChange={e => updateForm("scope_change_count", parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Contract Type</label>
                  <select
                    className="form-select"
                    value={form.client_type}
                    onChange={e => updateForm("client_type", e.target.value)}
                  >
                    {CLIENT_TYPES.map(ct => (
                      <option key={ct.value} value={ct.value}>{ct.label}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Employee Cost Ratio</label>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={form.employee_cost_ratio}
                    onChange={e => updateForm("employee_cost_ratio", parseFloat(e.target.value) || 0.57)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Attrition Events</label>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    value={form.attrition_events}
                    onChange={e => updateForm("attrition_events", parseInt(e.target.value) || 0)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Burn Rate Variance</label>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={1}
                    step={0.01}
                    value={form.weekly_burn_rate_variance}
                    onChange={e => updateForm("weekly_burn_rate_variance", parseFloat(e.target.value) || 0.1)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Junior Mix (%)</label>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={form.seniority_mix_junior}
                    onChange={e => updateForm("seniority_mix_junior", parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Mid Mix (%)</label>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={form.seniority_mix_mid}
                    onChange={e => updateForm("seniority_mix_mid", parseFloat(e.target.value) || 0)}
                  />
                </div>
                <div className="form-group">
                  <label className="form-label">Senior Mix (%)</label>
                  <input
                    className="form-input"
                    type="number"
                    min={0}
                    max={1}
                    step={0.05}
                    value={form.seniority_mix_senior}
                    onChange={e => updateForm("seniority_mix_senior", parseFloat(e.target.value) || 0)}
                  />
                </div>
              </div>
              <button
                className="btn btn-primary"
                onClick={handlePredict}
                disabled={loading}
                style={{ width: "100%" }}
              >
                {loading ? <span className="loading-spinner"></span> : "⚡"}
                Run Prediction
              </button>
            </div>
          )}

          {/* Sample Projects Table */}
          {samples.length > 0 && (
            <div className="panel glass" style={{ marginBottom: 32, overflowX: "auto" }}>
              <div className="panel-header">
                <div className="panel-icon glass" style={{ background: "rgba(123, 63, 228, 0.15)" }}>
                  📋
                </div>
                <div className="panel-title">Sample Projects — Click to Predict</div>
              </div>
              <table className="sample-table">
                <thead>
                  <tr>
                    <th>Industry</th>
                    <th>Team</th>
                    <th>Budget</th>
                    <th>Duration</th>
                    <th>Scope Δ</th>
                    <th>Contract</th>
                    <th>ECR</th>
                    <th>Attrition</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {samples.map((s, i) => (
                    <tr key={i} onClick={() => handleSelectSample(s)}>
                      <td>{String(s.features.industry_type || "—")}</td>
                      <td>{String(s.features.team_size || "—")}</td>
                      <td>{formatCurrency(Number(s.features.budget_planned_usd || 0), currency)}</td>
                      <td>{String(s.features.duration_planned_weeks || "—")}w</td>
                      <td>{String(s.features.scope_change_count || "0")}</td>
                      <td style={{ fontSize: 11 }}>{String(s.features.client_type || "—").replace(/_/g, " ")}</td>
                      <td>{Number(s.features.employee_cost_ratio || 0).toFixed(2)}</td>
                      <td>{String(s.features.attrition_events || "0")}</td>
                      <td>
                        <span
                          className={`risk-badge risk-${s.prediction.risk_class}`}
                          style={{ fontSize: 10, padding: "3px 10px" }}
                        >
                          {riskLabel(String(s.prediction.risk_class))}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Prediction Results */}
          {result && (
            <>
              <div className="divider" />
              <div className="results-grid">
                {/* Risk Level Panel */}
                <div className="panel glass">
                  <div className="panel-header">
                    <div className="panel-icon glass" style={{
                      background: `${riskColor(result.risk_class)}20`,
                    }}>
                      {result.risk_class === "on_track" ? "✓" : result.risk_class === "at_risk" ? "⚠" : "✕"}
                    </div>
                    <div className="panel-title">Risk Assessment</div>
                  </div>
                  <span className={`risk-badge risk-${result.risk_class}`}>
                    {riskLabel(result.risk_class)}
                  </span>
                  <div className="risk-value" style={{ color: riskColor(result.risk_class) }}>
                    {(result.risk_confidence * 100).toFixed(1)}%
                  </div>
                  <div className="risk-confidence">Prediction Confidence</div>

                  {/* Class Probability Bars */}
                  <div style={{ marginTop: 20 }}>
                    {Object.entries(result.class_probabilities)
                      .sort(([, a], [, b]) => b - a)
                      .map(([cls, prob]) => (
                        <div className="prob-bar" key={cls}>
                          <div className="prob-label-row">
                            <span className="prob-label">{riskLabel(cls)}</span>
                            <span className="prob-value" style={{ color: riskColor(cls) }}>
                              {(prob * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="prob-track">
                            <div
                              className="prob-fill"
                              style={{
                                width: `${prob * 100}%`,
                                background: riskColor(cls),
                              }}
                            />
                          </div>
                        </div>
                      ))}
                  </div>
                </div>

                {/* Cost Panel */}
                <div className="panel glass">
                  <div className="panel-header">
                    <div className="panel-icon glass" style={{ background: "rgba(46, 92, 255, 0.15)" }}>
                      $
                    </div>
                    <div className="panel-title">Cost Analysis</div>
                  </div>
                  <div className="cost-row">
                    <span className="cost-label">Planned Budget</span>
                    <span className="cost-value">
                      {formatCurrency(result.budget_planned_usd, currency)}
                    </span>
                  </div>
                  <div className="cost-row">
                    <span className="cost-label">Predicted Final Cost</span>
                    <span className="cost-value" style={{ color: riskColor(result.risk_class) }}>
                      {currency === "USD"
                        ? formatCurrency(result.predicted_final_cost_usd, "USD")
                        : formatCurrency(result.predicted_final_cost_usd, "INR")}
                    </span>
                  </div>
                  <div className="cost-row">
                    <span className="cost-label">Cost Overrun</span>
                    <span className={`cost-overrun ${result.overrun_percentage > 0 ? "positive" : "negative"}`}>
                      {result.overrun_percentage > 0 ? "+" : ""}
                      {result.overrun_percentage.toFixed(1)}%
                    </span>
                  </div>

                  {/* Overrun Visual Bar */}
                  <div className="overrun-bar-container">
                    <div className="overrun-bar-labels">
                      <span>Budget</span>
                      <span>{result.overrun_percentage > 0 ? "Over Budget" : "Under Budget"}</span>
                    </div>
                    <div className="overrun-bar-track">
                      <div
                        className={`overrun-bar-fill ${
                          result.overrun_percentage > 20 ? "danger"
                            : result.overrun_percentage > 5 ? "warning"
                              : "safe"
                        }`}
                        style={{
                          width: `${Math.min(Math.max(result.predicted_overrun_ratio * 50, 5), 100)}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>

                {/* SHAP Factors Panel — Full Width */}
                <div className="panel glass full-panel">
                  <div className="panel-header">
                    <div className="panel-icon glass" style={{ background: "rgba(123, 63, 228, 0.15)" }}>
                      🔍
                    </div>
                    <div className="panel-title">Key Contributing Factors (SHAP Analysis)</div>
                  </div>
                  {result.top_factors.map((factor, i) => (
                    <div className="factor-card" key={i}>
                      <div className="factor-header">
                        <span className="factor-name">
                          {factor.feature.replace(/_/g, " ").replace(/^(industry type|client type)\s*/i, "")}
                        </span>
                        <span className={`factor-impact ${factor.impact}`}>
                          {factor.impact === "increases_risk" ? "↑ Risk" : "↓ Risk"}
                        </span>
                      </div>
                      <div className="factor-description">{factor.description}</div>
                    </div>
                  ))}
                </div>

                {/* RL Recommendations Panel */}
                {result.recommendations && result.recommendations.length > 0 && (
                  <div className="panel glass full-panel">
                    <div className="panel-header">
                      <div className="panel-icon glass" style={{ background: "rgba(34, 197, 94, 0.15)" }}>
                        💡
                      </div>
                      <div>
                        <div className="panel-title">Recommended Interventions (RL Agent)</div>
                        <div style={{ fontSize: "10.5px", color: "var(--text-muted)", marginTop: "2px", fontWeight: "normal" }}>
                          Estimated via simulated counterfactual analysis, not observed real-world outcomes.
                        </div>
                      </div>
                    </div>
                    {result.recommendations.map((rec, i) => (
                      <div className="factor-card" key={i} style={{ borderLeft: "3px solid #22C55E" }}>
                        <div className="factor-header">
                          <span className="factor-name" style={{ color: "#4ADE80" }}>
                            {rec.action}
                          </span>
                          <span className="factor-impact reduces_risk">
                            {rec.expected_risk_reduction > 0
                              ? `↓ ${(rec.expected_risk_reduction * 100).toFixed(1)}% risk`
                              : "Maintain"}
                          </span>
                        </div>
                        <div className="factor-description">{rec.description}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </>
          )}

          {/* Empty State */}
          {!result && samples.length === 0 && (
            <div style={{
              textAlign: "center",
              padding: "80px 20px",
              color: "var(--text-muted)",
            }}>
              <div style={{ fontSize: 48, marginBottom: 16, opacity: 0.3 }}>Δ</div>
              <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 8, color: "var(--text-secondary)" }}>
                No predictions yet
              </div>
              <div style={{ fontSize: 13 }}>
                Load sample projects or enter custom project parameters to get started
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer style={{
        padding: "20px 0",
        borderTop: "1px solid var(--glass-border)",
        position: "relative",
        zIndex: 2,
      }}>
        <div className="container" style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          flexWrap: "wrap",
          gap: "12px",
          fontSize: 11,
          color: "var(--text-muted)",
        }}>
          <span>DELTA — Project Cost-Overrun & Delivery-Risk Prediction</span>
          <div style={{ display: "flex", gap: 16 }}>
            <a href="https://github.com/Dhusyanth209/delta" target="_blank" rel="noopener noreferrer" style={{ color: "var(--text-muted)", textDecoration: "none", transition: "color 0.2s" }} onMouseOver={e => e.currentTarget.style.color = "var(--text-primary)"} onMouseOut={e => e.currentTarget.style.color = "var(--text-muted)"}>GitHub Repo</a>
            <a href="https://github.com/Dhusyanth209/delta/blob/main/docs/README.md" target="_blank" rel="noopener noreferrer" style={{ color: "var(--text-muted)", textDecoration: "none", transition: "color 0.2s" }} onMouseOver={e => e.currentTarget.style.color = "var(--text-primary)"} onMouseOut={e => e.currentTarget.style.color = "var(--text-muted)"}>Documentation</a>
            <a href="https://github.com/Dhusyanth209/delta/blob/main/docs/VIDEO_SCRIPT.md" target="_blank" rel="noopener noreferrer" style={{ color: "var(--text-muted)", textDecoration: "none", transition: "color 0.2s" }} onMouseOver={e => e.currentTarget.style.color = "var(--text-primary)"} onMouseOut={e => e.currentTarget.style.color = "var(--text-muted)"}>Demo Video</a>
          </div>
          <span>Hackathon Submission · Open Innovation Track</span>
        </div>
      </footer>
    </>
  );
}
