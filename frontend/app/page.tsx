"use client";

import { useState, useCallback, useRef, useEffect } from "react";
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

  // Copilot state
  const [copilotOpen, setCopilotOpen] = useState(false);
  const [copilotMessages, setCopilotMessages] = useState<{role: string; content: string}[]>([]);
  const [copilotInput, setCopilotInput] = useState("");
  const [copilotLoading, setCopilotLoading] = useState(false);
  const copilotMessagesEnd = useRef<HTMLDivElement>(null);

  // What-If Simulation state
  const [simTeamDelta, setSimTeamDelta] = useState(0);
  const [simScopeDelta, setSimScopeDelta] = useState(0);
  const [simClientType, setSimClientType] = useState<string | null>(null);
  const [simResult, setSimResult] = useState<any>(null);
  const [simLoading, setSimLoading] = useState(false);

  // Executive Report state
  const [reportOpen, setReportOpen] = useState(false);
  const [reportContent, setReportContent] = useState<string | null>(null);
  const [reportLoading, setReportLoading] = useState(false);
  const [reportCopied, setReportCopied] = useState(false);

  // Slack Alert state
  const [slackLoading, setSlackLoading] = useState(false);
  const [slackPreview, setSlackPreview] = useState<any>(null);
  const [slackStatus, setSlackStatus] = useState<string | null>(null);

  // Bulk Upload state
  const [uploadOpen, setUploadOpen] = useState(false);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [batchResults, setBatchResults] = useState<any>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Heatmap state
  const [heatmapData, setHeatmapData] = useState<any>(null);
  const [heatmapLoading, setHeatmapLoading] = useState(false);
  const [heatmapTopN, setHeatmapTopN] = useState(8);
  const [hoveredCell, setHoveredCell] = useState<{row: number; col: number} | null>(null);

  // Theme state
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  // Compare state
  const [compareList, setCompareList] = useState<number[]>([]);
  const [compareOpen, setCompareOpen] = useState(false);

  // Landing state
  const [showLanding, setShowLanding] = useState(true);
  const [landingFading, setLandingFading] = useState(false);

  // Email Alert state
  const [emailModalOpen, setEmailModalOpen] = useState(false);
  const [emailRecipient, setEmailRecipient] = useState("pmo-alerts@enterprise.com");
  const [emailLoading, setEmailLoading] = useState(false);
  const [emailResult, setEmailResult] = useState<any>(null);

  // Toast Notification state
  const [toasts, setToasts] = useState<Array<{ id: number; message: string; type: "success" | "error" | "info" }>>([]);

  // Risk Trajectory state
  const [trajectoryData, setTrajectoryData] = useState<any>(null);
  const [trajectoryLoading, setTrajectoryLoading] = useState(false);

  // Bookmarks & History state
  const [bookmarks, setBookmarks] = useState<Array<{
    id: string;
    label: string;
    timestamp: number;
    features: any;
    result: any;
    risk_class: string;
    overrun_pct: number;
  }>>([]);
  const [historyDrawerOpen, setHistoryDrawerOpen] = useState(false);

  const addToast = useCallback((message: string, type: "success" | "error" | "info" = "info") => {
    const id = Date.now() + Math.random();
    setToasts(prev => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts(prev => prev.filter(t => t.id !== id));
    }, 4000);
  }, []);

  // Initialize theme from localStorage
  useEffect(() => {
    const saved = localStorage.getItem("delta-theme") as "dark" | "light" | null;
    if (saved) {
      setTheme(saved);
      document.documentElement.setAttribute("data-theme", saved);
    }
    // Load bookmarks from localStorage
    try {
      const savedBookmarks = localStorage.getItem("delta-bookmarks");
      if (savedBookmarks) setBookmarks(JSON.parse(savedBookmarks));
    } catch {}
  }, []);

  const toggleTheme = useCallback(() => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("delta-theme", next);
  }, [theme]);

  // Bookmark helpers
  const saveBookmark = useCallback(() => {
    if (!form || !result) return;
    const industry = form.industry_type || "Project";
    const id = `bm-${Date.now()}`;
    const label = `${industry} — ${form.team_size} ppl, $${(form.budget_planned_usd / 1000).toFixed(0)}K`;
    const bm = {
      id,
      label,
      timestamp: Date.now(),
      features: { ...form },
      result: { ...result },
      risk_class: result.risk_class,
      overrun_pct: result.overrun_percentage,
    };
    setBookmarks(prev => {
      const next = [bm, ...prev].slice(0, 50); // cap at 50
      localStorage.setItem("delta-bookmarks", JSON.stringify(next));
      return next;
    });
    addToast(`Bookmarked: ${label}`, "success");
  }, [form, result, addToast]);

  const deleteBookmark = useCallback((id: string) => {
    setBookmarks(prev => {
      const next = prev.filter(b => b.id !== id);
      localStorage.setItem("delta-bookmarks", JSON.stringify(next));
      return next;
    });
    addToast("Bookmark removed", "info");
  }, [addToast]);

  const restoreBookmark = useCallback((bm: any) => {
    setForm(bm.features);
    setResult(bm.result);
    setTrajectoryData(null);
    setHistoryDrawerOpen(false);
    addToast(`Restored: ${bm.label}`, "success");
  }, [addToast]);

  const clearAllBookmarks = useCallback(() => {
    setBookmarks([]);
    localStorage.removeItem("delta-bookmarks");
    addToast("All bookmarks cleared", "info");
  }, [addToast]);

  useEffect(() => {
    copilotMessagesEnd.current?.scrollIntoView({ behavior: "smooth" });
  }, [copilotMessages]);

  const QUICK_QUESTIONS = [
    "Why is this project at risk?",
    "What's driving the cost overrun?",
    "What should I do first?",
    "Tell me about the team composition",
    "📚 What does PMBOK say about this risk?",
    "📚 How should I handle attrition per PMBOK?",
  ];

  const handleCopilotSend = useCallback(async (question?: string) => {
    const q = question || copilotInput.trim();
    if (!q) return;
    if (!result) {
      setCopilotMessages(prev => [
        ...prev,
        { role: "user", content: q },
        { role: "assistant", content: "Please run a prediction first so I have project data to analyze." },
      ]);
      setCopilotInput("");
      return;
    }

    const userMsg = { role: "user", content: q };
    setCopilotMessages(prev => [...prev, userMsg]);
    setCopilotInput("");
    setCopilotLoading(true);

    try {
      const res = await fetch(`${API_BASE}/copilot/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: q,
          project_features: form,
          prediction_result: result,
          chat_history: copilotMessages.slice(-6),
        }),
      });
      if (!res.ok) throw new Error(`API error: ${res.status}`);
      const data = await res.json();
      setCopilotMessages(prev => [...prev, { role: "assistant", content: data.answer }]);
    } catch {
      setCopilotMessages(prev => [
        ...prev,
        { role: "assistant", content: "Sorry, I couldn't connect to the copilot service. Please try again." },
      ]);
    } finally {
      setCopilotLoading(false);
    }
  }, [copilotInput, result, form, copilotMessages]);

  const handleSimulate = useCallback(async (teamDelta: number, scopeDelta: number, clientTypeOverride?: string | null) => {
    if (!form) return;
    setSimLoading(true);
    try {
      const res = await fetch(`${API_BASE}/simulate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseline_features: form,
          team_size_delta: teamDelta,
          scope_change_delta: scopeDelta,
          client_type: clientTypeOverride || undefined,
        }),
      });
      if (!res.ok) throw new Error(`Simulation error: ${res.status}`);
      const data = await res.json();
      setSimResult(data);
    } catch (err: any) {
      console.error("Simulation failed:", err);
    } finally {
      setSimLoading(false);
    }
  }, [form]);

  const handleGenerateReport = useCallback(async () => {
    if (!form || !result) return;
    setReportLoading(true);
    setReportOpen(true);
    try {
      const res = await fetch(`${API_BASE}/report`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_features: form,
          prediction_result: result,
          simulation_result: simResult || undefined,
        }),
      });
      if (!res.ok) throw new Error(`Report error: ${res.status}`);
      const data = await res.json();
      setReportContent(data.markdown_content);
    } catch (err: any) {
      console.error("Report generation failed:", err);
      setReportContent("Failed to generate report. Please try again.");
    } finally {
      setReportLoading(false);
    }
  }, [form, result, simResult]);

  const handleSlackAlert = useCallback(async () => {
    if (!form || !result) return;
    setSlackLoading(true);
    setSlackStatus(null);
    try {
      const res = await fetch(`${API_BASE}/alerts/slack`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_features: form,
          prediction_result: result,
        }),
      });
      if (!res.ok) throw new Error(`Slack error: ${res.status}`);
      const data = await res.json();
      setSlackStatus(data.status);
      if (data.status === "dry_run") {
        setSlackPreview(data);
        addToast("Slack alert generated (dry-run preview ready)", "info");
      } else if (data.status === "sent") {
        setSlackPreview(data);
        addToast("Slack alert posted successfully!", "success");
      }
    } catch (err: any) {
      console.error("Slack alert failed:", err);
      setSlackStatus("error");
      addToast(`Slack alert error: ${err.message}`, "error");
    } finally {
      setSlackLoading(false);
    }
  }, [form, result, addToast]);

  const handleSendEmailAlert = useCallback(async () => {
    if (!form || !result || !emailRecipient) return;
    setEmailLoading(true);
    try {
      const res = await fetch(`${API_BASE}/alerts/email`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          recipient_email: emailRecipient,
          project_features: form,
          prediction_result: result,
        }),
      });
      if (!res.ok) throw new Error(`Email error: ${res.status}`);
      const data = await res.json();
      setEmailResult(data);
      if (data.status === "sent") {
        addToast(`Email alert delivered to ${emailRecipient}`, "success");
      } else {
        addToast(`Email preview generated for ${emailRecipient}`, "info");
      }
    } catch (err: any) {
      console.error("Email alert failed:", err);
      addToast(`Failed to generate email alert: ${err.message}`, "error");
    } finally {
      setEmailLoading(false);
    }
  }, [form, result, emailRecipient, addToast]);

  const handleUpload = useCallback(async (file: File) => {
    setUploadLoading(true);
    setBatchResults(null);
    try {
      const formData = new window.FormData();
      formData.append("file", file);
      const res = await fetch(`${API_BASE}/projects/upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || `Upload error: ${res.status}`);
      }
      const data = await res.json();
      setBatchResults(data);
    } catch (err: any) {
      setError(err.message || "Upload failed");
    } finally {
      setUploadLoading(false);
    }
  }, []);

  const handleSelectBatchProject = useCallback((proj: any) => {
    const f = proj.project_features;
    setForm({
      industry_type: f.industry_type || "BFSI",
      team_size: f.team_size || 20,
      seniority_mix_junior: f.seniority_mix_junior || 0.3,
      seniority_mix_mid: f.seniority_mix_mid || 0.4,
      seniority_mix_senior: f.seniority_mix_senior || 0.3,
      budget_planned_usd: f.budget_planned_usd || 300000,
      duration_planned_weeks: f.duration_planned_weeks || 20,
      scope_change_count: f.scope_change_count || 3,
      client_type: f.client_type || "fixed_bid",
      employee_cost_ratio: f.employee_cost_ratio || 0.57,
      attrition_events: f.attrition_events || 1,
      weekly_burn_rate_variance: f.weekly_burn_rate_variance || 0.1,
    });
    setResult({
      risk_class: proj.risk_class,
      risk_confidence: proj.risk_confidence,
      predicted_overrun_ratio: 1 + (proj.overrun_percentage / 100),
      predicted_final_cost_usd: proj.predicted_final_cost_usd,
      predicted_final_cost_inr: proj.predicted_final_cost_inr,
      budget_planned_usd: proj.budget_planned_usd,
      budget_planned_inr: proj.budget_planned_usd * USD_TO_INR,
      overrun_percentage: proj.overrun_percentage,
      top_factors: proj.top_factors || [],
      class_probabilities: {},
      recommendations: proj.recommendations || [],
    });
    setSimResult(null);
    setBatchResults(null);
    setUploadOpen(false);
  }, []);

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
    <div className="app">
      {/* Toast Notification Container */}
      <div className="toast-container">
        {toasts.map(t => (
          <div key={t.id} className={`toast toast-${t.type}`}>
            <span>{t.type === "success" ? "✓" : t.type === "error" ? "✕" : "ℹ"}</span>
            <span>{t.message}</span>
          </div>
        ))}
      </div>

      {/* Email Alert Modal */}
      {emailModalOpen && (
        <div className="modal-backdrop" onClick={() => setEmailModalOpen(false)}>
          <div className="modal-card" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div className="modal-title">
                <span>📧</span> Executive Email Risk Alert
              </div>
              <button
                className="copilot-close"
                onClick={() => setEmailModalOpen(false)}
                style={{ fontSize: 18 }}
              >
                ✕
              </button>
            </div>
            <div className="modal-body">
              <label style={{ fontSize: 12, color: "var(--text-secondary)", display: "block", marginBottom: 6 }}>
                Recipient Email Address:
              </label>
              <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
                <input
                  type="email"
                  className="form-input"
                  style={{ flex: 1 }}
                  value={emailRecipient}
                  onChange={e => setEmailRecipient(e.target.value)}
                  placeholder="e.g. pmo-head@company.com"
                />
                <button
                  className="sim-btn active"
                  style={{ padding: "8px 16px", fontSize: 12 }}
                  onClick={handleSendEmailAlert}
                  disabled={emailLoading}
                >
                  {emailLoading ? "Sending..." : "Send Alert"}
                </button>
              </div>

              {emailResult && (
                <div style={{ marginTop: 12 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 6 }}>
                    Email Preview ({emailResult.status === "sent" ? "✅ Sent" : "ℹ️ Dry-Run Preview"})
                  </div>
                  <div style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 600, marginBottom: 8, padding: "6px 10px", background: "var(--glass-bg)", borderRadius: 6 }}>
                    Subject: {emailResult.subject}
                  </div>
                  <div
                    style={{
                      maxHeight: 260,
                      overflowY: "auto",
                      border: "1px solid var(--glass-border)",
                      borderRadius: 8,
                      background: "#0A0E1A",
                      padding: 12,
                      fontSize: 12,
                    }}
                    dangerouslySetInnerHTML={{ __html: emailResult.html_preview }}
                  />
                </div>
              )}
            </div>
            <div className="modal-footer">
              <button className="sim-btn" onClick={() => setEmailModalOpen(false)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Landing Page Overlay */}
      {showLanding && (
        <div className={`landing-overlay ${landingFading ? 'fadeout' : ''}`}>
          {/* Floating Orbs */}
          <div className="landing-orb landing-orb-1" />
          <div className="landing-orb landing-orb-2" />
          <div className="landing-orb landing-orb-3" />

          <div className="landing-content">
            <div className="landing-logo">Δ</div>

            <h1 className="landing-headline">
              Predict Project Risk<br /><em>Before It's Too Late</em>
            </h1>

            <p className="landing-sub">
              Employee costs rose <strong>206%</strong> while revenue grew only <strong>185%</strong> over a decade.
              Late detection of delivery risk means problems compound before intervention.
              DELTA uses AI to give you early warning.
            </p>

            <button
              className="landing-cta"
              onClick={() => {
                setLandingFading(true);
                setTimeout(() => setShowLanding(false), 500);
              }}
            >
              Launch Dashboard <span style={{ fontSize: 20 }}>→</span>
            </button>

            <div className="landing-features">
              <div className="landing-feature-card">
                <div className="landing-feature-icon">🧠</div>
                <div className="landing-feature-title">AI Copilot</div>
                <div className="landing-feature-desc">Ask questions about any project's risk factors in plain English</div>
              </div>
              <div className="landing-feature-card">
                <div className="landing-feature-icon">📊</div>
                <div className="landing-feature-title">SHAP Explainability</div>
                <div className="landing-feature-desc">See exactly why a project is flagged as at-risk or failed</div>
              </div>
              <div className="landing-feature-card">
                <div className="landing-feature-icon">🔮</div>
                <div className="landing-feature-title">What-If Simulator</div>
                <div className="landing-feature-desc">Test hiring, scope, and budget scenarios before deciding</div>
              </div>
            </div>

            <div className="landing-stats">
              <div className="landing-stat">
                <div className="landing-stat-val">950</div>
                <div className="landing-stat-label">Projects Trained</div>
              </div>
              <div className="landing-stat">
                <div className="landing-stat-val">71.6%</div>
                <div className="landing-stat-label">Accuracy</div>
              </div>
              <div className="landing-stat">
                <div className="landing-stat-val">12</div>
                <div className="landing-stat-label">Risk Factors</div>
              </div>
              <div className="landing-stat">
                <div className="landing-stat-val">29</div>
                <div className="landing-stat-label">Engineered Features</div>
              </div>
            </div>

            <div className="landing-tech">
              <span className="landing-tech-pill">XGBoost</span>
              <span className="landing-tech-pill">SHAP</span>
              <span className="landing-tech-pill">FastAPI</span>
              <span className="landing-tech-pill">Next.js</span>
              <span className="landing-tech-pill">ReportLab</span>
              <span className="landing-tech-pill">Docker</span>
            </div>
          </div>
        </div>
      )}

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
            <label className="theme-toggle-label" onClick={toggleTheme}>
              {theme === "dark" ? "🌙" : "☀️"}
              <div className="theme-toggle" />
            </label>
            <button
              className="sim-btn"
              style={{ padding: "6px 12px", fontSize: 12, display: "flex", alignItems: "center", gap: 6, background: historyDrawerOpen ? "rgba(46, 92, 255, 0.2)" : "rgba(255,255,255,0.05)", border: "1px solid var(--glass-border)", position: "relative" }}
              onClick={() => setHistoryDrawerOpen(prev => !prev)}
            >
              📑 History
              {bookmarks.length > 0 && (
                <span style={{ background: "#2E5CFF", color: "#fff", borderRadius: "50%", width: 16, height: 16, fontSize: 9, display: "flex", alignItems: "center", justifyContent: "center", fontWeight: 700 }}>
                  {bookmarks.length}
                </span>
              )}
            </button>
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
            <button
              className="btn btn-secondary"
              onClick={() => setUploadOpen(!uploadOpen)}
            >
              {uploadOpen ? "✕ Hide Upload" : "📤 Upload Projects"}
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

          {/* Bulk Upload Zone */}
          {uploadOpen && (
            <div className="form-panel glass" style={{ marginBottom: 24 }}>
              <div className="panel-header">
                <div className="panel-icon glass" style={{ background: "rgba(46, 92, 255, 0.15)" }}>
                  📤
                </div>
                <div className="panel-title">Bulk Project Upload</div>
                <a
                  href={`${API_BASE}/projects/template`}
                  download
                  style={{ fontSize: 12, color: "#60A5FA", textDecoration: "none", marginLeft: "auto", display: "flex", alignItems: "center", gap: 4 }}
                >
                  📥 Download Template CSV
                </a>
              </div>

              <div
                style={{
                  border: "2px dashed rgba(46, 92, 255, 0.3)",
                  borderRadius: "var(--radius-md)",
                  padding: "32px 24px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: "rgba(46, 92, 255, 0.04)",
                  transition: "border-color 0.2s, background 0.2s",
                }}
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => { e.preventDefault(); e.currentTarget.style.borderColor = "#2E5CFF"; e.currentTarget.style.background = "rgba(46, 92, 255, 0.1)"; }}
                onDragLeave={(e) => { e.currentTarget.style.borderColor = "rgba(46, 92, 255, 0.3)"; e.currentTarget.style.background = "rgba(46, 92, 255, 0.04)"; }}
                onDrop={(e) => {
                  e.preventDefault();
                  e.currentTarget.style.borderColor = "rgba(46, 92, 255, 0.3)";
                  e.currentTarget.style.background = "rgba(46, 92, 255, 0.04)";
                  const f = e.dataTransfer.files[0];
                  if (f) handleUpload(f);
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv,.xlsx"
                  style={{ display: "none" }}
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) handleUpload(f);
                    e.target.value = "";
                  }}
                />
                {uploadLoading ? (
                  <div style={{ color: "var(--text-muted)", fontSize: 14 }}>
                    <span className="loading-spinner" style={{ marginRight: 8 }}></span>
                    Processing uploaded projects...
                  </div>
                ) : (
                  <>
                    <div style={{ fontSize: 28, marginBottom: 8 }}>📂</div>
                    <div style={{ color: "var(--text-primary)", fontSize: 14, fontWeight: 600, marginBottom: 4 }}>
                      Drag & Drop CSV or Excel file here
                    </div>
                    <div style={{ color: "var(--text-muted)", fontSize: 12 }}>
                      or click to browse — Supports .csv and .xlsx
                    </div>
                  </>
                )}
              </div>
            </div>
          )}

          {/* Batch Results — Portfolio Summary + Table */}
          {batchResults && (
            <div style={{ marginBottom: 24 }}>
              {/* Portfolio Summary Cards */}
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 12, marginBottom: 16 }}>
                <div className="sim-metric-box">
                  <div className="sim-metric-label">Total Projects</div>
                  <div className="sim-metric-val" style={{ fontSize: 22 }}>{batchResults.total_projects}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{batchResults.successful_predictions} predicted successfully</div>
                </div>
                <div className="sim-metric-box">
                  <div className="sim-metric-label" style={{ color: "#22C55E" }}>🟢 On Track</div>
                  <div className="sim-metric-val" style={{ fontSize: 22, color: "#22C55E" }}>{batchResults.portfolio_summary?.risk_distribution?.on_track || 0}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>projects on schedule</div>
                </div>
                <div className="sim-metric-box">
                  <div className="sim-metric-label" style={{ color: "#F59E0B" }}>🟡 At Risk</div>
                  <div className="sim-metric-val" style={{ fontSize: 22, color: "#F59E0B" }}>{batchResults.portfolio_summary?.risk_distribution?.at_risk || 0}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>projects need attention</div>
                </div>
                <div className="sim-metric-box">
                  <div className="sim-metric-label" style={{ color: "#EF4444" }}>🔴 Failed</div>
                  <div className="sim-metric-val" style={{ fontSize: 22, color: "#EF4444" }}>{batchResults.portfolio_summary?.risk_distribution?.failed || 0}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>projects in danger</div>
                </div>
                <div className="sim-metric-box">
                  <div className="sim-metric-label">Avg Overrun</div>
                  <div className="sim-metric-val" style={{ fontSize: 22, color: (batchResults.portfolio_summary?.average_overrun_pct || 0) > 0 ? "#F87171" : "#34D399" }}>
                    {(batchResults.portfolio_summary?.average_overrun_pct || 0) > 0 ? "+" : ""}{(batchResults.portfolio_summary?.average_overrun_pct || 0).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>portfolio average</div>
                </div>
                <div className="sim-metric-box">
                  <div className="sim-metric-label">Total Cost Variance</div>
                  <div className="sim-metric-val" style={{ fontSize: 18, color: (batchResults.portfolio_summary?.total_cost_variance_usd || 0) > 0 ? "#F87171" : "#34D399" }}>
                    {currency === "USD"
                      ? `${(batchResults.portfolio_summary?.total_cost_variance_usd || 0) > 0 ? "+" : ""}$${Math.abs(batchResults.portfolio_summary?.total_cost_variance_usd || 0).toLocaleString()}`
                      : `${(batchResults.portfolio_summary?.total_cost_variance_inr || 0) > 0 ? "+" : ""}₹${Math.abs(batchResults.portfolio_summary?.total_cost_variance_inr || 0).toLocaleString()}`}
                  </div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>portfolio total</div>
                </div>
              </div>

              {/* Batch Results Table */}
              <div className="panel glass" style={{ overflow: "hidden" }}>
                <div className="panel-header" style={{ padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div className="panel-title" style={{ fontSize: 13 }}>📊 Batch Prediction Results — Click any row to drill down</div>
                  {compareList.length >= 2 && (
                    <button
                      className="sim-btn active"
                      style={{ padding: "5px 14px", fontSize: 11 }}
                      onClick={() => setCompareOpen(!compareOpen)}
                    >
                      {compareOpen ? "✕ Close Compare" : `📊 Compare ${compareList.length} Projects`}
                    </button>
                  )}
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table className="sample-table">
                    <thead>
                      <tr>
                        <th style={{ width: 36 }}>
                          <span style={{ fontSize: 9, color: "var(--text-muted)" }}>SEL</span>
                        </th>
                        <th>#</th>
                        <th>Industry</th>
                        <th>Team</th>
                        <th>Budget</th>
                        <th>Duration</th>
                        <th>Contract</th>
                        <th>Risk</th>
                        <th>Overrun</th>
                        <th>Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {batchResults.predictions?.filter((p: any) => p.status === "success").map((p: any, i: number) => (
                        <tr key={i} style={{ cursor: "pointer", background: compareList.includes(i) ? "rgba(46, 92, 255, 0.08)" : undefined }}>
                          <td onClick={(e) => { e.stopPropagation(); setCompareList(prev => prev.includes(i) ? prev.filter(x => x !== i) : prev.length >= 3 ? prev : [...prev, i]); }}>
                            <div style={{ width: 16, height: 16, borderRadius: 4, border: `2px solid ${compareList.includes(i) ? '#2E5CFF' : 'var(--glass-border)'}`, background: compareList.includes(i) ? '#2E5CFF' : 'transparent', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer', transition: 'all 0.15s' }}>
                              {compareList.includes(i) && <span style={{ color: '#fff', fontSize: 10, lineHeight: 1 }}>✓</span>}
                            </div>
                          </td>
                          <td style={{ fontSize: 11, color: "var(--text-muted)" }} onClick={() => handleSelectBatchProject(p)}>{p.row_index + 1}</td>
                          <td onClick={() => handleSelectBatchProject(p)}>{p.project_features?.industry_type || "—"}</td>
                          <td onClick={() => handleSelectBatchProject(p)}>{p.project_features?.team_size || "—"}</td>
                          <td onClick={() => handleSelectBatchProject(p)}>{formatCurrency(p.budget_planned_usd || 0, currency)}</td>
                          <td onClick={() => handleSelectBatchProject(p)}>{p.project_features?.duration_planned_weeks || "—"}w</td>
                          <td style={{ fontSize: 11 }} onClick={() => handleSelectBatchProject(p)}>{(p.project_features?.client_type || "—").replace(/_/g, " ")}</td>
                          <td onClick={() => handleSelectBatchProject(p)}>
                            <span className={`risk-badge risk-${p.risk_class}`} style={{ fontSize: 10, padding: "3px 10px" }}>
                              {riskLabel(p.risk_class)}
                            </span>
                          </td>
                          <td style={{ color: p.overrun_percentage > 0 ? "#F87171" : "#34D399", fontWeight: 600, fontSize: 12 }} onClick={() => handleSelectBatchProject(p)}>
                            {p.overrun_percentage > 0 ? "+" : ""}{p.overrun_percentage.toFixed(1)}%
                          </td>
                          <td style={{ fontSize: 12 }} onClick={() => handleSelectBatchProject(p)}>{(p.risk_confidence * 100).toFixed(0)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Comparison Panel */}
              {compareOpen && compareList.length >= 2 && (() => {
                const successProjs = batchResults.predictions?.filter((p: any) => p.status === "success") || [];
                const selected = compareList.map(i => successProjs[i]).filter(Boolean);
                if (selected.length < 2) return null;

                // Find best/worst for each metric
                const overruns = selected.map((p: any) => p.overrun_percentage);
                const confidences = selected.map((p: any) => p.risk_confidence);
                const costs = selected.map((p: any) => p.predicted_final_cost_usd);
                const bestOverrun = Math.min(...overruns);
                const worstOverrun = Math.max(...overruns);
                const bestConf = Math.max(...confidences);

                const riskRank: Record<string, number> = { on_track: 0, at_risk: 1, failed: 2 };

                const metrics = [
                  { label: "Risk Level", key: "risk", render: (p: any) => riskLabel(p.risk_class), color: (p: any) => riskColor(p.risk_class) },
                  { label: "Confidence", key: "conf", render: (p: any) => `${(p.risk_confidence * 100).toFixed(0)}%` },
                  { label: "Overrun %", key: "overrun", render: (p: any) => `${p.overrun_percentage > 0 ? "+" : ""}${p.overrun_percentage.toFixed(1)}%`, color: (p: any) => p.overrun_percentage > 0 ? "#F87171" : "#34D399" },
                  { label: "Budget", key: "budget", render: (p: any) => formatCurrency(p.budget_planned_usd || 0, currency) },
                  { label: "Predicted Cost", key: "cost", render: (p: any) => formatCurrency(p.predicted_final_cost_usd || 0, currency) },
                  { label: "Industry", key: "industry", render: (p: any) => p.project_features?.industry_type || "—" },
                  { label: "Team Size", key: "team", render: (p: any) => p.project_features?.team_size || "—" },
                  { label: "Contract", key: "contract", render: (p: any) => (p.project_features?.client_type || "—").replace(/_/g, " ") },
                ];

                return (
                  <div className="panel glass" style={{ overflow: "hidden", marginTop: 16 }}>
                    <div className="panel-header" style={{ padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                      <div className="panel-title" style={{ fontSize: 13 }}>📊 Project Comparison — {selected.length} projects</div>
                      <button className="sim-btn" style={{ padding: "4px 10px", fontSize: 10 }} onClick={() => { setCompareList([]); setCompareOpen(false); }}>Clear</button>
                    </div>
                    <div style={{ padding: "0 16px 16px", overflowX: "auto" }}>
                      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                        <thead>
                          <tr>
                            <th style={{ textAlign: "left", padding: "10px 12px", color: "var(--text-muted)", fontSize: 10, fontWeight: 600, borderBottom: "1px solid var(--glass-border)" }}>METRIC</th>
                            {selected.map((p: any, i: number) => (
                              <th key={i} style={{ textAlign: "center", padding: "10px 12px", borderBottom: "1px solid var(--glass-border)", minWidth: 140 }}>
                                <span className={`risk-badge risk-${p.risk_class}`} style={{ fontSize: 9, padding: "2px 8px", marginRight: 6 }}>{riskLabel(p.risk_class).charAt(0)}</span>
                                <span style={{ color: "var(--text-primary)", fontWeight: 600, fontSize: 11 }}>Project #{p.row_index + 1}</span>
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {metrics.map((m, mi) => (
                            <tr key={mi}>
                              <td style={{ padding: "8px 12px", color: "var(--text-secondary)", fontWeight: 500, fontSize: 11, borderBottom: "1px solid var(--glass-border)" }}>{m.label}</td>
                              {selected.map((p: any, pi: number) => {
                                const val = m.render(p);
                                let cellBg = "transparent";
                                let indicator = "";
                                // Diff highlights for overrun
                                if (m.key === "overrun") {
                                  if (p.overrun_percentage === bestOverrun && bestOverrun !== worstOverrun) { cellBg = "rgba(34, 197, 94, 0.08)"; indicator = " 🏆"; }
                                  if (p.overrun_percentage === worstOverrun && bestOverrun !== worstOverrun) { cellBg = "rgba(239, 68, 68, 0.08)"; indicator = " ⚠"; }
                                }
                                if (m.key === "risk") {
                                  const ranks = selected.map((s: any) => riskRank[s.risk_class] ?? 1);
                                  const bestRank = Math.min(...ranks);
                                  const worstRank = Math.max(...ranks);
                                  if ((riskRank[p.risk_class] ?? 1) === bestRank && bestRank !== worstRank) { cellBg = "rgba(34, 197, 94, 0.08)"; indicator = " 🏆"; }
                                  if ((riskRank[p.risk_class] ?? 1) === worstRank && bestRank !== worstRank) { cellBg = "rgba(239, 68, 68, 0.08)"; indicator = " ⚠"; }
                                }
                                return (
                                  <td key={pi} style={{ textAlign: "center", padding: "8px 12px", fontWeight: 600, color: m.color ? m.color(p) : "var(--text-primary)", borderBottom: "1px solid var(--glass-border)", background: cellBg, transition: "background 0.2s" }}>
                                    {val}{indicator}
                                  </td>
                                );
                              })}
                            </tr>
                          ))}
                          {/* Top SHAP factors row */}
                          <tr>
                            <td style={{ padding: "10px 12px", color: "var(--text-secondary)", fontWeight: 500, fontSize: 11, verticalAlign: "top" }}>Top Risk Factors</td>
                            {selected.map((p: any, pi: number) => (
                              <td key={pi} style={{ padding: "8px 12px", verticalAlign: "top" }}>
                                {(p.top_factors || []).slice(0, 3).map((f: any, fi: number) => (
                                  <div key={fi} style={{ marginBottom: 6 }}>
                                    <div style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 10, marginBottom: 2 }}>
                                      <span style={{ color: f.impact === "increases_risk" ? "#FCA5A5" : "#86EFAC" }}>{f.impact === "increases_risk" ? "↑" : "↓"}</span>
                                      <span style={{ color: "var(--text-primary)", fontWeight: 500 }}>{f.feature?.replace(/_/g, " ")}</span>
                                    </div>
                                    <div style={{ height: 4, borderRadius: 2, background: "var(--glass-border)", overflow: "hidden" }}>
                                      <div style={{ height: "100%", width: `${Math.min(f.magnitude * 200, 100)}%`, borderRadius: 2, background: f.impact === "increases_risk" ? "#EF4444" : "#22C55E", transition: "width 0.5s" }} />
                                    </div>
                                  </div>
                                ))}
                              </td>
                            ))}
                          </tr>
                          {/* Recommendations row */}
                          <tr>
                            <td style={{ padding: "10px 12px", color: "var(--text-secondary)", fontWeight: 500, fontSize: 11, verticalAlign: "top" }}>Top Action</td>
                            {selected.map((p: any, pi: number) => {
                              const rec = p.recommendations?.[0];
                              return (
                                <td key={pi} style={{ padding: "8px 12px", fontSize: 10, color: "var(--text-secondary)", verticalAlign: "top" }}>
                                  {rec ? (
                                    <div>
                                      <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: 2 }}>{rec.action}</div>
                                      <div>{rec.description?.slice(0, 50)}</div>
                                      <div style={{ color: "#34D399", marginTop: 2 }}>-{(rec.expected_risk_reduction * 100).toFixed(0)}% risk</div>
                                    </div>
                                  ) : <span style={{ color: "var(--text-muted)" }}>—</span>}
                                </td>
                              );
                            })}
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </div>
                );
              })()}

              {/* Heatmap Section */}
              <div className="panel glass" style={{ overflow: "hidden", marginTop: 16 }}>
                <div className="panel-header" style={{ padding: "12px 16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div className="panel-title" style={{ fontSize: 13 }}>🗺️ Risk Factor Heatmap</div>
                  <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                    <button
                      className="sim-btn"
                      style={{ padding: "5px 12px", fontSize: 11 }}
                      onClick={() => {
                        const newN = heatmapTopN === 8 ? 20 : 8;
                        setHeatmapTopN(newN);
                        // Refetch with new top_n
                        if (batchResults?.predictions) {
                          setHeatmapLoading(true);
                          const projectFeatures = batchResults.predictions
                            .filter((p: any) => p.status === "success")
                            .map((p: any) => p.project_features);
                          fetch(`${API_BASE}/heatmap/data?top_n=${newN}`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(projectFeatures),
                          }).then(r => r.json()).then(data => {
                            setHeatmapData(data);
                            setHeatmapLoading(false);
                          }).catch(() => setHeatmapLoading(false));
                        }
                      }}
                    >
                      {heatmapTopN === 8 ? "Show All Factors" : "Top 8 Only"}
                    </button>
                    {!heatmapData && (
                      <button
                        className="sim-btn active"
                        style={{ padding: "5px 14px", fontSize: 11 }}
                        disabled={heatmapLoading}
                        onClick={() => {
                          if (!batchResults?.predictions) return;
                          setHeatmapLoading(true);
                          const projectFeatures = batchResults.predictions
                            .filter((p: any) => p.status === "success")
                            .map((p: any) => p.project_features);
                          fetch(`${API_BASE}/heatmap/data?top_n=${heatmapTopN}`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify(projectFeatures),
                          }).then(r => r.json()).then(data => {
                            setHeatmapData(data);
                            setHeatmapLoading(false);
                          }).catch(() => setHeatmapLoading(false));
                        }}
                      >
                        {heatmapLoading ? "Loading..." : "Generate Heatmap"}
                      </button>
                    )}
                  </div>
                </div>

                {heatmapLoading && (
                  <div style={{ padding: 32, textAlign: "center", color: "var(--text-muted)" }}>
                    <span className="loading-spinner" style={{ marginRight: 8 }}></span>
                    Computing SHAP values across all projects...
                  </div>
                )}

                {heatmapData && !heatmapLoading && (
                  <div style={{ padding: "0 16px 16px", overflowX: "auto" }}>
                    {/* Color Legend */}
                    <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 12, fontSize: 11, color: "var(--text-muted)" }}>
                      <span>Reduces Risk</span>
                      <div style={{ display: "flex", height: 12, borderRadius: 6, overflow: "hidden", width: 160 }}>
                        <div style={{ flex: 1, background: "#22C55E" }} />
                        <div style={{ flex: 1, background: "#6BD67E" }} />
                        <div style={{ flex: 1, background: "#A3E635" }} />
                        <div style={{ flex: 1, background: "#FACC15" }} />
                        <div style={{ flex: 1, background: "#FB923C" }} />
                        <div style={{ flex: 1, background: "#EF4444" }} />
                        <div style={{ flex: 1, background: "#DC2626" }} />
                      </div>
                      <span>Increases Risk</span>
                      <span style={{ marginLeft: 16 }}>Showing {heatmapData.columns?.length || 0} of {heatmapData.total_features_available || 0} features</span>
                    </div>

                    {/* Heatmap Grid */}
                    <div style={{ display: "grid", gridTemplateColumns: `140px repeat(${heatmapData.columns?.length || 1}, minmax(80px, 1fr))`, gap: 2, fontSize: 11 }}>
                      {/* Header row */}
                      <div style={{ padding: "8px 6px", fontWeight: 700, color: "var(--text-muted)", fontSize: 10 }}>PROJECT</div>
                      {heatmapData.columns?.map((col: any, ci: number) => (
                        <div key={ci} style={{ padding: "8px 4px", fontWeight: 600, color: "var(--text-secondary)", fontSize: 9, textAlign: "center", lineHeight: 1.3, wordBreak: "break-word" }}>
                          {col.label}
                        </div>
                      ))}

                      {/* Data rows */}
                      {heatmapData.projects?.map((proj: any, ri: number) => (
                        <>
                          {/* Row header */}
                          <div key={`rh-${ri}`} style={{ padding: "6px", display: "flex", alignItems: "center", gap: 6, cursor: "pointer", borderRadius: 4, background: "rgba(255,255,255,0.02)" }}
                            onClick={() => {
                              const bp = batchResults?.predictions?.filter((p: any) => p.status === "success")?.[ri];
                              if (bp) handleSelectBatchProject(bp);
                            }}
                          >
                            <span className={`risk-badge risk-${proj.risk_class}`} style={{ fontSize: 8, padding: "2px 6px" }}>
                              {riskLabel(proj.risk_class).charAt(0)}
                            </span>
                            <span style={{ color: "var(--text-primary)", fontSize: 10, fontWeight: 500 }}>
                              #{proj.index + 1} {proj.industry}
                            </span>
                          </div>

                          {/* Cells */}
                          {heatmapData.matrix?.[ri]?.map((cell: any, ci: number) => {
                            const n = cell.normalized; // -1 to +1
                            const absN = Math.abs(n);
                            let bg: string;
                            if (n > 0) {
                              // Red spectrum for risk-increasing
                              const r = Math.round(239 + (220 - 239) * (1 - absN));
                              const g = Math.round(68 + (180 - 68) * (1 - absN));
                              const b = Math.round(68 + (180 - 68) * (1 - absN));
                              bg = `rgba(${r}, ${g}, ${b}, ${0.15 + absN * 0.65})`;
                            } else {
                              // Green spectrum for risk-reducing
                              const r = Math.round(34 + (180 - 34) * (1 - absN));
                              const g = Math.round(197 + (210 - 197) * (1 - absN));
                              const b = Math.round(94 + (180 - 94) * (1 - absN));
                              bg = `rgba(${r}, ${g}, ${b}, ${0.15 + absN * 0.65})`;
                            }
                            const isHovered = hoveredCell?.row === ri && hoveredCell?.col === ci;
                            return (
                              <div
                                key={`c-${ri}-${ci}`}
                                style={{
                                  background: bg,
                                  borderRadius: 4,
                                  padding: "6px 4px",
                                  textAlign: "center",
                                  cursor: "default",
                                  position: "relative",
                                  transition: "transform 0.15s, box-shadow 0.15s",
                                  transform: isHovered ? "scale(1.08)" : "scale(1)",
                                  boxShadow: isHovered ? "0 4px 16px rgba(0,0,0,0.4)" : "none",
                                  zIndex: isHovered ? 10 : 1,
                                }}
                                onMouseEnter={() => setHoveredCell({ row: ri, col: ci })}
                                onMouseLeave={() => setHoveredCell(null)}
                              >
                                <div style={{ fontSize: 11, fontWeight: 600, color: n > 0 ? "#FCA5A5" : "#86EFAC" }}>
                                  {n > 0 ? "↑" : "↓"} {(absN * 100).toFixed(0)}%
                                </div>
                                {/* Tooltip */}
                                {isHovered && (
                                  <div style={{
                                    position: "absolute",
                                    bottom: "calc(100% + 8px)",
                                    left: "50%",
                                    transform: "translateX(-50%)",
                                    background: "rgba(15, 15, 30, 0.95)",
                                    border: "1px solid rgba(255,255,255,0.15)",
                                    borderRadius: 8,
                                    padding: "10px 14px",
                                    minWidth: 180,
                                    zIndex: 100,
                                    boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
                                    fontSize: 11,
                                    whiteSpace: "nowrap",
                                  }}>
                                    <div style={{ fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>
                                      {heatmapData.columns[ci]?.label}
                                    </div>
                                    <div style={{ color: n > 0 ? "#FCA5A5" : "#86EFAC", marginBottom: 2 }}>
                                      {cell.direction === "increases_risk" ? "⬆ Increases Risk" : "⬇ Reduces Risk"}
                                    </div>
                                    <div style={{ color: "var(--text-muted)" }}>
                                      SHAP: {cell.raw_shap.toFixed(4)} | Intensity: {(absN * 100).toFixed(1)}%
                                    </div>
                                    <div style={{ color: "var(--text-muted)", marginTop: 4 }}>
                                      Project: #{heatmapData.projects[ri]?.index + 1} {heatmapData.projects[ri]?.industry} ({riskLabel(heatmapData.projects[ri]?.risk_class)})
                                    </div>
                                  </div>
                                )}
                              </div>
                            );
                          })}
                        </>
                      ))}
                    </div>
                  </div>
                )}

                {!heatmapData && !heatmapLoading && (
                  <div style={{ padding: "24px 16px", textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
                    Click "Generate Heatmap" to visualize SHAP risk factors across all projects
                  </div>
                )}
              </div>
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

          {/* Loading Shimmer Skeleton */}
          {loading && (
            <div style={{ marginTop: 24 }}>
              <div className="skeleton skeleton-text" style={{ width: "30%", height: 20, marginBottom: 16 }} />
              <div className="skeleton-grid">
                <div className="skeleton skeleton-card" />
                <div className="skeleton skeleton-card" />
                <div className="skeleton skeleton-card" />
              </div>
            </div>
          )}

          {/* Prediction Results */}
          {result && (
            <>
              <div className="divider" />

              {/* Action Bar for Executive Reports & Slack Alerts */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, flexWrap: "wrap", gap: 12 }}>
                <div style={{ fontSize: 18, fontWeight: 700, color: "var(--text-primary)" }}>
                  Project Risk & Overrun Analysis
                </div>
                <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                  <button
                    className="sim-btn active"
                    style={{ padding: "10px 18px", fontSize: 13, display: "flex", alignItems: "center", gap: 8, boxShadow: "0 4px 16px rgba(46, 92, 255, 0.3)" }}
                    onClick={handleGenerateReport}
                    disabled={reportLoading}
                  >
                    <span>📄</span> {reportLoading ? "Generating..." : "Export PMO Report"}
                  </button>
                  <button
                    className="sim-btn"
                    style={{ padding: "10px 18px", fontSize: 13, display: "flex", alignItems: "center", gap: 8, background: "rgba(46, 92, 255, 0.08)", border: "1px solid rgba(46, 92, 255, 0.25)" }}
                    onClick={async () => {
                      if (!form || !result) return;
                      try {
                        const res = await fetch(`${API_BASE}/report/pdf`, {
                          method: "POST",
                          headers: { "Content-Type": "application/json" },
                          body: JSON.stringify({
                            project_features: form,
                            prediction_result: result,
                            simulation_result: simResult,
                          }),
                        });
                        if (!res.ok) throw new Error("PDF generation failed");
                        const blob = await res.blob();
                        const url = window.URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = "DELTA_Risk_Report.pdf";
                        a.click();
                        window.URL.revokeObjectURL(url);
                      } catch (err) {
                        console.error("PDF download failed:", err);
                      }
                    }}
                  >
                    <span>📥</span> Download PDF
                  </button>
                  <button
                    className="sim-btn"
                    style={{ padding: "10px 18px", fontSize: 13, display: "flex", alignItems: "center", gap: 8, background: "rgba(34, 197, 94, 0.1)", border: "1px solid rgba(34, 197, 94, 0.3)" }}
                    onClick={saveBookmark}
                  >
                    <span>⭐</span> Bookmark
                  </button>
                  {(result.risk_class === "at_risk" || result.risk_class === "failed") && (
                    <>
                      <button
                        className="sim-btn"
                        style={{ padding: "10px 18px", fontSize: 13, display: "flex", alignItems: "center", gap: 8, background: result.risk_class === "failed" ? "rgba(239, 68, 68, 0.15)" : "rgba(251, 191, 36, 0.15)", border: `1px solid ${result.risk_class === "failed" ? "rgba(239, 68, 68, 0.4)" : "rgba(251, 191, 36, 0.4)"}` }}
                        onClick={handleSlackAlert}
                        disabled={slackLoading}
                      >
                        <span>🔔</span> {slackLoading ? "Sending..." : "Send Slack Alert"}
                      </button>
                      <button
                        className="sim-btn"
                        style={{ padding: "10px 18px", fontSize: 13, display: "flex", alignItems: "center", gap: 8, background: "rgba(59, 130, 246, 0.15)", border: "1px solid rgba(59, 130, 246, 0.4)" }}
                        onClick={() => {
                          setEmailModalOpen(true);
                          if (!emailResult) handleSendEmailAlert();
                        }}
                      >
                        <span>📧</span> Send Email Alert
                      </button>
                    </>
                  )}
                </div>
              </div>

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

                {/* Risk Trajectory Timeline */}
                <div className="panel glass" style={{ overflow: "hidden" }}>
                  <div className="panel-header" style={{ padding: "14px 20px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                    <div className="panel-title" style={{ fontSize: 14 }}>📈 Risk Trajectory — Milestone Evolution</div>
                    <button
                      className="sim-btn active"
                      style={{ padding: "6px 14px", fontSize: 11 }}
                      disabled={trajectoryLoading}
                      onClick={async () => {
                        if (!form) return;
                        setTrajectoryLoading(true);
                        try {
                          const res = await fetch(`${API_BASE}/trajectory`, {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ project_features: form }),
                          });
                          if (!res.ok) throw new Error(`Trajectory error: ${res.status}`);
                          const data = await res.json();
                          setTrajectoryData(data);
                          addToast("Risk trajectory computed across 6 milestones", "success");
                        } catch (err: any) {
                          console.error("Trajectory failed:", err);
                          addToast(`Trajectory error: ${err.message}`, "error");
                        } finally {
                          setTrajectoryLoading(false);
                        }
                      }}
                    >
                      {trajectoryLoading ? "Computing..." : trajectoryData ? "↻ Refresh" : "Compute Trajectory"}
                    </button>
                  </div>

                  {trajectoryData && (() => {
                    const ms = trajectoryData.milestones || [];
                    const healthColor = (h: string) => h === "healthy" ? "#22C55E" : h === "critical" ? "#EF4444" : "#F59E0B";
                    const riskRankNum = (r: string) => r === "on_track" ? 0 : r === "at_risk" ? 1 : 2;

                    // SVG sparkline points
                    const svgW = 600;
                    const svgH = 80;
                    const points = ms.map((m: any, i: number) => ({
                      x: (i / Math.max(ms.length - 1, 1)) * (svgW - 40) + 20,
                      y: svgH - 12 - (riskRankNum(m.risk_class) / 2) * (svgH - 24),
                    }));
                    const pathD = points.map((p: any, i: number) => `${i === 0 ? "M" : "L"} ${p.x} ${p.y}`).join(" ");

                    return (
                      <div style={{ padding: "0 20px 20px" }}>
                        {/* Escalation Alert */}
                        {trajectoryData.risk_escalation_point && (
                          <div style={{ padding: "10px 14px", borderRadius: 8, background: "rgba(239, 68, 68, 0.08)", border: "1px solid rgba(239, 68, 68, 0.2)", marginBottom: 16, fontSize: 12, color: "#FCA5A5", display: "flex", alignItems: "center", gap: 8 }}>
                            <span style={{ fontSize: 16 }}>⚠</span>
                            <span>Risk escalation detected at <strong style={{ color: "#F87171" }}>{trajectoryData.risk_escalation_point}</strong> milestone — consider early intervention before this phase.</span>
                          </div>
                        )}

                        {/* SVG Sparkline */}
                        <div style={{ overflowX: "auto", marginBottom: 20 }}>
                          <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} style={{ width: "100%", maxWidth: svgW }}>
                            <line x1="20" y1={svgH - 12} x2={svgW - 20} y2={svgH - 12} stroke="var(--glass-border)" strokeWidth="1" />
                            <line x1="20" y1={svgH / 2} x2={svgW - 20} y2={svgH / 2} stroke="var(--glass-border)" strokeWidth="0.5" strokeDasharray="4 4" />
                            <line x1="20" y1="12" x2={svgW - 20} y2="12" stroke="var(--glass-border)" strokeWidth="1" />
                            <text x="2" y={svgH - 8} fontSize="8" fill="var(--text-muted)">OK</text>
                            <text x="2" y={svgH / 2 + 3} fontSize="8" fill="var(--text-muted)">Risk</text>
                            <text x="2" y="16" fontSize="8" fill="var(--text-muted)">Fail</text>
                            <path d={pathD} fill="none" stroke="url(#trajGrad)" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
                            <defs>
                              <linearGradient id="trajGrad" x1="0" y1="0" x2="1" y2="0">
                                <stop offset="0%" stopColor="#22C55E" />
                                <stop offset="50%" stopColor="#F59E0B" />
                                <stop offset="100%" stopColor="#EF4444" />
                              </linearGradient>
                            </defs>
                            {points.map((p: any, i: number) => (
                              <g key={i}>
                                <circle cx={p.x} cy={p.y} r="5" fill={healthColor(ms[i].health)} stroke="#0A0E1A" strokeWidth="2" />
                                <text x={p.x} y={svgH - 1} textAnchor="middle" fontSize="8" fill="var(--text-muted)">{ms[i].milestone_label}</text>
                              </g>
                            ))}
                          </svg>
                        </div>

                        {/* Milestone Cards Row */}
                        <div style={{ display: "grid", gridTemplateColumns: `repeat(${ms.length}, 1fr)`, gap: 8 }}>
                          {ms.map((m: any, i: number) => (
                            <div key={i} style={{
                              background: "var(--glass-bg)",
                              border: `1px solid ${m.health === "critical" ? "rgba(239,68,68,0.3)" : m.health === "warning" ? "rgba(245,158,11,0.3)" : "var(--glass-border)"}`,
                              borderRadius: 10,
                              padding: "12px 10px",
                              textAlign: "center",
                            }}>
                              <div style={{ width: 8, height: 8, borderRadius: "50%", background: healthColor(m.health), margin: "0 auto 8px", boxShadow: `0 0 8px ${healthColor(m.health)}40` }} />
                              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>{m.milestone_label}</div>
                              <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 6 }}>Week {m.week_number}</div>
                              <span className={`risk-badge risk-${m.risk_class}`} style={{ fontSize: 9, padding: "2px 8px" }}>
                                {riskLabel(m.risk_class)}
                              </span>
                              <div style={{ fontSize: 11, fontWeight: 600, color: m.overrun_percentage > 0 ? "#F87171" : "#34D399", marginTop: 6 }}>
                                {m.overrun_percentage > 0 ? "+" : ""}{m.overrun_percentage.toFixed(1)}%
                              </div>
                              <div style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 4 }}>
                                {m.top_factor?.replace(/_/g, " ") || "—"}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    );
                  })()}

                  {!trajectoryData && !trajectoryLoading && (
                    <div style={{ padding: "24px 20px", textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
                      Click &quot;Compute Trajectory&quot; to simulate how this project&apos;s risk evolves across 6 milestone phases (Kickoff → Planning → Build → Testing → UAT → Go-Live).
                    </div>
                  )}
                </div>

                {/* Expanded What-If Simulation Panel */}
                <div className="sim-panel full-panel">
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
                    <div>
                      <div className="sim-title" style={{ fontSize: 18 }}>
                        <span>⚡</span> Interactive What-If Counterfactual Scenario Simulator
                      </div>
                      <div className="sim-subtitle" style={{ fontSize: 13, marginTop: 4 }}>
                        Simulate real-time operational shifts against the trained XGBoost model to quantify financial and risk impact before committing resources.
                      </div>
                    </div>
                    {simResult && (
                      <span className={`sim-delta-badge ${simResult.delta.is_improvement ? "pos" : "neg"}`} style={{ fontSize: 14, padding: "8px 16px" }}>
                        {simResult.delta.is_improvement ? "✓ Favorable Outcome" : "⚠ Adverse Impact"}
                      </span>
                    )}
                  </div>

                  {/* Parameter Controls Grid */}
                  <div className="sim-grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 20, marginTop: 20 }}>
                    {/* Team Size Controls + State Explanation */}
                    <div className="sim-control-group" style={{ background: "rgba(255,255,255,0.02)", padding: 16, borderRadius: "var(--radius-md)", border: "1px solid var(--glass-border)" }}>
                      <div className="sim-control-label" style={{ fontSize: 13, display: "flex", justifyContent: "space-between" }}>
                        <span>👥 Team Size Adjustment</span>
                        <span style={{ color: "var(--accent-blue)" }}>Baseline: {form.team_size} members</span>
                      </div>
                      <div className="sim-btn-row" style={{ marginTop: 8 }}>
                        {[-3, -2, -1, 0, 1, 2, 3].map((d) => (
                          <button
                            key={d}
                            className={`sim-btn ${simTeamDelta === d ? "active" : ""}`}
                            style={{ flex: 1, padding: "8px 0" }}
                            onClick={() => {
                              setSimTeamDelta(d);
                              handleSimulate(d, simScopeDelta, simClientType);
                            }}
                          >
                            {d > 0 ? `+${d}` : d}
                          </button>
                        ))}
                      </div>
                      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 10, lineHeight: 1.5 }}>
                        {simTeamDelta === 0 && "• Operating at baseline staffing level."}
                        {simTeamDelta > 0 && `• Adding ${simTeamDelta} developer(s) increases throughput but raises labor burn rate by est. $${(simTeamDelta * 12000).toLocaleString()}/mo.`}
                        {simTeamDelta < 0 && `• Reducing team by ${Math.abs(simTeamDelta)} member(s) cuts immediate labor expense but elevates bottleneck & burn variance risks.`}
                      </div>
                    </div>

                    {/* Scope Changes Controls + State Explanation */}
                    <div className="sim-control-group" style={{ background: "rgba(255,255,255,0.02)", padding: 16, borderRadius: "var(--radius-md)", border: "1px solid var(--glass-border)" }}>
                      <div className="sim-control-label" style={{ fontSize: 13, display: "flex", justifyContent: "space-between" }}>
                        <span>📝 Scope Change Requests</span>
                        <span style={{ color: "var(--accent-blue)" }}>Baseline: {form.scope_change_count} changes</span>
                      </div>
                      <div className="sim-btn-row" style={{ marginTop: 8 }}>
                        {[-3, -2, -1, 0, 1, 2, 3].map((d) => (
                          <button
                            key={d}
                            className={`sim-btn ${simScopeDelta === d ? "active" : ""}`}
                            style={{ flex: 1, padding: "8px 0" }}
                            onClick={() => {
                              setSimScopeDelta(d);
                              handleSimulate(simTeamDelta, d, simClientType);
                            }}
                          >
                            {d > 0 ? `+${d}` : d}
                          </button>
                        ))}
                      </div>
                      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 10, lineHeight: 1.5 }}>
                        {simScopeDelta === 0 && "• Evaluating under baseline scope change frequency."}
                        {simScopeDelta < 0 && `• Freeze/Negotiate ${Math.abs(simScopeDelta)} scope change(s): Significantly reduces re-work overhead and cost overrun probability.`}
                        {simScopeDelta > 0 && `• Accepting ${simScopeDelta} additional scope change(s): Expands feature set but compounds schedule slippage and cost overrun risk.`}
                      </div>
                    </div>

                    {/* Contract Type Controls + State Explanation */}
                    <div className="sim-control-group" style={{ background: "rgba(255,255,255,0.02)", padding: 16, borderRadius: "var(--radius-md)", border: "1px solid var(--glass-border)" }}>
                      <div className="sim-control-label" style={{ fontSize: 13, display: "flex", justifyContent: "space-between" }}>
                        <span>📑 Contract Structure</span>
                        <span style={{ color: "var(--accent-blue)" }}>Current: {simClientType || form.client_type}</span>
                      </div>
                      <div className="sim-btn-row" style={{ marginTop: 8 }}>
                        {[
                          { label: "Fixed Bid", val: "fixed_bid" },
                          { label: "Outcome Based", val: "outcome_based" },
                          { label: "Time & Material", val: "time_and_material" },
                        ].map((c) => (
                          <button
                            key={c.val}
                            className={`sim-btn ${(simClientType || form.client_type) === c.val ? "active" : ""}`}
                            style={{ flex: 1, padding: "8px 4px", fontSize: 11 }}
                            onClick={() => {
                              setSimClientType(c.val);
                              handleSimulate(simTeamDelta, simScopeDelta, c.val);
                            }}
                          >
                            {c.label}
                          </button>
                        ))}
                      </div>
                      <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 10, lineHeight: 1.5 }}>
                        {(simClientType || form.client_type) === "fixed_bid" && "• Fixed Bid: High vendor margin risk if scope/burn varies, penalizes cost overruns heavily."}
                        {(simClientType || form.client_type) === "outcome_based" && "• Outcome-Based: Aligns milestone delivery with risk sharing, reducing financial penalty on overruns."}
                        {(simClientType || form.client_type) === "time_and_material" && "• Time & Material: Passes cost variance to client, reducing direct project insolvency risk."}
                      </div>
                    </div>
                  </div>

                  {/* Detailed Simulation Outcome Display */}
                  {simResult && (
                    <div className={`sim-banner ${simResult.delta.is_improvement ? "improved" : "worsened"}`} style={{ flexDirection: "column", alignItems: "stretch", gap: 16, marginTop: 16, padding: 20 }}>
                      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid var(--glass-border)", paddingBottom: 12 }}>
                        <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>
                          Simulation Analysis Summary
                        </div>
                        <div style={{ fontSize: 12, color: simResult.delta.is_improvement ? "#10B981" : "#EF4444", fontWeight: 600 }}>
                          {simResult.delta.risk_changed
                            ? `Risk Status Changed: ${simResult.delta.baseline_risk} ➔ ${simResult.delta.simulated_risk}`
                            : `Risk Status Maintained: ${simResult.delta.simulated_risk}`}
                        </div>
                      </div>

                      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))", gap: 16 }}>
                        <div className="sim-metric-box">
                          <div className="sim-metric-label">Simulated Cost</div>
                          <div className="sim-metric-val" style={{ fontSize: 20 }}>
                            {currency === "USD"
                              ? `$${simResult.simulated_prediction.predicted_final_cost_usd.toLocaleString()}`
                              : `₹${simResult.simulated_prediction.predicted_final_cost_inr.toLocaleString()}`}
                          </div>
                          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                            Baseline: {currency === "USD" ? `$${simResult.baseline_prediction.predicted_final_cost_usd.toLocaleString()}` : `₹${simResult.baseline_prediction.predicted_final_cost_inr.toLocaleString()}`}
                          </div>
                        </div>

                        <div className="sim-metric-box">
                          <div className="sim-metric-label">Net Variance ($\Delta$)</div>
                          <div className="sim-metric-val" style={{ fontSize: 20, color: simResult.delta.cost_diff_usd <= 0 ? "#10B981" : "#EF4444" }}>
                            {simResult.delta.cost_diff_usd <= 0 ? "-" : "+"}
                            {currency === "USD"
                              ? `$${Math.abs(simResult.delta.cost_diff_usd).toLocaleString()}`
                              : `₹${Math.abs(simResult.delta.cost_diff_inr).toLocaleString()}`}
                          </div>
                          <div style={{ fontSize: 11, color: simResult.delta.cost_diff_usd <= 0 ? "#10B981" : "#EF4444" }}>
                            {simResult.delta.cost_diff_usd <= 0 ? "↓ Direct Labor / Scope Savings" : "↑ Additional Overhead Cost"}
                          </div>
                        </div>

                        <div className="sim-metric-box">
                          <div className="sim-metric-label">Overrun Shift</div>
                          <div className="sim-metric-val" style={{ fontSize: 20 }}>
                            {simResult.simulated_prediction.overrun_percentage > 0 ? `+${simResult.simulated_prediction.overrun_percentage.toFixed(1)}%` : `${simResult.simulated_prediction.overrun_percentage.toFixed(1)}%`}
                          </div>
                          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                            Shift: {simResult.delta.overrun_diff_pct > 0 ? `+${simResult.delta.overrun_diff_pct.toFixed(1)}%` : `${simResult.delta.overrun_diff_pct.toFixed(1)}%`} from baseline
                          </div>
                        </div>

                        <div className="sim-metric-box">
                          <div className="sim-metric-label">Confidence Score</div>
                          <div className="sim-metric-val" style={{ fontSize: 20 }}>
                            {(simResult.simulated_prediction.risk_confidence * 100).toFixed(0)}%
                          </div>
                          <div style={{ fontSize: 11, color: "var(--text-muted)" }}>
                            {simResult.delta.confidence_diff >= 0 ? `+${(simResult.delta.confidence_diff * 100).toFixed(1)}%` : `${(simResult.delta.confidence_diff * 100).toFixed(1)}%`} confidence shift
                          </div>
                        </div>
                      </div>

                      {/* DELTA Copilot — Embedded Scenario AI Advisor (Placed in designated layout space) */}
                      <div style={{
                        background: "rgba(13, 18, 36, 0.85)",
                        border: "1px solid rgba(46, 92, 255, 0.35)",
                        borderRadius: "var(--radius-md)",
                        padding: "16px 18px",
                        marginTop: "16px",
                        boxShadow: "0 4px 20px rgba(0, 0, 0, 0.3)",
                        display: "flex",
                        flexDirection: "column",
                        gap: "12px",
                      }}>
                        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "10px" }}>
                          <div style={{ fontSize: "13px", fontWeight: "700", color: "var(--text-primary)", display: "flex", alignItems: "center", gap: "8px" }}>
                            <span style={{ fontSize: "16px" }}>🤖</span> DELTA Copilot — Scenario AI Advisor
                          </div>
                          <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
                            <span style={{ fontSize: "11px", color: "#60A5FA", background: "rgba(46, 92, 255, 0.15)", padding: "3px 10px", borderRadius: "12px", border: "1px solid rgba(46, 92, 255, 0.3)", fontWeight: 600 }}>
                              📚 PMBOK & SHAP Grounded
                            </span>
                          </div>
                        </div>

                        {/* Messages Area */}
                        <div className="copilot-messages" style={{ minHeight: "140px", maxHeight: "240px", background: "rgba(0,0,0,0.3)", borderRadius: "var(--radius-sm)", padding: "12px", border: "1px solid rgba(255,255,255,0.06)", overflowY: "auto" }}>
                          {copilotMessages.length === 0 ? (
                            <div style={{ fontSize: "12.5px", color: "var(--text-muted)", textAlign: "center", padding: "24px 12px", lineHeight: "1.6" }}>
                              💬 Ask DELTA Copilot about this scenario risk, financial variance, or PMBOK recommendations using the quick chips or text box below...
                            </div>
                          ) : (
                            copilotMessages.map((msg, i) => (
                              <div key={i} className={`copilot-msg ${msg.role === "user" ? "user" : "assistant"}`}>
                                <div style={{ fontWeight: 600, fontSize: "11px", marginBottom: "4px", opacity: 0.85 }}>
                                  {msg.role === "user" ? "You" : "🤖 DELTA Copilot"}
                                </div>
                                <div>
                                  {msg.content.split("\n").map((line, j) => (
                                    <span key={j}>
                                      {line.split(/\*\*(.*?)\*\*/g).map((part, k) =>
                                        k % 2 === 1 ? <strong key={k}>{part}</strong> : part
                                      )}
                                      {j < msg.content.split("\n").length - 1 && <br />}
                                    </span>
                                  ))}
                                </div>
                              </div>
                            ))
                          )}
                          {copilotLoading && (
                            <div className="copilot-typing">
                              <span /><span /><span />
                            </div>
                          )}
                          <div ref={copilotMessagesEnd} />
                        </div>

                        {/* Quick Chips */}
                        <div className="copilot-chips" style={{ borderTop: "none", padding: 0 }}>
                          {QUICK_QUESTIONS.map((q, i) => (
                            <button key={i} className="copilot-chip" type="button" onClick={() => handleCopilotSend(q)}>{q}</button>
                          ))}
                        </div>

                        {/* Input Row */}
                        <div className="copilot-input-row" style={{ borderTop: "none", padding: 0 }}>
                          <input
                            id="embedded-copilot-input"
                            className="copilot-input"
                            placeholder="Ask about this simulation outcome, trade-offs, or PMBOK guidelines..."
                            value={copilotInput}
                            onChange={e => setCopilotInput(e.target.value)}
                            onKeyDown={e => {
                              if (e.key === "Enter" && !copilotLoading) {
                                e.preventDefault();
                                handleCopilotSend();
                              }
                            }}
                            disabled={copilotLoading}
                            autoComplete="off"
                          />
                          <button
                            className="copilot-send"
                            type="button"
                            onClick={() => handleCopilotSend()}
                            disabled={copilotLoading || !copilotInput.trim()}
                          >
                            Send
                          </button>
                        </div>
                      </div>

                      {/* Highlighted Analytical Inference Box */}
                      <div style={{
                        background: "rgba(46, 92, 255, 0.08)",
                        border: "1px solid rgba(46, 92, 255, 0.3)",
                        borderRadius: "var(--radius-md)",
                        padding: "16px 20px",
                        marginTop: "16px",
                      }}>
                        <div style={{ fontSize: "13px", fontWeight: "700", color: "#60A5FA", marginBottom: "8px", display: "flex", alignItems: "center", gap: "8px" }}>
                          <span>🔍</span> Scenario Outcome Text Inference
                        </div>
                        <div style={{ fontSize: "12.5px", color: "var(--text-primary)", lineHeight: "1.7" }}>
                          Under this simulated scenario, the project yields a predicted final cost of{" "}
                          <strong>
                            {currency === "USD"
                              ? `$${simResult.simulated_prediction.predicted_final_cost_usd.toLocaleString()}`
                              : `₹${simResult.simulated_prediction.predicted_final_cost_inr.toLocaleString()}`}
                          </strong>{" "}
                          (baseline:{" "}
                          {currency === "USD"
                            ? `$${simResult.baseline_prediction.predicted_final_cost_usd.toLocaleString()}`
                            : `₹${simResult.baseline_prediction.predicted_final_cost_inr.toLocaleString()}`}
                          ), producing a net variance of{" "}
                          <strong style={{ color: simResult.delta.cost_diff_usd <= 0 ? "#34D399" : "#F87171" }}>
                            {simResult.delta.cost_diff_usd <= 0 ? "-" : "+"}
                            {currency === "USD"
                              ? `$${Math.abs(simResult.delta.cost_diff_usd).toLocaleString()}`
                              : `₹${Math.abs(simResult.delta.cost_diff_inr).toLocaleString()}`}
                          </strong>{" "}
                          ({simResult.delta.cost_diff_usd <= 0 ? "labor/scope savings" : "additional overhead cost"}).{" "}
                          The overall overrun ratio stands at <strong>{simResult.simulated_prediction.overrun_percentage > 0 ? `+${simResult.simulated_prediction.overrun_percentage.toFixed(1)}%` : `${simResult.simulated_prediction.overrun_percentage.toFixed(1)}%`}</strong>{" "}
                          ({simResult.delta.overrun_diff_pct > 0 ? `+${simResult.delta.overrun_diff_pct.toFixed(1)}% shift` : `${simResult.delta.overrun_diff_pct.toFixed(1)}% shift`} from baseline), while the project risk status is{" "}
                          <strong>
                            {simResult.delta.risk_changed
                              ? `changed from ${simResult.delta.baseline_risk.toUpperCase()} ➔ ${simResult.delta.simulated_risk.toUpperCase()}`
                              : `MAINTAINED as ${simResult.delta.simulated_risk.toUpperCase()}`}
                          </strong>{" "}
                          with <strong>{(simResult.simulated_prediction.risk_confidence * 100).toFixed(0)}%</strong> model confidence.
                        </div>
                      </div>

                      {/* Point-Wise Executive Analysis & State Explanations */}
                      <div style={{
                        background: "rgba(0, 0, 0, 0.3)",
                        border: "1px solid var(--glass-border)",
                        borderRadius: "var(--radius-md)",
                        padding: "20px",
                        marginTop: "8px",
                      }}>
                        <div style={{
                          fontSize: "13px",
                          fontWeight: "700",
                          color: "var(--text-primary)",
                          marginBottom: "14px",
                          display: "flex",
                          alignItems: "center",
                          gap: "8px",
                          borderBottom: "1px solid rgba(255,255,255,0.08)",
                          paddingBottom: "8px"
                        }}>
                          <span>📋</span> Scenario State Analysis & Operational Insights (Point-Wise Breakdown)
                        </div>

                        <div style={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                          gap: "20px"
                        }}>
                          {/* Point 1: Operational Impact */}
                          <div style={{ background: "rgba(255,255,255,0.02)", padding: "14px", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.05)" }}>
                            <div style={{ fontSize: "12px", fontWeight: "700", color: "#60A5FA", marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
                              <span>👥</span> Operational Trade-Offs
                            </div>
                            <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.7" }}>
                              <li>
                                <strong>Staffing:</strong> {simTeamDelta === 0 ? "Baseline team size maintained." : simTeamDelta > 0 ? `Added ${simTeamDelta} developer(s). Increases throughput but adds est. $${(simTeamDelta * 12000).toLocaleString()}/mo labor burn.` : `Reduced by ${Math.abs(simTeamDelta)} developer(s). Direct cost reduction, but elevates burn rate variance risk.`}
                              </li>
                              <li>
                                <strong>Scope Control:</strong> {simScopeDelta === 0 ? "Baseline scope change count." : simScopeDelta < 0 ? `Freezing ${Math.abs(simScopeDelta)} scope change(s) prevents project scope creep and cuts re-work overhead.` : `Accepting ${simScopeDelta} new scope change(s) introduces delivery bottleneck risks.`}
                              </li>
                              <li>
                                <strong>Contract Risk:</strong> {(simClientType || form.client_type) === "fixed_bid" ? "Fixed Bid contract penalizes cost overruns heavily against vendor margins." : (simClientType || form.client_type) === "outcome_based" ? "Outcome-Based contract shares risk based on milestone deliverables." : "Time & Material passes burn variance to client."}
                              </li>
                            </ul>
                          </div>

                          {/* Point 2: Financial Delta */}
                          <div style={{ background: "rgba(255,255,255,0.02)", padding: "14px", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.05)" }}>
                            <div style={{ fontSize: "12px", fontWeight: "700", color: simResult.delta.is_improvement ? "#34D399" : "#F87171", marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
                              <span>📊</span> Financial & Risk Variance
                            </div>
                            <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.7" }}>
                              <li>
                                <strong>Net Impact:</strong> {simResult.delta.cost_diff_usd <= 0 ? <span style={{ color: "#34D399" }}>Est. Savings of {currency === "USD" ? `$${Math.abs(simResult.delta.cost_diff_usd).toLocaleString()}` : `₹${Math.abs(simResult.delta.cost_diff_inr).toLocaleString()}`}</span> : <span style={{ color: "#F87171" }}>Est. Cost Excess of {currency === "USD" ? `$${Math.abs(simResult.delta.cost_diff_usd).toLocaleString()}` : `₹${Math.abs(simResult.delta.cost_diff_inr).toLocaleString()}`}</span>}
                              </li>
                              <li>
                                <strong>Overrun Shift:</strong> {simResult.delta.overrun_diff_pct === 0 ? "No change from baseline overrun." : simResult.delta.overrun_diff_pct < 0 ? `Overrun reduced by ${Math.abs(simResult.delta.overrun_diff_pct).toFixed(1)}%` : `Overrun increased by ${simResult.delta.overrun_diff_pct.toFixed(1)}%`}
                              </li>
                              <li>
                                <strong>Risk Status:</strong> {simResult.delta.risk_changed ? `Status changed from ${simResult.delta.baseline_risk.toUpperCase()} ➔ ${simResult.delta.simulated_risk.toUpperCase()}` : `Maintained as ${simResult.delta.simulated_risk.toUpperCase()} (${(simResult.simulated_prediction.risk_confidence * 100).toFixed(0)}% model confidence)`}
                              </li>
                            </ul>
                          </div>

                          {/* Point 3: Strategic Recommendation */}
                          <div style={{ background: "rgba(255,255,255,0.02)", padding: "14px", borderRadius: "var(--radius-sm)", border: "1px solid rgba(255,255,255,0.05)" }}>
                            <div style={{ fontSize: "12px", fontWeight: "700", color: "#FBBF24", marginBottom: "8px", display: "flex", alignItems: "center", gap: "6px" }}>
                              <span>💡</span> Strategic PM Guidance
                            </div>
                            <ul style={{ margin: 0, paddingLeft: "16px", fontSize: "12px", color: "var(--text-secondary)", lineHeight: "1.7" }}>
                              {simResult.delta.is_improvement ? (
                                <>
                                  <li><strong>Recommended Action:</strong> Proceed with this scenario. It provides a net positive financial buffer.</li>
                                  <li><strong>Key Benefit:</strong> Lower risk classification without exceeding approved budget limits.</li>
                                </>
                              ) : (
                                <>
                                  <li><strong>Recommended Action:</strong> Counter-balance by reducing scope changes by 1-2 items to offset labor cost increases.</li>
                                  <li><strong>Mitigation Strategy:</strong> Consider negotiating contract structure to Outcome-Based pricing.</li>
                                </>
                              )}
                            </ul>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
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

      {/* Executive Report Modal Drawer */}
      {reportOpen && (
        <div className="report-modal-overlay" onClick={() => setReportOpen(false)}>
          <div className="report-modal" onClick={(e) => e.stopPropagation()}>
            <div className="report-modal-header">
              <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 8 }}>
                <span>📄</span> PMO Executive Audit Report
              </div>
              <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
                <button
                  className="sim-btn"
                  onClick={() => {
                    if (reportContent) {
                      navigator.clipboard.writeText(reportContent);
                      setReportCopied(true);
                      setTimeout(() => setReportCopied(false), 2000);
                    }
                  }}
                  disabled={!reportContent}
                >
                  {reportCopied ? "✓ Copied!" : "📋 Copy Markdown"}
                </button>
                <button
                  className="sim-btn active"
                  onClick={() => window.print()}
                  disabled={!reportContent}
                >
                  🖨️ Print / Save as PDF
                </button>
                <button className="copilot-close" onClick={() => setReportOpen(false)}>✕</button>
              </div>
            </div>

            <div className="report-modal-body">
              {reportLoading ? (
                <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)" }}>
                  Generating PMO Executive Audit Report...
                </div>
              ) : (
                reportContent
              )}
            </div>
          </div>
        </div>
      )}

      {/* Slack Alert Preview Modal */}
      {slackPreview && (
        <div className="report-modal-overlay" onClick={() => setSlackPreview(null)}>
          <div className="report-modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 560 }}>
            <div className="report-modal-header">
              <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", display: "flex", alignItems: "center", gap: 8 }}>
                <span>🔔</span> Slack Risk Alert {slackPreview.status === "sent" ? "— Sent ✓" : "— Dry Run Preview"}
              </div>
              <button className="copilot-close" onClick={() => setSlackPreview(null)}>✕</button>
            </div>

            <div className="report-modal-body" style={{ padding: "20px 24px" }}>
              {slackPreview.status === "dry_run" && (
                <div style={{ background: "rgba(251, 191, 36, 0.1)", border: "1px solid rgba(251, 191, 36, 0.3)", borderRadius: "var(--radius-sm)", padding: "10px 14px", marginBottom: 16, fontSize: 12, color: "#FBBF24" }}>
                  ⚠ No SLACK_WEBHOOK_URL configured. This is a preview of what would be sent.
                </div>
              )}
              {slackPreview.status === "sent" && (
                <div style={{ background: "rgba(16, 185, 129, 0.1)", border: "1px solid rgba(16, 185, 129, 0.3)", borderRadius: "var(--radius-sm)", padding: "10px 14px", marginBottom: 16, fontSize: 12, color: "#10B981" }}>
                  ✓ Alert delivered to Slack successfully.
                </div>
              )}

              {/* Render Slack blocks as a preview card */}
              <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: "var(--radius-md)", padding: "16px", border: "1px solid rgba(255,255,255,0.08)" }}>
                {slackPreview.slack_payload?.blocks?.map((block: any, i: number) => {
                  if (block.type === "header") {
                    return <div key={i} style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)", marginBottom: 12 }}>{block.text?.text}</div>;
                  }
                  if (block.type === "divider") {
                    return <hr key={i} style={{ border: "none", borderTop: "1px solid rgba(255,255,255,0.1)", margin: "12px 0" }} />;
                  }
                  if (block.type === "section" && block.fields) {
                    return (
                      <div key={i} style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px 16px", marginBottom: 10 }}>
                        {block.fields.map((f: any, j: number) => (
                          <div key={j} style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.5 }}>
                            {f.text.split("\n").map((line: string, k: number) => (
                              <div key={k} style={k === 0 ? { fontWeight: 700, color: "var(--text-primary)", fontSize: 11, marginBottom: 2 } : {}}>{line.replace(/\*/g, "").replace(/`/g, "")}</div>
                            ))}
                          </div>
                        ))}
                      </div>
                    );
                  }
                  if (block.type === "section" && block.text) {
                    return (
                      <div key={i} style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 10, whiteSpace: "pre-wrap" }}>
                        {block.text.text.replace(/\*/g, "").split("\n").map((line: string, j: number) => (
                          <div key={j}>{line}</div>
                        ))}
                      </div>
                    );
                  }
                  if (block.type === "context") {
                    return (
                      <div key={i} style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 8, fontStyle: "italic" }}>
                        {block.elements?.[0]?.text?.replace(/_/g, "").replace(/\*/g, "")}
                      </div>
                    );
                  }
                  return null;
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* History Drawer Overlay */}
      {historyDrawerOpen && (
        <div
          style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.5)", zIndex: 9998, backdropFilter: "blur(2px)" }}
          onClick={() => setHistoryDrawerOpen(false)}
        />
      )}

      {/* History Drawer */}
      <div
        style={{
          position: "fixed", top: 0, right: historyDrawerOpen ? 0 : -420, width: 400, height: "100vh",
          background: "var(--bg-primary)", borderLeft: "1px solid var(--glass-border)",
          zIndex: 9999, transition: "right 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          display: "flex", flexDirection: "column", boxShadow: historyDrawerOpen ? "-8px 0 32px rgba(0,0,0,0.3)" : "none",
        }}
      >
        {/* Drawer Header */}
        <div style={{ padding: "20px", borderBottom: "1px solid var(--glass-border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)" }}>📑 Saved Predictions</div>
            <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 4 }}>{bookmarks.length} bookmark{bookmarks.length !== 1 ? "s" : ""} saved</div>
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            {bookmarks.length > 0 && (
              <button
                className="sim-btn"
                style={{ padding: "4px 10px", fontSize: 10, color: "#EF4444", border: "1px solid rgba(239,68,68,0.3)", background: "rgba(239,68,68,0.05)" }}
                onClick={clearAllBookmarks}
              >
                Clear All
              </button>
            )}
            <button
              className="sim-btn"
              style={{ padding: "4px 10px", fontSize: 14 }}
              onClick={() => setHistoryDrawerOpen(false)}
            >
              ✕
            </button>
          </div>
        </div>

        {/* Drawer Body */}
        <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px" }}>
          {bookmarks.length === 0 ? (
            <div style={{ textAlign: "center", padding: "60px 20px", color: "var(--text-muted)" }}>
              <div style={{ fontSize: 40, marginBottom: 12 }}>📌</div>
              <div style={{ fontSize: 13, marginBottom: 6 }}>No bookmarks yet</div>
              <div style={{ fontSize: 11 }}>Run a prediction and click ⭐ Bookmark to save it here.</div>
            </div>
          ) : (
            bookmarks.map((bm) => (
              <div
                key={bm.id}
                style={{
                  background: "var(--glass-bg)", border: "1px solid var(--glass-border)", borderRadius: 12,
                  padding: "14px 16px", marginBottom: 10, cursor: "pointer",
                  transition: "border-color 0.2s, transform 0.15s",
                }}
                onClick={() => restoreBookmark(bm)}
                onMouseEnter={e => { (e.currentTarget as HTMLElement).style.borderColor = "#2E5CFF"; (e.currentTarget as HTMLElement).style.transform = "translateX(-2px)"; }}
                onMouseLeave={e => { (e.currentTarget as HTMLElement).style.borderColor = "var(--glass-border)"; (e.currentTarget as HTMLElement).style.transform = "translateX(0)"; }}
              >
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 8 }}>
                  <div style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)", flex: 1 }}>{bm.label}</div>
                  <button
                    className="sim-btn"
                    style={{ padding: "2px 6px", fontSize: 10, color: "#EF4444", border: "none", background: "transparent", flexShrink: 0 }}
                    onClick={e => { e.stopPropagation(); deleteBookmark(bm.id); }}
                  >
                    🗑
                  </button>
                </div>
                <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 6 }}>
                  <span className={`risk-badge risk-${bm.risk_class}`} style={{ fontSize: 9, padding: "2px 8px" }}>
                    {riskLabel(bm.risk_class)}
                  </span>
                  <span style={{ fontSize: 11, fontWeight: 600, color: bm.overrun_pct > 0 ? "#F87171" : "#34D399" }}>
                    {bm.overrun_pct > 0 ? "+" : ""}{bm.overrun_pct.toFixed(1)}% overrun
                  </span>
                </div>
                <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
                  {new Date(bm.timestamp).toLocaleDateString()} {new Date(bm.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
