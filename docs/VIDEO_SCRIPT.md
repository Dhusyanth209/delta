# DELTA — Demo Video Script (2:30)

> **Target Length**: 2 minutes 30 seconds  
> **Format**: Screen recording with voiceover narration  
> **Resolution**: 1920×1080 recommended

---

## 0:00–0:20 — Problem Statement (20s)

### [SCREEN: Title slide — DELTA logo (Δ) with tagline on dark background]

**NARRATION:**
> "Every year, IT services companies lose millions to projects that overrun their budgets — and the painful part is, the warning signs were there all along. Employee costs are rising faster than revenue — two hundred and six percent growth in costs versus a hundred and eighty-five percent in revenue over the last decade. When a project starts slipping under a fixed-bid or outcome-based contract, every week of late detection compounds the loss. DELTA is built to catch those signals early."

### [SCREEN: Fade to dashboard at localhost:3000 — empty state with Δ icon]

---

## 0:20–1:00 — Dashboard Demo (40s)

### [SCREEN: Click "Load Sample Projects" button]

**NARRATION:**
> "Here's the live dashboard. I'm loading sample projects from our test set — these are real predictions from the trained model, not hardcoded values."

### [SCREEN: Sample projects table appears with 8 projects showing industry, team size, budget, risk level badges]

> "You can see a mix of risk levels — on-track in green, at-risk in amber, failed in red. Let me click on this BFSI project with a large budget and several scope changes..."

### [SCREEN: Click on an at-risk or failed project — prediction results appear below]

> "The model predicts this project is at risk with 97% confidence, projecting a 25% cost overrun. But what's more useful than the label is the *why*."

### [SCREEN: Scroll down to the SHAP factors panel]

> "These are the top three factors driving this prediction, powered by SHAP explainability. It's telling us: high scope-change count on a fixed-bid contract is the biggest risk factor, employee costs above the 57% industry baseline are squeezing margins, and team attrition is adding lateral-hire premiums. This is language a project manager can act on."

### [SCREEN: Toggle currency from USD to INR — values update]

> "And for our Indian IT context, we can toggle between dollar and rupee values."

### [SCREEN: Click on an on-track project for contrast]

> "Compare that to this smaller project — low scope changes, stable team, T&M contract. The model gives it a green on-track rating with 99% confidence and only a 0.2% overrun."

---

## 1:00–2:00 — Code & Model Walkthrough (60s)

### [SCREEN: Switch to VS Code / editor — open data/generate_dataset.py]

**NARRATION:**
> "Let me quickly walk through how this works under the hood. The dataset is synthetic — 950 IT project records — but the parameters are grounded in real industry research."

### [SCREEN: Scroll to show the research-cited comments in the code]

> "Every key assumption is cited. The employee cost ratio is centered at 57% — that's the real industry average. Attrition events are modeled at 13-14% annualized, with each departure triggering a 25-30% lateral-hire cost premium. These numbers come from published research, not guesswork."

### [SCREEN: Switch to model/train_model.py — show the metrics output section]

> "The model itself is XGBoost — a gradient-boosted classifier for risk labels, plus a separate regressor for continuous cost prediction."

### [SCREEN: Open model/artifacts/metrics.json or show the terminal output]

> "Our real accuracy numbers: 71.6% on a three-class problem with intentional noise. That's deliberate — real data has unexplained variance, and a 98% accuracy model would mean our synthetic data was too clean, not that our model was better."

### [SCREEN: Open model/artifacts/shap_summary.png]

> "And here's the SHAP analysis showing which features the model weighs most. Scope change count and employee cost ratio are at the top — which aligns with what the research tells us about margin pressure drivers."

### [SCREEN: Open model/artifacts/confusion_matrix.png]

> "The confusion matrix shows the model handles all three classes reasonably well — it's not just memorizing the majority class."

---

## 2:00–2:30 — Impact & Closing (30s)

### [SCREEN: Return to dashboard with a prediction visible]

**NARRATION:**
> "So what would this mean for a real PMO team? Instead of waiting for a quarterly review to discover a project is bleeding money, they'd get an early warning — with specific, explainable factors they can address."

### [SCREEN: Show a slide or text overlay with key numbers]

> "The research benchmarks suggest AI-augmented project management can deliver 30-40% operational cost reduction with a 6-12 month ROI. Our target market isn't the top-5 IT giants who build their own tools — it's the mid-cap firms facing the same margin pressure without the R&D budget."

### [SCREEN: Return to dashboard]

> "To be clear — this model is trained on synthetic data. The approach works, the explainability is real, but validation on actual company data is the necessary next step. That's what we'd need a partnership to prove."

### [SCREEN: Show GitHub repo link]

> "DELTA. Early-warning intelligence for IT project delivery. Check out the repo — and thanks for watching."

### [SCREEN: Fade to black with GitHub URL + team name]

---

## Recording Checklist

- [ ] Screen resolution set to 1920×1080
- [ ] Browser zoom at 100%
- [ ] Backend running on localhost:8000
- [ ] Frontend running on localhost:3000
- [ ] Browser dev tools closed (clean view)
- [ ] Microphone levels tested
- [ ] Total runtime: under 2:30
- [ ] GitHub repo set to public
- [ ] Video uploaded and set to public/unlisted
