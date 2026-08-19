/* ============================================================
   PROJECTS DATA
   ------------------------------------------------------------
   To add a project: add ONE object to the array below. The
   render layer (render.js) builds cards from this schema, so
   nothing else needs to change.

   Fields:
     id           (string)   — unique slug (used for anchors)
     title        (string)   — project name
     tagline      (string)   — one-line method summary
     year         (number)   — when the work was done
     role         (string)   — your role on the project
     category     (string)   — "credit" | "policy" (drives accent)
     stack        (array)    — tools actually used in the code
     tags         (array)    — method / domain labels
     metrics      (array)    — [{ label, value, kind }] where
                               kind = "num" (big headline) or
                                     "text" (small chip)
      description  (string)   — 2–4 sentences, plain prose
      question     (string)   — concise problem framing
      result       (string)   — strongest supported result or diagnostic
      diagnostics  (array)    — method and validation details
      limitations  (array)    — material caveats to disclose
      githubUrl    (string)   — link to the project directory
      liveUrl      (string)   — live demo / dashboard (or null)
      artifactsUrl (string)   — hosted report / PDF (or null)
      featured     (boolean)  — true = large top card (one at a time)
   ============================================================ */

export const projects = [
  {
    id: "financial-inclusion-causal",
    title: "Banco del Bienestar & Private Financial Infrastructure",
    tagline: "A dynamic causal study of a 2022 public-bank rollout in municipalities without a development-bank branch at baseline",
    year: 2026,
    month: "June",
    role: "Sole author · causal evaluation",
    category: "policy",
    stack: ["Python", "pandas", "csdid", "CNBV quarterly data"],
    tags: ["Causal inference", "Staggered DiD", "Policy evaluation"],
    metrics: [
      { label: "Panel coverage", value: "16 quarters", kind: "num" },
      { label: "Observations", value: "39,459", kind: "num" },
      { label: "ATT estimate", value: "≈ −0.006", kind: "text" },
      { label: "Standard error", value: "≈ 0.008", kind: "text" },
    ],
    description:
      "This project asks whether the 2022 Banco del Bienestar branch expansion affected private financial infrastructure in municipalities without a development-bank branch at baseline. The design uses a municipal-level CNBV panel and a staggered Callaway–Sant'Anna difference-in-differences estimator, with the outcome defined as log(1 + corresponsales). No statistically significant treatment effect was detected; the result is stable across three estimation specifications and two control-group definitions.",
    question:
      "Did the 2022 branch expansion change private financial infrastructure in Mexican municipalities without a development-bank branch at baseline?",
    result:
      "No statistically significant effect was detected. ATT is approximately −0.006 (SE ≈ 0.008) across the reported specifications.",
    diagnostics: [
      "Callaway–Sant'Anna estimator for staggered adoption",
      "Event-study check of the parallel-trends assumption",
      "Three specifications · two control-group definitions",
    ],
    limitations: [
      "Post-treatment window is short, with limited follow-up for early cohorts",
      "Banco del Bienestar is proxied by aggregate development-bank branches",
      "Independent private-sector expansion cannot be fully separated from a response to treatment",
    ],
    githubUrl:
      "https://github.com/Mathias70473/data-science-portfolio/tree/main/03-banco-bienestar-financial-inclusion",
    liveUrl: null,
    artifactsUrl: null,
    featured: true,
  },
  {
    id: "payment-default-prediction",
    title: "Payment Default Prediction",
    tagline: "A transparent threshold policy for asymmetric lending costs",
    year: 2026,
    month: "January",
    role: "Modeling · threshold analysis",
    category: "credit",
    stack: ["Python", "statsmodels", "scikit-learn", "scipy", "UCI credit-card data"],
    tags: ["Classification", "Cost-sensitive learning", "Feature engineering"],
    metrics: [
      { label: "Recall", value: "94.8%", kind: "num" },
      { label: "Cost reduction", value: "−77.9%", kind: "num" },
      { label: "Dataset", value: "30,000 accounts", kind: "text" },
    ],
    description:
      "A logistic-regression model predicts next-month credit-card default from payment history and engineered features. Instead of using a default 0.5 cutoff, the classification threshold is selected from a 25:1 cost ratio between missed defaults and false alarms, then evaluated on a stratified holdout.",
    question: "How should a lender set a decision threshold when the cost of a missed default is asymmetric?",
    result:
      "At a 0.05 threshold, the model reaches 94.8% recall and reduces estimated total cost by 77.9% on the 6,000-row holdout.",
    diagnostics: [
      "80/20 stratified holdout · 30,000 accounts",
      "ROC-AUC 0.7154",
      "25:1 cost ratio between missed defaults and false alarms",
    ],
    limitations: [
      "Single 2005 Taiwan credit-card dataset",
      "Precision falls to 22.4% at the selected threshold",
      "Single holdout rather than time-based validation",
    ],
    githubUrl:
      "https://github.com/Mathias70473/data-science-portfolio/tree/main/01-payment-default-prediction",
    liveUrl: null,
    artifactsUrl: null,
    featured: false,
  },
  {
    id: "credit-risk-scorecard",
    title: "Credit Risk Scorecard",
    tagline: "WoE / IV feature selection into a PDO-scaled logistic scorecard",
    year: 2026,
    month: "April",
    role: "Modeling · feature engineering",
    category: "credit",
    stack: ["Python", "pandas", "scikit-learn", "statsmodels", "WoE / IV"],
    tags: ["Scorecard", "Logistic regression", "Credit risk"],
    metrics: [
      { label: "ROC-AUC", value: "0.8115", kind: "num" },
      { label: "Gini", value: "0.62", kind: "num" },
      { label: "KS", value: "0.51", kind: "num" },
      { label: "Mean score", value: "613", kind: "num" },
      { label: "Score range", value: "504–715", kind: "text" },
    ],
    description:
      "This project builds an interpretable credit scorecard on the German Credit Dataset. The pipeline applies Weight-of-Evidence binning and Information Value screening, fits a logistic model on WoE-encoded features, and scales the result to a score using the PDO method so the output can be read by a risk team.",
    question: "How can model output become an interpretable applicant score rather than a black-box probability?",
    result:
      "The WoE/IV pipeline retains 8 of 20 variables and produces scores from 504 to 715, with a mean score of 613.",
    diagnostics: [
      "ROC-AUC 0.8115 · Gini 0.62 · KS 0.51",
      "Checking-account status is the strongest selected signal (IV 0.6660)",
      "Borderline applicants are flagged for manual review",
    ],
    limitations: [
      "Sample size is limited to 1,000 observations",
      "Dataset dates to 1990s Germany",
      "No out-of-time stability test is reported",
    ],
    githubUrl:
      "https://github.com/Mathias70473/data-science-portfolio/tree/main/02-credit-risk-scorecard",
    liveUrl: null,
    artifactsUrl: null,
    featured: false,
  },
];
