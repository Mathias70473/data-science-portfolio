/* ============================================================
   CREDIT SCORE SIMULATOR
   An illustrative in-page demo of a logistic-regression
   scorecard (matching the "Credit Risk Scorecard" project).

   The three inputs map to the scorecard's documented 504–715
   range. Weights are PLACEHOLDER — swap them for the real
   model's coefficients when available.
   ============================================================ */

const INPUTS = [
  {
    id: "income",
    label: "Monthly income (MXN)",
    min: 5000,
    max: 80000,
    step: 1000,
    value: 25000,
    weight: 0.0008,
  },
  {
    id: "utilization",
    label: "Credit utilization (%)",
    min: 0,
    max: 100,
    step: 1,
    value: 35,
    weight: -0.35,
  },
  {
    id: "payment",
    label: "On-time payment history (0–100)",
    min: 0,
    max: 100,
    step: 1,
    value: 90,
    weight: 0.55,
  },
];

const BASE_SCORE = 504;
const MAX_SCORE = 715;

function scoreFrom(values) {
  let raw = BASE_SCORE;
  for (const input of INPUTS) {
    const v = Number(values[input.id]);
    const normalized = (v - input.min) / (input.max - input.min);
    raw += normalized * input.weight * (MAX_SCORE - BASE_SCORE) * 1.5;
  }
  return Math.round(Math.min(MAX_SCORE, Math.max(BASE_SCORE, raw)));
}

function formatValue(el) {
  const value = Number(el.value);
  if (el.id === "income") {
    return `${new Intl.NumberFormat("en-US").format(value)} MXN`;
  }
  return `${value}%`;
}

export function initDemo() {
  const controls = document.getElementById("demo-controls");
  const scoreEl = document.getElementById("demo-score");
  const bandEl = document.getElementById("demo-band");

  if (!controls || !scoreEl || !bandEl) return;

  function update() {
    const values = {};
    controls.querySelectorAll("input[type=range]").forEach((el) => {
      const value = el.value;
      const displayValue = formatValue(el);
      values[el.dataset.input] = value;
      el.setAttribute("aria-valuetext", displayValue);
      const out = el.parentElement.querySelector("output");
      if (out) out.value = displayValue;
    });

    const score = scoreFrom(values);

    scoreEl.textContent = score;
    if (bandEl) bandEl.textContent = "Illustrative only";
  }

  controls.addEventListener("input", update);
  update();
}
