/* ============================================================
   RENDER LOGIC
   Pure functions: take project objects, return HTML strings.
   All interpolated user strings pass through escapeHtml().
   ============================================================ */

export function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function metricHtml(metric) {
  if (metric.kind === "num") {
    return `
      <span class="metric metric--num">
        <span class="metric__value">${escapeHtml(metric.value)}</span>
        <span class="metric__label">${escapeHtml(metric.label)}</span>
      </span>`;
  }
  return `
    <span class="metric metric--text">${escapeHtml(metric.value)}</span>`;
}

function metricsHtml(metrics) {
  return metrics.map(metricHtml).join("");
}

function stackHtml(stack) {
  return stack
    .map((s) => `<span class="stack">${escapeHtml(s)}</span>`)
    .join("");
}

function tagHtml(tags) {
  return tags.map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("");
}

function dateText(p) {
  return p.month ? `${p.month} ${p.year}` : String(p.year);
}

function evidenceHtml(p) {
  const items = [
    { label: "Research question", value: p.question },
    { label: "Method", value: p.diagnostics?.join(" · ") },
    { label: "Finding", value: p.result },
  ];

  const present = items.filter((item) => item.value);
  if (!present.length) return "";

  return `
    <dl class="card__evidence" aria-label="Project evidence">
      ${present
        .map(
          (item) => `
            <div class="card__evidence-item">
              <dt>${escapeHtml(item.label)}</dt>
              <dd>${escapeHtml(item.value)}</dd>
            </div>`
        )
        .join("")}
    </dl>`;
}

function linksHtml(p) {
  const links = [
    `<a class="card__link" href="${escapeHtml(p.githubUrl)}" target="_blank" rel="noopener">View project on GitHub →</a>`,
  ];
  if (p.liveUrl) {
    links.unshift(
      `<a class="card__link card__link--live" href="${escapeHtml(p.liveUrl)}" target="_blank" rel="noopener">Live demo →</a>`
    );
  }
  if (p.artifactsUrl) {
    links.push(
      `<a class="card__link" href="${escapeHtml(p.artifactsUrl)}" target="_blank" rel="noopener">Report ↓</a>`
    );
  }
  return links.join("");
}

export function buildCard(p, { featured = false } = {}) {
  const cls = featured ? "card card--featured" : "card";
  const eyebrow = featured
    ? `<p class="eyebrow eyebrow--light">Featured · ${escapeHtml(
        p.category === "policy" ? "Policy evaluation" : "Credit risk"
      )}</p>`
    : `<p class="eyebrow eyebrow--card">${escapeHtml(
        p.category === "policy" ? "Policy evaluation" : "Credit risk"
      )} · ${escapeHtml(dateText(p))}</p>`;

  const meta = featured
    ? `<p class="card__meta">${escapeHtml(p.role)} · ${escapeHtml(
        dateText(p)
      )}</p>`
    : "";

  return `
    <article class="${cls}" data-category="${escapeHtml(p.category)}" id="${escapeHtml(
    p.id
  )}">
      ${eyebrow}
      <h3 class="card__title">${escapeHtml(p.title)}</h3>
      <p class="card__tagline">${escapeHtml(p.tagline)}</p>
      ${meta}
      <div class="card__metrics">${metricsHtml(p.metrics)}</div>
      <p class="card__desc">${escapeHtml(p.description)}</p>
      ${evidenceHtml(p)}
      ${featured && p.limitations ? `
        <div class="card__limitations">
          <p class="card__limitations-label">Caveats</p>
          <ul>${p.limitations.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>
        </div>` : ""}
      <div class="card__stack">${stackHtml(p.stack)}</div>
      <div class="card__footer">
        <div class="card__tags">${tagHtml(p.tags)}</div>
        <div class="card__links">${linksHtml(p)}</div>
      </div>
    </article>
  `;
}
