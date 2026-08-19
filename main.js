/* ============================================================
   APP ENTRY
   Imports data + render, mounts cards, wires filter/sort,
   and initializes the demo.
   ============================================================ */

import { projects } from "./data/projects.js";
import { buildCard } from "./render.js";
import { initDemo } from "./demo.js";

let activeFilter = "all";

function sortByYearDesc(list) {
  return [...list].sort((a, b) => b.year - a.year);
}

function render() {
  const featuredMount = document.getElementById("featured-project");
  const gridMount = document.getElementById("project-grid");
  if (!featuredMount || !gridMount) return;

  const filtered =
    activeFilter === "all"
      ? projects
      : projects.filter((p) => p.category === activeFilter);

  const featured = filtered.filter((p) => p.featured);
  const rest = sortByYearDesc(filtered.filter((p) => !p.featured));

  featuredMount.innerHTML = featured.map((p) => buildCard(p, { featured: true })).join("");
  gridMount.innerHTML = rest.map((p) => buildCard(p)).join("");
}

function initFilters() {
  const filterBar = document.getElementById("filter-bar");
  if (!filterBar) return;

  filterBar.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-filter]");
    if (!btn) return;

    activeFilter = btn.dataset.filter;
    filterBar
      .querySelectorAll("button")
      .forEach((b) => {
        const selected = b === btn;
        b.classList.toggle("is-active", selected);
        b.setAttribute("aria-pressed", String(selected));
      });
    render();
  });
}

function init() {
  render();
  initFilters();
  initDemo();
}

document.addEventListener("DOMContentLoaded", init);
