#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import html
import shutil
import socket
import subprocess
import textwrap
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from weatlib import (
    EmbeddingStore,
    build_sqlite_index,
    download_file,
    evaluate_proposal,
    extract_text_member,
    load_json,
    normalize_token,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
SPECS_DIR = PROJECT_ROOT / "specs"
RUNS_DIR = PROJECT_ROOT / "runs"
STATE_DIR = PROJECT_ROOT / "state"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
DASHBOARD_HTML = DASHBOARD_DIR / "index.html"

DATA_CONFIG = CONFIG_DIR / "data_source.json"
RESEARCH_CONFIG = CONFIG_DIR / "research_config.json"
PROMPT_PATH = PROMPTS_DIR / "proposer_system.md"
SOURCE_MANIFEST = STATE_DIR / "source_manifest.json"
LEADERBOARD_PATH = STATE_DIR / "leaderboard.csv"
HISTORY_PATH = STATE_DIR / "history.csv"
BEST_FINDING_PATH = STATE_DIR / "best_finding.json"

DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 1200
DEFAULT_OLLAMA_THINK = "auto"

MODEL_ALIASES = {
    "qwen3.5:37b": "qwen3.5:35b",
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def ensure_dirs() -> None:
    for path in (RUNS_DIR, STATE_DIR, DASHBOARD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def canonical_model_name(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def model_family(model: str) -> str:
    canonical = canonical_model_name(model)
    if canonical.startswith("gpt-oss"):
        return "gpt-oss"
    if canonical.startswith("gemma4"):
        return "gemma4"
    if canonical.startswith("qwen3.5"):
        return "qwen3.5"
    return "generic"


def source_manifest() -> dict[str, Any]:
    if not SOURCE_MANIFEST.exists():
        raise SystemExit("Active source is missing. Run `prepare` first.")
    return load_json(SOURCE_MANIFEST)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def history_summary(limit: int = 12) -> str:
    rows = load_csv_rows(HISTORY_PATH)
    if not rows:
        return "No prior findings."
    tail = rows[-limit:]
    lines = []
    for row in tail:
        lines.append(
            f"- {row['proposal_id']} | {row['bias_name']} | accepted={row['accepted']} | "
            f"abs_d={row['abs_effect_size']} | p={row['directional_p_value']} | {row['rationale_code']}"
        )
    return "\n".join(lines)


def build_prepare_manifest(text_path: Path, index_path: Path, source_name: str, source_note: str) -> dict[str, Any]:
    meta = build_sqlite_index(text_path, index_path)
    manifest = {
        "embedding_name": source_name,
        "text_path": str(text_path),
        "index_path": str(index_path),
        "row_count": meta["row_count"],
        "dimension": meta["dimension"],
        "source_note": source_note,
        "prepared_at": now_stamp(),
    }
    write_json(SOURCE_MANIFEST, manifest)
    return manifest


def safe_float(value: Any) -> float | None:
    if value in (None, "", "NA"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_dashboard_payload() -> dict[str, Any]:
    ensure_dirs()
    source = load_json(SOURCE_MANIFEST) if SOURCE_MANIFEST.exists() else None
    best = load_json(BEST_FINDING_PATH) if BEST_FINDING_PATH.exists() else None

    run_dirs = sorted([path for path in RUNS_DIR.iterdir() if path.is_dir()], reverse=True)
    run_summaries = []
    findings = []

    for run_dir in run_dirs:
        results_path = run_dir / "results.json"
        if not results_path.exists():
            continue
        payload = load_json(results_path)
        results = payload.get("results", [])
        backend = run_dir.name.split("-")[-1]
        accepted_count = sum(1 for item in results if item.get("accepted"))
        run_summaries.append(
            {
                "run_id": run_dir.name,
                "backend": backend,
                "result_count": len(results),
                "accepted_count": accepted_count,
                "path": str(run_dir),
            }
        )
        for result in results:
            metrics = result.get("metrics", {})
            findings.append(
                {
                    "run_id": run_dir.name,
                    "backend": backend,
                    "proposal_id": result.get("proposal_id", ""),
                    "discipline": result.get("discipline", ""),
                    "bias_name": result.get("bias_name", ""),
                    "hypothesis": result.get("hypothesis", ""),
                    "signature": result.get("signature", ""),
                    "accepted": bool(result.get("accepted")),
                    "rationale_code": result.get("rationale_code", ""),
                    "error": result.get("error", ""),
                    "effect_size": safe_float(metrics.get("effect_size")),
                    "abs_effect_size": safe_float(metrics.get("abs_effect_size")),
                    "directional_p_value": safe_float(metrics.get("directional_p_value")),
                    "supported_orientation": metrics.get("supported_orientation", ""),
                    "canonical_sets": result.get("canonical_sets", {}),
                    "dropped_terms": result.get("dropped_terms", {}),
                    "run_path": str(run_dir),
                }
            )

    disciplines = sorted({item["discipline"] for item in findings if item.get("discipline")})
    return {
        "generated_at": now_stamp(),
        "source": source,
        "best_finding": best,
        "run_summaries": run_summaries,
        "findings": findings,
        "disciplines": disciplines,
    }


def dashboard_html(payload: dict[str, Any]) -> str:
    data_json = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")
    best = payload.get("best_finding")
    best_name = best.get("bias_name", "None yet") if best else "None yet"
    best_d = safe_float(best.get("metrics", {}).get("abs_effect_size")) if best else None
    best_d_text = "n/a" if best_d is None else f"{best_d:.3f}"
    source = payload.get("source") or {}
    source_name = source.get("embedding_name", "No active source")
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AutoWEAT Dashboard</title>
  <style>
    :root {{
      --bg: #f5f0e8;
      --panel: #fffaf3;
      --ink: #1f1a17;
      --muted: #6a5f57;
      --accent: #0f6a5b;
      --accent-soft: #d8efe8;
      --line: #e3d8ca;
      --bad: #8b2e2e;
      --good: #0d6b3f;
      --chip: #efe2d1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fdf7ef 0, #f5f0e8 45%, #ece2d1 100%);
    }}
    .page {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }}
    h1, h2, h3 {{ margin: 0; }}
    p {{ margin: 0; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.4fr 1fr;
      gap: 18px;
      margin-bottom: 20px;
    }}
    .panel {{
      background: rgba(255,250,243,0.92);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 18px;
      box-shadow: 0 10px 35px rgba(61, 41, 20, 0.08);
    }}
    .hero-title {{
      font-size: 34px;
      line-height: 1.05;
      margin-bottom: 10px;
      letter-spacing: -0.02em;
    }}
    .hero-sub {{
      color: var(--muted);
      line-height: 1.45;
      max-width: 60ch;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .stat {{
      padding: 12px;
      border-radius: 14px;
      background: var(--panel);
      border: 1px solid var(--line);
    }}
    .stat-label {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    .stat-value {{
      font-size: 22px;
      font-weight: 700;
    }}
    .filters {{
      display: grid;
      grid-template-columns: repeat(5, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 16px;
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
    }}
    select, input {{
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      border-radius: 12px;
      padding: 10px 12px;
      color: var(--ink);
      font: inherit;
    }}
    .toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .count {{
      color: var(--muted);
      font-size: 14px;
    }}
    .toolbar-controls {{
      display: flex;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .inline-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
    }}
    .inline-toggle input {{
      width: auto;
      margin: 0;
      padding: 0;
    }}
    .cards {{
      display: grid;
      gap: 14px;
    }}
    .card {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.88);
      border-radius: 18px;
      padding: 16px;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      align-items: start;
      gap: 16px;
      margin-bottom: 10px;
    }}
    .title {{
      font-size: 22px;
      line-height: 1.1;
      margin-bottom: 4px;
    }}
    .meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      background: var(--chip);
      font-size: 12px;
    }}
    .chip.accepted {{
      background: #daf2df;
      color: var(--good);
    }}
    .chip.rejected {{
      background: #f4dfdf;
      color: var(--bad);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 12px 0;
    }}
    .metric {{
      padding: 10px 12px;
      border: 1px solid var(--line);
      border-radius: 12px;
      background: var(--panel);
    }}
    .metric span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 4px;
    }}
    .metric strong {{
      font-size: 18px;
    }}
    .hypothesis {{
      color: var(--ink);
      line-height: 1.5;
      margin-bottom: 12px;
    }}
    .sets {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .set-box {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: #fffdf9;
    }}
    .set-box h4 {{
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .terms {{
      line-height: 1.55;
      font-size: 14px;
    }}
    .footer {{
      margin-top: 18px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 980px) {{
      .hero, .filters, .metrics, .sets {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="panel">
        <h1 class="hero-title">AutoWEAT Dashboard</h1>
        <p class="hero-sub">A live local dashboard for socially meaningful WEAT findings. Every round rewrites this page from the run artifacts, so you can sort by strongest effect, newest run, significance, discipline, or inspect the exact canonical seed words used in each accepted and rejected test.</p>
        <div class="stats">
          <div class="stat">
            <span class="stat-label">Active Source</span>
            <div class="stat-value">__SOURCE_NAME__</div>
          </div>
          <div class="stat">
            <span class="stat-label">Current Best Finding</span>
            <div class="stat-value">__BEST_NAME__</div>
          </div>
          <div class="stat">
            <span class="stat-label">Best |d|</span>
            <div class="stat-value">__BEST_D__</div>
          </div>
        </div>
      </div>
      <div class="panel">
        <h2 style="font-size:20px; margin-bottom:10px;">How To Use</h2>
        <p class="hero-sub" style="max-width:none;">Filter by run, discipline, acceptance status, or search any title, hypothesis, or seed word. Sort by latest run, largest absolute effect size, smallest p-value, or alphabetical bias name. Every card shows the exact final `X/Y/A/B` seed words used in the evaluated WEAT.</p>
        <div class="footer" style="margin-top:14px;">
          Generated at: <span id="generated-at-label">__GENERATED_AT__</span><br>
          Source manifest: __SOURCE_MANIFEST_PATH__
        </div>
      </div>
    </section>

    <section class="panel">
      <div class="filters">
        <label>Run
          <select id="run-filter"></select>
        </label>
        <label>Discipline
          <select id="discipline-filter"></select>
        </label>
        <label>Acceptance
          <select id="accepted-filter">
            <option value="all">All</option>
            <option value="accepted">Accepted only</option>
            <option value="rejected">Rejected only</option>
          </select>
        </label>
        <label>Sort By
          <select id="sort-filter">
            <option value="latest">Latest run</option>
            <option value="abs_effect_desc">Highest |d|</option>
            <option value="p_asc">Lowest p-value</option>
            <option value="name_asc">Bias name</option>
          </select>
        </label>
        <label>Search
          <input id="search-filter" type="search" placeholder="Search bias, rationale, or seed words">
        </label>
      </div>
      <div class="toolbar">
        <div class="count" id="result-count"></div>
        <div class="toolbar-controls">
          <label class="inline-toggle">
            <input id="refresh-toggle" type="checkbox" checked>
            Auto-refresh every 15s
          </label>
        </div>
      </div>
      <div class="cards" id="cards"></div>
    </section>
  </div>

  <script id="dashboard-data" type="application/json">__DATA_JSON__</script>
  <script>
    const payload = JSON.parse(document.getElementById('dashboard-data').textContent);
    const findings = payload.findings || [];
    const runFilter = document.getElementById('run-filter');
    const disciplineFilter = document.getElementById('discipline-filter');
    const acceptedFilter = document.getElementById('accepted-filter');
    const sortFilter = document.getElementById('sort-filter');
    const searchFilter = document.getElementById('search-filter');
    const resultCount = document.getElementById('result-count');
    const cards = document.getElementById('cards');
    const refreshToggle = document.getElementById('refresh-toggle');
    const generatedAtLabel = document.getElementById('generated-at-label');
    const storagePrefix = 'auto-weat-dashboard:';
    let refreshTimer = null;

    function safeStorageGet(key) {{
      try {{
        return window.localStorage.getItem(key);
      }} catch (_error) {{
        return null;
      }}
    }}

    function safeStorageSet(key, value) {{
      try {{
        window.localStorage.setItem(key, value);
      }} catch (_error) {{
        return null;
      }}
      return value;
    }}

    function addOptions(select, values, allLabel) {{
      select.innerHTML = '';
      const all = document.createElement('option');
      all.value = 'all';
      all.textContent = allLabel;
      select.appendChild(all);
      values.forEach(value => {{
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      }});
    }}

    addOptions(runFilter, (payload.run_summaries || []).map(item => item.run_id), 'All runs');
    addOptions(disciplineFilter, payload.disciplines || [], 'All disciplines');

    function cardHTML(item) {{
      const acceptedClass = item.accepted ? 'accepted' : 'rejected';
      const acceptedText = item.accepted ? 'Accepted' : 'Rejected';
      const metrics = [
        ['Effect size', item.effect_size],
        ['|d|', item.abs_effect_size],
        ['Directional p', item.directional_p_value],
        ['Orientation', item.supported_orientation || 'n/a'],
      ];
      const sets = item.canonical_sets || {{}};
      const setEntries = [
        ['X targets', sets.x_terms || []],
        ['Y targets', sets.y_terms || []],
        ['A attributes', sets.a_terms || []],
        ['B attributes', sets.b_terms || []],
      ];
      const dropped = item.dropped_terms || {{}};
      const droppedFlat = [
        ...(dropped.x_terms || []).map(value => `X:${{value}}`),
        ...(dropped.y_terms || []).map(value => `Y:${{value}}`),
        ...(dropped.a_terms || []).map(value => `A:${{value}}`),
        ...(dropped.b_terms || []).map(value => `B:${{value}}`),
      ];
      return `
        <article class="card">
          <div class="card-top">
            <div>
              <div class="title">${{escapeHtml(item.bias_name || '(untitled)')}}</div>
              <div class="meta">
                <span class="chip">${{escapeHtml(item.discipline || 'unknown discipline')}}</span>
                <span class="chip">${{escapeHtml(item.run_id)}}</span>
                <span class="chip">${{escapeHtml(item.proposal_id || '')}}</span>
                <span class="chip ${acceptedClass}">${{acceptedText}}</span>
                <span class="chip">${{escapeHtml(item.rationale_code || '')}}</span>
              </div>
            </div>
          </div>
          <p class="hypothesis">${{escapeHtml(item.hypothesis || '')}}</p>
          <div class="metrics">
            ${metrics.map(([label, value]) => `
              <div class="metric">
                <span>${{label}}</span>
                <strong>${{formatValue(value)}}</strong>
              </div>
            `).join('')}
          </div>
          <div class="sets">
            ${setEntries.map(([label, values]) => `
              <div class="set-box">
                <h4>${{label}}</h4>
                <div class="terms">${{escapeHtml((values || []).join(', '))}}</div>
              </div>
            `).join('')}
          </div>
          ${droppedFlat.length ? `<div class="footer">Dropped terms: ${{escapeHtml(droppedFlat.join(', '))}}</div>` : ''}
          ${item.error ? `<div class="footer">Error: ${{escapeHtml(item.error)}}</div>` : ''}
        </article>
      `;
    }}

    function escapeHtml(value) {{
      return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function formatValue(value) {{
      if (value === null || value === undefined || value === '') return 'n/a';
      if (typeof value === 'number') return value.toFixed(4);
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed.toFixed(4) : String(value);
    }}

    function searchableText(item) {{
      const sets = item.canonical_sets || {{}};
      return [
        item.bias_name,
        item.discipline,
        item.hypothesis,
        item.rationale_code,
        ...(sets.x_terms || []),
        ...(sets.y_terms || []),
        ...(sets.a_terms || []),
        ...(sets.b_terms || []),
      ].join(' ').toLowerCase();
    }}

    function filteredFindings() {{
      const runValue = runFilter.value;
      const disciplineValue = disciplineFilter.value;
      const acceptedValue = acceptedFilter.value;
      const query = searchFilter.value.trim().toLowerCase();

      let rows = findings.filter(item => {{
        if (runValue !== 'all' && item.run_id !== runValue) return false;
        if (disciplineValue !== 'all' && item.discipline !== disciplineValue) return false;
        if (acceptedValue === 'accepted' && !item.accepted) return false;
        if (acceptedValue === 'rejected' && item.accepted) return false;
        if (query && !searchableText(item).includes(query)) return false;
        return true;
      }});

      const sortValue = sortFilter.value;
      rows = rows.slice().sort((a, b) => {{
        if (sortValue === 'latest') return b.run_id.localeCompare(a.run_id) || a.proposal_id.localeCompare(b.proposal_id);
        if (sortValue === 'abs_effect_desc') return (b.abs_effect_size ?? -Infinity) - (a.abs_effect_size ?? -Infinity);
        if (sortValue === 'p_asc') return (a.directional_p_value ?? Infinity) - (b.directional_p_value ?? Infinity);
        if (sortValue === 'name_asc') return (a.bias_name || '').localeCompare(b.bias_name || '');
        return 0;
      }});
      return rows;
    }}

    function persistUiState() {{
      safeStorageSet(storagePrefix + 'run-filter', runFilter.value);
      safeStorageSet(storagePrefix + 'discipline-filter', disciplineFilter.value);
      safeStorageSet(storagePrefix + 'accepted-filter', acceptedFilter.value);
      safeStorageSet(storagePrefix + 'sort-filter', sortFilter.value);
      safeStorageSet(storagePrefix + 'search-filter', searchFilter.value);
      safeStorageSet(storagePrefix + 'refresh-toggle', refreshToggle.checked ? 'true' : 'false');
    }}

    function restoreUiState() {{
      const savedRun = safeStorageGet(storagePrefix + 'run-filter');
      const savedDiscipline = safeStorageGet(storagePrefix + 'discipline-filter');
      const savedAccepted = safeStorageGet(storagePrefix + 'accepted-filter');
      const savedSort = safeStorageGet(storagePrefix + 'sort-filter');
      const savedSearch = safeStorageGet(storagePrefix + 'search-filter');
      const savedRefresh = safeStorageGet(storagePrefix + 'refresh-toggle');

      if (savedRun && [...runFilter.options].some(option => option.value === savedRun)) runFilter.value = savedRun;
      if (savedDiscipline && [...disciplineFilter.options].some(option => option.value === savedDiscipline)) disciplineFilter.value = savedDiscipline;
      if (savedAccepted && [...acceptedFilter.options].some(option => option.value === savedAccepted)) acceptedFilter.value = savedAccepted;
      if (savedSort && [...sortFilter.options].some(option => option.value === savedSort)) sortFilter.value = savedSort;
      if (savedSearch) searchFilter.value = savedSearch;
      refreshToggle.checked = savedRefresh === null ? true : savedRefresh === 'true';
    }}

    function syncAutoRefresh() {{
      if (refreshTimer) {{
        window.clearTimeout(refreshTimer);
        refreshTimer = null;
      }}
      if (!refreshToggle.checked) return;
      refreshTimer = window.setTimeout(() => {{
        window.location.href = window.location.pathname + window.location.search + '#refresh-' + Date.now();
        window.location.reload();
      }}, 15000);
    }}

    function render() {{
      const rows = filteredFindings();
      resultCount.textContent = `${rows.length} finding${rows.length === 1 ? '' : 's'} shown`;
      cards.innerHTML = rows.map(cardHTML).join('');
      if (!rows.length) {{
        cards.innerHTML = '<article class="card"><div class="title">No findings match the current filters.</div></article>';
      }}
      if (generatedAtLabel) generatedAtLabel.textContent = payload.generated_at || 'unknown';
      persistUiState();
      syncAutoRefresh();
    }}

    [runFilter, disciplineFilter, acceptedFilter, sortFilter, searchFilter].forEach(node => {{
      node.addEventListener('input', render);
      node.addEventListener('change', render);
    }});
    refreshToggle.addEventListener('change', render);

    restoreUiState();
    render();
  </script>
</body>
</html>
"""
    return (
        template.replace("__SOURCE_NAME__", html.escape(source_name))
        .replace("__BEST_NAME__", html.escape(best_name))
        .replace("__BEST_D__", html.escape(best_d_text))
        .replace("__GENERATED_AT__", html.escape(payload.get("generated_at", "")))
        .replace("__SOURCE_MANIFEST_PATH__", html.escape(str(SOURCE_MANIFEST)))
        .replace("__DATA_JSON__", data_json)
        .replace("{{", "{")
        .replace("}}", "}")
    )


def rebuild_dashboard() -> None:
    ensure_dirs()
    payload = collect_dashboard_payload()
    DASHBOARD_HTML.write_text(dashboard_html(payload), encoding="utf-8")


def command_reset(_args: argparse.Namespace) -> None:
    ensure_dirs()
    deleted_run_dirs = 0
    for path in RUNS_DIR.iterdir():
        if path.is_dir():
            shutil.rmtree(path)
            deleted_run_dirs += 1
        else:
            path.unlink()

    deleted_state_files = []
    for path in (LEADERBOARD_PATH, HISTORY_PATH, BEST_FINDING_PATH):
        if path.exists():
            path.unlink()
            deleted_state_files.append(path.name)

    rebuild_dashboard()
    print(
        json.dumps(
            {
                "reset_at": now_stamp(),
                "deleted_run_dirs": deleted_run_dirs,
                "deleted_state_files": deleted_state_files,
                "kept_source_manifest": SOURCE_MANIFEST.exists(),
                "dashboard_html": str(DASHBOARD_HTML),
            },
            indent=2,
        )
    )


def command_prepare(args: argparse.Namespace) -> None:
    ensure_dirs()
    data_config = load_json(DATA_CONFIG)

    if args.txt_path:
        text_path = args.txt_path
        index_path = args.index_path or (PROJECT_ROOT / "data" / "index" / f"{text_path.stem}.sqlite")
        manifest = build_prepare_manifest(
            text_path=text_path,
            index_path=index_path,
            source_name=text_path.stem,
            source_note="Prepared from explicit --txt-path override.",
        )
        rebuild_dashboard()
        print(json.dumps(manifest, indent=2))
        return

    zip_path = resolve_project_path(data_config["zip_path"])
    text_path = resolve_project_path(data_config["text_path"])
    index_path = resolve_project_path(data_config["index_path"])

    if args.download and not zip_path.exists():
        download_file(data_config["download_url"], zip_path)

    if not text_path.exists():
        if not zip_path.exists():
            raise SystemExit("Missing zip and extracted text. Re-run with `prepare --download`.")
        extract_text_member(zip_path, text_path)

    manifest = build_prepare_manifest(
        text_path=text_path,
        index_path=index_path,
        source_name=data_config["embedding_name"],
        source_note=data_config["note"],
    )
    rebuild_dashboard()
    print(json.dumps(manifest, indent=2))


def command_status(_args: argparse.Namespace) -> None:
    rebuild_dashboard()
    manifest = source_manifest()
    best = load_json(BEST_FINDING_PATH) if BEST_FINDING_PATH.exists() else None
    payload = {"source": manifest, "best_finding": best, "dashboard_html": str(DASHBOARD_HTML)}
    print(json.dumps(payload, indent=2))


def resolve_ollama_think(model: str, think: str | None) -> str | None:
    family = model_family(model)
    if think in (None, "", "auto"):
        if family == "gpt-oss":
            return "high"
        if family in ("gemma4", "qwen3.5"):
            return "true"
        return None
    if family == "gpt-oss" and think == "true":
        return "high"
    if family == "gpt-oss" and think == "false":
        return "low"
    if family in ("gemma4", "qwen3.5") and think in ("low", "medium", "high"):
        return "true"
    if think in ("low", "medium", "high"):
        return think
    if think == "true":
        return "true"
    if think == "false":
        return "false"
    raise ValueError(f"Unsupported --ollama-think value: {think}")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def extract_json_payload(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    if not cleaned:
        raise ValueError("Model returned an empty response.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start_positions = [idx for idx, char in enumerate(cleaned) if char in "[{"]
    end_positions = [idx for idx, char in enumerate(cleaned) if char in "]}"]
    for start in start_positions:
        for end in reversed(end_positions):
            if end <= start:
                continue
            candidate = cleaned[start : end + 1].strip()
            if not candidate:
                continue
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
    preview = cleaned[:500].replace("\n", "\\n")
    raise ValueError(f"Could not parse model response as JSON. Preview: {preview}")


def ollama_generate(
    prompt: str,
    model: str,
    host: str,
    timeout_seconds: int,
    think: str | None = None,
    json_mode: bool = True,
) -> dict[str, Any]:
    canonical_model = canonical_model_name(model)
    family = model_family(canonical_model)

    options: dict[str, Any] = {"temperature": 0}
    if family == "gpt-oss":
        options = {"temperature": 1.0, "top_p": 1.0}
    elif family == "gemma4":
        options = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
    elif family == "qwen3.5":
        options = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}

    effective_prompt = prompt
    if family == "gemma4" and think == "true":
        effective_prompt = "<|think|>\n" + prompt

    payload = {
        "model": canonical_model,
        "prompt": effective_prompt,
        "stream": False,
        "options": options,
    }
    if json_mode:
        payload["format"] = "json"
    if family == "gemma4":
        pass
    elif think in ("true", "false"):
        payload["think"] = (think == "true")
    elif think is not None:
        payload["think"] = think
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            return json.load(response)
    except (TimeoutError, socket.timeout) as exc:
        raise RuntimeError(
            f"Ollama request timed out after {timeout_seconds} seconds for model {model}. "
            "Try a larger --ollama-timeout value."
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc


def social_science_rules_backend(n_proposals: int) -> list[dict[str, Any]]:
    library = [
        {
            "discipline": "organizational psychology",
            "bias_name": "Gender-career association",
            "hypothesis": "Male terms may align more with career and status than female terms do.",
            "x_terms": ["man", "male", "brother", "father", "son", "uncle", "husband", "he"],
            "y_terms": ["woman", "female", "sister", "mother", "daughter", "aunt", "wife", "she"],
            "a_terms": ["career", "office", "salary", "professional", "corporation", "management", "executive", "business"],
            "b_terms": ["home", "parents", "children", "family", "marriage", "wedding", "relatives", "house"],
        },
        {
            "discipline": "sociology",
            "bias_name": "Rich-poor competence",
            "hypothesis": "Wealth terms may align more with competence and status than poverty terms do.",
            "x_terms": ["rich", "wealthy", "affluent", "elite", "prosperous", "privileged", "millionaire", "capitalist"],
            "y_terms": ["poor", "needy", "destitute", "workingclass", "impoverished", "unemployed", "welfare", "homeless"],
            "a_terms": ["competent", "intelligent", "successful", "capable", "efficient", "educated", "skilled", "professional"],
            "b_terms": ["lazy", "ignorant", "weak", "dependent", "chaotic", "unstable", "failing", "burdensome"],
        },
    ]
    proposals = []
    stamp = now_stamp().replace("-", "")
    for idx, item in enumerate(library[:n_proposals], start=1):
        proposals.append({"proposal_id": f"rules_{stamp}_{idx:02d}", **item})
    return proposals


def ollama_backend(
    n_proposals: int,
    model: str,
    host: str,
    timeout_seconds: int,
    think: str | None,
    run_dir: Path,
) -> list[dict[str, Any]]:
    prompt_template = PROMPT_PATH.read_text(encoding="utf-8").strip()
    research_config = load_json(RESEARCH_CONFIG)
    proposal_limits = research_config.get("proposal_limits", {})
    prompt = textwrap.dedent(
        f"""
        {prompt_template}

        Research interests:
        {json.dumps(research_config['domains_of_interest'], indent=2)}

        Candidate set size guidance:
        - minimum candidates per set: {proposal_limits.get('min_candidate_terms_per_set', 16)}
        - maximum candidates per set: {proposal_limits.get('max_candidate_terms_per_set', 20)}
        - evaluator keeps only the first 8 unique in-vocabulary tokens per set

        Structured history:
        {history_summary()}

        Generate exactly {n_proposals} proposals.
        """
    ).strip()
    (run_dir / "ollama_prompt.txt").write_text(prompt, encoding="utf-8")

    resolved_think = resolve_ollama_think(model, think)
    canonical_model = canonical_model_name(model)
    family = model_family(canonical_model)
    write_json(
        run_dir / "ollama_request_settings.json",
        {
            "requested_model": model,
            "effective_model": canonical_model,
            "model_family": family,
            "host": host,
            "timeout_seconds": timeout_seconds,
            "think": resolved_think,
        },
    )

    fallback_prompt = (
        prompt
        + "\n\nReturn only a single JSON object matching the schema. "
        + "Do not emit markdown fences or commentary."
    )

    attempts = [("json_mode", True, prompt), ("plain_json_fallback", False, fallback_prompt)]
    errors: list[str] = []
    for label, json_mode, attempt_prompt in attempts:
        body = ollama_generate(
            attempt_prompt,
            canonical_model,
            host,
            timeout_seconds,
            think=resolved_think,
            json_mode=json_mode,
        )
        raw = body.get("response", "")
        write_json(run_dir / f"ollama_response_body_{label}.json", body)
        (run_dir / f"ollama_raw_response_{label}.txt").write_text(str(raw), encoding="utf-8")
        try:
            parsed = extract_json_payload(str(raw))
        except ValueError as exc:
            errors.append(f"{label}: {exc}")
            continue
        if not isinstance(parsed, dict) or "proposals" not in parsed:
            errors.append(f"{label}: missing top-level proposals key")
            continue
        return parsed["proposals"]

    raise ValueError(" ; ".join(errors))


def load_manual_proposals(path: Path) -> list[dict[str, Any]]:
    return load_json(path)["proposals"]


def get_proposals(args: argparse.Namespace, run_dir: Path) -> list[dict[str, Any]]:
    if args.backend == "rules":
        return social_science_rules_backend(args.n_proposals)
    if args.backend == "manual":
        if not args.manual_proposals:
            raise ValueError("--manual-proposals is required for the manual backend")
        return load_manual_proposals(args.manual_proposals)
    if args.backend == "ollama":
        if not args.model:
            raise ValueError("--model is required for the ollama backend")
        return ollama_backend(
            n_proposals=args.n_proposals,
            model=args.model,
            host=args.ollama_host,
            timeout_seconds=args.ollama_timeout,
            think=args.ollama_think,
            run_dir=run_dir,
        )
    raise ValueError(f"Unknown backend: {args.backend}")


def existing_signatures() -> set[str]:
    return {row["signature"] for row in load_csv_rows(LEADERBOARD_PATH) if row.get("signature")}


def append_leaderboard(results: list[dict[str, Any]]) -> None:
    rows = load_csv_rows(LEADERBOARD_PATH)
    for result in results:
        metrics = result.get("metrics", {})
        rows.append(
            {
                "proposal_id": result.get("proposal_id", ""),
                "discipline": result.get("discipline", ""),
                "bias_name": result.get("bias_name", ""),
                "signature": result.get("signature", ""),
                "effect_size": metrics.get("effect_size", ""),
                "abs_effect_size": metrics.get("abs_effect_size", ""),
                "directional_p_value": metrics.get("directional_p_value", ""),
                "supported_orientation": metrics.get("supported_orientation", ""),
                "accepted": result.get("accepted", ""),
                "rationale_code": result.get("rationale_code", ""),
                "error": result.get("error", ""),
            }
        )
    with LEADERBOARD_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "proposal_id",
                "discipline",
                "bias_name",
                "signature",
                "effect_size",
                "abs_effect_size",
                "directional_p_value",
                "supported_orientation",
                "accepted",
                "rationale_code",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def append_history(results: list[dict[str, Any]]) -> None:
    rows = load_csv_rows(HISTORY_PATH)
    for result in results:
        metrics = result.get("metrics", {})
        rows.append(
            {
                "proposal_id": result.get("proposal_id", ""),
                "discipline": result.get("discipline", ""),
                "bias_name": result.get("bias_name", ""),
                "abs_effect_size": metrics.get("abs_effect_size", ""),
                "directional_p_value": metrics.get("directional_p_value", ""),
                "accepted": result.get("accepted", ""),
                "rationale_code": result.get("rationale_code", ""),
            }
        )
    with HISTORY_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "proposal_id",
                "discipline",
                "bias_name",
                "abs_effect_size",
                "directional_p_value",
                "accepted",
                "rationale_code",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def update_best_finding(results: list[dict[str, Any]]) -> None:
    accepted = [result for result in results if result.get("accepted")]
    if not accepted:
        return
    best = max(accepted, key=lambda item: item["metrics"]["abs_effect_size"])
    current = load_json(BEST_FINDING_PATH) if BEST_FINDING_PATH.exists() else None
    if current is None or float(best["metrics"]["abs_effect_size"]) > float(current["metrics"]["abs_effect_size"]):
        write_json(BEST_FINDING_PATH, best)


def write_summary(run_dir: Path, results: list[dict[str, Any]]) -> None:
    lines = ["# Round Summary", ""]
    for result in results:
        lines.append(f"## {result.get('proposal_id', '')} - {result.get('bias_name', '')}")
        lines.append(f"- discipline: {result.get('discipline', '')}")
        if result.get("error"):
            lines.append(f"- error: {result['error']}")
            lines.append(f"- rationale_code: {result.get('rationale_code', '')}")
            lines.append("")
            continue
        metrics = result["metrics"]
        lines.append(f"- effect_size: {metrics['effect_size']}")
        lines.append(f"- abs_effect_size: {metrics['abs_effect_size']}")
        lines.append(f"- directional_p_value: {metrics['directional_p_value']}")
        lines.append(f"- supported_orientation: {metrics['supported_orientation']}")
        lines.append(f"- accepted: {result.get('accepted')}")
        lines.append(f"- rationale_code: {result.get('rationale_code', '')}")
        lines.append("")
    (run_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate_round(proposals: list[dict[str, Any]], run_dir: Path) -> list[dict[str, Any]]:
    ensure_dirs()
    research_config = load_json(RESEARCH_CONFIG)
    manifest = source_manifest()
    current_signatures = existing_signatures()
    results = []
    with EmbeddingStore(manifest) as store:
        for proposal in proposals:
            result = evaluate_proposal(proposal, store, research_config, current_signatures)
            if result.get("signature"):
                current_signatures.add(result["signature"])
            results.append(result)
    write_json(run_dir / "results.json", {"results": results})
    append_leaderboard(results)
    append_history(results)
    update_best_finding(results)
    write_summary(run_dir, results)
    rebuild_dashboard()
    return results


def command_round(args: argparse.Namespace) -> None:
    source_manifest()
    run_dir = RUNS_DIR / f"{now_stamp()}-{args.backend}"
    run_dir.mkdir(parents=True, exist_ok=True)
    proposals = get_proposals(args, run_dir)
    write_json(run_dir / "proposals.json", {"proposals": proposals})
    evaluate_round(proposals, run_dir)


def command_loop(args: argparse.Namespace) -> None:
    source_manifest()
    for _ in range(args.rounds):
        round_args = argparse.Namespace(**vars(args))
        command_round(round_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AutoWEAT autoresearch runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    reset = subparsers.add_parser("reset", help="Delete prior runs and state while keeping the prepared source.")
    reset.set_defaults(func=command_reset)

    prepare = subparsers.add_parser("prepare", help="Download/extract/index an embedding source.")
    prepare.add_argument("--download", action="store_true")
    prepare.add_argument("--txt-path", type=Path, default=None)
    prepare.add_argument("--index-path", type=Path, default=None)
    prepare.set_defaults(func=command_prepare)

    status = subparsers.add_parser("status", help="Show the active embedding source and best finding.")
    status.set_defaults(func=command_status)

    round_parser = subparsers.add_parser("round", help="Run one discovery round.")
    round_parser.add_argument("--backend", choices=("rules", "manual", "ollama"), default="rules")
    round_parser.add_argument("--n-proposals", type=int, default=3)
    round_parser.add_argument("--manual-proposals", type=Path, default=None)
    round_parser.add_argument("--model", type=str, default=None)
    round_parser.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST)
    round_parser.add_argument("--ollama-timeout", type=int, default=DEFAULT_OLLAMA_TIMEOUT_SECONDS)
    round_parser.add_argument("--ollama-think", type=str, default=DEFAULT_OLLAMA_THINK)
    round_parser.set_defaults(func=command_round)

    loop = subparsers.add_parser("loop", help="Run several discovery rounds.")
    loop.add_argument("--backend", choices=("rules", "manual", "ollama"), default="rules")
    loop.add_argument("--rounds", type=int, default=3)
    loop.add_argument("--n-proposals", type=int, default=3)
    loop.add_argument("--manual-proposals", type=Path, default=None)
    loop.add_argument("--model", type=str, default=None)
    loop.add_argument("--ollama-host", type=str, default=DEFAULT_OLLAMA_HOST)
    loop.add_argument("--ollama-timeout", type=int, default=DEFAULT_OLLAMA_TIMEOUT_SECONDS)
    loop.add_argument("--ollama-think", type=str, default=DEFAULT_OLLAMA_THINK)
    loop.set_defaults(func=command_loop)

    return parser


def main() -> None:
    ensure_dirs()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
