"use strict";

// ── State ──────────────────────────────────────────────────────────────────
let cvMarkdown = "";
let matchData  = null;
let refCvText  = "";      // reference CV text read from uploaded file
let aiSelStart = 0;       // persisted selection for AI edit
let aiSelEnd   = 0;

// ── DOM refs ───────────────────────────────────────────────────────────────
const cvEditor     = document.getElementById("cv-editor");
const cvPreview    = document.getElementById("cv-preview");
const statusBar    = document.getElementById("status-bar");
const btnGenerate  = document.getElementById("btn-generate");
const btnExport    = document.getElementById("btn-export");
const btnMatch     = document.getElementById("btn-match");
const btnAiEdit    = document.getElementById("btn-ai-edit");
const wordCount    = document.getElementById("preview-word-count");
const matchPanel   = document.getElementById("match-panel");
const btnTheme     = document.getElementById("btn-theme");
const dropZone     = document.getElementById("ref-cv-drop");
const dropInput    = document.getElementById("ref-cv-input");
const dropFilename = document.getElementById("ref-cv-filename");
const dropClear    = document.getElementById("ref-cv-clear");

// ── Theme toggle ───────────────────────────────────────────────────────────
const prefersDark = window.matchMedia("(prefers-color-scheme: dark)");
let darkMode = prefersDark.matches;

function applyTheme() {
  document.documentElement.setAttribute("data-theme", darkMode ? "dark" : "light");
  btnTheme.textContent = darkMode ? "Light" : "Dark";
}
applyTheme();
btnTheme.addEventListener("click", () => { darkMode = !darkMode; applyTheme(); });
prefersDark.addEventListener("change", e => { darkMode = e.matches; applyTheme(); });

// ── Outer tabs (Generate | Edit CV) ───────────────────────────────────────
document.querySelectorAll(".outer-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".outer-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".outer-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    const panel = document.getElementById("outer-" + tab.dataset.outer);
    if (panel) panel.classList.add("active");
  });
});

// ── Inner tabs (GitHub | Job Description | Other Details) ─────────────────
document.querySelectorAll(".inner-tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".inner-tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".inner-panel").forEach(p => p.classList.remove("active"));
    tab.classList.add("active");
    const panel = document.getElementById("inner-" + tab.dataset.inner);
    if (panel) panel.classList.add("active");
  });
});

// ── Reference CV file upload ───────────────────────────────────────────────
function handleRefCvFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    refCvText = e.target.result || "";
    dropFilename.textContent = file.name;
    dropClear.classList.add("visible");
  };
  reader.readAsText(file);
}

dropInput.addEventListener("change", () => handleRefCvFile(dropInput.files[0]));
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) { dropInput.files = e.dataTransfer.files; handleRefCvFile(file); }
});
dropClear.addEventListener("click", e => {
  e.stopPropagation();
  refCvText = "";
  dropFilename.textContent = "";
  dropClear.classList.remove("visible");
  dropInput.value = "";
});

// ── Collect "Other Details" fields into a formatted string ─────────────────
function buildDetailsText() {
  const parts = [];
  const v = id => document.getElementById(id).value.trim();

  if (v("detail-name"))     parts.push("Full name: "      + v("detail-name"));
  if (v("detail-email"))    parts.push("Email: "          + v("detail-email"));
  if (v("detail-phone"))    parts.push("Phone: "          + v("detail-phone"));
  if (v("detail-linkedin")) parts.push("LinkedIn: "       + v("detail-linkedin"));
  if (v("detail-website"))  parts.push("Portfolio/site: " + v("detail-website"));
  if (v("detail-location")) parts.push("Location: "       + v("detail-location"));
  if (v("detail-skills"))   parts.push("Additional skills to highlight: " + v("detail-skills"));
  if (v("detail-bio"))      parts.push("Summary/bio hint: " + v("detail-bio"));

  return parts.join("\n");
}

// ── Markdown → HTML renderer ───────────────────────────────────────────────
function mdToHtml(md) {
  if (!md) return "";
  let html = md
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/&lt;span class="source-tag"&gt;(.*?)&lt;\/span&gt;/gs,
             '<span class="source-tag">$1</span>')
    .replace(/^### (.+)$/gm, "<h3>$1</h3>")
    .replace(/^## (.+)$/gm,  "<h2>$1</h2>")
    .replace(/^# (.+)$/gm,   "<h1>$1</h1>")
    .replace(/^---$/gm, "<hr>")
    .replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>")
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/\*(.+?)\*/g, "<em>$1</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");

  const lines = html.split("\n");
  const out = [];
  let inUl = false, inOl = false;
  for (const line of lines) {
    const isUl = /^- (.+)$/.test(line);
    const isOl = /^\d+\. (.+)$/.test(line);
    if (isUl) {
      if (!inUl) { out.push("<ul>"); inUl = true; }
      out.push("<li>" + line.replace(/^- /, "") + "</li>");
    } else if (isOl) {
      if (!inOl) { out.push("<ol>"); inOl = true; }
      out.push("<li>" + line.replace(/^\d+\. /, "") + "</li>");
    } else {
      if (inUl) { out.push("</ul>"); inUl = false; }
      if (inOl) { out.push("</ol>"); inOl = false; }
      if (/^<h[123]>|^<hr>/.test(line)) out.push(line);
      else if (line.trim() !== "") out.push("<p>" + line + "</p>");
    }
  }
  if (inUl) out.push("</ul>");
  if (inOl) out.push("</ol>");
  return out.join("\n");
}

function updatePreview() {
  const md = cvEditor.value;
  cvMarkdown = md;
  if (!md.trim()) {
    cvPreview.innerHTML = `<div class="preview-empty">
      <div class="preview-empty-icon">&#9675;</div>
      <div class="preview-empty-text">Enter your GitHub username and click Generate CV</div>
    </div>`;
    wordCount.textContent = "";
    return;
  }
  cvPreview.innerHTML = mdToHtml(md);
  wordCount.textContent = md.trim().split(/\s+/).length + " words";
}
cvEditor.addEventListener("input", updatePreview);

// ── Status helpers ─────────────────────────────────────────────────────────
function setStatus(msg, type) {
  statusBar.innerHTML = type === "loading"
    ? `<span class="spinner"></span> ${escHtml(msg)}`
    : `<span class="${type === "ok" ? "status-ok" : type === "err" ? "status-err" : ""}">${escHtml(msg)}</span>`;
}
function escHtml(s) {
  return String(s)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}

// ── Auto-fill from URL path (e.g. /torvalds fills username) ───────────────
(function autoFillFromUrl() {
  const path = window.location.pathname.replace(/^\//, "").trim();
  if (path && !path.includes("/")) {
    const el = document.getElementById("github-username");
    if (el && !el.value) el.value = decodeURIComponent(path);
  }
})();

// ── Generate ───────────────────────────────────────────────────────────────
async function runGenerate() {
  const username = document.getElementById("github-username").value.trim();
  if (!username) {
    // Switch to GitHub inner tab to show the error in context
    document.querySelector('[data-inner="github"]').click();
    setStatus("Enter a GitHub username.", "err");
    return;
  }

  btnGenerate.disabled = true;
  btnGenerate.textContent = "Generating...";
  setStatus("Collecting GitHub profile...", "loading");
  matchPanel.classList.remove("visible");

  // Build extra_details from Other Details fields
  const detailsText = buildDetailsText();

  // Combine reference CV text and extra details into a single context string
  // so the existing API field can carry both without a schema change.
  let referenceContext = "";
  if (refCvText.trim()) referenceContext += refCvText.trim();
  if (detailsText)      referenceContext += (referenceContext ? "\n\n---\n" : "") + detailsText;

  try {
    const resp = await fetch("/api/v1/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        github_username: username,
        jd_text: document.getElementById("jd-text").value.trim() || null,
        reference_cv: referenceContext || null,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    cvEditor.value = data.cv_markdown;
    updatePreview();

    // Push username into the URL without reloading
    if (history.pushState) history.pushState({}, "", "/" + encodeURIComponent(username));

    // Switch to Edit CV tab so user can see and refine
    document.querySelector('[data-outer="editor"]').click();

    setStatus("Generated via " + data.stages_completed.join(" > "), "ok");
    btnExport.disabled = false;
    btnMatch.disabled = !document.getElementById("jd-text").value.trim();
  } catch (e) {
    setStatus("Generation failed: " + e.message, "err");
  } finally {
    btnGenerate.disabled = false;
    btnGenerate.textContent = "Generate CV";
  }
}

btnGenerate.addEventListener("click", runGenerate);

// Enable match button when JD has text and CV exists
document.getElementById("jd-text").addEventListener("input", () => {
  btnMatch.disabled = !(document.getElementById("jd-text").value.trim() && cvEditor.value.trim());
});

// ── Toolbar ────────────────────────────────────────────────────────────────
document.querySelectorAll(".tool-btn[data-action]").forEach(btn => {
  btn.addEventListener("click", () => {
    const action = btn.dataset.action;
    const ta = cvEditor;

    if (action === "undo") { document.execCommand("undo"); updatePreview(); return; }
    if (action === "redo") { document.execCommand("redo"); updatePreview(); return; }

    const start = ta.selectionStart, end = ta.selectionEnd;
    const sel    = ta.value.slice(start, end);
    const before = ta.value.slice(0, start);
    const after  = ta.value.slice(end);
    let rep = sel, off = 0;

    if (action === "bold")   { rep = `**${sel || "bold text"}**`;   off = sel ? 0 : -2; }
    if (action === "italic") { rep = `*${sel || "italic text"}*`;   off = sel ? 0 : -1; }
    if (action === "h2")     { rep = `\n## ${sel || "Section"}\n`;  }
    if (action === "h3")     { rep = `\n### ${sel || "Heading"}\n`; }

    if (action === "bullet" || action === "number") {
      if (sel) {
        // Prefix every non-blank selected line
        rep = sel.split("\n").map((l, i) =>
          l.trim() ? (action === "bullet" ? `- ${l}` : `${i + 1}. ${l}`) : l
        ).join("\n");
      } else {
        // No selection — find the start of the current line and prefix it.
        // If the line is already blank, insert a new list item on a new line.
        const lineStart = before.lastIndexOf("\n") + 1;
        const lineText  = ta.value.slice(lineStart, end).trimEnd();
        const prefix    = action === "bullet" ? "- " : "1. ";
        if (lineText.trim() === "") {
          // Blank line: insert "- \n" or "1. \n" and place cursor after prefix
          rep = prefix;
          off = 0;
        } else {
          // Non-blank line: prepend prefix to the whole line
          const lineEnd = ta.value.indexOf("\n", end);
          const fullLine = ta.value.slice(lineStart, lineEnd === -1 ? undefined : lineEnd);
          const newFull  = prefix + fullLine;
          ta.value = ta.value.slice(0, lineStart) + newFull +
                     (lineEnd === -1 ? "" : ta.value.slice(lineEnd));
          ta.setSelectionRange(lineStart + newFull.length, lineStart + newFull.length);
          ta.focus();
          updatePreview();
          return;
        }
      }
    }

    ta.value = before + rep + after;
    const np = start + rep.length + off;
    ta.setSelectionRange(np, np);
    ta.focus();
    updatePreview();
  });
});

// ── AI Edit ────────────────────────────────────────────────────────────────
// Capture selection on every pointer/key interaction so we know the range
// even after the button click steals focus from the textarea.
cvEditor.addEventListener("mouseup",  captureSelection);
cvEditor.addEventListener("keyup",    captureSelection);
cvEditor.addEventListener("select",   captureSelection);

function captureSelection() {
  aiSelStart = cvEditor.selectionStart;
  aiSelEnd   = cvEditor.selectionEnd;
  const hint = document.getElementById("ai-sel-hint");
  if (hint) {
    const chars = aiSelEnd - aiSelStart;
    hint.textContent = chars > 0 ? chars + " chars selected" : "";
  }
}

btnAiEdit.addEventListener("click", async () => {
  const selected    = cvEditor.value.slice(aiSelStart, aiSelEnd).trim();
  const instruction = document.getElementById("ai-instruction").value.trim();

  if (!selected)    { setStatus("Select a section of the CV first.", "err"); return; }
  if (!instruction) { setStatus("Type an instruction for the AI.", "err"); return; }

  // Restore the visual selection so the user can see what will be replaced
  cvEditor.focus();
  cvEditor.setSelectionRange(aiSelStart, aiSelEnd);
  cvEditor.classList.add("ai-active");

  btnAiEdit.disabled = true;
  btnAiEdit.textContent = "Editing...";
  setStatus("AI is rewriting the selected section...", "loading");

  try {
    const resp = await fetch("/api/v1/edit", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        full_cv: cvEditor.value,
        selected_text: selected,
        instruction: instruction,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const data = await resp.json();
    cvEditor.value = cvEditor.value.slice(0, aiSelStart) + data.replacement + cvEditor.value.slice(aiSelEnd);
    cvEditor.setSelectionRange(aiSelStart, aiSelStart + data.replacement.length);
    cvEditor.focus();
    updatePreview();
    document.getElementById("ai-instruction").value = "";
    setStatus("AI edit applied.", "ok");
  } catch (e) {
    setStatus("AI edit failed: " + e.message, "err");
  } finally {
    cvEditor.classList.remove("ai-active");
    btnAiEdit.disabled = false;
    btnAiEdit.textContent = "AI Edit";
  }
});

// ── Match ──────────────────────────────────────────────────────────────────
btnMatch.addEventListener("click", async () => {
  const jd = document.getElementById("jd-text").value.trim();
  const cv = cvEditor.value.trim();
  if (!jd || !cv) { setStatus("Both CV and JD are required for matching.", "err"); return; }

  btnMatch.disabled = true;
  btnMatch.textContent = "Matching...";
  setStatus("Analysing CV against job description...", "loading");

  try {
    const resp = await fetch("/api/v1/match", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cv_text: cv, jd_text: jd }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    matchData = await resp.json();
    renderMatchPanel(matchData);
    const rs = document.querySelector(".right-scroll");
    if (rs) setTimeout(() => rs.scrollTo({ top: rs.scrollHeight, behavior: "smooth" }), 80);
    setStatus("Match score: " + matchData.score + "/100", matchData.score >= 70 ? "ok" : "err");
  } catch (e) {
    setStatus("Match failed: " + e.message, "err");
  } finally {
    btnMatch.disabled = false;
    btnMatch.textContent = "Match CV to JD";
  }
});

function renderMatchPanel(data) {
  const sc = data.score >= 70 ? "score-high" : data.score >= 40 ? "score-mid" : "score-low";
  let mHtml = "", gHtml = "";

  if (data.matched.length) {
    mHtml = `<div class="match-section-title">Matched (${data.matched.length})</div>`;
    data.matched.forEach(m => {
      mHtml += `<div class="match-item match-item-matched">
        <div class="match-item-skill">${escHtml(m.skill)}</div>
        <div class="match-item-evidence">${escHtml(m.evidence)}</div>
      </div>`;
    });
  }

  if (data.gaps.length) {
    gHtml = `<div class="match-section-title">Gaps (${data.gaps.length})</div>`;
    data.gaps.forEach(g => {
      gHtml += `<div class="match-item match-item-gap">
        <div class="match-item-skill">${escHtml(g.requirement)}</div>
        <div class="match-item-suggestion">Suggestion: ${escHtml(g.suggestion)}</div>
      </div>`;
    });
  }

  matchPanel.innerHTML = `
    <div class="match-header">
      <div class="match-score-badge ${sc}">${data.score}</div>
      <div class="match-header-content">
        <div class="match-label">JD Match Score</div>
        <div class="match-summary">${escHtml(data.summary)}</div>
      </div>
    </div>
    <div class="match-body">
      <div class="match-cols">
        <div>${mHtml}</div>
        <div>${gHtml}</div>
      </div>
    </div>`;
  matchPanel.classList.add("visible");
}

// ── Export PDF ─────────────────────────────────────────────────────────────
btnExport.addEventListener("click", async () => {
  const md = cvEditor.value.trim();
  if (!md) { setStatus("No CV content to export.", "err"); return; }

  btnExport.disabled = true;
  btnExport.textContent = "Exporting...";
  setStatus("Generating PDF...", "loading");

  try {
    const username = document.getElementById("github-username").value.trim() || "cv";
    const resp = await fetch("/api/v1/export-pdf", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ cv_markdown: md, filename: username }),
    });
    if (resp.status === 503) {
      const err = await resp.json().catch(() => ({ detail: "PDF tools unavailable" }));
      throw new Error(err.detail);
    }
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({ detail: resp.statusText }));
      throw new Error(err.detail || resp.statusText);
    }
    const blob = await resp.blob();
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement("a");
    a.href = url; a.download = username + ".pdf";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    setStatus("PDF downloaded.", "ok");
  } catch (e) {
    setStatus("PDF export failed: " + e.message, "err");
  } finally {
    btnExport.disabled = false;
    btnExport.textContent = "Export PDF";
  }
});

// ── Initialise ────────────────────────────────────────────────────────────
updatePreview();
