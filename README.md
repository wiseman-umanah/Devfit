# DevFit

**Your GitHub profile is your CV. DevFit just writes it for you.**

DevFit takes a public GitHub username and generates a professional, ATS-safe CV — grounded entirely in what is actually in the profile. No hallucinations, no invented skills, no filler phrases. Every claim traces to a real GitHub artefact.

Optionally paste a job description and DevFit tailors the CV to the role, then scores how well the profile matches it.

---

## Quick start (Docker)

```bash
git clone <repo>
cd devfit
cp .env.example .env
# Edit .env — add your GROQ_API_KEY (required) and optionally GITHUB_TOKEN
docker compose up
```

Open **http://localhost:8000** in your browser.

---

## Quick start (local)

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd devfit
uv sync
cp .env.example .env
# Edit .env — add GROQ_API_KEY
uv run devfit-server
```

Open **http://localhost:8000**.

---

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | **Yes** | Groq inference API key. Get one free at [console.groq.com](https://console.groq.com). |
| `GITHUB_TOKEN` | No | Personal-access token for the **server** (not the user's). Raises rate limit from 60 to 5,000 req/hr. Generate at [github.com/settings/tokens](https://github.com/settings/tokens) — zero scopes required. |
| `LOG_LEVEL` | No | `INFO` (default) or `DEBUG`. |
| `DEVFIT_ENV` | No | `development` (default) or `production`. |

---

## How it works

```
GitHub username (+ optional JD + optional details)
        │
        ▼
GitHub Collector   — async, per-run cache
        │  ArtefactBundle: repos, language stats, READMEs,
        │  contribution graph, account metadata
        ▼
4-Agent CV Pipeline
  Agent 1 — Generator  (gpt-oss-20b)    drafts the full CV
  Agent 2 — Guard      (llama-guard-86m) safety + injection filter
  Agent 3 — Reviewer   (gpt-oss-120b)   quality audit (6 rules, JSON verdict)
  Agent 4 — Polisher   (llama-22m)      tightens phrasing, removes redundancy
        │
        ▼
Post-processing
  — em-dashes, emojis, filler phrases stripped
  — star counts below 3 suppressed
  — contact details from "Other Details" tab injected
        │
        ▼
Live preview  →  Edit CV  →  Export PDF
```

When a job description is provided the pipeline uses `tailored_cv.txt` instead of `standalone_cv.txt` and enables the **Match CV to JD** button, which scores matched skills and gaps side-by-side in the preview panel.

---

## Web UI

The UI is a two-panel split view accessible at `/` or `/<github-username>` (e.g. `http://localhost:8000/torvalds` pre-fills the username field).

**Left panel — Generate tab**

| Sub-tab | What goes here |
|---|---|
| GitHub | Username + optional reference CV (drag-and-drop file upload, read in-memory only) |
| Job Description | Paste a JD to tailor the CV and enable the Match button |
| Other Details | Name, email, phone, LinkedIn, portfolio, location, extra skills, bio hint |

**Left panel — Edit CV tab**

Markdown editor with formatting toolbar (Bold, Italic, H2, H3, bullet/numbered list, Undo, Redo) and an AI Edit bar — select any text, type an instruction, click AI Edit.

**Right panel**

Live Markdown preview (CV canvas is always white, even in dark mode) with the JD match report directly below it after a match run.

**Top bar:** Match CV to JD · Export PDF · Dark/Light toggle

---

## REST API

The server exposes a versioned JSON API used by the UI (and available for direct use):

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `GET` | `/{username}` | Web UI with username pre-filled |
| `GET` | `/health` | Liveness probe → `{"status": "ok"}` |
| `POST` | `/api/v1/generate` | GitHub username → CV Markdown |
| `POST` | `/api/v1/edit` | AI-rewrite a selected CV section |
| `POST` | `/api/v1/match` | CV + JD → score, matched skills, gaps |
| `POST` | `/api/v1/export-pdf` | CV Markdown → PDF binary stream |
| `POST` | `/api/v1/analyze` | Full JD-fit pipeline (evidence-grounded) |

Interactive docs at **http://localhost:8000/docs**.

---

## PDF export

PDF export uses **pandoc** + **Chrome headless** on the server. If either is unavailable the endpoint returns `503` and the UI shows a clear message — the Markdown CV is always available as a fallback.

Inside Docker, pandoc and Chrome are **not** included by default (they significantly inflate image size). To enable PDF export in Docker, uncomment the relevant lines in the `Dockerfile` and rebuild.

---

## The JD-fit pipeline (`/api/v1/analyze`)

For deeper analysis the full evidence-grounded pipeline is also available:

```
JD text + GitHub username
        │
        ▼
JD Analyzer         → list[Claim] (atomic, checkable assertions)
        ▼
Evidence Matcher    → keyword scoring; skip likely-unverifiable claims
+ First-Pass Classifier  (Groq, concurrent)
        ▼
Independent Verifier
  Layer 1: rule-based (date arithmetic, language presence, zero-activity)
  Layer 2: ConstrainedLLM — can confirm or downgrade; CANNOT upgrade
        ▼
Fit Report + Tailored CV + Evidence Appendix + Improvement Suggestions
```

Every claim in the output is classified **Verified**, **Contradicted**, or **Unverifiable** against real GitHub artefacts.

---

## Development

```bash
uv sync                               # install all deps including dev
uv run pytest -m "not integration"    # unit tests (no network, ~10s)
uv run pytest -m integration          # network tests (requires live keys)
uv run ruff check src/ tests/         # lint
uv run basedpyright src/              # type check
```

All three must pass clean before committing.

---

## Stack

| Component | Choice |
|---|---|
| Language | Python 3.12 |
| Package manager | `uv` (lockfile-pinned) |
| Web framework | FastAPI + Uvicorn |
| Templating | Jinja2 + static files |
| LLM | Groq (`openai/gpt-oss-120b`, `gpt-oss-20b`, `llama-guard`) |
| HTTP client | `httpx` (async) |
| Schema / validation | Pydantic v2 |
| Config | pydantic-settings (`.env` merge) |
| Lint / format | ruff |
| Type checking | basedpyright |
| Container | Docker (multi-stage, non-root, slim) |

---

## Project layout

```
devfit/
├── src/devfit/
│   ├── schema.py              Claim, Artefact, Verdict — Pydantic v2, frozen
│   ├── config.py              Settings, get_settings()
│   ├── github/                client.py, collector.py, bundle.py
│   ├── pipeline/              analyzer.py, matcher.py, classifier.py
│   ├── verifier/              rules.py, llm.py, verifier.py
│   ├── output/                cv.py, standalone_cv.py, agents.py,
│   │                          cv_reviewer.py, cv_utils.py, report.py,
│   │                          appendix.py, improvements.py, pdf.py
│   ├── baseline/              BaselinePipeline (hallucination benchmark)
│   ├── api/
│   │   ├── app.py             FastAPI app, routes
│   │   ├── static/            devfit.css, devfit.js
│   │   ├── templates/         index.html (Jinja2)
│   │   └── routers/           generate, edit, match, export_pdf, analyze, health
│   └── prompts/               all LLM prompts as .txt files
├── tests/                     195 unit tests, mirrors src/devfit/
├── eval/                      10 real-profile test cases, ground truth labels
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```
