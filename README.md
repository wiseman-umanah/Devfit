# DevFit

**Hallucination is not a bug to patch. It's the architecture to replace.**

DevFit takes a job description and a public GitHub username. It produces a fit report and a tailored CV where every material claim is classified as **Verified**, **Contradicted**, or **Unverifiable** against real GitHub artefacts — and the CV only contains claims that survived that test.

The refusal mechanism is the product.

---

## The Problem

Most JD → CV tools do embedding similarity between resume and JD text, then generate a keyword-optimized rewrite. Plausible is not the same as true. The output sounds great and cannot be defended.

DevFit's entire design makes hallucination structurally expensive: every claim that reaches the final CV must survive an independent verification stage or be refused. The system does not soften this constraint when it's inconvenient.

---

## How It Works

```
JD text + GitHub username
         │
         ▼
[Stage 3] GitHub Collector        ← async concurrent, per-run cache
         │  ArtefactBundle (repos, language stats, READMEs,
         │  contribution graph, account metadata)
         ▼
[Stage 4] JD Analyzer             ← Groq, temperature=0.0
         │  list[Claim] — atomic, checkable assertions
         ▼
[Stage 5] Evidence Matcher        ← keyword scoring, skip likely-unverifiable
         │  + First-Pass Classifier (Groq, concurrent)
         │  → draft Verdicts (verified / contradicted / unverifiable)
         ▼
[Stage 6] Independent Verifier    ← two independent layers
         │  Layer 1: Rule-based (date arithmetic, language presence, zero-activity)
         │  Layer 2: ConstrainedLLM (Groq, confirm/downgrade ONLY — cannot upgrade)
         │  → final Verdicts + VerifierDecisions (full trajectory)
         ▼
[Stage 7] Report + CV Generator   ← evidence-linked Markdown
         │  fit_report.md — score built ONLY from verified/contradicted
         │  cv.md — every verified line carries an artefact pointer
         │  evidence_appendix.md — full pointer index
         ▼
[Stage 8] Human Checkpoint        ← approve / edit / abort
         │  No file is written without explicit approval.
         ▼
  output/<run_id>/
    ├── fit_report.md
    ├── cv.md
    ├── evidence_appendix.md
    └── trajectory_log.jsonl      ← every stage event + verifier decisions
```

The verifier is **independent**: Layer 1 runs synchronously. If a rule resolves a claim, Layer 2 (LLM) never sees it. If Layer 2 runs, it can only confirm or downgrade — never upgrade a verdict.

---

## Installation

Requires Python 3.12 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone <repo>
cd devfit
uv sync
cp .env.example .env
# Fill in GROQ_API_KEY (required) and GITHUB_TOKEN (optional, raises rate limit)
```

---

## Usage

```bash
# Production run
devfit --jd path/to/jd.txt --github torvalds

# With optional resume cross-check
devfit --jd path/to/jd.txt --github torvalds --resume resume.txt

# Include unverifiable claims in CV with explicit marker
devfit --jd "Senior Python engineer..." --github tiangolo --include-unverifiable

# Verbose dev mode
devfit-dev --jd jd.txt --github sindresorhus
```

Output is written to `./output/<run_id>/` after you approve at the human checkpoint.

---

## Evaluation

10 real public GitHub profiles were selected to cover strong fits, clear mismatches, partial fits, and an engineered contradicted case. 53 material claims were manually labeled before any pipeline run.

```bash
# Run the full eval (requires live keys, ~30 min)
uv run python eval/run_eval.py

# Run a single case
uv run python eval/run_eval.py --cases case_01_strong_fit

# Run only DevFit (skip Baseline)
uv run python eval/run_eval.py --devfit-only

# Score results against ground truth
uv run python eval/score.py \
    --ground-truth eval/ground_truth.json \
    --devfit-outputs eval/devfit_outputs/ \
    --baseline-outputs eval/baseline_outputs/
```

Metrics reported:

| Metric | Baseline | DevFit |
|---|---|---|
| Hallucination Rate | (TBD after live run) | (TBD) |
| Misclassification Rate | — | (TBD) |
| CV Claims Without Evidence | 100% | Target: 0% |

See [`eval/README.md`](eval/README.md) for full test-case descriptions and labeling methodology.

---

## Development

```bash
# Unit tests only (no network)
uv run pytest -m "not integration"

# Integration tests (requires GITHUB_TOKEN)
uv run pytest -m integration

# Lint
uv run ruff check src/ tests/ eval/

# Type check
uv run basedpyright src/
```

All three must pass clean before marking a stage complete.

---

## Stack

| Component | Choice | Reason |
|---|---|---|
| Runtime | Python 3.12 | Stable async, typed StrEnum |
| Package manager | `uv` | Fast, reproducible lockfile |
| LLM | Groq `openai/gpt-oss-120b` | Fast inference, low cost |
| Async HTTP | `httpx` | Native async, used by Groq SDK |
| Schema | Pydantic v2 | Model validators enforce invariants |
| Config | pydantic-settings | `.env` + env var merge |
| Lint/Format | ruff | Single tool, 88-char limit |
| Types | basedpyright | Strict, stricter than mypy |

---

## Trajectory Logging

Every pipeline run writes a `trajectory_log.jsonl` to the output directory. Each line is a JSON event:

```json
{"timestamp": "2024-01-15T10:23:11Z", "stage": "verification_complete", "data": {"total_verdicts": 12, "verified": 7, "contradicted": 2, "unverifiable": 3, "downgraded_count": 1}}
{"timestamp": "2024-01-15T10:23:11Z", "stage": "verifier_decision", "data": {"claim_id": "jd-003", "layer": 2, "was_downgraded": true, "reason": "pointer not found in evidence"}}
{"timestamp": "2024-01-15T10:23:12Z", "stage": "human_checkpoint", "data": {"action": "approved"}}
```

The trajectory is the evidence trail that shows *why* each claim was classified the way it was.

---

## Project Layout

```
devfit/
├── src/devfit/
│   ├── schema.py           # Claim, Artefact, Verdict (Pydantic v2, frozen)
│   ├── config.py           # Settings via pydantic-settings, get_settings()
│   ├── cli.py              # devfit / devfit-dev entry points
│   ├── github/             # client.py, collector.py, bundle.py
│   ├── pipeline/           # analyzer.py, matcher.py, classifier.py
│   ├── verifier/           # rules.py, llm.py, verifier.py
│   ├── output/             # report.py, cv.py, appendix.py, trajectory.py
│   ├── checkpoint/         # HumanCheckpoint (approve/edit/abort)
│   ├── baseline/           # BaselinePipeline (hallucination benchmark)
│   ├── api/                # FastAPI app (optional web interface)
│   └── prompts/            # all LLM prompts as .txt files
├── eval/
│   ├── test_cases/         # 10 cases: jd.txt + github_username.txt
│   ├── ground_truth.json   # 53 manually labeled claims (locked)
│   ├── run_eval.py         # batch eval runner
│   └── score.py            # Hallucination Rate + Misclassification Rate
└── tests/                  # mirrors src/devfit/ layout, 109 unit tests
```
