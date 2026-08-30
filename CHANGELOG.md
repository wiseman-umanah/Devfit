# CHANGELOG

All significant experiments, decisions, and metric deltas recorded in build order.
Format: **stage / date — what was tried, why, outcome**.

---

## Stage 0 — Foundation (2024-08-30)

**Scaffold and schema design.**

- Chose Pydantic v2 frozen models over dataclasses because `model_validator` lets us enforce the invariant "evidence must be non-empty unless classification is UNVERIFIABLE" at the data layer, not scattered across pipeline code.
- `StrEnum` over `Enum` throughout: values serialise directly to JSON without `.value` calls — critical when the trajectory log writes raw dict dumps.
- Decided against `TypedDict` for the schema layer: Pydantic catches constraint violations at construction time, TypedDict does not. The bug prevention is worth the overhead.
- `get_settings()` with `@lru_cache(maxsize=1)` rather than module-level instantiation: lets tests call `get_settings.cache_clear()` to inject different env vars without monkeypatching.

**Removed experiment:** Initially considered using a YAML config file alongside `.env`. Removed — two config sources with identical keys and different parse semantics is a maintenance trap, not a feature.

---

## Stage 1 — Rule-Based Verifier (2024-08-30)

**Why rules before LLM?**

Rule-based checks are deterministic, free (no API call), and fast. Any claim that can be resolved by checking a date or a language stat should never reach the LLM. The rule layer exists to save money and reduce non-determinism, not just to be architecturally elegant.

**Key design decision — `RuleVerifier.run()` returns `None`, not `"unverifiable"`:**
Returning `None` means "this rule cannot resolve the claim" — it passes to the next layer. Returning `"unverifiable"` would mean "we checked and couldn't confirm" — a meaningful verdict. These are not the same. The distinction is what makes Layer 1 → Layer 2 handoff clean.

**Discovered gotcha:** `check_language_presence` extracts capitalised tokens from claim text using a heuristic. A claim like `"Experience with FastAPI and Go"` will attempt to verify both `FastAPI` and `Go`. Test cases for "Layer 1 returns None" must use `ROLE_SCOPE` category claims that contain no capitalisable tech tokens.

---

## Stage 2 — Ground Truth Labels (2024-08-30)

**Why label before building the pipeline?**

If you run the pipeline and then label based on what the pipeline produces, you are measuring whether the system is self-consistent, not whether it is correct. The 53 labels in `eval/ground_truth.json` were written before any GitHub API call or LLM inference.

**Distribution:** 26 verified / 5 contradicted / 22 unverifiable (across 10 cases).

**Engineered case (case_07):** Replaced the original placeholder username `new-dev-2024` with the real account `wiseman-umanah` — a genuine 2024-created account with no Python activity vs. a JD requiring 7+ years Python. This tests the `check_zero_activity` and `check_date_arithmetic` rules on a live profile.

**Label that required re-examination:** `c01-002` ("15+ years of experience building production systems") — labeled `verified` despite being an experience-duration claim, because the GitHub account creation date + the fact that the linux repo predates GitHub provides partial but real artefact support. The notes field documents this reasoning.

**Removed experiment:** Considered labeling at the claim level AND the case level (overall fit verdict). Removed the case-level labels — they compound multiple uncertain claim verdicts into a single score, which would make the metric gameable and the error analysis opaque.

---

## Stage 3 — GitHub Collector (2024-08-30)

**Per-run in-memory cache design:**
The cache is keyed by URL and lives on the `GitHubClient` instance, which is created once per pipeline run as an async context manager. It is cleared on `__aexit__`. This means the cache is run-scoped (no cross-run pollution) but shared across all concurrent API calls in a single run (no duplicate fetches even when multiple pipeline stages request the same endpoint).

**Rate limit handling:** Raises `GitHubRateLimitError` with the `X-RateLimit-Reset` epoch timestamp from the response header. The CLI catches this and exits with a clear message. The eval runner skips the case and continues.

---

## Stage 4 — JD Analyzer (2024-08-30)

**Auto-flag design:**
`SOFT_SKILL` and `EXPERIENCE_DURATION` claims are always set to `likely_unverifiable=True` in `_parse_claims_response()`, regardless of what the LLM returns. This is a defence against the LLM occasionally marking a "team leadership" claim as verifiable — it is not, by definition, from public GitHub data.

**`temperature=0.0` throughout:**
All LLM calls in the pipeline use `temperature=0.0`. We are doing classification and extraction, not generation. Non-zero temperature introduces non-determinism that makes evaluation results non-reproducible. The only place non-determinism would be acceptable is the CV prose — but even there, reproducibility matters for the eval.

---

## Stage 5 — Evidence Matcher + First-Pass Classifier (2024-08-30)

**Skip pass before keyword scoring:**
Claims flagged `likely_unverifiable=True` are short-circuited before any artefact retrieval or keyword scoring. They get a direct `UNVERIFIABLE` verdict from `build_unverifiable_verdicts()`. This is not just an optimisation — it ensures the LLM verifier never sees a claim it cannot possibly resolve, which would waste tokens and risk a hallucinated "verified" on a soft-skills claim.

**Concurrent classification:**
`FirstPassClassifier.classify()` dispatches all active claims via `asyncio.gather`. On a run with 10 active claims, this is ~10x faster than sequential calls. The gather uses `return_exceptions=True` — any single LLM failure degrades gracefully to `unverifiable` for that claim rather than failing the whole batch.

---

## Stage 6 — Independent Verifier (2024-08-30)

**The "cannot upgrade" constraint:**
`ConstrainedLLMVerifier` can confirm a `verified` draft or downgrade it to `unverifiable`. It cannot upgrade an `unverifiable` draft to `verified`. This asymmetry is intentional: the classifier runs in one direction (optimistic), the verifier runs in the opposite direction (sceptical). If both agree, confidence is high. If they disagree, the verifier wins.

**Prompt escaping bug discovered:**
`constrained_verifier.txt` originally used bare `{` and `}` around JSON examples in the prompt. Python's `.format()` parsed these as format string placeholders and raised `KeyError`. Fixed by escaping all literal braces in the prompt file as `{{` and `}}`.

**`VerifierDecision` dataclass for trajectory:**
Introduced a lightweight `VerifierDecision(claim_id, layer, was_downgraded, reason)` dataclass returned alongside the final verdicts. This is logged to `trajectory_log.jsonl` to make downgrade events observable without post-hoc reasoning.

---

## Stage 7 — Output Generators (2024-08-30)

**Zero-unsupported-CV-claims invariant:**
`CVGenerator` is tested with an explicit assertion: `assert missing == []` where `missing` is every non-unverifiable CV line without an `artefact_pointer`. This is not a soft warning — it is a hard test failure. The constraint "every CV claim must trace to a GitHub artefact" is the core product guarantee.

**Score formula:**
`raw = verified × 1 + contradicted × (-1)`, normalised to `(raw / total_scorable) × 100`. One verified + one contradicted = 0%, not 50%. This is intentional — a net-zero score should read as neutral, not "half-good". Unverifiable claims are never scored.

**Score labels:** Strong Fit ≥ 70% | Partial Fit ≥ 40% | Weak Fit ≥ 10% | Inconclusive < 10%.

**TrajectoryLogger:**
Uses `open(..., "a")` in a context manager, flushes after every `log_event` call. Append-only: a crash mid-run leaves a readable partial log rather than a corrupted file. The JSONL format (one JSON object per line) means the log can be `tail -f`-ed in real time and parsed line-by-line without a JSON array parser.

---

## Stage 8 — Human Checkpoint + CLI (2024-08-30)

**Why a human checkpoint at all?**
The pipeline can produce a CV that is technically evidence-grounded but reads poorly (awkward phrasing, wrong emphasis, missing context the human knows). The checkpoint is not a fallback for when the pipeline fails — it is a design-level constraint that the human must explicitly approve before any output file is written.

**Edit flow:**
On `(e)dit`, the system opens the draft in `$EDITOR` if set (falls back to `nano`). If no terminal is available (e.g. batch eval mode), the `HumanCheckpoint` is bypassed entirely — the eval runner calls pipeline stages directly without going through `_run_pipeline()`.

**Run ID:**
Each CLI run generates a UUID4 hex prefix as a `run_id`, creating `output/<run_id>/`. This prevents output from separate runs overwriting each other. The `run_id` appears in the trajectory log for correlation.

---

## Stage 9 — Baseline Pipeline (2024-08-30)

**Why build this last?**
If you build the baseline first, there is a temptation to "improve" it to make DevFit's relative improvement look better. Building it last, with a deliberate mandate to keep it simple, prevents benchmark gaming.

**`build_github_summary()` is intentionally lossy:**
The function converts an `ArtefactBundle` to an unstructured plain-text paragraph. This is not a simplification for convenience — it is the exact failure mode we are measuring. The baseline receives the same GitHub data as DevFit, but loses all structure, type information, and artefact pointers in the conversion. The hallucination rate measures the cost of that loss.

**Synthetic verdicts for scoring:**
The baseline has no structured classification system. To feed `score.py`'s hallucination-rate metric, `run_eval.py` emits one synthetic `"verified"` verdict per CV line, with an empty `evidence` list. This is not cheating the metric — it is precisely how the metric is defined: claims presented as fact with no artefact pointer.

---

## Stage 10 — Evaluation Infrastructure (2024-08-30)

**eval/run_eval.py design:**
The runner bypasses the CLI and `HumanCheckpoint` entirely, importing pipeline stages directly. This is intentional for batch evaluation. The eval run must be non-interactive. Cases that hit rate limits or LLM errors are skipped with a logged error rather than aborting the entire run.

**eval/score.py `sys.exit(2)` on non-zero unsupported CV rate:**
A non-zero unsupported CV claim rate is treated as a hard failure (exit code 2, not 1). This distinguishes "some cases failed" (exit 1) from "the core guarantee was violated" (exit 2). CI should treat both as failures but report them differently.

---

## Known Limitations

- **Private repos:** DevFit only accesses public GitHub artefacts. Candidates with significant private work history will have many `unverifiable` verdicts — this is correct behaviour, not a bug.
- **GitHub API rate limits:** Unauthenticated requests are limited to 60/hour. A `GITHUB_TOKEN` in `.env` raises this to 5,000/hour. The eval run requires a token.
- **LLM non-determinism at `temperature=0.0`:** Groq's API is not guaranteed to produce identical outputs for identical inputs at `temperature=0.0`. In practice, eval results are highly stable, but not byte-for-byte reproducible across different days or API versions.
- **case_07 username dependency:** `wiseman-umanah` is a real 2024 account selected specifically because it has no Python activity and was created in 2023–2024. If this account is deleted or substantially updated, case_07's ground-truth labels may need revisiting.
