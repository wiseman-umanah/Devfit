# DevFit Evaluation Set

## Overview

10 test cases covering all required coverage types (per TRD §5.3).
Ground truth labels are in [`ground_truth.json`](./ground_truth.json).

**This file is fixed before any pipeline run. Do not update labels after
running DevFit — doing so corrupts the evaluation and invalidates the
Measured Improvement scores.**

---

## Test Cases

| Case | GitHub Username | JD Type | Coverage Type |
|------|----------------|---------|---------------|
| 01 | `torvalds` | Senior Systems Engineer — Linux Kernel | Strong technical fit |
| 02 | `gvanrossum` | Senior Python Engineer — Language Tooling | Partial fit (account age unverifiable) |
| 03 | `sindresorhus` | Senior Frontend Engineer — Node.js/npm | Strong technical fit (JS) |
| 04 | `tiangolo` | Senior Python Backend — FastAPI | Strong technical fit (Python) |
| 05 | `antirez` | Senior Python Engineer — ML Pipelines | Clear mismatch (C expert, not Python) |
| 06 | `tj` | Senior Go Engineer — Cloud-Native | Partial fit (soft skills unverifiable) |
| 07 | `wiseman-umanah` | Senior Python Engineer — Backend | **Engineered contradicted case** |
| 08 | `yyx990803` | Senior Frontend — Vue.js | Strong fit for Vue; would mismatch Go |
| 09 | `kennethreitz` | Senior Python — HTTP Libraries | Overclaiming-inviting profile |
| 10 | `jessfraz` | Senior Go — Container Security | Partial fit (employment history unverifiable) |

---

## Claim Coverage Summary

| Classification | Count | Cases |
|----------------|-------|-------|
| `verified` | 26 | 01,02,03,04,05,06,08,09,10 |
| `contradicted` | 5 | 05 (×3), 07 (×2) |
| `unverifiable` | 21 | all cases |
| **Total** | **52** | |

---

## Engineered Contradicted Case (Case 07) ⚠️

This case is mandatory evidence for the video and evaluation table.

**Setup required before running the pipeline:**

1. Open [`test_cases/case_07_engineered_contradicted/github_username.txt`](./test_cases/case_07_engineered_contradicted/github_username.txt)
2. Replace `new-dev-2024` with a real public GitHub account that:
   - Was created in **2023 or 2024** (< 18 months old at time of eval run)
   - Has at least a few Python repos (to make it plausible it could be mis-labelled)
   - Has **no public activity predating 2023** (no older forks, no earlier commits)
3. Update the corresponding labels in `ground_truth.json` for `c07-001` and `c07-003`
   if the account age changes the exact contradiction signal.

**What the verifier must demonstrate:**

```
Claim:    "7+ years professional Python experience"
Evidence: GitHub account created 2024-xx-xx (< 1 year old)
Rule:     check_date_arithmetic → CONTRADICTED (confidence: 0.95)
LLM:      does not run (rule layer finalised this)
```

This trajectory must appear in the submission's agent trajectory logs.

---

## Label Methodology

Every label was assigned **before** any pipeline code ran against these profiles.
Labels are based on manual inspection of each GitHub profile as of the labelling date.

**Classification criteria used:**

- `verified` — a specific, named public artefact (repo, language stat, README excerpt)
  directly and unambiguously supports the claim. No inference required.
- `contradicted` — a specific artefact directly conflicts with the claim
  (e.g. zero Go bytes in language stats for an "expert Go" claim).
- `unverifiable` — the claim is either a soft skill, an employment-context assertion
  (years at a company, team leadership, professional role), or otherwise not
  determinable from public GitHub data alone. This is NOT an error — it is the
  expected outcome for ~40% of JD claims.

---

## Running the Evaluation

```bash
# Run DevFit on all cases
uv run python eval/run_all.py --system devfit --output eval/devfit_outputs/

# Run baseline on all cases
uv run python eval/run_all.py --system baseline --output eval/baseline_outputs/

# Score results
uv run python eval/score.py \
    --ground-truth eval/ground_truth.json \
    --devfit-outputs eval/devfit_outputs/ \
    --baseline-outputs eval/baseline_outputs/
```
