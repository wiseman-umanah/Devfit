"""
Evaluation runner — executes DevFit and Baseline pipelines on all 10 test cases.

Saves per-case output JSON files to:
    eval/devfit_outputs/<case_id>.json
    eval/baseline_outputs/<case_id>.json

Each output file contains:
    {
        "case_id": "case_01_strong_fit",
        "username": "torvalds",
        "verdicts": [ {claim_id, classification, evidence: [...]}, ... ],
        "cv_claims": [ {text, artefact_pointer, is_unverifiable}, ... ]
    }

Usage
-----
.. code-block:: bash

    uv run python eval/run_eval.py [--cases case_01] [--devfit-only] [--baseline-only]

Notes
-----
- Requires GROQ_API_KEY and (optionally) GITHUB_TOKEN in the environment.
- Skips the human checkpoint — batch mode runs non-interactively.
- On GitHub rate-limit error the case is skipped and marked as failed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_EVAL_DIR = Path(__file__).parent
_TEST_CASES_DIR = _EVAL_DIR / "test_cases"
_DEVFIT_OUTPUTS_DIR = _EVAL_DIR / "devfit_outputs"
_BASELINE_OUTPUTS_DIR = _EVAL_DIR / "baseline_outputs"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list.  Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    argparse.Namespace
        Parsed namespace with attributes: ``cases``, ``devfit_only``,
        ``baseline_only``.
    """
    parser = argparse.ArgumentParser(
        description="Run DevFit and Baseline pipelines on all eval test cases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        metavar="CASE_ID",
        default=None,
        help=(
            "Optional subset of case directory names to run, "
            "e.g. --cases case_01_strong_fit case_07_engineered_contradicted. "
            "Defaults to all cases."
        ),
    )
    parser.add_argument(
        "--devfit-only",
        action="store_true",
        default=False,
        help="Run only the DevFit pipeline; skip Baseline.",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        default=False,
        help="Run only the Baseline pipeline; skip DevFit.",
    )
    return parser.parse_args(argv)


def _discover_cases(subset: list[str] | None) -> list[Path]:
    """
    Return sorted list of test-case directories to evaluate.

    Parameters
    ----------
    subset : list[str] | None
        Case directory names to include, or ``None`` for all.

    Returns
    -------
    list[Path]
        Sorted list of existing test-case directories.
    """
    if subset:
        dirs = [_TEST_CASES_DIR / name for name in subset]
        missing = [d for d in dirs if not d.is_dir()]
        if missing:
            logger.error("Unknown test cases: %s", [d.name for d in missing])
            sys.exit(1)
        return dirs
    return sorted(d for d in _TEST_CASES_DIR.iterdir() if d.is_dir())


def _serialize_verdicts(
    verdicts: list[object],
) -> list[dict[str, object]]:
    """
    Serialise a list of ``Verdict`` Pydantic models to plain dicts.

    Parameters
    ----------
    verdicts : list[object]
        ``Verdict`` model instances.

    Returns
    -------
    list[dict[str, object]]
        JSON-serialisable dicts with ``claim_id``, ``classification``,
        and ``evidence`` (list of artefact pointer strings).
    """
    result: list[dict[str, object]] = []
    for v in verdicts:
        evidence_pointers: list[str] = []
        for artefact in getattr(v, "evidence", []):
            ptr = getattr(artefact, "pointer", "")
            if ptr:
                evidence_pointers.append(ptr)
        result.append(
            {
                "claim_id": getattr(v, "claim_id", ""),
                "classification": str(getattr(v, "classification", "")),
                "evidence": evidence_pointers,
            }
        )
    return result


def _serialize_cv_lines(cv_lines: list[object]) -> list[dict[str, object]]:
    """
    Serialise a list of ``CVLine`` objects to plain dicts.

    Parameters
    ----------
    cv_lines : list[object]
        ``CVLine`` instances from the ``CVGenerator``.

    Returns
    -------
    list[dict[str, object]]
        JSON-serialisable dicts with ``text``, ``artefact_pointer``,
        and ``is_unverifiable``.
    """
    return [
        {
            "text": getattr(cl, "text", ""),
            "artefact_pointer": getattr(cl, "artefact_pointer", "") or "",
            "is_unverifiable": bool(getattr(cl, "is_unverifiable", False)),
        }
        for cl in cv_lines
    ]


async def _run_devfit_case(
    case_dir: Path,
    output_dir: Path,
) -> bool:
    """
    Run the DevFit pipeline (no human checkpoint) on a single test case.

    Parameters
    ----------
    case_dir : Path
        Directory containing ``jd.txt`` and ``github_username.txt``.
    output_dir : Path
        Directory to write the output JSON file.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any failure.
    """
    from devfit.github.client import GitHubClient, GitHubRateLimitError
    from devfit.github.collector import GitHubCollector
    from devfit.output import CVGenerator, FitReportGenerator
    from devfit.pipeline import (
        EvidenceMatcher,
        FirstPassClassifier,
        JDAnalyzer,
    )
    from devfit.verifier import IndependentVerifier

    case_id = case_dir.name
    jd_text = (case_dir / "jd.txt").read_text(encoding="utf-8")
    username = (case_dir / "github_username.txt").read_text(encoding="utf-8").strip()
    logger.info("[DevFit] %s (%s) — start", case_id, username)

    try:
        # Stage 3: collect GitHub artefacts
        async with GitHubClient() as client:
            bundle = await GitHubCollector(client).collect(username)
    except GitHubRateLimitError as exc:
        logger.error("[DevFit] %s — rate limit: %s", case_id, exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("[DevFit] %s — collection failed: %s", case_id, exc)
        return False

    # Stage 4: analyse JD
    try:
        jd_claims = await JDAnalyzer().analyze(jd_text)
    except Exception as exc:  # noqa: BLE001
        logger.error("[DevFit] %s — JD analysis failed: %s", case_id, exc)
        return False

    claims_by_id = {c.id: c.text for c in jd_claims}

    # Stage 5: match + classify
    matcher = EvidenceMatcher()
    matched = matcher.match(jd_claims, bundle)
    skip_verdicts = matcher.build_unverifiable_verdicts(matched)

    try:
        draft_verdicts = await FirstPassClassifier().classify(matched)
    except Exception as exc:  # noqa: BLE001
        logger.error("[DevFit] %s — classification failed: %s", case_id, exc)
        return False

    # Stage 6: verify
    active_matched = [mc for mc in matched if not mc.skipped]
    try:
        final_verdicts, _ = await IndependentVerifier().verify_all(
            draft_verdicts, active_matched
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[DevFit] %s — verification failed: %s", case_id, exc)
        return False

    all_verdicts = skip_verdicts + final_verdicts

    # Stage 7: generate CV + report (for eval metrics)
    jd_title = jd_text.splitlines()[0][:80] if jd_text else "Role"
    _, cv_lines = await CVGenerator().generate(
        all_verdicts, claims_by_id, username,
        jd_title=jd_title, bundle=bundle,
    )
    _ = FitReportGenerator().generate(all_verdicts, claims_by_id, username, jd_title)

    # Write output JSON (sync mkdir is fine — this runs in asyncio thread, no trio)
    await asyncio.get_event_loop().run_in_executor(
        None, lambda: output_dir.mkdir(parents=True, exist_ok=True)
    )
    out = {
        "case_id": case_id,
        "username": username,
        "verdicts": _serialize_verdicts(all_verdicts),
        "cv_claims": _serialize_cv_lines(cv_lines),
    }
    out_path = output_dir / f"{case_id}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("[DevFit] %s — done → %s", case_id, out_path)
    return True


async def _run_baseline_case(
    case_dir: Path,
    output_dir: Path,
) -> bool:
    """
    Run the Baseline pipeline on a single test case.

    Parameters
    ----------
    case_dir : Path
        Directory containing ``jd.txt`` and ``github_username.txt``.
    output_dir : Path
        Directory to write the output JSON file.

    Returns
    -------
    bool
        ``True`` on success, ``False`` on any failure.
    """
    from devfit.baseline.pipeline import BaselinePipeline
    from devfit.github.client import GitHubClient, GitHubRateLimitError
    from devfit.github.collector import GitHubCollector

    case_id = case_dir.name
    jd_text = (case_dir / "jd.txt").read_text(encoding="utf-8")
    username = (case_dir / "github_username.txt").read_text(encoding="utf-8").strip()
    logger.info("[Baseline] %s (%s) — start", case_id, username)

    try:
        async with GitHubClient() as client:
            bundle = await GitHubCollector(client).collect(username)
    except GitHubRateLimitError as exc:
        logger.error("[Baseline] %s — rate limit: %s", case_id, exc)
        return False
    except Exception as exc:  # noqa: BLE001
        logger.error("[Baseline] %s — collection failed: %s", case_id, exc)
        return False

    try:
        result = await BaselinePipeline().run(jd_text, bundle)
    except Exception as exc:  # noqa: BLE001
        logger.error("[Baseline] %s — LLM call failed: %s", case_id, exc)
        return False

    # Baseline has no structured verdicts — we emit a synthetic "verified"
    # verdict per non-empty CV line, all with empty evidence, to drive
    # the hallucination-rate metric.
    cv_lines_raw = [
        line.strip()
        for line in result.cv_markdown.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    synthetic_verdicts: list[dict[str, object]] = [
        {
            "claim_id": f"baseline-{i:03d}",
            "classification": "verified",
            "evidence": [],  # deliberate — no artefact pointers
        }
        for i, _ in enumerate(cv_lines_raw, start=1)
    ]

    await asyncio.get_event_loop().run_in_executor(
        None, lambda: output_dir.mkdir(parents=True, exist_ok=True)
    )
    out = {
        "case_id": case_id,
        "username": username,
        "verdicts": synthetic_verdicts,
        "cv_claims": [
            {"text": line, "artefact_pointer": "", "is_unverifiable": False}
            for line in cv_lines_raw
        ],
        "fit_comment": result.fit_comment,
    }
    out_path = output_dir / f"{case_id}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("[Baseline] %s — done → %s", case_id, out_path)
    return True


async def _main_async(args: argparse.Namespace) -> int:
    """
    Run all selected test cases.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments.

    Returns
    -------
    int
        Exit code: 0 on full success, 1 if any case failed.
    """
    cases = _discover_cases(args.cases)
    logger.info("Evaluating %d test case(s)", len(cases))

    failures: list[str] = []

    for case_dir in cases:
        if not args.baseline_only:
            ok = await _run_devfit_case(case_dir, _DEVFIT_OUTPUTS_DIR)
            if not ok:
                failures.append(f"devfit:{case_dir.name}")

        if not args.devfit_only:
            ok = await _run_baseline_case(case_dir, _BASELINE_OUTPUTS_DIR)
            if not ok:
                failures.append(f"baseline:{case_dir.name}")

    if failures:
        logger.error(
            "%d case(s) failed: %s",
            len(failures),
            ", ".join(failures),
        )
        return 1

    logger.info("All cases completed successfully.")
    return 0


def main() -> None:
    """
    Entry point for the eval runner.

    Parses arguments, configures logging, and drives the async evaluation loop.
    """
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    exit_code = asyncio.run(_main_async(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
