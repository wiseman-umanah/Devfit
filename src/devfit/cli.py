"""
CLI entry points for DevFit.

Commands
--------
``devfit``
    Production entry point.  Runs the full DevFit pipeline.

``devfit-dev``
    Development entry point.  Sets ``LOG_LEVEL=DEBUG`` and
    ``DEVFIT_ENV=development`` before delegating to the same pipeline,
    providing verbose output useful during active development.

Usage
-----
.. code-block:: bash

    devfit --jd jd.txt --github torvalds
    devfit --jd jd.txt --github torvalds --resume resume.txt
    devfit-dev --jd jd.txt --github torvalds

Both commands are registered as ``[project.scripts]`` in ``pyproject.toml``
so they are available directly after ``uv sync``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """
    Configure the root logger with a clean format.

    Parameters
    ----------
    level : str
        Python logging level string, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


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
        Parsed arguments with attributes: ``jd``, ``github``,
        ``resume``, ``output``, ``include_unverifiable``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="devfit",
        description=(
            "Generate an evidence-grounded CV and fit report.\n"
            "Every claim is verified against public GitHub artefacts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--jd",
        required=True,
        metavar="FILE_OR_TEXT",
        help="Path to a JD file, or inline JD text.",
    )
    parser.add_argument(
        "--github",
        required=True,
        metavar="USERNAME",
        help="Public GitHub username of the candidate.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="FILE",
        help="Optional path to a resume file (plain text or Markdown).",
    )
    parser.add_argument(
        "--output",
        default="output",
        metavar="DIR",
        help="Directory to write final output files (default: ./output).",
    )
    parser.add_argument(
        "--include-unverifiable",
        action="store_true",
        default=False,
        help=(
            "Include unverifiable claims in the CV with a visible marker.  "
            "Disabled by default."
        ),
    )
    return parser.parse_args(argv)


def _read_input(path_or_text: str) -> str:
    """
    Return text from a file path or pass through the string directly.

    Parameters
    ----------
    path_or_text : str
        Either a filesystem path to a readable text file, or inline text.

    Returns
    -------
    str
        The resolved text content.
    """
    candidate = Path(path_or_text)
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return path_or_text


async def _run_pipeline(
    jd_text: str,
    github_username: str,
    resume_text: str | None,
    output_dir: Path,
    include_unverifiable: bool,
) -> None:
    """
    Full end-to-end pipeline runner.

    Stages in order
    ---------------
    3 → GitHub Collector (ArtefactBundle)
    4 → JD Analyzer + optional Resume Analyzer (list[Claim])
    5 → Evidence Matcher + First-Pass Classifier (draft Verdicts)
    6 → Independent Verifier (final Verdicts + VerifierDecisions)
    7 → Fit Report + CV + Evidence Appendix generators
    8 → Human Checkpoint (approve / edit / abort)
    8 → Write final output files

    Parameters
    ----------
    jd_text : str
        Full job description text.
    github_username : str
        Candidate's public GitHub username.
    resume_text : str | None
        Optional resume text for additional claim cross-checking.
    output_dir : Path
        Parent directory; a ``<run_id>/`` sub-directory is created here.
    include_unverifiable : bool
        When ``True``, unverifiable claims are included in the CV with a
        visible ``[CANNOT BE CONFIRMED FROM GITHUB]`` marker.
    """
    from devfit.checkpoint import HumanCheckpoint
    from devfit.github.client import GitHubClient, GitHubRateLimitError
    from devfit.github.collector import GitHubCollector
    from devfit.output import (
        CVGenerator,
        EvidenceAppendix,
        FitReportGenerator,
        TrajectoryLogger,
    )
    from devfit.pipeline import (
        EvidenceMatcher,
        FirstPassClassifier,
        JDAnalyzer,
        ResumeAnalyzer,
    )
    from devfit.verifier import IndependentVerifier

    run_id = uuid.uuid4().hex[:8]
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Run ID: %s  →  %s", run_id, run_dir)

    with TrajectoryLogger(run_dir) as tlog:
        tlog.log_event(
            "pipeline_start",
            {
                "run_id": run_id,
                "github_username": github_username,
                "has_resume": resume_text is not None,
                "include_unverifiable": include_unverifiable,
            },
        )

        # ── Stage 3: GitHub Collector ────────────────────────────────────────
        logger.info("[3/8] Collecting GitHub artefacts for '%s'", github_username)
        try:
            async with GitHubClient() as client:
                bundle = await GitHubCollector(client).collect(github_username)
        except GitHubRateLimitError as exc:
            logger.error("GitHub rate limit hit: %s", exc)
            sys.exit(1)
        tlog.log_event(
            "github_collection",
            {"username": github_username, "artefact_count": len(bundle)},
        )

        # ── Stage 4: JD + Resume Analyzers ──────────────────────────────────
        logger.info("[4/8] Analyzing JD (%d chars)", len(jd_text))
        jd_claims = await JDAnalyzer().analyze(jd_text)
        tlog.log_event("jd_analysis", {"claim_count": len(jd_claims)})

        resume_claims = []
        if resume_text:
            logger.info("[4/8] Analyzing resume (%d chars)", len(resume_text))
            resume_claims = await ResumeAnalyzer().analyze(resume_text)
            tlog.log_event("resume_analysis", {"claim_count": len(resume_claims)})

        all_claims = jd_claims + resume_claims
        claims_by_id = {c.id: c.text for c in all_claims}

        # ── Stage 5: Evidence Matcher + First-Pass Classifier ────────────────
        logger.info(
            "[5/8] Matching evidence and classifying %d claims", len(all_claims)
        )
        matcher = EvidenceMatcher()
        matched = matcher.match(all_claims, bundle)
        skip_verdicts = matcher.build_unverifiable_verdicts(matched)
        tlog.log_event(
            "evidence_matching",
            {
                "total_claims": len(all_claims),
                "skipped": len(skip_verdicts),
                "to_classify": sum(1 for mc in matched if not mc.skipped),
            },
        )

        draft_verdicts = await FirstPassClassifier().classify(matched)
        tlog.log_event(
            "first_pass_classification",
            {"draft_verdict_count": len(draft_verdicts)},
        )

        # ── Stage 6: Independent Verifier ────────────────────────────────────
        logger.info(
            "[6/8] Running independent verifier on %d drafts", len(draft_verdicts)
        )
        active_matched = [mc for mc in matched if not mc.skipped]
        final_verdicts, decisions = await IndependentVerifier().verify_all(
            draft_verdicts, active_matched
        )
        all_verdicts = skip_verdicts + final_verdicts
        tlog.log_verifier_decisions(decisions)
        tlog.log_event(
            "verification_complete",
            {
                "total_verdicts": len(all_verdicts),
                "verified": sum(
                    1 for v in all_verdicts if v.classification == "verified"
                ),
                "contradicted": sum(
                    1 for v in all_verdicts if v.classification == "contradicted"
                ),
                "unverifiable": sum(
                    1 for v in all_verdicts if v.classification == "unverifiable"
                ),
                "downgraded_count": sum(1 for d in decisions if d.was_downgraded),
            },
        )

        # ── Stage 7: Report + CV + Appendix generators ───────────────────────
        logger.info("[7/8] Generating report, CV, and evidence appendix")
        jd_title = jd_text.splitlines()[0][:80] if jd_text else "Role"

        report_md = FitReportGenerator().generate(
            all_verdicts, claims_by_id, github_username, jd_title
        )
        cv_md, _ = CVGenerator().generate(
            all_verdicts, claims_by_id, github_username, include_unverifiable
        )
        appendix_md = EvidenceAppendix().generate(all_verdicts, claims_by_id)

        # ── Stage 8: Human Checkpoint ─────────────────────────────────────────
        logger.info("[8/8] Human checkpoint — presenting draft for review")
        result = HumanCheckpoint(tlog).run(report_md, cv_md)

        if result is None:
            logger.info("Aborted at human checkpoint.  No output files written.")
            sys.exit(0)

        final_report_md, final_cv_md = result

        # ── Write final output files ──────────────────────────────────────────
        (run_dir / "fit_report.md").write_text(final_report_md, encoding="utf-8")
        (run_dir / "cv.md").write_text(final_cv_md, encoding="utf-8")
        (run_dir / "evidence_appendix.md").write_text(appendix_md, encoding="utf-8")

        tlog.log_event(
            "output_written",
            {
                "run_dir": str(run_dir),
                "files": ["fit_report.md", "cv.md", "evidence_appendix.md",
                          "trajectory_log.jsonl"],
            },
        )

    logger.info("Done. Output written to: %s", run_dir)
    print(f"\nOutput written to: {run_dir}")


def _main(dev_mode: bool = False) -> None:
    """
    Shared entry-point logic for both ``devfit`` and ``devfit-dev``.

    Parameters
    ----------
    dev_mode : bool
        When ``True``, overrides ``LOG_LEVEL`` to ``DEBUG`` and sets
        ``DEVFIT_ENV=development`` before loading settings.
    """
    if dev_mode:
        os.environ.setdefault("LOG_LEVEL", "DEBUG")
        os.environ.setdefault("DEVFIT_ENV", "development")

    from devfit.config import get_settings

    settings = get_settings()
    _configure_logging(settings.log_level)

    args = _parse_args()
    jd_text = _read_input(args.jd)
    resume_text = _read_input(args.resume) if args.resume else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(
            _run_pipeline(
                jd_text=jd_text,
                github_username=args.github,
                resume_text=resume_text,
                output_dir=output_dir,
                include_unverifiable=args.include_unverifiable,
            )
        )
    except KeyboardInterrupt:
        logger.info("Aborted by user.")
        sys.exit(130)


def main() -> None:
    """
    Production CLI entry point (``devfit`` command).

    Registered in ``pyproject.toml`` under ``[project.scripts]``.
    """
    _main(dev_mode=False)


def main_dev() -> None:
    """
    Development CLI entry point (``devfit-dev`` command).

    Sets ``LOG_LEVEL=DEBUG`` and ``DEVFIT_ENV=development`` before running
    the pipeline, providing verbose output useful during active development.

    Registered in ``pyproject.toml`` under ``[project.scripts]``.
    """
    _main(dev_mode=True)
