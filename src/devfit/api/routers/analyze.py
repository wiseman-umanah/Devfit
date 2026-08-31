"""
Analyze router -- ``POST /api/v1/analyze``.

Runs the full DevFit JD-fit pipeline: GitHub collection, claim analysis,
evidence matching, independent verification, and report generation.

This endpoint is the HTTP equivalent of the ``devfit-fit`` CLI command.
The human checkpoint step is skipped; all output is returned as JSON.
Nothing is written to disk.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["analyze"])
logger = logging.getLogger(__name__)


class AnalyzeRequest(BaseModel):
    """
    Request body for the analyze endpoint.

    Parameters
    ----------
    jd_text : str
        The full job description text.
    github_username : str
        The candidate's public GitHub username.
    resume_text : str | None
        Optional resume text.  When provided, DevFit additionally
        cross-checks resume claims against GitHub evidence.
    include_unverifiable : bool
        When ``True``, unverifiable claims are included in the CV output
        with a prominent marker.  Defaults to ``False``.
    """

    jd_text: str = Field(..., min_length=50)
    github_username: str = Field(..., min_length=1, max_length=39)
    resume_text: str | None = None
    include_unverifiable: bool = False


class AnalyzeResponse(BaseModel):
    """
    Response body for the analyze endpoint.

    Parameters
    ----------
    run_id : str
        Unique identifier for this pipeline run (8-char hex prefix).
    fit_report : str
        Markdown-formatted fit report.
    cv : str
        Markdown-formatted tailored CV.
    evidence_appendix : str
        Markdown-formatted evidence appendix.
    improvements : str
        Markdown-formatted improvement suggestions.
    verified_count : int
        Number of claims verified against GitHub artefacts.
    contradicted_count : int
        Number of claims contradicted by GitHub artefacts.
    unverifiable_count : int
        Number of claims that could not be verified.
    """

    run_id: str
    fit_report: str
    cv: str
    evidence_appendix: str
    improvements: str
    verified_count: int
    contradicted_count: int
    unverifiable_count: int


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the full DevFit JD-fit pipeline",
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the full DevFit verification pipeline and return results as JSON.

    Stages in order: GitHub collection -> JD analysis -> optional resume
    analysis -> evidence matching -> first-pass classification ->
    independent verification -> report + CV + appendix + improvements.

    The human checkpoint step is skipped; all output is returned directly.
    Nothing is written to disk.

    Parameters
    ----------
    request : AnalyzeRequest
        The incoming request body.

    Returns
    -------
    AnalyzeResponse
        Fit report, CV, evidence appendix, improvements, and verdict counts.

    Raises
    ------
    HTTPException
        ``503`` if the GitHub API is rate-limited.
        ``500`` for unexpected pipeline errors.
    """
    import uuid

    from devfit.github.client import GitHubClient, GitHubRateLimitError
    from devfit.github.collector import GitHubCollector
    from devfit.output import (
        CVGenerator,
        EvidenceAppendix,
        FitReportGenerator,
        ImprovementGenerator,
    )
    from devfit.pipeline import (
        EvidenceMatcher,
        FirstPassClassifier,
        JDAnalyzer,
        ResumeAnalyzer,
    )
    from devfit.verifier import IndependentVerifier

    run_id = uuid.uuid4().hex[:8]

    # ── Stage 1: GitHub Collector ────────────────────────────────────────────
    try:
        async with GitHubClient() as client:
            bundle = await GitHubCollector(client).collect(request.github_username)
    except GitHubRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"GitHub rate limit exceeded: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("GitHub collection failed for '%s'", request.github_username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to collect GitHub data: {exc}",
        ) from exc

    # ── Stage 2: JD + optional resume analysis ───────────────────────────────
    try:
        jd_claims = await JDAnalyzer().analyze(request.jd_text)
        resume_claims = []
        if request.resume_text:
            resume_claims = await ResumeAnalyzer().analyze(request.resume_text)
        all_claims = jd_claims + resume_claims
        claims_by_id = {c.id: c.text for c in all_claims}
        claims_by_id_to_category = {c.id: c.category for c in all_claims}

        # ── Stage 3: Evidence matching + first-pass classification ───────────
        matcher = EvidenceMatcher()
        matched = matcher.match(all_claims, bundle)
        skip_verdicts = matcher.build_unverifiable_verdicts(matched)
        draft_verdicts = await FirstPassClassifier().classify(matched)

        # ── Stage 4: Independent verification ───────────────────────────────
        active_matched = [mc for mc in matched if not mc.skipped]
        final_verdicts, _ = await IndependentVerifier().verify_all(
            draft_verdicts, active_matched
        )
        all_verdicts = skip_verdicts + final_verdicts

        # ── Stage 5: Report + CV + appendix + improvements ──────────────────
        jd_title = request.jd_text.splitlines()[0][:80]
        report_md = FitReportGenerator().generate(
            all_verdicts, claims_by_id, request.github_username, jd_title
        )
        cv_md, _ = await CVGenerator().generate(
            all_verdicts,
            claims_by_id,
            request.github_username,
            request.include_unverifiable,
            jd_title=jd_title,
            bundle=bundle,
        )
        appendix_md = EvidenceAppendix().generate(all_verdicts, claims_by_id)
        improvements_md = await ImprovementGenerator().generate(
            all_verdicts,
            claims_by_id,
            claims_by_id_to_category,
            jd_title=jd_title,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed for '%s'", request.github_username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pipeline failed: {exc}",
        ) from exc

    return AnalyzeResponse(
        run_id=run_id,
        fit_report=report_md,
        cv=cv_md,
        evidence_appendix=appendix_md,
        improvements=improvements_md,
        verified_count=sum(1 for v in all_verdicts if v.classification == "verified"),
        contradicted_count=sum(
            1 for v in all_verdicts if v.classification == "contradicted"
        ),
        unverifiable_count=sum(
            1 for v in all_verdicts if v.classification == "unverifiable"
        ),
    )
