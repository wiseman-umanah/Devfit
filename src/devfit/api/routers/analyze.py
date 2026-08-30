"""
Analyze router — ``POST /api/v1/analyze``.

This endpoint is a thin HTTP wrapper around the same pipeline used by the
CLI.  All heavy lifting happens in the pipeline and verifier sub-packages.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["analyze"])


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
        Unique identifier for this pipeline run.
    fit_report : str
        Markdown-formatted fit report.
    cv : str
        Markdown-formatted tailored CV.
    evidence_appendix : str
        Markdown-formatted evidence appendix.
    """

    run_id: str
    fit_report: str
    cv: str
    evidence_appendix: str


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    status_code=status.HTTP_200_OK,
    summary="Run the full DevFit pipeline",
)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Run the full DevFit verification pipeline and return results.

    This endpoint is asynchronous — the pipeline uses ``asyncio.gather``
    internally for concurrent GitHub API requests.

    Parameters
    ----------
    request : AnalyzeRequest
        The incoming request body.

    Returns
    -------
    AnalyzeResponse
        Fit report, CV, and evidence appendix for the given JD + username.

    Raises
    ------
    HTTPException
        ``422`` if request validation fails (handled by FastAPI automatically).
        ``503`` if the GitHub API is rate-limited.
        ``500`` for unexpected pipeline errors.
    """
    # TODO(stage-5): wire in the full pipeline once Stages 3-5 are complete.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Pipeline not yet implemented.  See steps.txt Stage 5.",
    )
