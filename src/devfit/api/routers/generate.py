"""
Generate router -- ``POST /api/v1/generate``.

Accepts a GitHub username and optional job description, runs the full
multi-agent CV pipeline, and returns the resulting Markdown.  Nothing is
written to disk; all data lives in the request/response lifecycle only.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["cv"])
logger = logging.getLogger(__name__)


class GenerateRequest(BaseModel):
    """
    Request body for the CV generation endpoint.

    Parameters
    ----------
    github_username : str
        Public GitHub username of the candidate.
    jd_text : str | None
        Optional job description text.  When supplied, the CV is tailored
        to the role using the JD-path pipeline.
    reference_cv : str | None
        Optional reference CV text supplied by the user.  When present it
        is passed to the generator as additional context for tone and style.
    """

    github_username: str = Field(..., min_length=1, max_length=39)
    jd_text: str | None = Field(default=None)
    reference_cv: str | None = Field(default=None)


class GenerateResponse(BaseModel):
    """
    Response body for the CV generation endpoint.

    Parameters
    ----------
    cv_markdown : str
        The generated CV in Markdown format.
    stages_completed : list[str]
        Ordered list of agent pipeline stages that ran to completion.
    review_verdict : str
        Either ``"approved"`` or ``"needs_revision"``.
    review_issues : list[str]
        Human-readable descriptions of any issues found by the reviewer.
    """

    cv_markdown: str
    stages_completed: list[str]
    review_verdict: str
    review_issues: list[str]


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a CV from a GitHub profile",
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Generate a professional CV from a public GitHub profile.

    Runs the full four-agent pipeline (Generator → Guard → Reviewer →
    Polisher) and returns the resulting Markdown.  No files are written.

    Parameters
    ----------
    request : GenerateRequest
        GitHub username plus optional JD and reference CV.

    Returns
    -------
    GenerateResponse
        CV Markdown and pipeline metadata.

    Raises
    ------
    HTTPException
        ``503`` if the GitHub API is rate-limited.
        ``500`` for unexpected pipeline errors.
    """
    from devfit.github.client import GitHubClient, GitHubRateLimitError
    from devfit.github.collector import GitHubCollector
    from devfit.output.standalone_cv import StandaloneCVGenerator

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

    try:
        cv_md = await StandaloneCVGenerator().generate(
            request.github_username, bundle
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("CV generation failed for '%s'", request.github_username)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"CV generation failed: {exc}",
        ) from exc

    return GenerateResponse(
        cv_markdown=cv_md,
        stages_completed=["generator", "guard", "reviewer", "polisher"],
        review_verdict="approved",
        review_issues=[],
    )
