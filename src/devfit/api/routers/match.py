"""
Match router -- ``POST /api/v1/match``.

Compares the current CV against a job description and returns a match
score, a list of matched skills, and a list of gaps with suggestions for
how to address each one.  Nothing is written to disk.
"""

from __future__ import annotations

import json
import logging
import re

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["cv"])
logger = logging.getLogger(__name__)

_MATCH_PROMPT = """\
You are a senior technical recruiter.
Analyse how well this CV matches the job description.

JOB DESCRIPTION:
{jd_text}

CANDIDATE CV:
{cv_text}

Return a JSON object with this exact structure:
{{
  "score": <integer 0-100>,
  "matched": [
    {{"skill": "<skill or requirement>", "evidence": "<where it appears in the CV>"}}
  ],
  "gaps": [
    {{
      "requirement": "<JD requirement not met>",
      "suggestion": "<specific, actionable suggestion to address this gap in the CV>"
    }}
  ],
  "summary": "<2-3 sentence overall assessment>"
}}

Rules:
- score 0-100 based on how many JD requirements are clearly met by the CV.
- matched: only list items that are genuinely evidenced in the CV text.
- gaps: only list genuine requirements from the JD that are absent or weak in the CV.
- suggestions must be specific: name the exact GitHub project or skill to highlight.
- Return ONLY the JSON. No preamble, no explanation.
"""


class MatchRequest(BaseModel):
    """
    Request body for the match endpoint.

    Parameters
    ----------
    cv_text : str
        The current CV in Markdown or plain text.
    jd_text : str
        The job description to match against.
    """

    cv_text: str = Field(..., min_length=50)
    jd_text: str = Field(..., min_length=50)


class MatchedItem(BaseModel):
    """A single matched skill or requirement."""

    skill: str
    evidence: str


class GapItem(BaseModel):
    """A single gap between the JD and the CV."""

    requirement: str
    suggestion: str


class MatchResponse(BaseModel):
    """
    Response body for the match endpoint.

    Parameters
    ----------
    score : int
        Match score from 0 (no match) to 100 (perfect match).
    matched : list[MatchedItem]
        Requirements from the JD that are met by the CV.
    gaps : list[GapItem]
        Requirements missing or weak in the CV, with suggestions.
    summary : str
        2-3 sentence overall assessment.
    """

    score: int
    matched: list[MatchedItem]
    gaps: list[GapItem]
    summary: str


def _parse_match_response(raw: str) -> MatchResponse:
    """
    Parse the LLM JSON response into a ``MatchResponse``.

    Strips markdown code fences.  Falls back to a minimal response on error.

    Parameters
    ----------
    raw : str
        Raw LLM output.

    Returns
    -------
    MatchResponse
        Parsed response or a safe fallback.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
        return MatchResponse(
            score=int(data.get("score", 0)),
            matched=[
                MatchedItem(
                    skill=str(m.get("skill", "")),
                    evidence=str(m.get("evidence", "")),
                )
                for m in data.get("matched", [])
                if isinstance(m, dict)
            ],
            gaps=[
                GapItem(
                    requirement=str(g.get("requirement", "")),
                    suggestion=str(g.get("suggestion", "")),
                )
                for g in data.get("gaps", [])
                if isinstance(g, dict)
            ],
            summary=str(data.get("summary", "")),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Match response parse failed: %s", exc)
        return MatchResponse(
            score=0,
            matched=[],
            gaps=[],
            summary="Could not parse match analysis. Please try again.",
        )


@router.post(
    "/match",
    response_model=MatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Match a CV against a job description",
)
async def match(request: MatchRequest) -> MatchResponse:
    """
    Compare a CV against a job description and return a structured match report.

    Parameters
    ----------
    request : MatchRequest
        CV text and job description text.

    Returns
    -------
    MatchResponse
        Score, matched items, gaps with suggestions, and overall summary.

    Raises
    ------
    HTTPException
        ``500`` if the LLM call fails.
    """
    from groq import AsyncGroq

    from devfit.config import get_settings

    settings = get_settings()
    prompt = _MATCH_PROMPT.format(
        jd_text=request.jd_text[:3000],
        cv_text=request.cv_text[:3000],
    )

    try:
        client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
        response = await client.chat.completions.create(
            model=settings.groq_model_reviewer,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ""
        await client.close()
    except Exception as exc:  # noqa: BLE001
        logger.exception("Match LLM call failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Match analysis failed: {exc}",
        ) from exc

    return _parse_match_response(raw)
