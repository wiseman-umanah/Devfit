"""
JD Analyzer and Resume Analyzer for the DevFit pipeline.

Responsibilities
----------------
``JDAnalyzer``
    Calls the Groq LLM with the JD text and the ``jd_analyzer.txt`` prompt.
    Returns a list of atomic ``Claim`` objects, each tagged by category.
    Automatically pre-flags ``SOFT_SKILL`` and ``EXPERIENCE_DURATION`` claims
    as ``likely_unverifiable=True`` so the Evidence Matcher skips retrieval.

``ResumeAnalyzer``
    Same extraction logic as ``JDAnalyzer`` but applied to resume text.
    Returns claims with ``source=ClaimSource.RESUME``.
    Resume analysis is optional — the pipeline runs identically without it.

Both classes are async-first and use ``httpx`` via the ``groq`` SDK internally.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from groq import AsyncGroq

from devfit.config import get_settings
from devfit.schema import Claim, ClaimCategory, ClaimSource

logger = logging.getLogger(__name__)

# Path to prompt templates — resolved relative to this file so the package
# works correctly regardless of the working directory.
_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Categories that are almost always unverifiable from public GitHub data.
_LIKELY_UNVERIFIABLE_CATEGORIES = frozenset(
    [ClaimCategory.SOFT_SKILL, ClaimCategory.EXPERIENCE_DURATION]
)


def _load_prompt(name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Parameters
    ----------
    name : str
        Filename without extension, e.g. ``"jd_analyzer"``.

    Returns
    -------
    str
        Raw prompt template string with ``{placeholder}`` slots.

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_claims_response(
    raw_json: str,
    source: ClaimSource,
    id_prefix: str,
) -> list[Claim]:
    """
    Parse the LLM's JSON array response into a list of ``Claim`` objects.

    Assigns stable IDs using ``id_prefix`` + a zero-padded counter.
    Applies the ``likely_unverifiable`` auto-flag based on category.

    Parameters
    ----------
    raw_json : str
        Raw string returned by the LLM — expected to be a JSON array.
    source : ClaimSource
        Source to assign to every parsed claim.
    id_prefix : str
        Short prefix for generated claim IDs, e.g. ``"jd"`` or ``"cv"``.

    Returns
    -------
    list[Claim]
        Validated ``Claim`` objects.  Empty list on parse failure (logged).
    """
    # Strip markdown code fences the LLM sometimes wraps responses in
    text = raw_json.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (```json or ```) and closing fence
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )

    try:
        raw_list: list[dict[str, object]] = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM claims response as JSON: %s", exc)
        logger.debug("Raw response was: %s", raw_json[:500])
        return []

    claims: list[Claim] = []
    for idx, item in enumerate(raw_list, start=1):
        try:
            category = ClaimCategory(str(item.get("category", "other")))
        except ValueError:
            category = ClaimCategory.OTHER

        likely_unverifiable = bool(item.get("likely_unverifiable", False))
        # Always enforce the flag for known-unverifiable categories
        if category in _LIKELY_UNVERIFIABLE_CATEGORIES:
            likely_unverifiable = True

        claim_id = f"{id_prefix}-{idx:03d}"
        claims.append(
            Claim(
                id=claim_id,
                text=str(item.get("text", "")),
                source=source,
                category=category,
                likely_unverifiable=likely_unverifiable,
            )
        )

    logger.debug("Parsed %d claims (source=%s)", len(claims), source)
    return claims


class JDAnalyzer:
    """
    Extract atomic, checkable claims from a job description via Groq.

    Each call creates a new Groq async client using the API key from
    ``get_settings()``.  Reuse a single ``JDAnalyzer`` instance within
    a pipeline run to avoid re-loading the prompt on every call.

    Examples
    --------
    >>> analyzer = JDAnalyzer()
    >>> claims = await analyzer.analyze(jd_text)
    """

    def __init__(self) -> None:
        """Initialise the analyzer, loading the prompt template once."""
        self._prompt_template = _load_prompt("jd_analyzer")
        settings = get_settings()
        self._client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value()
        )
        self._model = settings.groq_model

    async def analyze(self, jd_text: str) -> list[Claim]:
        """
        Extract atomic claims from a raw job description.

        Calls the Groq LLM with the ``jd_analyzer`` prompt.  The LLM must
        return a JSON array — any non-JSON response logs an error and returns
        an empty list rather than raising, so the pipeline can continue with
        whatever claims were successfully extracted from other sources.

        Parameters
        ----------
        jd_text : str
            Raw job description text.

        Returns
        -------
        list[Claim]
            Atomic ``Claim`` objects with ``source=jd_requirement``.
            ``SOFT_SKILL`` and ``EXPERIENCE_DURATION`` claims are
            automatically flagged ``likely_unverifiable=True``.
        """
        logger.info("Analyzing JD (%d chars)", len(jd_text))
        prompt = self._prompt_template.format(jd_text=jd_text)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,  # deterministic for consistent claim extraction
            max_tokens=4096,
        )

        raw = response.choices[0].message.content or ""
        claims = _parse_claims_response(
            raw,
            source=ClaimSource.JD_REQUIREMENT,
            id_prefix=f"jd-{uuid.uuid4().hex[:6]}",
        )
        logger.info("JD analysis produced %d claims", len(claims))
        return claims


class ResumeAnalyzer:
    """
    Extract atomic, checkable claims from optional resume text via Groq.

    Identical extraction logic to ``JDAnalyzer`` but applied to resume content.
    Resume analysis is **optional** — the pipeline runs without it.  When a
    resume is provided, DevFit cross-checks its claims against GitHub evidence
    in addition to the JD requirements.

    Examples
    --------
    >>> analyzer = ResumeAnalyzer()
    >>> claims = await analyzer.analyze(resume_text)
    """

    def __init__(self) -> None:
        """Initialise the analyzer, loading the resume prompt template once."""
        self._prompt_template = _load_prompt("resume_analyzer")
        settings = get_settings()
        self._client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value()
        )
        self._model = settings.groq_model

    async def analyze(self, resume_text: str) -> list[Claim]:
        """
        Extract atomic claims from resume text.

        Parameters
        ----------
        resume_text : str
            Raw resume text (plain text or Markdown).

        Returns
        -------
        list[Claim]
            Atomic ``Claim`` objects with ``source=ClaimSource.RESUME``.
            ``SOFT_SKILL`` and ``EXPERIENCE_DURATION`` claims are
            automatically flagged ``likely_unverifiable=True``.
        """
        logger.info("Analyzing resume (%d chars)", len(resume_text))
        prompt = self._prompt_template.format(resume_text=resume_text)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=4096,
        )

        raw = response.choices[0].message.content or ""
        claims = _parse_claims_response(
            raw,
            source=ClaimSource.RESUME,
            id_prefix=f"cv-{uuid.uuid4().hex[:6]}",
        )
        logger.info("Resume analysis produced %d claims", len(claims))
        return claims
