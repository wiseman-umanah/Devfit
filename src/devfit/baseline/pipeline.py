"""
Baseline pipeline — single Groq prompt, no artefact bundle, no verification.

This module exists purely as a comparison point for the evaluation.  It
deliberately omits every quality safeguard DevFit provides: no claim
extraction, no evidence matching, no rule layer, no LLM verifier, and no
artefact pointers in the output.  The output is whatever the LLM produces in
one unconstrained generation.

Do NOT improve this pipeline.  Its purpose is to demonstrate the hallucination
problem that DevFit solves.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from groq import AsyncGroq

from devfit.config import get_settings
from devfit.github.bundle import ArtefactBundle
from devfit.schema import ArtefactType

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Parameters
    ----------
    name : str
        Filename without extension, e.g. ``"baseline"``.

    Returns
    -------
    str
        Raw prompt template string.

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def build_github_summary(bundle: ArtefactBundle) -> str:
    """
    Convert an ``ArtefactBundle`` into an unstructured plain-text summary.

    This deliberately throws away the structured evidence that DevFit uses,
    producing instead a lossy paragraph the baseline LLM receives as context.
    No artefact pointers, no type tagging — just readable prose.

    Parameters
    ----------
    bundle : ArtefactBundle
        Structured evidence collected for a GitHub profile.

    Returns
    -------
    str
        A short, unstructured description of the GitHub profile.
    """
    lines: list[str] = []

    # Account metadata (username, account age, bio)
    for a in bundle.by_type(ArtefactType.ACCOUNT_METADATA):
        lines.append(f"GitHub user: {a.pointer}")
        if a.extracted_fact:
            lines.append(a.extracted_fact)

    # Language stats — top languages only
    lang_facts = [
        a.extracted_fact
        for a in bundle.by_type(ArtefactType.LANGUAGE_STATS)
        if a.extracted_fact
    ]
    if lang_facts:
        lines.append("Languages: " + "; ".join(lang_facts[:5]))

    # A few repo names
    repo_pointers = [
        a.pointer
        for a in bundle.by_type(ArtefactType.REPO)
        if a.pointer
    ][:5]
    if repo_pointers:
        lines.append("Repos (sample): " + ", ".join(repo_pointers))

    # Contribution graph summary
    for a in bundle.by_type(ArtefactType.CONTRIBUTION_GRAPH):
        if a.extracted_fact:
            lines.append(a.extracted_fact)
            break  # one line is enough

    return "\n".join(lines) if lines else "No GitHub data available."


@dataclass(frozen=True)
class BaselineResult:
    """
    Output of the baseline pipeline for a single run.

    Parameters
    ----------
    cv_markdown : str
        Raw Markdown CV produced by the LLM with no verification.
    fit_comment : str
        Short fit comment (2–3 sentences) from the LLM.
    raw_response : str
        Full unprocessed LLM response for traceability.
    """

    cv_markdown: str
    fit_comment: str
    raw_response: str


class BaselinePipeline:
    """
    Single-prompt baseline system used as the hallucination benchmark.

    Calls Groq once per run.  Receives only a JD text and an unstructured
    GitHub profile summary (no artefact bundle, no evidence pointers).

    Examples
    --------
    >>> pipeline = BaselinePipeline()
    >>> result = await pipeline.run(jd_text, bundle, resume_text=None)
    >>> print(result.cv_markdown)
    """

    def __init__(self) -> None:
        """Initialise the pipeline, loading the prompt template once."""
        self._prompt_template = _load_prompt("baseline")
        settings = get_settings()
        self._client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value()
        )
        self._model = settings.groq_model

    async def run(
        self,
        jd_text: str,
        bundle: ArtefactBundle,
        resume_text: str | None = None,
    ) -> BaselineResult:
        """
        Run the baseline pipeline.

        Parameters
        ----------
        jd_text : str
            Full job description text.
        bundle : ArtefactBundle
            Collected GitHub artefacts (converted to unstructured summary).
        resume_text : str | None
            Optional candidate resume.  When supplied it is appended to the
            prompt under a ``Candidate resume`` heading.

        Returns
        -------
        BaselineResult
            Raw LLM output split into CV and fit comment sections.
        """
        github_summary = build_github_summary(bundle)

        if resume_text:
            resume_section = f"Candidate resume:\n{resume_text}"
        else:
            resume_section = ""

        prompt = self._prompt_template.format(
            jd_text=jd_text,
            github_summary=github_summary,
            resume_section=resume_section,
        )

        logger.info("Baseline: sending single prompt to %s", self._model)

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ""
        logger.debug("Baseline raw response (%d chars)", len(raw))

        # Split on --- separator; everything before is CV, after is fit comment
        parts = raw.split("---", maxsplit=1)
        cv_markdown = parts[0].strip()
        fit_comment = parts[1].strip() if len(parts) > 1 else ""

        return BaselineResult(
            cv_markdown=cv_markdown,
            fit_comment=fit_comment,
            raw_response=raw,
        )
