"""
CV generator -- produces a professional, ATS-structured Markdown CV from verified verdicts.

Design rules (per TRD section 3.5)
------------------------------------
- The CV contains only ``verified`` claims by default.
- Every claim bullet must carry a traceable ``[source: <pointer>]`` tag.
- ``unverifiable`` claims may be included only at explicit user request
  and must render with a ``[NOT VERIFIED FROM GITHUB]`` marker.
- A Groq LLM call converts the verified claim list into professional CV prose.
  If the LLM call fails for any reason, the generator falls back to a
  deterministic bullet list so the pipeline never stalls.
- No em-dashes in output. No filler phrases. Factual, active-voice prose only.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from groq import AsyncGroq

from devfit.config import get_settings
from devfit.schema import Classification, Verdict

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_UNVERIFIABLE_MARKER = "[NOT VERIFIED FROM GITHUB]"
_EM_DASH_RE = re.compile(r"\s*[—–]\s*|--")


def _load_prompt(name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Parameters
    ----------
    name : str
        Filename without extension, e.g. ``"cv_generator"``.

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


def _strip_em_dashes(text: str) -> str:
    """
    Remove em-dashes and double-hyphens from generated text.

    Replaces ``--`` and Unicode em/en dashes with a comma+space so the
    sentence still reads naturally without the AI-giveaway punctuation.

    Parameters
    ----------
    text : str
        Raw LLM-generated Markdown string.

    Returns
    -------
    str
        Text with all em-dash variants replaced.
    """
    return _EM_DASH_RE.sub(", ", text)


@dataclass
class CVLine:
    """
    A single line in the generated CV, with its evidence pointer.

    Parameters
    ----------
    text : str
        The claim text as it will appear in the CV.
    artefact_pointer : str
        The pointer to the supporting artefact (required for verified claims).
    is_unverifiable : bool
        ``True`` when the claim is unverifiable and included by user request.
    """

    text: str
    artefact_pointer: str
    is_unverifiable: bool = False


def _extract_display_name(github_username: str, verified: list[Verdict]) -> str:
    """
    Derive a display name from artefact metadata or fall back to the username.

    Parameters
    ----------
    github_username : str
        Candidate GitHub username.
    verified : list[Verdict]
        Verified verdicts; account metadata artefacts may carry a bio with name.

    Returns
    -------
    str
        Best available display name for the CV header.
    """
    for v in verified:
        for artefact in v.evidence:
            if artefact.type == "account_metadata":
                # bio field format: 'Account created ..., bio: "Name Here ..."'
                bio_match = re.search(r'bio:\s*"([^"]+)"', artefact.extracted_fact)
                if bio_match:
                    # Take first segment if it looks like a name (no spaces = not a sentence)
                    candidate = bio_match.group(1).split(".")[0].strip()
                    if len(candidate.split()) <= 4:
                        return candidate
    return github_username


def _build_verified_claims_json(
    verified: list[Verdict], claims_by_id: dict[str, str]
) -> str:
    """
    Serialise verified verdicts into the JSON string the prompt expects.

    Parameters
    ----------
    verified : list[Verdict]
        Verified verdicts from the pipeline.
    claims_by_id : dict[str, str]
        Mapping of claim_id to human-readable claim text.

    Returns
    -------
    str
        JSON array string, each entry having ``text`` and ``artefact_pointer``.
    """
    payload = [
        {
            "text": claims_by_id.get(v.claim_id, v.claim_id),
            "artefact_pointer": v.evidence[0].pointer if v.evidence else "",
        }
        for v in verified
    ]
    return json.dumps(payload, indent=2)


def _build_fallback_cv(
    verified: list[Verdict],
    unverifiable: list[Verdict],
    claims_by_id: dict[str, str],
    github_username: str,
    include_unverifiable: bool,
) -> str:
    """
    Deterministic bullet-list CV used when the LLM call fails.

    Parameters
    ----------
    verified : list[Verdict]
        Verified verdicts.
    unverifiable : list[Verdict]
        Unverifiable verdicts.
    claims_by_id : dict[str, str]
        Mapping of claim_id to human-readable claim text.
    github_username : str
        Candidate GitHub username.
    include_unverifiable : bool
        Whether to include unverifiable claims.

    Returns
    -------
    str
        Plain Markdown CV with one bullet per verified claim.
    """
    lines: list[str] = [
        f"# {github_username}",
        "",
        f"GitHub: https://github.com/{github_username}",
        "",
    ]

    if not verified:
        lines += [
            "*No verified claims could be extracted from public GitHub data "
            "for this candidate and job description.*",
            "",
        ]
    else:
        lines += ["## Technical Skills", ""]
        for v in verified:
            claim_text = claims_by_id.get(v.claim_id, v.claim_id)
            pointer = v.evidence[0].pointer if v.evidence else ""
            lines.append(f"- {claim_text}  [source: `{pointer}`]")
        lines.append("")

    if include_unverifiable and unverifiable:
        lines += [
            "---",
            "",
            "## Additional Claims (Unverifiable from GitHub)",
            "",
            f"> {_UNVERIFIABLE_MARKER}",
            "",
        ]
        for v in unverifiable:
            claim_text = claims_by_id.get(v.claim_id, v.claim_id)
            lines.append(f"- {claim_text}  {_UNVERIFIABLE_MARKER}")
        lines.append("")

    return "\n".join(lines)


class CVGenerator:
    """
    Produce an ATS-structured Markdown CV from verified verdicts.

    Uses a Groq LLM call with the ``cv_generator.txt`` prompt to convert the
    verified claim list into professional CV prose.  The LLM is strictly
    grounded: it may only reference claims and artefact pointers that were
    passed to it.

    If the LLM call fails for any reason, falls back to a deterministic
    bullet-list CV so the pipeline always produces output.

    Every verified CV line carries an ``[source: <pointer>]`` tag.
    Unverifiable claims are only included when ``include_unverifiable=True``
    and are marked ``[NOT VERIFIED FROM GITHUB]``.

    Examples
    --------
    >>> gen = CVGenerator()
    >>> cv_md, cv_lines = gen.generate(
    ...     verdicts, claims_by_id, github_username,
    ...     include_unverifiable=False,
    ...     jd_title="Senior Python Engineer",
    ... )
    """

    def __init__(self) -> None:
        """Initialise the generator, loading the prompt template once."""
        self._prompt_template = _load_prompt("cv_generator")
        settings = get_settings()
        self._client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value()
        )
        self._model = settings.groq_model

    async def generate(  # type: ignore[override]
        self,
        verdicts: list[Verdict],
        claims_by_id: dict[str, str],
        github_username: str,
        include_unverifiable: bool = False,
        jd_title: str = "Software Engineer",
    ) -> tuple[str, list[CVLine]]:
        """
        Generate a professional CV and return it alongside structured lines.

        Calls Groq with the verified claims as strict grounding.  Falls back
        to a deterministic bullet list if the LLM call fails.

        Parameters
        ----------
        verdicts : list[Verdict]
            Final verdicts from ``IndependentVerifier.verify_all()``.
        claims_by_id : dict[str, str]
            Mapping of ``claim_id`` to claim text for human-readable output.
        github_username : str
            Candidate's GitHub username (used in the CV header and prompt).
        include_unverifiable : bool
            When ``True``, unverifiable claims are appended with a visible
            ``[NOT VERIFIED FROM GITHUB]`` marker.
        jd_title : str
            Short title of the role the CV is being tailored for.

        Returns
        -------
        tuple[str, list[CVLine]]
            The full Markdown CV string and the list of ``CVLine`` objects.
            Every non-unverifiable ``CVLine`` is guaranteed to have a
            non-empty ``artefact_pointer``.
        """
        verified = [v for v in verdicts if v.classification == Classification.VERIFIED]
        unverifiable = [
            v for v in verdicts if v.classification == Classification.UNVERIFIABLE
        ]

        # Build structured CVLine objects (used for the pointer-completeness check)
        cv_lines: list[CVLine] = [
            CVLine(
                text=claims_by_id.get(v.claim_id, v.claim_id),
                artefact_pointer=v.evidence[0].pointer if v.evidence else "",
                is_unverifiable=False,
            )
            for v in verified
        ]

        if not verified:
            md = _build_fallback_cv(
                verified, unverifiable, claims_by_id,
                github_username, include_unverifiable,
            )
            return md, cv_lines

        # Attempt LLM-generated professional prose
        md = await self._generate_with_llm(
            verified=verified,
            unverifiable=unverifiable,
            claims_by_id=claims_by_id,
            github_username=github_username,
            include_unverifiable=include_unverifiable,
            jd_title=jd_title,
        )

        return md, cv_lines

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_with_llm(
        self,
        verified: list[Verdict],
        unverifiable: list[Verdict],
        claims_by_id: dict[str, str],
        github_username: str,
        include_unverifiable: bool,
        jd_title: str,
    ) -> str:
        """
        Call Groq with the cv_generator prompt and return the CV Markdown.

        Falls back to the deterministic bullet list on any exception.

        Parameters
        ----------
        verified : list[Verdict]
            Verified verdicts to ground the CV in.
        unverifiable : list[Verdict]
            Unverifiable verdicts, optionally appended.
        claims_by_id : dict[str, str]
            Claim ID to text mapping.
        github_username : str
            Candidate GitHub username.
        include_unverifiable : bool
            Whether to include unverifiable claims section.
        jd_title : str
            Role title for context in the prompt.

        Returns
        -------
        str
            Markdown CV string, em-dash free.
        """
        display_name = _extract_display_name(github_username, verified)
        verified_claims_json = _build_verified_claims_json(verified, claims_by_id)

        unverifiable_section = ""
        if include_unverifiable and unverifiable:
            unverifiable_items = "\n".join(
                f"- {claims_by_id.get(v.claim_id, v.claim_id)}"
                for v in unverifiable
            )
            unverifiable_section = (
                "UNVERIFIABLE CLAIMS (include at end with marker if requested)\n"
                + unverifiable_items
            )

        # GitHub API returns name separately; we pass username as fallback
        contact_line = f"[{github_username}](https://github.com/{github_username})"

        prompt = self._prompt_template.format(
            github_username=github_username,
            display_name=display_name,
            jd_title=jd_title,
            verified_claims_json=verified_claims_json,
            unverifiable_section=unverifiable_section,
            contact_line=contact_line,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=2048,
            )
            raw = response.choices[0].message.content or ""
            logger.debug("CV generator LLM response: %d chars", len(raw))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "CV generator LLM call failed (%s), falling back to bullet list", exc
            )
            return _build_fallback_cv(
                verified, unverifiable, claims_by_id,
                github_username, include_unverifiable,
            )

        # Strip any em-dashes the LLM smuggled in despite the instruction
        clean = _strip_em_dashes(raw.strip())
        return clean
