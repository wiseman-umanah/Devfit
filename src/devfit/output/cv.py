"""
CV generator -- produces a professional, ATS-structured, JD-tailored Markdown CV.

Design rules (per TRD section 3.5)
------------------------------------
- The CV is about the candidate, not the JD. The ArtefactBundle provides the
  candidate's real GitHub profile: name, bio, repos, languages, activity.
- Verified claims from the JD provide the factual backbone -- what the pipeline
  confirmed the candidate can actually demonstrate.
- Every verified bullet carries a traceable [source: <pointer>] tag, wrapped in
  a hidden span so the PDF export strips it from the printed document.
- Unverifiable claims included only at explicit request, marked
  [NOT VERIFIED FROM GITHUB].
- No em-dashes, no emojis, no filler phrases in output. Active-voice prose only.
- Falls back to a deterministic CV if the LLM call fails.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from groq import AsyncGroq

from devfit.config import get_settings
from devfit.output.cv_utils import _extract_profile, _post_process
from devfit.schema import ArtefactType, Classification, Verdict

if TYPE_CHECKING:
    from devfit.github.bundle import ArtefactBundle

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_UNVERIFIABLE_MARKER = "[NOT VERIFIED FROM GITHUB]"


def _load_prompt(name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Parameters
    ----------
    name : str
        Filename without extension, e.g. ``"tailored_cv"``.

    Returns
    -------
    str
        Raw prompt template string.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


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
        JSON array, each entry with ``text`` and ``artefact_pointer``.
    """
    payload = [
        {
            "text": claims_by_id.get(v.claim_id, v.claim_id),
            "artefact_pointer": v.evidence[0].pointer if v.evidence else "",
        }
        for v in verified
    ]
    return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Fallback (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _build_fallback_cv(
    verified: list[Verdict],
    unverifiable: list[Verdict],
    claims_by_id: dict[str, str],
    github_username: str,
    display_name: str,
    include_unverifiable: bool,
    profile: dict[str, str],
) -> str:
    """
    Deterministic bullet-list CV used when the LLM call fails.

    Incorporates real profile data (languages, repos, activity) even without
    the LLM so the output is still candidate-centric.

    Parameters
    ----------
    verified : list[Verdict]
        Verified verdicts.
    unverifiable : list[Verdict]
        Unverifiable verdicts.
    claims_by_id : dict[str, str]
        Mapping of claim_id to claim text.
    github_username : str
        Candidate GitHub username.
    display_name : str
        Display name for the CV header.
    include_unverifiable : bool
        Whether to include unverifiable claims.
    profile : dict[str, str]
        Extracted profile fields from the bundle.

    Returns
    -------
    str
        Plain Markdown CV.
    """
    lines: list[str] = [
        f"# {display_name}",
        "Software Engineer",
        f"github.com/{github_username}",
        "",
    ]

    if not verified:
        lines += [
            "## Summary",
            "",
            "No verified claims could be extracted from public GitHub data "
            "for this candidate and job description.",
            "",
        ]
    else:
        # ── Summary ────────────────────────────────────────────────────────
        bio = profile.get("bio", "")
        top_langs = profile.get("top_languages", "")
        lines += ["## Summary", ""]
        summary_parts: list[str] = []
        if bio and bio != "No bio available.":
            summary_parts.append(bio.rstrip(".") + ".")
        if top_langs and top_langs != "unknown":
            import re as _re
            lang_names = _re.findall(r"([A-Za-z+#]+)\s*\(", top_langs)
            if lang_names:
                summary_parts.append(
                    f"Primary languages include {', '.join(lang_names[:4])}."
                )
        lines.append(" ".join(summary_parts) if summary_parts else "")
        lines.append("")

        # ── Projects (from verified verdicts) ──────────────────────────────
        lines += ["## Projects", ""]
        for v in verified:
            claim_text = claims_by_id.get(v.claim_id, v.claim_id)
            pointer = v.evidence[0].pointer if v.evidence else ""
            lines.append(f"- {claim_text}")
            if pointer:
                lines.append(
                    f'  <span class="source-tag">[source: {pointer}]</span>'
                )
        lines.append("")

        # ── Technical Skills ───────────────────────────────────────────────
        if top_langs and top_langs != "unknown":
            import re as _re
            lang_names = _re.findall(r"([A-Za-z+#.]+)\s*\(", top_langs)
            if lang_names:
                lines += [
                    "## Technical Skills",
                    "",
                    f"Languages: {', '.join(lang_names[:8])}",
                    "",
                ]

    # ── Open Source Activity ───────────────────────────────────────────────
    pub_repos = profile.get("public_repos", "")
    account_created = profile.get("account_created", "")
    recent_activity = profile.get("recent_activity", "")
    if pub_repos and pub_repos != "unknown":
        lines += ["## Open Source Activity", ""]
        created_str = (
            f", active since {account_created}" if account_created != "unknown" else ""
        )
        lines.append(
            f"- Maintains {pub_repos} public repositories on GitHub{created_str}."
        )
        if recent_activity and recent_activity != "unknown":
            lines.append(f"- {recent_activity.rstrip('.')}.")
        lines.append("")

    if include_unverifiable and unverifiable:
        lines += [
            "---",
            "",
            "## Additional Claims (Not Verified from GitHub)",
            "",
            f"> {_UNVERIFIABLE_MARKER}",
            "",
        ]
        for v in unverifiable:
            claim_text = claims_by_id.get(v.claim_id, v.claim_id)
            lines.append(f"- {claim_text}  {_UNVERIFIABLE_MARKER}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CVGenerator
# ---------------------------------------------------------------------------


class CVGenerator:
    """
    Produce a professional ATS-structured Markdown CV from verified verdicts.

    The generator is candidate-centric: it passes the candidate's real GitHub
    profile (repos, languages, activity, bio) to the LLM alongside the list
    of verified claims, so the output reads as a description of a person, not
    a compliance checklist.

    The LLM is strictly grounded: it may only assert facts traceable to the
    ArtefactBundle or the verified claims list.

    Falls back to a deterministic bullet list if the LLM call fails.

    Examples
    --------
    >>> gen = CVGenerator()
    >>> cv_md, cv_lines = gen.generate(
    ...     verdicts, claims_by_id, github_username,
    ...     bundle=bundle,
    ...     jd_title="Senior Python Engineer",
    ... )
    """

    def __init__(self) -> None:
        """Initialise the generator, loading the prompt template once."""
        self._prompt_template = _load_prompt("tailored_cv")
        settings = get_settings()
        self._client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value()
        )
        self._model = settings.groq_model

    async def generate(
        self,
        verdicts: list[Verdict],
        claims_by_id: dict[str, str],
        github_username: str,
        include_unverifiable: bool = False,
        jd_title: str = "Software Engineer",
        bundle: ArtefactBundle | None = None,
    ) -> tuple[str, list[CVLine]]:
        """
        Generate a professional CV and return it alongside structured lines.

        Calls Groq with the candidate's real GitHub profile and the verified
        claims as grounding.  Falls back to a deterministic bullet list if
        the LLM call fails.

        Parameters
        ----------
        verdicts : list[Verdict]
            Final verdicts from ``IndependentVerifier.verify_all()``.
        claims_by_id : dict[str, str]
            Mapping of ``claim_id`` to claim text for human-readable output.
        github_username : str
            Candidate's GitHub username.
        include_unverifiable : bool
            When ``True``, unverifiable claims are appended with
            ``[NOT VERIFIED FROM GITHUB]`` markers.
        jd_title : str
            Short title of the role the CV is tailored for.
        bundle : ArtefactBundle | None
            Collected GitHub artefacts.  When supplied, real profile data
            (name, bio, repos, languages, activity) is embedded in the
            prompt so the LLM can write a candidate-centric CV.

        Returns
        -------
        tuple[str, list[CVLine]]
            The full Markdown CV string and the list of ``CVLine`` objects.
            Every non-unverifiable ``CVLine`` has a non-empty
            ``artefact_pointer`` (pointer-completeness invariant).
        """
        verified = [v for v in verdicts if v.classification == Classification.VERIFIED]
        unverifiable = [
            v for v in verdicts if v.classification == Classification.UNVERIFIABLE
        ]
        profile = _extract_profile(github_username, bundle)
        display_name = profile["display_name"]

        # Build structured CVLine objects (pointer-completeness check data)
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
                github_username, display_name, include_unverifiable, profile,
            )
            return md, cv_lines

        md = await self._generate_with_llm(
            verified=verified,
            unverifiable=unverifiable,
            claims_by_id=claims_by_id,
            github_username=github_username,
            display_name=display_name,
            include_unverifiable=include_unverifiable,
            jd_title=jd_title,
            profile=profile,
        )
        return md, cv_lines

    async def _generate_with_llm(
        self,
        verified: list[Verdict],
        unverifiable: list[Verdict],
        claims_by_id: dict[str, str],
        github_username: str,
        display_name: str,
        include_unverifiable: bool,
        jd_title: str,
        profile: dict[str, str],
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
        display_name : str
            Best available display name for the header.
        include_unverifiable : bool
            Whether to include unverifiable claims section.
        jd_title : str
            Role title for context in the prompt.
        profile : dict[str, str]
            Extracted profile fields from the bundle.

        Returns
        -------
        str
            Markdown CV string, em-dash free.
        """
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

        prompt = self._prompt_template.format(
            github_username=github_username,
            display_name=display_name,
            jd_title=jd_title,
            verified_claims_json=verified_claims_json,
            unverifiable_section=unverifiable_section,
            account_created=profile["account_created"],
            public_repos=profile["public_repos"],
            bio=profile["bio"],
            top_languages=profile["top_languages"],
            recent_activity=profile["recent_activity"],
            repo_list=profile["repo_list"],
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
                "CV generator LLM call failed (%s), using fallback", exc
            )
            return _build_fallback_cv(
                verified, unverifiable, claims_by_id,
                github_username, display_name, include_unverifiable, profile,
            )

        return _post_process(raw)
