"""
CV generator -- produces a professional, ATS-structured Markdown CV.

Design rules (per TRD section 3.5)
------------------------------------
- The CV is about the candidate, not the JD. The ArtefactBundle provides the
  candidate's real GitHub profile: name, bio, repos, languages, activity.
- Verified claims from the JD provide the factual backbone -- what the pipeline
  confirmed the candidate can actually demonstrate.
- Every claim bullet carries a traceable [source: <pointer>] tag.
- Unverifiable claims included only at explicit request, marked
  [NOT VERIFIED FROM GITHUB].
- No em-dashes in output. No filler phrases. Active-voice prose only.
- Falls back to a deterministic bullet list if the LLM call fails.
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
from devfit.schema import ArtefactType, Classification, Verdict

if TYPE_CHECKING:
    from devfit.github.bundle import ArtefactBundle

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
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _strip_em_dashes(text: str) -> str:
    """
    Replace em-dashes and double-hyphens with a comma and space.

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


# ---------------------------------------------------------------------------
# Bundle profile extraction
# ---------------------------------------------------------------------------


def _extract_profile(
    github_username: str, bundle: ArtefactBundle | None
) -> dict[str, str]:
    """
    Extract structured candidate profile fields from an ArtefactBundle.

    Returns a dict with keys used directly as prompt template variables:
    ``display_name``, ``account_created``, ``public_repos``, ``bio``,
    ``top_languages``, ``recent_activity``, ``repo_list``.

    Parameters
    ----------
    github_username : str
        Candidate GitHub username (fallback when bundle is missing).
    bundle : ArtefactBundle | None
        Collected GitHub artefacts.  When ``None``, all fields fall back
        to the username and placeholder strings.

    Returns
    -------
    dict[str, str]
        Flat string values ready for prompt interpolation.
    """
    profile: dict[str, str] = {
        "display_name": github_username,
        "account_created": "unknown",
        "public_repos": "unknown",
        "bio": "No bio available.",
        "top_languages": "unknown",
        "recent_activity": "unknown",
        "repo_list": "No repositories available.",
    }

    if bundle is None:
        return profile

    # Account metadata
    for a in bundle.by_type(ArtefactType.ACCOUNT_METADATA):
        fact = a.extracted_fact
        # Extract account creation date
        date_match = re.search(r"Account created (\d{4}-\d{2}-\d{2})", fact)
        if date_match:
            profile["account_created"] = date_match.group(1)
        # Extract public repo count
        repos_match = re.search(r"(\d+) public repos", fact)
        if repos_match:
            profile["public_repos"] = repos_match.group(1)
        # Extract bio
        bio_match = re.search(r'bio:\s*"([^"]+)"', fact)
        if bio_match:
            profile["bio"] = bio_match.group(1)
            # Try to extract a real name from bio (first 1-4 word segment)
            name_candidate = bio_match.group(1).split(".")[0].strip()
            if 1 < len(name_candidate.split()) <= 4:
                profile["display_name"] = name_candidate

    # Language stats
    for a in bundle.by_type(ArtefactType.LANGUAGE_STATS):
        # Strip "Language breakdown across top repos: " prefix if present
        fact = a.extracted_fact
        if ":" in fact:
            fact = fact.split(":", 1)[1].strip()
        profile["top_languages"] = fact
        break

    # Top repos (by star count, capped at 8 for readability)
    repo_artefacts = bundle.by_type(ArtefactType.REPO)
    if repo_artefacts:
        repo_lines = [f"  - {a.extracted_fact}" for a in repo_artefacts[:8]]
        profile["repo_list"] = "\n".join(repo_lines)

    # Contribution graph
    for a in bundle.by_type(ArtefactType.CONTRIBUTION_GRAPH):
        profile["recent_activity"] = a.extracted_fact
        break

    return profile


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
        "",
        f"GitHub: https://github.com/{github_username}",
        "",
    ]

    # Brief profile summary line
    bio = profile.get("bio", "")
    if bio and bio != "No bio available.":
        lines += [f"*{bio}*", ""]

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

    # Always include real GitHub stats if available
    top_langs = profile.get("top_languages", "")
    if top_langs and top_langs != "unknown":
        lines += ["## GitHub Profile", ""]
        lines.append(f"- Top languages: {top_langs}")
        activity = profile.get("recent_activity", "")
        if activity and activity != "unknown":
            lines.append(f"- Activity: {activity}")
        pub_repos = profile.get("public_repos", "")
        if pub_repos and pub_repos != "unknown":
            lines.append(f"- Public repositories: {pub_repos}")
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
        self._prompt_template = _load_prompt("cv_generator")
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

        return _strip_em_dashes(raw.strip())
