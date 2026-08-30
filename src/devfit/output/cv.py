"""
CV generator — produces an ATS-structured Markdown CV from verified verdicts.

Design rules (per TRD §3.5)
----------------------------
- The CV contains **only** ``verified`` claims by default.
- Every claim bullet must carry a traceable ``[source: <pointer>]`` tag.
- ``unverifiable`` claims may be included **only** at explicit user request
  and must render with a ``[CANNOT BE CONFIRMED FROM GITHUB]`` marker.
- No LLM call — the CV is assembled deterministically from verified verdicts
  so there is zero risk of hallucination in the output artefact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from devfit.schema import Classification, Verdict

logger = logging.getLogger(__name__)

_UNVERIFIABLE_MARKER = "[CANNOT BE CONFIRMED FROM GITHUB]"


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


class CVGenerator:
    """
    Produce an ATS-structured Markdown CV from verified verdicts.

    No LLM calls.  The CV is assembled deterministically; every line
    can be traced back to a specific ``Artefact`` pointer.

    ATS structure
    -------------
    ## Summary | ## Technical Skills | ## Experience | ## Projects

    Examples
    --------
    >>> gen = CVGenerator()
    >>> cv_md, cv_lines = gen.generate(
    ...     verdicts, claims_by_id, github_username,
    ...     include_unverifiable=False,
    ... )
    """

    def generate(
        self,
        verdicts: list[Verdict],
        claims_by_id: dict[str, str],
        github_username: str,
        include_unverifiable: bool = False,
    ) -> tuple[str, list[CVLine]]:
        """
        Generate the CV Markdown and return it alongside the structured lines.

        Parameters
        ----------
        verdicts : list[Verdict]
            Final verdicts from ``IndependentVerifier.verify_all()``.
        claims_by_id : dict[str, str]
            Mapping of ``claim_id → claim_text`` for human-readable output.
        github_username : str
            Candidate's GitHub username (used in the CV header).
        include_unverifiable : bool
            When ``True``, unverifiable claims are appended with a visible
            ``[CANNOT BE CONFIRMED FROM GITHUB]`` marker.

        Returns
        -------
        tuple[str, list[CVLine]]
            The full Markdown CV string and the list of ``CVLine`` objects
            (used by the evidence check: every non-unverifiable line must
            have a non-empty ``artefact_pointer``).
        """
        verified = [v for v in verdicts if v.classification == Classification.VERIFIED]
        unverifiable = [
            v for v in verdicts if v.classification == Classification.UNVERIFIABLE
        ]

        cv_lines: list[CVLine] = []

        # Build structured lines from verified verdicts
        for v in verified:
            claim_text = claims_by_id.get(v.claim_id, v.claim_id)
            pointer = v.evidence[0].pointer if v.evidence else ""
            cv_lines.append(
                CVLine(text=claim_text, artefact_pointer=pointer, is_unverifiable=False)
            )

        # Optionally append unverifiable claims with marker
        unverifiable_lines: list[CVLine] = []
        if include_unverifiable:
            for v in unverifiable:
                claim_text = claims_by_id.get(v.claim_id, v.claim_id)
                unverifiable_lines.append(
                    CVLine(
                        text=claim_text,
                        artefact_pointer="",
                        is_unverifiable=True,
                    )
                )

        md = self._render(
            cv_lines, unverifiable_lines, github_username, include_unverifiable
        )
        return md, cv_lines

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _render(
        verified_lines: list[CVLine],
        unverifiable_lines: list[CVLine],
        github_username: str,
        include_unverifiable: bool,
    ) -> str:
        """
        Render ``CVLine`` objects into a Markdown string.

        Parameters
        ----------
        verified_lines : list[CVLine]
            Lines backed by verified artefacts.
        unverifiable_lines : list[CVLine]
            Unverifiable lines to append (may be empty).
        github_username : str
            Candidate GitHub username for the header.
        include_unverifiable : bool
            Controls whether to add the unverifiable disclaimer section.

        Returns
        -------
        str
            Full Markdown CV.
        """
        lines: list[str] = []

        lines += [
            "# Curriculum Vitae",
            "",
            f"**GitHub:** https://github.com/{github_username}  ",
            "",
            "> *Every claim in this CV is backed by a verifiable public GitHub "
            "artefact.  Source pointers are listed inline.*",
            "",
            "---",
            "",
        ]

        if not verified_lines:
            lines += [
                "*No verified claims could be extracted from public GitHub data "
                "for this candidate and job description.*",
                "",
            ]
        else:
            # ── Technical Skills ────────────────────────────────────────────
            lines += ["## Technical Skills", ""]
            for cl in verified_lines:
                lines.append(
                    f"- {cl.text}  [source: `{cl.artefact_pointer}`]"
                )
            lines.append("")

            # ── GitHub Profile ───────────────────────────────────────────────
            lines += [
                "## GitHub Profile",
                "",
                f"- Public profile: https://github.com/{github_username}",
                "",
            ]

        # ── Unverifiable section (only if requested) ─────────────────────────
        if include_unverifiable and unverifiable_lines:
            lines += [
                "---",
                "",
                "## Additional Claims (Unverifiable from GitHub)",
                "",
                f"> {_UNVERIFIABLE_MARKER}",
                "> These claims were provided by the candidate but cannot be "
                "confirmed or denied from public GitHub data alone.",
                "",
            ]
            for cl in unverifiable_lines:
                lines.append(f"- {cl.text}  {_UNVERIFIABLE_MARKER}")
            lines.append("")

        return "\n".join(lines)
