"""
Evidence appendix — renders every artefact pointer and extracted fact for all claims.

The appendix is a required output artefact: every claim in the final CV
must have a traceable pointer back to a GitHub artefact, and the appendix
provides that traceable index for human review.
"""

from __future__ import annotations

import logging

from devfit.schema import Classification, Verdict

logger = logging.getLogger(__name__)


class EvidenceAppendix:
    """
    Generate a Markdown evidence appendix from a list of ``Verdict`` objects.

    Lists every artefact pointer and its ``extracted_fact`` for verified and
    contradicted claims.  Unverifiable claims are noted without artefacts.

    Examples
    --------
    >>> gen = EvidenceAppendix()
    >>> appendix_md = gen.generate(verdicts, claims_by_id)
    """

    def generate(
        self,
        verdicts: list[Verdict],
        claims_by_id: dict[str, str],
    ) -> str:
        """
        Produce the Markdown evidence appendix.

        Parameters
        ----------
        verdicts : list[Verdict]
            Final verdicts from ``IndependentVerifier.verify_all()``.
        claims_by_id : dict[str, str]
            Mapping of ``claim_id → claim_text`` for human-readable output.

        Returns
        -------
        str
            Full Markdown evidence appendix as a string.
        """
        lines: list[str] = [
            "# Evidence Appendix",
            "",
            "Every claim in the DevFit report is listed here with its supporting "
            "artefact pointer(s).  Unverifiable claims carry no artefact.",
            "",
            "---",
            "",
        ]

        for v in verdicts:
            claim_text = claims_by_id.get(v.claim_id, v.claim_id)
            badge = {
                Classification.VERIFIED: "✓ VERIFIED",
                Classification.CONTRADICTED: "✗ CONTRADICTED",
                Classification.UNVERIFIABLE: "? UNVERIFIABLE",
            }[v.classification]

            lines.append(f"### `{v.claim_id}` — {badge}")
            lines.append("")
            lines.append(f"**Claim:** {claim_text}")
            lines.append("")

            if v.evidence:
                for art in v.evidence:
                    lines.append(f"- **Artefact type:** `{art.type}`")
                    lines.append(f"  - **Pointer:** `{art.pointer}`")
                    lines.append(f"  - **Fact:** {art.extracted_fact}")
            else:
                lines.append(
                    "- *No artefact — this claim cannot be confirmed "
                    "or denied from public GitHub data.*"
                )

            verified_by: list[str] = []
            if v.rule_checked:
                verified_by.append("rule layer")
            if v.llm_confirmed:
                verified_by.append("constrained LLM verifier")
            if verified_by:
                lines.append("")
                lines.append(f"Verified by: {', '.join(verified_by)}")

            lines.append("")

        return "\n".join(lines)
