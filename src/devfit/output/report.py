"""
Fit-report generator — produces a Markdown fit report from ``Verdict`` objects.

Design rules (per TRD §3.5)
----------------------------
- Score and rationale are built **only** from ``verified`` and ``contradicted``
  verdicts.
- ``unverifiable`` claims appear in a separate section and are **never** scored
  as a gap or a strength — not a word about them in the score section.
- The report is deterministic: same verdicts → same output, every time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from devfit.schema import Classification, Verdict

logger = logging.getLogger(__name__)

# Score weights: each verified claim adds +1, each contradicted subtracts -1.
# Final score is expressed as a percentage of maximum possible.
_VERIFIED_WEIGHT = 1
_CONTRADICTED_WEIGHT = -1


@dataclass(frozen=True)
class FitScore:
    """
    Numeric fit score derived from verified/contradicted verdicts.

    Parameters
    ----------
    verified_count : int
        Number of verified claims.
    contradicted_count : int
        Number of contradicted claims.
    total_scorable : int
        Total claims that contributed to the score (verified + contradicted).
    score_pct : float
        Fit score as a percentage in ``[0.0, 100.0]``.
    label : str
        Human-readable label: ``"Strong Fit"``, ``"Partial Fit"``, or
        ``"Weak Fit / Mismatch"``.
    """

    verified_count: int
    contradicted_count: int
    total_scorable: int
    score_pct: float
    label: str


def _compute_score(verdicts: list[Verdict]) -> FitScore:
    """
    Compute a fit score from a list of verdicts.

    Parameters
    ----------
    verdicts : list[Verdict]
        All verdicts for a pipeline run (unverifiable ones are ignored).

    Returns
    -------
    FitScore
        Numeric score and label.
    """
    verified = [v for v in verdicts if v.classification == Classification.VERIFIED]
    contradicted = [
        v for v in verdicts if v.classification == Classification.CONTRADICTED
    ]
    total_scorable = len(verified) + len(contradicted)

    if total_scorable == 0:
        return FitScore(
            verified_count=0,
            contradicted_count=0,
            total_scorable=0,
            score_pct=0.0,
            label="Inconclusive",
        )

    raw = len(verified) * _VERIFIED_WEIGHT + len(contradicted) * _CONTRADICTED_WEIGHT
    # Normalise to [0, 100] relative to max possible (all verified)
    score_pct = max(0.0, min(100.0, (raw / total_scorable) * 100))

    if score_pct >= 70:
        label = "Strong Fit"
    elif score_pct >= 40:
        label = "Partial Fit"
    else:
        label = "Weak Fit / Mismatch"

    return FitScore(
        verified_count=len(verified),
        contradicted_count=len(contradicted),
        total_scorable=total_scorable,
        score_pct=score_pct,
        label=label,
    )


class FitReportGenerator:
    """
    Generate a Markdown fit report from a list of ``Verdict`` objects.

    The report is self-contained Markdown: no external template files,
    no LLM calls, fully deterministic.

    Examples
    --------
    >>> gen = FitReportGenerator()
    >>> report_md = gen.generate(verdicts, claims_by_id, github_username, jd_title)
    """

    def generate(
        self,
        verdicts: list[Verdict],
        claims_by_id: dict[str, str],
        github_username: str,
        jd_title: str,
    ) -> str:
        """
        Produce a Markdown fit report.

        Parameters
        ----------
        verdicts : list[Verdict]
            Final verdicts from ``IndependentVerifier.verify_all()``.
        claims_by_id : dict[str, str]
            Mapping of ``claim_id → claim_text`` for human-readable output.
        github_username : str
            Candidate's GitHub username (used in the report header).
        jd_title : str
            Short title of the job description (used in the report header).

        Returns
        -------
        str
            Full Markdown fit report as a string.
        """
        score = _compute_score(verdicts)
        verified = [v for v in verdicts if v.classification == Classification.VERIFIED]
        contradicted = [
            v for v in verdicts if v.classification == Classification.CONTRADICTED
        ]
        unverifiable = [
            v for v in verdicts if v.classification == Classification.UNVERIFIABLE
        ]

        lines: list[str] = []

        # ── Header ──────────────────────────────────────────────────────────
        lines += [
            "# DevFit Fit Report",
            "",
            f"**Candidate:** `{github_username}`  ",
            f"**Role:** {jd_title}  ",
            "",
            "---",
            "",
        ]

        # ── Score ────────────────────────────────────────────────────────────
        lines += [
            f"## Fit Score: {score.label} ({score.score_pct:.0f}/100)",
            "",
            "| Metric | Count |",
            "|--------|-------|",
            f"| Verified claims | {score.verified_count} |",
            f"| Contradicted claims | {score.contradicted_count} |",
            f"| Unverifiable claims | {len(unverifiable)} |",
            f"| Total scorable | {score.total_scorable} |",
            "",
            "> Score formula: each verified claim adds +1 point, each contradicted "
            "claim subtracts 1 point, normalised to 100.  "
            "Unverifiable claims are excluded from scoring.",
            "",
            "---",
            "",
        ]

        # ── Verified section ─────────────────────────────────────────────────
        if verified:
            lines += [f"## Verified Claims ({len(verified)})", ""]
            for v in verified:
                claim_text = claims_by_id.get(v.claim_id, v.claim_id)
                pointer = v.evidence[0].pointer if v.evidence else "—"
                lines.append(f"- **{claim_text}**")
                lines.append(f"  - Evidence: `{pointer}`")
                if v.evidence:
                    lines.append(f"  - *{v.evidence[0].extracted_fact}*")
                if v.rule_checked:
                    method = "rule"
                elif v.llm_confirmed:
                    method = "llm"
                else:
                    method = "classifier"
                lines.append(f"  - Verified by: {method} layer")
                lines.append("")
        else:
            lines += ["## Verified Claims", "", "*None.*", ""]

        # ── Contradicted section ─────────────────────────────────────────────
        if contradicted:
            lines += [f"## Contradicted Claims ({len(contradicted)})", ""]
            for v in contradicted:
                claim_text = claims_by_id.get(v.claim_id, v.claim_id)
                pointer = v.evidence[0].pointer if v.evidence else "—"
                lines.append(f"- **{claim_text}**")
                lines.append(f"  - Contradicting evidence: `{pointer}`")
                if v.evidence:
                    lines.append(f"  - *{v.evidence[0].extracted_fact}*")
                lines.append("")
        else:
            lines += ["## Contradicted Claims", "", "*None.*", ""]

        lines.append("---")
        lines.append("")

        # ── Unverifiable section (never scored) ───────────────────────────────
        lines += [
            f"## Unverifiable Claims ({len(unverifiable)})",
            "",
            "> These claims cannot be confirmed or denied from public GitHub data.",
            "> They are listed here for transparency but contribute **zero** to "
            "the fit score.",
            "",
        ]
        if unverifiable:
            for v in unverifiable:
                claim_text = claims_by_id.get(v.claim_id, v.claim_id)
                lines.append(f"- {claim_text}")
            lines.append("")
        else:
            lines += ["*None.*", ""]

        return "\n".join(lines)
