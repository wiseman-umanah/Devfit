"""
Rule-based verifier layer — deterministic, no LLM, no external I/O.

Every function in this module must be:
- Fully deterministic (same inputs → same output, always).
- Free of network calls, filesystem reads, and LLM invocations.
- Unit-testable with synthetic ``ArtefactBundle`` fixtures.

Rules implemented
-----------------
``check_date_arithmetic``
    Evaluates claims of the form "N+ years experience with X" by comparing
    against the GitHub account creation date and (where available) the
    earliest public commit in language X.

``check_language_presence``
    Evaluates claims of the form "expert in / proficient in <language>" by
    checking whether that language appears in the ``LANGUAGE_STATS`` artefact
    with a meaningful byte count.

``check_zero_activity``
    Contradicts claims of substantial work in a language/framework when the
    ``ArtefactBundle`` contains zero repos or commits mentioning that
    technology.

Return contract
---------------
Every rule returns ``Verdict | None``.  ``None`` means the rule cannot
decisively resolve the claim — the claim passes to the LLM layer.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime

from devfit.github.bundle import ArtefactBundle
from devfit.schema import (
    ArtefactType,
    Claim,
    ClaimCategory,
    Classification,
    Verdict,
)

logger = logging.getLogger(__name__)

# Minimum byte count for a language to be considered "present".
_LANG_PRESENCE_THRESHOLD_BYTES = 500

# Regex to extract "N+ years" or "N years" patterns from claim text.
_YEARS_PATTERN = re.compile(r"(\d+)\+?\s+years?", re.IGNORECASE)


def check_date_arithmetic(
    claim: Claim,
    bundle: ArtefactBundle,
) -> Verdict | None:
    """
    Verify or contradict experience-duration claims using account age.

    Applies only to claims with ``category == EXPERIENCE_DURATION``.
    Extracts the stated number of years from the claim text and compares
    against the GitHub account creation date.  If the account is younger
    than the claimed experience, the claim is ``CONTRADICTED``.

    Parameters
    ----------
    claim : Claim
        The claim to evaluate.
    bundle : ArtefactBundle
        The artefact bundle for the candidate's GitHub profile.

    Returns
    -------
    Verdict | None
        A ``Verdict`` if the rule can resolve the claim, otherwise ``None``.
    """
    if claim.category != ClaimCategory.EXPERIENCE_DURATION:
        return None

    match = _YEARS_PATTERN.search(claim.text)
    if not match:
        return None

    claimed_years = int(match.group(1))

    metadata = bundle.by_type(ArtefactType.ACCOUNT_METADATA)
    if not metadata:
        return None

    # Extract account age from extracted_fact:
    # e.g. "Account created 2020-01-15 (5y 3m ago)"
    fact = metadata[0].extracted_fact
    age_match = re.search(r"Account created (\d{4}-\d{2}-\d{2})", fact)
    if not age_match:
        return None

    try:
        created_dt = datetime.fromisoformat(age_match.group(1)).replace(tzinfo=UTC)
    except ValueError:
        return None

    account_age_years = (datetime.now(UTC) - created_dt).days / 365.25

    if account_age_years < claimed_years:
        logger.debug(
            "Date rule CONTRADICTED claim '%s': account age %.1fy < claimed %dy",
            claim.id,
            account_age_years,
            claimed_years,
        )
        return Verdict(
            claim_id=claim.id,
            classification=Classification.CONTRADICTED,
            confidence=0.95,
            evidence=[metadata[0]],
            rule_checked=True,
            llm_confirmed=False,
        )

    return None


def check_language_presence(
    claim: Claim,
    bundle: ArtefactBundle,
) -> Verdict | None:
    """
    Verify or contradict language-skill claims using the language stats artefact.

    Applies only to ``TECHNICAL_SKILL`` claims.  Extracts language names from
    the claim text and checks whether they appear in the ``LANGUAGE_STATS``
    artefact with more than ``_LANG_PRESENCE_THRESHOLD_BYTES`` bytes.

    Parameters
    ----------
    claim : Claim
        The claim to evaluate.
    bundle : ArtefactBundle
        The artefact bundle for the candidate's GitHub profile.

    Returns
    -------
    Verdict | None
        A ``Verdict`` if the rule can resolve the claim, otherwise ``None``.
    """
    if claim.category != ClaimCategory.TECHNICAL_SKILL:
        return None

    lang_artefacts = bundle.by_type(ArtefactType.LANGUAGE_STATS)
    if not lang_artefacts:
        return None

    lang_fact = lang_artefacts[0].extracted_fact.lower()

    # Extract candidate language names from claim text
    # Heuristic: capitalised words are likely language names
    candidates = re.findall(r"\b([A-Z][a-zA-Z+#]+)\b", claim.text)
    if not candidates:
        return None

    for lang in candidates:
        lang_lower = lang.lower()
        if lang_lower in lang_fact:
            # Check byte count to ensure meaningful presence
            byte_match = re.search(
                rf"{re.escape(lang_lower)}\s*\(([0-9,]+)\s*bytes\)", lang_fact
            )
            if byte_match:
                byte_count = int(byte_match.group(1).replace(",", ""))
                if byte_count >= _LANG_PRESENCE_THRESHOLD_BYTES:
                    return Verdict(
                        claim_id=claim.id,
                        classification=Classification.VERIFIED,
                        confidence=0.90,
                        evidence=[lang_artefacts[0]],
                        rule_checked=True,
                        llm_confirmed=False,
                    )
        else:
            # Language completely absent from stats
            logger.debug(
                "Language rule CONTRADICTED claim '%s': '%s' not in language stats",
                claim.id,
                lang,
            )
            return Verdict(
                claim_id=claim.id,
                classification=Classification.CONTRADICTED,
                confidence=0.88,
                evidence=[lang_artefacts[0]],
                rule_checked=True,
                llm_confirmed=False,
            )

    return None


def check_zero_activity(
    claim: Claim,
    bundle: ArtefactBundle,
) -> Verdict | None:
    """
    Contradict claims of substantial work when no repos or commits exist.

    Checks whether the ``ArtefactBundle`` contains any ``REPO`` artefacts
    that mention the technology named in the claim.  If none are found and the
    claim asserts significant experience, the claim is ``CONTRADICTED``.

    Applies only to ``TECHNICAL_SKILL`` claims not already handled by
    ``check_language_presence``.

    Parameters
    ----------
    claim : Claim
        The claim to evaluate.
    bundle : ArtefactBundle
        The artefact bundle for the candidate's GitHub profile.

    Returns
    -------
    Verdict | None
        A ``Verdict`` if the rule can resolve the claim, otherwise ``None``.
    """
    if claim.category != ClaimCategory.TECHNICAL_SKILL:
        return None

    # Only apply when "expert" / "extensive" / "5+ years" type language is present
    strong_claim = re.search(
        r"\b(expert|extensive|senior|5\+|proficient|specialist)\b",
        claim.text,
        re.IGNORECASE,
    )
    if not strong_claim:
        return None

    repo_artefacts = bundle.by_type(ArtefactType.REPO)
    if not repo_artefacts:
        # No repos at all.  Use account metadata as evidence if available;
        # if even that is absent we cannot produce a contradicted verdict
        # (schema requires non-empty evidence for contradicted).
        metadata = bundle.by_type(ArtefactType.ACCOUNT_METADATA)
        if not metadata:
            return None
        return Verdict(
            claim_id=claim.id,
            classification=Classification.CONTRADICTED,
            confidence=0.80,
            evidence=metadata,
            rule_checked=True,
            llm_confirmed=False,
        )

    # Extract technology keywords from claim text
    techs = re.findall(r"\b([A-Z][a-zA-Z+#.]+)\b", claim.text)
    all_repo_facts = " ".join(a.extracted_fact.lower() for a in repo_artefacts)

    for tech in techs:
        if tech.lower() not in all_repo_facts:
            logger.debug(
                "Zero-activity rule CONTRADICTED claim '%s': "
                "'%s' absent from all repos",
                claim.id,
                tech,
            )
            return Verdict(
                claim_id=claim.id,
                classification=Classification.CONTRADICTED,
                confidence=0.75,
                evidence=[repo_artefacts[0]],
                rule_checked=True,
                llm_confirmed=False,
            )

    return None


class RuleVerifier:
    """
    Orchestrate all deterministic rules for a single claim.

    Rules are evaluated in priority order; the first rule that returns a
    non-``None`` ``Verdict`` wins.  ``None`` is returned only when no rule
    can resolve the claim, indicating the claim should be forwarded to the
    constrained LLM verifier.

    Examples
    --------
    >>> verifier = RuleVerifier()
    >>> verdict = verifier.run(claim, bundle)
    >>> if verdict is None:
    ...     # Forward to LLM layer
    ...     pass
    """

    _RULES = [
        check_date_arithmetic,
        check_language_presence,
        check_zero_activity,
    ]

    def run(self, claim: Claim, bundle: ArtefactBundle) -> Verdict | None:
        """
        Run all rules against a claim and return the first decisive verdict.

        Parameters
        ----------
        claim : Claim
            The claim to evaluate.
        bundle : ArtefactBundle
            The artefact bundle for the candidate's GitHub profile.

        Returns
        -------
        Verdict | None
            The first non-``None`` verdict produced by any rule, or ``None``
            if no rule can decisively resolve the claim.
        """
        for rule_fn in self._RULES:
            result = rule_fn(claim, bundle)
            if result is not None:
                return result
        return None
