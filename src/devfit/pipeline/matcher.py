"""
Evidence Matcher.

Pairs each ``Claim`` with relevant ``Artefact`` objects from the
``ArtefactBundle`` before the first-pass classifier runs.

Design
------
Matching is done in two passes, fastest-first:

1. **Skip pass** — claims flagged ``likely_unverifiable=True`` receive no
   candidate artefacts and are immediately short-circuited to a pre-built
   ``unverifiable`` verdict, saving LLM calls for soft-skill and duration
   claims that GitHub cannot resolve.

2. **Keyword pass** — for verifiable claims, the matcher extracts meaningful
   tokens from the claim text and scores each artefact by how many tokens
   appear in its ``extracted_fact``.  The top-N artefacts are returned as
   candidates for the classifier.

The matcher is deterministic and makes no network or LLM calls.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from devfit.github.bundle import ArtefactBundle
from devfit.schema import (
    Artefact,
    Claim,
    Classification,
    Verdict,
)

logger = logging.getLogger(__name__)

# Number of top-scoring artefacts to forward to the classifier per claim.
_MAX_CANDIDATES = 6

# Common English stop-words to ignore when scoring keyword matches.
_STOP_WORDS = frozenset(
    {
        "a", "an", "the", "and", "or", "in", "of", "for", "to", "with",
        "on", "at", "by", "is", "are", "was", "be", "has", "have",
        "strong", "deep", "proven", "experience", "familiarity", "skills",
        "knowledge", "ability", "understanding", "background", "record",
        "track", "prior", "excellent", "substantial", "significant",
    }
)


@dataclass
class MatchedClaim:
    """
    A ``Claim`` paired with its candidate ``Artefact`` objects.

    Parameters
    ----------
    claim : Claim
        The original claim to be verified.
    candidates : list[Artefact]
        Artefacts ranked by keyword relevance to the claim.
        Empty for ``likely_unverifiable`` claims.
    skipped : bool
        ``True`` when the claim was short-circuited (``likely_unverifiable``).
    """

    claim: Claim
    candidates: list[Artefact] = field(default_factory=list)
    skipped: bool = False


class EvidenceMatcher:
    """
    Match each ``Claim`` with relevant artefacts from an ``ArtefactBundle``.

    Returns a list of ``MatchedClaim`` objects.  Claims flagged
    ``likely_unverifiable`` are marked ``skipped=True`` with an empty
    candidate list so the classifier can fast-path them to ``unverifiable``.

    Examples
    --------
    >>> matcher = EvidenceMatcher()
    >>> matched = matcher.match(claims, bundle)
    >>> for mc in matched:
    ...     print(mc.claim.id, len(mc.candidates))
    """

    def match(
        self, claims: list[Claim], bundle: ArtefactBundle
    ) -> list[MatchedClaim]:
        """
        Match all claims against artefacts in the bundle.

        Parameters
        ----------
        claims : list[Claim]
            Claims extracted by the JD or resume analyzer.
        bundle : ArtefactBundle
            Artefact bundle built by the GitHub Collector.

        Returns
        -------
        list[MatchedClaim]
            One ``MatchedClaim`` per input claim, preserving order.
        """
        all_artefacts = bundle.all()
        results: list[MatchedClaim] = []

        for claim in claims:
            if claim.likely_unverifiable:
                logger.debug(
                    "Skipping claim '%s' (likely_unverifiable)", claim.id
                )
                results.append(MatchedClaim(claim=claim, skipped=True))
                continue

            candidates = self._rank_artefacts(claim.text, all_artefacts)
            logger.debug(
                "Claim '%s' matched %d candidates", claim.id, len(candidates)
            )
            results.append(MatchedClaim(claim=claim, candidates=candidates))

        return results

    def build_unverifiable_verdicts(
        self, matched: list[MatchedClaim]
    ) -> list[Verdict]:
        """
        Build ``unverifiable`` verdicts for all skipped claims.

        These claims are short-circuited before reaching the LLM — they will
        never be verified or contradicted from GitHub data.

        Parameters
        ----------
        matched : list[MatchedClaim]
            Output from ``match()``.

        Returns
        -------
        list[Verdict]
            One ``unverifiable`` verdict per skipped claim.
        """
        return [
            Verdict(
                claim_id=mc.claim.id,
                classification=Classification.UNVERIFIABLE,
                confidence=1.0,
                evidence=[],
                rule_checked=False,
                llm_confirmed=False,
            )
            for mc in matched
            if mc.skipped
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> set[str]:
        """
        Extract lowercase alphabetic tokens from text, removing stop words.

        Parameters
        ----------
        text : str
            Input text to tokenise.

        Returns
        -------
        set[str]
            Unique meaningful tokens.
        """
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9+#.]*", text.lower())
        return {t for t in tokens if t not in _STOP_WORDS and len(t) > 2}

    def _rank_artefacts(
        self, claim_text: str, artefacts: list[Artefact]
    ) -> list[Artefact]:
        """
        Rank artefacts by keyword overlap with the claim text.

        Parameters
        ----------
        claim_text : str
            The claim's text field.
        artefacts : list[Artefact]
            All artefacts from the bundle.

        Returns
        -------
        list[Artefact]
            Top-N artefacts sorted by descending keyword overlap score.
            Returns all artefacts if there are fewer than ``_MAX_CANDIDATES``.
        """
        claim_tokens = self._tokenise(claim_text)
        if not claim_tokens:
            return artefacts[:_MAX_CANDIDATES]

        scored: list[tuple[int, Artefact]] = []
        for art in artefacts:
            art_tokens = self._tokenise(art.extracted_fact + " " + art.pointer)
            score = len(claim_tokens & art_tokens)
            scored.append((score, art))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [art for _, art in scored[:_MAX_CANDIDATES]]
