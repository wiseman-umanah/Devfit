"""
Independent Verifier — the core quality gate of DevFit.

Orchestrates two strictly ordered, independent layers:

Layer 1 — Rule-based (``RuleVerifier``)
    Deterministic, no LLM, no network calls.  Runs date-arithmetic,
    language-presence, and zero-activity checks.  Claims resolved here are
    **finalised** and never forwarded to Layer 2.

Layer 2 — Constrained LLM (``ConstrainedLLMVerifier``)
    Only runs when Layer 1 returns ``None``.  Uses a narrow prompt to
    confirm or reject the first-pass classifier's draft verdict against the
    specific evidence it cited.  Cannot upgrade a verdict; can only confirm
    or downgrade.

Architectural independence requirement (TRD §3.4)
-------------------------------------------------
The two layers must be able to **disagree** with each other and with the
first-pass classifier.  This module enforces that independence: Layer 1 can
contradict a classifier-proposed ``verified`` verdict; Layer 2 can downgrade
a Layer-1-confirmed verdict is not applicable (Layer 1 finalises, bypassing 2).

Trajectory logging
------------------
Every verifier decision is emitted as a structured log entry at ``INFO``
level, suitable for capture by the trajectory logger in the CLI layer.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

from devfit.pipeline.matcher import MatchedClaim
from devfit.schema import Artefact, Claim, Classification, Verdict
from devfit.verifier.llm import ConstrainedLLMVerifier
from devfit.verifier.rules import RuleVerifier

logger = logging.getLogger(__name__)


@dataclass
class VerifierDecision:
    """
    Structured record of a single verifier decision for trajectory logging.

    Parameters
    ----------
    claim_id : str
        ID of the claim that was verified.
    claim_text : str
        Text of the claim, for human-readable logs.
    layer : str
        ``"rule"`` when Layer 1 resolved; ``"llm"`` when Layer 2 ran;
        ``"unverifiable_skip"`` for pre-skipped claims.
    draft_classification : Classification | None
        What the first-pass classifier proposed (``None`` for skips).
    final_classification : Classification
        The verdict classification after the verifier ran.
    was_downgraded : bool
        ``True`` if the verifier changed the classification.
    reason : str
        Short human-readable explanation.
    artefact_pointers : list[str]
        Pointers of artefacts cited in the final verdict.
    """

    claim_id: str
    claim_text: str
    layer: str
    draft_classification: Classification | None
    final_classification: Classification
    was_downgraded: bool
    reason: str
    artefact_pointers: list[str] = field(default_factory=list)


class IndependentVerifier:
    """
    Orchestrate both verifier layers to produce final ``Verdict`` objects.

    Accepts the draft verdicts from the first-pass classifier and the original
    matched claims (needed to pass the artefact bundle back to Layer 1).

    All Layer 2 (LLM) calls are dispatched concurrently via ``asyncio.gather``
    for claims that Layer 1 could not resolve.

    Examples
    --------
    >>> verifier = IndependentVerifier()
    >>> final_verdicts, decisions = await verifier.verify_all(
    ...     draft_verdicts, matched_claims
    ... )
    """

    def __init__(self) -> None:
        """Initialise both verifier layers."""
        self._rule_verifier = RuleVerifier()
        self._llm_verifier = ConstrainedLLMVerifier()

    async def verify_all(
        self,
        draft_verdicts: list[Verdict],
        matched_claims: list[MatchedClaim],
    ) -> tuple[list[Verdict], list[VerifierDecision]]:
        """
        Run both verifier layers over all draft verdicts.

        Layer 1 (rules) runs synchronously first for each claim.  Claims the
        rule layer resolves are finalised immediately.  The remaining claims
        are forwarded to Layer 2 (LLM) concurrently.

        Parameters
        ----------
        draft_verdicts : list[Verdict]
            Draft verdicts from ``FirstPassClassifier.classify()``.
            Does **not** include pre-skipped unverifiable verdicts — pass
            those in separately via ``unverifiable_verdicts``.
        matched_claims : list[MatchedClaim]
            The corresponding matched claims (same order as
            ``draft_verdicts``).  Used to supply the artefact bundle
            to the rule layer.

        Returns
        -------
        tuple[list[Verdict], list[VerifierDecision]]
            Final verdicts (same order as inputs) and a list of
            ``VerifierDecision`` records for trajectory logging.
        """
        # Build lookup: claim_id → MatchedClaim for rule layer access
        mc_index: dict[str, MatchedClaim] = {
            mc.claim.id: mc for mc in matched_claims
        }

        final_verdicts: list[Verdict] = []
        decisions: list[VerifierDecision] = []
        llm_queue: list[tuple[int, Verdict, list[Artefact], Claim]] = []

        # ── Layer 1: rule-based (synchronous, no await needed) ──────────────
        for idx, draft in enumerate(draft_verdicts):
            mc = mc_index.get(draft.claim_id)
            if mc is None:
                # No matched claim found — pass through unchanged
                final_verdicts.append(draft)
                continue

            from devfit.github.bundle import ArtefactBundle

            bundle = ArtefactBundle(artefacts=mc.candidates)
            rule_verdict = self._rule_verifier.run(mc.claim, bundle)

            if rule_verdict is not None:
                # Layer 1 resolved — finalise and skip Layer 2
                was_downgraded = (
                    rule_verdict.classification != draft.classification
                )
                decision = VerifierDecision(
                    claim_id=draft.claim_id,
                    claim_text=mc.claim.text,
                    layer="rule",
                    draft_classification=draft.classification,
                    final_classification=rule_verdict.classification,
                    was_downgraded=was_downgraded,
                    reason=(
                        f"Rule layer finalised as {rule_verdict.classification}"
                    ),
                    artefact_pointers=[a.pointer for a in rule_verdict.evidence],
                )
                if was_downgraded:
                    logger.info(
                        "VERIFIER [rule] DOWNGRADED claim '%s': %s → %s",
                        draft.claim_id,
                        draft.classification,
                        rule_verdict.classification,
                    )
                else:
                    logger.debug(
                        "VERIFIER [rule] confirmed claim '%s' as %s",
                        draft.claim_id, rule_verdict.classification,
                    )
                final_verdicts.append(rule_verdict)
                decisions.append(decision)
            else:
                # Layer 1 could not resolve — queue for Layer 2
                # Placeholder to preserve ordering
                final_verdicts.append(draft)  # will be replaced
                llm_queue.append((idx, draft, mc.candidates, mc.claim))

        # ── Layer 2: constrained LLM (concurrent) ───────────────────────────
        if llm_queue:
            logger.info(
                "VERIFIER [llm] checking %d claims concurrently", len(llm_queue)
            )
            llm_tasks = [
                self._llm_verifier.verify(draft, candidates)
                for _, draft, candidates, _ in llm_queue
            ]
            llm_results: list[Verdict] = list(
                await asyncio.gather(*llm_tasks)
            )

            for (idx, draft, _candidates, claim), final in zip(
                llm_queue, llm_results, strict=True
            ):
                was_downgraded = (
                    final.classification != draft.classification
                )
                decision = VerifierDecision(
                    claim_id=draft.claim_id,
                    claim_text=claim.text,
                    layer="llm",
                    draft_classification=draft.classification,
                    final_classification=final.classification,
                    was_downgraded=was_downgraded,
                    reason=(
                        f"LLM {'downgraded' if was_downgraded else 'confirmed'} "
                        f"as {final.classification}"
                    ),
                    artefact_pointers=[a.pointer for a in final.evidence],
                )
                if was_downgraded:
                    logger.info(
                        "VERIFIER [llm] DOWNGRADED claim '%s': %s → %s",
                        draft.claim_id,
                        draft.classification,
                        final.classification,
                    )
                final_verdicts[idx] = final
                decisions.append(decision)

        return final_verdicts, decisions
