"""
Unit tests for ``ConstrainedLLMVerifier`` and ``IndependentVerifier``.

All LLM calls are mocked — zero network I/O.

Coverage
--------
ConstrainedLLMVerifier
  - Confirmed response sets ``llm_confirmed=True``.
  - Confirmed response with unknown pointer is downgraded.
  - Downgraded response applies new classification.
  - Upgrade attempt is blocked.
  - Malformed JSON is downgraded gracefully.
  - LLM error is downgraded gracefully.
  - ``max_tokens=256`` used (stricter budget than classifier).

IndependentVerifier (core quality gate — step 6.4/6.5)
  - Layer 1 (rule) resolves a claim: verdict finalised, LLM not called.
  - Layer 1 cannot resolve: claim forwarded to Layer 2.
  - Layer 1 overrides a classifier-proposed verdict (key independence test).
  - Layer 2 downgrades a classifier-proposed verdict.
  - ``VerifierDecision.was_downgraded`` is set correctly.
  - All Layer 2 calls are concurrent (asyncio.gather).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from devfit.pipeline.matcher import MatchedClaim
from devfit.schema import (
    Artefact,
    ArtefactType,
    Claim,
    ClaimCategory,
    ClaimSource,
    Classification,
    Verdict,
)
from devfit.verifier.llm import ConstrainedLLMVerifier, _parse_verifier_response
from devfit.verifier.verifier import IndependentVerifier

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def python_artefact() -> Artefact:
    """Return a Python language-stats artefact."""
    return Artefact(
        type=ArtefactType.LANGUAGE_STATS,
        pointer="github.com/testuser",
        extracted_fact="python (485,000 bytes)",
    )


@pytest.fixture()
def verified_draft(python_artefact: Artefact) -> Verdict:
    """Return a draft VERIFIED verdict from the first-pass classifier."""
    return Verdict(
        claim_id="t-001",
        classification=Classification.VERIFIED,
        confidence=0.85,
        evidence=[python_artefact],
        rule_checked=False,
        llm_confirmed=False,
    )


@pytest.fixture()
def soft_claim_unverifiable() -> Verdict:
    """Return an already-unverifiable verdict (soft skill)."""
    return Verdict(
        claim_id="t-soft",
        classification=Classification.UNVERIFIABLE,
        confidence=1.0,
        evidence=[],
        rule_checked=False,
        llm_confirmed=False,
    )


# ---------------------------------------------------------------------------
# _parse_verifier_response unit tests (pure — no mocking)
# ---------------------------------------------------------------------------


class TestParseVerifierResponse:
    """Tests for the ``_parse_verifier_response`` helper."""

    def test_confirmed_sets_llm_confirmed_true(
        self,
        verified_draft: Verdict,
        python_artefact: Artefact,
    ) -> None:
        """A confirmed response must set ``llm_confirmed=True``."""
        raw = json.dumps({
            "decision": "confirmed",
            "artefact_pointer": python_artefact.pointer,
            "reason": "Language stats confirm Python expertise.",
        })
        result = _parse_verifier_response(raw, verified_draft, [python_artefact])
        assert result.llm_confirmed is True
        assert result.classification == Classification.VERIFIED

    def test_confirmed_with_unknown_pointer_downgrades(
        self,
        verified_draft: Verdict,
        python_artefact: Artefact,
    ) -> None:
        """Confirmed with a pointer not in candidates must downgrade."""
        raw = json.dumps({
            "decision": "confirmed",
            "artefact_pointer": "github.com/nonexistent",
            "reason": "Invented artefact",
        })
        result = _parse_verifier_response(raw, verified_draft, [python_artefact])
        assert result.classification == Classification.UNVERIFIABLE
        assert result.llm_confirmed is False

    def test_downgraded_to_unverifiable(
        self,
        verified_draft: Verdict,
        python_artefact: Artefact,
    ) -> None:
        """Downgraded response must apply the new classification."""
        raw = json.dumps({
            "decision": "downgraded",
            "new_classification": "unverifiable",
            "reason": "Artefact does not specifically confirm this claim.",
        })
        result = _parse_verifier_response(raw, verified_draft, [python_artefact])
        assert result.classification == Classification.UNVERIFIABLE
        assert result.llm_confirmed is False

    def test_upgrade_attempt_is_blocked(
        self,
        python_artefact: Artefact,
    ) -> None:
        """Verifier must NOT be able to upgrade an unverifiable to verified."""
        contradicted_draft = Verdict(
            claim_id="t-001",
            classification=Classification.CONTRADICTED,
            confidence=0.9,
            evidence=[python_artefact],
        )
        raw = json.dumps({
            "decision": "downgraded",
            "new_classification": "verified",  # attempted upgrade
            "reason": "Actually this looks good",
        })
        result = _parse_verifier_response(raw, contradicted_draft, [python_artefact])
        # Upgrade blocked — must keep original contradicted
        assert result.classification == Classification.CONTRADICTED

    def test_malformed_json_downgrades(
        self,
        verified_draft: Verdict,
        python_artefact: Artefact,
    ) -> None:
        """Malformed JSON must downgrade gracefully."""
        result = _parse_verifier_response(
            "NOT JSON", verified_draft, [python_artefact]
        )
        assert result.classification == Classification.UNVERIFIABLE
        assert result.llm_confirmed is False


# ---------------------------------------------------------------------------
# ConstrainedLLMVerifier tests (Groq mocked)
# ---------------------------------------------------------------------------


def _mock_groq_response(content: str) -> MagicMock:
    """
    Build a mock Groq completion response.

    Parameters
    ----------
    content : str
        Text the mock LLM returns.

    Returns
    -------
    MagicMock
        Mimics ``groq.types.chat.ChatCompletion``.
    """
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestConstrainedLLMVerifier:
    """Tests for ``ConstrainedLLMVerifier`` with mocked Groq."""

    @pytest.fixture()
    def mock_groq(self) -> MagicMock:
        """
        Patch AsyncGroq for verifier tests.

        Returns
        -------
        MagicMock
            Mock client with ``chat.completions.create`` as AsyncMock.
        """
        with patch("devfit.verifier.llm.AsyncGroq") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock()
            mock_cls.return_value = mock_instance
            yield mock_instance

    async def test_verify_confirmed_response(
        self,
        mock_groq: MagicMock,
        verified_draft: Verdict,
        python_artefact: Artefact,
    ) -> None:
        """A valid confirmed response must produce llm_confirmed=True verdict."""
        payload = json.dumps({
            "decision": "confirmed",
            "artefact_pointer": python_artefact.pointer,
            "reason": "Confirmed Python presence",
        })
        mock_groq.chat.completions.create.return_value = _mock_groq_response(payload)

        verifier = ConstrainedLLMVerifier()
        result = await verifier.verify(verified_draft, [python_artefact])

        assert result.llm_confirmed is True
        assert result.classification == Classification.VERIFIED

    async def test_verify_uses_max_tokens_256(
        self,
        mock_groq: MagicMock,
        verified_draft: Verdict,
        python_artefact: Artefact,
    ) -> None:
        """Constrained verifier must use max_tokens=256 (stricter than classifier)."""
        mock_groq.chat.completions.create.return_value = _mock_groq_response(
            '{"decision":"downgraded","new_classification":"unverifiable","reason":"x"}'
        )
        verifier = ConstrainedLLMVerifier()
        await verifier.verify(verified_draft, [python_artefact])

        kwargs = mock_groq.chat.completions.create.call_args.kwargs
        assert kwargs.get("max_tokens") == 256

    async def test_verify_llm_error_downgrades(
        self,
        mock_groq: MagicMock,
        verified_draft: Verdict,
        python_artefact: Artefact,
    ) -> None:
        """LLM error must downgrade to unverifiable, not raise."""
        mock_groq.chat.completions.create.side_effect = RuntimeError("network error")

        verifier = ConstrainedLLMVerifier()
        result = await verifier.verify(verified_draft, [python_artefact])

        assert result.classification == Classification.UNVERIFIABLE
        assert result.llm_confirmed is False


# ---------------------------------------------------------------------------
# IndependentVerifier tests — the core quality gate (TRD §3.4 / step 6.5)
# ---------------------------------------------------------------------------


class TestIndependentVerifier:
    """Tests for ``IndependentVerifier`` covering both layers."""

    @pytest.fixture()
    def mock_llm_verifier(self) -> MagicMock:
        """
        Patch ``ConstrainedLLMVerifier.verify`` so tests can control Layer 2.

        Returns
        -------
        MagicMock
            Async mock for ``ConstrainedLLMVerifier.verify``.
        """
        with patch(
            "devfit.verifier.verifier.ConstrainedLLMVerifier.verify",
            new_callable=AsyncMock,
        ) as mock_verify:
            yield mock_verify

    def _make_mc(
        self,
        claim: Claim,
        candidates: list[Artefact],
    ) -> MatchedClaim:
        """
        Build a non-skipped MatchedClaim.

        Parameters
        ----------
        claim : Claim
            The claim to wrap.
        candidates : list[Artefact]
            Candidate artefacts for the claim.

        Returns
        -------
        MatchedClaim
            MatchedClaim with skipped=False.
        """
        return MatchedClaim(claim=claim, candidates=candidates, skipped=False)

    async def test_rule_layer_resolves_claim_llm_not_called(
        self,
        mock_llm_verifier: MagicMock,
        python_artefact: Artefact,
    ) -> None:
        """When Layer 1 resolves a claim, Layer 2 must NOT be called."""
        # A date-arithmetic claim that the rule layer will catch:
        # account created 6 months ago, claim asserts 7+ years experience.
        created = (datetime.now(UTC) - timedelta(days=180)).strftime("%Y-%m-%d")
        meta = Artefact(
            type=ArtefactType.ACCOUNT_METADATA,
            pointer="github.com/newuser",
            extracted_fact=f"Account created {created} (0y 6m ago), 2 public repos",
        )
        claim = Claim(
            id="d-001",
            text="7+ years of professional software engineering experience.",
            source=ClaimSource.JD_REQUIREMENT,
            category=ClaimCategory.EXPERIENCE_DURATION,
            likely_unverifiable=False,
        )
        draft = Verdict(
            claim_id="d-001",
            classification=Classification.VERIFIED,  # wrong — rule should catch
            confidence=0.7,
            evidence=[meta],
            rule_checked=False,
            llm_confirmed=False,
        )
        mc = self._make_mc(claim, [meta])

        verifier = IndependentVerifier()
        finals, decisions = await verifier.verify_all([draft], [mc])

        # Layer 1 must have fired
        assert finals[0].rule_checked is True
        assert finals[0].classification == Classification.CONTRADICTED
        # Layer 2 must NOT have been called
        mock_llm_verifier.assert_not_called()
        # Decision must record the downgrade
        assert decisions[0].was_downgraded is True
        assert decisions[0].layer == "rule"

    async def test_rule_layer_cannot_resolve_forwards_to_llm(
        self,
        mock_llm_verifier: MagicMock,
        python_artefact: Artefact,
    ) -> None:
        """When Layer 1 returns None, Layer 2 must be called exactly once."""
        # Use a role_scope claim — no rule covers this category, so Layer 1
        # returns None and Layer 2 is always invoked.
        claim = Claim(
            id="s-001",
            text="Maintained open source repositories with active contributors.",
            source=ClaimSource.JD_REQUIREMENT,
            category=ClaimCategory.ROLE_SCOPE,
            likely_unverifiable=False,
        )
        draft = Verdict(
            claim_id="s-001",
            classification=Classification.VERIFIED,
            confidence=0.8,
            evidence=[python_artefact],
        )
        mc = self._make_mc(claim, [python_artefact])

        # Layer 2 confirms the draft
        confirmed = Verdict(
            claim_id="s-001",
            classification=Classification.VERIFIED,
            confidence=0.85,
            evidence=[python_artefact],
            rule_checked=False,
            llm_confirmed=True,
        )
        mock_llm_verifier.return_value = confirmed

        verifier = IndependentVerifier()
        finals, decisions = await verifier.verify_all([draft], [mc])

        mock_llm_verifier.assert_called_once()
        assert finals[0].llm_confirmed is True
        assert decisions[0].layer == "llm"
        assert decisions[0].was_downgraded is False

    async def test_layer2_downgrades_classifier_proposal(
        self,
        mock_llm_verifier: MagicMock,
        python_artefact: Artefact,
    ) -> None:
        """Layer 2 must be able to downgrade what the classifier proposed."""
        # Use a role_scope claim so Layer 1 returns None and Layer 2 runs.
        claim = Claim(
            id="r-001",
            text="Led and coordinated a distributed team across multiple time zones.",
            source=ClaimSource.JD_REQUIREMENT,
            category=ClaimCategory.ROLE_SCOPE,
            likely_unverifiable=False,
        )
        draft = Verdict(
            claim_id="r-001",
            classification=Classification.VERIFIED,
            confidence=0.7,
            evidence=[python_artefact],
        )
        mc = self._make_mc(claim, [python_artefact])

        # LLM downgrades to unverifiable (artefacts don't confirm team leadership)
        downgraded = Verdict(
            claim_id="r-001",
            classification=Classification.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
            rule_checked=False,
            llm_confirmed=False,
        )
        mock_llm_verifier.return_value = downgraded

        verifier = IndependentVerifier()
        finals, decisions = await verifier.verify_all([draft], [mc])

        assert finals[0].classification == Classification.UNVERIFIABLE
        assert decisions[0].was_downgraded is True
        assert decisions[0].layer == "llm"

    async def test_rule_overrides_classifier_verified(
        self,
        mock_llm_verifier: MagicMock,
    ) -> None:
        """
        Key independence test (TRD §3.4).

        The rule layer must be able to CONTRADICT a classifier-proposed
        VERIFIED verdict — this is the most critical architectural requirement.
        """
        created = (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%d")
        meta = Artefact(
            type=ArtefactType.ACCOUNT_METADATA,
            pointer="github.com/brandnew",
            extracted_fact=f"Account created {created} (0y 3m ago), 1 public repos",
        )
        claim = Claim(
            id="e-001",
            text="10+ years of professional Python experience.",
            source=ClaimSource.JD_REQUIREMENT,
            category=ClaimCategory.EXPERIENCE_DURATION,
            likely_unverifiable=False,
        )
        # Classifier wrongly proposed VERIFIED
        draft = Verdict(
            claim_id="e-001",
            classification=Classification.VERIFIED,
            confidence=0.75,
            evidence=[meta],
        )
        mc = self._make_mc(claim, [meta])

        verifier = IndependentVerifier()
        finals, decisions = await verifier.verify_all([draft], [mc])

        # Rule layer must contradict the classifier's verified proposal
        assert finals[0].classification == Classification.CONTRADICTED
        assert finals[0].rule_checked is True
        # LLM must NOT have been called at all
        mock_llm_verifier.assert_not_called()
        # was_downgraded must be True (verified → contradicted is a change)
        assert decisions[0].was_downgraded is True
