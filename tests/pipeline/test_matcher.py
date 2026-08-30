"""
Unit tests for ``EvidenceMatcher`` and ``FirstPassClassifier``.

All LLM calls are mocked — zero network I/O.

Coverage
--------
EvidenceMatcher
  - ``likely_unverifiable`` claims are skipped with empty candidates.
  - Non-skipped claims receive ranked candidates.
  - Keyword scoring ranks relevant artefacts higher.
  - ``build_unverifiable_verdicts`` produces correct schema-valid verdicts.

FirstPassClassifier
  - Valid classifier JSON is parsed into correct Verdict.
  - Evidence pointers are resolved back to full Artefact objects.
  - Missing pointer causes downgrade to unverifiable.
  - Malformed JSON is handled gracefully.
  - LLM errors are handled gracefully (default to unverifiable).
  - Skipped claims are excluded from LLM calls.
  - All active claims are processed concurrently (gather called once).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from devfit.github.bundle import ArtefactBundle
from devfit.pipeline.classifier import FirstPassClassifier, _parse_classifier_response
from devfit.pipeline.matcher import EvidenceMatcher, MatchedClaim
from devfit.schema import (
    Artefact,
    ArtefactType,
    Claim,
    ClaimCategory,
    ClaimSource,
    Classification,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def python_claim() -> Claim:
    """Return a verifiable Python technical-skill claim."""
    return Claim(
        id="t-001",
        text="Expert Python developer",
        source=ClaimSource.JD_REQUIREMENT,
        category=ClaimCategory.TECHNICAL_SKILL,
        likely_unverifiable=False,
    )


@pytest.fixture()
def soft_claim() -> Claim:
    """Return a soft-skill claim flagged likely_unverifiable."""
    return Claim(
        id="t-002",
        text="Excellent communication and leadership skills",
        source=ClaimSource.JD_REQUIREMENT,
        category=ClaimCategory.SOFT_SKILL,
        likely_unverifiable=True,
    )


@pytest.fixture()
def python_artefact() -> Artefact:
    """Return a Python language-stats artefact."""
    return Artefact(
        type=ArtefactType.LANGUAGE_STATS,
        pointer="github.com/testuser",
        extracted_fact="python (300,000 bytes), javascript (20,000 bytes)",
    )


@pytest.fixture()
def repo_artefact() -> Artefact:
    """Return a repo artefact unrelated to Python."""
    return Artefact(
        type=ArtefactType.REPO,
        pointer="github.com/testuser/go-server",
        extracted_fact="go-server (Go, 5 stars, updated 2024-01-01)",
    )


@pytest.fixture()
def bundle(
    python_artefact: Artefact, repo_artefact: Artefact
) -> ArtefactBundle:
    """Return a bundle with two artefacts."""
    return ArtefactBundle(artefacts=[python_artefact, repo_artefact])


# ---------------------------------------------------------------------------
# EvidenceMatcher tests
# ---------------------------------------------------------------------------


class TestEvidenceMatcher:
    """Tests for ``EvidenceMatcher``."""

    def test_skips_likely_unverifiable_claims(
        self, soft_claim: Claim, bundle: ArtefactBundle
    ) -> None:
        """Claims flagged likely_unverifiable must be skipped with no candidates."""
        matcher = EvidenceMatcher()
        results = matcher.match([soft_claim], bundle)

        assert len(results) == 1
        assert results[0].skipped is True
        assert results[0].candidates == []

    def test_matches_verifiable_claims_with_candidates(
        self, python_claim: Claim, bundle: ArtefactBundle
    ) -> None:
        """Verifiable claims must receive candidate artefacts."""
        matcher = EvidenceMatcher()
        results = matcher.match([python_claim], bundle)

        assert len(results) == 1
        assert results[0].skipped is False
        assert len(results[0].candidates) > 0

    def test_keyword_ranking_puts_relevant_artefact_first(
        self,
        python_claim: Claim,
        python_artefact: Artefact,
        repo_artefact: Artefact,
        bundle: ArtefactBundle,
    ) -> None:
        """Python language-stats artefact must rank above Go repo for a Python claim."""
        matcher = EvidenceMatcher()
        results = matcher.match([python_claim], bundle)
        # python_artefact shares 'python' token with claim; go-server does not
        assert results[0].candidates[0].pointer == python_artefact.pointer

    def test_build_unverifiable_verdicts_for_skipped(
        self, soft_claim: Claim, bundle: ArtefactBundle
    ) -> None:
        """Skipped claims must produce schema-valid UNVERIFIABLE verdicts."""
        matcher = EvidenceMatcher()
        matched = matcher.match([soft_claim], bundle)
        verdicts = matcher.build_unverifiable_verdicts(matched)

        assert len(verdicts) == 1
        v = verdicts[0]
        assert v.claim_id == soft_claim.id
        assert v.classification == Classification.UNVERIFIABLE
        assert v.evidence == []
        assert v.confidence == 1.0

    def test_mixed_claims_preserves_order(
        self,
        python_claim: Claim,
        soft_claim: Claim,
        bundle: ArtefactBundle,
    ) -> None:
        """Output order must match input claim order."""
        matcher = EvidenceMatcher()
        results = matcher.match([python_claim, soft_claim], bundle)

        assert results[0].claim.id == python_claim.id
        assert results[1].claim.id == soft_claim.id

    def test_empty_bundle_returns_empty_candidates(
        self, python_claim: Claim
    ) -> None:
        """An empty bundle must return empty candidates, not raise."""
        matcher = EvidenceMatcher()
        empty_bundle = ArtefactBundle(artefacts=[])
        results = matcher.match([python_claim], empty_bundle)

        assert results[0].candidates == []
        assert results[0].skipped is False


# ---------------------------------------------------------------------------
# _parse_classifier_response unit tests
# ---------------------------------------------------------------------------


class TestParseClassifierResponse:
    """Tests for the ``_parse_classifier_response`` helper."""

    def test_parses_verified_with_pointer(
        self, python_artefact: Artefact
    ) -> None:
        """Valid verified response with matching pointer must produce Verdict."""
        raw = json.dumps({
            "claim_id": "t-001",
            "classification": "verified",
            "confidence": 0.9,
            "evidence_pointers": [python_artefact.pointer],
            "reasoning": "Language stats confirm Python expertise.",
        })
        verdict = _parse_classifier_response(raw, "t-001", [python_artefact])
        assert verdict.classification == Classification.VERIFIED
        assert verdict.confidence == 0.9
        assert len(verdict.evidence) == 1
        assert verdict.evidence[0].pointer == python_artefact.pointer

    def test_downgrade_to_unverifiable_when_pointer_missing(
        self, python_artefact: Artefact
    ) -> None:
        """Verified response with pointer not in candidates must downgrade."""
        raw = json.dumps({
            "classification": "verified",
            "confidence": 0.9,
            "evidence_pointers": ["github.com/nonexistent"],
        })
        verdict = _parse_classifier_response(raw, "t-001", [python_artefact])
        assert verdict.classification == Classification.UNVERIFIABLE

    def test_unverifiable_with_empty_pointers(self) -> None:
        """Unverifiable classification with no pointers must be valid."""
        raw = json.dumps({
            "classification": "unverifiable",
            "confidence": 0.8,
            "evidence_pointers": [],
        })
        verdict = _parse_classifier_response(raw, "t-001", [])
        assert verdict.classification == Classification.UNVERIFIABLE
        assert verdict.evidence == []

    def test_malformed_json_returns_unverifiable(self) -> None:
        """Malformed JSON must return unverifiable verdict, not raise."""
        verdict = _parse_classifier_response("NOT JSON {{", "t-001", [])
        assert verdict.classification == Classification.UNVERIFIABLE
        assert verdict.confidence == 0.0

    def test_strips_markdown_fence(
        self, python_artefact: Artefact
    ) -> None:
        """Responses wrapped in ```json fences must be parsed correctly."""
        inner = json.dumps({
            "classification": "verified",
            "confidence": 0.85,
            "evidence_pointers": [python_artefact.pointer],
        })
        fenced = f"```json\n{inner}\n```"
        verdict = _parse_classifier_response(fenced, "t-001", [python_artefact])
        assert verdict.classification == Classification.VERIFIED


# ---------------------------------------------------------------------------
# FirstPassClassifier tests (Groq mocked)
# ---------------------------------------------------------------------------


def _make_groq_response(content: str) -> MagicMock:
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


class TestFirstPassClassifier:
    """Tests for ``FirstPassClassifier`` with mocked Groq."""

    @pytest.fixture()
    def mock_groq(self) -> MagicMock:
        """
        Patch AsyncGroq so no real API calls are made.

        Returns
        -------
        MagicMock
            Mock client with ``chat.completions.create`` as AsyncMock.
        """
        with patch("devfit.pipeline.classifier.AsyncGroq") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat.completions.create = AsyncMock()
            mock_cls.return_value = mock_instance
            yield mock_instance

    async def test_classify_returns_draft_verdicts(
        self,
        mock_groq: MagicMock,
        python_claim: Claim,
        python_artefact: Artefact,
    ) -> None:
        """Classifier must return one draft verdict per active matched claim."""
        payload = json.dumps({
            "classification": "verified",
            "confidence": 0.88,
            "evidence_pointers": [python_artefact.pointer],
        })
        mock_groq.chat.completions.create.return_value = (
            _make_groq_response(payload)
        )
        mc = MatchedClaim(claim=python_claim, candidates=[python_artefact])
        classifier = FirstPassClassifier()
        verdicts = await classifier.classify([mc])

        assert len(verdicts) == 1
        assert verdicts[0].classification == Classification.VERIFIED
        assert verdicts[0].llm_confirmed is False  # proposer, not confirmer

    async def test_classify_skips_skipped_claims(
        self,
        mock_groq: MagicMock,
        soft_claim: Claim,
    ) -> None:
        """Skipped claims must not produce an LLM call."""
        mc = MatchedClaim(claim=soft_claim, candidates=[], skipped=True)
        classifier = FirstPassClassifier()
        verdicts = await classifier.classify([mc])

        assert verdicts == []
        mock_groq.chat.completions.create.assert_not_called()

    async def test_classify_handles_llm_error_gracefully(
        self,
        mock_groq: MagicMock,
        python_claim: Claim,
        python_artefact: Artefact,
    ) -> None:
        """LLM errors must produce unverifiable verdict, not raise."""
        mock_groq.chat.completions.create.side_effect = RuntimeError("timeout")
        mc = MatchedClaim(claim=python_claim, candidates=[python_artefact])
        classifier = FirstPassClassifier()
        verdicts = await classifier.classify([mc])

        assert len(verdicts) == 1
        assert verdicts[0].classification == Classification.UNVERIFIABLE

    async def test_classify_uses_temperature_zero(
        self,
        mock_groq: MagicMock,
        python_claim: Claim,
    ) -> None:
        """All LLM calls must use temperature=0.0."""
        mock_groq.chat.completions.create.return_value = (
            _make_groq_response('{"classification":"unverifiable","confidence":0,"evidence_pointers":[]}')
        )
        mc = MatchedClaim(claim=python_claim, candidates=[])
        classifier = FirstPassClassifier()
        await classifier.classify([mc])

        kwargs = mock_groq.chat.completions.create.call_args.kwargs
        assert kwargs.get("temperature") == 0.0
