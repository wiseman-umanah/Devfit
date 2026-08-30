"""
Unit tests for ``JDAnalyzer`` and ``ResumeAnalyzer``.

All tests mock the Groq client — zero real LLM calls, zero network I/O.

Coverage
--------
- Correct claim source and category assignment.
- Auto-flagging of ``likely_unverifiable`` for soft_skill and experience_duration.
- Markdown code-fence stripping from LLM response.
- Graceful handling of malformed JSON from the LLM (empty list, no exception).
- ``ResumeAnalyzer`` assigns ``ClaimSource.RESUME``.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from devfit.pipeline.analyzer import (
    JDAnalyzer,
    ResumeAnalyzer,
    _parse_claims_response,
)
from devfit.schema import ClaimCategory, ClaimSource

# ---------------------------------------------------------------------------
# _parse_claims_response unit tests (pure function — no mocking needed)
# ---------------------------------------------------------------------------


class TestParseClaimsResponse:
    """Tests for the ``_parse_claims_response`` helper."""

    def test_parses_valid_array(self) -> None:
        """A valid JSON array produces correctly typed Claim objects."""
        raw = json.dumps([
            {
                "id": "c-001",
                "text": "Expert Python developer",
                "source": "jd_requirement",
                "category": "technical_skill",
                "likely_unverifiable": False,
            }
        ])
        claims = _parse_claims_response(raw, ClaimSource.JD_REQUIREMENT, "jd")
        assert len(claims) == 1
        assert claims[0].text == "Expert Python developer"
        assert claims[0].source == ClaimSource.JD_REQUIREMENT
        assert claims[0].category == ClaimCategory.TECHNICAL_SKILL
        assert claims[0].likely_unverifiable is False

    def test_auto_flags_soft_skill_as_unverifiable(self) -> None:
        """SOFT_SKILL claims must be auto-flagged even if LLM sets it False."""
        raw = json.dumps([
            {
                "text": "Excellent team leadership skills",
                "category": "soft_skill",
                "likely_unverifiable": False,  # LLM got this wrong
            }
        ])
        claims = _parse_claims_response(raw, ClaimSource.JD_REQUIREMENT, "jd")
        assert claims[0].likely_unverifiable is True

    def test_auto_flags_experience_duration_as_unverifiable(self) -> None:
        """EXPERIENCE_DURATION claims must be auto-flagged."""
        raw = json.dumps([
            {
                "text": "5+ years of professional Python experience",
                "category": "experience_duration",
                "likely_unverifiable": False,
            }
        ])
        claims = _parse_claims_response(raw, ClaimSource.JD_REQUIREMENT, "jd")
        assert claims[0].likely_unverifiable is True

    def test_strips_markdown_code_fence(self) -> None:
        """LLM responses wrapped in ```json ... ``` must be parsed correctly."""
        inner = json.dumps([
            {"text": "Python expert", "category": "technical_skill"}
        ])
        fenced = f"```json\n{inner}\n```"
        claims = _parse_claims_response(fenced, ClaimSource.JD_REQUIREMENT, "jd")
        assert len(claims) == 1

    def test_returns_empty_list_on_invalid_json(self) -> None:
        """Malformed JSON must return an empty list without raising."""
        claims = _parse_claims_response(
            "NOT VALID JSON {{{{", ClaimSource.JD_REQUIREMENT, "jd"
        )
        assert claims == []

    def test_assigns_stable_id_prefix(self) -> None:
        """Each claim ID must start with the provided prefix."""
        raw = json.dumps([
            {"text": "Go expertise", "category": "technical_skill"},
            {"text": "5 years Go", "category": "experience_duration"},
        ])
        claims = _parse_claims_response(raw, ClaimSource.JD_REQUIREMENT, "myprefix")
        assert claims[0].id.startswith("myprefix-")
        assert claims[1].id.startswith("myprefix-")
        # IDs must be unique
        assert claims[0].id != claims[1].id

    def test_unknown_category_falls_back_to_other(self) -> None:
        """An unrecognised category string must fall back to ``OTHER``."""
        raw = json.dumps([
            {"text": "Some claim", "category": "invented_category"}
        ])
        claims = _parse_claims_response(raw, ClaimSource.RESUME, "cv")
        assert claims[0].category == ClaimCategory.OTHER

    def test_resume_source_preserved(self) -> None:
        """Claims parsed with RESUME source must have that source set."""
        raw = json.dumps([
            {"text": "Built a microservice in Python", "category": "technical_skill"}
        ])
        claims = _parse_claims_response(raw, ClaimSource.RESUME, "cv")
        assert claims[0].source == ClaimSource.RESUME


# ---------------------------------------------------------------------------
# JDAnalyzer tests (Groq client mocked)
# ---------------------------------------------------------------------------


def _make_mock_groq_response(content: str) -> MagicMock:
    """
    Build a mock Groq completion response with the given content string.

    Parameters
    ----------
    content : str
        The text the mock LLM should return.

    Returns
    -------
    MagicMock
        Mimics ``groq.types.chat.ChatCompletion``.
    """
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


class TestJDAnalyzer:
    """Tests for ``JDAnalyzer`` with a mocked Groq client."""

    @pytest.fixture()
    def mock_groq(self) -> MagicMock:
        """
        Patch ``AsyncGroq`` so no real API calls are made.

        Returns
        -------
        MagicMock
            Mock client whose ``chat.completions.create`` is an ``AsyncMock``.
        """
        with patch("devfit.pipeline.analyzer.AsyncGroq") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat = MagicMock()
            mock_instance.chat.completions = MagicMock()
            mock_instance.chat.completions.create = AsyncMock()
            mock_cls.return_value = mock_instance
            yield mock_instance

    async def test_analyze_returns_claims(self, mock_groq: MagicMock) -> None:
        """Analyzer must return a non-empty list for a typical JD."""
        payload = json.dumps([
            {
                "text": "Expert Python developer",
                "category": "technical_skill",
                "likely_unverifiable": False,
            },
            {
                "text": "5+ years of experience",
                "category": "experience_duration",
                "likely_unverifiable": True,
            },
        ])
        mock_groq.chat.completions.create.return_value = (
            _make_mock_groq_response(payload)
        )

        analyzer = JDAnalyzer()
        claims = await analyzer.analyze(
            "We need an expert Python developer with 5+ years."
        )

        assert len(claims) == 2
        assert all(c.source == ClaimSource.JD_REQUIREMENT for c in claims)

    async def test_analyze_uses_temperature_zero(self, mock_groq: MagicMock) -> None:
        """The LLM call must use temperature=0.0 for deterministic output."""
        mock_groq.chat.completions.create.return_value = (
            _make_mock_groq_response("[]")
        )
        analyzer = JDAnalyzer()
        await analyzer.analyze("Some JD")

        call_kwargs = mock_groq.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == 0.0

    async def test_analyze_graceful_on_bad_json(self, mock_groq: MagicMock) -> None:
        """Malformed LLM response must return an empty list, not raise."""
        mock_groq.chat.completions.create.return_value = (
            _make_mock_groq_response("I cannot help with that request.")
        )
        analyzer = JDAnalyzer()
        claims = await analyzer.analyze("Some JD")
        assert claims == []

    async def test_soft_skill_auto_flagged(self, mock_groq: MagicMock) -> None:
        """Soft skill claims must be auto-flagged regardless of LLM output."""
        payload = json.dumps([
            {
                "text": "Strong communication skills",
                "category": "soft_skill",
                "likely_unverifiable": False,
            }
        ])
        mock_groq.chat.completions.create.return_value = (
            _make_mock_groq_response(payload)
        )
        analyzer = JDAnalyzer()
        claims = await analyzer.analyze("JD text")
        assert claims[0].likely_unverifiable is True


# ---------------------------------------------------------------------------
# ResumeAnalyzer tests (Groq client mocked)
# ---------------------------------------------------------------------------


class TestResumeAnalyzer:
    """Tests for ``ResumeAnalyzer`` with a mocked Groq client."""

    @pytest.fixture()
    def mock_groq(self) -> MagicMock:
        """
        Patch ``AsyncGroq`` for ResumeAnalyzer tests.

        Returns
        -------
        MagicMock
            Mock with ``chat.completions.create`` as an ``AsyncMock``.
        """
        with patch("devfit.pipeline.analyzer.AsyncGroq") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.chat = MagicMock()
            mock_instance.chat.completions = MagicMock()
            mock_instance.chat.completions.create = AsyncMock()
            mock_cls.return_value = mock_instance
            yield mock_instance

    async def test_analyze_assigns_resume_source(
        self, mock_groq: MagicMock
    ) -> None:
        """Resume claims must have ``source=ClaimSource.RESUME``."""
        payload = json.dumps([
            {
                "text": "Built a REST API using FastAPI",
                "category": "technical_skill",
                "likely_unverifiable": False,
            }
        ])
        mock_groq.chat.completions.create.return_value = (
            _make_mock_groq_response(payload)
        )
        analyzer = ResumeAnalyzer()
        claims = await analyzer.analyze("I built a REST API using FastAPI.")
        assert claims[0].source == ClaimSource.RESUME

    async def test_analyze_empty_resume_returns_empty(
        self, mock_groq: MagicMock
    ) -> None:
        """An empty resume response from the LLM must return an empty list."""
        mock_groq.chat.completions.create.return_value = (
            _make_mock_groq_response("[]")
        )
        analyzer = ResumeAnalyzer()
        claims = await analyzer.analyze("Minimal resume.")
        assert claims == []
