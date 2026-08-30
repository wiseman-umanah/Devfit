"""
Unit tests for the baseline pipeline.

Coverage
--------
build_github_summary
  - Empty bundle produces fallback string.
  - Account metadata and languages are included in summary.
  - Repo pointers included (capped at 5).
  - Contribution graph fact included.
  - Bundle with only repos (no account metadata) still works.

BaselinePipeline.run
  - CV and fit comment are split on --- separator.
  - Response with no --- puts all content in cv_markdown, empty fit_comment.
  - raw_response preserved exactly.
  - Prompt includes jd_text and github_summary.
  - resume_section present when resume_text supplied.
  - resume_section empty when resume_text is None.
  - Returns BaselineResult dataclass (frozen, correct fields).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from devfit.baseline.pipeline import (
    BaselinePipeline,
    BaselineResult,
    build_github_summary,
)
from devfit.github.bundle import ArtefactBundle
from devfit.schema import Artefact, ArtefactType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_artefact(
    artefact_type: ArtefactType,
    pointer: str,
    extracted_fact: str = "",
) -> Artefact:
    """Build a minimal Artefact for test use."""
    return Artefact(
        type=artefact_type,
        pointer=pointer,
        extracted_fact=extracted_fact,
    )


@pytest.fixture()
def empty_bundle() -> ArtefactBundle:
    """Bundle with no artefacts."""
    return ArtefactBundle(artefacts=[])


@pytest.fixture()
def rich_bundle() -> ArtefactBundle:
    """Bundle with account metadata, languages, repos, and contribution graph."""
    return ArtefactBundle(
        artefacts=[
            _make_artefact(
                ArtefactType.ACCOUNT_METADATA,
                "torvalds",
                "Account created 1996, over 28 years of activity.",
            ),
            _make_artefact(
                ArtefactType.LANGUAGE_STATS,
                "torvalds/language_stats",
                "C (97%), Makefile (2%)",
            ),
            _make_artefact(ArtefactType.REPO, "torvalds/linux"),
            _make_artefact(ArtefactType.REPO, "torvalds/subsurface"),
            _make_artefact(
                ArtefactType.CONTRIBUTION_GRAPH,
                "torvalds/contributions",
                "312 contributions in the last year.",
            ),
        ]
    )


@pytest.fixture()
def mock_groq_response() -> MagicMock:
    """Return a mock Groq completion response that includes the --- separator."""
    msg = MagicMock()
    msg.content = (
        "# Tailored CV\n\nSenior engineer.\n\n---\n\nStrong fit for kernel role."
    )
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# build_github_summary tests
# ---------------------------------------------------------------------------


class TestBuildGithubSummary:
    """Tests for the unstructured GitHub summary builder."""

    def test_empty_bundle_returns_fallback(
        self, empty_bundle: ArtefactBundle
    ) -> None:
        """Empty bundle must produce the fallback message."""
        summary = build_github_summary(empty_bundle)
        assert summary == "No GitHub data available."

    def test_account_metadata_included(self, rich_bundle: ArtefactBundle) -> None:
        """Account metadata pointer and extracted_fact must appear in summary."""
        summary = build_github_summary(rich_bundle)
        assert "torvalds" in summary
        assert "28 years" in summary

    def test_languages_included(self, rich_bundle: ArtefactBundle) -> None:
        """Language facts must appear in summary."""
        summary = build_github_summary(rich_bundle)
        assert "C (97%)" in summary

    def test_repos_included(self, rich_bundle: ArtefactBundle) -> None:
        """Repo pointers must appear in summary."""
        summary = build_github_summary(rich_bundle)
        assert "torvalds/linux" in summary

    def test_contribution_graph_included(self, rich_bundle: ArtefactBundle) -> None:
        """Contribution graph extracted_fact must appear in summary."""
        summary = build_github_summary(rich_bundle)
        assert "312 contributions" in summary

    def test_repos_only_bundle(self) -> None:
        """Bundle with only repos (no metadata) must not crash."""
        bundle = ArtefactBundle(
            artefacts=[
                _make_artefact(ArtefactType.REPO, "user/repo-a"),
                _make_artefact(ArtefactType.REPO, "user/repo-b"),
            ]
        )
        summary = build_github_summary(bundle)
        assert "user/repo-a" in summary
        assert "No GitHub data available." not in summary

    def test_repo_cap_at_five(self) -> None:
        """Only the first 5 repos should appear in the summary."""
        artefacts = [
            _make_artefact(ArtefactType.REPO, f"user/repo-{i}")
            for i in range(10)
        ]
        bundle = ArtefactBundle(artefacts=artefacts)
        summary = build_github_summary(bundle)
        for i in range(5, 10):
            assert f"repo-{i}" not in summary


# ---------------------------------------------------------------------------
# BaselinePipeline.run tests
# ---------------------------------------------------------------------------


class TestBaselinePipeline:
    """Tests for the BaselinePipeline class."""

    def _make_pipeline(self) -> BaselinePipeline:
        """Return a BaselinePipeline with Groq client creation mocked out."""
        with (
            patch("devfit.baseline.pipeline.get_settings") as mock_settings,
            patch("devfit.baseline.pipeline.AsyncGroq"),
        ):
            mock_settings.return_value.groq_api_key.get_secret_value.return_value = (
                "fake-key"
            )
            mock_settings.return_value.groq_model = "openai/gpt-oss-120b"
            return BaselinePipeline()

    @pytest.mark.asyncio()
    async def test_splits_on_separator(
        self,
        empty_bundle: ArtefactBundle,
        mock_groq_response: MagicMock,
    ) -> None:
        """CV and fit comment are split correctly on the --- separator."""
        pipeline = self._make_pipeline()
        pipeline._client = AsyncMock()
        pipeline._client.chat.completions.create = AsyncMock(
            return_value=mock_groq_response
        )

        result = await pipeline.run("Senior engineer needed.", empty_bundle)

        assert "Tailored CV" in result.cv_markdown
        assert "Strong fit" in result.fit_comment

    @pytest.mark.asyncio()
    async def test_no_separator_all_in_cv(
        self,
        empty_bundle: ArtefactBundle,
    ) -> None:
        """When no --- separator is present, all content goes into cv_markdown."""
        msg = MagicMock()
        msg.content = "Just a plain CV with no separator."
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]

        pipeline = self._make_pipeline()
        pipeline._client = AsyncMock()
        pipeline._client.chat.completions.create = AsyncMock(return_value=resp)

        result = await pipeline.run("Some JD.", empty_bundle)

        assert result.cv_markdown == "Just a plain CV with no separator."
        assert result.fit_comment == ""

    @pytest.mark.asyncio()
    async def test_raw_response_preserved(
        self,
        empty_bundle: ArtefactBundle,
        mock_groq_response: MagicMock,
    ) -> None:
        """raw_response must equal the full LLM output."""
        pipeline = self._make_pipeline()
        pipeline._client = AsyncMock()
        pipeline._client.chat.completions.create = AsyncMock(
            return_value=mock_groq_response
        )

        result = await pipeline.run("JD text.", empty_bundle)

        assert result.raw_response == mock_groq_response.choices[0].message.content

    @pytest.mark.asyncio()
    async def test_resume_section_present_when_supplied(
        self,
        empty_bundle: ArtefactBundle,
    ) -> None:
        """The prompt must include the resume text when resume_text is provided."""
        msg = MagicMock()
        msg.content = "CV content\n---\nFit comment"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]

        captured_prompt: list[str] = []

        async def capture(  # type: ignore[return]
            model: str, messages: list, **kwargs: object
        ) -> object:
            captured_prompt.append(messages[0]["content"])
            return resp

        pipeline = self._make_pipeline()
        pipeline._client = AsyncMock()
        pipeline._client.chat.completions.create = capture

        await pipeline.run("JD text.", empty_bundle, resume_text="I worked at Acme.")

        assert "I worked at Acme." in captured_prompt[0]

    @pytest.mark.asyncio()
    async def test_resume_section_absent_when_none(
        self,
        empty_bundle: ArtefactBundle,
    ) -> None:
        """The prompt must not contain a resume heading when resume_text is None."""
        msg = MagicMock()
        msg.content = "CV\n---\nComment"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]

        captured_prompt: list[str] = []

        async def capture(  # type: ignore[return]
            model: str, messages: list, **kwargs: object
        ) -> object:
            captured_prompt.append(messages[0]["content"])
            return resp

        pipeline = self._make_pipeline()
        pipeline._client = AsyncMock()
        pipeline._client.chat.completions.create = capture

        await pipeline.run("JD text.", empty_bundle, resume_text=None)

        assert "Candidate resume:" not in captured_prompt[0]

    @pytest.mark.asyncio()
    async def test_returns_frozen_dataclass(
        self,
        empty_bundle: ArtefactBundle,
        mock_groq_response: MagicMock,
    ) -> None:
        """Result must be a frozen BaselineResult dataclass."""
        pipeline = self._make_pipeline()
        pipeline._client = AsyncMock()
        pipeline._client.chat.completions.create = AsyncMock(
            return_value=mock_groq_response
        )

        result = await pipeline.run("JD.", empty_bundle)

        assert isinstance(result, BaselineResult)
        with pytest.raises((AttributeError, TypeError)):
            result.cv_markdown = "mutated"  # type: ignore[misc]
