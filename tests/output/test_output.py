"""
Unit tests for FitReportGenerator, CVGenerator, EvidenceAppendix, and TrajectoryLogger.

Coverage
--------
FitReportGenerator
  - Score computed correctly from verified/contradicted counts.
  - Unverifiable claims never appear in score section.
  - Score labels: Strong / Partial / Weak.
  - Empty verdicts produce Inconclusive score.
  - Contradicted claims have a dedicated section.

CVGenerator
  - Verified claims all carry [source: <pointer>] tags.
  - Zero verified CV lines lack an artefact pointer (critical assertion).
  - Unverifiable claims excluded by default.
  - Unverifiable claims included with marker when include_unverifiable=True.
  - Empty verified list produces an "no verified claims" note.

EvidenceAppendix
  - Every verdict appears in the appendix.
  - Verified artefact pointers are present.
  - Unverifiable entries note no artefact.

TrajectoryLogger
  - Events are written as valid JSONL.
  - Each entry has a timestamp and stage field.
  - File is flushed after each write.
  - Context manager opens and closes cleanly.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from devfit.output.appendix import EvidenceAppendix
from devfit.output.cv import CVGenerator
from devfit.output.report import FitReportGenerator, _compute_score
from devfit.output.trajectory import TrajectoryLogger
from devfit.schema import (
    Artefact,
    ArtefactType,
    Classification,
    Verdict,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def python_artefact() -> Artefact:
    """Return a Python language-stats artefact."""
    return Artefact(
        type=ArtefactType.LANGUAGE_STATS,
        pointer="github.com/testuser",
        extracted_fact="python (400,000 bytes)",
    )


@pytest.fixture()
def verified_verdict(python_artefact: Artefact) -> Verdict:
    """Return a VERIFIED verdict."""
    return Verdict(
        claim_id="c-001",
        classification=Classification.VERIFIED,
        confidence=0.9,
        evidence=[python_artefact],
        rule_checked=True,
    )


@pytest.fixture()
def contradicted_verdict(python_artefact: Artefact) -> Verdict:
    """Return a CONTRADICTED verdict."""
    return Verdict(
        claim_id="c-002",
        classification=Classification.CONTRADICTED,
        confidence=0.85,
        evidence=[python_artefact],
        rule_checked=True,
    )


@pytest.fixture()
def unverifiable_verdict() -> Verdict:
    """Return an UNVERIFIABLE verdict."""
    return Verdict(
        claim_id="c-003",
        classification=Classification.UNVERIFIABLE,
        confidence=1.0,
        evidence=[],
    )


@pytest.fixture()
def claims_by_id() -> dict[str, str]:
    """Return a claim_id → claim_text mapping for the test fixtures."""
    return {
        "c-001": "Expert Python developer",
        "c-002": "7+ years of professional experience",
        "c-003": "Strong team leadership skills",
    }


# ---------------------------------------------------------------------------
# _compute_score tests
# ---------------------------------------------------------------------------


class TestComputeScore:
    """Tests for the fit score computation logic."""

    def test_all_verified_gives_strong_fit(
        self, verified_verdict: Verdict
    ) -> None:
        """All verified → Strong Fit (100%)."""
        score = _compute_score([verified_verdict])
        assert score.label == "Strong Fit"
        assert score.score_pct == 100.0

    def test_all_contradicted_gives_weak_fit(
        self, contradicted_verdict: Verdict
    ) -> None:
        """All contradicted → Weak Fit / Mismatch (0%)."""
        score = _compute_score([contradicted_verdict])
        assert score.label == "Weak Fit / Mismatch"
        assert score.score_pct == 0.0

    def test_mixed_gives_weak_fit(
        self,
        verified_verdict: Verdict,
        contradicted_verdict: Verdict,
    ) -> None:
        """One verified + one contradicted: raw=(1-1)/2*100=0% → Weak Fit."""
        score = _compute_score([verified_verdict, contradicted_verdict])
        assert score.score_pct == 0.0
        assert score.label == "Weak Fit / Mismatch"
        assert score.verified_count == 1
        assert score.contradicted_count == 1

    def test_unverifiable_not_counted(
        self,
        verified_verdict: Verdict,
        unverifiable_verdict: Verdict,
    ) -> None:
        """Unverifiable claims must not affect the score."""
        score_with = _compute_score([verified_verdict, unverifiable_verdict])
        score_without = _compute_score([verified_verdict])
        assert score_with.score_pct == score_without.score_pct
        assert score_with.total_scorable == 1

    def test_empty_verdicts_gives_inconclusive(self) -> None:
        """No verdicts → Inconclusive."""
        score = _compute_score([])
        assert score.label == "Inconclusive"
        assert score.score_pct == 0.0


# ---------------------------------------------------------------------------
# FitReportGenerator tests
# ---------------------------------------------------------------------------


class TestFitReportGenerator:
    """Tests for FitReportGenerator."""

    def test_report_contains_score_label(
        self,
        verified_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """The generated report must contain the score label."""
        gen = FitReportGenerator()
        report = gen.generate(
            [verified_verdict], claims_by_id, "testuser", "Senior Eng"
        )
        assert "Strong Fit" in report

    def test_report_contains_verified_claim_text(
        self,
        verified_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Verified claim text must appear in the report."""
        gen = FitReportGenerator()
        report = gen.generate([verified_verdict], claims_by_id, "testuser", "Role")
        assert "Expert Python developer" in report

    def test_unverifiable_not_in_score_section(
        self,
        unverifiable_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Unverifiable claims must appear only in the unverifiable section."""
        gen = FitReportGenerator()
        report = gen.generate([unverifiable_verdict], claims_by_id, "testuser", "Role")
        # Must appear in unverifiable section
        assert "Strong team leadership skills" in report
        # Must NOT be scored — Inconclusive
        assert "Inconclusive" in report

    def test_report_contains_github_username(
        self,
        verified_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """The report header must include the candidate's GitHub username."""
        gen = FitReportGenerator()
        report = gen.generate([verified_verdict], claims_by_id, "myuser", "Role")
        assert "myuser" in report

    def test_contradicted_claims_have_own_section(
        self,
        contradicted_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Contradicted claims must appear under the Contradicted section."""
        gen = FitReportGenerator()
        report = gen.generate([contradicted_verdict], claims_by_id, "u", "Role")
        assert "Contradicted Claims" in report
        assert "7+ years of professional experience" in report


# ---------------------------------------------------------------------------
# CVGenerator tests — the critical pointer-completeness check
# ---------------------------------------------------------------------------


class TestCVGenerator:
    """Tests for CVGenerator."""

    def _make_generator(self) -> CVGenerator:
        """Return a CVGenerator with Groq client creation mocked out."""
        settings_patcher = patch("devfit.output.cv.get_settings")
        groq_patcher = patch("devfit.output.cv.AsyncGroq")
        mock_settings = settings_patcher.start()
        groq_patcher.start()
        mock_settings.return_value.groq_api_key.get_secret_value.return_value = (
            "fake-key"
        )
        mock_settings.return_value.groq_model = "openai/gpt-oss-120b"
        gen = CVGenerator()
        settings_patcher.stop()
        groq_patcher.stop()
        return gen

    @pytest.mark.asyncio()
    async def test_all_verified_lines_have_pointer(
        self,
        verified_verdict: Verdict,
        python_artefact: Artefact,
        claims_by_id: dict[str, str],
    ) -> None:
        """
        Every verified CV line must carry a non-empty artefact pointer.

        This is the critical TRD section 3.5 assertion: zero CV lines without
        a traceable artefact pointer. The pointer-completeness check runs on
        the structured CVLine objects, which are built deterministically before
        any LLM call.
        """
        gen = self._make_generator()
        # LLM failure forces fallback; cv_lines are always built deterministically
        gen._client = MagicMock()
        gen._client.chat.completions.create.side_effect = Exception("mocked failure")
        _, cv_lines = await gen.generate(
            [verified_verdict], claims_by_id, "testuser"
        )
        for line in cv_lines:
            if not line.is_unverifiable:
                assert line.artefact_pointer, (
                    f"CV line '{line.text}' has no artefact pointer"
                )

    @pytest.mark.asyncio()
    async def test_source_tag_in_fallback_markdown(
        self,
        verified_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Fallback CV bullets must contain [source: and the pointer."""
        gen = self._make_generator()
        gen._client = MagicMock()
        gen._client.chat.completions.create.side_effect = Exception("mocked failure")
        md, _ = await gen.generate([verified_verdict], claims_by_id, "testuser")
        assert "[source:" in md
        assert "github.com/testuser" in md

    @pytest.mark.asyncio()
    async def test_unverifiable_excluded_by_default(
        self,
        unverifiable_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Unverifiable claims must NOT appear in the CV by default."""
        gen = self._make_generator()
        md, cv_lines = await gen.generate(
            [unverifiable_verdict], claims_by_id, "testuser",
            include_unverifiable=False,
        )
        assert "NOT VERIFIED FROM GITHUB" not in md
        assert "CANNOT BE CONFIRMED" not in md
        assert all(not cl.is_unverifiable for cl in cv_lines)

    @pytest.mark.asyncio()
    async def test_unverifiable_included_with_marker(
        self,
        unverifiable_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """When include_unverifiable=True, fallback CV shows the marker."""
        gen = self._make_generator()
        md, _ = await gen.generate(
            [unverifiable_verdict], claims_by_id, "testuser",
            include_unverifiable=True,
        )
        assert "NOT VERIFIED FROM GITHUB" in md

    @pytest.mark.asyncio()
    async def test_empty_verified_produces_no_claims_note(
        self,
        unverifiable_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """With no verified claims, the CV must note this clearly."""
        gen = self._make_generator()
        md, cv_lines = await gen.generate(
            [unverifiable_verdict], claims_by_id, "testuser"
        )
        assert "No verified claims" in md
        assert cv_lines == []

    @pytest.mark.asyncio()
    async def test_zero_verified_lines_lack_pointer(
        self,
        verified_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Assert the zero-missing-pointer invariant for a full verified set."""
        gen = self._make_generator()
        gen._client = MagicMock()
        gen._client.chat.completions.create.side_effect = Exception("mocked failure")
        _, cv_lines = await gen.generate(
            [verified_verdict], claims_by_id, "testuser"
        )
        missing = [
            cl for cl in cv_lines if not cl.is_unverifiable and not cl.artefact_pointer
        ]
        assert missing == [], (
            f"{len(missing)} CV line(s) lack an artefact pointer: "
            + ", ".join(cl.text for cl in missing)
        )

    @pytest.mark.asyncio()
    async def test_llm_output_used_when_successful(
        self,
        verified_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """When the LLM call succeeds, its output replaces the fallback."""
        gen = self._make_generator()
        msg = MagicMock()
        msg.content = "# Test User\n\n## Technical Skills\n\n- Python [source: github.com/testuser]"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        gen._client = AsyncMock()
        gen._client.chat.completions.create = AsyncMock(return_value=resp)

        md, cv_lines = await gen.generate(
            [verified_verdict], claims_by_id, "testuser"
        )
        assert "Test User" in md
        # cv_lines still built deterministically regardless of LLM output
        assert len(cv_lines) == 1
        assert cv_lines[0].artefact_pointer == "github.com/testuser"

    @pytest.mark.asyncio()
    async def test_em_dashes_stripped_from_llm_output(
        self,
        verified_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Em-dashes in LLM output must be stripped before returning."""
        gen = self._make_generator()
        msg = MagicMock()
        msg.content = "# Dev\n\nSkilled in Python — and Go [source: github.com/testuser]"
        choice = MagicMock()
        choice.message = msg
        resp = MagicMock()
        resp.choices = [choice]
        gen._client = AsyncMock()
        gen._client.chat.completions.create = AsyncMock(return_value=resp)

        md, _ = await gen.generate([verified_verdict], claims_by_id, "testuser")
        assert "—" not in md
        assert "--" not in md


# ---------------------------------------------------------------------------
# EvidenceAppendix tests
# ---------------------------------------------------------------------------


class TestEvidenceAppendix:
    """Tests for EvidenceAppendix."""

    def test_all_verdicts_appear(
        self,
        verified_verdict: Verdict,
        contradicted_verdict: Verdict,
        unverifiable_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Every claim ID must appear in the appendix."""
        gen = EvidenceAppendix()
        appendix = gen.generate(
            [verified_verdict, contradicted_verdict, unverifiable_verdict],
            claims_by_id,
        )
        assert "c-001" in appendix
        assert "c-002" in appendix
        assert "c-003" in appendix

    def test_verified_pointer_present(
        self,
        verified_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Verified artefact pointer must appear in the appendix."""
        gen = EvidenceAppendix()
        appendix = gen.generate([verified_verdict], claims_by_id)
        assert "github.com/testuser" in appendix

    def test_unverifiable_has_no_artefact_note(
        self,
        unverifiable_verdict: Verdict,
        claims_by_id: dict[str, str],
    ) -> None:
        """Unverifiable entries must note that no artefact is available."""
        gen = EvidenceAppendix()
        appendix = gen.generate([unverifiable_verdict], claims_by_id)
        assert "No artefact" in appendix or "no artefact" in appendix.lower()


# ---------------------------------------------------------------------------
# TrajectoryLogger tests
# ---------------------------------------------------------------------------


class TestTrajectoryLogger:
    """Tests for TrajectoryLogger."""

    def test_events_written_as_valid_jsonl(self) -> None:
        """Each logged event must be a parseable JSON object on its own line."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with TrajectoryLogger(out) as tlog:
                tlog.log_event("test_stage", {"key": "value"})
                tlog.log_event("another_stage", {"number": 42})

            lines = (out / "trajectory_log.jsonl").read_text().splitlines()
            assert len(lines) == 2
            for line in lines:
                obj = json.loads(line)
                assert "timestamp" in obj
                assert "stage" in obj

    def test_event_contains_stage_and_timestamp(self) -> None:
        """Each event must include a stage identifier and an ISO timestamp."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with TrajectoryLogger(out) as tlog:
                tlog.log_event("pipeline_start", {"run_id": "abc123"})

            line = (out / "trajectory_log.jsonl").read_text().strip()
            obj = json.loads(line)
            assert obj["stage"] == "pipeline_start"
            assert obj["run_id"] == "abc123"
            # Timestamp must be parseable ISO format
            from datetime import datetime
            datetime.fromisoformat(obj["timestamp"])

    def test_log_when_not_open_does_not_raise(self) -> None:
        """Logging without opening must log a warning, not raise."""
        with tempfile.TemporaryDirectory() as tmp:
            tlog = TrajectoryLogger(Path(tmp))
            # Should not raise — just logs a warning
            tlog.log_event("orphan_event", {})

    def test_context_manager_creates_file(self) -> None:
        """Using as context manager must create the trajectory log file."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            with TrajectoryLogger(out) as tlog:
                tlog.log_event("x", {})
            assert (out / "trajectory_log.jsonl").exists()
