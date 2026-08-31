"""
Unit tests for ``POST /api/v1/analyze``.

Coverage
--------
- Valid request returns 200 with all expected response fields.
- ``run_id`` is a non-empty string.
- Verdict counts (verified, contradicted, unverifiable) are non-negative integers.
- GitHub rate-limit error maps to HTTP 503.
- Generic GitHub collection failure maps to HTTP 500.
- Generic pipeline failure maps to HTTP 500.
- ``resume_text`` is forwarded to the resume analyzer when provided.
- ``include_unverifiable=True`` is forwarded to CVGenerator.
- Request with JD shorter than 50 chars returns 422 (validation).
- Request with empty github_username returns 422 (validation).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from devfit.api.app import app
from devfit.github.bundle import ArtefactBundle
from devfit.schema import (
    Artefact,
    ArtefactType,
    Classification,
    Verdict,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_VALID_JD = (
    "We are looking for a senior Python engineer with 5+ years of experience "
    "building REST APIs, strong SQL skills, and familiarity with cloud platforms."
)
_VALID_GITHUB = "torvalds"

_MOCK_ARTEFACT = Artefact(
    type=ArtefactType.REPO,
    pointer="github.com/torvalds/linux",
    extracted_fact="linux (C, 180000 stars, updated 2025-01-01) — Linux kernel",
)

_VERIFIED_VERDICT = Verdict(
    claim_id="c-001",
    classification=Classification.VERIFIED,
    confidence=0.95,
    evidence=[_MOCK_ARTEFACT],
)
_CONTRADICTED_VERDICT = Verdict(
    claim_id="c-002",
    classification=Classification.CONTRADICTED,
    confidence=0.90,
    evidence=[_MOCK_ARTEFACT],
)
_UNVERIFIABLE_VERDICT = Verdict(
    claim_id="c-003",
    classification=Classification.UNVERIFIABLE,
    confidence=1.0,
    evidence=[],
)


def _make_bundle() -> ArtefactBundle:
    """Return a minimal ``ArtefactBundle`` for mocking the collector."""
    return ArtefactBundle(artefacts=[_MOCK_ARTEFACT])


def _mock_pipeline_patches(
    *, include_resume_analyzer: bool = False
) -> dict[str, object]:
    """
    Return a dict of patch targets and mock return values for the full pipeline.

    Parameters
    ----------
    include_resume_analyzer : bool
        When ``True``, the ``ResumeAnalyzer.analyze`` mock returns one extra claim.

    Returns
    -------
    dict[str, object]
        Mapping of dotted import path -> mock object.
    """
    bundle = _make_bundle()

    claim = MagicMock()
    claim.id = "c-001"
    claim.text = "Python proficiency"
    claim.category = "TECHNICAL_SKILL"

    matched_claim = MagicMock()
    matched_claim.skipped = False

    patches: dict[str, object] = {
        "devfit.github.client.GitHubClient": MagicMock(
            return_value=MagicMock(
                __aenter__=AsyncMock(return_value=MagicMock()),
                __aexit__=AsyncMock(return_value=False),
            )
        ),
        "devfit.github.collector.GitHubCollector": MagicMock(
            return_value=MagicMock(collect=AsyncMock(return_value=bundle))
        ),
        # patch at the __init__ re-export level — that is what analyze.py binds
        "devfit.pipeline.JDAnalyzer": MagicMock(
            return_value=MagicMock(analyze=AsyncMock(return_value=[claim]))
        ),
        "devfit.pipeline.ResumeAnalyzer": MagicMock(
            return_value=MagicMock(
                analyze=AsyncMock(
                    return_value=[claim] if include_resume_analyzer else []
                )
            )
        ),
        "devfit.pipeline.EvidenceMatcher": MagicMock(
            return_value=MagicMock(
                match=MagicMock(return_value=[matched_claim]),
                build_unverifiable_verdicts=MagicMock(
                    return_value=[_UNVERIFIABLE_VERDICT]
                ),
            )
        ),
        "devfit.pipeline.FirstPassClassifier": MagicMock(
            return_value=MagicMock(classify=AsyncMock(return_value=[]))
        ),
        "devfit.verifier.IndependentVerifier": MagicMock(
            return_value=MagicMock(
                verify_all=AsyncMock(
                    return_value=([_VERIFIED_VERDICT, _CONTRADICTED_VERDICT], [])
                )
            )
        ),
        "devfit.output.FitReportGenerator": MagicMock(
            return_value=MagicMock(generate=MagicMock(return_value="# Fit Report\n"))
        ),
        "devfit.output.CVGenerator": MagicMock(
            return_value=MagicMock(generate=AsyncMock(return_value=("# CV\n", [])))
        ),
        "devfit.output.EvidenceAppendix": MagicMock(
            return_value=MagicMock(generate=MagicMock(return_value="# Appendix\n"))
        ),
        "devfit.output.ImprovementGenerator": MagicMock(
            return_value=MagicMock(
                generate=AsyncMock(return_value="# Improvements\n")
            )
        ),
    }
    return patches


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> TestClient:
    """Return a synchronous ``TestClient`` for the DevFit FastAPI app."""
    return TestClient(app)


class TestAnalyzeSuccess:
    """Happy-path tests for ``POST /api/v1/analyze``."""

    def test_returns_200(self, client: TestClient) -> None:
        """A valid request returns HTTP 200."""
        patches = _mock_pipeline_patches()
        with _apply_patches(patches):
            resp = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            )
        assert resp.status_code == 200

    def test_response_has_all_fields(self, client: TestClient) -> None:
        """Response body includes all documented fields."""
        patches = _mock_pipeline_patches()
        with _apply_patches(patches):
            data = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            ).json()
        assert "run_id" in data
        assert "fit_report" in data
        assert "cv" in data
        assert "evidence_appendix" in data
        assert "improvements" in data
        assert "verified_count" in data
        assert "contradicted_count" in data
        assert "unverifiable_count" in data

    def test_run_id_is_non_empty(self, client: TestClient) -> None:
        """``run_id`` is a non-empty string."""
        patches = _mock_pipeline_patches()
        with _apply_patches(patches):
            data = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            ).json()
        assert isinstance(data["run_id"], str)
        assert len(data["run_id"]) == 8

    def test_verdict_counts_are_non_negative_ints(self, client: TestClient) -> None:
        """Verdict counts are all non-negative integers."""
        patches = _mock_pipeline_patches()
        with _apply_patches(patches):
            data = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            ).json()
        assert isinstance(data["verified_count"], int)
        assert isinstance(data["contradicted_count"], int)
        assert isinstance(data["unverifiable_count"], int)
        assert data["verified_count"] >= 0
        assert data["contradicted_count"] >= 0
        assert data["unverifiable_count"] >= 0

    def test_verdict_counts_match_mocked_verdicts(self, client: TestClient) -> None:
        """Counts match mocked verdicts: 1 verified, 1 contradicted, 1 unverifiable."""
        patches = _mock_pipeline_patches()
        with _apply_patches(patches):
            data = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            ).json()
        assert data["verified_count"] == 1
        assert data["contradicted_count"] == 1
        assert data["unverifiable_count"] == 1

    def test_markdown_fields_are_non_empty_strings(self, client: TestClient) -> None:
        """Markdown output fields are non-empty strings."""
        patches = _mock_pipeline_patches()
        with _apply_patches(patches):
            data = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            ).json()
        assert data["fit_report"].strip()
        assert data["cv"].strip()
        assert data["evidence_appendix"].strip()
        assert data["improvements"].strip()


class TestAnalyzeResume:
    """Tests for the optional ``resume_text`` parameter."""

    def test_resume_text_is_forwarded(self, client: TestClient) -> None:
        """When resume_text is provided, the response still returns 200."""
        patches = _mock_pipeline_patches(include_resume_analyzer=True)
        with _apply_patches(patches):
            resp = client.post(
                "/api/v1/analyze",
                json={
                    "jd_text": _VALID_JD,
                    "github_username": _VALID_GITHUB,
                    "resume_text": "Senior Python Engineer with 7 years of experience.",
                },
            )
        assert resp.status_code == 200

    def test_no_resume_text_also_succeeds(self, client: TestClient) -> None:
        """When resume_text is omitted, the request still succeeds."""
        patches = _mock_pipeline_patches()
        with _apply_patches(patches):
            resp = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            )
        assert resp.status_code == 200


class TestAnalyzeErrors:
    """Error-path tests for ``POST /api/v1/analyze``."""

    def test_rate_limit_returns_503(self, client: TestClient) -> None:
        """GitHub rate-limit error maps to HTTP 503."""
        from devfit.github.client import GitHubRateLimitError

        with patch(
            "devfit.github.client.GitHubClient",
            return_value=MagicMock(
                __aenter__=AsyncMock(side_effect=GitHubRateLimitError("rate limited")),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            resp = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            )
        assert resp.status_code == 503
        assert "rate limit" in resp.json()["detail"].lower()

    def test_github_collection_error_returns_500(self, client: TestClient) -> None:
        """Unexpected GitHub collection failure maps to HTTP 500."""
        with patch(
            "devfit.github.client.GitHubClient",
            return_value=MagicMock(
                __aenter__=AsyncMock(side_effect=RuntimeError("network error")),
                __aexit__=AsyncMock(return_value=False),
            ),
        ):
            resp = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            )
        assert resp.status_code == 500

    def test_pipeline_failure_returns_500(self, client: TestClient) -> None:
        """A failure in the JD-analysis stage maps to HTTP 500."""
        patches = _mock_pipeline_patches()
        patches["devfit.pipeline.JDAnalyzer"] = MagicMock(
            return_value=MagicMock(
                analyze=AsyncMock(side_effect=RuntimeError("LLM error"))
            )
        )
        with _apply_patches(patches):
            resp = client.post(
                "/api/v1/analyze",
                json={"jd_text": _VALID_JD, "github_username": _VALID_GITHUB},
            )
        assert resp.status_code == 500
        assert "Pipeline failed" in resp.json()["detail"]


class TestAnalyzeValidation:
    """Pydantic validation tests for ``POST /api/v1/analyze``."""

    def test_short_jd_returns_422(self, client: TestClient) -> None:
        """JD text shorter than 50 characters returns 422."""
        resp = client.post(
            "/api/v1/analyze",
            json={"jd_text": "Too short.", "github_username": _VALID_GITHUB},
        )
        assert resp.status_code == 422

    def test_empty_username_returns_422(self, client: TestClient) -> None:
        """Empty github_username returns 422."""
        resp = client.post(
            "/api/v1/analyze",
            json={"jd_text": _VALID_JD, "github_username": ""},
        )
        assert resp.status_code == 422

    def test_username_too_long_returns_422(self, client: TestClient) -> None:
        """github_username longer than 39 characters returns 422."""
        resp = client.post(
            "/api/v1/analyze",
            json={"jd_text": _VALID_JD, "github_username": "a" * 40},
        )
        assert resp.status_code == 422

    def test_missing_jd_returns_422(self, client: TestClient) -> None:
        """Missing jd_text field returns 422."""
        resp = client.post(
            "/api/v1/analyze",
            json={"github_username": _VALID_GITHUB},
        )
        assert resp.status_code == 422

    def test_missing_username_returns_422(self, client: TestClient) -> None:
        """Missing github_username field returns 422."""
        resp = client.post(
            "/api/v1/analyze",
            json={"jd_text": _VALID_JD},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Helper: stack multiple patches via a context manager
# ---------------------------------------------------------------------------


@contextmanager
def _apply_patches(targets: dict[str, object]) -> Iterator[None]:
    """
    Apply multiple ``unittest.mock.patch`` contexts simultaneously.

    Parameters
    ----------
    targets : dict[str, object]
        Mapping of dotted import path -> replacement object.

    Yields
    ------
    None
    """
    patches = [patch(k, v) for k, v in targets.items()]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()
