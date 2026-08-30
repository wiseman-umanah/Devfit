"""
Unit tests for the deterministic rule-based verifier layer.

These tests use only synthetic ``ArtefactBundle`` fixtures — zero network
calls, zero LLM calls.  Every test must be fully deterministic.

Coverage requirements (per TRD §2 and steps.txt §1.5):
- Each rule must have at least one CONFIRMED, one CONTRADICTED, and one
  CANNOT-RESOLVE (returns ``None``) test case.
"""

from __future__ import annotations

import pytest

from devfit.github.bundle import ArtefactBundle
from devfit.schema import (
    Artefact,
    ArtefactType,
    Claim,
    ClaimCategory,
    ClaimSource,
    Classification,
)
from devfit.verifier.rules import (
    RuleVerifier,
    check_date_arithmetic,
    check_language_presence,
    check_zero_activity,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def young_account_bundle() -> ArtefactBundle:
    """
    Bundle whose account was created only 6 months ago.

    Returns
    -------
    ArtefactBundle
        Suitable for testing date-arithmetic contradictions.
    """
    from datetime import UTC, datetime, timedelta

    created = (datetime.now(UTC) - timedelta(days=180)).strftime("%Y-%m-%d")
    return ArtefactBundle(
        artefacts=[
            Artefact(
                type=ArtefactType.ACCOUNT_METADATA,
                pointer="github.com/newuser",
                extracted_fact=f"Account created {created} (0y 6m ago), 3 public repos",
            )
        ]
    )


@pytest.fixture()
def no_go_bundle() -> ArtefactBundle:
    """
    Bundle with language stats that contain zero Go bytes.

    Returns
    -------
    ArtefactBundle
        Suitable for testing language-presence contradictions.
    """
    return ArtefactBundle(
        artefacts=[
            Artefact(
                type=ArtefactType.LANGUAGE_STATS,
                pointer="github.com/testuser",
                extracted_fact=(
                    "Language breakdown across top repos: python (200,000 bytes), "
                    "javascript (50,000 bytes)"
                ),
            )
        ]
    )


@pytest.fixture()
def duration_claim_7y() -> Claim:
    """
    Return a ``EXPERIENCE_DURATION`` claim asserting 7 years of experience.

    Returns
    -------
    Claim
        ``EXPERIENCE_DURATION`` claim with 7-year assertion.
    """
    return Claim(
        id="c-dur-7",
        text="7+ years of professional software engineering experience.",
        source=ClaimSource.JD_REQUIREMENT,
        category=ClaimCategory.EXPERIENCE_DURATION,
        likely_unverifiable=True,
    )


@pytest.fixture()
def go_expert_claim() -> Claim:
    """
    Return an expert-level Go ``TECHNICAL_SKILL`` claim.

    Returns
    -------
    Claim
        ``TECHNICAL_SKILL`` claim for Go expertise.
    """
    return Claim(
        id="c-go-1",
        text="Expert Go developer with extensive backend experience.",
        source=ClaimSource.JD_REQUIREMENT,
        category=ClaimCategory.TECHNICAL_SKILL,
        likely_unverifiable=False,
    )


@pytest.fixture()
def soft_skill_claim() -> Claim:
    """
    Return a ``SOFT_SKILL`` leadership claim that no rule should resolve.

    Returns
    -------
    Claim
        ``SOFT_SKILL`` claim for leadership.
    """
    return Claim(
        id="c-soft-1",
        text="Strong team leadership and communication skills.",
        source=ClaimSource.JD_REQUIREMENT,
        category=ClaimCategory.SOFT_SKILL,
        likely_unverifiable=True,
    )


# ---------------------------------------------------------------------------
# check_date_arithmetic
# ---------------------------------------------------------------------------


class TestCheckDateArithmetic:
    """Tests for the ``check_date_arithmetic`` rule."""

    def test_contradicts_when_account_younger_than_claimed(
        self,
        duration_claim_7y: Claim,
        young_account_bundle: ArtefactBundle,
    ) -> None:
        """Account 6 months old must CONTRADICT a 7-year claim."""
        verdict = check_date_arithmetic(duration_claim_7y, young_account_bundle)
        assert verdict is not None
        assert verdict.classification == Classification.CONTRADICTED
        assert verdict.rule_checked is True
        assert verdict.evidence  # must have supporting artefact

    def test_cannot_resolve_non_duration_claim(
        self,
        go_expert_claim: Claim,
        young_account_bundle: ArtefactBundle,
    ) -> None:
        """A TECHNICAL_SKILL claim must return None (rule does not apply)."""
        result = check_date_arithmetic(go_expert_claim, young_account_bundle)
        assert result is None

    def test_cannot_resolve_when_no_metadata(
        self,
        duration_claim_7y: Claim,
        bundle_empty: ArtefactBundle,
    ) -> None:
        """Without account metadata the rule cannot resolve — return None."""
        result = check_date_arithmetic(duration_claim_7y, bundle_empty)
        assert result is None


# ---------------------------------------------------------------------------
# check_language_presence
# ---------------------------------------------------------------------------


class TestCheckLanguagePresence:
    """Tests for the ``check_language_presence`` rule."""

    def test_contradicts_when_language_absent(
        self,
        go_expert_claim: Claim,
        no_go_bundle: ArtefactBundle,
    ) -> None:
        """Go absent from language stats must CONTRADICT the expert-Go claim."""
        verdict = check_language_presence(go_expert_claim, no_go_bundle)
        assert verdict is not None
        assert verdict.classification == Classification.CONTRADICTED
        assert verdict.rule_checked is True

    def test_verifies_when_language_present(
        self,
        bundle_with_python: ArtefactBundle,
    ) -> None:
        """Python present with substantial bytes must VERIFY a Python claim."""
        python_claim = Claim(
            id="c-py-1",
            text="Python backend developer.",
            source=ClaimSource.JD_REQUIREMENT,
            category=ClaimCategory.TECHNICAL_SKILL,
        )
        verdict = check_language_presence(python_claim, bundle_with_python)
        assert verdict is not None
        assert verdict.classification == Classification.VERIFIED

    def test_cannot_resolve_soft_skill(
        self,
        soft_skill_claim: Claim,
        bundle_with_python: ArtefactBundle,
    ) -> None:
        """A SOFT_SKILL claim must return None."""
        result = check_language_presence(soft_skill_claim, bundle_with_python)
        assert result is None

    def test_cannot_resolve_when_no_language_stats(
        self,
        go_expert_claim: Claim,
        bundle_empty: ArtefactBundle,
    ) -> None:
        """Without language stats the rule cannot resolve — return None."""
        result = check_language_presence(go_expert_claim, bundle_empty)
        assert result is None


# ---------------------------------------------------------------------------
# check_zero_activity
# ---------------------------------------------------------------------------


class TestCheckZeroActivity:
    """Tests for the ``check_zero_activity`` rule."""

    def test_contradicts_expert_claim_with_no_repos(
        self,
        go_expert_claim: Claim,
    ) -> None:
        """Expert claim with account metadata but zero repos must be CONTRADICTED."""
        bundle = ArtefactBundle(
            artefacts=[
                Artefact(
                    type=ArtefactType.ACCOUNT_METADATA,
                    pointer="github.com/newuser",
                    extracted_fact=(
                        "Account created 2024-01-01 (1y 0m ago), 0 public repos"
                    ),
                )
            ]
        )
        verdict = check_zero_activity(go_expert_claim, bundle)
        assert verdict is not None
        assert verdict.classification == Classification.CONTRADICTED
        assert verdict.rule_checked is True

    def test_returns_none_when_bundle_has_no_evidence_at_all(
        self,
        go_expert_claim: Claim,
        bundle_empty: ArtefactBundle,
    ) -> None:
        """With no artefacts at all, rule cannot produce a verdict — return None."""
        result = check_zero_activity(go_expert_claim, bundle_empty)
        assert result is None

    def test_cannot_resolve_weak_claim(
        self,
        bundle_with_python: ArtefactBundle,
    ) -> None:
        """A claim without strong language (expert/extensive/etc.) returns None."""
        weak_claim = Claim(
            id="c-weak-1",
            text="Some familiarity with Go.",
            source=ClaimSource.JD_REQUIREMENT,
            category=ClaimCategory.TECHNICAL_SKILL,
        )
        result = check_zero_activity(weak_claim, bundle_with_python)
        assert result is None

    def test_cannot_resolve_soft_skill(
        self,
        soft_skill_claim: Claim,
        bundle_empty: ArtefactBundle,
    ) -> None:
        """SOFT_SKILL claims must return None regardless of bundle contents."""
        result = check_zero_activity(soft_skill_claim, bundle_empty)
        assert result is None


# ---------------------------------------------------------------------------
# RuleVerifier orchestration
# ---------------------------------------------------------------------------


class TestRuleVerifier:
    """Tests for the ``RuleVerifier`` orchestrator."""

    def test_returns_first_decisive_verdict(
        self,
        duration_claim_7y: Claim,
        young_account_bundle: ArtefactBundle,
    ) -> None:
        """Should return a verdict when any rule resolves the claim."""
        verifier = RuleVerifier()
        result = verifier.run(duration_claim_7y, young_account_bundle)
        assert result is not None
        assert result.classification == Classification.CONTRADICTED

    def test_returns_none_when_no_rule_resolves(
        self,
        soft_skill_claim: Claim,
        bundle_with_python: ArtefactBundle,
    ) -> None:
        """Must return None when no rule can resolve the claim."""
        verifier = RuleVerifier()
        result = verifier.run(soft_skill_claim, bundle_with_python)
        assert result is None
