"""
Unit tests for the core Pydantic schema models.

Tests cover:
- Valid model construction for all four models.
- Enum value validation (invalid strings must be rejected).
- The ``Verdict`` model validator: evidence must be non-empty for
  ``verified`` and ``contradicted`` classifications.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from devfit.schema import (
    Artefact,
    ArtefactType,
    Claim,
    ClaimCategory,
    ClaimSource,
    Classification,
    GroundTruthLabel,
    Verdict,
)


class TestClaim:
    """Tests for the ``Claim`` model."""

    def test_valid_construction(self) -> None:
        """A fully-specified Claim constructs without error."""
        claim = Claim(
            id="c-001",
            text="Expert in Python",
            source=ClaimSource.JD_REQUIREMENT,
            category=ClaimCategory.TECHNICAL_SKILL,
        )
        assert claim.id == "c-001"
        assert claim.likely_unverifiable is False

    def test_invalid_source_raises(self) -> None:
        """An invalid source enum value must raise ValidationError."""
        with pytest.raises(ValidationError):
            Claim(
                id="c-002",
                text="...",
                source="bad_source",  # type: ignore[arg-type]
                category=ClaimCategory.OTHER,
            )

    def test_invalid_category_raises(self) -> None:
        """An invalid category enum value must raise ValidationError."""
        with pytest.raises(ValidationError):
            Claim(
                id="c-003",
                text="...",
                source=ClaimSource.RESUME,
                category="made_up_category",  # type: ignore[arg-type]
            )


class TestArtefact:
    """Tests for the ``Artefact`` model."""

    def test_valid_construction(self) -> None:
        """A valid Artefact constructs without error."""
        art = Artefact(
            type=ArtefactType.REPO,
            pointer="github.com/user/repo",
            extracted_fact="Python repo with 10 stars",
        )
        assert art.pointer == "github.com/user/repo"

    def test_invalid_type_raises(self) -> None:
        """An invalid artefact type must raise ValidationError."""
        with pytest.raises(ValidationError):
            Artefact(
                type="blog_post",  # type: ignore[arg-type]
                pointer="...",
                extracted_fact="...",
            )


class TestVerdict:
    """Tests for the ``Verdict`` model, including the evidence validator."""

    def test_verified_with_evidence(self, sample_artefact: Artefact) -> None:
        """A verified verdict with evidence constructs without error."""
        verdict = Verdict(
            claim_id="c-001",
            classification=Classification.VERIFIED,
            confidence=0.9,
            evidence=[sample_artefact],
            rule_checked=True,
        )
        assert verdict.classification == Classification.VERIFIED

    def test_unverifiable_without_evidence(self) -> None:
        """An unverifiable verdict may have an empty evidence list."""
        verdict = Verdict(
            claim_id="c-001",
            classification=Classification.UNVERIFIABLE,
            confidence=1.0,
            evidence=[],
        )
        assert verdict.evidence == []

    def test_verified_without_evidence_raises(self) -> None:
        """A verified verdict with empty evidence must raise ValidationError."""
        with pytest.raises(ValidationError, match="no supporting evidence"):
            Verdict(
                claim_id="c-001",
                classification=Classification.VERIFIED,
                confidence=0.9,
                evidence=[],
            )

    def test_contradicted_without_evidence_raises(self) -> None:
        """A contradicted verdict with empty evidence must raise ValidationError."""
        with pytest.raises(ValidationError, match="no supporting evidence"):
            Verdict(
                claim_id="c-001",
                classification=Classification.CONTRADICTED,
                confidence=0.85,
                evidence=[],
            )

    def test_confidence_out_of_range_raises(self, sample_artefact: Artefact) -> None:
        """Confidence values outside [0, 1] must raise ValidationError."""
        with pytest.raises(ValidationError):
            Verdict(
                claim_id="c-001",
                classification=Classification.VERIFIED,
                confidence=1.5,
                evidence=[sample_artefact],
            )


class TestGroundTruthLabel:
    """Tests for the ``GroundTruthLabel`` model."""

    def test_valid_construction(self) -> None:
        """A valid ground-truth label constructs without error."""
        label = GroundTruthLabel(
            claim_id="c-001",
            correct_classification=Classification.VERIFIED,
        )
        assert label.labeled_by == "builder"

    def test_invalid_classification_raises(self) -> None:
        """An invalid classification string must raise ValidationError."""
        with pytest.raises(ValidationError):
            GroundTruthLabel(
                claim_id="c-001",
                correct_classification="maybe",  # type: ignore[arg-type]
            )
