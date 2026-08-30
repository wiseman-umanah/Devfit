"""
Frozen Pydantic v2 models shared across all DevFit pipeline stages.

Design constraints
------------------
- All fields use explicit Python type annotations (no bare ``Any``).
- Enums are ``str`` enums so they serialise cleanly to JSON.
- ``Verdict.evidence`` must be non-empty unless ``classification`` is
  ``Classification.UNVERIFIABLE``; a model validator enforces this.
- Do **not** add ad-hoc fields to these models.  If a new field is
  genuinely required, extend the model deliberately and record the change
  in ``CHANGELOG.md``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ClaimSource(StrEnum):
    """Identifies where a ``Claim`` originated."""

    JD_REQUIREMENT = "jd_requirement"
    RESUME = "resume"
    BIO = "bio"


class ClaimCategory(StrEnum):
    """Semantic category of a ``Claim``, used to pre-flag unverifiable items."""

    TECHNICAL_SKILL = "technical_skill"
    EXPERIENCE_DURATION = "experience_duration"
    ROLE_SCOPE = "role_scope"
    SOFT_SKILL = "soft_skill"
    OTHER = "other"


class ArtefactType(StrEnum):
    """The kind of GitHub artefact backing a piece of evidence."""

    REPO = "repo"
    COMMIT = "commit"
    README = "readme"
    LANGUAGE_STATS = "language_stats"
    CONTRIBUTION_GRAPH = "contribution_graph"
    ACCOUNT_METADATA = "account_metadata"


class Classification(StrEnum):
    """Three-way verdict classification — the core DevFit output signal."""

    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNVERIFIABLE = "unverifiable"


# ---------------------------------------------------------------------------
# Core models
# ---------------------------------------------------------------------------


class Claim(BaseModel):
    """
    A single atomic, checkable statement extracted from a JD, resume, or bio.

    Parameters
    ----------
    id : str
        Unique identifier for this claim within a run (e.g. ``"c-001"``).
    text : str
        The atomic statement to be verified — should be a single checkable
        assertion, not a compound sentence.
    source : ClaimSource
        Where the claim came from.
    category : ClaimCategory
        Semantic category; ``SOFT_SKILL`` and ``EXPERIENCE_DURATION`` claims
        are pre-flagged because they almost always resolve as
        ``UNVERIFIABLE`` against GitHub data.
    likely_unverifiable : bool
        Set to ``True`` by the JD Analyzer when ``category`` suggests the
        claim cannot be confirmed from public GitHub artefacts.  The Evidence
        Matcher will skip retrieval for these, saving API calls.
    """

    id: str
    text: str
    source: ClaimSource
    category: ClaimCategory
    likely_unverifiable: bool = False


class Artefact(BaseModel):
    """
    A single piece of GitHub evidence that supports or contradicts a claim.

    Parameters
    ----------
    type : ArtefactType
        The kind of GitHub data this artefact represents.
    pointer : str
        URL or identifier that uniquely locates the artefact, e.g.
        ``"github.com/user/repo"`` or ``"github.com/user/repo/commit/abc123"``.
    extracted_fact : str
        A human-readable summary of what this artefact actually shows,
        written in the past tense (e.g. ``"Repo contains 14 Go files"``).
    """

    type: ArtefactType
    pointer: str
    extracted_fact: str


class Verdict(BaseModel):
    """
    The final classification of a single ``Claim`` after all verification layers.

    Parameters
    ----------
    claim_id : str
        Foreign key back to ``Claim.id``.
    classification : Classification
        The authoritative three-way verdict.
    confidence : float
        Confidence score in ``[0.0, 1.0]``.  Deterministic rule verdicts
        should carry ``1.0``; LLM-only verdicts typically land below ``0.9``.
    evidence : list[Artefact]
        The artefacts that support this verdict.  **Must be non-empty** unless
        ``classification`` is ``UNVERIFIABLE``.
    rule_checked : bool
        ``True`` if a deterministic rule in the rule-based layer contributed
        to (or finalised) this verdict.
    llm_confirmed : bool
        ``True`` if the constrained LLM verifier ran and agreed with the
        proposed classification.
    """

    claim_id: str
    classification: Classification
    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence: list[Artefact] = Field(default_factory=list)
    rule_checked: bool = False
    llm_confirmed: bool = False

    @model_validator(mode="after")
    def _require_evidence_when_not_unverifiable(self) -> Verdict:
        """
        Enforce that non-unverifiable verdicts always carry at least one artefact.

        Returns
        -------
        Verdict
            The validated model instance.

        Raises
        ------
        ValueError
            If ``classification`` is not ``UNVERIFIABLE`` and ``evidence`` is
            empty.
        """
        if (
            self.classification != Classification.UNVERIFIABLE
            and not self.evidence
        ):
            raise ValueError(
                f"Verdict for claim '{self.claim_id}' has classification "
                f"'{self.classification}' but no supporting evidence.  "
                "Provide at least one Artefact or set classification to "
                "'unverifiable'."
            )
        return self


class GroundTruthLabel(BaseModel):
    """
    A manually assigned ground-truth classification for a single claim.

    This model is used exclusively by the evaluation scripts in ``eval/``.
    Labels must be fixed **before** any pipeline run to prevent confirmation
    bias corrupting the evaluation results.

    Parameters
    ----------
    claim_id : str
        Foreign key back to ``Claim.id``.
    correct_classification : Classification
        The human-assigned correct classification.
    labeled_by : str
        Identifier of the labeller (typically ``"builder"``).
    notes : str
        Optional free-text rationale for the label, useful for edge cases.
    """

    claim_id: str
    correct_classification: Classification
    labeled_by: str = "builder"
    notes: str = ""
