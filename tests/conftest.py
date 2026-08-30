"""
Shared pytest fixtures for the DevFit test suite.

Fixtures
--------
sample_claim_technical
    A ``Claim`` with ``category=TECHNICAL_SKILL``.
sample_claim_duration
    A ``Claim`` with ``category=EXPERIENCE_DURATION``.
sample_artefact
    A minimal ``Artefact`` for test use.
bundle_with_python
    An ``ArtefactBundle`` containing Python language stats and account
    metadata indicating an account created 3 years ago.
bundle_empty
    An ``ArtefactBundle`` with no artefacts.
"""

from __future__ import annotations

import pytest

from devfit.github.bundle import ArtefactBundle
from devfit.schema import Artefact, ArtefactType, Claim, ClaimCategory, ClaimSource


@pytest.fixture()
def sample_claim_technical() -> Claim:
    """
    Return a ``TECHNICAL_SKILL`` claim asserting Python expertise.

    Returns
    -------
    Claim
        A claim suitable for testing the language-presence rule.
    """
    return Claim(
        id="c-001",
        text="Expert Python developer with 5+ years building production services.",
        source=ClaimSource.JD_REQUIREMENT,
        category=ClaimCategory.TECHNICAL_SKILL,
        likely_unverifiable=False,
    )


@pytest.fixture()
def sample_claim_duration() -> Claim:
    """
    Return an ``EXPERIENCE_DURATION`` claim asserting 7 years of experience.

    Returns
    -------
    Claim
        A claim suitable for testing the date-arithmetic rule.
    """
    return Claim(
        id="c-002",
        text="7+ years professional software engineering experience.",
        source=ClaimSource.JD_REQUIREMENT,
        category=ClaimCategory.EXPERIENCE_DURATION,
        likely_unverifiable=True,
    )


@pytest.fixture()
def sample_artefact() -> Artefact:
    """
    Return a minimal repo ``Artefact`` for generic test use.

    Returns
    -------
    Artefact
        A repo artefact pointing at a fictional repository.
    """
    return Artefact(
        type=ArtefactType.REPO,
        pointer="github.com/testuser/my-python-app",
        extracted_fact=(
            "my-python-app (Python, 42 stars, updated 2024-11-01) — async web service"
        ),
    )


@pytest.fixture()
def bundle_with_python() -> ArtefactBundle:
    """
    Return an ``ArtefactBundle`` with Python language stats and a 3-year-old account.

    Returns
    -------
    ArtefactBundle
        Bundle suitable for verifying Python skill and medium account age.
    """
    return ArtefactBundle(
        artefacts=[
            Artefact(
                type=ArtefactType.ACCOUNT_METADATA,
                pointer="github.com/testuser",
                extracted_fact=(
                    "Account created 2022-01-15 (3y 2m ago), 18 public repos"
                ),
            ),
            Artefact(
                type=ArtefactType.LANGUAGE_STATS,
                pointer="github.com/testuser",
                extracted_fact=(
                    "Language breakdown across top repos: "
                    "python (485,210 bytes), javascript (32,000 bytes)"
                ),
            ),
            Artefact(
                type=ArtefactType.REPO,
                pointer="github.com/testuser/api-service",
                extracted_fact=(
                    "api-service (Python, 12 stars, updated 2024-10-01)"
                    " — FastAPI service"
                ),
            ),
        ]
    )


@pytest.fixture()
def bundle_empty() -> ArtefactBundle:
    """
    Return an ``ArtefactBundle`` containing no artefacts.

    Returns
    -------
    ArtefactBundle
        Empty bundle for testing edge-case handling.
    """
    return ArtefactBundle(artefacts=[])
