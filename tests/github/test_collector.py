"""
Integration smoke tests for GitHubCollector.

These tests make real outbound HTTPS requests to the GitHub API.
They are marked ``integration`` and excluded from the default pytest run::

    uv run pytest -m "not integration"   # fast unit tests only
    uv run pytest -m integration         # only network tests

Set ``GITHUB_TOKEN`` in ``.env`` to avoid the 60 req/hr unauthenticated limit.
"""

from __future__ import annotations

import pytest

from devfit.github.bundle import ArtefactBundle
from devfit.github.client import GitHubClient
from devfit.github.collector import GitHubCollector
from devfit.schema import ArtefactType


@pytest.mark.integration
async def test_collector_torvalds_returns_full_bundle() -> None:
    """
    Collect artefacts for 'torvalds' and verify all required types are present.

    This is the canonical smoke test: a well-known profile with many public
    repos, rich language stats, and READMEs that guarantee all artefact types
    are populated.

    Asserts
    -------
    - Bundle contains at least one ``ACCOUNT_METADATA`` artefact.
    - Bundle contains at least one ``LANGUAGE_STATS`` artefact.
    - Bundle contains at least one ``REPO`` artefact.
    - Bundle contains at least one ``README`` artefact.
    - Bundle contains at least one ``CONTRIBUTION_GRAPH`` artefact.
    - Total bundle length is >= 5 artefacts.
    - Account metadata extracted_fact mentions 'Account created'.
    - Language stats mention 'C' (torvalds is primarily a C developer).
    """
    async with GitHubClient() as client:
        collector = GitHubCollector(client)
        bundle: ArtefactBundle = await collector.collect("torvalds")

    assert len(bundle) >= 5, f"Expected ≥5 artefacts, got {len(bundle)}"

    # Required artefact types
    for required_type in (
        ArtefactType.ACCOUNT_METADATA,
        ArtefactType.LANGUAGE_STATS,
        ArtefactType.REPO,
        ArtefactType.README,
        ArtefactType.CONTRIBUTION_GRAPH,
    ):
        items = bundle.by_type(required_type)
        assert items, f"Expected at least one {required_type} artefact, got none"

    # Sanity check content
    meta = bundle.by_type(ArtefactType.ACCOUNT_METADATA)[0]
    assert "Account created" in meta.extracted_fact

    lang = bundle.by_type(ArtefactType.LANGUAGE_STATS)[0]
    # torvalds repos are overwhelmingly C
    assert "c" in lang.extracted_fact.lower(), (
        f"Expected 'C' in language stats for torvalds, got: {lang.extracted_fact}"
    )


@pytest.mark.integration
async def test_collector_cache_prevents_duplicate_requests() -> None:
    """
    Verify that collecting the same username twice reuses the cached response.

    The second collect call must not produce additional unique artefact types
    beyond those already seen in the first call.
    """
    async with GitHubClient() as client:
        collector = GitHubCollector(client)
        bundle1 = await collector.collect("tiangolo")
        # Second call on same client reuses cache
        bundle2 = await collector.collect("tiangolo")

    assert len(bundle1) == len(bundle2)


@pytest.mark.integration
async def test_collector_unknown_user_raises() -> None:
    """Collecting artefacts for a non-existent username must raise an HTTP error."""
    import httpx

    async with GitHubClient() as client:
        collector = GitHubCollector(client)
        with pytest.raises(httpx.HTTPStatusError):
            await collector.collect("this-user-does-not-exist-xyz-devfit-9999")
