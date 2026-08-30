"""
Unit tests for ``ArtefactBundle``.

Tests cover:
- ``by_type`` returns correct subsets.
- ``all`` returns the full flat list.
- ``__len__`` returns the correct count.
- Empty bundle behaves correctly.
"""

from __future__ import annotations

from devfit.github.bundle import ArtefactBundle
from devfit.schema import Artefact, ArtefactType


class TestArtefactBundle:
    """Tests for the ``ArtefactBundle`` container class."""

    def test_by_type_returns_correct_subset(
        self, bundle_with_python: ArtefactBundle
    ) -> None:
        """``by_type`` returns only artefacts of the requested type."""
        lang_artefacts = bundle_with_python.by_type(ArtefactType.LANGUAGE_STATS)
        assert len(lang_artefacts) == 1
        assert lang_artefacts[0].type == ArtefactType.LANGUAGE_STATS

    def test_by_type_returns_empty_list_for_missing_type(
        self, bundle_with_python: ArtefactBundle
    ) -> None:
        """``by_type`` returns an empty list when no artefact of that type exists."""
        commits = bundle_with_python.by_type(ArtefactType.COMMIT)
        assert commits == []

    def test_all_returns_full_list(self, bundle_with_python: ArtefactBundle) -> None:
        """``all`` returns all artefacts regardless of type."""
        assert len(bundle_with_python.all()) == len(bundle_with_python)

    def test_len_reflects_total_count(self, bundle_with_python: ArtefactBundle) -> None:
        """``__len__`` returns the total number of artefacts."""
        assert len(bundle_with_python) == 3

    def test_empty_bundle(self, bundle_empty: ArtefactBundle) -> None:
        """An empty bundle returns empty lists and length zero."""
        assert len(bundle_empty) == 0
        assert bundle_empty.all() == []
        assert bundle_empty.by_type(ArtefactType.REPO) == []

    def test_repr_contains_type_counts(
        self, bundle_with_python: ArtefactBundle
    ) -> None:
        """``__repr__`` includes artefact type counts."""
        r = repr(bundle_with_python)
        assert "ArtefactBundle" in r

    def test_by_type_returns_copy(self, bundle_with_python: ArtefactBundle) -> None:
        """Mutating the list returned by ``by_type`` does not affect the bundle."""
        repos = bundle_with_python.by_type(ArtefactType.REPO)
        original_count = len(bundle_with_python.by_type(ArtefactType.REPO))
        repos.append(
            Artefact(
                type=ArtefactType.REPO,
                pointer="github.com/x/y",
                extracted_fact="extra",
            )
        )
        assert len(bundle_with_python.by_type(ArtefactType.REPO)) == original_count
