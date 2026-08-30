"""
ArtefactBundle — a typed container of GitHub evidence keyed by ArtefactType.

The bundle is the canonical output of the ``GitHubCollector``.  All downstream
pipeline stages (Evidence Matcher, Rule Verifier, LLM Verifier) receive an
``ArtefactBundle`` and must **not** accept raw GitHub API response dicts.
Passing raw dicts downstream is the exact failure mode the baseline exhibits.
"""

from __future__ import annotations

from collections import defaultdict

from devfit.schema import Artefact, ArtefactType


class ArtefactBundle:
    """
    Immutable-ish container of ``Artefact`` objects grouped by type.

    Designed for O(1) lookup by ``ArtefactType`` — downstream stages can
    retrieve all artefacts of a given type in a single call rather than
    scanning a flat list.

    Parameters
    ----------
    artefacts : list[Artefact]
        The full set of artefacts collected for a single GitHub profile.

    Examples
    --------
    >>> bundle = ArtefactBundle(artefacts=[...])
    >>> lang_artefacts = bundle.by_type(ArtefactType.LANGUAGE_STATS)
    """

    def __init__(self, artefacts: list[Artefact]) -> None:
        """
        Build the internal index from a flat list of artefacts.

        Parameters
        ----------
        artefacts : list[Artefact]
            Flat list of all collected artefacts for one GitHub profile.
        """
        self._index: dict[ArtefactType, list[Artefact]] = defaultdict(list)
        for artefact in artefacts:
            self._index[artefact.type].append(artefact)
        # Keep a flat copy for iteration
        self._all: list[Artefact] = list(artefacts)

    def by_type(self, artefact_type: ArtefactType) -> list[Artefact]:
        """
        Return all artefacts of the given type.

        Parameters
        ----------
        artefact_type : ArtefactType
            The type of artefact to retrieve.

        Returns
        -------
        list[Artefact]
            May be empty if no artefacts of that type were collected.
        """
        return list(self._index[artefact_type])

    def all(self) -> list[Artefact]:
        """
        Return all artefacts as a flat list.

        Returns
        -------
        list[Artefact]
            All artefacts in insertion order.
        """
        return list(self._all)

    def __len__(self) -> int:
        """Return the total number of artefacts in this bundle."""
        return len(self._all)

    def __repr__(self) -> str:
        """Return a concise developer-readable representation."""
        counts = {t.value: len(v) for t, v in self._index.items() if v}
        return f"ArtefactBundle({counts})"
