"""
Async GitHub REST API client and structured ArtefactBundle builder.

All network I/O in this sub-package is performed with ``httpx.AsyncClient``
so the GitHub Collector can run concurrently with other async tasks.

Modules
-------
client
    Low-level async HTTP client with per-run in-memory cache and optional
    PAT authentication.
collector
    High-level ``GitHubCollector`` that fetches all required endpoints and
    assembles an ``ArtefactBundle``.
bundle
    ``ArtefactBundle`` data structure — a typed container keyed by
    ``ArtefactType`` for O(1) lookup by downstream pipeline stages.
"""
