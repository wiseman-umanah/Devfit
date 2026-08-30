"""
Low-level async HTTP client for the GitHub REST API.

Features
--------
- Per-run in-memory response cache keyed by URL — prevents redundant API
  calls when multiple pipeline stages request the same endpoint.
- Optional ``Authorization: Bearer <token>`` header when ``GITHUB_TOKEN``
  is set, raising the rate limit from 60 to 5,000 requests per hour.
- Graceful rate-limit handling: raises ``GitHubRateLimitError`` with a
  clear message including the ``X-RateLimit-Reset`` epoch timestamp.
- All methods are ``async`` and must be called within an asyncio event loop.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from devfit.config import get_settings

logger = logging.getLogger(__name__)


class GitHubRateLimitError(Exception):
    """
    Raised when the GitHub API returns a 429 or 403 rate-limit response.

    Parameters
    ----------
    reset_at : int
        Unix epoch timestamp at which the rate limit resets.
    """

    def __init__(self, reset_at: int) -> None:
        """
        Initialise with the reset epoch timestamp.

        Parameters
        ----------
        reset_at : int
            Unix epoch timestamp at which the rate limit resets.
        """
        self.reset_at = reset_at
        super().__init__(
            f"GitHub API rate limit exceeded.  Resets at epoch {reset_at}.  "
            "Set GITHUB_TOKEN in .env for a higher limit."
        )


class GitHubClient:
    """
    Async GitHub REST API client with per-run response caching.

    One instance should be created per pipeline run and shared across all
    stages to maximise cache hits.  Use as an async context manager to
    ensure the underlying ``httpx.AsyncClient`` is properly closed.

    Parameters
    ----------
    base_url : str
        Base URL for the GitHub API.  Defaults to ``https://api.github.com``.
    token : str | None
        GitHub personal-access token.  When ``None``, requests are
        unauthenticated (60 req/hr rate limit).

    Examples
    --------
    >>> async with GitHubClient() as client:
    ...     profile = await client.get("/users/torvalds")
    """

    def __init__(
        self,
        base_url: str | None = None,
        token: str | None = None,
    ) -> None:
        """
        Initialise the client, building auth headers from settings if needed.

        Parameters
        ----------
        base_url : str | None
            Override the GitHub API base URL (useful for testing).
        token : str | None
            Override the GitHub token (useful for testing).
        """
        settings = get_settings()
        self._base_url = base_url or settings.github_api_base
        resolved_token = token or (
            settings.github_token.get_secret_value()
            if settings.github_token
            else None
        )
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if resolved_token:
            headers["Authorization"] = f"Bearer {resolved_token}"

        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=30.0,
            follow_redirects=True,
        )
        # In-memory cache: url → parsed JSON.  Cleared when the client closes.
        self._cache: dict[str, Any] = {}

    async def get(self, path: str) -> Any:
        """
        Perform a GET request, returning the parsed JSON response body.

        Results are cached for the lifetime of this client instance so that
        subsequent calls to the same ``path`` within a single pipeline run
        never touch the network.

        Parameters
        ----------
        path : str
            API path relative to the base URL, e.g. ``"/users/torvalds"``.

        Returns
        -------
        Any
            Parsed JSON response (dict or list depending on the endpoint).

        Raises
        ------
        GitHubRateLimitError
            When the API returns HTTP 429 or a 403 with a rate-limit body.
        httpx.HTTPStatusError
            For any other non-2xx response.
        """
        if path in self._cache:
            logger.debug("Cache hit: %s", path)
            return self._cache[path]

        logger.debug("GET %s", path)
        response = await self._http.get(path)

        if response.status_code in (429, 403):
            reset_at = int(response.headers.get("X-RateLimit-Reset", 0))
            raise GitHubRateLimitError(reset_at)

        response.raise_for_status()
        data: Any = response.json()
        self._cache[path] = data
        return data

    async def __aenter__(self) -> GitHubClient:
        """Return self to support use as an async context manager."""
        return self

    async def __aexit__(self, *_: object) -> None:
        """Close the underlying HTTP client and clear the cache."""
        await self._http.aclose()
        self._cache.clear()
