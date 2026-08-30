"""
GitHubCollector — fetches all required endpoints and builds an ArtefactBundle.

Concurrency strategy
--------------------
All independent API requests (repos, language stats, README files) are
dispatched concurrently with ``asyncio.gather`` to minimise wall-clock time.
The ``GitHubClient`` cache ensures no URL is fetched more than once even if
multiple concurrent tasks request the same endpoint.

Rate-limit behaviour
--------------------
If a ``GitHubRateLimitError`` is raised during collection, the collector
logs the reset timestamp and re-raises — callers must handle the error.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from datetime import UTC, datetime
from typing import Any

from devfit.github.bundle import ArtefactBundle
from devfit.github.client import GitHubClient
from devfit.schema import Artefact, ArtefactType

logger = logging.getLogger(__name__)

# Maximum number of repos to deep-inspect (README + language stats).
# Top repos are chosen by star count to prioritise signal-dense content.
_TOP_REPO_LIMIT = 10


class GitHubCollector:
    """
    Collect public GitHub signals for a username and return an ArtefactBundle.

    Parameters
    ----------
    client : GitHubClient
        An open ``GitHubClient`` instance (must be used within its async
        context manager or otherwise kept alive during collection).

    Examples
    --------
    >>> async with GitHubClient() as client:
    ...     collector = GitHubCollector(client)
    ...     bundle = await collector.collect("torvalds")
    """

    def __init__(self, client: GitHubClient) -> None:
        """
        Initialise the collector with a shared client.

        Parameters
        ----------
        client : GitHubClient
            Open ``GitHubClient`` — shared to maximise cache hits.
        """
        self._client = client

    async def collect(self, username: str) -> ArtefactBundle:
        """
        Fetch all public signals for *username* and return an ArtefactBundle.

        Endpoints fetched
        -----------------
        - ``/users/{username}`` — account metadata (creation date, bio, etc.)
        - ``/users/{username}/repos`` — public repo list
        - ``/repos/{owner}/{repo}/languages`` — language breakdown (top repos)
        - ``/repos/{owner}/{repo}/readme`` — README content (top repos)
        - ``/users/{username}/events/public`` — recent public commit activity

        Parameters
        ----------
        username : str
            Public GitHub username.

        Returns
        -------
        ArtefactBundle
            Structured artefact container ready for downstream pipeline stages.
        """
        logger.info("Collecting GitHub data for '%s'", username)

        profile_task = self._client.get(f"/users/{username}")
        repos_task = self._client.get(
            f"/users/{username}/repos?per_page=100&sort=pushed&type=owner"
        )
        events_task = self._client.get(
            f"/users/{username}/events/public?per_page=100"
        )

        profile_raw, repos_raw, events_raw = await asyncio.gather(
            profile_task, repos_task, events_task
        )

        artefacts: list[Artefact] = []

        # --- Account metadata ---
        artefacts.append(self._build_account_metadata(username, profile_raw))

        # --- Language stats (concurrent across top repos by stars) ---
        # repos_raw is list[Any] from the GitHub API.
        top_repos: list[dict[str, Any]] = sorted(
            repos_raw,
            key=lambda r: int(r.get("stargazers_count") or 0),
            reverse=True,
        )[:_TOP_REPO_LIMIT]

        lang_tasks = [
            self._client.get(f"/repos/{username}/{r['name']}/languages")
            for r in top_repos
        ]
        readme_tasks = [
            self._client.get(f"/repos/{username}/{r['name']}/readme")
            for r in top_repos
        ]

        lang_results, readme_results = await asyncio.gather(
            asyncio.gather(*lang_tasks, return_exceptions=True),
            asyncio.gather(*readme_tasks, return_exceptions=True),
        )

        # Merge language stats across all top repos
        combined_langs: dict[str, int] = {}
        for result in lang_results:
            if isinstance(result, dict):
                for lang, count in result.items():
                    combined_langs[lang] = combined_langs.get(lang, 0) + int(count)

        if combined_langs:
            sorted_langs = sorted(
                combined_langs.items(), key=lambda x: x[1], reverse=True
            )
            top_langs = ", ".join(
                f"{lang} ({count:,} bytes)" for lang, count in sorted_langs[:10]
            )
            artefacts.append(
                Artefact(
                    type=ArtefactType.LANGUAGE_STATS,
                    pointer=f"github.com/{username}",
                    extracted_fact=f"Language breakdown across top repos: {top_langs}",
                )
            )

        # Per-repo artefacts
        for repo, readme_result in zip(top_repos, readme_results, strict=False):
            repo_name = str(repo.get("name", ""))
            artefacts.append(self._build_repo_artefact(username, repo))
            if isinstance(readme_result, dict):
                artefacts.append(
                    self._build_readme_artefact(username, repo_name, readme_result)
                )

        # Contribution graph proxy from public events
        artefacts.append(self._build_contribution_artefact(username, events_raw))

        bundle = ArtefactBundle(artefacts=artefacts)
        logger.info("Bundle built for '%s': %r", username, bundle)
        return bundle

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_account_metadata(username: str, profile: dict[str, Any]) -> Artefact:
        """
        Convert a raw GitHub ``/users/{username}`` response to an Artefact.

        Parameters
        ----------
        username : str
            GitHub username.
        profile : dict[str, Any]
            Raw API response dict.

        Returns
        -------
        Artefact
            Account metadata artefact.
        """
        created_raw = str(profile.get("created_at", "unknown"))
        public_repos = int(profile.get("public_repos", 0) or 0)
        bio = str(profile.get("bio") or "").strip()

        try:
            created_dt = datetime.fromisoformat(
                created_raw.rstrip("Z")
            ).replace(tzinfo=UTC)
            created_str = created_dt.strftime("%Y-%m-%d")
            age_days = (datetime.now(UTC) - created_dt).days
            age_str = f"{age_days // 365}y {(age_days % 365) // 30}m"
        except ValueError:
            created_str = created_raw
            age_str = "unknown"

        fact = (
            f"Account created {created_str} ({age_str} ago), "
            f"{public_repos} public repos"
        )
        if bio:
            fact += f", bio: \"{bio[:120]}\""

        return Artefact(
            type=ArtefactType.ACCOUNT_METADATA,
            pointer=f"github.com/{username}",
            extracted_fact=fact,
        )

    @staticmethod
    def _build_repo_artefact(username: str, repo: dict[str, Any]) -> Artefact:
        """
        Convert a raw GitHub repo dict to a repo Artefact.

        Parameters
        ----------
        username : str
            Repository owner's GitHub username.
        repo : dict[str, Any]
            Single entry from the ``/users/{username}/repos`` response.

        Returns
        -------
        Artefact
            Repo-level artefact.
        """
        name = str(repo.get("name", ""))
        lang = str(repo.get("language") or "unknown")
        stars = int(repo.get("stargazers_count", 0) or 0)
        desc = str(repo.get("description") or "").strip()[:100]
        updated = str(repo.get("updated_at", ""))[:10]

        fact = f"{name} ({lang}, {stars} stars, updated {updated})"
        if desc:
            fact += f" — {desc}"

        return Artefact(
            type=ArtefactType.REPO,
            pointer=f"github.com/{username}/{name}",
            extracted_fact=fact,
        )

    @staticmethod
    def _build_readme_artefact(
        username: str, repo_name: str, readme_raw: dict[str, Any]
    ) -> Artefact:
        """
        Decode a base64-encoded README response and extract a summary excerpt.

        Parameters
        ----------
        username : str
            Repository owner's GitHub username.
        repo_name : str
            Repository name.
        readme_raw : dict[str, Any]
            Raw API response from ``/repos/{owner}/{repo}/readme``.

        Returns
        -------
        Artefact
            README artefact with the first 500 characters of decoded content.
        """
        content_b64 = str(readme_raw.get("content", "")).replace("\n", "")
        try:
            content = base64.b64decode(content_b64).decode("utf-8", errors="replace")
            excerpt = content[:500].strip()
        except Exception:
            excerpt = "(could not decode README)"

        return Artefact(
            type=ArtefactType.README,
            pointer=f"github.com/{username}/{repo_name}/blob/HEAD/README.md",
            extracted_fact=f"README excerpt: {excerpt}",
        )

    @staticmethod
    def _build_contribution_artefact(
        username: str, events: list[dict[str, Any]]
    ) -> Artefact:
        """
        Summarise public commit activity from the events feed.

        Parameters
        ----------
        username : str
            GitHub username.
        events : list[dict[str, Any]]
            Raw response from ``/users/{username}/events/public``.

        Returns
        -------
        Artefact
            Contribution-graph proxy artefact.
        """
        push_events = [e for e in events if e.get("type") == "PushEvent"]
        commit_count = sum(
            len((e.get("payload") or {}).get("commits") or [])
            for e in push_events
        )
        repos_pushed = {
            str((e.get("repo") or {}).get("name") or "")
            for e in push_events
        }

        fact = (
            f"{commit_count} public commits across {len(repos_pushed)} repos "
            f"in the most recent 100 events"
        )
        return Artefact(
            type=ArtefactType.CONTRIBUTION_GRAPH,
            pointer=f"github.com/{username}",
            extracted_fact=fact,
        )
