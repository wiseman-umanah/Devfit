"""
Optional database session factory.

Import guard
------------
This module raises a clear ``ImportError`` at import time if the ``db``
optional dependency group is not installed, rather than producing a
confusing ``ModuleNotFoundError`` deep inside SQLAlchemy internals.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any


def _require_db() -> None:
    """
    Raise a helpful error when the ``db`` extra is not installed.

    Raises
    ------
    ImportError
        Always, when SQLAlchemy is not installed.
    """
    try:
        import sqlalchemy.ext.asyncio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Database support requires the 'db' optional dependency group.  "
            "Install it with: uv sync --extra db"
        ) from exc


def create_session_factory(database_url: str) -> Any:
    """
    Create an async SQLAlchemy session factory for the given database URL.

    Supported URL schemes
    ---------------------
    ``sqlite+aiosqlite:///<path>``
        Local SQLite database.
    ``postgresql+asyncpg://<user>:<pass>@<host>/<db>``
        PostgreSQL via asyncpg.

    Parameters
    ----------
    database_url : str
        An async-compatible SQLAlchemy database URL.

    Returns
    -------
    async_sessionmaker[AsyncSession]
        A factory that produces ``AsyncSession`` instances.

    Raises
    ------
    ImportError
        When the ``db`` extra is not installed.
    """
    _require_db()
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    engine = create_async_engine(database_url, echo=False, future=True)
    return async_sessionmaker(engine, expire_on_commit=False)


@asynccontextmanager
async def get_async_session(
    factory: Any,
) -> AsyncGenerator[Any, None]:
    """
    Yield a single database session, committing on success and rolling back on error.

    Parameters
    ----------
    factory : Any
        Session factory created by ``create_session_factory()``.

    Yields
    ------
    AsyncSession
        An open SQLAlchemy async session.

    Raises
    ------
    ImportError
        When the ``db`` extra is not installed.
    """
    _require_db()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
