"""
Optional database sub-package (SQLAlchemy 2.x async).

This sub-package is **not imported** unless the ``db`` optional dependency
group is installed (``uv sync --extra db``).  The rest of the application
must function correctly when this sub-package is absent.

Database support
----------------
Two async backends are supported:

``sqlite+aiosqlite:///<path>``
    Suitable for local development and single-machine eval runs.

``postgresql+asyncpg://<user>:<pass>@<host>/<db>``
    Suitable for production deployments.

Switch by setting ``DATABASE_URL`` in the ``.env`` file.  When
``DATABASE_URL`` is absent or ``None``, all DB functionality is disabled
and the app runs in stateless mode.

Modules
-------
session
    ``get_async_session`` — async context manager yielding an
    ``AsyncSession``.  Raises ``RuntimeError`` when the ``db`` extra is
    not installed.
models
    SQLAlchemy ORM models mirroring the core Pydantic schema.
migrations
    Alembic environment and migration scripts.
"""
