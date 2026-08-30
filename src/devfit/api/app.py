"""
FastAPI application factory — optional HTTP surface for DevFit.

The API is entirely optional.  The CLI (``devfit`` / ``devfit-dev``) is the
primary interface.  This module exposes the FastAPI ``app`` instance and a
``main()`` entry point for use with Uvicorn.

Routers
-------
health
    ``GET /health`` — liveness probe.
analyze
    ``POST /analyze`` — full pipeline run (JD + GitHub username → report + CV).
"""

from __future__ import annotations

from fastapi import FastAPI

from devfit import __version__
from devfit.api.routers import analyze, health

app = FastAPI(
    title="DevFit",
    description=(
        "Evidence-grounded CV and fit-report generator.  "
        "Every claim is verified against public GitHub artefacts."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.include_router(health.router)
app.include_router(analyze.router, prefix="/api/v1")


def main() -> None:
    """
    Start the Uvicorn development server.

    This function is the entry point for the ``devfit-server`` script
    defined in ``pyproject.toml``.  It is **not** used by the CLI.
    """
    import uvicorn

    uvicorn.run(
        "devfit.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
