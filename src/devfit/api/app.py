"""
FastAPI application factory for DevFit.

Routes
------
GET  /                     → CV generator UI (Jinja2 template, no username pre-filled)
GET  /{username}           → CV generator UI with GitHub username pre-filled
GET  /health               → Liveness probe
POST /api/v1/generate      → GitHub username (+ optional JD) → CV Markdown
POST /api/v1/edit          → Highlighted text + instruction → rewritten section
POST /api/v1/match         → CV + JD → match score + gap analysis
POST /api/v1/export-pdf    → CV Markdown → PDF binary stream
POST /api/v1/analyze       → Full JD-fit pipeline (JD required)

Static assets
-------------
All CSS and JS are served from ``src/devfit/api/static/`` under the ``/static``
URL prefix.  The HTML template lives in ``src/devfit/api/templates/index.html``.

Design notes
------------
- No CV state is stored server-side.  The browser holds the CV Markdown in a
  ``let cvMarkdown`` variable only.  Nothing persists between page reloads.
- The reference CV is read by the browser via ``FileReader`` — it never leaves
  the client except as a JSON string in the ``/api/v1/generate`` body.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from devfit import __version__
from devfit.api.routers import analyze, edit, export_pdf, generate, health, match

_HERE = Path(__file__).parent
_TEMPLATES = Jinja2Templates(directory=str(_HERE / "templates"))

app = FastAPI(
    title="DevFit",
    description=(
        "AI CV generator grounded in your GitHub profile. "
        "Every claim is verifiable."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

app.include_router(health.router)
app.include_router(generate.router, prefix="/api/v1")
app.include_router(edit.router, prefix="/api/v1")
app.include_router(match.router, prefix="/api/v1")
app.include_router(export_pdf.router, prefix="/api/v1")
app.include_router(analyze.router, prefix="/api/v1")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def ui_root(request: Request) -> HTMLResponse:
    """
    Serve the CV generator UI with no username pre-filled.

    Returns
    -------
    HTMLResponse
        Rendered ``index.html`` template.
    """
    return _TEMPLATES.TemplateResponse(
        request, "index.html", {"username": ""}
    )


@app.get("/{username}", response_class=HTMLResponse, include_in_schema=False)
async def ui_username(request: Request, username: str) -> HTMLResponse:
    """
    Serve the CV generator UI with the GitHub username pre-filled.

    Navigating to ``/torvalds`` pre-populates the username field so the
    user only needs to click Generate.

    Parameters
    ----------
    request : Request
        The incoming FastAPI request.
    username : str
        GitHub username extracted from the URL path.

    Returns
    -------
    HTMLResponse
        Rendered ``index.html`` template with ``username`` context variable.
    """
    return _TEMPLATES.TemplateResponse(
        request, "index.html", {"username": username}
    )


def main() -> None:
    """
    Start the Uvicorn server for the DevFit web UI.

    Registered in ``pyproject.toml`` under ``[project.scripts]`` as
    ``devfit-server``.
    """
    import uvicorn

    uvicorn.run(
        "devfit.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
