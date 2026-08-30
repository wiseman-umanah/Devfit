"""
Health-check router.

Provides a simple ``GET /health`` endpoint for liveness probes.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """
    Response body for the health endpoint.

    Parameters
    ----------
    status : str
        Always ``"ok"`` when the service is healthy.
    version : str
        Current application version string.
    """

    status: str
    version: str


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """
    Return service health status.

    Returns
    -------
    HealthResponse
        ``{"status": "ok", "version": "<version>"}``
    """
    from devfit import __version__

    return HealthResponse(status="ok", version=__version__)
