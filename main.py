"""
DevFit — production entry point.

This file exists for the case where someone runs ``python main.py`` directly.
The canonical way to run DevFit is via the ``devfit-server`` command
registered in ``pyproject.toml``, or via Docker:

    docker compose up
"""

from __future__ import annotations

import uvicorn


def main() -> None:
    """Start the DevFit web server."""
    uvicorn.run(
        "devfit.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
