"""
DevFit — production entry point.

This file exists only to satisfy the case where someone runs
``python main.py`` directly.  The canonical way to run DevFit is via
the ``devfit`` or ``devfit-dev`` CLI commands registered in
``pyproject.toml`` (available after ``uv sync``).
"""

from devfit.cli import main

if __name__ == "__main__":
    main()
