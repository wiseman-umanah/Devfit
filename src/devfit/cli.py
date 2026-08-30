"""
CLI entry points for DevFit.

Commands
--------
``devfit``
    Production entry point.  Runs the full DevFit pipeline.

``devfit-dev``
    Development entry point.  Sets ``LOG_LEVEL=DEBUG`` and
    ``DEVFIT_ENV=development`` before delegating to the same pipeline,
    providing verbose output useful during active development.

Usage
-----
.. code-block:: bash

    devfit --jd jd.txt --github torvalds
    devfit --jd jd.txt --github torvalds --resume resume.txt
    devfit-dev --jd jd.txt --github torvalds

Both commands are registered as ``[project.scripts]`` in ``pyproject.toml``
so they are available directly after ``uv sync``.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _configure_logging(level: str) -> None:
    """
    Configure the root logger with a clean format.

    Parameters
    ----------
    level : str
        Python logging level string, e.g. ``"INFO"`` or ``"DEBUG"``.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse CLI arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list.  Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: ``jd``, ``github``,
        ``resume``, ``output``, ``include_unverifiable``.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="devfit",
        description=(
            "Generate an evidence-grounded CV and fit report.\n"
            "Every claim is verified against public GitHub artefacts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--jd",
        required=True,
        metavar="FILE_OR_TEXT",
        help="Path to a JD file, or inline JD text.",
    )
    parser.add_argument(
        "--github",
        required=True,
        metavar="USERNAME",
        help="Public GitHub username of the candidate.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="FILE",
        help="Optional path to a resume file (plain text or Markdown).",
    )
    parser.add_argument(
        "--output",
        default="output",
        metavar="DIR",
        help="Directory to write final output files (default: ./output).",
    )
    parser.add_argument(
        "--include-unverifiable",
        action="store_true",
        default=False,
        help=(
            "Include unverifiable claims in the CV with a visible marker.  "
            "Disabled by default."
        ),
    )
    return parser.parse_args(argv)


def _read_input(path_or_text: str) -> str:
    """
    Return text from a file path or pass through the string directly.

    Parameters
    ----------
    path_or_text : str
        Either a filesystem path to a readable text file, or inline text.

    Returns
    -------
    str
        The resolved text content.
    """
    candidate = Path(path_or_text)
    if candidate.exists() and candidate.is_file():
        return candidate.read_text(encoding="utf-8")
    return path_or_text


async def _run_pipeline(
    jd_text: str,
    github_username: str,
    resume_text: str | None,
    output_dir: Path,
    include_unverifiable: bool,
) -> None:
    """
    Async pipeline runner — wires together all stages end-to-end.

    This function will be fleshed out incrementally as stages are implemented
    (see steps.txt Stages 3–8).  Currently it raises ``NotImplementedError``
    to clearly signal that the pipeline is not yet complete.

    Parameters
    ----------
    jd_text : str
        Full job description text.
    github_username : str
        Candidate's public GitHub username.
    resume_text : str | None
        Optional resume text.
    output_dir : Path
        Directory where output files will be written after human approval.
    include_unverifiable : bool
        Whether to include unverifiable claims in the CV output.

    Raises
    ------
    NotImplementedError
        Until the pipeline stages are wired in (see steps.txt).
    """
    # TODO(stage-3): wire in GitHubCollector
    # TODO(stage-4): wire in JDAnalyzer
    # TODO(stage-5): wire in EvidenceMatcher + FirstPassClassifier
    # TODO(stage-6): wire in IndependentVerifier
    # TODO(stage-7): wire in FitReportGenerator + CVGenerator
    # TODO(stage-8): wire in HumanCheckpoint
    raise NotImplementedError(
        "Pipeline not yet implemented.  Follow steps.txt build order."
    )


def _main(dev_mode: bool = False) -> None:
    """
    Shared entry-point logic for both ``devfit`` and ``devfit-dev``.

    Parameters
    ----------
    dev_mode : bool
        When ``True``, overrides ``LOG_LEVEL`` to ``DEBUG`` and sets
        ``DEVFIT_ENV=development`` before loading settings.
    """
    if dev_mode:
        os.environ.setdefault("LOG_LEVEL", "DEBUG")
        os.environ.setdefault("DEVFIT_ENV", "development")

    from devfit.config import get_settings

    settings = get_settings()
    _configure_logging(settings.log_level)

    args = _parse_args()
    jd_text = _read_input(args.jd)
    resume_text = _read_input(args.resume) if args.resume else None
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        asyncio.run(
            _run_pipeline(
                jd_text=jd_text,
                github_username=args.github,
                resume_text=resume_text,
                output_dir=output_dir,
                include_unverifiable=args.include_unverifiable,
            )
        )
    except NotImplementedError as exc:
        logger.error("Pipeline stub: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Aborted by user.")
        sys.exit(130)


def main() -> None:
    """
    Production CLI entry point (``devfit`` command).

    Registered in ``pyproject.toml`` under ``[project.scripts]``.
    """
    _main(dev_mode=False)


def main_dev() -> None:
    """
    Development CLI entry point (``devfit-dev`` command).

    Sets ``LOG_LEVEL=DEBUG`` and ``DEVFIT_ENV=development`` before running
    the pipeline, providing verbose output useful during active development.

    Registered in ``pyproject.toml`` under ``[project.scripts]``.
    """
    _main(dev_mode=True)
