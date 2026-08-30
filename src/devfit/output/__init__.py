"""
Output sub-package: fit-report, CV, evidence-appendix, and trajectory logger.

Public API
----------
``FitReportGenerator``
    Produces a Markdown fit report.  Score built only from verified/contradicted.
    Unverifiable claims listed separately, never scored.

``CVGenerator``
    Produces an ATS-structured Markdown CV.  Every verified claim line carries
    a ``[source: <pointer>]`` tag.  Unverifiable claims only on explicit request,
    always marked ``[CANNOT BE CONFIRMED FROM GITHUB]``.

``EvidenceAppendix``
    Renders the full artefact pointer index for every claim in the run.

``TrajectoryLogger``
    Append-only JSONL logger for all pipeline events and verifier decisions.
"""

from devfit.output.appendix import EvidenceAppendix
from devfit.output.cv import CVGenerator, CVLine
from devfit.output.report import FitReportGenerator, FitScore
from devfit.output.trajectory import TrajectoryLogger

__all__ = [
    "FitReportGenerator",
    "FitScore",
    "CVGenerator",
    "CVLine",
    "EvidenceAppendix",
    "TrajectoryLogger",
]
