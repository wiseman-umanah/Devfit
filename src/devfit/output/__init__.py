"""
Output sub-package: fit-report, CV, evidence-appendix, improvements, and trajectory logger.

Public API
----------
``FitReportGenerator``
    Produces a Markdown fit report.  Score built only from verified/contradicted.
    Unverifiable claims listed separately, never scored.

``CVGenerator``
    Produces a professional ATS-structured Markdown CV via Groq LLM, grounded
    strictly in verified claims.  Falls back to a deterministic bullet list if
    the LLM call fails.  Every verified line carries a ``[source: <pointer>]``
    tag.  Unverifiable claims only on explicit request, marked
    ``[NOT VERIFIED FROM GITHUB]``.

``ImprovementGenerator``
    Produces concrete GitHub improvement suggestions for contradicted and
    unverifiable-technical claims.  Falls back to a bullet list on LLM failure.

``EvidenceAppendix``
    Renders the full artefact pointer index for every claim in the run.

``TrajectoryLogger``
    Append-only JSONL logger for all pipeline events and verifier decisions.
"""

from devfit.output.appendix import EvidenceAppendix
from devfit.output.cv import CVGenerator, CVLine
from devfit.output.improvements import ImprovementGenerator
from devfit.output.pdf import export_cv_to_pdf
from devfit.output.report import FitReportGenerator, FitScore
from devfit.output.trajectory import TrajectoryLogger

__all__ = [
    "FitReportGenerator",
    "FitScore",
    "CVGenerator",
    "CVLine",
    "ImprovementGenerator",
    "EvidenceAppendix",
    "TrajectoryLogger",
    "export_cv_to_pdf",
]
