"""
Output sub-package: CV generation, review, fit-report, evidence appendix, and trajectory logger.

Public API
----------
``StandaloneCVGenerator``
    Produces a professional general-purpose CV from a GitHub ArtefactBundle
    alone.  No job description or claim verdicts required.  Uses the
    ``standalone_cv.txt`` prompt.  Falls back to a deterministic CV on LLM
    failure.  Post-processing (em-dash, emoji, filler, source-tag) applied
    unconditionally.

``CVReviewer``
    Second-pass quality auditor.  Reviews a generated CV draft against the
    candidate's GitHub profile and returns a structured ``ReviewResult``
    flagging any violations (em-dashes, emojis, filler phrases, invented facts,
    ATS structure issues, missing sections).  Auto-fixes Rule 1-3 violations
    via post-processing.  Never blocks the pipeline on failure.

``ReviewResult``
    Structured verdict from ``CVReviewer.review()``.  ``approved`` is ``True``
    when no violations were found.

``CVGenerator``
    Produces a JD-tailored Markdown CV from verified claim verdicts.  Falls back
    to a deterministic CV if the LLM call fails.  Every verified line carries a
    ``[source: <pointer>]`` tag hidden in the PDF export.

``FitReportGenerator``
    Produces a Markdown fit report.  Score built only from verified/contradicted.
    Unverifiable claims listed separately, never scored.

``ImprovementGenerator``
    Produces concrete GitHub improvement suggestions for contradicted and
    unverifiable-technical claims.

``EvidenceAppendix``
    Renders the full artefact pointer index for every claim in the run.

``TrajectoryLogger``
    Append-only JSONL logger for all pipeline events and verifier decisions.
"""

from devfit.output.agents import AgentPipelineResult, CVAgentPipeline
from devfit.output.appendix import EvidenceAppendix
from devfit.output.cv import CVGenerator, CVLine
from devfit.output.cv_reviewer import CVReviewer, ReviewResult
from devfit.output.improvements import ImprovementGenerator
from devfit.output.pdf import export_cv_to_pdf
from devfit.output.report import FitReportGenerator, FitScore
from devfit.output.standalone_cv import StandaloneCVGenerator
from devfit.output.trajectory import TrajectoryLogger

__all__ = [
    "FitReportGenerator",
    "FitScore",
    "CVGenerator",
    "CVLine",
    "StandaloneCVGenerator",
    "CVAgentPipeline",
    "AgentPipelineResult",
    "CVReviewer",
    "ReviewResult",
    "ImprovementGenerator",
    "EvidenceAppendix",
    "TrajectoryLogger",
    "export_cv_to_pdf",
]
