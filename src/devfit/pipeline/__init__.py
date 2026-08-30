"""
Pipeline sub-package.

JD Analyzer, Resume Analyzer, Evidence Matcher, and First-Pass Classifier.

Build order (per TRD §9)
------------------------
These stages are built **after** the rule-based verifier layer and **after**
the test-set ground-truth labels are frozen (see ``eval/``).

Public API
----------
``JDAnalyzer``
    Parses raw JD text into a list of ``Claim`` objects via a Groq LLM call.
    Pre-flags ``SOFT_SKILL`` and ``EXPERIENCE_DURATION`` claims as
    ``likely_unverifiable=True`` to save Evidence Matcher retrieval effort.

``ResumeAnalyzer``
    Same extraction logic applied to optional resume text.
    Returns claims with ``source=ClaimSource.RESUME``.

Stubs (implemented in later stages)
------------------------------------
``EvidenceMatcher``   — Stage 5
``FirstPassClassifier`` — Stage 5
"""

from devfit.pipeline.analyzer import JDAnalyzer, ResumeAnalyzer

__all__ = ["JDAnalyzer", "ResumeAnalyzer"]
