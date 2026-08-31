"""
DevFit — GitHub-profile-driven CV generator and evidence-grounded fit reporter.

Public surface
--------------
Importing this package gives access to the top-level version string only.
All pipeline stages are accessed through their respective sub-packages.

Sub-packages
------------
devfit.schema
    Frozen Pydantic v2 models shared by all pipeline stages.
devfit.config
    Settings loaded from environment variables / .env file.
devfit.github
    Async GitHub REST API client and ArtefactBundle builder.
devfit.pipeline
    JD Analyzer, Evidence Matcher, and First-Pass Classifier.
devfit.verifier
    Independent Verifier: rule-based layer + constrained LLM layer.
devfit.output
    CV generators, fit-report, evidence-appendix, and PDF export.
devfit.baseline
    Deliberately simple single-prompt baseline for comparison.
devfit.db (optional)
    SQLAlchemy models and async session factory.
    Only imported when the ``db`` extra is installed.
devfit.api
    FastAPI application — web UI and REST API served by ``devfit-server``.
"""

__version__ = "0.1.0"
