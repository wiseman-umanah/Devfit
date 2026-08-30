"""
DevFit — evidence-grounded CV and fit-report generator.

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
    Fit-report, CV, and evidence-appendix generators.
devfit.checkpoint
    Mandatory human-review checkpoint (CLI interactive).
devfit.baseline
    Deliberately simple single-prompt baseline for comparison.
devfit.db (optional)
    SQLAlchemy models and async session factory.
    Only imported when the ``db`` extra is installed.
devfit.api
    FastAPI application (optional HTTP surface).
devfit.cli
    Click-based CLI entry points: ``devfit`` and ``devfit-dev``.
"""

__version__ = "0.1.0"
