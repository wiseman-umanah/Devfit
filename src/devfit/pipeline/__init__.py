"""
Pipeline sub-package: JD Analyzer, Evidence Matcher, and First-Pass Classifier.

Build order (per TRD §9)
------------------------
These stages are built **after** the rule-based verifier layer and **after**
the test-set ground-truth labels are frozen (see ``eval/``).

Modules
-------
analyzer
    ``JDAnalyzer`` — parses raw JD text into a list of ``Claim`` objects
    via a Groq LLM call.  Prompts are loaded from
    ``src/devfit/prompts/jd_analyzer.txt``.
matcher
    ``EvidenceMatcher`` — pairs each ``Claim`` with candidate ``Artefact``
    objects from the ``ArtefactBundle``.
classifier
    ``FirstPassClassifier`` — produces draft ``Verdict`` objects via a Groq
    LLM call.  This is the **proposer**, not the quality gate.
"""
