"""
Output sub-package: fit-report, CV, and evidence-appendix generators.

Modules
-------
report
    ``FitReportGenerator`` — produces a Markdown fit report from ``Verdict``
    objects.  Only ``verified`` and ``contradicted`` verdicts are scored.
    ``unverifiable`` verdicts appear in a separate section and are never
    included in the score.
cv
    ``CVGenerator`` — produces an ATS-structured Markdown CV.  Every claim
    line must trace back to an ``Artefact`` pointer.  Unverifiable claims
    may only be included at explicit user request and are marked accordingly.
appendix
    ``EvidenceAppendix`` — renders the full list of artefact pointers with
    their ``extracted_fact`` summaries for every claim in the run.
"""
