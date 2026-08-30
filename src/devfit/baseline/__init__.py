"""
Baseline sub-package — deliberately simple single-prompt pipeline.

Purpose
-------
The baseline is built **last** and kept intentionally inferior to DevFit.
Its sole purpose is to provide a fair comparison point for measuring
DevFit's hallucination-rate improvement.

Constraints (per TRD §4)
------------------------
- Single direct LLM prompt (no tool calls, no structured artefact bundle).
- Input: JD text + a short, unstructured GitHub profile summary.
- Output: a CV + a short fit comment.
- No classification, no verification, no artefact pointers, no rule layer.
- Must run on the **identical** test cases as DevFit.

Modules
-------
pipeline
    ``BaselinePipeline`` — the single-prompt Groq call.
"""
