"""
Independent Verifier sub-package — the core quality gate.

Architecture
------------
The verifier runs in two strictly ordered layers:

1. **Rule-based layer** (``rules`` module) — deterministic, zero LLM calls,
   zero external I/O.  Runs date arithmetic, language-presence checks, and
   zero-activity contradiction checks.  Claims resolved here are finalised
   and never sent to the LLM layer.

2. **Constrained LLM layer** (``llm`` module) — a Groq call with a narrower
   prompt than the first-pass classifier.  It must cite the exact artefact
   pointer it is relying on, or return ``unverifiable``.

Both layers must be able to **disagree** with the first-pass classifier and
with each other.  If they always agree, the "independent" framing is false.

Modules
-------
rules
    ``RuleVerifier`` — deterministic rule layer (no LLM, no network).
llm
    ``ConstrainedLLMVerifier`` — narrow Groq re-check layer.
verifier
    ``IndependentVerifier`` — orchestrates the two layers.
"""
