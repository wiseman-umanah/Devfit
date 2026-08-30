"""
Independent Verifier sub-package — the core quality gate.

Architecture
------------
The verifier runs in two strictly ordered, independent layers:

1. **Rule-based layer** (``rules`` module) — deterministic, zero LLM calls,
   zero external I/O.  Runs date arithmetic, language-presence checks, and
   zero-activity contradiction checks.  Claims resolved here are finalised
   and never sent to the LLM layer.

2. **Constrained LLM layer** (``llm`` module) — a Groq call with a narrower
   prompt than the first-pass classifier.  It must cite the exact artefact
   pointer it is relying on, or return ``unverifiable``.

Both layers must be able to **disagree** with the first-pass classifier and
with each other.  If they always agree, the "independent" framing is false.

Public API
----------
``IndependentVerifier``
    Orchestrates Layer 1 then Layer 2.  Returns final verdicts and
    ``VerifierDecision`` records for trajectory logging.

``VerifierDecision``
    Structured record of each verifier decision (layer, draft vs final
    classification, was_downgraded, reason).

``RuleVerifier``
    Deterministic rule layer — use standalone for unit testing.

``ConstrainedLLMVerifier``
    LLM layer — only processes claims the rule layer cannot resolve.
"""

from devfit.verifier.llm import ConstrainedLLMVerifier
from devfit.verifier.rules import RuleVerifier
from devfit.verifier.verifier import IndependentVerifier, VerifierDecision

__all__ = [
    "IndependentVerifier",
    "VerifierDecision",
    "RuleVerifier",
    "ConstrainedLLMVerifier",
]
