"""
Constrained LLM verifier — Layer 2 of the Independent Verifier.

This is a **narrower, stricter** Groq call than the first-pass classifier.
Its only job is to confirm or reject a *specific proposed verdict* against
*specific evidence* — it is not allowed to re-classify from scratch.

Key invariants (enforced here, not just in the prompt)
------------------------------------------------------
- Must cite the exact artefact pointer from the draft verdict, or the
  response is treated as a downgrade to ``unverifiable``.
- Can only confirm or downgrade — never upgrade a verdict.
- Uses a different prompt file (``constrained_verifier.txt``) so the two
  LLM stages are structurally distinct and can disagree.
- Sets ``Verdict.llm_confirmed = True`` only when the LLM confirmed the
  proposed classification.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from groq import AsyncGroq

from devfit.config import get_settings
from devfit.schema import Artefact, Classification, Verdict

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"


def _load_prompt(name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Parameters
    ----------
    name : str
        Filename without extension.

    Returns
    -------
    str
        Raw prompt template string.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_verifier_response(
    raw: str,
    draft: Verdict,
    candidates: list[Artefact],
) -> Verdict:
    """
    Parse the constrained verifier JSON response and apply confirm/downgrade.

    Never upgrades a verdict.  Any parse error or missing pointer results in
    a downgrade to ``unverifiable``.

    Parameters
    ----------
    raw : str
        Raw LLM response string.
    draft : Verdict
        The draft verdict being re-checked.
    candidates : list[Artefact]
        Artefacts the verifier was given to work with.

    Returns
    -------
    Verdict
        Confirmed (``llm_confirmed=True``) or downgraded verdict.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(
            line for line in text.splitlines()
            if not line.strip().startswith("```")
        )

    try:
        data: dict[str, object] = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Constrained verifier returned invalid JSON for claim '%s': "
            "%s — downgrading",
            draft.claim_id, exc,
        )
        return _downgrade(draft, "Verifier returned unparseable response")

    decision = str(data.get("decision", "downgraded")).lower()

    if decision == "confirmed":
        # Verify the cited pointer actually exists in candidates
        pointer = str(data.get("artefact_pointer", ""))
        pointer_index = {a.pointer: a for a in candidates}
        if pointer not in pointer_index:
            logger.warning(
                "Verifier confirmed claim '%s' but cited unknown "
                "pointer '%s' — downgrading",
                draft.claim_id, pointer,
            )
            return _downgrade(
                draft, f"Cited pointer '{pointer}' not in evidence bundle"
            )

        logger.debug(
            "Verifier CONFIRMED claim '%s' as '%s'",
            draft.claim_id, draft.classification,
        )
        return Verdict(
            claim_id=draft.claim_id,
            classification=draft.classification,
            # slight boost on confirmation
            confidence=min(draft.confidence + 0.05, 1.0),
            evidence=draft.evidence,
            rule_checked=draft.rule_checked,
            llm_confirmed=True,
        )

    # Downgrade path
    raw_new = str(data.get("new_classification", "unverifiable")).lower()
    try:
        new_classification = Classification(raw_new)
    except ValueError:
        new_classification = Classification.UNVERIFIABLE

    # Cannot upgrade — enforce the constraint
    _upgrade_map = {
        (Classification.CONTRADICTED, Classification.VERIFIED),
        (Classification.UNVERIFIABLE, Classification.VERIFIED),
        (Classification.UNVERIFIABLE, Classification.CONTRADICTED),
    }
    if (draft.classification, new_classification) in _upgrade_map:
        logger.warning(
            "Verifier attempted to UPGRADE claim '%s' from '%s' to "
            "'%s' — ignoring, keeping original classification",
            draft.claim_id, draft.classification, new_classification,
        )
        new_classification = draft.classification

    reason = str(data.get("reason", "Verifier did not confirm"))
    logger.info(
        "Verifier DOWNGRADED claim '%s': '%s' → '%s' — %s",
        draft.claim_id, draft.classification, new_classification, reason,
    )
    return _downgrade(draft, reason, new_classification)


def _downgrade(
    draft: Verdict,
    reason: str,
    new_classification: Classification = Classification.UNVERIFIABLE,
) -> Verdict:
    """
    Build a downgraded verdict from a draft, preserving claim identity.

    Parameters
    ----------
    draft : Verdict
        The original draft verdict being downgraded.
    reason : str
        Human-readable reason logged for trajectory purposes.
    new_classification : Classification
        Target classification (defaults to ``UNVERIFIABLE``).

    Returns
    -------
    Verdict
        Downgraded verdict with ``llm_confirmed=False``.
    """
    logger.debug("Downgrade reason for claim '%s': %s", draft.claim_id, reason)
    # UNVERIFIABLE requires empty evidence; keep evidence for CONTRADICTED
    evidence = (
        draft.evidence if new_classification == Classification.CONTRADICTED else []
    )
    return Verdict(
        claim_id=draft.claim_id,
        classification=new_classification,
        confidence=0.0,
        evidence=evidence,
        rule_checked=draft.rule_checked,
        llm_confirmed=False,
    )


class ConstrainedLLMVerifier:
    """
    Re-check a draft verdict with a narrower, stricter Groq prompt.

    This is Layer 2 of the ``IndependentVerifier``.  It only runs when the
    rule-based Layer 1 returns ``None`` (i.e. could not decisively resolve
    the claim).

    The prompt used here (``constrained_verifier.txt``) is structurally
    different from ``first_pass_classifier.txt`` — it asks the LLM to
    confirm or reject a specific proposal, not classify from scratch.

    Examples
    --------
    >>> verifier = ConstrainedLLMVerifier()
    >>> final_verdict = await verifier.verify(draft_verdict, candidates)
    """

    def __init__(self) -> None:
        """Initialise the verifier, loading the prompt template once."""
        self._prompt_template = _load_prompt("constrained_verifier")
        settings = get_settings()
        self._client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value()
        )
        self._model = settings.groq_model

    async def verify(
        self,
        draft: Verdict,
        candidates: list[Artefact],
    ) -> Verdict:
        """
        Confirm or downgrade a draft verdict via constrained LLM call.

        Parameters
        ----------
        draft : Verdict
            Draft verdict from the first-pass classifier.
        candidates : list[Artefact]
            The candidate artefacts that were visible to the classifier.

        Returns
        -------
        Verdict
            Confirmed verdict (``llm_confirmed=True``) or a downgraded
            verdict (``llm_confirmed=False``).
        """
        cited_pointers = [a.pointer for a in draft.evidence]
        artefact_content = "\n\n".join(
            f"[{a.pointer}]\n{a.extracted_fact}"
            for a in candidates
        )

        prompt = self._prompt_template.format(
            claim_text=draft.claim_id,  # claim_id is the identifier here
            proposed_classification=draft.classification,
            cited_pointers=", ".join(cited_pointers) if cited_pointers else "(none)",
            artefact_content=artefact_content or "(no artefacts provided)",
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(
                "Constrained verifier LLM call failed for claim '%s': %s — downgrading",
                draft.claim_id, exc,
            )
            return _downgrade(draft, f"LLM call failed: {exc}")

        return _parse_verifier_response(raw, draft, candidates)
