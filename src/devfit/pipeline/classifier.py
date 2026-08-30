"""
First-Pass Classifier — proposes draft ``Verdict`` objects via Groq LLM.

This is the **proposer** stage, not the quality gate.  Every draft verdict
it produces will be independently re-checked by the ``IndependentVerifier``
in Stage 6 before it is allowed to stand.

Design
------
- ``likely_unverifiable`` claims are never sent here; the ``EvidenceMatcher``
  already short-circuited them.
- One Groq call per claim (claims are independent — concurrent with
  ``asyncio.gather`` over the full batch).
- Uses ``first_pass_classifier.txt`` prompt, which is deliberately broader
  than the constrained verifier prompt.
- Returns a draft ``Verdict`` with ``llm_confirmed=False`` and
  ``rule_checked=False`` — those flags are set by the verifier layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from groq import AsyncGroq

from devfit.config import get_settings
from devfit.pipeline.matcher import MatchedClaim
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

    Raises
    ------
    FileNotFoundError
        If the prompt file does not exist.
    """
    path = _PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def _parse_classifier_response(
    raw: str,
    claim_id: str,
    candidates: list[Artefact],
) -> Verdict:
    """
    Parse the first-pass classifier JSON response into a draft ``Verdict``.

    Falls back to ``unverifiable`` with zero evidence on any parse error so
    the pipeline never stalls — the verifier will handle the downgrade.

    Parameters
    ----------
    raw : str
        Raw LLM response string.
    claim_id : str
        Claim ID to assign to the verdict.
    candidates : list[Artefact]
        Candidate artefacts provided to the classifier (used to resolve
        pointer references back to full ``Artefact`` objects).

    Returns
    -------
    Verdict
        Draft verdict with ``rule_checked=False`` and ``llm_confirmed=False``.
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
            "First-pass classifier returned invalid JSON for '%s': %s",
            claim_id, exc,
        )
        return Verdict(
            claim_id=claim_id,
            classification=Classification.UNVERIFIABLE,
            confidence=0.0,
            evidence=[],
        )

    try:
        classification = Classification(str(data.get("classification", "unverifiable")))
    except ValueError:
        classification = Classification.UNVERIFIABLE

    raw_confidence = data.get("confidence", 0.5)
    try:
        confidence = float(raw_confidence)  # type: ignore[arg-type]
        confidence = max(0.0, min(1.0, confidence))
    except (TypeError, ValueError):
        confidence = 0.5

    # Resolve pointer strings back to full Artefact objects
    pointer_index = {a.pointer: a for a in candidates}
    evidence_pointers: list[str] = []
    raw_pointers = data.get("evidence_pointers", [])
    if isinstance(raw_pointers, list):
        evidence_pointers = [str(p) for p in raw_pointers]

    evidence = [
        pointer_index[p] for p in evidence_pointers if p in pointer_index
    ]

    # Enforce schema: non-unverifiable must have evidence
    if classification != Classification.UNVERIFIABLE and not evidence:
        logger.warning(
            "Classifier proposed '%s' for claim '%s' with no resolvable "
            "evidence pointers — downgrading to unverifiable",
            classification, claim_id,
        )
        classification = Classification.UNVERIFIABLE
        confidence = 0.0

    return Verdict(
        claim_id=claim_id,
        classification=classification,
        confidence=confidence,
        evidence=evidence,
        rule_checked=False,
        llm_confirmed=False,
    )


class FirstPassClassifier:
    """
    Classify evidence-matched claims via Groq LLM (proposer stage).

    Processes the full batch concurrently using ``asyncio.gather``.
    Claims that were short-circuited by the matcher (``skipped=True``)
    are passed through directly from the matcher's pre-built unverifiable
    verdicts and never reach this class.

    Examples
    --------
    >>> classifier = FirstPassClassifier()
    >>> draft_verdicts = await classifier.classify(matched_claims)
    """

    def __init__(self) -> None:
        """Initialise the classifier, loading the prompt template once."""
        self._prompt_template = _load_prompt("first_pass_classifier")
        settings = get_settings()
        self._client = AsyncGroq(
            api_key=settings.groq_api_key.get_secret_value()
        )
        self._model = settings.groq_model

    async def classify(
        self, matched_claims: list[MatchedClaim]
    ) -> list[Verdict]:
        """
        Produce a draft ``Verdict`` for every non-skipped matched claim.

        Skipped claims (``likely_unverifiable``) must have their verdicts
        built by ``EvidenceMatcher.build_unverifiable_verdicts()`` before
        calling this method; they are excluded here.

        All LLM calls are dispatched concurrently via ``asyncio.gather``.

        Parameters
        ----------
        matched_claims : list[MatchedClaim]
            Output from ``EvidenceMatcher.match()`` — skipped claims are
            filtered out automatically.

        Returns
        -------
        list[Verdict]
            Draft verdicts in the same order as the non-skipped input claims.
        """
        active = [mc for mc in matched_claims if not mc.skipped]
        if not active:
            return []

        logger.info(
            "First-pass classifier: classifying %d claims concurrently",
            len(active),
        )
        tasks = [self._classify_one(mc) for mc in active]
        verdicts: list[Verdict] = await asyncio.gather(*tasks)
        return list(verdicts)

    async def _classify_one(self, mc: MatchedClaim) -> Verdict:
        """
        Classify a single ``MatchedClaim`` with one Groq call.

        Parameters
        ----------
        mc : MatchedClaim
            A non-skipped matched claim with its candidate artefacts.

        Returns
        -------
        Verdict
            Draft verdict for this claim.
        """
        claim_json = json.dumps(
            {
                "id": mc.claim.id,
                "text": mc.claim.text,
                "category": mc.claim.category,
                "source": mc.claim.source,
            },
            indent=2,
        )
        artefacts_json = json.dumps(
            [
                {
                    "pointer": a.pointer,
                    "type": a.type,
                    "extracted_fact": a.extracted_fact,
                }
                for a in mc.candidates
            ],
            indent=2,
        )

        prompt = self._prompt_template.format(
            claim_json=claim_json,
            artefacts_json=artefacts_json,
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=512,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:
            logger.error(
                "LLM call failed for claim '%s': %s — defaulting to unverifiable",
                mc.claim.id, exc,
            )
            raw = (
                '{"classification": "unverifiable", '
                '"confidence": 0.0, "evidence_pointers": []}'
            )

        return _parse_classifier_response(raw, mc.claim.id, mc.candidates)
