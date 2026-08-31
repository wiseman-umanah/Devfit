"""
Edit router -- ``POST /api/v1/edit``.

Accepts a highlighted section of the CV, the full CV context, and a plain
English instruction from the user.  Returns the replacement text for that
section only.  Nothing is written to disk.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(tags=["cv"])
logger = logging.getLogger(__name__)

_EDIT_PROMPT = """\
You are a professional CV editor. The user has highlighted a section of their CV
and given you an instruction. Rewrite ONLY the highlighted section according to
the instruction. Do not touch any other part of the CV.

RULES:
1. Return ONLY the rewritten section. No explanation, no preamble.
2. No em-dashes (— or –). Use commas or periods.
3. No emojis.
4. No filler phrases: "passionate about", "results-driven", "dynamic",
   "proven track record", "detail-oriented".
5. Stay factual — do not add claims not present in the original text or context.
6. Match the formatting style of the surrounding CV (Markdown).

FULL CV CONTEXT (for reference only — do not rewrite this):
{full_cv}

HIGHLIGHTED SECTION TO REWRITE:
{selected_text}

USER INSTRUCTION:
{instruction}

Rewritten section:"""


class EditRequest(BaseModel):
    """
    Request body for the AI edit endpoint.

    Parameters
    ----------
    full_cv : str
        The complete current CV Markdown (context for the model).
    selected_text : str
        The highlighted section the user wants to edit.
    instruction : str
        Plain-English instruction, e.g. "Make this more specific" or
        "Rewrite as past-tense bullets".
    """

    full_cv: str = Field(..., min_length=10)
    selected_text: str = Field(..., min_length=1)
    instruction: str = Field(..., min_length=3, max_length=500)


class EditResponse(BaseModel):
    """
    Response body for the AI edit endpoint.

    Parameters
    ----------
    replacement : str
        The rewritten section, ready to replace the selected text.
    """

    replacement: str


@router.post(
    "/edit",
    response_model=EditResponse,
    status_code=status.HTTP_200_OK,
    summary="AI-edit a highlighted CV section",
)
async def edit(request: EditRequest) -> EditResponse:
    """
    Rewrite a highlighted CV section according to the user's instruction.

    Parameters
    ----------
    request : EditRequest
        Full CV context, selected text, and edit instruction.

    Returns
    -------
    EditResponse
        The replacement text for the selected section.

    Raises
    ------
    HTTPException
        ``500`` if the LLM call fails.
    """
    from groq import AsyncGroq

    from devfit.config import get_settings
    from devfit.output.cv_utils import _post_process

    settings = get_settings()
    prompt = _EDIT_PROMPT.format(
        full_cv=request.full_cv[:4000],
        selected_text=request.selected_text,
        instruction=request.instruction,
    )

    try:
        client = AsyncGroq(api_key=settings.groq_api_key.get_secret_value())
        response = await client.chat.completions.create(
            model=settings.groq_model_reviewer,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content or ""
        await client.close()
    except Exception as exc:  # noqa: BLE001
        logger.exception("AI edit LLM call failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI edit failed: {exc}",
        ) from exc

    return EditResponse(replacement=_post_process(raw.strip()))
