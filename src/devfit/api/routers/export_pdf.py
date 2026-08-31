"""
Export PDF router -- ``POST /api/v1/export-pdf``.

Accepts CV Markdown, converts it to PDF in a temporary directory, streams
the PDF bytes back, and immediately discards the temp files.  Nothing
persists after the response is sent.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel, Field

router = APIRouter(tags=["cv"])
logger = logging.getLogger(__name__)


class ExportPdfRequest(BaseModel):
    """
    Request body for the PDF export endpoint.

    Parameters
    ----------
    cv_markdown : str
        The CV in Markdown format.
    filename : str
        Suggested filename for the download (without ``.pdf`` extension).
    """

    cv_markdown: str = Field(..., min_length=10)
    filename: str = Field(default="cv", min_length=1, max_length=80)


@router.post(
    "/export-pdf",
    summary="Export CV Markdown as a PDF",
    response_class=Response,
    responses={
        200: {
            "content": {"application/pdf": {}},
            "description": "PDF file stream.",
        },
        503: {"description": "pandoc or Chrome not available on this server."},
        500: {"description": "PDF generation failed."},
    },
)
async def export_pdf(request: ExportPdfRequest) -> Response:
    """
    Convert CV Markdown to a PDF and return it as a binary stream.

    Uses pandoc + Chrome headless.  The temp directory is deleted immediately
    after the PDF bytes are read into memory — nothing persists on disk.

    Parameters
    ----------
    request : ExportPdfRequest
        CV Markdown and desired filename.

    Returns
    -------
    Response
        PDF bytes with ``Content-Disposition: attachment`` header.

    Raises
    ------
    HTTPException
        ``503`` if pandoc or Chrome are not available on the server.
        ``500`` if PDF generation fails for any other reason.
    """
    from devfit.output.pdf import _find_chrome, _find_pandoc, export_cv_to_pdf

    if not _find_pandoc():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="pandoc is not installed on this server. PDF export unavailable.",
        )
    if not _find_chrome():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chrome/Chromium is not installed. PDF export unavailable.",
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        md_path = tmp_path / "cv.md"
        pdf_path = tmp_path / "cv.pdf"
        md_path.write_text(request.cv_markdown, encoding="utf-8")

        success = export_cv_to_pdf(md_path, pdf_path)
        if not success or not pdf_path.exists():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF generation failed. Check server logs.",
            )

        pdf_bytes = pdf_path.read_bytes()

    safe_name = "".join(
        c for c in request.filename if c.isalnum() or c in "-_ "
    ).strip() or "cv"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}.pdf"',
            "Content-Length": str(len(pdf_bytes)),
        },
    )
