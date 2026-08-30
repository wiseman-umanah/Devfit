"""
Mandatory human-review checkpoint.

This is a required competition deliverable — it must:
1. Present the draft report + CV to the user in the terminal.
2. Require explicit ``(a)pprove``, ``(e)dit``, or ``(q)abort`` action.
3. Log the interaction to the trajectory log.
4. Never write final output files without explicit approval.

The checkpoint interaction is captured in the trajectory log so it can be
exported as part of the agent trajectory submission package.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from devfit.output.trajectory import TrajectoryLogger

logger = logging.getLogger(__name__)

_PROMPT = (
    "\n[DevFit] Review complete.  Choose an action:\n"
    "  (a) Approve — write final output files\n"
    "  (e) Edit   — open draft in $EDITOR for changes\n"
    "  (q) Abort  — discard and exit\n"
    "Choice [a/e/q]: "
)


class HumanCheckpoint:
    """
    Present the draft report and CV for human review before finalising.

    The checkpoint is non-negotiable: no output files are written until the
    user explicitly approves.  This satisfies competition Rule 5 (qualified
    human reviewer for anything significantly affecting someone).

    Parameters
    ----------
    tlog : TrajectoryLogger
        Open trajectory logger — the checkpoint interaction is appended here.

    Examples
    --------
    >>> checkpoint = HumanCheckpoint(tlog)
    >>> approved_report, approved_cv = await checkpoint.run(report_md, cv_md)
    """

    def __init__(self, tlog: TrajectoryLogger) -> None:
        """
        Initialise the checkpoint.

        Parameters
        ----------
        tlog : TrajectoryLogger
            Open trajectory logger for recording the interaction.
        """
        self._tlog = tlog

    def run(
        self,
        report_md: str,
        cv_md: str,
    ) -> tuple[str, str] | None:
        """
        Present the draft and wait for explicit human action.

        Prints the full report and CV to stdout, then prompts the user.
        On edit, opens ``$EDITOR`` (falls back to ``nano``) with a combined
        draft and reads back any changes.

        Parameters
        ----------
        report_md : str
            Draft fit report Markdown.
        cv_md : str
            Draft CV Markdown.

        Returns
        -------
        tuple[str, str] | None
            ``(final_report_md, final_cv_md)`` on approval, or ``None`` if
            the user aborted.
        """
        self._print_draft(report_md, cv_md)

        while True:
            try:
                raw = input(_PROMPT).strip().lower()
            except (EOFError, KeyboardInterrupt):
                raw = "q"

            if raw == "a":
                self._tlog.log_event(
                    "human_checkpoint",
                    {"action": "approved", "edits": None},
                )
                logger.info("Human checkpoint: APPROVED")
                return report_md, cv_md

            if raw == "e":
                edited_report, edited_cv, edits_summary = self._open_editor(
                    report_md, cv_md
                )
                self._tlog.log_event(
                    "human_checkpoint",
                    {"action": "edited", "edits": edits_summary},
                )
                logger.info("Human checkpoint: EDITED")
                # Show edited version and prompt again
                self._print_draft(edited_report, edited_cv)
                report_md, cv_md = edited_report, edited_cv
                continue

            if raw == "q":
                self._tlog.log_event(
                    "human_checkpoint",
                    {"action": "aborted", "edits": None},
                )
                logger.info("Human checkpoint: ABORTED")
                return None

            print("  Please enter 'a', 'e', or 'q'.", file=sys.stderr)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _print_draft(report_md: str, cv_md: str) -> None:
        """
        Print the draft report and CV to stdout with clear separators.

        Parameters
        ----------
        report_md : str
            Draft fit report Markdown.
        cv_md : str
            Draft CV Markdown.
        """
        sep = "=" * 72
        print(f"\n{sep}")
        print("  DRAFT FIT REPORT")
        print(sep)
        print(report_md)
        print(f"\n{sep}")
        print("  DRAFT CV")
        print(sep)
        print(cv_md)
        print(sep)

    @staticmethod
    def _open_editor(
        report_md: str, cv_md: str
    ) -> tuple[str, str, str]:
        """
        Open the combined draft in ``$EDITOR`` and read back changes.

        The report and CV are written to a temp file separated by a
        ``--- CV ---`` marker.  After the editor closes, the file is
        split back into report and CV sections.

        Parameters
        ----------
        report_md : str
            Current report draft.
        cv_md : str
            Current CV draft.

        Returns
        -------
        tuple[str, str, str]
            ``(new_report_md, new_cv_md, edits_summary)`` where
            ``edits_summary`` is a short string for the trajectory log.
        """
        separator = "\n\n<!-- --- CV --- -->\n\n"
        combined = report_md + separator + cv_md

        editor = os.environ.get("EDITOR", "nano")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".md",
            encoding="utf-8",
            delete=False,
        ) as tmp:
            tmp.write(combined)
            tmp_path = Path(tmp.name)

        try:
            subprocess.run([editor, str(tmp_path)], check=False)
            edited = tmp_path.read_text(encoding="utf-8")
        finally:
            tmp_path.unlink(missing_ok=True)

        if separator in edited:
            parts = edited.split(separator, maxsplit=1)
            new_report = parts[0].strip()
            new_cv = parts[1].strip()
        else:
            # Separator removed by user — treat whole text as report
            new_report = edited.strip()
            new_cv = cv_md

        original_len = len(combined)
        edited_len = len(new_report) + len(new_cv)
        edits_summary = (
            f"chars before={original_len} after={edited_len} "
            f"delta={edited_len - original_len:+d}"
        )
        return new_report, new_cv, edits_summary
