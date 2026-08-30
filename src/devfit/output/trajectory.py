"""
Trajectory logger — records every pipeline decision to ``trajectory_log.jsonl``.

Every significant event (verifier decisions, LLM calls, human checkpoint
interaction) is appended as a newline-delimited JSON object so the full agent
trajectory can be exported for the competition submission.

This module is intentionally side-effect-free until ``TrajectoryLogger.open()``
is called — it writes nothing until a run is active.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """
    Append-only JSONL trajectory log for a single pipeline run.

    Opens a file at ``output_dir/trajectory_log.jsonl`` and appends one
    JSON object per event.  The file is flushed after every write so that
    partial trajectories are preserved if the process is interrupted.

    Use as a context manager to ensure the file is always closed cleanly.

    Parameters
    ----------
    output_dir : Path
        Directory where the trajectory log will be written.

    Examples
    --------
    >>> with TrajectoryLogger(output_dir) as tlog:
    ...     tlog.log_event("pipeline_start", {"github_username": "torvalds"})
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialise the logger.  Does not open the file until ``open()`` is called.

        Parameters
        ----------
        output_dir : Path
            Directory that must exist before ``open()`` is called.
        """
        self._path = output_dir / "trajectory_log.jsonl"
        self._file: Any = None

    def open(self) -> TrajectoryLogger:
        """
        Open the trajectory log file for appending.

        Returns
        -------
        TrajectoryLogger
            Self, for use as a context manager.
        """
        self._file = self._path.open("a", encoding="utf-8")
        return self

    def close(self) -> None:
        """Close the trajectory log file if open."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def log_event(self, stage: str, data: dict[str, Any]) -> None:
        """
        Append a single event to the trajectory log.

        Parameters
        ----------
        stage : str
            Pipeline stage identifier, e.g. ``"jd_analysis"``,
            ``"verifier_decision"``, ``"human_checkpoint"``.
        data : dict[str, Any]
            Arbitrary key-value payload for this event.
        """
        if self._file is None:
            logger.warning("TrajectoryLogger not open — skipping event '%s'", stage)
            return
        entry = {
            "timestamp": datetime.now(UTC).isoformat(),
            "stage": stage,
            **data,
        }
        self._file.write(json.dumps(entry, default=str) + "\n")
        self._file.flush()

    def log_verifier_decisions(self, decisions: list[Any]) -> None:
        """
        Append one trajectory event per verifier decision.

        Parameters
        ----------
        decisions : list[VerifierDecision]
            Output from ``IndependentVerifier.verify_all()``.
        """
        for decision in decisions:
            self.log_event(
                "verifier_decision",
                {
                    "claim_id": decision.claim_id,
                    "claim_text": decision.claim_text,
                    "layer": decision.layer,
                    "draft_classification": (
                        str(decision.draft_classification)
                        if decision.draft_classification
                        else None
                    ),
                    "final_classification": str(decision.final_classification),
                    "was_downgraded": decision.was_downgraded,
                    "reason": decision.reason,
                    "artefact_pointers": decision.artefact_pointers,
                },
            )

    def __enter__(self) -> TrajectoryLogger:
        """Open the log file and return self."""
        return self.open()

    def __exit__(self, *_: object) -> None:
        """Close the log file on context exit."""
        self.close()
