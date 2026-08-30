"""
Mandatory human-review checkpoint sub-package.

The checkpoint is a required deliverable for the competition submission — it
must be captured in the trajectory log and must require explicit user action
before any final artefact is written to disk.

Public API
----------
``HumanCheckpoint``
    Presents draft report + CV in the terminal.
    Prompts for approve / edit / abort.
    Logs the interaction to ``trajectory_log.jsonl``.
    Returns ``None`` if the user aborted — callers must check.
"""

from devfit.checkpoint.checkpoint import HumanCheckpoint

__all__ = ["HumanCheckpoint"]
