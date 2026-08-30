"""
Mandatory human-review checkpoint sub-package.

The checkpoint is a required deliverable for the competition submission — it
must be captured in the trajectory log and must require explicit user action
before any final artefact is written to disk.

Modules
-------
checkpoint
    ``HumanCheckpoint`` — presents the draft report + CV in the terminal,
    prompts for approve / edit / abort, and logs the interaction to
    ``trajectory_log.jsonl``.
"""
