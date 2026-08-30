r"""
Evaluation scoring script for DevFit vs Baseline.

Computes three metrics (reported separately, never blended):

1. Hallucination Rate
   For both Baseline and DevFit: percentage of claims presented as
   ``verified`` or ``contradicted`` in the output that have zero supporting
   artefact pointers — i.e. claims asserted as fact with no evidence.

2. Misclassification Rate  (DevFit only)
   Percentage of DevFit verdicts that disagree with the ground-truth label.

3. Unsupported CV Claim Rate  (DevFit only)
   Percentage of claims in the final DevFit CV that lack an artefact pointer.
   Target: 0%.

Usage
-----
.. code-block:: bash

    uv run python eval/score.py \
        --ground-truth eval/ground_truth.json \
        --devfit-outputs eval/devfit_outputs/ \
        --baseline-outputs eval/baseline_outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def load_json(path: Path) -> list[dict[str, object]] | dict[str, object]:
    """
    Load and parse a JSON file.

    Parameters
    ----------
    path : Path
        Path to a JSON file.

    Returns
    -------
    list[dict[str, object]] | dict[str, object]
        Parsed JSON content.

    Raises
    ------
    SystemExit
        If the file does not exist or cannot be parsed.
    """
    if not path.exists():
        logger.error("File not found: %s", path)
        sys.exit(1)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error in %s: %s", path, exc)
        sys.exit(1)


def compute_hallucination_rate(verdicts: list[dict[str, object]]) -> float:
    """
    Compute the hallucination rate for a list of verdict dicts.

    A verdict is considered a hallucination when it has classification
    ``verified`` or ``contradicted`` but an empty ``evidence`` list —
    i.e. the system asserted a fact without citing any artefact.

    Parameters
    ----------
    verdicts : list[dict[str, object]]
        List of verdict objects from a pipeline output file.

    Returns
    -------
    float
        Hallucination rate as a fraction in ``[0.0, 1.0]``.
        Returns ``0.0`` if the verdict list is empty.
    """
    if not verdicts:
        return 0.0
    factual = [
        v for v in verdicts
        if v.get("classification") in ("verified", "contradicted")
    ]
    if not factual:
        return 0.0
    hallucinated = [
        v for v in factual
        if not v.get("evidence")
    ]
    return len(hallucinated) / len(factual)


def compute_misclassification_rate(
    verdicts: list[dict[str, object]],
    ground_truth: dict[str, str],
) -> float:
    """
    Compute the misclassification rate for DevFit verdicts vs ground truth.

    Only evaluates claims that appear in both the verdict list and the
    ground-truth dict.  Claims present in ground truth but absent from
    verdicts are counted as misclassified (the system failed to produce
    a verdict).

    Parameters
    ----------
    verdicts : list[dict[str, object]]
        DevFit verdict objects from a pipeline output file.
    ground_truth : dict[str, str]
        Mapping of ``claim_id → correct_classification``.

    Returns
    -------
    float
        Misclassification rate as a fraction in ``[0.0, 1.0]``.
    """
    if not ground_truth:
        return 0.0
    verdict_map = {str(v["claim_id"]): str(v["classification"]) for v in verdicts}
    misclassified = sum(
        1
        for claim_id, correct in ground_truth.items()
        if verdict_map.get(claim_id) != correct
    )
    return misclassified / len(ground_truth)


def compute_unsupported_cv_rate(cv_claims: list[dict[str, object]]) -> float:
    """
    Compute the rate of CV claims that lack an artefact pointer.

    Parameters
    ----------
    cv_claims : list[dict[str, object]]
        List of CV claim objects, each expected to have an
        ``artefact_pointer`` field.

    Returns
    -------
    float
        Fraction of CV claims without a pointer, in ``[0.0, 1.0]``.
        Returns ``0.0`` if the list is empty.
    """
    if not cv_claims:
        return 0.0
    unsupported = [c for c in cv_claims if not c.get("artefact_pointer")]
    return len(unsupported) / len(cv_claims)


def collect_outputs(output_dir: Path) -> list[dict[str, object]]:
    """
    Collect all verdict JSON files from a pipeline output directory.

    Each case output is expected to be a JSON file named ``<case_id>.json``
    containing a dict with a ``"verdicts"`` list.

    Parameters
    ----------
    output_dir : Path
        Directory containing per-case pipeline output files.

    Returns
    -------
    list[dict[str, object]]
        Flat list of all verdict dicts across all output files.
    """
    all_verdicts: list[dict[str, object]] = []
    if not output_dir.exists():
        logger.warning("Output directory not found: %s — skipping", output_dir)
        return all_verdicts
    for output_file in sorted(output_dir.glob("*.json")):
        data = load_json(output_file)
        if isinstance(data, dict):
            verdicts = data.get("verdicts", [])
            if isinstance(verdicts, list):
                all_verdicts.extend(verdicts)
    return all_verdicts


def main() -> None:
    """
    Run the evaluation, printing a results table to stdout.

    Parses CLI arguments, loads ground truth and pipeline outputs, and
    computes + prints the three metrics for Baseline and DevFit.
    """
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Score DevFit vs Baseline against ground-truth labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Path to eval/ground_truth.json",
    )
    parser.add_argument(
        "--devfit-outputs",
        required=True,
        type=Path,
        help="Directory containing DevFit per-case output JSON files.",
    )
    parser.add_argument(
        "--baseline-outputs",
        required=True,
        type=Path,
        help="Directory containing Baseline per-case output JSON files.",
    )
    args = parser.parse_args()

    # Load ground truth
    raw_labels = load_json(args.ground_truth)
    if not isinstance(raw_labels, list):
        logger.error("ground_truth.json must be a JSON array")
        sys.exit(1)
    ground_truth: dict[str, str] = {
        str(label["claim_id"]): str(label["correct_classification"])
        for label in raw_labels
    }
    logger.info("Loaded %d ground-truth labels", len(ground_truth))

    # Collect outputs
    devfit_verdicts = collect_outputs(args.devfit_outputs)
    baseline_verdicts = collect_outputs(args.baseline_outputs)
    logger.info(
        "Loaded %d DevFit verdicts, %d Baseline verdicts",
        len(devfit_verdicts),
        len(baseline_verdicts),
    )

    # Compute metrics
    devfit_hall = compute_hallucination_rate(devfit_verdicts)
    baseline_hall = compute_hallucination_rate(baseline_verdicts)
    devfit_misclass = compute_misclassification_rate(devfit_verdicts, ground_truth)
    devfit_unsupported = compute_unsupported_cv_rate(devfit_verdicts)

    # Print results table
    print()
    print("=" * 60)
    print("  DevFit Evaluation Results")
    print("=" * 60)
    print()
    print("HEADLINE METRIC")
    print(f"  Hallucination Rate — Baseline : {baseline_hall:.1%}")
    print(f"  Hallucination Rate — DevFit   : {devfit_hall:.1%}")
    improvement = baseline_hall - devfit_hall
    print(f"  Improvement                   : {improvement:+.1%}")
    print()
    print("DEVFIT-ONLY METRICS")
    print(f"  Misclassification Rate        : {devfit_misclass:.1%}")
    print(f"  CV Claims Without Evidence    : {devfit_unsupported:.1%}  (target: 0%)")
    print()
    print(f"  Ground-truth labels           : {len(ground_truth)}")
    print(f"  DevFit verdicts evaluated     : {len(devfit_verdicts)}")
    print(f"  Baseline verdicts evaluated   : {len(baseline_verdicts)}")
    print("=" * 60)
    print()

    # Non-zero unsupported CV claims is a hard failure
    if devfit_unsupported > 0:
        logger.error(
            "FAIL: %.1f%% of DevFit CV claims lack an artefact pointer. "
            "Target is 0%%.",
            devfit_unsupported * 100,
        )
        sys.exit(2)


if __name__ == "__main__":
    main()
