"""
Progress Tracking Utilities for Research Loop
==============================================

Functions for tracking and displaying progress of the autonomous research agent.
"""

import json
from pathlib import Path
from typing import Any


def count_hypothesis_status(project_dir: Path) -> dict[str, int]:
    """
    Count hypotheses by status in research_roadmap.json.

    Args:
        project_dir: Directory containing research_roadmap.json

    Returns:
        Dict with counts: {pending, in_progress, validated, invalidated, inconclusive, total}
    """
    roadmap_file = project_dir / "research_roadmap.json"

    counts = {
        "pending": 0,
        "in_progress": 0,
        "validated": 0,
        "invalidated": 0,
        "inconclusive": 0,
        "total": 0
    }

    if not roadmap_file.exists():
        return counts

    try:
        with open(roadmap_file, "r") as f:
            data = json.load(f)

        hypotheses = data.get("hypotheses", [])
        counts["total"] = len(hypotheses)

        for h in hypotheses:
            status = h.get("status", "pending")
            if status in counts:
                counts[status] += 1
            else:
                counts["pending"] += 1

        return counts
    except (json.JSONDecodeError, IOError):
        return counts


def get_next_hypothesis(project_dir: Path) -> dict[str, Any] | None:
    """
    Get the highest-priority pending hypothesis.

    Returns:
        Hypothesis dict or None if no pending hypotheses
    """
    roadmap_file = project_dir / "research_roadmap.json"

    if not roadmap_file.exists():
        return None

    try:
        with open(roadmap_file, "r") as f:
            data = json.load(f)

        pending = [h for h in data.get("hypotheses", []) if h.get("status") == "pending"]
        if not pending:
            return None

        # Sort by priority (lower number = higher priority)
        pending.sort(key=lambda x: x.get("priority", 999))
        return pending[0]
    except (json.JSONDecodeError, IOError):
        return None


def print_session_header(session_num: int, is_initializer: bool) -> None:
    """Print a formatted header for the session."""
    session_type = "RESEARCH INITIALIZER" if is_initializer else "RESEARCH AGENT"

    print("\n" + "=" * 70)
    print(f"  SESSION {session_num}: {session_type}")
    print("=" * 70)
    print()


def print_progress_summary(project_dir: Path) -> None:
    """Print a summary of current research progress."""
    counts = count_hypothesis_status(project_dir)

    if counts["total"] > 0:
        completed = counts["validated"] + counts["invalidated"] + counts["inconclusive"]
        percentage = (completed / counts["total"]) * 100

        print(f"\n{'=' * 50}")
        print("  RESEARCH PROGRESS")
        print(f"{'=' * 50}")
        print(f"  Total hypotheses:  {counts['total']}")
        print(f"  Pending:           {counts['pending']}")
        print(f"  In Progress:       {counts['in_progress']}")
        print(f"  Validated:         {counts['validated']}")
        print(f"  Invalidated:       {counts['invalidated']}")
        print(f"  Inconclusive:      {counts['inconclusive']}")
        print(f"  Completion:        {completed}/{counts['total']} ({percentage:.1f}%)")
        print(f"{'=' * 50}")

        # Show next hypothesis
        next_h = get_next_hypothesis(project_dir)
        if next_h:
            print(f"\n  Next: [{next_h['id']}] {next_h['title']}")
            print(f"         Impact: {next_h.get('expected_impact', 'unknown')}, "
                  f"Effort: {next_h.get('effort', 'unknown')}")
    else:
        print("\nProgress: research_roadmap.json not yet created")
