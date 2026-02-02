#!/usr/bin/env python3
"""
Filter treatment edges from DRKG for fair Node2Vec comparison.

PURPOSE:
    Remove treatment-related edges from DRKG to enable fair comparison with TxGNN.
    TxGNN evaluates on diseases with NO treatment edges - our kNN method benefits
    from Node2Vec embeddings trained on graphs including treatment edges.

EDGES REMOVED:
    - GNBR::T::Compound:Disease (54,020) - treats
    - DRUGBANK::treats::Compound:Disease (4,968)
    - Hetionet::CtD::Compound:Disease (755) - treats
    - Hetionet::CpD::Compound:Disease (390) - palliates
    - GNBR::Pa::Compound:Disease (2,619) - alleviates
    - GNBR::Pr::Compound:Disease (966) - prevents

EDGES KEPT:
    - GNBR::Sa::Compound:Disease (16,923) - side effects (not treatment)
    - All other edge types (gene-disease, drug-gene, etc.)

OUTPUT:
    data/raw/drkg/drkg_no_treatment.tsv (~5.81M edges)
"""

import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DRKG_PATH = PROJECT_ROOT / "data" / "raw" / "drkg" / "drkg.tsv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw" / "drkg" / "drkg_no_treatment.tsv"

# Edge types to remove (treatment-related)
TREATMENT_EDGE_TYPES = frozenset([
    "GNBR::T::Compound:Disease",
    "DRUGBANK::treats::Compound:Disease",
    "Hetionet::CtD::Compound:Disease",
    "Hetionet::CpD::Compound:Disease",
    "GNBR::Pa::Compound:Disease",
    "GNBR::Pr::Compound:Disease",
])


def main() -> None:
    print("=" * 70)
    print("FILTER DRKG TREATMENT EDGES")
    print("=" * 70)
    print()

    if not DRKG_PATH.exists():
        print(f"ERROR: DRKG not found at {DRKG_PATH}")
        sys.exit(1)

    print(f"Input:  {DRKG_PATH}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Edge types to remove:")
    for et in sorted(TREATMENT_EDGE_TYPES):
        print(f"  - {et}")
    print()

    removed_counts: Counter[str] = Counter()
    kept_counts: Counter[str] = Counter()
    total_lines = 0
    kept_lines = 0

    with open(DRKG_PATH, "r") as fin, open(OUTPUT_PATH, "w") as fout:
        for line in fin:
            total_lines += 1
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue

            edge_type = parts[1]

            if edge_type in TREATMENT_EDGE_TYPES:
                removed_counts[edge_type] += 1
            else:
                fout.write(line)
                kept_lines += 1
                # Track kept edge types (sample first 100K for memory)
                if kept_lines <= 100000:
                    kept_counts[edge_type] += 1

    total_removed = sum(removed_counts.values())

    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print(f"Total input edges:   {total_lines:,}")
    print(f"Edges removed:       {total_removed:,} ({100*total_removed/total_lines:.2f}%)")
    print(f"Edges kept:          {kept_lines:,} ({100*kept_lines/total_lines:.2f}%)")
    print()

    print("Removed by type:")
    for et in sorted(TREATMENT_EDGE_TYPES):
        count = removed_counts.get(et, 0)
        print(f"  {count:>6,}  {et}")
    print()

    print(f"Output written to: {OUTPUT_PATH}")
    print()

    # Verify the output
    expected_removed = 54020 + 4968 + 755 + 390 + 2619 + 966  # = 63,718
    if abs(total_removed - expected_removed) > 100:
        print(f"WARNING: Expected ~{expected_removed:,} removed edges, got {total_removed:,}")
    else:
        print("Verification: Edge counts match expected values")


if __name__ == "__main__":
    main()
