#!/usr/bin/env python3
"""h904: Diff baseline vs demoted tier precision summary from h393 outputs."""

import re
import sys
from pathlib import Path


TIER_LINE = re.compile(
    r"(GOLDEN|HIGH|MEDIUM|LOW|FILTER)\s*:\s*Full=\s*([\d.]+)%\s*\|\s*Holdout=\s*([\d.]+)%\s*±\s*([\d.]+)%"
)


def parse(path: Path) -> dict:
    text = path.read_text()
    out = {}
    for m in TIER_LINE.finditer(text):
        tier, full, hold, std = m.groups()
        out[tier] = (float(full), float(hold), float(std))
    return out


def main() -> None:
    baseline_path = Path("data/analysis/h904_baseline.txt")
    demoted_path = Path("data/analysis/h904_demoted_output.txt")

    if not baseline_path.exists():
        sys.exit(f"missing {baseline_path}")
    if not demoted_path.exists():
        sys.exit(f"missing {demoted_path}")

    base = parse(baseline_path)
    demo = parse(demoted_path)

    print("=" * 78)
    print("h904: BASELINE vs DEMOTED (5-seed holdout)")
    print("=" * 78)
    print(f"{'Tier':<8}{'Baseline':<25}{'Demoted':<25}{'Δ holdout':<10}")
    print("-" * 78)
    for tier in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        if tier not in base or tier not in demo:
            print(f"{tier:<8}missing in one file")
            continue
        bf, bh, bs = base[tier]
        df, dh, ds = demo[tier]
        delta = dh - bh
        print(
            f"{tier:<8}{bh:5.1f}% ± {bs:.1f}% (full {bf:5.1f}%)   "
            f"{dh:5.1f}% ± {ds:.1f}% (full {df:5.1f}%)   "
            f"{delta:+.2f}pp"
        )


if __name__ == "__main__":
    main()
