#!/usr/bin/env python3
"""
Runner script for automated literature mining.

Usage:
    # Mine all NOVEL MEDIUM predictions
    python scripts/run_literature_mining.py --tier MEDIUM --status NOVEL

    # Mine top 200 NOVEL predictions by composite score
    python scripts/run_literature_mining.py --all-novel --top 200

    # Mine with LLM abstract classification
    python scripts/run_literature_mining.py --tier HIGH --status NOVEL --use-llm

    # Resume interrupted mining (uses cache)
    python scripts/run_literature_mining.py --tier MEDIUM --resume

    # Show summary of cached results
    python scripts/run_literature_mining.py --summary
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.literature_miner import LiteratureMiner


def load_predictions(tier: str | None = None,
                     status: str | None = None,
                     top_n: int | None = None,
                     all_novel: bool = False) -> list[dict]:
    """Load predictions from deliverable JSON."""
    json_path = (Path(__file__).parent.parent / "data" / "deliverables" /
                 "drug_repurposing_predictions_with_confidence.json")

    if not json_path.exists():
        print(f"ERROR: Deliverable not found: {json_path}")
        print("Run scripts/h420_regenerate_deliverable.py first.")
        sys.exit(1)

    with open(json_path) as f:
        all_preds = json.load(f)
    print(f"Loaded {len(all_preds)} predictions from deliverable")

    # Filter by tier
    if tier:
        tiers = [t.strip().upper() for t in tier.split(",")]
        all_preds = [p for p in all_preds if p.get("confidence_tier") in tiers]
        print(f"  After tier filter ({tier}): {len(all_preds)}")

    # Filter by literature status
    if status:
        statuses = [s.strip().upper() for s in status.split(",")]
        all_preds = [p for p in all_preds if p.get("literature_status", "").upper() in statuses]
        print(f"  After status filter ({status}): {len(all_preds)}")

    # Filter all novel (not known indication)
    if all_novel:
        all_preds = [p for p in all_preds if not p.get("is_known_indication", False)]
        print(f"  After novel filter: {len(all_preds)}")

    # Sort by composite quality score (higher = more promising)
    all_preds.sort(key=lambda p: p.get("composite_quality_score", 0), reverse=True)

    # Top N
    if top_n and top_n < len(all_preds):
        all_preds = all_preds[:top_n]
        print(f"  After top-{top_n} filter: {len(all_preds)}")

    return all_preds


def print_summary(miner: LiteratureMiner) -> None:
    """Print summary statistics."""
    stats = miner.summary_stats()
    print("\n" + "=" * 60)
    print("LITERATURE MINING CACHE SUMMARY")
    print("=" * 60)
    print(f"Total cached entries: {stats['total']}")
    print(f"Mean evidence score: {stats['mean_score']}")
    print(f"Adverse effects detected: {stats['adverse_effects_detected']}")
    print(f"With LLM classification: {stats['with_llm_classification']}")

    if stats.get("by_level"):
        print("\nBy evidence level:")
        for level, count in sorted(stats["by_level"].items(),
                                    key=lambda x: x[1], reverse=True):
            pct = 100 * count / stats["total"]
            print(f"  {level:<20} {count:>5} ({pct:.1f}%)")


def generate_report(results: list, output_path: Path) -> None:
    """Generate a summary report of mining results."""
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_mined": len(results),
        "by_level": {},
        "adverse_effects": [],
        "strong_evidence": [],
        "gt_gap_candidates": [],
    }

    for r in results:
        level = r.evidence_level
        report["by_level"][level] = report["by_level"].get(level, 0) + 1

        if r.evidence_level == "ADVERSE_EFFECT":
            report["adverse_effects"].append({
                "drug": r.drug_name,
                "disease": r.disease_name,
                "score": r.evidence_score,
                "llm_summary": r.llm_summary,
            })

        if r.evidence_level == "STRONG_EVIDENCE":
            report["strong_evidence"].append({
                "drug": r.drug_name,
                "disease": r.disease_name,
                "score": r.evidence_score,
                "trials": r.trial_count,
                "phases": r.trial_phases,
                "pubs": r.pubmed_total,
                "llm_summary": r.llm_summary,
            })

    # GT gap candidates: strong/moderate evidence but not known indication
    for r in results:
        if r.evidence_level in ("STRONG_EVIDENCE", "MODERATE_EVIDENCE"):
            report["gt_gap_candidates"].append({
                "drug": r.drug_name,
                "disease": r.disease_name,
                "evidence_level": r.evidence_level,
                "score": r.evidence_score,
                "trials": r.trial_count,
                "pubs": r.pubmed_total,
            })

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {output_path}")

    # Print highlights
    print("\n" + "=" * 60)
    print("MINING RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total mined: {report['total_mined']}")
    for level, count in sorted(report["by_level"].items(),
                                key=lambda x: x[1], reverse=True):
        pct = 100 * count / report["total_mined"] if report["total_mined"] else 0
        print(f"  {level:<20} {count:>5} ({pct:.1f}%)")

    if report["adverse_effects"]:
        print(f"\nAdverse effects detected: {len(report['adverse_effects'])}")
        for ae in report["adverse_effects"][:10]:
            print(f"  {ae['drug']} → {ae['disease']}: {ae['llm_summary']}")

    if report["strong_evidence"]:
        print(f"\nStrong evidence (potential GT gaps): {len(report['strong_evidence'])}")
        for se in report["strong_evidence"][:10]:
            phases_str = ", ".join(se["phases"]) if se["phases"] else "N/A"
            print(f"  {se['drug']} → {se['disease']}: "
                  f"{se['trials']} trials ({phases_str}), {se['pubs']} pubs")

    print(f"\nTotal GT gap candidates: {len(report['gt_gap_candidates'])}")


def main():
    parser = argparse.ArgumentParser(
        description="Run literature mining on drug repurposing predictions")
    parser.add_argument("--tier", type=str, default=None,
                        help="Filter by confidence tier (GOLDEN,HIGH,MEDIUM,LOW)")
    parser.add_argument("--status", type=str, default=None,
                        help="Filter by literature status (NOVEL,LIKELY_GT_GAP,etc)")
    parser.add_argument("--top", type=int, default=None,
                        help="Top N predictions by composite score")
    parser.add_argument("--all-novel", action="store_true",
                        help="Only mine novel (non-GT) predictions")
    parser.add_argument("--use-llm", action="store_true",
                        help="Use Claude Haiku for abstract classification")
    parser.add_argument("--resume", action="store_true",
                        help="Resume interrupted mining (skip cached)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Re-query even if cached")
    parser.add_argument("--summary", action="store_true",
                        help="Show summary of cached results")
    parser.add_argument("--save-every", type=int, default=25,
                        help="Save cache every N pairs (default: 25)")

    args = parser.parse_args()

    # Initialize miner
    miner = LiteratureMiner(use_llm=args.use_llm)

    # Summary mode
    if args.summary:
        print_summary(miner)
        return

    # Load and filter predictions
    if not args.tier and not args.all_novel and not args.top:
        print("ERROR: Specify --tier, --all-novel, or --top to select predictions.")
        print("Use --summary to view cached results.")
        parser.print_help()
        sys.exit(1)

    preds = load_predictions(
        tier=args.tier,
        status=args.status,
        top_n=args.top,
        all_novel=args.all_novel,
    )

    if not preds:
        print("No predictions match the specified filters.")
        return

    # Convert to pairs format
    pairs = [{"drug_name": p["drug_name"], "disease_name": p["disease_name"]}
             for p in preds]

    # Remove duplicates (same drug-disease pair can appear in deliverable)
    seen = set()
    unique_pairs = []
    for pair in pairs:
        key = f"{pair['drug_name'].lower()}|{pair['disease_name'].lower()}"
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)
    print(f"\nUnique pairs to mine: {len(unique_pairs)}")

    # Estimate time
    cached_count = sum(1 for p in unique_pairs
                       if LiteratureMiner._cache_key(p["drug_name"], p["disease_name"])
                       in miner.cache)
    fresh_count = len(unique_pairs) - cached_count
    # ~2 sec per fresh query (PubMed rate limit + ClinicalTrials)
    est_time = fresh_count * 2
    print(f"Cached: {cached_count}, Fresh queries needed: {fresh_count}")
    if est_time > 60:
        print(f"Estimated time: {est_time // 60}m {est_time % 60}s")
    elif fresh_count > 0:
        print(f"Estimated time: {est_time}s")

    # Mine
    start = time.time()
    results = miner.mine_batch(
        unique_pairs,
        save_every=args.save_every,
        force_refresh=args.force_refresh,
    )
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.0f}s")

    # Generate report
    report_name = f"literature_mining_report"
    if args.tier:
        report_name += f"_{args.tier.lower()}"
    if args.status:
        report_name += f"_{args.status.lower()}"
    report_path = Path(__file__).parent.parent / "data" / "validation" / f"{report_name}.json"
    generate_report(results, report_path)

    # Print overall summary
    print_summary(miner)


if __name__ == "__main__":
    main()
