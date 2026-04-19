#!/usr/bin/env python3
"""h995: Autoimmune biologic family-mis-selection audit.

h991 pooled 5-seed diagnostic found autoimmune bio_p30|bio = 13.9% vs
sm_p30|sm = 39.1% (-25.2pp). In a category where biologics ARE the gold
standard, a 14% hit rate on biologic slots means the kNN is surfacing the
WRONG biologic family (anti-CD20 instead of anti-TNF, oncology mAbs
instead of immunology mAbs, etc.).

This script measures, for each autoimmune holdout disease and each of the
~3-4 biologic slots in top-30:
    - hit (in expanded GT)
    - SuffixMatch: drug's USAN suffix matches any suffix in the disease's
      biologic GT
    - TargetMatch: drug's target-gene set intersects the union of target
      sets of drugs in the disease's biologic GT

Aggregate:
    hit_rate(match=True)  vs  hit_rate(match=False)

Decision rule (h995 step 3):
    If hit_rate_match / hit_rate_mismatch >= 3x AND n >= 30 in each bucket,
    propose a family-match filter (h995 step 4 ship test).
    Else invalidate h995 as an unlikely filter.

Also measures the same per-category globally to sanity-check whether the
pattern is autoimmune-specific or generic.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from production_predictor import DrugRepurposingPredictor  # noqa: E402
from h393_holdout_tier_validation import (  # noqa: E402
    split_diseases,
    recompute_gt_structures,
    restore_gt_structures,
)
from h939_biologic_target_overlap_audit import is_biologic  # noqa: E402


SEEDS = [42, 123, 456, 789, 2024]
TOP_N = 30


# USAN suffix order matters — longer, more specific suffixes first so that
# 'kinra' wins over 'mab'/'ase' on anakinra, 'plase' over 'ase', etc.
SUFFIX_ORDER = (
    "kinra", "cept", "parin", "ferax", "plase", "streptim",
    "genase", "hyase", "nase", "pase", "lase",
    "tropin", "relin", "tide", "mab",
)


_TRAILING_QUALIFIERS = (
    " pegol", " sodium", " alfa", " beta", " gamma", " delta",
    " acetate", " calcium", " disodium", " chloride", "-alfa",
    "-beta", "-1a", "-1b", "-2a", "-2b",
)


def _strip_trailing_qualifiers(nm: str) -> str:
    changed = True
    while changed:
        changed = False
        for q in _TRAILING_QUALIFIERS:
            if nm.endswith(q):
                nm = nm[: -len(q)].rstrip()
                changed = True
    return nm


def usan_suffix(drug_name: str | None) -> str | None:
    """Return the best-matching USAN suffix for a biologic name, or None."""
    if not drug_name:
        return None
    nm = _strip_trailing_qualifiers(drug_name.lower().strip())
    for suf in SUFFIX_ORDER:
        if nm.endswith(suf):
            return suf
    # Keyword-based fallback families (insulin/interferon/heparin/vaccine/…)
    for kw in (
        "insulin", "interferon", "erythropoietin", "darbepoetin",
        "heparin", "filgrastim", "globulin", "immunoglobulin",
        "antibody", "vaccine", "factor", "von willebrand",
        "botulinum", "glucagon", "somatropin", "somatostatin",
        "hyaluronidase", "chymotrypsin", "collagenase", "trypsin",
        "pancrelipase", "pancreatin", "interleukin", "streptokinase",
        "urokinase", "alteplase",
    ):
        if kw in nm:
            return kw
    return None


def usan_substem(drug_name: str | None) -> str | None:
    """For -mab drugs, return the USAN substem that encodes target family.
    Pattern: <prefix><target><source>mab where target is one of:
      li(m)(u) = immunomodulator
      tu(m)(u) = tumor
      ci(r)(u) = cardiovascular
      ki(n)(u) = interleukin
      vi(r)(u) = viral
      bac(u) = bacterial
      fu(n)(g)(u) = fungal
      neu(r) = neural
      tox = toxin

    Returns 3-letter target code (e.g. 'lim', 'tum', 'kin') or None.
    Only meaningful for -mab.
    """
    if not drug_name:
        return None
    nm = _strip_trailing_qualifiers(drug_name.lower().strip())
    if not nm.endswith("mab"):
        return None
    # Strip mab + source letter (u/o/xi/zu). Use regex.
    m = re.match(r"^(.*?)(xi|zu|o|u)(mab)$", nm)
    if not m:
        return None
    stem_no_src = m.group(1)
    # Canonical USAN 2-letter target substems
    TWO = ("li", "tu", "ki", "vi", "ba", "fu", "ne", "ci", "to", "le")
    if len(stem_no_src) >= 2 and stem_no_src[-2:] in TWO:
        return stem_no_src[-2:]
    # Fallback: last 3 chars as a fingerprint
    return stem_no_src[-3:] if len(stem_no_src) >= 3 else stem_no_src


def load_expanded_gt(path: Path) -> Dict[str, Set[str]]:
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, Set[str]] = {}
    for dis_id, drugs in raw.items():
        s: Set[str] = set()
        for d in drugs:
            if isinstance(d, str):
                s.add(d)
            elif isinstance(d, dict):
                did = d.get("drug_id") or d.get("drug")
                if did:
                    s.add(did)
        out[dis_id] = s
    return out


def evaluate_seed(
    predictor: DrugRepurposingPredictor,
    expanded_gt: Dict[str, Set[str]],
    biologic_pool: Set[str],
    holdout_ids: List[str],
) -> List[Dict]:
    """Return list of per-slot records for biologic slots only."""
    slot_records: List[Dict] = []

    for dis_id in holdout_ids:
        gt = expanded_gt.get(dis_id, set())
        if not gt:
            continue
        bio_gt = gt & biologic_pool
        # We still include diseases with 0 bio_gt — needed to
        # evaluate "mismatch" as well; but with no bio_gt, suffix
        # & target sets are empty so every candidate is mismatch.

        try:
            result = predictor.predict(dis_id, top_n=TOP_N,
                                       include_filtered=True)
        except Exception:
            continue
        preds = result.predictions
        if not preds:
            continue

        name = predictor.disease_names.get(dis_id, dis_id)
        cat = predictor.categorize_disease(name)

        # Build reference sets from bio_gt
        gt_suffixes: Set[str] = set()
        gt_substems: Set[str] = set()
        gt_target_union: Set[str] = set()
        for g in bio_gt:
            suf = usan_suffix(predictor.drug_id_to_name.get(g))
            if suf:
                gt_suffixes.add(suf)
            sub = usan_substem(predictor.drug_id_to_name.get(g))
            if sub:
                gt_substems.add(sub)
            tgts = predictor.drug_targets.get(g)
            if tgts:
                gt_target_union |= tgts

        for p in preds[:TOP_N]:
            if p.drug_id not in biologic_pool:
                continue
            drug_name = predictor.drug_id_to_name.get(p.drug_id, "")
            suf = usan_suffix(drug_name)
            sub = usan_substem(drug_name)
            tgts = predictor.drug_targets.get(p.drug_id, set())

            suffix_match = (suf is not None and suf in gt_suffixes)
            substem_match = (sub is not None and sub in gt_substems)
            target_match = bool(tgts and gt_target_union and
                                (tgts & gt_target_union))

            slot_records.append({
                "disease_id": dis_id,
                "category": cat,
                "bio_gt_size": len(bio_gt),
                "drug_id": p.drug_id,
                "drug_name": drug_name,
                "suffix": suf,
                "substem": sub,
                "n_targets": len(tgts),
                "hit": bool(p.drug_id in gt),
                "suffix_match": bool(suffix_match),
                "substem_match": bool(substem_match),
                "target_match": bool(target_match),
                "rank": p.rank,
            })

    return slot_records


def summarize(records: List[Dict], rule_key: str) -> Dict:
    match = [r for r in records if r[rule_key]]
    miss = [r for r in records if not r[rule_key]]
    n_match = len(match)
    n_miss = len(miss)
    h_match = sum(1 for r in match if r["hit"])
    h_miss = sum(1 for r in miss if r["hit"])
    hr_match = (h_match / n_match) if n_match else 0.0
    hr_miss = (h_miss / n_miss) if n_miss else 0.0
    ratio = hr_match / hr_miss if hr_miss > 0 else float("inf")
    return {
        "rule": rule_key,
        "n_match": n_match,
        "n_miss": n_miss,
        "hits_match": h_match,
        "hits_miss": h_miss,
        "hit_rate_match": hr_match,
        "hit_rate_miss": hr_miss,
        "ratio": ratio if ratio != float("inf") else None,
    }


def main():
    print("=" * 78)
    print("h995: Autoimmune biologic family-mis-selection audit")
    print("=" * 78)

    predictor = DrugRepurposingPredictor()
    expanded_gt = load_expanded_gt(
        predictor.reference_dir / "expanded_ground_truth.json"
    )
    print(f"Expanded GT: {len(expanded_gt)} diseases")

    biologic_pool = {
        d for d in predictor.drug_id_to_name
        if is_biologic(predictor.drug_id_to_name.get(d))
    }
    print(f"Biologic pool: {len(biologic_pool)} of "
          f"{len(predictor.drug_id_to_name)} drugs")

    all_diseases = [
        d for d in predictor.ground_truth if d in predictor.embeddings
    ]
    print(f"Evaluable diseases (GT ∩ embeddings): {len(all_diseases)}")
    print(f"Seeds: {SEEDS}\n")

    all_records: List[Dict] = []

    for seed in SEEDS:
        print(f"--- Seed {seed} ---")
        train_ids, holdout_ids = split_diseases(all_diseases, seed)
        train_set = set(train_ids)
        originals = recompute_gt_structures(predictor, train_set)
        try:
            recs = evaluate_seed(predictor, expanded_gt,
                                 biologic_pool, holdout_ids)
        finally:
            restore_gt_structures(predictor, originals)
        all_records.extend({**r, "seed": seed} for r in recs)

        # autoimmune quick scalar
        ai = [r for r in recs if r["category"] == "autoimmune"]
        print(f"  n_holdout={len(holdout_ids)} "
              f"n_biologic_slots={len(recs)} "
              f"autoimmune_biologic_slots={len(ai)} "
              f"autoimmune_bio_hits={sum(1 for r in ai if r['hit'])}")

    # Autoimmune subset
    ai_records = [r for r in all_records if r["category"] == "autoimmune"]
    print(f"\nAutoimmune pooled biologic slots: {len(ai_records)} "
          f"(5 seeds)")
    print(f"Autoimmune slot hit rate: "
          f"{100 * sum(1 for r in ai_records if r['hit']) / max(len(ai_records), 1):.2f}% "
          f"({sum(1 for r in ai_records if r['hit'])} / {len(ai_records)})")

    # Summaries by rule
    print("\n" + "=" * 78)
    print("AUTOIMMUNE — hit rate by family-match rule")
    print("=" * 78)
    ai_summaries = {}
    for rule in ("suffix_match", "substem_match", "target_match"):
        s = summarize(ai_records, rule)
        ai_summaries[rule] = s
        ratio_s = f"{s['ratio']:.2f}x" if s["ratio"] else "inf"
        print(f"  [{rule}] match n={s['n_match']} hits={s['hits_match']} "
              f"({100*s['hit_rate_match']:.2f}%)  |  "
              f"miss n={s['n_miss']} hits={s['hits_miss']} "
              f"({100*s['hit_rate_miss']:.2f}%)  "
              f"ratio={ratio_s}")

    # Decision
    print("\n" + "=" * 78)
    print("DECISION (autoimmune)")
    print("=" * 78)
    decisive = []
    for rule, s in ai_summaries.items():
        if (s["n_match"] >= 30 and s["n_miss"] >= 30
                and s["ratio"] is not None and s["ratio"] >= 3.0):
            decisive.append(rule)
    if decisive:
        print(f"PROCEED: rules {decisive} meet ≥3x ratio AND n≥30 each bucket.")
        print("Next step: implement filter + 5-seed ship test "
              "(bio_p30|bio lift ≥5pp on autoimmune, bio_r30 drop ≤2pp).")
    else:
        print("INVALIDATE (diagnostic): no rule meets ≥3x ratio AND n≥30 gate "
              "on autoimmune. Biologic family-match is not a usable filter.")

    # Per-category context (target_match only — most semantically meaningful)
    print("\n" + "=" * 78)
    print("PER CATEGORY — target_match hit-rate match vs miss (pooled 5 seeds)")
    print("=" * 78)
    cats: Dict[str, List[Dict]] = defaultdict(list)
    for r in all_records:
        cats[r["category"]].append(r)
    print(f"{'category':<18s} {'slots':>6s} {'match_n':>8s} "
          f"{'hr_match':>9s} {'miss_n':>7s} {'hr_miss':>8s} {'ratio':>7s}")
    per_cat = {}
    for cat in sorted(cats, key=lambda c: len(cats[c]), reverse=True):
        s = summarize(cats[cat], "target_match")
        per_cat[cat] = s
        ratio_s = f"{s['ratio']:.2f}x" if s["ratio"] else "inf"
        print(f"{cat:<18s} {len(cats[cat]):>6d} {s['n_match']:>8d} "
              f"{100*s['hit_rate_match']:>7.2f}% {s['n_miss']:>7d} "
              f"{100*s['hit_rate_miss']:>6.2f}% {ratio_s:>7s}")

    # Top-15 most-frequent autoimmune biologic mispicks (hit=0, target_match=0)
    print("\n" + "=" * 78)
    print("TOP-15 mispicked autoimmune biologics (hit=0 AND target_match=0)")
    print("=" * 78)
    mispick = defaultdict(int)
    for r in ai_records:
        if not r["hit"] and not r["target_match"]:
            mispick[(r["drug_name"], r["suffix"])] += 1
    for (nm, suf), n in sorted(mispick.items(),
                               key=lambda x: -x[1])[:15]:
        print(f"  {n:>4d}  {nm:<30s} suffix={suf}")

    # Top-15 missing bio hits (bio_gt with target_match=0 in top-30)
    # i.e. what biologic hits DID slip through — inverse audit
    print("\n" + "=" * 78)
    print("TOP-15 autoimmune biologic HITS by family-match type")
    print("=" * 78)
    hit_classify = defaultdict(int)
    for r in ai_records:
        if r["hit"]:
            hit_classify[(r["drug_name"], r["target_match"],
                         r["suffix_match"])] += 1
    for (nm, tm, sm), n in sorted(hit_classify.items(),
                                  key=lambda x: -x[1])[:15]:
        print(f"  {n:>4d}  {nm:<30s} target_match={tm} suffix_match={sm}")

    # Save
    out_path = ROOT / "data/analysis/h995_autoimmune_family_audit.json"
    payload = {
        "hypothesis": "h995",
        "title": "Autoimmune biologic family-mis-selection audit",
        "seeds": SEEDS,
        "top_n": TOP_N,
        "n_biologic_pool": len(biologic_pool),
        "n_autoimmune_slots_pooled": len(ai_records),
        "autoimmune_hit_rate": (
            sum(1 for r in ai_records if r["hit"])
            / max(len(ai_records), 1)
        ),
        "autoimmune_by_rule": ai_summaries,
        "decisive_rules_on_autoimmune": decisive,
        "per_category_target_match": per_cat,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nSaved -> {out_path}")

    # Also save per-slot CSV-style records for drill-down
    recs_path = ROOT / "data/analysis/h995_slot_records.json"
    with open(recs_path, "w") as f:
        json.dump(all_records, f, indent=2, default=str)
    print(f"Saved -> {recs_path}")


if __name__ == "__main__":
    main()
