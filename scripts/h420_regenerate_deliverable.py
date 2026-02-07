#!/usr/bin/env python3
"""
h420: Regenerate production deliverable with current tier rules.

The existing deliverable has 58% stale categories (h349).
After h395, h396, h398, h399, h415 modified tier rules,
the deliverable needs regeneration.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.production_predictor import DrugRepurposingPredictor, ConfidenceTier, classify_literature_status

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False
    print("WARNING: openpyxl not installed, will save as CSV instead")


def load_gene_overlap_data() -> tuple:
    """Load drug target and disease gene data for overlap computation.

    Returns (drug_targets, disease_genes) dicts.
    """
    base = Path(__file__).parent.parent / "data" / "reference"
    drug_targets: Dict[str, list] = {}
    disease_genes: Dict[str, list] = {}

    dt_path = base / "drug_targets.json"
    if dt_path.exists():
        with open(dt_path) as f:
            drug_targets = json.load(f)
        print(f"Loaded drug targets: {len(drug_targets)} drugs")

    dg_path = base / "disease_genes.json"
    if dg_path.exists():
        with open(dg_path) as f:
            disease_genes = json.load(f)
        print(f"Loaded disease genes: {len(disease_genes)} diseases")

    return drug_targets, disease_genes


def compute_gene_overlap(drug_id: str, disease_id: str,
                         drug_targets: Dict[str, list],
                         disease_genes: Dict[str, list]) -> int:
    """Count shared genes between drug targets and disease-associated genes."""
    db_id = drug_id.replace('drkg:Compound::', '')
    mesh_id = disease_id.replace('drkg:Disease::', '')
    drug_genes = set(str(g) for g in drug_targets.get(db_id, []))
    dis_genes = set(str(g) for g in disease_genes.get(mesh_id, []))
    return len(drug_genes & dis_genes)


def load_self_referential_data() -> Dict[str, Dict]:
    """Load h504 self-referential analysis data.

    Returns dict of disease_id -> {self_only_pct, n_gt, therapeutic_island}.
    """
    sr_path = Path(__file__).parent.parent / "data" / "analysis" / "h504_self_referential.json"
    if not sr_path.exists():
        print(f"WARNING: {sr_path} not found, self-referential annotations will be empty")
        return {}

    with open(sr_path) as f:
        sr_data = json.load(f)

    result: Dict[str, Dict] = {}
    for entry in sr_data:
        disease_id = entry["disease_id"]
        n_gt = entry["n_gt"]
        self_only_pct = entry["self_only_pct"]
        # h517: Therapeutic island = GT>5 and 100% self-referential
        therapeutic_island = n_gt > 5 and self_only_pct == 100.0
        result[disease_id] = {
            "self_referential_pct": self_only_pct,
            "therapeutic_island": therapeutic_island,
        }

    n_islands = sum(1 for v in result.values() if v["therapeutic_island"])
    print(f"Loaded self-referential data: {len(result)} diseases, {n_islands} therapeutic islands")
    return result


def main():
    start = time.time()
    print("=" * 70)
    print("h420: Regenerate Production Deliverable")
    print("=" * 70)

    predictor = DrugRepurposingPredictor()
    gt_data = predictor.ground_truth

    # h517: Load self-referential annotations
    self_ref_data = load_self_referential_data()

    # h546: Load gene overlap data
    drug_targets, disease_genes_data = load_gene_overlap_data()

    # h570: Load per-disease holdout precision
    disease_holdout_precision: Dict[str, float] = {}
    dhp_path = Path(__file__).parent.parent / "data" / "reference" / "disease_holdout_precision.json"
    if dhp_path.exists():
        with open(dhp_path) as f:
            dhp_data = json.load(f)
        for did, info in dhp_data.items():
            if info.get("holdout_precision") is not None:
                disease_holdout_precision[did] = info["holdout_precision"]
        print(f"Loaded disease holdout precision: {len(disease_holdout_precision)} diseases")

    # h616: Load expanded GT for completeness ratio
    gt_completeness: Dict[str, float] = {}
    exp_gt_path = Path(__file__).parent.parent / "data" / "reference" / "expanded_ground_truth.json"
    if exp_gt_path.exists():
        with open(exp_gt_path) as f:
            exp_gt_data = json.load(f)
        for did in predictor.ground_truth:
            int_count = len(predictor.ground_truth[did])
            exp_drugs = exp_gt_data.get(did, [])
            exp_count = len(exp_drugs) if isinstance(exp_drugs, list) else 0
            if int_count > 0:
                gt_completeness[did] = round(exp_count / int_count, 1)
        print(f"Computed GT completeness: {len(gt_completeness)} diseases")

    # Literature mining evidence cache
    lit_cache: Dict[str, dict] = {}
    lit_cache_path = Path(__file__).parent.parent / "data" / "validation" / "literature_mining_cache.json"
    if lit_cache_path.exists():
        with open(lit_cache_path) as f:
            lit_cache = json.load(f)
        print(f"Loaded literature mining cache: {len(lit_cache)} entries")

    # Get all diseases with embeddings
    all_diseases = [d for d in predictor.embeddings if d in predictor.disease_names]
    print(f"Diseases with embeddings: {len(all_diseases)}")

    # Generate predictions for all diseases
    all_predictions = []
    tier_counts = defaultdict(int)
    category_counts = defaultdict(int)

    for idx, disease_id in enumerate(all_diseases):
        disease_name = predictor.disease_names.get(disease_id, disease_id)

        try:
            result = predictor.predict(disease_name, top_n=30, include_filtered=True)
        except Exception as e:
            continue

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - start
            print(f"  Processed {idx+1}/{len(all_diseases)} diseases... ({elapsed:.0f}s)")

        for pred in result.predictions:
            is_gt = disease_id in gt_data and pred.drug_id in gt_data[disease_id]

            # h481: Classify literature status
            lit_status, soc_class = classify_literature_status(
                pred.drug_name, disease_name, result.category, is_gt
            )

            # h517: Self-referential annotations
            sr_info = self_ref_data.get(disease_id, {})
            self_ref_pct = sr_info.get("self_referential_pct", "")
            therapeutic_island = sr_info.get("therapeutic_island", False)

            # h546: Gene overlap annotation
            gene_overlap = compute_gene_overlap(
                pred.drug_id, disease_id, drug_targets, disease_genes_data
            )

            all_predictions.append({
                'disease_name': disease_name,
                'disease_id': disease_id,
                'drug_name': pred.drug_name,
                'drug_id': pred.drug_id,
                'rank': pred.rank,
                'knn_score': round(pred.knn_score, 6),
                'normalized_score': round(pred.norm_score, 4),
                'confidence_tier': pred.confidence_tier.value,
                'tier_rule': pred.category_specific_tier or 'standard',
                'category': result.category,
                'disease_tier': result.disease_tier,
                'train_frequency': pred.train_frequency,
                'mechanism_support': pred.mechanism_support,
                'has_targets': pred.has_targets,
                'is_known_indication': is_gt,
                'rescue_applied': pred.category_rescue_applied,
                'transe_consilience': pred.transe_consilience,
                'rank_bucket_precision': pred.rank_bucket_precision,
                'category_holdout_precision': pred.category_holdout_precision,
                'literature_status': lit_status,
                'soc_drug_class': soc_class or '',
                'self_referential_pct': self_ref_pct,
                'therapeutic_island': therapeutic_island,
                'gene_overlap_count': gene_overlap,
                'disease_holdout_precision': disease_holdout_precision.get(disease_id, ''),
                'gt_completeness_ratio': gt_completeness.get(disease_id, ''),
            })

            # Literature mining evidence (from automated cache)
            lit_key = f"{pred.drug_name.lower()}|{disease_name.lower()}"
            lit_entry = lit_cache.get(lit_key, {})
            all_predictions[-1]['literature_evidence_level'] = lit_entry.get('evidence_level', 'NOT_ASSESSED')
            all_predictions[-1]['literature_evidence_score'] = lit_entry.get('evidence_score', 0.0)

            # h592: Compute composite quality score for experiment prioritization
            p = all_predictions[-1]
            rank_val = p['rank']
            rank_score = max(0, (20 - rank_val) / 19) if rank_val <= 20 else 0
            ns = p['normalized_score'] if p['normalized_score'] else 0
            transe_val = 1.0 if p['transe_consilience'] else 0.0
            go_val = 1.0 if (p.get('gene_overlap_count') or 0) > 0 else 0.0
            mech_val = 1.0 if p['mechanism_support'] else 0.0
            dhp_raw = p.get('disease_holdout_precision')
            dq_val = float(dhp_raw) / 100 if dhp_raw and dhp_raw != '' else 0.0
            sr_raw = p.get('self_referential_pct')
            nsr_val = 1.0 - (float(sr_raw) / 100 if sr_raw and sr_raw != '' else 0.5)
            p['composite_quality_score'] = round(
                rank_score * 1.5 + ns * 1.0 + transe_val * 1.0 +
                go_val * 1.0 + mech_val * 0.5 + dq_val * 1.0 + nsr_val * 0.5,
                2
            )

            # h631: MEDIUM quality quartile based on TransE, mechanism, rank
            # h629 holdout results (5-seed, expanded GT):
            #   Q1 (TransE+mechanism OR TransE+rank<=5): 60-72% holdout
            #   Q2 (TransE OR mechanism+rank<=10): 50-57% holdout
            #   Q3 (mechanism OR rank<=5): 44-54% holdout
            #   Q4 (none of the above): 31% holdout
            if p['confidence_tier'] == 'MEDIUM':
                has_transe = bool(p.get('transe_consilience'))
                has_mech = bool(p.get('mechanism_support'))
                rank_val_q = p.get('rank', 99)

                if has_transe and (has_mech or rank_val_q <= 5):
                    p['medium_quality'] = 'Q1'  # 60-72% holdout
                elif has_transe or (has_mech and rank_val_q <= 10):
                    p['medium_quality'] = 'Q2'  # 50-57% holdout
                elif has_mech or rank_val_q <= 5:
                    p['medium_quality'] = 'Q3'  # 44-54% holdout
                else:
                    p['medium_quality'] = 'Q4'  # ~31% holdout
            else:
                p['medium_quality'] = ''

            tier_counts[pred.confidence_tier.value] += 1
            category_counts[result.category] += 1

    elapsed = time.time() - start
    print(f"\nGenerated {len(all_predictions)} predictions in {elapsed:.0f}s")

    # Print tier distribution
    print(f"\n{'Tier':<10} {'Count':>8} {'Pct':>7}")
    print("-" * 30)
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        count = tier_counts[tier]
        pct = 100 * count / len(all_predictions) if all_predictions else 0
        print(f"{tier:<10} {count:>8} {pct:>6.1f}%")

    print(f"\nTop categories:")
    for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = 100 * count / len(all_predictions)
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Save to file
    output_dir = Path(__file__).parent.parent / "data" / "deliverables"
    output_dir.mkdir(parents=True, exist_ok=True)

    if HAS_OPENPYXL:
        output_path = output_dir / "drug_repurposing_predictions_with_confidence.xlsx"
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "Predictions"

        # Header
        headers = list(all_predictions[0].keys())
        ws.append(headers)

        # Data
        for pred in all_predictions:
            ws.append([pred[h] for h in headers])

        # Format
        for col in ws.columns:
            max_length = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_length + 2, 40)

        wb.save(str(output_path))
        print(f"\nSaved to {output_path}")

        # Also save JSON for programmatic use
        def _json_safe(obj: object) -> object:
            """Convert numpy types to Python native for JSON serialization."""
            import numpy as np
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj

        json_preds = [
            {k: _json_safe(v) for k, v in p.items()} for p in all_predictions
        ]
        json_path = output_dir / "drug_repurposing_predictions_with_confidence.json"
        with open(json_path, 'w') as jf:
            json.dump(json_preds, jf, indent=2)
        print(f"Saved JSON to {json_path}")
    else:
        output_path = output_dir / "drug_repurposing_predictions_with_confidence.csv"
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_predictions[0].keys())
            writer.writeheader()
            writer.writerows(all_predictions)
        print(f"\nSaved to {output_path}")

    # Compare with old deliverable if it exists
    old_path = output_dir / "drug_repurposing_predictions_with_confidence.xlsx"
    if HAS_OPENPYXL and old_path.exists() and old_path != output_path:
        print("\nNote: Old deliverable exists but comparison skipped (different format)")

    total_time = time.time() - start
    print(f"\nTotal time: {total_time:.0f}s")
    print(f"Total predictions: {len(all_predictions)}")
    print(f"Unique diseases: {len(set(p['disease_id'] for p in all_predictions))}")
    print(f"Unique drugs: {len(set(p['drug_id'] for p in all_predictions))}")

    # Summary statistics
    gt_preds = [p for p in all_predictions if p['is_known_indication']]
    print(f"\nKnown indications in predictions: {len(gt_preds)}")
    for tier in ['GOLDEN', 'HIGH', 'MEDIUM', 'LOW', 'FILTER']:
        tier_preds = [p for p in all_predictions if p['confidence_tier'] == tier]
        tier_gt = [p for p in tier_preds if p['is_known_indication']]
        prec = 100 * len(tier_gt) / len(tier_preds) if tier_preds else 0
        print(f"  {tier}: {len(tier_gt)}/{len(tier_preds)} = {prec:.1f}%")


if __name__ == '__main__':
    main()
