#!/usr/bin/env python3
"""
h376: Ensemble Coverage Analysis - Which Diseases Benefit from Ensemble?

Analyzes disease-level characteristics that predict ensemble benefit.
Based on h369-h372 findings about MinRank ensemble.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import DrugRepurposingPredictor
from disease_categorizer import categorize_disease
from sklearn.metrics.pairwise import cosine_similarity


def compute_knn_scores(predictor: DrugRepurposingPredictor, disease_id: str, k: int = 20, leave_one_out: bool = True) -> Dict[str, float]:
    """Compute kNN drug scores for a disease (replicates predict() logic).

    Args:
        leave_one_out: If True, exclude the disease itself from neighbors (proper evaluation)
    """
    if disease_id not in predictor.embeddings:
        return {}

    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]

    # For leave-one-out: exclude the disease itself
    if leave_one_out and disease_id in predictor.train_diseases:
        self_idx = predictor.train_diseases.index(disease_id)
        sims[self_idx] = -1  # Ensure it's not selected as neighbor

    # Get k+1 neighbors to account for potential self-exclusion
    top_k_idx = np.argsort(sims)[-(k+1):]

    drug_scores: Dict[str, float] = defaultdict(float)
    count = 0
    for idx in reversed(top_k_idx):  # Highest similarity first
        neighbor_disease = predictor.train_diseases[idx]
        if leave_one_out and neighbor_disease == disease_id:
            continue
        if count >= k:
            break

        neighbor_sim = sims[idx]
        for drug_id in predictor.ground_truth[neighbor_disease]:
            if drug_id in predictor.embeddings:
                drug_scores[drug_id] += neighbor_sim
        count += 1

    return dict(drug_scores)


def minrank_ensemble(knn_scores: Dict[str, float], target_scores: Dict[str, int]) -> Dict[str, Tuple[int, float]]:
    """Apply MinRank ensemble: return min rank and combined score for each drug."""
    all_drugs = set(knn_scores.keys()) | set(target_scores.keys())

    if not all_drugs:
        return {}

    # Convert to ranks
    knn_ranked = sorted(knn_scores.items(), key=lambda x: -x[1])
    knn_ranks = {drug: i + 1 for i, (drug, _) in enumerate(knn_ranked)}

    target_ranked = sorted(target_scores.items(), key=lambda x: -x[1])
    target_ranks = {drug: i + 1 for i, (drug, _) in enumerate(target_ranked)}

    # MinRank fusion
    result = {}
    max_rank = len(all_drugs)
    for drug in all_drugs:
        knn_r = knn_ranks.get(drug, max_rank)
        target_r = target_ranks.get(drug, max_rank)
        min_r = min(knn_r, target_r)
        combined_score = knn_scores.get(drug, 0) + target_scores.get(drug, 0)
        result[drug] = (min_r, combined_score)

    return result


def evaluate_methods(
    predictor: DrugRepurposingPredictor,
    disease_id: str,
    gt_drugs: List[str]
) -> Dict[str, bool]:
    """Evaluate kNN, Target, and MinRank methods for a disease. Return hit@30 status."""
    # kNN scores
    knn_scores = compute_knn_scores(predictor, disease_id)

    # Target scores from predictor
    target_scores = predictor._get_target_scores(disease_id)

    if not knn_scores:
        return {
            "knn_hit": False,
            "target_hit": False,
            "ensemble_hit": False,
            "knn_only": False,
            "target_only": False,
            "ensemble_rescues": False,
            "ensemble_hurts": False,
            "error": "no_knn_scores"
        }

    # kNN ranking
    knn_ranked = sorted(knn_scores.items(), key=lambda x: -x[1])[:30]
    knn_top30 = {d for d, _ in knn_ranked}

    # Target ranking
    target_ranked = sorted(target_scores.items(), key=lambda x: -x[1])[:30]
    target_top30 = {d for d, _ in target_ranked}

    # MinRank ensemble
    ensemble = minrank_ensemble(knn_scores, target_scores)
    ensemble_ranked = sorted(ensemble.items(), key=lambda x: (x[1][0], -x[1][1]))[:30]
    ensemble_top30 = {d for d, _ in ensemble_ranked}

    # Check hits
    knn_hit = any(d in knn_top30 for d in gt_drugs)
    target_hit = any(d in target_top30 for d in gt_drugs)
    ensemble_hit = any(d in ensemble_top30 for d in gt_drugs)

    return {
        "knn_hit": knn_hit,
        "target_hit": target_hit,
        "ensemble_hit": ensemble_hit,
        "knn_only": knn_hit and not target_hit,
        "target_only": target_hit and not knn_hit,
        "ensemble_rescues": ensemble_hit and not knn_hit and not target_hit,
        "ensemble_hurts": not ensemble_hit and (knn_hit or target_hit)
    }


def load_drug_atc(predictor: DrugRepurposingPredictor) -> Dict[str, str]:
    """Load drug to ATC code mapping for class diversity."""
    path = predictor.reference_dir / "drugbank_lookup.json"
    with open(path) as f:
        data = json.load(f)

    atc = {}
    for drug_id, info in data.items():
        if "atc_codes" in info and info["atc_codes"]:
            # Take first 3 chars of first ATC code as drug class
            atc[drug_id] = info["atc_codes"][0][:3]

    return atc


def analyze_diseases(n_diseases: int = 300) -> Dict:
    """Analyze ensemble benefit across diseases."""
    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()
    drug_atc = load_drug_atc(predictor)

    # Use train_diseases as our evaluation set
    # ground_truth maps disease_id -> list of drug_ids
    gt = predictor.ground_truth

    print(f"Found {len(gt)} diseases with ground truth")

    # Build evaluable diseases list
    evaluable_diseases = []
    for disease_id, drugs in gt.items():
        if disease_id not in predictor.embeddings:
            continue

        disease_name = predictor.disease_names.get(disease_id, disease_id)
        gene_count = len(predictor.disease_genes.get(disease_id, []))

        # Get drug class diversity
        drug_classes = set()
        for d in drugs:
            if d in drug_atc:
                drug_classes.add(drug_atc[d][:3])

        evaluable_diseases.append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "gt_drugs": list(drugs) if isinstance(drugs, set) else drugs,
            "gt_drug_count": len(drugs),
            "gene_count": gene_count,
            "drug_classes": len(drug_classes)
        })

    print(f"Total evaluable diseases: {len(evaluable_diseases)}")

    # Sample if too many
    if len(evaluable_diseases) > n_diseases:
        np.random.seed(42)
        indices = np.random.choice(len(evaluable_diseases), n_diseases, replace=False)
        evaluable_diseases = [evaluable_diseases[i] for i in sorted(indices)]
        print(f"Sampled to {n_diseases} diseases")

    # Evaluate each disease
    results = []
    for i, d in enumerate(evaluable_diseases):
        if (i + 1) % 50 == 0:
            print(f"Evaluating {i+1}/{len(evaluable_diseases)}...")

        try:
            eval_result = evaluate_methods(predictor, d["disease_id"], d["gt_drugs"])
        except Exception as e:
            print(f"  Error on {d['disease_id']}: {e}")
            continue

        if eval_result.get("error"):
            continue

        # Add disease features
        category = categorize_disease(d["disease_name"])
        if category is None:
            category = "other"
        result = {
            **d,
            "category": category,
            **eval_result
        }
        results.append(result)

    print(f"Evaluated {len(results)} diseases successfully")

    if not results:
        return {"error": "No results", "n_evaluated": 0}

    # Aggregate analysis
    analysis = analyze_results(results)

    return {
        "n_evaluated": len(results),
        "summary": analysis,
        "diseases": results
    }


def analyze_results(results: List[Dict]) -> Dict:
    """Analyze what predicts ensemble benefit."""
    # Overall metrics
    knn_recall = sum(1 for r in results if r["knn_hit"]) / len(results)
    target_recall = sum(1 for r in results if r["target_hit"]) / len(results)
    ensemble_recall = sum(1 for r in results if r["ensemble_hit"]) / len(results)

    # Ensemble benefit cases
    ensemble_rescues = [r for r in results if r["ensemble_rescues"]]
    ensemble_hurts = [r for r in results if r["ensemble_hurts"]]

    # Analyze by gene count quartiles
    gene_counts = sorted([r["gene_count"] for r in results])
    q1 = np.percentile(gene_counts, 25)
    q2 = np.percentile(gene_counts, 50)
    q3 = np.percentile(gene_counts, 75)

    def quartile(gc):
        if gc <= q1:
            return "Q1_low"
        elif gc <= q2:
            return "Q2"
        elif gc <= q3:
            return "Q3"
        else:
            return "Q4_high"

    gene_quartile_results = defaultdict(list)
    for r in results:
        gene_quartile_results[quartile(r["gene_count"])].append(r)

    gene_quartile_summary = {}
    for q, rs in sorted(gene_quartile_results.items()):
        gene_quartile_summary[q] = {
            "n": len(rs),
            "knn_recall": sum(1 for r in rs if r["knn_hit"]) / len(rs) if rs else 0,
            "target_recall": sum(1 for r in rs if r["target_hit"]) / len(rs) if rs else 0,
            "ensemble_recall": sum(1 for r in rs if r["ensemble_hit"]) / len(rs) if rs else 0,
            "ensemble_delta": (sum(1 for r in rs if r["ensemble_hit"]) - sum(1 for r in rs if r["knn_hit"])) / len(rs) if rs else 0
        }

    # Analyze by drug count
    drug_count_thresholds = [1, 3, 5, 10]
    drug_count_results = {}
    for threshold in drug_count_thresholds:
        subset = [r for r in results if r["gt_drug_count"] >= threshold]
        if subset:
            drug_count_results[f"ge_{threshold}_drugs"] = {
                "n": len(subset),
                "knn_recall": sum(1 for r in subset if r["knn_hit"]) / len(subset),
                "target_recall": sum(1 for r in subset if r["target_hit"]) / len(subset),
                "ensemble_recall": sum(1 for r in subset if r["ensemble_hit"]) / len(subset),
                "ensemble_delta": (sum(1 for r in subset if r["ensemble_hit"]) - sum(1 for r in subset if r["knn_hit"])) / len(subset)
            }

    # Analyze by category
    category_results = defaultdict(list)
    for r in results:
        category_results[r["category"]].append(r)

    category_summary = {}
    for cat, rs in sorted(category_results.items()):
        if len(rs) >= 3:  # Only report categories with 3+ diseases
            category_summary[cat] = {
                "n": len(rs),
                "knn_recall": sum(1 for r in rs if r["knn_hit"]) / len(rs) if rs else 0,
                "target_recall": sum(1 for r in rs if r["target_hit"]) / len(rs) if rs else 0,
                "ensemble_recall": sum(1 for r in rs if r["ensemble_hit"]) / len(rs) if rs else 0,
                "ensemble_delta_pp": round(100 * (sum(1 for r in rs if r["ensemble_hit"]) - sum(1 for r in rs if r["knn_hit"])) / len(rs), 1) if rs else 0
            }

    # Analyze by drug class diversity
    diversity_results = {"low_diversity": [], "high_diversity": []}
    for r in results:
        if r["drug_classes"] <= 2:
            diversity_results["low_diversity"].append(r)
        else:
            diversity_results["high_diversity"].append(r)

    diversity_summary = {}
    for div_type, rs in diversity_results.items():
        if rs:
            diversity_summary[div_type] = {
                "n": len(rs),
                "knn_recall": sum(1 for r in rs if r["knn_hit"]) / len(rs),
                "target_recall": sum(1 for r in rs if r["target_hit"]) / len(rs),
                "ensemble_recall": sum(1 for r in rs if r["ensemble_hit"]) / len(rs),
                "ensemble_delta": (sum(1 for r in rs if r["ensemble_hit"]) - sum(1 for r in rs if r["knn_hit"])) / len(rs)
            }

    # Analyze by target-kNN gap (from h370)
    gap_results = {"small_gap": [], "large_gap": []}
    for r in results:
        target_better = r["target_hit"] and not r["knn_hit"]
        knn_better = r["knn_hit"] and not r["target_hit"]
        if target_better or knn_better:
            gap_results["large_gap"].append(r)
        else:
            gap_results["small_gap"].append(r)

    gap_summary = {}
    for gap_type, rs in gap_results.items():
        if rs:
            gap_summary[gap_type] = {
                "n": len(rs),
                "knn_recall": sum(1 for r in rs if r["knn_hit"]) / len(rs),
                "target_recall": sum(1 for r in rs if r["target_hit"]) / len(rs),
                "ensemble_recall": sum(1 for r in rs if r["ensemble_hit"]) / len(rs),
                "ensemble_delta": (sum(1 for r in rs if r["ensemble_hit"]) - max(sum(1 for r in rs if r["knn_hit"]), sum(1 for r in rs if r["target_hit"]))) / len(rs)
            }

    return {
        "overall": {
            "knn_recall": round(knn_recall, 3),
            "target_recall": round(target_recall, 3),
            "ensemble_recall": round(ensemble_recall, 3),
            "ensemble_delta_pp": round(100 * (ensemble_recall - knn_recall), 1)
        },
        "ensemble_rescue_count": len(ensemble_rescues),
        "ensemble_hurt_count": len(ensemble_hurts),
        "by_gene_quartile": gene_quartile_summary,
        "by_drug_count": drug_count_results,
        "by_category": category_summary,
        "by_drug_class_diversity": diversity_summary,
        "by_method_gap": gap_summary,
        "gene_quartile_boundaries": {
            "Q1_max": float(q1),
            "Q2_max": float(q2),
            "Q3_max": float(q3)
        },
        "rescue_examples": [{"disease": r["disease_name"], "category": r["category"], "genes": r["gene_count"]} for r in ensemble_rescues[:10]],
        "hurt_examples": [{"disease": r["disease_name"], "category": r["category"], "genes": r["gene_count"]} for r in ensemble_hurts[:10]]
    }


def main():
    print("=" * 60)
    print("h376: Ensemble Coverage Analysis")
    print("=" * 60)

    results = analyze_diseases(n_diseases=300)

    if results.get("n_evaluated", 0) == 0:
        print("ERROR: No diseases evaluated")
        return

    # Save full results
    output_path = Path(__file__).parent.parent / "data/analysis/h376_ensemble_coverage.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to {output_path}")

    # Print summary
    summary = results["summary"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n1. OVERALL PERFORMANCE:")
    print(f"   kNN Recall@30: {summary['overall']['knn_recall']:.1%}")
    print(f"   Target Recall@30: {summary['overall']['target_recall']:.1%}")
    print(f"   Ensemble Recall@30: {summary['overall']['ensemble_recall']:.1%}")
    print(f"   Ensemble Delta: {summary['overall']['ensemble_delta_pp']:+.1f} pp")
    print(f"   Diseases rescued by ensemble: {summary['ensemble_rescue_count']}")
    print(f"   Diseases hurt by ensemble: {summary['ensemble_hurt_count']}")

    print("\n2. BY GENE COUNT QUARTILE:")
    for q, data in sorted(summary["by_gene_quartile"].items()):
        delta = data["ensemble_delta"] * 100
        print(f"   {q}: n={data['n']}, kNN={data['knn_recall']:.1%}, Target={data['target_recall']:.1%}, Ens={data['ensemble_recall']:.1%}, Δ={delta:+.1f}pp")

    print("\n3. BY DRUG COUNT:")
    for dc, data in sorted(summary["by_drug_count"].items()):
        delta = data["ensemble_delta"] * 100
        print(f"   {dc}: n={data['n']}, kNN={data['knn_recall']:.1%}, Ens={data['ensemble_recall']:.1%}, Δ={delta:+.1f}pp")

    print("\n4. BY CATEGORY (n>=3):")
    cat_sorted = sorted(summary["by_category"].items(), key=lambda x: -x[1]["n"])
    for cat, data in cat_sorted[:12]:
        print(f"   {cat}: n={data['n']}, kNN={data['knn_recall']:.1%}, Target={data['target_recall']:.1%}, Ens={data['ensemble_recall']:.1%}, Δ={data['ensemble_delta_pp']:+.1f}pp")

    print("\n5. BY DRUG CLASS DIVERSITY:")
    for div, data in summary["by_drug_class_diversity"].items():
        delta = data["ensemble_delta"] * 100
        print(f"   {div}: n={data['n']}, kNN={data['knn_recall']:.1%}, Ens={data['ensemble_recall']:.1%}, Δ={delta:+.1f}pp")

    print("\n6. BY METHOD GAP (small = both agree, large = one is better):")
    for gap, data in summary["by_method_gap"].items():
        delta = data["ensemble_delta"] * 100
        print(f"   {gap}: n={data['n']}, kNN={data['knn_recall']:.1%}, Target={data['target_recall']:.1%}, Ens={data['ensemble_recall']:.1%}, Δ={delta:+.1f}pp")

    print("\n7. GENE COUNT QUARTILE BOUNDARIES:")
    bounds = summary["gene_quartile_boundaries"]
    print(f"   Q1 (low): 0-{bounds['Q1_max']:.0f} genes")
    print(f"   Q2: {bounds['Q1_max']:.0f}-{bounds['Q2_max']:.0f} genes")
    print(f"   Q3: {bounds['Q2_max']:.0f}-{bounds['Q3_max']:.0f} genes")
    print(f"   Q4 (high): >{bounds['Q3_max']:.0f} genes")

    # Rescue examples
    if summary["rescue_examples"]:
        print("\n8. EXAMPLE DISEASES RESCUED BY ENSEMBLE:")
        for ex in summary["rescue_examples"][:5]:
            print(f"   - {ex['disease']} ({ex['category']}, {ex['genes']} genes)")

    # Hurt examples
    if summary["hurt_examples"]:
        print("\n9. EXAMPLE DISEASES HURT BY ENSEMBLE:")
        for ex in summary["hurt_examples"][:5]:
            print(f"   - {ex['disease']} ({ex['category']}, {ex['genes']} genes)")

    # Key findings
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Find best category for ensemble
    if summary["by_category"]:
        best_cat = max(summary["by_category"].items(), key=lambda x: x[1]["ensemble_delta_pp"])
        worst_cat = min(summary["by_category"].items(), key=lambda x: x[1]["ensemble_delta_pp"])
        print(f"\n- Best category for ensemble: {best_cat[0]} ({best_cat[1]['ensemble_delta_pp']:+.1f} pp)")
        print(f"- Worst category for ensemble: {worst_cat[0]} ({worst_cat[1]['ensemble_delta_pp']:+.1f} pp)")

    # Gene count effect
    gene_q = summary["by_gene_quartile"]
    q1_delta = gene_q.get("Q1_low", {}).get("ensemble_delta", 0)
    q4_delta = gene_q.get("Q4_high", {}).get("ensemble_delta", 0)
    print(f"\n- Low gene count (Q1) ensemble delta: {q1_delta*100:+.1f} pp")
    print(f"- High gene count (Q4) ensemble delta: {q4_delta*100:+.1f} pp")

    # Drug diversity effect
    div = summary["by_drug_class_diversity"]
    low_div = div.get("low_diversity", {}).get("ensemble_delta", 0)
    high_div = div.get("high_diversity", {}).get("ensemble_delta", 0)
    print(f"\n- Low drug class diversity ensemble delta: {low_div*100:+.1f} pp")
    print(f"- High drug class diversity ensemble delta: {high_div*100:+.1f} pp")

    # Method gap effect
    gap = summary["by_method_gap"]
    small_gap = gap.get("small_gap", {}).get("ensemble_delta", 0)
    large_gap = gap.get("large_gap", {}).get("ensemble_delta", 0)
    print(f"\n- Small gap (methods agree) ensemble delta: {small_gap*100:+.1f} pp")
    print(f"- Large gap (one method better) ensemble delta: {large_gap*100:+.1f} pp")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
