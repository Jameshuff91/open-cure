#!/usr/bin/env python3
"""
h381: Category-Specific Ensemble Routing

Tests whether applying MinRank ensemble only to categories where it helps
(autoimmune, metabolic, cancer) improves overall R@30.

From h376:
- Benefiting categories: autoimmune (+7.7pp), metabolic (+8.3pp), cancer (+2.0pp)
- Also benefiting but small n: rare_genetic (+25pp,n=4), hematological (+20pp,n=5), musculoskeletal (+20pp,n=5)
- Hurting categories: CV (-14.3pp), neuro (-11.1pp), immune (-12.5pp), respiratory (-16.7pp)

Evaluation approach:
1. LOO evaluation comparing: kNN-only vs category-routed ensemble
2. Test multiple routing sets (conservative vs aggressive)
3. Production context test (with tier rules)
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from production_predictor import DrugRepurposingPredictor
from disease_categorizer import categorize_disease
from sklearn.metrics.pairwise import cosine_similarity


def compute_knn_scores(predictor: DrugRepurposingPredictor, disease_id: str, k: int = 20) -> Dict[str, float]:
    """Compute kNN drug scores for a disease with leave-one-out."""
    if disease_id not in predictor.embeddings:
        return {}

    test_emb = predictor.embeddings[disease_id].reshape(1, -1)
    sims = cosine_similarity(test_emb, predictor.train_embeddings)[0]

    # Exclude the disease itself
    if disease_id in predictor.train_diseases:
        self_idx = predictor.train_diseases.index(disease_id)
        sims[self_idx] = -1

    top_k_idx = np.argsort(sims)[-(k+1):]

    drug_scores: Dict[str, float] = defaultdict(float)
    count = 0
    for idx in reversed(top_k_idx):
        neighbor_disease = predictor.train_diseases[idx]
        if neighbor_disease == disease_id:
            continue
        if count >= k:
            break

        neighbor_sim = sims[idx]
        for drug_id in predictor.ground_truth[neighbor_disease]:
            if drug_id in predictor.embeddings:
                drug_scores[drug_id] += neighbor_sim
        count += 1

    return dict(drug_scores)


def minrank_ensemble(knn_scores: Dict[str, float], target_scores: Dict[str, int]) -> List[str]:
    """Apply MinRank ensemble and return top-30 drug IDs."""
    all_drugs = set(knn_scores.keys()) | set(target_scores.keys())
    if not all_drugs:
        return []

    knn_ranked = sorted(knn_scores.items(), key=lambda x: -x[1])
    knn_ranks = {drug: i + 1 for i, (drug, _) in enumerate(knn_ranked)}

    target_ranked = sorted(target_scores.items(), key=lambda x: -x[1])
    target_ranks = {drug: i + 1 for i, (drug, _) in enumerate(target_ranked)}

    max_rank = len(all_drugs)
    fusion = []
    for drug in all_drugs:
        knn_r = knn_ranks.get(drug, max_rank)
        target_r = target_ranks.get(drug, max_rank)
        min_r = min(knn_r, target_r)
        combined = knn_scores.get(drug, 0) + target_scores.get(drug, 0)
        fusion.append((drug, min_r, combined))

    fusion.sort(key=lambda x: (x[1], -x[2]))
    return [drug for drug, _, _ in fusion[:30]]


def evaluate_routing(predictor: DrugRepurposingPredictor,
                     ensemble_categories: Set[str],
                     n_diseases: int = 0) -> Dict:
    """Evaluate category-routed ensemble vs pure kNN.

    Args:
        ensemble_categories: Categories to apply ensemble to
        n_diseases: 0 = all diseases
    """
    gt = predictor.ground_truth

    evaluable = []
    for disease_id, drugs in gt.items():
        if disease_id not in predictor.embeddings:
            continue
        disease_name = predictor.disease_names.get(disease_id, disease_id)
        category = categorize_disease(disease_name) or "other"
        drug_list = list(drugs) if isinstance(drugs, set) else drugs
        evaluable.append({
            "disease_id": disease_id,
            "disease_name": disease_name,
            "category": category,
            "gt_drugs": drug_list
        })

    if n_diseases > 0 and len(evaluable) > n_diseases:
        np.random.seed(42)
        indices = np.random.choice(len(evaluable), n_diseases, replace=False)
        evaluable = [evaluable[i] for i in sorted(indices)]

    print(f"Evaluating {len(evaluable)} diseases...")
    print(f"Ensemble categories: {ensemble_categories}")

    # Track results
    knn_hits = 0
    routed_hits = 0

    cat_knn = defaultdict(lambda: {"hits": 0, "total": 0})
    cat_routed = defaultdict(lambda: {"hits": 0, "total": 0})

    changes = []  # Track where routing changed the outcome

    for i, d in enumerate(evaluable):
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(evaluable)}...")

        disease_id = d["disease_id"]
        category = d["category"]
        gt_drugs = set(d["gt_drugs"])

        # kNN scores (always computed)
        knn_scores = compute_knn_scores(predictor, disease_id)
        if not knn_scores:
            continue

        # kNN top-30
        knn_top30 = sorted(knn_scores.items(), key=lambda x: -x[1])[:30]
        knn_top30_drugs = {drug for drug, _ in knn_top30}
        knn_hit = bool(gt_drugs & knn_top30_drugs)

        # Routed: use ensemble for selected categories, kNN for others
        if category in ensemble_categories:
            target_scores = predictor._get_target_scores(disease_id)
            if target_scores:
                routed_top30_drugs = set(minrank_ensemble(knn_scores, target_scores))
            else:
                routed_top30_drugs = knn_top30_drugs
        else:
            routed_top30_drugs = knn_top30_drugs

        routed_hit = bool(gt_drugs & routed_top30_drugs)

        knn_hits += knn_hit
        routed_hits += routed_hit

        cat_knn[category]["total"] += 1
        cat_routed[category]["total"] += 1
        cat_knn[category]["hits"] += knn_hit
        cat_routed[category]["hits"] += routed_hit

        if knn_hit != routed_hit:
            changes.append({
                "disease": d["disease_name"],
                "category": category,
                "knn_hit": knn_hit,
                "routed_hit": routed_hit,
                "direction": "rescued" if routed_hit and not knn_hit else "hurt"
            })

    total = sum(cat_knn[c]["total"] for c in cat_knn)

    result = {
        "ensemble_categories": sorted(ensemble_categories),
        "n_evaluated": total,
        "overall": {
            "knn_recall": knn_hits / total if total > 0 else 0,
            "routed_recall": routed_hits / total if total > 0 else 0,
            "delta_pp": round(100 * (routed_hits - knn_hits) / total, 2) if total > 0 else 0,
            "rescued": sum(1 for c in changes if c["direction"] == "rescued"),
            "hurt": sum(1 for c in changes if c["direction"] == "hurt"),
        },
        "by_category": {},
        "changes": changes
    }

    for cat in sorted(cat_knn.keys()):
        n = cat_knn[cat]["total"]
        if n >= 3:
            k_r = cat_knn[cat]["hits"] / n
            r_r = cat_routed[cat]["hits"] / n
            result["by_category"][cat] = {
                "n": n,
                "knn_recall": round(k_r, 3),
                "routed_recall": round(r_r, 3),
                "delta_pp": round(100 * (r_r - k_r), 1),
                "ensemble_applied": cat in ensemble_categories
            }

    return result


def main():
    print("=" * 60)
    print("h381: Category-Specific Ensemble Routing")
    print("=" * 60)

    print("\nLoading predictor...")
    predictor = DrugRepurposingPredictor()

    # Define routing sets to test
    routing_sets = {
        "knn_only": set(),  # Baseline - no ensemble
        "conservative": {"autoimmune", "metabolic", "cancer"},  # h376 findings
        "aggressive": {"autoimmune", "metabolic", "cancer", "rare_genetic", "hematological", "musculoskeletal"},
        "all_positive": {"autoimmune", "metabolic", "cancer", "rare_genetic", "hematological", "musculoskeletal", "dermatological", "gastrointestinal", "genetic", "ophthalmological", "psychiatric", "renal"},  # All non-negative categories
    }

    all_results = {}

    for name, categories in routing_sets.items():
        print(f"\n{'='*40}")
        print(f"Testing: {name}")
        print(f"{'='*40}")

        result = evaluate_routing(predictor, categories, n_diseases=300)
        all_results[name] = result

        o = result["overall"]
        print(f"\n  Overall: kNN {o['knn_recall']:.1%} → Routed {o['routed_recall']:.1%} ({o['delta_pp']:+.1f} pp)")
        print(f"  Rescued: {o['rescued']}, Hurt: {o['hurt']}")

        if result["changes"]:
            print(f"\n  Changes:")
            for c in result["changes"][:5]:
                print(f"    {c['direction']}: {c['disease']} ({c['category']})")

    # Summary comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Strategy':<20} {'kNN R@30':>10} {'Routed R@30':>12} {'Delta':>8} {'Rescued':>8} {'Hurt':>6}")
    print("-" * 64)
    for name in routing_sets:
        o = all_results[name]["overall"]
        print(f"{name:<20} {o['knn_recall']:>9.1%} {o['routed_recall']:>11.1%} {o['delta_pp']:>+7.1f}pp {o['rescued']:>7} {o['hurt']:>5}")

    # Category detail for conservative (primary strategy)
    print("\n\nCATEGORY DETAIL (conservative routing):")
    print(f"{'Category':<20} {'n':>4} {'kNN':>8} {'Routed':>8} {'Delta':>8} {'Ensemble?':>10}")
    print("-" * 58)
    for cat, info in sorted(all_results["conservative"]["by_category"].items(), key=lambda x: -x[1]["delta_pp"]):
        marker = "YES" if info["ensemble_applied"] else ""
        print(f"{cat:<20} {info['n']:>4} {info['knn_recall']:>7.1%} {info['routed_recall']:>7.1%} {info['delta_pp']:>+7.1f}pp {marker:>10}")

    # Save results
    output_path = Path(__file__).parent.parent / "data/analysis/h381_category_ensemble_routing.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Verdict
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    conservative = all_results["conservative"]["overall"]
    if conservative["delta_pp"] > 0:
        print(f"\nCATEGORY-SPECIFIC ROUTING WORKS: +{conservative['delta_pp']:.1f} pp overall")
        print("Recommendation: Enable ensemble for autoimmune/metabolic/cancer in production")
    elif conservative["delta_pp"] == 0:
        print(f"\nNO OVERALL CHANGE: {conservative['delta_pp']:.1f} pp")
        print("Ensemble routing is neutral. Changes cancel out.")
    else:
        print(f"\nENSEMBLE ROUTING HURTS: {conservative['delta_pp']:.1f} pp overall")
        print("Even selective routing cannot overcome ensemble noise.")

    # Check if any strategy works
    best_strategy = max(all_results.items(), key=lambda x: x[1]["overall"]["delta_pp"])
    if best_strategy[0] != "knn_only" and best_strategy[1]["overall"]["delta_pp"] > 0:
        print(f"\nBest strategy: {best_strategy[0]} ({best_strategy[1]['overall']['delta_pp']:+.1f} pp)")
    else:
        print(f"\nNo strategy improves over pure kNN.")


if __name__ == "__main__":
    main()
