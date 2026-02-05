#!/usr/bin/env python3
"""
h165: Per-Disease-Category Precision Calibration

Compute precision by (disease_category, confidence_tier) pairs to see if
category-specific calibration would be more accurate than tier-only calibration.

SUCCESS: Identify categories where tier-based precision is miscalibrated
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

SEEDS = [42, 123, 456, 789, 1024]

# Disease categories from production_predictor.py
CATEGORY_KEYWORDS = {
    "infectious": [
        "infection", "infectious", "bacterial", "viral", "fungal", "parasitic",
        "tuberculosis", "hepatitis", "HIV", "malaria", "sepsis", "pneumonia",
        "meningitis", "abscess", "cellulitis", "osteomyelitis", "endocarditis",
        "encephalitis", "influenza", "covid", "coronavirus",
    ],
    "autoimmune": [
        "autoimmune", "rheumatoid", "lupus", "sclerosis", "psoriasis",
        "inflammatory bowel", "crohn", "ulcerative colitis", "arthritis",
        "spondylitis", "myasthenia", "hashimoto", "graves", "sjogren",
        "dermatomyositis", "polymyositis", "vasculitis",
    ],
    "cancer": [
        "cancer", "carcinoma", "leukemia", "lymphoma", "melanoma", "sarcoma",
        "myeloma", "tumor", "neoplasm", "malignant", "oncology", "metastatic",
        "glioma", "glioblastoma", "adenocarcinoma",
    ],
    "cardiovascular": [
        "heart", "cardiac", "cardiovascular", "hypertension", "arrhythmia",
        "coronary", "atherosclerosis", "myocardial", "angina", "stroke",
        "vascular", "thrombosis", "embolism", "aneurysm", "cardiomyopathy",
    ],
    "metabolic": [
        "diabetes", "metabolic", "obesity", "hyperlipidemia", "hypercholesterolemia",
        "hypertriglyceridemia", "gout", "hyperuricemia", "thyroid", "porphyria",
        "lipodystrophy", "glycogen storage",
    ],
    "neurological": [
        "neurological", "alzheimer", "parkinson", "epilepsy", "seizure",
        "dementia", "neuropathy", "migraine", "headache", "ataxia",
        "huntington", "amyotrophic", "multiple sclerosis",
    ],
    "respiratory": [
        "respiratory", "asthma", "copd", "pulmonary", "bronchitis", "emphysema",
        "cystic fibrosis", "pulmonary fibrosis", "pneumonitis", "pleurisy",
    ],
    "dermatological": [
        "dermatological", "skin", "dermatitis", "eczema", "acne", "rosacea",
        "urticaria", "pruritus", "psoriasis", "vitiligo", "alopecia",
    ],
    "gastrointestinal": [
        "gastrointestinal", "gastric", "intestinal", "hepatic", "liver",
        "cirrhosis", "pancreatitis", "cholecystitis", "gastritis", "colitis",
        "ulcer", "reflux", "gerd", "dyspepsia",
    ],
    "psychiatric": [
        "psychiatric", "depression", "anxiety", "bipolar", "schizophrenia",
        "psychosis", "ocd", "ptsd", "adhd", "insomnia", "sleep disorder",
    ],
    "ophthalmic": [
        "ophthalmic", "eye", "ocular", "retinal", "glaucoma", "macular",
        "cataract", "uveitis", "conjunctivitis", "keratitis",
    ],
    "hematological": [
        "hematological", "anemia", "thrombocytopenia", "neutropenia",
        "hemophilia", "sickle cell", "thalassemia", "polycythemia",
    ],
}


def categorize_disease(disease_name: str) -> str:
    """Categorize a disease by keywords."""
    disease_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in disease_lower:
                return category
    return "other"


def load_ground_truth() -> Dict[str, set]:
    """Load ground truth drug-disease pairs."""
    gt_file = REFERENCE_DIR / "expanded_ground_truth.json"
    with open(gt_file) as f:
        gt_data = json.load(f)

    gt_dict = defaultdict(set)
    # gt_data is a dict: {disease_id: [list of drug names]}
    for disease_id, drugs in gt_data.items():
        for drug in drugs:
            if isinstance(drug, dict):
                gt_dict[disease_id].add(drug.get("name", "").lower())
            else:
                gt_dict[disease_id].add(drug.lower())
    return gt_dict


def load_drug_targets() -> Dict[str, set]:
    """Load drug targets for mechanism support."""
    target_file = REFERENCE_DIR / "drug_targets.json"
    if not target_file.exists():
        return {}
    with open(target_file) as f:
        return json.load(f)


def load_disease_genes() -> Dict[str, set]:
    """Load disease genes for mechanism support."""
    gene_file = REFERENCE_DIR / "disease_genes.json"
    if not gene_file.exists():
        return {}
    with open(gene_file) as f:
        return json.load(f)


def compute_mechanism_support(drug: str, disease: str, drug_targets: Dict, disease_genes: Dict) -> bool:
    """Check if drug has mechanism support for disease."""
    if not drug_targets or not disease_genes:
        return False

    targets = drug_targets.get(drug.lower(), set())
    if isinstance(targets, list):
        targets = set(targets)

    genes = disease_genes.get(disease, set())
    if isinstance(genes, list):
        genes = set(genes)

    return len(targets & genes) > 0


def assign_tier(rank: int, freq: int, has_mech: bool, is_tier1: bool, drug_class_rescue: str = None) -> str:
    """
    Assign confidence tier based on production rules.
    Simplified version for analysis.
    """
    # FILTER criteria
    if rank > 20:
        return "FILTER"
    if freq <= 2 and not has_mech:
        return "FILTER"

    # GOLDEN criteria (Tier 1 or rescued)
    if is_tier1 and freq >= 10 and has_mech:
        return "GOLDEN"
    if drug_class_rescue:
        return "GOLDEN"

    # HIGH criteria
    if freq >= 15 and has_mech:
        return "HIGH"
    if rank <= 5 and freq >= 10 and has_mech:
        return "HIGH"

    # MEDIUM criteria
    if freq >= 5 and has_mech:
        return "MEDIUM"
    if freq >= 10:
        return "MEDIUM"

    return "LOW"


def run_evaluation(seed: int):
    """Run one seed of evaluation."""
    np.random.seed(seed)

    # Load data
    gt_dict = load_ground_truth()
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    # Load embeddings from CSV
    emb_df = pd.read_csv(EMBEDDINGS_DIR / "node2vec_256_named.csv")
    emb_df.set_index("entity", inplace=True)

    # Separate disease and drug embeddings
    disease_ids = [idx for idx in emb_df.index if "Disease::" in idx]
    emb_cols = [c for c in emb_df.columns if c.startswith("dim_")]

    disease_emb = emb_df.loc[disease_ids, emb_cols].values
    disease_idx = {d: i for i, d in enumerate(disease_ids)}

    # Load disease name mappings from mesh_mappings_from_agents.json
    # Build reverse mapping: MESH ID -> disease name
    disease_name_map = {}
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_mappings = json.load(f)
    for batch_key, batch_data in mesh_mappings.items():
        if batch_key == "metadata":
            continue
        for disease_name, mesh_id in batch_data.items():
            if mesh_id:
                disease_name_map[f"Disease::MESH:{mesh_id}"] = disease_name

    # Convert GT dict keys to match embedding format (remove drkg: prefix)
    gt_dict_normalized = {}
    for k, v in gt_dict.items():
        # Normalize key: drkg:Disease::MESH:X -> Disease::MESH:X
        norm_key = k.replace("drkg:", "") if k.startswith("drkg:") else k
        # Normalize drug values: drkg:compound::xxx -> compound name
        norm_drugs = set()
        for drug in v:
            drug_norm = drug.replace("drkg:", "") if drug.startswith("drkg:") else drug
            # Extract drug name if in format Compound::XXX or compound::xxx
            if "::" in drug_norm:
                norm_drugs.add(drug_norm.split("::")[-1].lower())
            else:
                norm_drugs.add(drug_norm.lower())
        gt_dict_normalized[norm_key] = norm_drugs

    # Get evaluable diseases
    evaluable = [d for d in disease_ids if d in gt_dict_normalized and len(gt_dict_normalized[d]) > 0]
    evaluable = [d for d in evaluable if d in disease_idx]

    # Train/test split
    np.random.shuffle(evaluable)
    split_idx = int(len(evaluable) * 0.8)
    train_diseases = set(evaluable[:split_idx])
    test_diseases = evaluable[split_idx:]

    # Compute disease similarity
    disease_sim = cosine_similarity(disease_emb)

    # Count drug frequency in training set
    drug_freq = defaultdict(int)
    for d in train_diseases:
        for drug in gt_dict_normalized.get(d, []):
            drug_freq[drug] += 1

    results = []

    for disease in test_diseases:
        disease_name = disease_name_map.get(disease, disease)
        category = categorize_disease(disease_name)

        # Determine if Tier 1
        is_tier1 = category in ["autoimmune", "dermatological", "psychiatric"]

        # Get similar diseases
        d_idx = disease_idx[disease]
        sims = disease_sim[d_idx]

        # Exclude self and non-training diseases
        similar_indices = []
        for i, d in enumerate(disease_ids):
            if d != disease and d in train_diseases:
                similar_indices.append((i, sims[i]))

        similar_indices.sort(key=lambda x: -x[1])
        top_k = similar_indices[:20]

        # Rank drugs by kNN weighted frequency
        drug_scores = defaultdict(float)
        for idx, sim in top_k:
            neighbor = disease_ids[idx]
            for drug in gt_dict_normalized.get(neighbor, []):
                drug_scores[drug] += sim

        ranked_drugs = sorted(drug_scores.items(), key=lambda x: -x[1])

        # Get ground truth drugs
        gt_drugs = gt_dict_normalized.get(disease, set())

        # Evaluate each prediction
        for rank, (drug, _) in enumerate(ranked_drugs[:30], 1):
            freq = drug_freq.get(drug, 0)
            has_mech = compute_mechanism_support(drug, disease, drug_targets, disease_genes)
            tier = assign_tier(rank, freq, has_mech, is_tier1)
            is_hit = drug.lower() in {g.lower() for g in gt_drugs}

            results.append({
                "disease": disease,
                "disease_name": disease_name,
                "category": category,
                "drug": drug,
                "rank": rank,
                "freq": freq,
                "has_mech": has_mech,
                "tier": tier,
                "is_hit": is_hit,
                "seed": seed,
            })

    return results


def main():
    print("h165: Per-Disease-Category Precision Calibration")
    print("=" * 80)

    all_results = []
    for seed in SEEDS:
        print(f"Running seed {seed}...")
        results = run_evaluation(seed)
        all_results.extend(results)

    df = pd.DataFrame(all_results)

    # Overall tier precision (baseline)
    print("\n" + "=" * 80)
    print("OVERALL TIER PRECISION (Current Calibration)")
    print("=" * 80)

    tier_stats = df.groupby("tier").agg(
        n=("is_hit", "count"),
        hits=("is_hit", "sum"),
    )
    tier_stats["precision"] = (tier_stats["hits"] / tier_stats["n"] * 100).round(1)

    tier_order = ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
    for tier in tier_order:
        if tier in tier_stats.index:
            row = tier_stats.loc[tier]
            print(f"  {tier:8s}: {row['precision']:5.1f}% ({int(row['hits'])}/{int(row['n'])})")

    # Per-category precision
    print("\n" + "=" * 80)
    print("PRECISION BY (CATEGORY, TIER)")
    print("=" * 80)

    category_tier_stats = df.groupby(["category", "tier"]).agg(
        n=("is_hit", "count"),
        hits=("is_hit", "sum"),
    ).reset_index()
    category_tier_stats["precision"] = (category_tier_stats["hits"] / category_tier_stats["n"] * 100).round(1)

    # Pivot for display
    pivot = category_tier_stats.pivot(index="category", columns="tier", values="precision")
    pivot = pivot.reindex(columns=tier_order)

    print("\n" + pivot.to_string())

    # Find categories where tier precision differs from overall
    print("\n" + "=" * 80)
    print("MISCALIBRATION ANALYSIS (Category vs Overall)")
    print("=" * 80)

    overall_precision = tier_stats["precision"].to_dict()

    miscalibrations = []
    for _, row in category_tier_stats.iterrows():
        cat = row["category"]
        tier = row["tier"]
        n = row["n"]
        prec = row["precision"]

        if tier not in overall_precision or n < 10:
            continue

        overall = overall_precision[tier]
        diff = prec - overall

        if abs(diff) >= 10:  # >10pp difference
            miscalibrations.append({
                "category": cat,
                "tier": tier,
                "n": int(n),
                "category_precision": prec,
                "overall_precision": overall,
                "difference": diff,
            })

    if miscalibrations:
        miscalibrations.sort(key=lambda x: -abs(x["difference"]))
        print("\nSignificant miscalibrations (>10pp difference, n>=10):")
        for m in miscalibrations:
            sign = "+" if m["difference"] > 0 else ""
            print(f"  {m['category']:15s} {m['tier']:8s}: "
                  f"{m['category_precision']:5.1f}% vs {m['overall_precision']:5.1f}% overall "
                  f"({sign}{m['difference']:+.1f}pp, n={m['n']})")
    else:
        print("\nNo significant miscalibrations found (all within 10pp)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\nTotal predictions: {len(df)}")
    print(f"Unique diseases: {df['disease'].nunique()}")
    print(f"Seeds used: {len(SEEDS)}")

    # Save results
    output = {
        "tier_precision": tier_stats.to_dict(),
        "category_tier_precision": category_tier_stats.to_dict(orient="records"),
        "miscalibrations": miscalibrations,
    }

    output_file = ANALYSIS_DIR / "h165_category_calibration.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
