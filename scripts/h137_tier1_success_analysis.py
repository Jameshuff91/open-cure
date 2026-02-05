#!/usr/bin/env python3
"""
h137: Why Do Tier 1 Categories Succeed?

h132 showed dramatic gap: Tier 1 + filters = 58% vs Tier 2/3 + filters = ~28%.
Understanding WHY Tier 1 succeeds could inform approaches for other categories.

Potential factors:
1. More drugs per disease? (broader treatment options)
2. Better mechanism coverage? (more target-gene overlaps)
3. More training data? (more diseases in category)
4. Shared drug profiles? (similar drugs across diseases)

SUCCESS CRITERIA: Identify at least 2 structural factors explaining Tier 1 success
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

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

# Category tiers from h71
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}
TIER_3_CATEGORIES = {'infectious', 'neurological', 'metabolic', 'respiratory', 'gastrointestinal', 'hematological'}

CATEGORY_KEYWORDS = {
    'autoimmune': ['autoimmune', 'lupus', 'rheumatoid', 'arthritis', 'scleroderma', 'myasthenia',
                   'multiple sclerosis', 'crohn', 'colitis', 'psoriasis', 'sjögren'],
    'infectious': ['infection', 'bacterial', 'viral', 'fungal', 'hiv', 'aids', 'hepatitis',
                   'tuberculosis', 'malaria', 'pneumonia', 'sepsis', 'meningitis'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'oncology', 'sarcoma', 'myeloma'],
    'cardiovascular': ['cardiac', 'heart', 'coronary', 'hypertension', 'arrhythmia',
                       'atherosclerosis', 'stroke', 'vascular', 'myocardial', 'angina'],
    'neurological': ['neurological', 'alzheimer', 'parkinson', 'epilepsy', 'neuropathy',
                     'dementia', 'huntington', 'sclerosis', 'brain'],
    'metabolic': ['diabetes', 'metabolic', 'obesity', 'thyroid', 'hyperlipidemia',
                  'hypercholesterolemia', 'gout', 'porphyria'],
    'psychiatric': ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'psychiatric',
                    'ptsd', 'ocd', 'adhd', 'psychosis'],
    'respiratory': ['respiratory', 'asthma', 'copd', 'pulmonary', 'lung', 'bronchitis',
                    'pneumonitis', 'fibrosis'],
    'gastrointestinal': ['gastrointestinal', 'gastric', 'intestinal', 'bowel', 'liver',
                         'hepatic', 'cirrhosis', 'pancreatitis', 'celiac'],
    'dermatological': ['skin', 'dermatitis', 'eczema', 'psoriasis', 'dermatological',
                       'acne', 'urticaria', 'vitiligo'],
    'ophthalmic': ['eye', 'retinal', 'glaucoma', 'macular', 'ophthalmic', 'uveitis',
                   'conjunctivitis', 'keratitis'],
    'hematological': ['anemia', 'leukemia', 'lymphoma', 'hemophilia', 'thrombocytopenia',
                      'neutropenia', 'hematological', 'myelodysplastic'],
}


def load_node2vec_embeddings() -> Dict[str, np.ndarray]:
    embeddings_path = EMBEDDINGS_DIR / "node2vec_256_named.csv"
    df = pd.read_csv(embeddings_path)
    dim_cols = [c for c in df.columns if c.startswith("dim_")]
    embeddings: Dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        entity = f"drkg:{row['entity']}"
        embeddings[entity] = row[dim_cols].values.astype(np.float32)
    return embeddings


def load_drugbank_lookup():
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_full = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_full


def load_mesh_mappings_from_file() -> Dict[str, str]:
    mesh_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    if not mesh_path.exists():
        return {}
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    mesh_mappings: Dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith("D") or mesh_str.startswith("C"):
                        mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"
    return mesh_mappings


def load_ground_truth(mesh_mappings, name_to_drug_id):
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt: Dict[str, Set[str]] = defaultdict(set)
    disease_names: Dict[str, str] = {}

    for _, row in df.iterrows():
        disease = str(row.get("disease name", "")).strip()
        drug = str(row.get("final normalized drug label", "")).strip()
        if not disease or not drug:
            continue

        disease_id = matcher.get_mesh_id(disease)
        if not disease_id:
            disease_id = mesh_mappings.get(disease.lower())
        if not disease_id:
            continue

        disease_names[disease_id] = disease
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt[disease_id].add(drug_id)

    return dict(gt), disease_names


def load_drug_targets() -> Dict[str, Set[str]]:
    targets_path = REFERENCE_DIR / "drug_targets.json"
    if not targets_path.exists():
        return {}
    with open(targets_path) as f:
        drug_targets = json.load(f)
    return {f"drkg:Compound::{k}": set(v) for k, v in drug_targets.items()}


def load_disease_genes() -> Dict[str, Set[str]]:
    genes_path = REFERENCE_DIR / "disease_genes.json"
    if not genes_path.exists():
        return {}
    with open(genes_path) as f:
        disease_genes = json.load(f)

    result = {}
    for k, v in disease_genes.items():
        gene_set = set(v)
        result[k] = gene_set
        if k.startswith('MESH:'):
            drkg_key = f"drkg:Disease::{k}"
            result[drkg_key] = gene_set
    return result


def categorize_disease(disease_name: str) -> str:
    name_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw in name_lower:
                return category
    return 'other'


def get_tier(category: str) -> int:
    if category in TIER_1_CATEGORIES:
        return 1
    elif category in TIER_2_CATEGORIES:
        return 2
    else:
        return 3


def main():
    print("h137: Why Do Tier 1 Categories Succeed?")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()

    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drug targets available: {len(drug_targets)}")
    print(f"  Disease genes available: {len(disease_genes)}")

    # Categorize all diseases
    disease_categories = {d: categorize_disease(disease_names.get(d, "")) for d in ground_truth}
    disease_tiers = {d: get_tier(cat) for d, cat in disease_categories.items()}

    # Group diseases by tier
    tier_diseases = {1: [], 2: [], 3: []}
    for d, tier in disease_tiers.items():
        tier_diseases[tier].append(d)

    print("\n" + "=" * 70)
    print("DISEASE DISTRIBUTION BY TIER")
    print("=" * 70)

    for tier in [1, 2, 3]:
        diseases = tier_diseases[tier]
        categories = [disease_categories[d] for d in diseases]
        cat_counts = defaultdict(int)
        for c in categories:
            cat_counts[c] += 1
        print(f"\nTier {tier}: {len(diseases)} diseases")
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {count}")

    # Factor 1: Drugs per disease
    print("\n" + "=" * 70)
    print("FACTOR 1: DRUGS PER DISEASE")
    print("=" * 70)

    drugs_per_disease = {d: len(drugs) for d, drugs in ground_truth.items()}

    for tier in [1, 2, 3]:
        diseases = tier_diseases[tier]
        avg_drugs = np.mean([drugs_per_disease[d] for d in diseases])
        med_drugs = np.median([drugs_per_disease[d] for d in diseases])
        max_drugs = max([drugs_per_disease[d] for d in diseases])
        min_drugs = min([drugs_per_disease[d] for d in diseases])
        print(f"Tier {tier}: avg={avg_drugs:.1f}, median={med_drugs:.0f}, range=[{min_drugs}, {max_drugs}]")

    # Factor 2: Mechanism coverage
    print("\n" + "=" * 70)
    print("FACTOR 2: MECHANISM SUPPORT RATE")
    print("=" * 70)

    for tier in [1, 2, 3]:
        diseases = tier_diseases[tier]
        total_pairs = 0
        mech_support_pairs = 0

        for disease_id in diseases:
            disease_gene_set = disease_genes.get(disease_id, set())
            for drug_id in ground_truth[disease_id]:
                drug_gene_set = drug_targets.get(drug_id, set())
                total_pairs += 1
                if len(disease_gene_set & drug_gene_set) > 0:
                    mech_support_pairs += 1

        rate = mech_support_pairs / total_pairs * 100 if total_pairs > 0 else 0
        print(f"Tier {tier}: {rate:.1f}% of drug-disease pairs have mechanism support ({mech_support_pairs}/{total_pairs})")

    # Factor 3: Drug frequency distribution
    print("\n" + "=" * 70)
    print("FACTOR 3: DRUG FREQUENCY DISTRIBUTION")
    print("=" * 70)

    all_drugs = set()
    for drugs in ground_truth.values():
        all_drugs.update(drugs)

    drug_freq = defaultdict(int)
    for disease_id, drugs in ground_truth.items():
        for drug_id in drugs:
            drug_freq[drug_id] += 1

    for tier in [1, 2, 3]:
        diseases = tier_diseases[tier]
        tier_drugs = set()
        for d in diseases:
            tier_drugs.update(ground_truth[d])

        tier_drug_freqs = [drug_freq[drug_id] for drug_id in tier_drugs]
        avg_freq = np.mean(tier_drug_freqs)
        med_freq = np.median(tier_drug_freqs)
        high_freq_count = sum(1 for f in tier_drug_freqs if f >= 10)

        print(f"Tier {tier}: avg freq={avg_freq:.1f}, median={med_freq:.0f}, drugs with freq>=10: {high_freq_count}/{len(tier_drugs)} ({high_freq_count/len(tier_drugs)*100:.1f}%)")

    # Factor 4: Drug overlap between diseases in same tier
    print("\n" + "=" * 70)
    print("FACTOR 4: DRUG OVERLAP WITHIN TIER")
    print("=" * 70)

    for tier in [1, 2, 3]:
        diseases = tier_diseases[tier]
        if len(diseases) < 2:
            continue

        # Calculate average Jaccard similarity of drug sets
        total_jaccard = 0
        n_pairs = 0

        for i, d1 in enumerate(diseases):
            drugs1 = ground_truth[d1]
            for d2 in diseases[i+1:]:
                drugs2 = ground_truth[d2]
                intersection = len(drugs1 & drugs2)
                union = len(drugs1 | drugs2)
                if union > 0:
                    jaccard = intersection / union
                    total_jaccard += jaccard
                    n_pairs += 1

        avg_jaccard = total_jaccard / n_pairs if n_pairs > 0 else 0
        print(f"Tier {tier}: avg drug Jaccard similarity = {avg_jaccard:.4f} ({n_pairs} disease pairs)")

    # Factor 5: Disease gene coverage
    print("\n" + "=" * 70)
    print("FACTOR 5: DISEASE GENE ANNOTATION RATE")
    print("=" * 70)

    for tier in [1, 2, 3]:
        diseases = tier_diseases[tier]
        has_genes = sum(1 for d in diseases if d in disease_genes and len(disease_genes[d]) > 0)
        avg_genes = np.mean([len(disease_genes.get(d, set())) for d in diseases])

        print(f"Tier {tier}: {has_genes}/{len(diseases)} diseases have gene annotations ({has_genes/len(diseases)*100:.1f}%), avg genes = {avg_genes:.1f}")

    # Factor 6: Embedding clustering (are Tier 1 diseases more similar to each other?)
    print("\n" + "=" * 70)
    print("FACTOR 6: EMBEDDING CLUSTERING (within-tier similarity)")
    print("=" * 70)

    for tier in [1, 2, 3]:
        diseases = [d for d in tier_diseases[tier] if d in emb_dict]
        if len(diseases) < 2:
            continue

        embs = np.array([emb_dict[d] for d in diseases])
        sims = cosine_similarity(embs)

        # Average off-diagonal similarity
        n = len(diseases)
        off_diag = sims[np.triu_indices(n, k=1)]
        avg_sim = np.mean(off_diag)

        print(f"Tier {tier}: avg within-tier cosine similarity = {avg_sim:.4f} ({len(diseases)} diseases)")

    # Factor 7: Common drug analysis
    print("\n" + "=" * 70)
    print("FACTOR 7: TOP DRUGS BY TIER")
    print("=" * 70)

    for tier in [1, 2, 3]:
        diseases = tier_diseases[tier]
        tier_drug_counts = defaultdict(int)
        for d in diseases:
            for drug_id in ground_truth[d]:
                tier_drug_counts[drug_id] += 1

        top_drugs = sorted(tier_drug_counts.items(), key=lambda x: -x[1])[:5]
        print(f"\nTier {tier} top drugs:")
        for drug_id, count in top_drugs:
            drug_name = id_to_name.get(drug_id, drug_id)
            pct = count / len(diseases) * 100
            print(f"  {drug_name}: {count} diseases ({pct:.1f}%)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: WHY TIER 1 SUCCEEDS")
    print("=" * 70)

    # Collect key metrics
    tier1_diseases = tier_diseases[1]
    tier1_mech_rate = sum(1 for d in tier1_diseases for drug in ground_truth[d]
                          if len(disease_genes.get(d, set()) & drug_targets.get(drug, set())) > 0) / \
                      sum(len(ground_truth[d]) for d in tier1_diseases) * 100

    tier23_diseases = tier_diseases[2] + tier_diseases[3]
    tier23_mech_rate = sum(1 for d in tier23_diseases for drug in ground_truth[d]
                           if len(disease_genes.get(d, set()) & drug_targets.get(drug, set())) > 0) / \
                       sum(len(ground_truth[d]) for d in tier23_diseases) * 100

    tier1_drugs = set()
    for d in tier1_diseases:
        tier1_drugs.update(ground_truth[d])
    tier1_high_freq = sum(1 for drug in tier1_drugs if drug_freq[drug] >= 10)

    tier23_drugs = set()
    for d in tier23_diseases:
        tier23_drugs.update(ground_truth[d])
    tier23_high_freq = sum(1 for drug in tier23_drugs if drug_freq[drug] >= 10)

    findings = []

    print("\nKey differentiators:")

    # Check mechanism rate
    if tier1_mech_rate > tier23_mech_rate * 1.2:
        finding = f"1. MECHANISM COVERAGE: Tier 1 = {tier1_mech_rate:.1f}% vs Tier 2/3 = {tier23_mech_rate:.1f}%"
        print(finding)
        findings.append(finding)
    else:
        print(f"1. Mechanism coverage: Tier 1 = {tier1_mech_rate:.1f}% vs Tier 2/3 = {tier23_mech_rate:.1f}% (similar)")

    # Check high-freq drug ratio
    tier1_ratio = tier1_high_freq / len(tier1_drugs) * 100
    tier23_ratio = tier23_high_freq / len(tier23_drugs) * 100
    if tier1_ratio > tier23_ratio * 1.2:
        finding = f"2. HIGH-FREQ DRUGS: Tier 1 = {tier1_ratio:.1f}% vs Tier 2/3 = {tier23_ratio:.1f}%"
        print(finding)
        findings.append(finding)
    else:
        print(f"2. High-freq drugs: Tier 1 = {tier1_ratio:.1f}% vs Tier 2/3 = {tier23_ratio:.1f}% (similar)")

    # Check drug overlap
    tier1_jaccard = sum(len(ground_truth[d1] & ground_truth[d2]) / len(ground_truth[d1] | ground_truth[d2])
                        for i, d1 in enumerate(tier1_diseases) for d2 in tier1_diseases[i+1:]
                        if len(ground_truth[d1] | ground_truth[d2]) > 0) / max(1, len(tier1_diseases) * (len(tier1_diseases)-1) // 2)

    tier23_jaccard = sum(len(ground_truth[d1] & ground_truth[d2]) / len(ground_truth[d1] | ground_truth[d2])
                         for i, d1 in enumerate(tier23_diseases) for d2 in tier23_diseases[i+1:]
                         if len(ground_truth[d1] | ground_truth[d2]) > 0) / max(1, len(tier23_diseases) * (len(tier23_diseases)-1) // 2)

    if tier1_jaccard > tier23_jaccard * 1.2:
        finding = f"3. DRUG OVERLAP: Tier 1 Jaccard = {tier1_jaccard:.4f} vs Tier 2/3 = {tier23_jaccard:.4f}"
        print(finding)
        findings.append(finding)
    else:
        print(f"3. Drug overlap: Tier 1 = {tier1_jaccard:.4f} vs Tier 2/3 = {tier23_jaccard:.4f} (similar)")

    # Success criteria
    print("\n" + "=" * 70)
    print("SUCCESS CRITERIA CHECK")
    print("=" * 70)

    if len(findings) >= 2:
        print(f"  VALIDATED: Found {len(findings)} structural factors explaining Tier 1 success")
        for f in findings:
            print(f"  - {f}")
        success = True
    else:
        print(f"  INCONCLUSIVE: Only found {len(findings)} distinguishing factors")
        success = False

    # Save results
    output = {
        'tier_distribution': {str(k): len(v) for k, v in tier_diseases.items()},
        'drugs_per_disease': {
            'tier1': float(np.mean([drugs_per_disease[d] for d in tier_diseases[1]])),
            'tier2': float(np.mean([drugs_per_disease[d] for d in tier_diseases[2]])),
            'tier3': float(np.mean([drugs_per_disease[d] for d in tier_diseases[3]])),
        },
        'mechanism_rate': {
            'tier1': float(tier1_mech_rate),
            'tier23': float(tier23_mech_rate),
        },
        'high_freq_ratio': {
            'tier1': float(tier1_ratio),
            'tier23': float(tier23_ratio),
        },
        'drug_jaccard': {
            'tier1': float(tier1_jaccard),
            'tier23': float(tier23_jaccard),
        },
        'key_findings': findings,
        'success': success,
    }

    results_file = ANALYSIS_DIR / "h137_tier1_success_analysis.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
