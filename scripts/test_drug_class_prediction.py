#!/usr/bin/env python3
"""
h171: Drug-Class-Based Prediction for Neurological

Instead of kNN (which predicts wrong drug classes), directly match disease types
to appropriate drug classes.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
from production_predictor import CATEGORY_KEYWORDS

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"

# Drug class definitions for neurological diseases
NEUROLOGICAL_DISEASE_DRUG_CLASSES = {
    # Epilepsy, seizure -> anticonvulsants
    'epilepsy': ['anticonvulsant', 'antiepileptic'],
    'seizure': ['anticonvulsant', 'antiepileptic'],

    # Parkinson's -> dopaminergic agents
    'parkinson': ['dopaminergic', 'levodopa', 'dopamine_agonist'],

    # Alzheimer's, dementia -> cholinesterase inhibitors, NMDA antagonists
    'alzheimer': ['cholinesterase_inhibitor', 'nmda_antagonist'],
    'dementia': ['cholinesterase_inhibitor', 'nmda_antagonist'],

    # Migraine -> triptans, ergots, CGRP inhibitors
    'migraine': ['triptan', 'ergot', 'cgrp_inhibitor'],
    'headache': ['triptan', 'nsaid'],

    # Neuropathy -> gabapentinoids, tricyclics for pain
    'neuropathy': ['gabapentinoid', 'tricyclic', 'topical_analgesic'],
    'neuralgia': ['anticonvulsant', 'tricyclic'],

    # Movement disorders
    'dyskinesia': ['dopaminergic', 'anticholinergic'],
    'dystonia': ['anticholinergic', 'muscle_relaxant'],

    # Sleep disorders
    'narcolepsy': ['stimulant', 'wake_promoting'],
    'insomnia': ['hypnotic', 'sedative'],
}

# Map drug classes to actual drugs
DRUG_CLASS_MEMBERS = {
    'anticonvulsant': [
        'carbamazepine', 'valproic acid', 'phenytoin', 'lamotrigine',
        'topiramate', 'levetiracetam', 'gabapentin', 'pregabalin',
        'oxcarbazepine', 'zonisamide', 'lacosamide', 'perampanel',
        'clobazam', 'clonazepam', 'brivaracetam', 'eslicarbazepine'
    ],
    'antiepileptic': [
        'carbamazepine', 'valproic acid', 'phenytoin', 'lamotrigine',
        'topiramate', 'levetiracetam', 'gabapentin', 'pregabalin'
    ],
    'dopaminergic': [
        'levodopa', 'carbidopa', 'pramipexole', 'ropinirole',
        'bromocriptine', 'apomorphine', 'rotigotine', 'amantadine',
        'entacapone', 'rasagiline', 'selegiline', 'safinamide'
    ],
    'levodopa': ['levodopa', 'carbidopa'],
    'dopamine_agonist': ['pramipexole', 'ropinirole', 'bromocriptine', 'apomorphine', 'rotigotine'],
    'cholinesterase_inhibitor': ['donepezil', 'rivastigmine', 'galantamine'],
    'nmda_antagonist': ['memantine'],
    'triptan': [
        'sumatriptan', 'rizatriptan', 'zolmitriptan', 'eletriptan',
        'naratriptan', 'almotriptan', 'frovatriptan', 'lasmiditan'
    ],
    'ergot': ['ergotamine', 'dihydroergotamine'],
    'cgrp_inhibitor': ['erenumab', 'fremanezumab', 'galcanezumab', 'ubrogepant', 'rimegepant'],
    'gabapentinoid': ['gabapentin', 'pregabalin'],
    'tricyclic': ['amitriptyline', 'nortriptyline', 'desipramine'],
    'topical_analgesic': ['lidocaine', 'capsaicin'],
    'anticholinergic': ['trihexyphenidyl', 'benztropine', 'biperiden'],
    'muscle_relaxant': ['baclofen', 'tizanidine', 'botulinum toxin'],
    'stimulant': ['amphetamine', 'methylphenidate', 'modafinil', 'armodafinil', 'solriamfetol'],
    'wake_promoting': ['modafinil', 'armodafinil', 'pitolisant', 'solriamfetol'],
    'hypnotic': ['zolpidem', 'eszopiclone', 'zaleplon'],
    'sedative': ['diazepam', 'lorazepam', 'temazepam'],
    'nsaid': ['ibuprofen', 'naproxen', 'aspirin', 'celecoxib'],
}


def categorize_disease(disease_name: str) -> str:
    """Categorize disease by keywords."""
    disease_lower = disease_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in disease_lower for kw in keywords):
            return category
    return 'other'


def get_neurological_subtype(disease_name: str) -> List[str]:
    """Get neurological disease subtype for drug class matching."""
    disease_lower = disease_name.lower()
    matching_classes = []

    for disease_key, drug_classes in NEUROLOGICAL_DISEASE_DRUG_CLASSES.items():
        if disease_key in disease_lower:
            matching_classes.extend(drug_classes)

    return list(set(matching_classes))


def get_drugs_for_classes(drug_classes: List[str], name_to_drug_id: Dict[str, str]) -> Dict[str, float]:
    """Get drugs from specified classes with scores."""
    drug_scores = {}

    for drug_class in drug_classes:
        members = DRUG_CLASS_MEMBERS.get(drug_class, [])
        for drug_name in members:
            drug_id = name_to_drug_id.get(drug_name.lower())
            if drug_id:
                # Each matching class adds 1 to score
                drug_scores[drug_id] = drug_scores.get(drug_id, 0) + 1

    return drug_scores


def load_mesh_mappings_from_file() -> Dict[str, str]:
    """Load MESH mappings from file."""
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


def load_drugbank_lookup() -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load DrugBank mappings."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_name_drkg = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}
    return name_to_id, id_to_name_drkg


def load_ground_truth(
    mesh_mappings: Dict[str, str],
    name_to_drug_id: Dict[str, str],
) -> Tuple[Dict[str, Set[str]], Dict[str, str]]:
    """Load ground truth with fuzzy matching."""
    df = pd.read_excel(REFERENCE_DIR / "everycure" / "indicationList.xlsx")
    fuzzy_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(fuzzy_mappings)

    gt_pairs: Dict[str, Set[str]] = defaultdict(set)
    disease_name_to_id: Dict[str, str] = {}

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
        drug_id = name_to_drug_id.get(drug.lower())
        if drug_id:
            gt_pairs[disease_id].add(drug_id)
            if disease_id not in disease_name_to_id:
                disease_name_to_id[disease_id] = disease

    return dict(gt_pairs), disease_name_to_id


def evaluate_recall_at_k(
    predictions: Dict[str, float],
    gt_drugs: Set[str],
    k: int = 30,
) -> float:
    """Compute Recall@k."""
    if not gt_drugs:
        return 0.0
    sorted_drugs = sorted(predictions.items(), key=lambda x: -x[1])
    top_k_drugs = set(d for d, _ in sorted_drugs[:k])
    hits = len(top_k_drugs & gt_drugs)
    return hits / len(gt_drugs)


def run_evaluation():
    """Evaluate drug-class-based prediction for neurological diseases."""
    print("Loading data...")
    mesh_mappings = load_mesh_mappings_from_file()
    name_to_drug_id, drug_id_to_name = load_drugbank_lookup()
    gt, disease_id_to_name = load_ground_truth(mesh_mappings, name_to_drug_id)

    # Find neurological diseases
    neuro_diseases = []
    for disease_id, disease_name in disease_id_to_name.items():
        if categorize_disease(disease_name) == 'neurological':
            neuro_diseases.append((disease_id, disease_name))

    print(f"\nNeurological diseases in GT: {len(neuro_diseases)}")

    # Evaluate each disease
    print("\n" + "="*70)
    print("DRUG-CLASS-BASED PREDICTION FOR NEUROLOGICAL DISEASES")
    print("="*70)

    total_recall = []
    total_coverage = []

    for disease_id, disease_name in neuro_diseases:
        gt_drugs = gt[disease_id]

        # Get matching drug classes
        drug_classes = get_neurological_subtype(disease_name)

        if not drug_classes:
            print(f"\n{disease_name}: No matching drug classes")
            total_recall.append(0)
            total_coverage.append(0)
            continue

        # Get drugs from those classes
        predicted_drugs = get_drugs_for_classes(drug_classes, name_to_drug_id)

        if not predicted_drugs:
            print(f"\n{disease_name}: Drug classes {drug_classes} but no drugs found in DrugBank")
            total_recall.append(0)
            total_coverage.append(0)
            continue

        # Compute recall
        recall = evaluate_recall_at_k(predicted_drugs, gt_drugs, k=30)

        # Compute coverage (what fraction of GT drugs are in our drug class predictions)
        coverage = len(set(predicted_drugs.keys()) & gt_drugs) / len(gt_drugs) if gt_drugs else 0

        total_recall.append(recall)
        total_coverage.append(coverage)

        # Details
        gt_drug_names = [drug_id_to_name.get(d, d) for d in gt_drugs]
        predicted_drug_names = sorted(
            [(drug_id_to_name.get(d, d), s) for d, s in predicted_drugs.items()],
            key=lambda x: -x[1]
        )[:10]

        hits = set(predicted_drugs.keys()) & gt_drugs
        hit_names = [drug_id_to_name.get(d, d) for d in hits]

        print(f"\n{disease_name}:")
        print(f"  Drug classes: {drug_classes}")
        print(f"  GT drugs ({len(gt_drugs)}): {gt_drug_names}")
        print(f"  Predicted drugs ({len(predicted_drugs)}): {[n for n, s in predicted_drug_names]}")
        print(f"  HITS ({len(hits)}): {hit_names}")
        print(f"  Recall@30: {recall*100:.1f}%, Coverage: {coverage*100:.1f}%")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\nDrug-Class-Based Prediction:")
    print(f"  Mean Recall@30: {np.mean(total_recall)*100:.1f}%")
    print(f"  Mean Coverage: {np.mean(total_coverage)*100:.1f}%")
    print(f"  Diseases with hits: {sum(1 for r in total_recall if r > 0)}/{len(total_recall)}")

    print(f"\nBaseline (kNN from h168):")
    print(f"  Mean Coverage: 18.0%")
    print(f"  Zero-coverage: 7/10 diseases")

    improvement = np.mean(total_coverage)*100 - 18.0
    print(f"\nImprovement: {improvement:+.1f} pp coverage")

    # Save results
    results = {
        'diseases': [
            {
                'disease': disease_name,
                'drug_classes': get_neurological_subtype(disease_name),
                'recall': total_recall[i] if i < len(total_recall) else 0,
                'coverage': total_coverage[i] if i < len(total_coverage) else 0
            }
            for i, (disease_id, disease_name) in enumerate(neuro_diseases)
        ],
        'mean_recall': float(np.mean(total_recall)),
        'mean_coverage': float(np.mean(total_coverage)),
        'knn_baseline_coverage': 0.18
    }

    output_path = ANALYSIS_DIR / "drug_class_prediction_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    run_evaluation()
