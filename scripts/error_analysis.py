#!/usr/bin/env python3
"""
Error analysis for drug repurposing predictions.

Analyzes why 52.5% of ground truth drugs are not in top-30 predictions.
Identifies patterns in:
1. Which drug types are missed (biologics, small molecules, etc.)
2. Which disease categories have low recall
3. Feature coverage gaps
4. Systematic biases
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, Set, List, Tuple
from collections import defaultdict, Counter

import numpy as np
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathway_features import PathwayEnrichment
from chemical_features import DrugFingerprinter, compute_tanimoto_similarity
from atc_features import ATCMapper

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"


def classify_drug_type(drug_name: str) -> str:
    """Classify drug by name pattern."""
    name_lower = drug_name.lower()

    # Biologics patterns
    if name_lower.endswith(('mab', 'umab', 'zumab', 'ximab')):
        return 'monoclonal_antibody'
    if name_lower.endswith(('cept', 'ept')):
        return 'fusion_protein'
    if name_lower.endswith(('ase', 'plase', 'kinase')):
        return 'enzyme'
    if name_lower.endswith(('lin', 'sulin')):
        return 'peptide_hormone'

    # Small molecule patterns
    if name_lower.endswith(('ib', 'nib', 'tinib')):
        return 'kinase_inhibitor'
    if name_lower.endswith('pril'):
        return 'ace_inhibitor'
    if name_lower.endswith('sartan'):
        return 'arb'
    if name_lower.endswith('statin'):
        return 'statin'
    if name_lower.endswith(('olol', 'alol')):
        return 'beta_blocker'
    if name_lower.endswith('prazole'):
        return 'ppi'
    if name_lower.endswith('cycline'):
        return 'antibiotic'
    if name_lower.endswith('cillin'):
        return 'penicillin'
    if name_lower.endswith('mycin'):
        return 'macrolide'

    return 'other'


def classify_disease_category(disease_name: str) -> str:
    """Classify disease by name pattern."""
    name_lower = disease_name.lower()

    if any(x in name_lower for x in ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'sarcoma', 'melanoma']):
        return 'cancer'
    if any(x in name_lower for x in ['diabetes', 'metabolic', 'obesity']):
        return 'metabolic'
    if any(x in name_lower for x in ['heart', 'cardiac', 'hypertension', 'coronary', 'arrhythmia']):
        return 'cardiovascular'
    if any(x in name_lower for x in ['alzheimer', 'parkinson', 'dementia', 'neurological', 'epilepsy']):
        return 'neurological'
    if any(x in name_lower for x in ['arthritis', 'lupus', 'autoimmune', 'crohn', 'colitis']):
        return 'autoimmune'
    if any(x in name_lower for x in ['infection', 'bacterial', 'viral', 'hiv', 'hepatitis']):
        return 'infectious'
    if any(x in name_lower for x in ['asthma', 'copd', 'pulmonary', 'respiratory']):
        return 'respiratory'
    if any(x in name_lower for x in ['depression', 'anxiety', 'schizophrenia', 'bipolar', 'psychiatric']):
        return 'psychiatric'

    return 'other'


def main() -> None:
    print("=" * 70)
    print("ERROR ANALYSIS")
    print("=" * 70)

    # Load all resources
    print("\n1. Loading resources...")

    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    checkpoint = torch.load(MODELS_DIR / "transe.pt", map_location="cpu", weights_only=False)
    embeddings = None
    if "entity_embeddings" in checkpoint:
        embeddings = checkpoint["entity_embeddings"].numpy()
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        for key in ["entity_embeddings.weight", "ent_embeddings.weight"]:
            if key in state:
                embeddings = state[key].numpy()
                break
    entity2id = checkpoint.get("entity2id", {})

    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets = {k: set(v) for k, v in json.load(f).items()}

    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes = {k: set(v) for k, v in json.load(f).items()}

    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)
    mesh_mappings = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id and str(mesh_id).startswith(("D", "C")):
                    mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_id}"

    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()}
    id_to_drug_name = {f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()}

    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt_raw = json.load(f)

    # Load feature modules
    pe = PathwayEnrichment()
    fingerprinter = DrugFingerprinter(use_cache=True)
    atc_mapper = ATCMapper()

    # Build drug embeddings lookup
    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    valid_drug_ids = []
    valid_drug_indices = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)
    drug_embs = embeddings[valid_drug_indices]
    drug_id_to_local_idx = {did: i for i, did in enumerate(valid_drug_ids)}

    print(f"   Drugs with embeddings: {len(valid_drug_ids)}")

    # Analyze each disease
    print("\n2. Running predictions and collecting errors...")

    # Stats collectors
    disease_stats = defaultdict(lambda: {'hits': 0, 'total': 0, 'drugs': []})
    drug_stats = defaultdict(lambda: {'hits': 0, 'total': 0, 'diseases': []})
    drug_type_stats = defaultdict(lambda: {'hits': 0, 'total': 0})
    disease_cat_stats = defaultdict(lambda: {'hits': 0, 'total': 0})

    # Feature coverage stats
    missed_drugs_no_fp = 0
    missed_drugs_no_targets = 0
    missed_drugs_no_atc = 0
    hit_drugs_with_fp = 0
    hit_drugs_with_targets = 0

    # Rank distribution
    missed_drug_ranks = []
    hit_drug_ranks = []

    diseases_evaluated = 0

    for disease_name, disease_data in tqdm(gt_raw.items(), desc="Diseases"):
        mesh_id = mesh_mappings.get(disease_name.lower())
        if not mesh_id:
            continue

        disease_idx = entity2id.get(mesh_id)
        if disease_idx is None:
            continue

        mesh_short = mesh_id.split("MESH:")[-1]
        disease_emb = embeddings[disease_idx]
        dis_genes = disease_genes.get(f"MESH:{mesh_short}", set())
        disease_category = classify_disease_category(disease_name)

        # Get GT drugs
        gt_drug_ids = set()
        gt_drug_names = []
        for drug_info in disease_data.get('drugs', []):
            drug_name = drug_info['name']
            drug_id = name_to_id.get(drug_name.lower())
            if drug_id and drug_id in drug_id_to_local_idx:
                gt_drug_ids.add(drug_id)
                gt_drug_names.append(drug_name)

        if not gt_drug_ids:
            continue

        diseases_evaluated += 1

        # Get GT fingerprints
        gt_fps = []
        for drug_name in gt_drug_names:
            fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
            if fp is not None:
                gt_fps.append(fp)

        # Score all drugs with quad boost
        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))
        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        base_features = np.hstack([concat_feats, product_feats, diff_feats])
        base_scores = model.predict_proba(base_features)[:, 1]

        # Apply quad boost
        boosted_scores = np.zeros(n_drugs)
        for i, drug_id in enumerate(valid_drug_ids):
            db_id = drug_id.split("::")[-1]
            drug_name = id_to_drug_name.get(drug_id, "")

            # Target overlap
            drug_genes = drug_targets.get(db_id, set())
            overlap = len(drug_genes & dis_genes)

            # ATC score
            atc_score = atc_mapper.get_mechanism_score(drug_name, disease_name)

            # Pathway overlap
            po, _, _ = pe.get_pathway_overlap(db_id, f"MESH:{mesh_short}")

            # Chemical similarity
            chem_sim = 0.0
            query_fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
            if query_fp is not None and gt_fps:
                for gt_fp in gt_fps:
                    sim = compute_tanimoto_similarity(query_fp, gt_fp)
                    chem_sim = max(chem_sim, sim)

            # Quad additive boost
            boost = 1 + 0.01 * min(overlap, 10) + 0.05 * atc_score + 0.01 * min(po, 10)
            if chem_sim > 0.7:
                boost *= 1.2
            boosted_scores[i] = base_scores[i] * boost

        # Get rankings
        rankings = np.argsort(boosted_scores)[::-1]
        drug_id_to_rank = {valid_drug_ids[idx]: rank for rank, idx in enumerate(rankings)}

        top_30_ids = set(valid_drug_ids[idx] for idx in rankings[:30])

        # Analyze each GT drug
        for drug_id in gt_drug_ids:
            drug_name = id_to_drug_name.get(drug_id, "")
            drug_type = classify_drug_type(drug_name)
            db_id = drug_id.split("::")[-1]
            rank = drug_id_to_rank.get(drug_id, 99999)

            is_hit = drug_id in top_30_ids

            # Update stats
            drug_stats[drug_name]['total'] += 1
            drug_stats[drug_name]['diseases'].append(disease_name)
            drug_type_stats[drug_type]['total'] += 1
            disease_cat_stats[disease_category]['total'] += 1
            disease_stats[disease_name]['total'] += 1
            disease_stats[disease_name]['drugs'].append(drug_name)

            if is_hit:
                drug_stats[drug_name]['hits'] += 1
                drug_type_stats[drug_type]['hits'] += 1
                disease_cat_stats[disease_category]['hits'] += 1
                disease_stats[disease_name]['hits'] += 1
                hit_drug_ranks.append(rank)

                # Feature coverage for hits
                if fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False) is not None:
                    hit_drugs_with_fp += 1
                if drug_targets.get(db_id):
                    hit_drugs_with_targets += 1
            else:
                missed_drug_ranks.append(rank)

                # Feature coverage for misses
                if fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False) is None:
                    missed_drugs_no_fp += 1
                if not drug_targets.get(db_id):
                    missed_drugs_no_targets += 1
                if not atc_mapper.get_atc_codes(drug_name):
                    missed_drugs_no_atc += 1

    # Print results
    print(f"\nDiseases evaluated: {diseases_evaluated}")

    total_gt = sum(d['total'] for d in drug_type_stats.values())
    total_hits = sum(d['hits'] for d in drug_type_stats.values())
    print(f"Total GT drugs: {total_gt}")
    print(f"Total hits: {total_hits} ({total_hits/total_gt:.1%})")
    print(f"Total misses: {total_gt - total_hits} ({(total_gt-total_hits)/total_gt:.1%})")

    # Drug type analysis
    print("\n" + "=" * 70)
    print("DRUG TYPE ANALYSIS")
    print("=" * 70)
    print(f"\n{'Drug Type':<25} {'Hits':<10} {'Total':<10} {'Recall':<10}")
    print("-" * 55)

    for drug_type in sorted(drug_type_stats.keys(), key=lambda x: drug_type_stats[x]['total'], reverse=True):
        stats = drug_type_stats[drug_type]
        recall = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{drug_type:<25} {stats['hits']:<10} {stats['total']:<10} {recall:.1%}")

    # Disease category analysis
    print("\n" + "=" * 70)
    print("DISEASE CATEGORY ANALYSIS")
    print("=" * 70)
    print(f"\n{'Disease Category':<25} {'Hits':<10} {'Total':<10} {'Recall':<10}")
    print("-" * 55)

    for cat in sorted(disease_cat_stats.keys(), key=lambda x: disease_cat_stats[x]['total'], reverse=True):
        stats = disease_cat_stats[cat]
        recall = stats['hits'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{cat:<25} {stats['hits']:<10} {stats['total']:<10} {recall:.1%}")

    # Feature coverage for misses
    print("\n" + "=" * 70)
    print("FEATURE COVERAGE (MISSES)")
    print("=" * 70)
    total_misses = total_gt - total_hits
    print(f"\nMissed drugs without fingerprint: {missed_drugs_no_fp} ({missed_drugs_no_fp/total_misses:.1%})")
    print(f"Missed drugs without targets: {missed_drugs_no_targets} ({missed_drugs_no_targets/total_misses:.1%})")
    print(f"Missed drugs without ATC: {missed_drugs_no_atc} ({missed_drugs_no_atc/total_misses:.1%})")

    # Rank distribution
    print("\n" + "=" * 70)
    print("RANK DISTRIBUTION")
    print("=" * 70)
    if missed_drug_ranks:
        print(f"\nMissed drugs (should be in top 30 but aren't):")
        print(f"  Median rank: {np.median(missed_drug_ranks):.0f}")
        print(f"  Mean rank: {np.mean(missed_drug_ranks):.0f}")
        print(f"  Top 100: {sum(1 for r in missed_drug_ranks if r < 100)} ({sum(1 for r in missed_drug_ranks if r < 100)/len(missed_drug_ranks):.1%})")
        print(f"  Top 500: {sum(1 for r in missed_drug_ranks if r < 500)} ({sum(1 for r in missed_drug_ranks if r < 500)/len(missed_drug_ranks):.1%})")
        print(f"  Top 1000: {sum(1 for r in missed_drug_ranks if r < 1000)} ({sum(1 for r in missed_drug_ranks if r < 1000)/len(missed_drug_ranks):.1%})")

    # Worst performing diseases
    print("\n" + "=" * 70)
    print("WORST PERFORMING DISEASES (0% Recall, >5 GT drugs)")
    print("=" * 70)

    zero_recall_diseases = [
        (name, stats) for name, stats in disease_stats.items()
        if stats['hits'] == 0 and stats['total'] >= 5
    ]
    zero_recall_diseases.sort(key=lambda x: x[1]['total'], reverse=True)

    print(f"\n{'Disease':<50} {'GT Drugs':<10}")
    print("-" * 60)
    for name, stats in zero_recall_diseases[:15]:
        print(f"{name[:48]:<50} {stats['total']:<10}")

    # Best performing diseases
    print("\n" + "=" * 70)
    print("BEST PERFORMING DISEASES (100% Recall, >5 GT drugs)")
    print("=" * 70)

    perfect_diseases = [
        (name, stats) for name, stats in disease_stats.items()
        if stats['hits'] == stats['total'] and stats['total'] >= 5
    ]
    perfect_diseases.sort(key=lambda x: x[1]['total'], reverse=True)

    print(f"\n{'Disease':<50} {'GT Drugs':<10}")
    print("-" * 60)
    for name, stats in perfect_diseases[:15]:
        print(f"{name[:48]:<50} {stats['total']:<10}")

    # Drugs that are always missed
    print("\n" + "=" * 70)
    print("DRUGS THAT ARE ALWAYS MISSED (≥5 diseases)")
    print("=" * 70)

    always_missed = [
        (name, stats) for name, stats in drug_stats.items()
        if stats['hits'] == 0 and stats['total'] >= 5
    ]
    always_missed.sort(key=lambda x: x[1]['total'], reverse=True)

    print(f"\n{'Drug':<40} {'Diseases':<10} {'Type':<20}")
    print("-" * 70)
    for name, stats in always_missed[:20]:
        drug_type = classify_drug_type(name)
        print(f"{name[:38]:<40} {stats['total']:<10} {drug_type:<20}")


if __name__ == "__main__":
    main()
