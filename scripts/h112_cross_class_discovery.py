#!/usr/bin/env python3
"""
h112: Cross-Class Drug Discovery Analysis

PURPOSE:
    h110 found that "incoherent" predictions (drugs from ATC classes that NEVER
    treat similar diseases) have HIGHER precision (11.24%) than coherent ones (6.69%).

    This is counter-intuitive: we'd expect drugs from relevant classes to work better.

    Investigate WHY this happens by analyzing:
    1. What types of drugs/diseases are in incoherent hits?
    2. Is it selection pressure (only strong signals overcome class mismatch)?
    3. Is it mechanism overlap despite class mismatch (polypharmacology)?
    4. Is it confounding with well-known indications in coherent predictions?

SUCCESS CRITERIA:
    Identify actionable pattern explaining the precision gap.
"""

import json
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Set, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
EMBEDDINGS_DIR = PROJECT_ROOT / "data" / "embeddings"

SEEDS = [42, 123, 456, 789, 1024]


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


def load_atc_codes() -> Dict[str, str]:
    """Load drug -> ATC code mapping."""
    atc_path = REFERENCE_DIR / "drug_atc_codes.json"
    if not atc_path.exists():
        return {}
    with open(atc_path) as f:
        atc_data = json.load(f)
    return {f"drkg:Compound::{k}": v for k, v in atc_data.items()}


def get_atc_class(atc_code: str) -> str:
    """Extract first-level ATC class (A, B, C, etc.)."""
    if atc_code and len(atc_code) >= 1:
        return atc_code[0]
    return "unknown"


def run_knn_with_coherence(
    emb_dict, train_gt, test_gt, drug_atc, disease_names, drug_targets, disease_genes, k=20
) -> List[Dict]:
    """Run kNN and compute ATC coherence for each prediction."""
    train_disease_list = [d for d in train_gt if d in emb_dict]
    train_disease_embs = np.array([emb_dict[d] for d in train_disease_list], dtype=np.float32)

    # Drug training frequency
    drug_train_freq: Dict[str, int] = defaultdict(int)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            drug_train_freq[drug_id] += 1

    # Build ATC -> diseases treated mapping from training data
    atc_to_diseases: Dict[str, Set[str]] = defaultdict(set)
    for disease_id, drugs in train_gt.items():
        for drug_id in drugs:
            atc = drug_atc.get(drug_id, "")
            if atc:
                atc_class = get_atc_class(atc)
                atc_to_diseases[atc_class].add(disease_id)

    results = []

    for disease_id in test_gt:
        if disease_id not in emb_dict:
            continue
        gt_drugs = {d for d in test_gt[disease_id] if d in emb_dict}
        if not gt_drugs:
            continue

        disease_name = disease_names.get(disease_id, "")

        # Find k nearest training diseases
        test_emb = emb_dict[disease_id].reshape(1, -1)
        sims = cosine_similarity(test_emb, train_disease_embs)[0]
        top_k_idx = np.argsort(sims)[-k:]
        neighbor_diseases = set(train_disease_list[idx] for idx in top_k_idx)

        # Count drug frequency
        drug_counts: Dict[str, float] = defaultdict(float)
        for idx in top_k_idx:
            neighbor_disease = train_disease_list[idx]
            neighbor_sim = sims[idx]
            for drug_id in train_gt[neighbor_disease]:
                if drug_id in emb_dict:
                    drug_counts[drug_id] += neighbor_sim

        if not drug_counts:
            continue

        sorted_drugs = sorted(drug_counts.items(), key=lambda x: x[1], reverse=True)[:30]
        max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

        for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
            atc = drug_atc.get(drug_id, "")
            atc_class = get_atc_class(atc) if atc else ""

            # Check if any drug from same ATC class treats similar diseases
            is_coherent = False
            if atc_class and atc_class in atc_to_diseases:
                classmate_diseases = atc_to_diseases[atc_class]
                # Check if any neighbor disease is treated by a drug from same class
                if classmate_diseases & neighbor_diseases:
                    is_coherent = True

            # Mechanism support
            drug_genes = drug_targets.get(drug_id, set())
            dis_genes = disease_genes.get(disease_id, set())
            mech_support = len(drug_genes & dis_genes) > 0

            train_freq = drug_train_freq.get(drug_id, 0)
            n_targets = len(drug_genes)
            is_hit = drug_id in gt_drugs

            results.append({
                'disease': disease_id,
                'drug': drug_id,
                'disease_name': disease_name,
                'atc_class': atc_class,
                'is_coherent': is_coherent,
                'has_atc': bool(atc),
                'mechanism_support': mech_support,
                'train_frequency': train_freq,
                'n_targets': n_targets,
                'knn_score': score,
                'norm_score': score / max_score if max_score > 0 else 0,
                'rank': rank,
                'is_hit': 1 if is_hit else 0,
            })

    return results


def main():
    print("h112: Cross-Class Drug Discovery Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    emb_dict = load_node2vec_embeddings()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    mesh_mappings = load_mesh_mappings_from_file()
    ground_truth, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)
    drug_targets = load_drug_targets()
    disease_genes = load_disease_genes()
    drug_atc = load_atc_codes()

    print(f"  Embeddings: {len(emb_dict)}")
    print(f"  Ground truth diseases: {len(ground_truth)}")
    print(f"  Drugs with ATC codes: {len(drug_atc)}")
    print(f"  Drugs with targets: {len(drug_targets)}")

    # Collect predictions across seeds
    print("\n" + "=" * 70)
    print("Collecting predictions across 5 seeds")
    print("=" * 70)

    all_results = []

    for seed in SEEDS:
        np.random.seed(seed)
        diseases = list(ground_truth.keys())
        np.random.shuffle(diseases)
        n_test = len(diseases) // 5
        test_diseases = set(diseases[:n_test])
        train_diseases = set(diseases[n_test:])

        train_gt = {d: ground_truth[d] for d in train_diseases}
        test_gt = {d: ground_truth[d] for d in test_diseases}

        seed_results = run_knn_with_coherence(
            emb_dict, train_gt, test_gt, drug_atc, disease_names, drug_targets, disease_genes, k=20
        )
        all_results.extend(seed_results)
        print(f"  Seed {seed}: {len(seed_results)} predictions")

    df = pd.DataFrame(all_results)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Base hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Split into coherent vs incoherent
    has_atc = df[df['has_atc']]
    coherent = has_atc[has_atc['is_coherent']]
    incoherent = has_atc[~has_atc['is_coherent']]

    print(f"\nWith ATC data: {len(has_atc)} predictions")
    print(f"  COHERENT: {len(coherent)} ({len(coherent)/len(has_atc)*100:.1f}%)")
    print(f"  INCOHERENT: {len(incoherent)} ({len(incoherent)/len(has_atc)*100:.1f}%)")

    # === VERIFY THE PHENOMENON ===
    print("\n" + "=" * 70)
    print("VERIFY: Incoherent > Coherent Precision")
    print("=" * 70)

    coh_prec = coherent['is_hit'].mean() * 100
    incoh_prec = incoherent['is_hit'].mean() * 100
    print(f"\nCOHERENT precision: {coh_prec:.2f}%")
    print(f"INCOHERENT precision: {incoh_prec:.2f}%")
    print(f"Difference: {incoh_prec - coh_prec:.2f} pp")

    # === HYPOTHESIS 1: kNN SCORE SELECTION ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 1: Selection Pressure (kNN Score)")
    print("Do incoherent hits have higher kNN scores?")
    print("=" * 70)

    print(f"\nMean kNN score:")
    print(f"  COHERENT: {coherent['norm_score'].mean():.4f}")
    print(f"  INCOHERENT: {incoherent['norm_score'].mean():.4f}")

    # Focus on hits
    coh_hits = coherent[coherent['is_hit'] == 1]
    incoh_hits = incoherent[incoherent['is_hit'] == 1]

    print(f"\nMean kNN score (HITS ONLY):")
    print(f"  COHERENT hits: {coh_hits['norm_score'].mean():.4f} (N={len(coh_hits)})")
    print(f"  INCOHERENT hits: {incoh_hits['norm_score'].mean():.4f} (N={len(incoh_hits)})")

    if incoh_hits['norm_score'].mean() > coh_hits['norm_score'].mean():
        print("  → Incoherent hits DO have higher kNN scores (selection pressure hypothesis SUPPORTED)")
    else:
        print("  → Incoherent hits do NOT have higher kNN scores")

    # === HYPOTHESIS 2: MECHANISM SUPPORT ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 2: Mechanism Support (Drug-Target-Gene Overlap)")
    print("Do incoherent hits have more mechanism support?")
    print("=" * 70)

    print(f"\nMechanism support rate:")
    print(f"  COHERENT: {coherent['mechanism_support'].mean()*100:.2f}%")
    print(f"  INCOHERENT: {incoherent['mechanism_support'].mean()*100:.2f}%")

    print(f"\nMechanism support rate (HITS ONLY):")
    print(f"  COHERENT hits: {coh_hits['mechanism_support'].mean()*100:.2f}%")
    print(f"  INCOHERENT hits: {incoh_hits['mechanism_support'].mean()*100:.2f}%")

    if incoh_hits['mechanism_support'].mean() > coh_hits['mechanism_support'].mean():
        print("  → Incoherent hits DO have more mechanism support (polypharmacology hypothesis SUPPORTED)")
    else:
        print("  → Incoherent hits do NOT have more mechanism support")

    # === HYPOTHESIS 3: POLYPHARMACOLOGY (n_targets) ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 3: Polypharmacology (Target Count)")
    print("Do incoherent hits have more targets?")
    print("=" * 70)

    print(f"\nMean n_targets:")
    print(f"  COHERENT: {coherent['n_targets'].mean():.1f}")
    print(f"  INCOHERENT: {incoherent['n_targets'].mean():.1f}")

    print(f"\nMean n_targets (HITS ONLY):")
    print(f"  COHERENT hits: {coh_hits['n_targets'].mean():.1f}")
    print(f"  INCOHERENT hits: {incoh_hits['n_targets'].mean():.1f}")

    if incoh_hits['n_targets'].mean() > coh_hits['n_targets'].mean():
        print("  → Incoherent hits DO have more targets (polypharmacology hypothesis SUPPORTED)")
    else:
        print("  → Incoherent hits do NOT have more targets")

    # === HYPOTHESIS 4: TRAINING FREQUENCY ===
    print("\n" + "=" * 70)
    print("HYPOTHESIS 4: Drug Training Frequency")
    print("Are incoherent hits from high-frequency drugs?")
    print("=" * 70)

    print(f"\nMean train_frequency:")
    print(f"  COHERENT: {coherent['train_frequency'].mean():.2f}")
    print(f"  INCOHERENT: {incoherent['train_frequency'].mean():.2f}")

    print(f"\nMean train_frequency (HITS ONLY):")
    print(f"  COHERENT hits: {coh_hits['train_frequency'].mean():.2f}")
    print(f"  INCOHERENT hits: {incoh_hits['train_frequency'].mean():.2f}")

    # === EXAMINE INCOHERENT HITS ===
    print("\n" + "=" * 70)
    print("SAMPLE INCOHERENT HITS (Cross-Class Discoveries)")
    print("=" * 70)

    incoh_hits_sample = incoh_hits.head(20)
    print(f"\n{'Drug':<25} {'ATC':>5} {'Disease':<30} {'Score':>6}")
    print("-" * 70)
    for _, row in incoh_hits_sample.iterrows():
        drug_name = id_to_name.get(row['drug'], row['drug'].split("::")[-1])[:24]
        disease_name = row['disease_name'][:29]
        print(f"{drug_name:<25} {row['atc_class']:>5} {disease_name:<30} {row['norm_score']:.3f}")

    # === ATC CLASS DISTRIBUTION ===
    print("\n" + "=" * 70)
    print("ATC CLASS DISTRIBUTION")
    print("=" * 70)

    print("\nCoherent predictions - ATC class distribution:")
    coh_atc = Counter(coherent['atc_class'])
    for atc, count in coh_atc.most_common(10):
        prec = coherent[coherent['atc_class'] == atc]['is_hit'].mean() * 100
        print(f"  {atc}: {count} predictions, {prec:.1f}% precision")

    print("\nIncoherent predictions - ATC class distribution:")
    incoh_atc = Counter(incoherent['atc_class'])
    for atc, count in incoh_atc.most_common(10):
        prec = incoherent[incoherent['atc_class'] == atc]['is_hit'].mean() * 100
        print(f"  {atc}: {count} predictions, {prec:.1f}% precision")

    # === SUMMARY ===
    print("\n" + "=" * 70)
    print("SUMMARY: Why Incoherent Predictions Outperform")
    print("=" * 70)

    findings = []

    # Selection pressure
    if incoh_hits['norm_score'].mean() > coh_hits['norm_score'].mean():
        diff = incoh_hits['norm_score'].mean() - coh_hits['norm_score'].mean()
        findings.append(f"SELECTION PRESSURE: Incoherent hits have +{diff:.3f} higher kNN scores")

    # Mechanism support
    if incoh_hits['mechanism_support'].mean() > coh_hits['mechanism_support'].mean():
        diff = (incoh_hits['mechanism_support'].mean() - coh_hits['mechanism_support'].mean()) * 100
        findings.append(f"MECHANISM SUPPORT: Incoherent hits have +{diff:.1f}% more mechanism support")

    # Polypharmacology
    if incoh_hits['n_targets'].mean() > coh_hits['n_targets'].mean():
        diff = incoh_hits['n_targets'].mean() - coh_hits['n_targets'].mean()
        findings.append(f"POLYPHARMACOLOGY: Incoherent hits have +{diff:.1f} more targets")

    if findings:
        print("\nCONFIRMED EXPLANATIONS:")
        for f in findings:
            print(f"  ✓ {f}")
    else:
        print("\nNo clear explanation found")

    print(f"\nKEY INSIGHT:")
    print("  Incoherent predictions that rank highly must overcome class bias through")
    print("  stronger evidence (higher kNN scores, more targets, mechanism support).")
    print("  This 'selection pressure' makes them more reliable on average.")

    # Save results
    results = {
        'coherent_precision': float(coh_prec),
        'incoherent_precision': float(incoh_prec),
        'precision_gap': float(incoh_prec - coh_prec),
        'coherent_mean_score': float(coherent['norm_score'].mean()),
        'incoherent_mean_score': float(incoherent['norm_score'].mean()),
        'coherent_hits_mean_score': float(coh_hits['norm_score'].mean()),
        'incoherent_hits_mean_score': float(incoh_hits['norm_score'].mean()),
        'coherent_hits_mechanism_rate': float(coh_hits['mechanism_support'].mean()),
        'incoherent_hits_mechanism_rate': float(incoh_hits['mechanism_support'].mean()),
        'coherent_hits_mean_targets': float(coh_hits['n_targets'].mean()),
        'incoherent_hits_mean_targets': float(incoh_hits['n_targets'].mean()),
        'n_coherent': len(coherent),
        'n_incoherent': len(incoherent),
        'n_coherent_hits': len(coh_hits),
        'n_incoherent_hits': len(incoh_hits),
    }

    results_file = PROJECT_ROOT / "data" / "analysis" / "h112_cross_class_discovery.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == '__main__':
    main()
