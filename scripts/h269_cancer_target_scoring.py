#!/usr/bin/env python3
"""
h269: Cancer-Specific Target-Based Scoring

Hypothesis: Cancer drugs work by targeting specific cancer genes (oncogenes, tumor suppressors).
We can improve cancer predictions by scoring drug-cancer pairs by target-gene overlap.

Steps:
1. Extract drug targets for cancer drugs
2. Build cancer gene lists (oncogenes, tumor suppressors, cancer pathways)
3. Score drug-cancer pairs by target overlap with cancer genes
4. Evaluate on held-out cancer diseases
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def load_cancer_genes():
    """Load comprehensive cancer gene list from multiple sources."""

    # Known oncogenes and tumor suppressors (curated from COSMIC/literature)
    oncogenes = {
        # Classic oncogenes
        'KRAS', 'NRAS', 'HRAS', 'BRAF', 'RAF1', 'ARAF',
        'PIK3CA', 'PIK3CB', 'AKT1', 'AKT2', 'MTOR',
        'MYC', 'MYCN', 'MYCL',
        'EGFR', 'ERBB2', 'ERBB3', 'MET', 'ALK', 'ROS1', 'RET', 'NTRK1', 'NTRK2', 'NTRK3',
        'FGFR1', 'FGFR2', 'FGFR3', 'FGFR4', 'PDGFRA', 'PDGFRB', 'KIT', 'FLT3',
        'ABL1', 'BCR', 'JAK1', 'JAK2', 'JAK3',
        'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CDK4', 'CDK6',
        'MDM2', 'MDM4',
        'BCL2', 'BCL2L1', 'MCL1',
        'NOTCH1', 'NOTCH2', 'NOTCH3',
        'WNT1', 'WNT2', 'WNT3', 'CTNNB1',
        'SMO', 'GLI1', 'GLI2',
        'TERT', 'TERC',
        'IDH1', 'IDH2',
        'SF3B1', 'U2AF1', 'SRSF2',
        'EZH2', 'NSD1', 'NSD2',
        # Fusion partners
        'ETV6', 'EWSR1', 'FUS', 'SS18', 'TMPRSS2',
    }

    tumor_suppressors = {
        # Classic tumor suppressors
        'TP53', 'RB1', 'CDKN2A', 'CDKN2B', 'CDKN1A', 'CDKN1B',
        'PTEN', 'NF1', 'NF2', 'TSC1', 'TSC2',
        'APC', 'AXIN1', 'AXIN2',
        'BRCA1', 'BRCA2', 'PALB2', 'BRIP1', 'RAD51C', 'RAD51D',
        'ATM', 'ATR', 'CHEK1', 'CHEK2', 'NBN',
        'MLH1', 'MSH2', 'MSH6', 'PMS2', 'MUTYH',
        'VHL', 'SDHB', 'SDHC', 'SDHD', 'FH',
        'WT1', 'SMARCB1', 'SMARCA4',
        'BAP1', 'ARID1A', 'ARID1B', 'ARID2',
        'KDM6A', 'KDM5C',
        'PTCH1', 'SUFU',
        'STK11', 'KEAP1',
        'FBXW7', 'CREBBP', 'EP300',
        'RUNX1', 'CEBPA', 'GATA3',
        'MEN1', 'RET', 'SDHA',
        'POLE', 'POLD1',
        # DNA repair
        'XRCC1', 'XRCC2', 'XRCC3', 'XRCC4', 'XRCC5', 'XRCC6',
        'ERCC1', 'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5',
        'FANCD2', 'FANCA', 'FANCB', 'FANCC', 'FANCE', 'FANCF', 'FANCG',
    }

    # Cell cycle genes
    cell_cycle = {
        'CCNA1', 'CCNA2', 'CCNB1', 'CCNB2', 'CCND1', 'CCND2', 'CCND3', 'CCNE1', 'CCNE2',
        'CDK1', 'CDK2', 'CDK4', 'CDK6', 'CDK7', 'CDK9',
        'CDC25A', 'CDC25B', 'CDC25C', 'CDC20', 'CDC6',
        'PLK1', 'PLK4', 'AURKA', 'AURKB',
        'BUB1', 'BUB1B', 'BUB3', 'MAD1L1', 'MAD2L1',
        'E2F1', 'E2F2', 'E2F3', 'E2F4', 'E2F5',
    }

    # Apoptosis genes
    apoptosis = {
        'BCL2', 'BCL2L1', 'BCL2L2', 'MCL1', 'BCL2A1',
        'BAX', 'BAK1', 'BOK', 'BID', 'BAD', 'BIK', 'BIM', 'PUMA', 'NOXA',
        'CASP3', 'CASP7', 'CASP8', 'CASP9', 'CASP10',
        'APAF1', 'CYCS', 'DIABLO',
        'XIAP', 'BIRC2', 'BIRC3', 'BIRC5',
        'FAS', 'FASLG', 'TNFRSF10A', 'TNFRSF10B',
    }

    # Immune checkpoint / immunotherapy targets
    immune = {
        'CD274', 'PDCD1', 'PDCD1LG2', 'CTLA4',
        'CD47', 'SIRPA',
        'LAG3', 'HAVCR2', 'TIGIT', 'BTLA',
        'CD19', 'CD20', 'CD22', 'CD38', 'BCMA',
        'HER2', 'EGFR', 'VEGFA', 'VEGFR1', 'VEGFR2',
    }

    # Angiogenesis
    angiogenesis = {
        'VEGFA', 'VEGFB', 'VEGFC', 'VEGFD', 'PGF',
        'KDR', 'FLT1', 'FLT4',
        'ANGPT1', 'ANGPT2', 'TIE1', 'TEK',
        'PDGFA', 'PDGFB', 'PDGFC', 'PDGFD', 'PDGFRA', 'PDGFRB',
    }

    # All cancer genes combined
    all_cancer_genes = oncogenes | tumor_suppressors | cell_cycle | apoptosis | immune | angiogenesis

    return {
        'oncogenes': oncogenes,
        'tumor_suppressors': tumor_suppressors,
        'cell_cycle': cell_cycle,
        'apoptosis': apoptosis,
        'immune': immune,
        'angiogenesis': angiogenesis,
        'all': all_cancer_genes
    }


def map_gene_symbols_to_ids(gene_symbols, gene_lookup_path='data/reference/gene_lookup.json'):
    """Map gene symbols to NCBI gene IDs."""
    with open(gene_lookup_path) as f:
        gene_lookup = json.load(f)

    symbol_to_id = {}
    for gene_id, info in gene_lookup.items():
        if isinstance(info, dict):
            symbol = info.get('symbol', '')
            symbol_to_id[symbol] = gene_id

    gene_ids = set()
    found_symbols = set()
    for symbol in gene_symbols:
        if symbol in symbol_to_id:
            gene_ids.add(symbol_to_id[symbol])
            found_symbols.add(symbol)

    return gene_ids, found_symbols


def load_cancer_gt():
    """Load cancer-related ground truth from Every Cure data."""
    gt = pd.read_excel('data/reference/everycure/indicationList.xlsx')
    disease_col = 'final normalized disease label'
    drug_col = 'final normalized drug label'

    cancer_terms = ['cancer', 'carcinoma', 'melanoma', 'leukemia', 'lymphoma',
                    'myeloma', 'sarcoma', 'tumor', 'neoplasm', 'glioma', 'blastoma']
    cancer_mask = gt[disease_col].str.lower().str.contains('|'.join(cancer_terms), na=False)
    cancer_gt = gt[cancer_mask]

    # Create disease -> drugs mapping
    disease_to_drugs = defaultdict(set)
    for _, row in cancer_gt.iterrows():
        disease = row[disease_col]
        drug = row[drug_col]
        if pd.notna(disease) and pd.notna(drug):
            disease_to_drugs[disease].add(drug)

    return disease_to_drugs


def load_drug_targets():
    """Load drug targets and create name-to-target mapping."""
    with open('data/reference/drug_targets.json') as f:
        drug_targets = json.load(f)

    with open('data/reference/drugbank_lookup.json') as f:
        drugbank = json.load(f)

    # Create name -> DrugBank ID mapping
    name_to_db = {}
    for db_id, info in drugbank.items():
        name = info if isinstance(info, str) else info.get('name', '')
        if name:
            name_to_db[name.lower()] = db_id

    return drug_targets, name_to_db


def score_drug_by_cancer_targets(drug_name, drug_targets, name_to_db, cancer_gene_ids):
    """Score a drug by how many of its targets are cancer genes."""
    db_id = name_to_db.get(drug_name.lower())
    if not db_id or db_id not in drug_targets:
        return 0, 0, set()

    targets = set(drug_targets[db_id])
    cancer_targets = targets & cancer_gene_ids

    total_targets = len(targets)
    cancer_target_count = len(cancer_targets)

    return cancer_target_count, total_targets, cancer_targets


def evaluate_cancer_target_scoring(seed=42, min_drugs=3, top_k=30):
    """Evaluate cancer-specific target-based scoring using disease holdout."""
    np.random.seed(seed)

    # Load data
    print("Loading cancer ground truth...")
    cancer_disease_drugs = load_cancer_gt()

    print("Loading drug targets...")
    drug_targets, name_to_db = load_drug_targets()

    print("Loading cancer gene lists...")
    cancer_genes = load_cancer_genes()
    cancer_gene_ids, found_symbols = map_gene_symbols_to_ids(cancer_genes['all'])
    print(f"  Mapped {len(found_symbols)}/{len(cancer_genes['all'])} cancer genes to IDs")

    # Filter to evaluable diseases (>= min_drugs drugs)
    evaluable_diseases = {d: drugs for d, drugs in cancer_disease_drugs.items()
                          if len(drugs) >= min_drugs}
    print(f"\nEvaluable cancer diseases (>= {min_drugs} drugs): {len(evaluable_diseases)}")

    # 80/20 train/test split by disease
    disease_list = list(evaluable_diseases.keys())
    np.random.shuffle(disease_list)
    split_idx = int(0.8 * len(disease_list))
    train_diseases = set(disease_list[:split_idx])
    test_diseases = set(disease_list[split_idx:])

    print(f"Train diseases: {len(train_diseases)}, Test diseases: {len(test_diseases)}")

    # Get all drugs from train diseases (for ranking)
    train_drugs = set()
    for disease in train_diseases:
        train_drugs.update(evaluable_diseases[disease])

    # Get drugs with target data
    all_cancer_drugs = set()
    for drugs in evaluable_diseases.values():
        all_cancer_drugs.update(drugs)

    drugs_with_targets = set()
    for drug in all_cancer_drugs:
        db_id = name_to_db.get(drug.lower())
        if db_id and db_id in drug_targets:
            drugs_with_targets.add(drug)

    print(f"Cancer drugs with targets: {len(drugs_with_targets)}/{len(all_cancer_drugs)}")

    # Score all drugs by cancer target count
    drug_scores = {}
    for drug in drugs_with_targets:
        cancer_count, total_count, _ = score_drug_by_cancer_targets(
            drug, drug_targets, name_to_db, cancer_gene_ids
        )
        if total_count > 0:
            drug_scores[drug] = {
                'cancer_targets': cancer_count,
                'total_targets': total_count,
                'cancer_ratio': cancer_count / total_count if total_count > 0 else 0,
                'cancer_score': cancer_count  # Simple count
            }

    print(f"\nDrugs with cancer target scores: {len(drug_scores)}")

    # Evaluate on test diseases
    hits = 0
    total_diseases = 0
    disease_results = []

    for disease in test_diseases:
        true_drugs = evaluable_diseases[disease]

        # Rank drugs by cancer target score (only consider drugs with scores)
        scored_drugs = [(d, drug_scores.get(d, {}).get('cancer_score', 0))
                       for d in drugs_with_targets]
        scored_drugs.sort(key=lambda x: -x[1])  # Descending

        top_drugs = [d for d, _ in scored_drugs[:top_k]]

        # Calculate hit
        true_positives = set(top_drugs) & true_drugs
        hit = 1 if len(true_positives) > 0 else 0
        hits += hit
        total_diseases += 1

        disease_results.append({
            'disease': disease,
            'true_drugs': len(true_drugs),
            'hit': hit,
            'true_positives': list(true_positives)[:5]
        })

    recall_at_k = hits / total_diseases if total_diseases > 0 else 0

    print(f"\n=== Results (Target-Based Scoring) ===")
    print(f"Test diseases: {total_diseases}")
    print(f"Hits: {hits}")
    print(f"Recall@{top_k}: {recall_at_k:.1%}")

    # Show top drugs by cancer target score
    print(f"\nTop 20 drugs by cancer target count:")
    top_scored = sorted(drug_scores.items(), key=lambda x: -x[1]['cancer_score'])[:20]
    for drug, scores in top_scored:
        print(f"  {drug}: {scores['cancer_score']} cancer targets / {scores['total_targets']} total ({scores['cancer_ratio']:.1%})")

    # Compare to random baseline
    random_hits = 0
    for disease in test_diseases:
        true_drugs = evaluable_diseases[disease]
        random_sample = list(drugs_with_targets)[:top_k]
        if set(random_sample) & true_drugs:
            random_hits += 1
    random_recall = random_hits / len(test_diseases)

    print(f"\nRandom baseline Recall@{top_k}: {random_recall:.1%}")
    print(f"Lift over random: {recall_at_k / random_recall:.2f}x" if random_recall > 0 else "")

    return recall_at_k, drug_scores, disease_results


def main():
    """Main evaluation function."""
    print("=" * 60)
    print("h269: Cancer-Specific Target-Based Scoring")
    print("=" * 60)

    # First, check cancer gene coverage
    cancer_genes = load_cancer_genes()
    print(f"\nCancer gene categories:")
    for category, genes in cancer_genes.items():
        if category != 'all':
            print(f"  {category}: {len(genes)} genes")
    print(f"  TOTAL: {len(cancer_genes['all'])} unique genes")

    cancer_gene_ids, found = map_gene_symbols_to_ids(cancer_genes['all'])
    print(f"\nMapped to IDs: {len(found)}/{len(cancer_genes['all'])} genes")

    # Run evaluation with multiple seeds
    print("\n" + "=" * 60)
    print("Running evaluation with 5 seeds...")
    print("=" * 60)

    recalls = []
    for seed in [42, 123, 456, 789, 1234]:
        print(f"\n--- Seed {seed} ---")
        recall, drug_scores, results = evaluate_cancer_target_scoring(seed=seed)
        recalls.append(recall)

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Mean Recall@30: {np.mean(recalls):.1%} ± {np.std(recalls):.1%}")
    print(f"Individual: {[f'{r:.1%}' for r in recalls]}")

    # Compare to kNN baseline from CLAUDE.md (37% with leakage, 26% without)
    knn_baseline = 0.26  # Fair comparison
    print(f"\nkNN baseline (no leakage): {knn_baseline:.1%}")
    print(f"Improvement: {100*(np.mean(recalls) - knn_baseline):.1f} pp")


if __name__ == '__main__':
    main()
