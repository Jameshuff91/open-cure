#!/usr/bin/env python3
"""
h163: Drug Class Precision Ranking

Find hidden high-precision drug classes by systematically analyzing ALL drug classes
across ALL disease categories. Since we don't have ATC codes, use drug name patterns
to infer class membership.

SUCCESS: Find new drug classes with >35% precision and n>=10 predictions
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List, Optional

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

# Drug class patterns based on naming conventions
DRUG_CLASSES = {
    # -mab suffix = monoclonal antibodies
    'monoclonal_antibody': {
        'suffix': ['mab'],
        'examples': ['adalimumab', 'infliximab', 'rituximab', 'trastuzumab'],
    },
    # -nib suffix = kinase inhibitors
    'kinase_inhibitor': {
        'suffix': ['nib'],
        'examples': ['imatinib', 'erlotinib', 'sunitinib', 'gefitinib'],
    },
    # -statin suffix = HMG-CoA reductase inhibitors
    'statin': {
        'suffix': ['statin'],
        'examples': ['atorvastatin', 'simvastatin', 'rosuvastatin'],
    },
    # -pril suffix = ACE inhibitors
    'ace_inhibitor': {
        'suffix': ['pril'],
        'examples': ['lisinopril', 'enalapril', 'captopril'],
    },
    # -sartan suffix = ARBs
    'arb': {
        'suffix': ['sartan'],
        'examples': ['losartan', 'valsartan', 'irbesartan'],
    },
    # -olol suffix = beta blockers
    'beta_blocker': {
        'suffix': ['olol'],
        'examples': ['metoprolol', 'atenolol', 'propranolol'],
    },
    # -pine suffix = calcium channel blockers (dihydropyridines)
    'calcium_channel_blocker': {
        'suffix': ['dipine'],
        'examples': ['amlodipine', 'nifedipine', 'felodipine'],
    },
    # -azole suffix = antifungals/antibiotics
    'azole': {
        'suffix': ['azole'],
        'examples': ['fluconazole', 'metronidazole', 'omeprazole'],
    },
    # -cillin suffix = penicillins
    'penicillin': {
        'suffix': ['cillin'],
        'examples': ['amoxicillin', 'ampicillin', 'penicillin'],
    },
    # -mycin suffix = macrolides/aminoglycosides
    'mycin': {
        'suffix': ['mycin'],
        'examples': ['azithromycin', 'erythromycin', 'gentamicin', 'vancomycin'],
    },
    # -cycline suffix = tetracyclines
    'tetracycline': {
        'suffix': ['cycline'],
        'examples': ['doxycycline', 'tetracycline', 'minocycline'],
    },
    # -floxacin suffix = fluoroquinolones
    'fluoroquinolone': {
        'suffix': ['floxacin'],
        'examples': ['ciprofloxacin', 'levofloxacin', 'moxifloxacin'],
    },
    # -cept suffix = receptor fusion proteins
    'receptor_fusion': {
        'suffix': ['cept'],
        'examples': ['etanercept', 'abatacept', 'aflibercept'],
    },
    # -one suffix (steroids)
    'corticosteroid': {
        'contains': ['prednis', 'cortis', 'dexameth', 'hydrocort', 'betameth', 'triamcin', 'budesoni'],
        'suffix': [],
        'examples': ['prednisone', 'dexamethasone', 'hydrocortisone'],
    },
    # NSAID patterns
    'nsaid': {
        'contains': ['ibuprofen', 'naproxen', 'diclofenac', 'indomethacin', 'celecoxib', 'aspirin', 'ketorolac'],
        'suffix': [],
        'examples': [],
    },
    # -prazole suffix = PPIs
    'ppi': {
        'suffix': ['prazole'],
        'examples': ['omeprazole', 'pantoprazole', 'esomeprazole'],
    },
    # -tidine suffix = H2 blockers
    'h2_blocker': {
        'suffix': ['tidine'],
        'examples': ['ranitidine', 'famotidine', 'cimetidine'],
    },
    # -taxel suffix = taxanes
    'taxane': {
        'suffix': ['taxel'],
        'examples': ['paclitaxel', 'docetaxel', 'cabazitaxel'],
    },
    # -platin suffix = platinum compounds
    'platinum': {
        'suffix': ['platin'],
        'examples': ['cisplatin', 'carboplatin', 'oxaliplatin'],
    },
    # -rubicin suffix = anthracyclines
    'anthracycline': {
        'suffix': ['rubicin'],
        'examples': ['doxorubicin', 'epirubicin', 'daunorubicin'],
    },
    # -mustine suffix = alkylating agents
    'alkylating': {
        'contains': ['cyclophosphamide', 'ifosfamide', 'melphalan', 'chlorambucil', 'busulfan'],
        'suffix': ['mustine'],
        'examples': [],
    },
    # -pam suffix = benzodiazepines
    'benzodiazepine': {
        'suffix': ['pam', 'zolam'],
        'examples': ['diazepam', 'lorazepam', 'alprazolam', 'midazolam'],
    },
    # -triptan suffix = triptans (migraine)
    'triptan': {
        'suffix': ['triptan'],
        'examples': ['sumatriptan', 'rizatriptan', 'zolmitriptan'],
    },
    # -setron suffix = 5-HT3 antagonists
    'setron': {
        'suffix': ['setron'],
        'examples': ['ondansetron', 'granisetron', 'palonosetron'],
    },
    # -lukast suffix = leukotriene antagonists
    'leukotriene': {
        'suffix': ['lukast'],
        'examples': ['montelukast', 'zafirlukast'],
    },
    # -glitazone suffix = thiazolidinediones
    'thiazolidinedione': {
        'suffix': ['glitazone'],
        'examples': ['pioglitazone', 'rosiglitazone'],
    },
    # -gliflozin suffix = SGLT2 inhibitors
    'sglt2_inhibitor': {
        'suffix': ['gliflozin'],
        'examples': ['empagliflozin', 'dapagliflozin', 'canagliflozin'],
    },
    # -tide suffix = peptide hormones/analogs
    'peptide': {
        'suffix': ['tide'],
        'examples': ['liraglutide', 'semaglutide', 'octreotide'],
    },
}

# Disease category keywords
DISEASE_CATEGORIES = {
    'autoimmune': ['autoimmune', 'lupus', 'rheumatoid', 'arthritis', 'scleroderma', 'myasthenia',
                   'multiple sclerosis', 'crohn', 'colitis', 'psoriasis', 'sjögren'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'sarcoma', 'myeloma'],
    'cardiovascular': ['cardiac', 'heart', 'coronary', 'hypertension', 'arrhythmia',
                       'atherosclerosis', 'stroke', 'angina', 'infarction'],
    'infectious': ['infection', 'bacterial', 'viral', 'fungal', 'hiv', 'aids', 'hepatitis',
                   'tuberculosis', 'malaria', 'pneumonia', 'sepsis'],
    'metabolic': ['diabetes', 'metabolic', 'obesity', 'hyperlipidemia', 'hyperglycemia',
                  'hypothyroid', 'hyperthyroid'],
    'neurological': ['parkinson', 'alzheimer', 'epilepsy', 'seizure', 'migraine',
                     'neuropathy', 'dementia', 'stroke'],
    'respiratory': ['asthma', 'copd', 'pulmonary', 'bronchitis', 'pneumonia', 'fibrosis'],
    'gi': ['gastro', 'intestin', 'ulcer', 'reflux', 'colitis', 'hepat', 'cirrhosis', 'pancreat'],
}


def classify_drug(drug_name: str) -> Optional[str]:
    """Classify drug into a drug class based on naming patterns."""
    drug_lower = drug_name.lower()

    for class_name, patterns in DRUG_CLASSES.items():
        # Check suffix patterns
        for suffix in patterns.get('suffix', []):
            if drug_lower.endswith(suffix):
                return class_name
        # Check contains patterns
        for contain in patterns.get('contains', []):
            if contain in drug_lower:
                return class_name

    return None


def classify_disease(disease_name: str) -> str:
    """Classify disease into a category."""
    disease_lower = disease_name.lower()
    for category, keywords in DISEASE_CATEGORIES.items():
        if any(kw in disease_lower for kw in keywords):
            return category
    return 'other'


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
        disease_name = str(row['disease name']).lower().strip()
        drug_name = str(row['final normalized drug label']).lower().strip()

        disease_id = mesh_mappings.get(disease_name)
        if not disease_id:
            disease_id = matcher.get_mesh_id(disease_name)
        if not disease_id:
            continue

        drug_id = name_to_drug_id.get(drug_name)
        if drug_id:
            gt[disease_id].add(drug_id)
            disease_names[disease_id] = disease_name

    return gt, disease_names


def knn_predictions(disease_id, train_diseases, gt, embeddings, id_to_name, k=20):
    """Generate kNN predictions for a disease."""
    if disease_id not in embeddings:
        return []

    query_emb = embeddings[disease_id].reshape(1, -1)
    train_with_emb = [d for d in train_diseases if d in embeddings and d != disease_id]
    if not train_with_emb:
        return []

    train_embs = np.vstack([embeddings[d] for d in train_with_emb])
    sims = cosine_similarity(query_emb, train_embs)[0]
    top_idx = np.argsort(sims)[-k:][::-1]
    neighbors = [train_with_emb[i] for i in top_idx]
    neighbor_sims = [sims[i] for i in top_idx]

    drug_scores: Dict[str, float] = defaultdict(float)
    for neighbor, sim in zip(neighbors, neighbor_sims):
        for drug in gt.get(neighbor, []):
            drug_scores[drug] += sim

    sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)

    predictions = []
    for rank, (drug_id, score) in enumerate(sorted_drugs[:30], 1):
        drug_name = id_to_name.get(drug_id, drug_id)
        predictions.append({
            'drug_id': drug_id,
            'drug_name': drug_name,
            'rank': rank,
            'score': score,
        })

    return predictions


def main():
    print("h163: Drug Class Precision Ranking")
    print("=" * 80)

    print("\nLoading data...")
    embeddings = load_node2vec_embeddings()
    mesh_mappings = load_mesh_mappings_from_file()
    name_to_drug_id, id_to_name = load_drugbank_lookup()
    gt, disease_names = load_ground_truth(mesh_mappings, name_to_drug_id)

    diseases = [d for d in gt.keys() if d in embeddings and len(gt[d]) >= 1]
    print(f"  Total diseases with GT and embeddings: {len(diseases)}")

    # Collect predictions across all seeds
    all_preds = []

    for seed in SEEDS:
        np.random.seed(seed)

        n_test = max(1, len(diseases) // 5)
        test_diseases = set(np.random.choice(diseases, n_test, replace=False))
        train_diseases = set(diseases) - test_diseases

        for disease in test_diseases:
            preds = knn_predictions(disease, train_diseases, gt, embeddings, id_to_name)
            gt_drugs = gt[disease]
            disease_cat = classify_disease(disease_names[disease])

            for p in preds:
                drug_class = classify_drug(p['drug_name'])
                is_hit = p['drug_id'] in gt_drugs

                all_preds.append({
                    'disease': disease_names[disease],
                    'disease_category': disease_cat,
                    'drug': p['drug_name'],
                    'drug_class': drug_class,
                    'rank': p['rank'],
                    'score': p['score'],
                    'is_hit': is_hit,
                    'seed': seed,
                })

    df = pd.DataFrame(all_preds)
    print(f"\nTotal predictions: {len(df)}")
    print(f"Overall hit rate: {df['is_hit'].mean()*100:.2f}%")

    # Analyze by drug class (overall)
    print("\n" + "=" * 80)
    print("OVERALL DRUG CLASS PRECISION")
    print("=" * 80)

    class_stats = []
    for drug_class in sorted(DRUG_CLASSES.keys()):
        class_df = df[df['drug_class'] == drug_class]
        if len(class_df) == 0:
            continue
        n = len(class_df)
        hits = class_df['is_hit'].sum()
        precision = hits / n * 100

        # Also check with rank filter
        rank5_df = class_df[class_df['rank'] <= 5]
        n_r5 = len(rank5_df)
        hits_r5 = rank5_df['is_hit'].sum()
        precision_r5 = hits_r5 / n_r5 * 100 if n_r5 > 0 else 0

        rank10_df = class_df[class_df['rank'] <= 10]
        n_r10 = len(rank10_df)
        hits_r10 = rank10_df['is_hit'].sum()
        precision_r10 = hits_r10 / n_r10 * 100 if n_r10 > 0 else 0

        class_stats.append({
            'class': drug_class,
            'n': n,
            'hits': int(hits),
            'precision': precision,
            'n_rank5': n_r5,
            'precision_rank5': precision_r5,
            'n_rank10': n_r10,
            'precision_rank10': precision_r10,
        })

    # Sort by precision
    class_stats.sort(key=lambda x: -x['precision'])

    print(f"\n{'Drug Class':<25} {'N':>6} {'Hits':>5} {'Prec':>8} | {'R<=5 N':>6} {'Prec':>8} | {'R<=10 N':>6} {'Prec':>8}")
    print("-" * 95)

    for s in class_stats:
        if s['n'] >= 5:  # Only show classes with enough data
            star = " ***" if s['precision'] >= 35 or s['precision_rank5'] >= 35 or s['precision_rank10'] >= 35 else ""
            print(f"{s['class']:<25} {s['n']:>6} {s['hits']:>5} {s['precision']:>7.1f}% | {s['n_rank5']:>6} {s['precision_rank5']:>7.1f}% | {s['n_rank10']:>6} {s['precision_rank10']:>7.1f}%{star}")

    # Find high-precision classes (not already in production)
    already_in_production = {'statin', 'corticosteroid', 'beta_blocker', 'taxane', 'alkylating',
                             'fluoroquinolone', 'tetracycline'}

    print("\n" + "=" * 80)
    print("NEW HIGH-PRECISION CLASSES (>35% with n>=10, not in production)")
    print("=" * 80)

    new_findings = []
    for s in class_stats:
        if s['class'] in already_in_production:
            continue
        # Check any criteria
        for criteria, prec_key, n_key in [
            ('overall', 'precision', 'n'),
            ('rank<=5', 'precision_rank5', 'n_rank5'),
            ('rank<=10', 'precision_rank10', 'n_rank10'),
        ]:
            if s[prec_key] >= 35 and s[n_key] >= 10:
                new_findings.append({
                    'class': s['class'],
                    'criteria': criteria,
                    'n': s[n_key],
                    'precision': s[prec_key],
                })

    if new_findings:
        for f in sorted(new_findings, key=lambda x: -x['precision']):
            print(f"  ✓ {f['class']} + {f['criteria']}: {f['precision']:.1f}% (n={f['n']})")
    else:
        print("  No new classes meeting threshold found")

    # Analyze by drug class per disease category
    print("\n" + "=" * 80)
    print("DRUG CLASS BY DISEASE CATEGORY (showing >30% precision with n>=5)")
    print("=" * 80)

    category_findings = []
    for disease_cat in DISEASE_CATEGORIES.keys():
        cat_df = df[df['disease_category'] == disease_cat]
        if len(cat_df) == 0:
            continue

        print(f"\n{disease_cat.upper()}")
        print("-" * 60)

        for drug_class in sorted(DRUG_CLASSES.keys()):
            class_cat_df = cat_df[cat_df['drug_class'] == drug_class]
            if len(class_cat_df) < 5:
                continue

            n = len(class_cat_df)
            hits = class_cat_df['is_hit'].sum()
            precision = hits / n * 100

            # Check with rank filter
            rank10_df = class_cat_df[class_cat_df['rank'] <= 10]
            n_r10 = len(rank10_df)
            hits_r10 = rank10_df['is_hit'].sum()
            precision_r10 = hits_r10 / n_r10 * 100 if n_r10 > 0 else 0

            if precision >= 30 or (precision_r10 >= 30 and n_r10 >= 5):
                star = "***" if precision >= 35 or precision_r10 >= 35 else ""
                print(f"  {drug_class:<25} overall: {precision:>5.1f}% (n={n:>3}) | rank<=10: {precision_r10:>5.1f}% (n={n_r10:>3}) {star}")

                if s['class'] not in already_in_production:
                    category_findings.append({
                        'disease_category': disease_cat,
                        'drug_class': drug_class,
                        'precision': precision,
                        'precision_rank10': precision_r10,
                        'n': n,
                        'n_rank10': n_r10,
                    })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\nClasses already in production (confirmed):")
    for cls in sorted(already_in_production):
        stats = next((s for s in class_stats if s['class'] == cls), None)
        if stats:
            print(f"  ✓ {cls}: {stats['precision']:.1f}% overall, {stats['precision_rank10']:.1f}% rank<=10")

    print("\nNew classes to consider:")
    if new_findings:
        for f in sorted(new_findings, key=lambda x: -x['precision'])[:5]:
            print(f"  → {f['class']} + {f['criteria']}: {f['precision']:.1f}% (n={f['n']})")
    else:
        print("  None found above threshold")

    # Save results
    results = {
        'class_stats': class_stats,
        'new_findings': new_findings,
        'category_findings': category_findings,
    }

    output_file = ANALYSIS_DIR / "h163_drug_class_precision.json"
    with open(output_file, 'w') as f:
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            return obj
        json.dump(results, f, indent=2, default=convert)

    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
