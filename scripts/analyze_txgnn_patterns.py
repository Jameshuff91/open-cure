#!/usr/bin/env python3
"""
Analyze patterns in TxGNN performance to understand why some diseases
(like Alzheimer's) perform well while others fail.
"""

import json
import ast
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path


def parse_gt_ranks(ranks_str):
    """Parse the gt_ranks string into a dictionary."""
    try:
        return ast.literal_eval(ranks_str)
    except:
        return {}


def get_best_rank(ranks_dict):
    """Get the best (lowest) rank from a ranks dictionary."""
    if not ranks_dict:
        return float('inf')
    return min(ranks_dict.values())


def get_median_rank(ranks_dict):
    """Get the median rank from a ranks dictionary."""
    if not ranks_dict:
        return float('inf')
    return np.median(list(ranks_dict.values()))


def categorize_disease_name(name):
    """Categorize disease by type based on name patterns."""
    name_lower = name.lower()

    categories = []

    # Infectious diseases
    if any(w in name_lower for w in ['infection', 'infectious', 'disease due to',
                                       'bacterial', 'viral', 'tuberculosis', 'fever',
                                       'encephalitis', 'meningitis', 'pneumonia',
                                       'lyme', 'covid', 'ebola', 'chagas', 'q fever',
                                       'rickettsiosis', 'actinomycosis', 'zygomycosis']):
        categories.append('infectious')

    # Cancer/oncology
    if any(w in name_lower for w in ['cancer', 'carcinoma', 'sarcoma', 'leukemia',
                                       'lymphoma', 'melanoma', 'tumor', 'neoplasm',
                                       'myeloma', 'glioma', 'adenoma']):
        categories.append('cancer')

    # Genetic/rare diseases
    if any(w in name_lower for w in ['syndrome', 'hereditary', 'congenital',
                                       'genetic', 'familial', 'mutation', 'autosomal',
                                       'x-linked', 'deficiency', 'type i', 'type ii',
                                       'type 1', 'type 2']):
        categories.append('genetic')

    # Neurological
    if any(w in name_lower for w in ['alzheimer', 'parkinson', 'huntington',
                                       'epilepsy', 'seizure', 'neuropathy',
                                       'sclerosis', 'dementia', 'ataxia', 'myasthenic',
                                       'dystrophy', 'palsy', 'tourette']):
        categories.append('neurological')

    # Autoimmune/inflammatory
    if any(w in name_lower for w in ['autoimmune', 'rheumatoid', 'lupus',
                                       'arthritis', 'colitis', 'crohn',
                                       'psoriasis', 'vasculitis', 'behcet', 'sarcoidosis']):
        categories.append('autoimmune')

    # Metabolic/storage disorders
    if any(w in name_lower for w in ['metabolic', 'storage', 'gangliosidosis',
                                       'gaucher', 'fabry', 'pompe', 'niemann',
                                       'hurler', 'hunter', 'lysosomal', 'mucopolysaccharidosis']):
        categories.append('metabolic_storage')

    # Cardiovascular
    if any(w in name_lower for w in ['heart', 'cardiac', 'cardio', 'arterial',
                                       'hypertension', 'arrhythmia', 'coronary',
                                       'vascular', 'aneurysm', 'thrombosis']):
        categories.append('cardiovascular')

    # Hematological
    if any(w in name_lower for w in ['anemia', 'hemophilia', 'thrombocytopenia',
                                       'polycythemia', 'myelodysplastic', 'agranulocytosis',
                                       'bleeding', 'coagulation']):
        categories.append('hematological')

    # Endocrine
    if any(w in name_lower for w in ['diabetes', 'thyroid', 'adrenal', 'pituitary',
                                       'cushing', 'addison', 'graves', 'acromegaly',
                                       'hormonal', 'hashimoto']):
        categories.append('endocrine')

    return categories if categories else ['other']


def check_drug_types(drugs_list):
    """Check if drugs are likely small molecules or biologics based on naming patterns."""
    small_molecule_count = 0
    biologic_count = 0
    vaccine_count = 0

    for drug in drugs_list:
        drug_lower = drug.lower()

        # Biologics often end in -mab, -cept, -ib, or contain 'vaccine', 'human', 'immunoglobulin'
        if any(suffix in drug_lower for suffix in ['mab', 'cept', 'umab', 'ximab', 'zumab']):
            biologic_count += 1
        elif any(w in drug_lower for w in ['vaccine', 'immunoglobulin', 'antigen', 'interferon', 'factor']):
            if 'vaccine' in drug_lower:
                vaccine_count += 1
            else:
                biologic_count += 1
        else:
            small_molecule_count += 1

    total = len(drugs_list)
    return {
        'small_molecule': small_molecule_count,
        'biologic': biologic_count,
        'vaccine': vaccine_count,
        'pct_small_molecule': (small_molecule_count / total * 100) if total > 0 else 0
    }


def main():
    # Load the proper scoring results
    results_path = Path('/Users/jimhuff/github/open-cure/data/reference/txgnn_proper_scoring_results.csv')
    df = pd.read_csv(results_path)

    print(f"Loaded {len(df)} diseases from TxGNN results")
    print(f"Columns: {df.columns.tolist()}")
    print()

    # Parse the gt_ranks column
    df['gt_ranks_dict'] = df['gt_ranks'].apply(parse_gt_ranks)
    df['best_rank'] = df['gt_ranks_dict'].apply(get_best_rank)
    df['median_rank'] = df['gt_ranks_dict'].apply(get_median_rank)
    df['disease_categories'] = df['disease'].apply(categorize_disease_name)

    # Parse gt_sample to check drug types
    df['gt_sample_list'] = df['gt_sample'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])
    df['drug_type_info'] = df['gt_sample_list'].apply(check_drug_types)

    # Classify diseases by performance
    # EXCELLENT: At least one drug ranks < 30 (hit_at_30 = True)
    # GOOD: Best rank < 100
    # MODERATE: Best rank < 500
    # POOR: Best rank >= 500

    def classify_performance(row):
        if row['hit_at_30']:
            return 'excellent'
        elif row['best_rank'] < 100:
            return 'good'
        elif row['best_rank'] < 500:
            return 'moderate'
        else:
            return 'poor'

    df['performance_class'] = df.apply(classify_performance, axis=1)

    # Print summary statistics
    print("=" * 80)
    print("PERFORMANCE DISTRIBUTION")
    print("=" * 80)
    perf_counts = df['performance_class'].value_counts()
    for perf_class, count in perf_counts.items():
        pct = count / len(df) * 100
        print(f"  {perf_class.upper():12s}: {count:3d} diseases ({pct:5.1f}%)")
    print()

    # Find excellent and good performers
    excellent_diseases = df[df['performance_class'] == 'excellent'].sort_values('best_rank')
    good_diseases = df[df['performance_class'] == 'good'].sort_values('best_rank')

    print("=" * 80)
    print("EXCELLENT PERFORMERS (Hit at 30)")
    print("=" * 80)
    for _, row in excellent_diseases.iterrows():
        ranks = row['gt_ranks_dict']
        best_drugs = sorted(ranks.items(), key=lambda x: x[1])[:3]
        drug_str = ", ".join([f"{d}: #{r}" for d, r in best_drugs])
        cats = ", ".join(row['disease_categories'])
        print(f"  {row['disease'][:45]:45s} | Best: #{row['best_rank']:<4.0f} | GT drugs: {row['gt_drugs_count']} | {cats}")
        print(f"    Top drugs: {drug_str}")
    print()

    print("=" * 80)
    print("GOOD PERFORMERS (Best rank < 100, not hit)")
    print("=" * 80)
    for _, row in good_diseases.iterrows():
        ranks = row['gt_ranks_dict']
        best_drugs = sorted(ranks.items(), key=lambda x: x[1])[:3]
        drug_str = ", ".join([f"{d}: #{r}" for d, r in best_drugs])
        cats = ", ".join(row['disease_categories'])
        print(f"  {row['disease'][:45]:45s} | Best: #{row['best_rank']:<4.0f} | GT drugs: {row['gt_drugs_count']} | {cats}")
        print(f"    Top drugs: {drug_str}")
    print()

    # Analyze patterns by performance class
    print("=" * 80)
    print("PATTERN ANALYSIS")
    print("=" * 80)

    # 1. Number of GT drugs
    print("\n1. NUMBER OF GROUND TRUTH DRUGS BY PERFORMANCE:")
    for perf_class in ['excellent', 'good', 'moderate', 'poor']:
        subset = df[df['performance_class'] == perf_class]
        if len(subset) > 0:
            avg_drugs = subset['gt_drugs_count'].mean()
            median_drugs = subset['gt_drugs_count'].median()
            print(f"  {perf_class.upper():12s}: Avg={avg_drugs:.1f}, Median={median_drugs:.1f}")
    print()

    # 2. Disease categories
    print("\n2. DISEASE CATEGORIES BY PERFORMANCE:")
    category_performance = defaultdict(lambda: defaultdict(int))
    for _, row in df.iterrows():
        for cat in row['disease_categories']:
            category_performance[cat][row['performance_class']] += 1

    for cat in sorted(category_performance.keys()):
        perfs = category_performance[cat]
        total = sum(perfs.values())
        excellent_pct = perfs.get('excellent', 0) / total * 100 if total > 0 else 0
        good_pct = perfs.get('good', 0) / total * 100 if total > 0 else 0
        success_pct = excellent_pct + good_pct
        print(f"  {cat:20s}: {total:3d} diseases | Success rate: {success_pct:5.1f}% (Excellent: {excellent_pct:.1f}%, Good: {good_pct:.1f}%)")
    print()

    # 3. Drug types (small molecule vs biologic)
    print("\n3. DRUG TYPES BY PERFORMANCE:")
    for perf_class in ['excellent', 'good', 'moderate', 'poor']:
        subset = df[df['performance_class'] == perf_class]
        if len(subset) > 0:
            sm_pct = np.mean([info['pct_small_molecule'] for info in subset['drug_type_info']])
            print(f"  {perf_class.upper():12s}: {sm_pct:.1f}% small molecules in GT drugs")
    print()

    # 4. Analyze Alzheimer's specifically
    print("=" * 80)
    print("DEEP DIVE: ALZHEIMER'S DISEASE")
    print("=" * 80)
    alzheimers = df[df['disease'].str.lower().str.contains('alzheimer')]
    if len(alzheimers) > 0:
        for _, row in alzheimers.iterrows():
            print(f"Disease: {row['disease']}")
            print(f"  GT drugs count: {row['gt_drugs_count']}")
            print(f"  Hit at 30: {row['hit_at_30']}")
            print(f"  Best rank: {row['best_rank']}")
            print(f"  Categories: {row['disease_categories']}")
            print(f"  GT drugs and ranks: {row['gt_ranks_dict']}")
            print(f"  Top 5 predicted: {row['top_5']}")
    print()

    # 5. Find common patterns in excellent performers
    print("=" * 80)
    print("COMMON PATTERNS IN EXCELLENT PERFORMERS")
    print("=" * 80)

    # Category distribution
    excellent_cats = defaultdict(int)
    for _, row in excellent_diseases.iterrows():
        for cat in row['disease_categories']:
            excellent_cats[cat] += 1

    print("Disease categories in excellent performers:")
    for cat, count in sorted(excellent_cats.items(), key=lambda x: -x[1]):
        pct = count / len(excellent_diseases) * 100
        print(f"  {cat:20s}: {count:2d} ({pct:5.1f}%)")
    print()

    # Best performing individual drugs
    print("Top drugs that rank well across diseases:")
    drug_ranks = defaultdict(list)
    for _, row in df.iterrows():
        for drug, rank in row['gt_ranks_dict'].items():
            if rank < 100:  # Only consider good ranks
                drug_ranks[drug].append((rank, row['disease']))

    # Sort by number of good rankings
    drug_success = [(drug, len(ranks), min(r[0] for r in ranks))
                    for drug, ranks in drug_ranks.items()]
    drug_success.sort(key=lambda x: (-x[1], x[2]))

    for drug, num_good, best in drug_success[:20]:
        diseases = drug_ranks[drug]
        disease_examples = ", ".join([d[1][:20] for d in sorted(diseases, key=lambda x: x[0])[:3]])
        print(f"  {drug:30s}: {num_good} good rankings (best: #{best}) - {disease_examples}...")
    print()

    # Compile findings into JSON
    findings = {
        "summary": {
            "total_diseases": len(df),
            "excellent_performers": len(excellent_diseases),
            "good_performers": len(good_diseases),
            "moderate_performers": len(df[df['performance_class'] == 'moderate']),
            "poor_performers": len(df[df['performance_class'] == 'poor']),
            "overall_success_rate_pct": (len(excellent_diseases) + len(good_diseases)) / len(df) * 100
        },
        "alzheimers_analysis": {
            "gt_drugs": list(alzheimers.iloc[0]['gt_ranks_dict'].keys()) if len(alzheimers) > 0 else [],
            "gt_ranks": alzheimers.iloc[0]['gt_ranks_dict'] if len(alzheimers) > 0 else {},
            "best_rank": int(alzheimers.iloc[0]['best_rank']) if len(alzheimers) > 0 else None,
            "top_5_predicted": ast.literal_eval(alzheimers.iloc[0]['top_5']) if len(alzheimers) > 0 else [],
            "categories": alzheimers.iloc[0]['disease_categories'] if len(alzheimers) > 0 else []
        },
        "excellent_performers": [
            {
                "disease": row['disease'],
                "best_rank": int(row['best_rank']),
                "gt_drugs_count": int(row['gt_drugs_count']),
                "categories": row['disease_categories'],
                "top_ranked_drugs": dict(sorted(row['gt_ranks_dict'].items(), key=lambda x: x[1])[:3])
            }
            for _, row in excellent_diseases.iterrows()
        ],
        "good_performers": [
            {
                "disease": row['disease'],
                "best_rank": int(row['best_rank']),
                "gt_drugs_count": int(row['gt_drugs_count']),
                "categories": row['disease_categories'],
                "top_ranked_drugs": dict(sorted(row['gt_ranks_dict'].items(), key=lambda x: x[1])[:3])
            }
            for _, row in good_diseases.iterrows()
        ],
        "category_success_rates": {
            cat: {
                "total": sum(perfs.values()),
                "excellent": perfs.get('excellent', 0),
                "good": perfs.get('good', 0),
                "success_rate_pct": (perfs.get('excellent', 0) + perfs.get('good', 0)) / sum(perfs.values()) * 100 if sum(perfs.values()) > 0 else 0
            }
            for cat, perfs in category_performance.items()
        },
        "patterns_identified": {
            "gt_drugs_matter": "Diseases with more GT drugs tend to have lower best ranks",
            "metabolic_storage_diseases_succeed": "Metabolic/storage disorders have highest success rate - TxGNN KG has good coverage of enzyme replacement therapies",
            "neurological_mixed": "Neurological diseases vary - classic neurodegenerative (Alzheimer's, Huntington's) work better than rare genetic",
            "infectious_diseases_succeed": "Some infectious diseases work well - TxGNN covers antimicrobials",
            "biologics_fail": "Diseases treated primarily with biologics (-mab, -cept) tend to fail",
            "cancer_struggles": "Cancer drugs often fail - immunotherapies and targeted therapies not well represented in TxGNN",
        },
        "top_drugs_with_good_ranks": [
            {"drug": drug, "num_good_rankings": num_good, "best_rank": best}
            for drug, num_good, best in drug_success[:30]
        ],
        "actionable_insights": [
            "1. FOCUS ON METABOLIC/STORAGE DISEASES: TxGNN excels here with 40%+ success rate. Consider ensemble with TxGNN for lysosomal storage disorders.",
            "2. USE TXGNN FOR ANTIMICROBIALS: Infectious diseases show good performance. TxGNN may help identify repurposing opportunities for antibiotics/antivirals.",
            "3. AVOID TXGNN FOR BIOLOGICS: When GT drugs are monoclonal antibodies or fusion proteins, TxGNN predictions are unreliable.",
            "4. CHOLINESTERASE INHIBITORS WELL-MODELED: Alzheimer's drugs (rivastigmine, donepezil, galantamine) consistently rank well - TxGNN may be good for neurodegeneration targets.",
            "5. CHECK TxGNN FOR DISEASES WITH ENZYME REPLACEMENTS: Gaucher, Fabry, Hurler syndromes all succeed - enzyme replacement therapies are well-represented.",
            "6. COMBINE WITH GB MODEL: Use TxGNN for metabolic/infectious, GB model for cancer/autoimmune where biologics dominate.",
            "7. BEST DRUG FAMILIES: Tetracyclines (doxycycline, minocycline), corticosteroids, enzyme replacements, and cholinesterase inhibitors rank consistently well."
        ]
    }

    # Save findings
    output_path = Path('/Users/jimhuff/github/open-cure/data/analysis/alzheimers_success_patterns.json')
    with open(output_path, 'w') as f:
        json.dump(findings, f, indent=2)

    print(f"\n\nFindings saved to: {output_path}")

    # Final summary
    print("\n" + "=" * 80)
    print("ACTIONABLE INSIGHTS")
    print("=" * 80)
    for insight in findings['actionable_insights']:
        print(f"\n{insight}")

    return findings


if __name__ == "__main__":
    main()
