#!/usr/bin/env python3
"""
h329: Drug Class Width Analysis

Hypothesis: h307 found broad classes (many diseases) have isolation as bad signal,
but statins (narrow class) have isolation as good signal.

This script quantifies "class width" (average GT diseases per drug in class)
and correlates with isolation signal quality.

Goal: Find if class width threshold predicts whether isolation is good/bad signal.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Drug class definitions from production_predictor.py
LOCAL_ANESTHETICS = {'lidocaine', 'bupivacaine', 'ropivacaine', 'prilocaine', 'mepivacaine', 'articaine'}
CORTICOSTEROID_DRUGS = {'prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone',
                        'hydrocortisone', 'betamethasone', 'triamcinolone', 'budesonide',
                        'fluticasone', 'beclomethasone', 'mometasone'}
TNF_INHIBITORS = {'infliximab', 'adalimumab', 'etanercept', 'certolizumab', 'golimumab'}
NSAID_DRUGS = {'ibuprofen', 'naproxen', 'diclofenac', 'celecoxib', 'meloxicam',
               'indomethacin', 'piroxicam', 'ketorolac', 'aspirin', 'ketoprofen'}
IL_INHIBITORS = {'secukinumab', 'ixekizumab', 'ustekinumab', 'tocilizumab', 'sarilumab',
                 'canakinumab', 'anakinra', 'risankizumab', 'guselkumab', 'tildrakizumab',
                 'brodalumab', 'bimekizumab'}
STATIN_DRUGS = {'atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin',
                'lovastatin', 'fluvastatin', 'pitavastatin'}

# Additional drug classes to analyze
ACE_INHIBITORS = {'lisinopril', 'enalapril', 'ramipril', 'captopril', 'benazepril',
                  'fosinopril', 'quinapril', 'trandolapril', 'perindopril'}
BETA_BLOCKERS = {'metoprolol', 'atenolol', 'carvedilol', 'propranolol', 'bisoprolol',
                 'nebivolol', 'labetalol', 'nadolol', 'sotalol'}
SSRIS = {'fluoxetine', 'sertraline', 'paroxetine', 'citalopram', 'escitalopram', 'fluvoxamine'}
PPIS = {'omeprazole', 'esomeprazole', 'lansoprazole', 'pantoprazole', 'rabeprazole'}
ANTIHISTAMINES = {'cetirizine', 'loratadine', 'fexofenadine', 'diphenhydramine', 'hydroxyzine'}
CALCIUM_CHANNEL_BLOCKERS = {'amlodipine', 'nifedipine', 'diltiazem', 'verapamil', 'felodipine'}
FLUOROQUINOLONES = {'ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'ofloxacin', 'norfloxacin'}
TAXANES = {'paclitaxel', 'docetaxel', 'cabazitaxel'}
PLATINUM_AGENTS = {'cisplatin', 'carboplatin', 'oxaliplatin'}
ANTICONVULSANTS = {'carbamazepine', 'valproic acid', 'phenytoin', 'gabapentin', 'pregabalin',
                   'levetiracetam', 'lamotrigine', 'topiramate', 'oxcarbazepine'}

ALL_DRUG_CLASSES = {
    'local_anesthetics': LOCAL_ANESTHETICS,
    'corticosteroids': CORTICOSTEROID_DRUGS,
    'tnf_inhibitors': TNF_INHIBITORS,
    'nsaids': NSAID_DRUGS,
    'il_inhibitors': IL_INHIBITORS,
    'statins': STATIN_DRUGS,
    'ace_inhibitors': ACE_INHIBITORS,
    'beta_blockers': BETA_BLOCKERS,
    'ssris': SSRIS,
    'ppis': PPIS,
    'antihistamines': ANTIHISTAMINES,
    'ccbs': CALCIUM_CHANNEL_BLOCKERS,
    'fluoroquinolones': FLUOROQUINOLONES,
    'taxanes': TAXANES,
    'platinum_agents': PLATINUM_AGENTS,
    'anticonvulsants': ANTICONVULSANTS,
}


def load_ground_truth() -> pd.DataFrame:
    """Load Every Cure ground truth."""
    gt_path = Path("data/reference/everycure/indicationList.xlsx")
    return pd.read_excel(gt_path)


def normalize_drug_name(name: str) -> str:
    """Normalize drug name for matching."""
    if not isinstance(name, str):
        return ""
    return name.lower().strip()


def find_drugs_in_gt(gt_df: pd.DataFrame, drug_names: Set[str]) -> Dict[str, Set[str]]:
    """Find drugs in GT and their diseases."""
    drug_to_diseases: Dict[str, Set[str]] = defaultdict(set)

    for _, row in gt_df.iterrows():
        drug_name = normalize_drug_name(row.get('final normalized drug label', ''))
        disease = str(row.get('final normalized disease label', '')).lower()

        for target_drug in drug_names:
            if target_drug in drug_name:
                drug_to_diseases[target_drug].add(disease)
                break

    return drug_to_diseases


def calculate_class_width(gt_df: pd.DataFrame, class_drugs: Set[str]) -> Tuple[float, int, Dict[str, int]]:
    """
    Calculate class width = average number of GT diseases per drug in class.

    Returns: (avg_diseases_per_drug, n_drugs_found, drug_counts)
    """
    drug_to_diseases = find_drugs_in_gt(gt_df, class_drugs)

    if not drug_to_diseases:
        return 0.0, 0, {}

    disease_counts = {drug: len(diseases) for drug, diseases in drug_to_diseases.items()}
    avg_width = np.mean(list(disease_counts.values()))

    return avg_width, len(drug_to_diseases), disease_counts


def load_predictions() -> pd.DataFrame:
    """Load predictions from deliverables file."""
    pred_path = Path("data/deliverables/drug_repurposing_predictions_with_confidence.xlsx")
    if pred_path.exists():
        return pd.read_excel(pred_path)
    return None


def calculate_isolation_precision(
    predictions_df: pd.DataFrame,
    class_drugs: Set[str],
    gt_df: pd.DataFrame
) -> Tuple[float, float, int, int]:
    """
    Calculate precision for isolated vs non-isolated predictions.

    Returns: (isolated_precision, non_isolated_precision, n_isolated, n_non_isolated)
    """
    if predictions_df is None:
        return 0.0, 0.0, 0, 0

    # Build GT lookup
    gt_pairs = set()
    for _, row in gt_df.iterrows():
        drug = normalize_drug_name(row.get('final normalized drug label', ''))
        disease = str(row.get('final normalized disease label', '')).lower()
        gt_pairs.add((drug, disease))

    # Group predictions by disease
    disease_predictions: Dict[str, List[str]] = defaultdict(list)
    for _, row in predictions_df.iterrows():
        drug = normalize_drug_name(str(row.get('drug_name', '')))
        disease = str(row.get('disease_name', '')).lower()
        disease_predictions[disease].append(drug)

    isolated_correct = 0
    isolated_total = 0
    non_isolated_correct = 0
    non_isolated_total = 0

    for disease, drugs_for_disease in disease_predictions.items():
        # Find class drugs predicted for this disease
        class_drugs_predicted = set()
        for drug in drugs_for_disease:
            for class_drug in class_drugs:
                if class_drug in drug:
                    class_drugs_predicted.add(class_drug)

        if not class_drugs_predicted:
            continue

        is_isolated = len(class_drugs_predicted) == 1

        for class_drug in class_drugs_predicted:
            # Check if this is a hit
            is_hit = False
            for drug in drugs_for_disease:
                if class_drug in drug:
                    # Check GT
                    for gt_drug, gt_disease in gt_pairs:
                        if class_drug in gt_drug and disease in gt_disease:
                            is_hit = True
                            break
                    if is_hit:
                        break

            if is_isolated:
                isolated_total += 1
                if is_hit:
                    isolated_correct += 1
            else:
                non_isolated_total += 1
                if is_hit:
                    non_isolated_correct += 1

    isolated_prec = 100 * isolated_correct / isolated_total if isolated_total > 0 else 0.0
    non_isolated_prec = 100 * non_isolated_correct / non_isolated_total if non_isolated_total > 0 else 0.0

    return isolated_prec, non_isolated_prec, isolated_total, non_isolated_total


def main():
    print("=" * 70)
    print("h329: Drug Class Width Analysis")
    print("=" * 70)

    # Load data
    print("\nLoading ground truth...")
    gt_df = load_ground_truth()
    print(f"  GT: {len(gt_df)} drug-disease pairs")

    print("\nLoading predictions...")
    pred_df = load_predictions()
    if pred_df is not None:
        print(f"  Predictions: {len(pred_df)} total")
    else:
        print("  WARNING: No predictions file found, will use GT-only width analysis")

    # Calculate class width for all classes
    print("\n" + "=" * 70)
    print("STEP 1: Calculate Class Width (avg GT diseases per drug)")
    print("=" * 70)

    results = []
    for class_name, class_drugs in ALL_DRUG_CLASSES.items():
        avg_width, n_found, drug_counts = calculate_class_width(gt_df, class_drugs)

        # Calculate isolation precision gap (if we have predictions)
        iso_prec, non_iso_prec, n_iso, n_non_iso = 0.0, 0.0, 0, 0
        if pred_df is not None:
            iso_prec, non_iso_prec, n_iso, n_non_iso = calculate_isolation_precision(
                pred_df, class_drugs, gt_df
            )

        results.append({
            'class': class_name,
            'avg_width': avg_width,
            'n_drugs_found': n_found,
            'drugs_in_class': len(class_drugs),
            'isolated_precision': iso_prec,
            'non_isolated_precision': non_iso_prec,
            'precision_gap': non_iso_prec - iso_prec,  # Positive = cohesion helps
            'n_isolated': n_iso,
            'n_non_isolated': n_non_iso,
        })

        print(f"\n{class_name}:")
        print(f"  Class width (avg diseases/drug): {avg_width:.1f}")
        print(f"  Drugs found in GT: {n_found}/{len(class_drugs)}")
        if drug_counts:
            top_drugs = sorted(drug_counts.items(), key=lambda x: -x[1])[:3]
            print(f"  Top drugs: {', '.join(f'{d}={c}' for d,c in top_drugs)}")
        if n_iso + n_non_iso > 0:
            print(f"  Isolated precision: {iso_prec:.1f}% (n={n_iso})")
            print(f"  Non-isolated precision: {non_iso_prec:.1f}% (n={n_non_iso})")
            print(f"  Gap (non-iso - iso): {non_iso_prec - iso_prec:+.1f} pp")

    # Create summary table
    print("\n" + "=" * 70)
    print("STEP 2: Summary Table - Width vs Isolation Signal")
    print("=" * 70)

    df = pd.DataFrame(results)
    df = df.sort_values('avg_width', ascending=False)

    print("\n{:20} {:>8} {:>6} {:>8} {:>8} {:>10}".format(
        "Class", "Width", "n", "Iso%", "Non-Iso%", "Gap"
    ))
    print("-" * 70)
    for _, row in df.iterrows():
        gap_str = f"{row['precision_gap']:+.1f}" if row['n_isolated'] + row['n_non_isolated'] > 0 else "N/A"
        iso_str = f"{row['isolated_precision']:.1f}" if row['n_isolated'] > 0 else "N/A"
        non_iso_str = f"{row['non_isolated_precision']:.1f}" if row['n_non_isolated'] > 0 else "N/A"
        print("{:20} {:>8.1f} {:>6} {:>8} {:>8} {:>10}".format(
            row['class'],
            row['avg_width'],
            row['n_drugs_found'],
            iso_str,
            non_iso_str,
            gap_str
        ))

    # Correlation analysis
    print("\n" + "=" * 70)
    print("STEP 3: Correlation Analysis")
    print("=" * 70)

    # Filter to classes with isolation data
    df_with_data = df[(df['n_isolated'] >= 5) & (df['n_non_isolated'] >= 5)]

    if len(df_with_data) >= 3:
        width = df_with_data['avg_width'].values
        gap = df_with_data['precision_gap'].values

        pearson_r, pearson_p = pearsonr(width, gap)
        spearman_r, spearman_p = spearmanr(width, gap)

        print(f"\nClasses with sufficient data: {len(df_with_data)}")
        print(f"\nPearson correlation (width vs gap): r={pearson_r:.3f}, p={pearson_p:.3f}")
        print(f"Spearman correlation (width vs gap): r={spearman_r:.3f}, p={spearman_p:.3f}")

        # Is there a threshold?
        print("\n\nWidth Threshold Analysis:")
        for threshold in [5, 10, 15, 20, 30, 50]:
            narrow = df_with_data[df_with_data['avg_width'] <= threshold]
            wide = df_with_data[df_with_data['avg_width'] > threshold]

            narrow_gap = narrow['precision_gap'].mean() if len(narrow) > 0 else float('nan')
            wide_gap = wide['precision_gap'].mean() if len(wide) > 0 else float('nan')

            print(f"  Threshold {threshold}: Narrow avg gap = {narrow_gap:+.1f} pp, Wide avg gap = {wide_gap:+.1f} pp")
    else:
        print("\nInsufficient data for correlation analysis (need >= 3 classes with data)")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Identify patterns
    cohesion_helps = df[(df['precision_gap'] > 5) & (df['n_isolated'] >= 5)]
    isolation_helps = df[(df['precision_gap'] < -5) & (df['n_isolated'] >= 5)]

    print(f"\nClasses where COHESION helps (gap > +5 pp): {len(cohesion_helps)}")
    for _, row in cohesion_helps.iterrows():
        print(f"  {row['class']}: width={row['avg_width']:.1f}, gap={row['precision_gap']:+.1f} pp")

    print(f"\nClasses where ISOLATION helps (gap < -5 pp): {len(isolation_helps)}")
    for _, row in isolation_helps.iterrows():
        print(f"  {row['class']}: width={row['avg_width']:.1f}, gap={row['precision_gap']:+.1f} pp")

    # Save results
    output_path = Path("data/analysis/h329_class_width_analysis.json")
    output = {
        "hypothesis": "h329 - Drug Class Width Analysis",
        "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        "class_results": df.to_dict(orient='records'),
        "correlation": {
            "pearson_r": float(pearson_r) if len(df_with_data) >= 3 else None,
            "pearson_p": float(pearson_p) if len(df_with_data) >= 3 else None,
            "spearman_r": float(spearman_r) if len(df_with_data) >= 3 else None,
            "spearman_p": float(spearman_p) if len(df_with_data) >= 3 else None,
        },
        "cohesion_helps_classes": cohesion_helps['class'].tolist() if len(cohesion_helps) > 0 else [],
        "isolation_helps_classes": isolation_helps['class'].tolist() if len(isolation_helps) > 0 else [],
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
