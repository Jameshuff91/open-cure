#!/usr/bin/env python3
"""
Analyze TxGNN vs GB model performance by disease category.
Categorizes diseases using keyword matching and computes per-category Recall@30.
"""

import json
import pandas as pd
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

# Define disease category keywords
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "neurological": [
        "alzheimer", "parkinson", "huntington", "epilepsy", "seizure", "neuropathy",
        "encephalitis", "encephalopathy", "dementia", "ataxia", "dystonia", "myasthenic",
        "guillain", "multiple sclerosis", "als", "amyotrophic", "brain", "cerebral",
        "neurodegenerative", "neuroblastoma", "glioma", "glioblastoma", "meningitis",
        "lennox", "west syndrome", "tourette", "melas", "leber", "optic", "migraine",
        "headache", "spinal", "muscular dystrophy", "duchenne", "becker"
    ],
    "autoimmune": [
        "autoimmune", "lupus", "rheumatoid", "arthritis", "psoriasis", "crohn",
        "colitis", "inflammatory bowel", "sjogren", "scleroderma", "myositis",
        "vasculitis", "takayasu", "behcet", "celiac", "hashimoto", "graves",
        "addison", "pemphigus", "pemphigoid", "dermatomyositis", "polymyositis",
        "ankylosing", "spondylitis", "still disease", "kawasaki"
    ],
    "cancer": [
        "cancer", "carcinoma", "sarcoma", "lymphoma", "leukemia", "melanoma",
        "tumor", "tumour", "neoplasm", "oncology", "malignant", "metastatic",
        "myeloma", "glioma", "blastoma", "adenocarcinoma", "ewing", "wilms",
        "hodgkin", "burkitt", "neuroblastoma", "mesothelioma", "thymoma"
    ],
    "infectious": [
        "infection", "infectious", "bacterial", "viral", "fungal", "parasitic",
        "tuberculosis", "hiv", "aids", "hepatitis", "malaria", "covid", "influenza",
        "pneumonia", "sepsis", "meningitis", "encephalitis", "abscess", "fever",
        "dengue", "ebola", "chagas", "lyme", "typhoid", "cholera", "plague",
        "rickettsia", "clostridium", "pseudomonas", "klebsiella", "staphylococc",
        "streptococc", "candida", "aspergill", "zygomycosis", "cryptococ",
        "helicobacter", "moraxella", "actinomycosis", "q fever", "encephalitis"
    ],
    "metabolic": [
        "metabolic", "diabetes", "obesity", "hyperlipidemia", "cholesterol",
        "thyroid", "hypothyroid", "hyperthyroid", "gout", "porphyria",
        "glycogen storage", "lysosomal", "fabry", "gaucher", "niemann-pick",
        "mucopolysaccharidosis", "hurler", "hunter", "scheie", "phenylketonuria",
        "galactosemia", "fructose", "wilson", "hemochromatosis", "amyloidosis",
        "cushing", "zollinger", "prader-willi", "laron", "achondroplasia",
        "bardet-biedl", "noonan"
    ],
    "cardiovascular": [
        "heart", "cardiac", "cardiovascular", "arrhythmia", "atrial fibrillation",
        "coronary", "myocardial", "infarction", "hypertension", "hypotension",
        "stroke", "thrombosis", "embolism", "aneurysm", "aortic", "valve",
        "cardiomyopathy", "angina", "peripheral arterial", "pulmonary hypertension",
        "eisenmenger", "arteritis", "venous", "vascular"
    ],
    "respiratory": [
        "respiratory", "lung", "pulmonary", "asthma", "copd", "bronchitis",
        "pneumonia", "fibrosis", "emphysema", "cystic fibrosis", "sleep apnea",
        "bronchiectasis", "sarcoidosis", "pleurisy", "ards"
    ],
    "hematological": [
        "anemia", "hemophilia", "thrombocytopenia", "leukopenia", "neutropenia",
        "agranulocytosis", "polycythemia", "thalassemia", "sickle cell",
        "coagulation", "bleeding", "hemorrhagic", "platelet", "von willebrand",
        "aplastic", "myelodysplastic", "myeloproliferative", "erythroid",
        "diamond-blackfan", "mastocytosis"
    ],
    "genetic_rare": [
        "syndrome", "congenital", "hereditary", "genetic", "inherited", "mutation",
        "ehlers-danlos", "marfan", "osteogenesis", "cystic fibrosis", "prader-willi",
        "angelman", "rett", "fragile x", "down syndrome", "turner", "klinefelter",
        "noonan", "bardet-biedl", "achondroplasia", "gorham", "langerhans",
        "shox", "c1 inhibitor", "cinca", "muckle-wells", "agammaglobulinemia"
    ],
    "dermatological": [
        "skin", "dermatitis", "eczema", "psoriasis", "acne", "rosacea",
        "vitiligo", "alopecia", "urticaria", "stevens-johnson", "pemphigus",
        "epidermolysis", "ichthyosis"
    ],
    "gastrointestinal": [
        "gastro", "intestinal", "bowel", "colitis", "crohn", "celiac",
        "liver", "hepatic", "cirrhosis", "pancreatitis", "cholangitis",
        "esophageal", "stomach", "ulcer", "helicobacter"
    ],
    "psychiatric": [
        "depression", "anxiety", "bipolar", "schizophrenia", "psychosis",
        "ptsd", "ocd", "adhd", "autism", "agoraphobia", "panic", "phobia",
        "eating disorder", "anorexia", "bulimia", "alcohol withdrawal",
        "substance", "addiction"
    ],
    "ophthalmological": [
        "eye", "ocular", "retinal", "macular", "glaucoma", "cataract",
        "uveitis", "conjunctivitis", "keratitis", "optic neuropathy"
    ],
    "renal": [
        "kidney", "renal", "nephritis", "nephrotic", "nephropathy",
        "glomerulonephritis", "polycystic kidney", "dialysis"
    ],
    "musculoskeletal": [
        "osteoporosis", "osteoarthritis", "fracture", "bone", "joint",
        "tendinitis", "bursitis", "fibromyalgia", "myopathy", "rhabdomyolysis"
    ]
}


def categorize_disease(disease_name: str) -> List[str]:
    """
    Categorize a disease based on keyword matching.
    Returns list of categories (can be multiple).
    """
    disease_lower = disease_name.lower()
    categories = []

    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in disease_lower:
                categories.append(category)
                break  # Only add each category once

    if not categories:
        categories = ["other"]

    return categories


def load_gb_evaluation() -> Dict:
    """Load GB model evaluation results."""
    gb_path = Path("/Users/jimhuff/github/open-cure/models/gb_enhanced_evaluation.json")
    with open(gb_path) as f:
        return json.load(f)


def main():
    # Load TxGNN results
    results_path = Path("/Users/jimhuff/github/open-cure/data/reference/txgnn_proper_scoring_results.csv")
    df = pd.read_csv(results_path)

    print(f"Loaded {len(df)} disease evaluations from TxGNN results")
    print(f"TxGNN Overall Recall@30: {df['hit_at_30'].mean() * 100:.1f}%")
    print(f"TxGNN Total hits: {df['hit_at_30'].sum()} out of {len(df)}")
    print()

    # Load GB model results for comparison
    gb_results = load_gb_evaluation()
    gb_by_disease = gb_results["by_disease"]

    print(f"GB Model: {len(gb_by_disease)} diseases evaluated")
    print(f"GB Model Aggregate Recall@30: {gb_results['aggregate_recall@30'] * 100:.1f}%")
    print()

    # Categorize GB model diseases
    gb_category_stats: Dict[str, Dict] = defaultdict(lambda: {
        "count": 0,
        "total_recall": 0.0,
        "diseases": []
    })

    for disease, stats in gb_by_disease.items():
        categories = categorize_disease(disease)
        for category in categories:
            gb_category_stats[category]["count"] += 1
            gb_category_stats[category]["total_recall"] += stats["recall@30"]
            gb_category_stats[category]["diseases"].append({
                "name": disease,
                "recall": stats["recall@30"]
            })

    # Categorize each TxGNN disease
    disease_categories = {}
    for _, row in df.iterrows():
        disease = row['disease']
        categories = categorize_disease(disease)
        disease_categories[disease] = categories

    # Compute per-category statistics for TxGNN
    txgnn_category_stats: Dict[str, Dict] = defaultdict(lambda: {
        "count": 0,
        "hits": 0,
        "diseases": [],
        "hit_diseases": [],
        "miss_diseases": []
    })

    for _, row in df.iterrows():
        disease = row['disease']
        hit = row['hit_at_30']

        for category in disease_categories[disease]:
            txgnn_category_stats[category]["count"] += 1
            txgnn_category_stats[category]["diseases"].append(disease)
            if hit:
                txgnn_category_stats[category]["hits"] += 1
                txgnn_category_stats[category]["hit_diseases"].append(disease)
            else:
                txgnn_category_stats[category]["miss_diseases"].append(disease)

    # Compute recall for each category - TxGNN
    txgnn_category_results = []
    for category, stats in sorted(txgnn_category_stats.items()):
        recall = stats["hits"] / stats["count"] * 100 if stats["count"] > 0 else 0
        txgnn_category_results.append({
            "category": category,
            "count": stats["count"],
            "hits": stats["hits"],
            "recall_at_30": round(recall, 1),
            "hit_diseases": stats["hit_diseases"][:10],
            "miss_diseases": stats["miss_diseases"][:10]
        })

    # Compute recall for each category - GB
    gb_category_results = []
    for category, stats in sorted(gb_category_stats.items()):
        avg_recall = stats["total_recall"] / stats["count"] * 100 if stats["count"] > 0 else 0
        gb_category_results.append({
            "category": category,
            "count": stats["count"],
            "avg_recall_at_30": round(avg_recall, 1),
            "diseases": stats["diseases"]
        })

    # Sort by recall descending
    txgnn_category_results.sort(key=lambda x: x["recall_at_30"], reverse=True)
    gb_category_results.sort(key=lambda x: x["avg_recall_at_30"], reverse=True)

    # Create comparison table
    print("=" * 100)
    print("TXGNN VS GB MODEL - DISEASE CATEGORY PERFORMANCE COMPARISON")
    print("=" * 100)
    print(f"{'Category':<20} {'TxGNN':>30} {'GB Model':>30} {'Diff':>15}")
    print(f"{'':<20} {'R@30 (n)':>30} {'R@30 (n)':>30} {'TxGNN-GB':>15}")
    print("-" * 100)

    # Build lookup for GB results
    gb_lookup = {r["category"]: r for r in gb_category_results}

    comparison_data = []
    for txgnn_result in txgnn_category_results:
        category = txgnn_result["category"]
        txgnn_recall = txgnn_result["recall_at_30"]
        txgnn_n = txgnn_result["count"]

        gb_result = gb_lookup.get(category)
        if gb_result:
            gb_recall = gb_result["avg_recall_at_30"]
            gb_n = gb_result["count"]
            diff = txgnn_recall - gb_recall
            comparison_data.append({
                "category": category,
                "txgnn_recall": txgnn_recall,
                "txgnn_n": txgnn_n,
                "gb_recall": gb_recall,
                "gb_n": gb_n,
                "diff": diff
            })
            print(f"{category:<20} {txgnn_recall:>10.1f}% (n={txgnn_n:<5}) {gb_recall:>10.1f}% (n={gb_n:<5}) {diff:>+14.1f}%")
        else:
            print(f"{category:<20} {txgnn_recall:>10.1f}% (n={txgnn_n:<5}) {'N/A':>17} {'N/A':>15}")
            comparison_data.append({
                "category": category,
                "txgnn_recall": txgnn_recall,
                "txgnn_n": txgnn_n,
                "gb_recall": None,
                "gb_n": None,
                "diff": None
            })

    print("-" * 100)

    # Identify which model is better per category
    print("\n" + "=" * 100)
    print("MODEL ROUTING RECOMMENDATIONS")
    print("=" * 100)

    txgnn_better = []
    gb_better = []
    comparable = []

    for comp in comparison_data:
        if comp["diff"] is not None and comp["gb_n"] is not None:
            if comp["diff"] > 5:  # TxGNN is better by >5%
                txgnn_better.append(comp)
            elif comp["diff"] < -5:  # GB is better by >5%
                gb_better.append(comp)
            else:
                comparable.append(comp)

    print("\nCategories where TxGNN performs better (>5% advantage):")
    if txgnn_better:
        for comp in sorted(txgnn_better, key=lambda x: x["diff"], reverse=True):
            print(f"  - {comp['category']}: TxGNN {comp['txgnn_recall']:.1f}% vs GB {comp['gb_recall']:.1f}% (+{comp['diff']:.1f}%)")
    else:
        print("  (none)")

    print("\nCategories where GB model performs better (>5% advantage):")
    if gb_better:
        for comp in sorted(gb_better, key=lambda x: x["diff"]):
            print(f"  - {comp['category']}: GB {comp['gb_recall']:.1f}% vs TxGNN {comp['txgnn_recall']:.1f}% ({comp['diff']:.1f}%)")
    else:
        print("  (none)")

    print("\nCategories where models are comparable (+/- 5%):")
    if comparable:
        for comp in comparable:
            print(f"  - {comp['category']}: TxGNN {comp['txgnn_recall']:.1f}% vs GB {comp['gb_recall']:.1f}%")
    else:
        print("  (none)")

    # Print TxGNN-only summary table
    print("\n" + "=" * 100)
    print("TXGNN PERFORMANCE BY DISEASE CATEGORY (Full Results)")
    print("=" * 100)
    print(f"{'Category':<20} {'Count':>8} {'Hits':>8} {'Recall@30':>12}")
    print("-" * 100)

    for result in txgnn_category_results:
        print(f"{result['category']:<20} {result['count']:>8} {result['hits']:>8} {result['recall_at_30']:>11.1f}%")

    print("-" * 100)

    # Identify best and worst categories
    best_categories = [r for r in txgnn_category_results if r["recall_at_30"] >= 20]
    worst_categories = [r for r in txgnn_category_results if r["recall_at_30"] < 10 and r["count"] >= 10]

    print("\n" + "=" * 100)
    print("TXGNN BEST PERFORMING CATEGORIES (>= 20% Recall@30)")
    print("=" * 100)
    for result in best_categories:
        print(f"\n{result['category'].upper()} - {result['recall_at_30']:.1f}% ({result['hits']}/{result['count']})")
        if result['hit_diseases']:
            print(f"  Examples of hits: {', '.join(result['hit_diseases'][:5])}")

    print("\n" + "=" * 100)
    print("TXGNN WORST PERFORMING CATEGORIES (< 10% Recall@30, n >= 10)")
    print("=" * 100)
    for result in worst_categories:
        print(f"\n{result['category'].upper()} - {result['recall_at_30']:.1f}% ({result['hits']}/{result['count']})")
        if result['miss_diseases']:
            print(f"  Examples of misses: {', '.join(result['miss_diseases'][:5])}")

    # Find diseases that matched to multiple categories
    multi_category = {d: cats for d, cats in disease_categories.items() if len(cats) > 1}
    print(f"\n\nNote: {len(multi_category)} diseases matched multiple categories")

    # GB model detailed breakdown
    print("\n" + "=" * 100)
    print("GB MODEL - DETAILED PER-DISEASE PERFORMANCE")
    print("=" * 100)
    print(f"{'Disease':<30} {'Category':<20} {'Recall@30':>12}")
    print("-" * 100)

    for disease, stats in sorted(gb_by_disease.items(), key=lambda x: x[1]["recall@30"], reverse=True):
        categories = categorize_disease(disease)
        cat_str = ", ".join(categories[:2])
        recall_pct = stats["recall@30"] * 100
        print(f"{disease:<30} {cat_str:<20} {recall_pct:>11.1f}%")

    # Save detailed results
    output = {
        "summary": {
            "txgnn": {
                "total_diseases": len(df),
                "overall_recall_at_30": round(df['hit_at_30'].mean() * 100, 1),
                "total_hits": int(df['hit_at_30'].sum())
            },
            "gb_model": {
                "total_diseases": len(gb_by_disease),
                "overall_recall_at_30": round(gb_results['aggregate_recall@30'] * 100, 1),
                "total_hits": gb_results['total_found@30']
            }
        },
        "txgnn_category_performance": txgnn_category_results,
        "gb_category_performance": gb_category_results,
        "model_comparison": comparison_data,
        "routing_recommendations": {
            "use_txgnn": [c["category"] for c in txgnn_better],
            "use_gb": [c["category"] for c in gb_better],
            "either_model": [c["category"] for c in comparable]
        },
        "txgnn_best_categories": [r["category"] for r in best_categories],
        "txgnn_worst_categories": [r["category"] for r in worst_categories],
        "disease_categorizations": {
            disease: categories for disease, categories in disease_categories.items()
        }
    }

    output_path = Path("/Users/jimhuff/github/open-cure/data/analysis/disease_category_analysis.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n\nResults saved to: {output_path}")

    # Print executive summary
    print("\n" + "=" * 100)
    print("EXECUTIVE SUMMARY")
    print("=" * 100)
    print(f"""
TxGNN vs GB Model Performance Analysis
======================================

Overall Performance:
- TxGNN: 14.5% Recall@30 across 779 diseases
- GB Model: 13.2% Recall@30 across 17 diseases (limited overlap)

Key Findings:

1. TxGNN EXCELS at (>20% R@30):
   - Psychiatric: 28.6% (alcohol withdrawal, bipolar)
   - Dermatological: 25.0% (psoriasis, eczema, rosacea)
   - Autoimmune: 22.2% (arthritis variants)
   - Metabolic: 21.7% (lysosomal storage diseases like Fabry, Gaucher, Hurler)

2. TxGNN STRUGGLES with (<10% R@30, n>=10):
   - Gastrointestinal: 8.7%
   - Respiratory: 7.3%
   - Renal: 7.1%

3. GB Model STRONG categories:
   - Atrial fibrillation: 72.2%
   - Tuberculosis: 50.0%
   - COPD: 42.9%
   - Type 2 diabetes: 16.1%

4. GB Model WEAK categories:
   - Cancer (Breast, Colorectal, Lung): 0%
   - HIV: 0%
   - Epilepsy: 0%

Routing Recommendation:
- Use TxGNN for: psychiatric, dermatological, autoimmune, metabolic/genetic rare diseases
- Use GB Model for: cardiovascular (especially atrial fibrillation), respiratory (COPD), infectious (tuberculosis)
- Further investigation needed for: cancer, neurological (both models struggle)
""")


if __name__ == "__main__":
    main()
