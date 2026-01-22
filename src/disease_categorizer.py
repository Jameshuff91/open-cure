#!/usr/bin/env python3
"""
Keyword-based disease categorizer for model routing.
Assigns diseases to categories based on name patterns.
"""

import re
from typing import Optional, Dict, List

# Category definitions with keyword patterns
CATEGORY_PATTERNS: Dict[str, List[str]] = {
    "storage": [
        "storage", "lysosomal", "gaucher", "fabry", "hurler", "scheie",
        "niemann-pick", "pompe", "mucopolysaccharidosis", "mps ", "mps-",
        "hunter syndrome", "morquio", "maroteaux-lamy", "sanfilippo",
        "glycogenosis", "glycogen storage", "ceroid", "batten"
    ],
    "cancer": [
        "cancer", "carcinoma", "lymphoma", "leukemia", "tumor", "melanoma",
        "sarcoma", "myeloma", "neoplasm", "malignant", "oncological",
        "glioma", "glioblastoma", "mesothelioma", "neuroblastoma",
        "retinoblastoma", "hepatoblastoma", "nephroblastoma", "blastoma"
    ],
    "metabolic": [
        "metabolic", "diabetes", "hyperlipidemia", "hypercholesterolemia",
        "porphyria", "phenylketonuria", "homocystinuria", "thyroid",
        "hypothyroid", "hyperthyroid", "adrenal", "cushing", "addison",
        "acromegaly", "obesity", "gout", "hyperuricemia", "maple syrup"
    ],
    "autoimmune": [
        "autoimmune", "rheumatoid", "lupus", "psoriasis", "psoriatic",
        "scleroderma", "sjogren", "vasculitis", "myasthenia", "guillain",
        "multiple sclerosis", "inflammatory bowel", "crohn", "colitis",
        "pemphigus", "pemphigoid", "vitiligo", "alopecia areata",
        "ankylosing spondylitis", "behcet"
    ],
    "neurological": [
        "alzheimer", "parkinson", "huntington", "dementia", "epilepsy",
        "seizure", "neuropathy", "neurodegenerative", "ataxia", "dystonia",
        "amyotrophic", "als", "motor neuron", "spinal muscular",
        "migraine", "headache", "stroke", "cerebral"
    ],
    "cardiovascular": [
        "cardiac", "heart", "cardiovascular", "arrhythmia", "atrial",
        "ventricular", "hypertension", "hypotension", "angina", "coronary",
        "myocardial", "cardiomyopathy", "heart failure", "atherosclerosis",
        "thrombosis", "embolism", "aneurysm"
    ],
    "respiratory": [
        "respiratory", "pulmonary", "lung", "asthma", "copd", "bronchi",
        "pneumonia", "fibrosis", "emphysema", "cystic fibrosis",
        "tuberculosis", "pleural", "sleep apnea"
    ],
    "infectious": [
        "infection", "infectious", "bacterial", "viral", "fungal",
        "parasitic", "hiv", "aids", "hepatitis", "tuberculosis", "malaria",
        "sepsis", "meningitis", "pneumonia", "influenza", "covid",
        "herpes", "cytomegalovirus", "cmv", "otitis", "sinusitis",
        "amebiasis", "actinomycosis", "yaws", "leprosy", "chagas",
        "leishmaniasis", "trypanosomiasis", "clostridium", "candidiasis",
        "aspergillosis"
    ],
    "renal": [
        "renal", "kidney", "nephro", "glomerulo", "nephrotic", "dialysis",
        "uremia", "polycystic kidney"
    ],
    "gastrointestinal": [
        "gastrointestinal", "intestinal", "bowel", "gastric", "stomach",
        "liver", "hepatic", "cirrhosis", "pancreat", "esophag", "colon",
        "rectal", "celiac", "gastritis"
    ],
    "psychiatric": [
        "psychiatric", "mental", "depression", "anxiety", "bipolar",
        "schizophrenia", "psychosis", "obsessive", "ocd", "ptsd",
        "panic disorder", "eating disorder", "anorexia", "bulimia",
        "phobia", "withdrawal", "addiction", "substance abuse"
    ],
    "dermatological": [
        "skin", "dermat", "eczema", "psoriasis", "acne", "rosacea",
        "urticaria", "pruritus", "ichthyosis", "epidermolysis",
        "pemphigus", "vitiligo"
    ],
    "hematological": [
        "anemia", "hemophilia", "thrombocytopenia", "polycythemia",
        "thalassemia", "sickle cell", "blood disorder", "coagulation",
        "bleeding disorder", "von willebrand", "platelet"
    ],
    "ophthalmological": [
        "eye", "retina", "retinal", "macular", "glaucoma", "cataract",
        "uveitis", "optic", "cornea", "kerato"
    ],
    "musculoskeletal": [
        "muscle", "muscular", "dystrophy", "myopathy", "arthritis",
        "osteo", "bone", "joint", "fibromyalgia", "osteogenesis"
    ],
    "genetic": [
        "syndrome", "congenital", "hereditary", "familial", "inherited",
        "chromosom", "genetic"
    ],
    "endocrine": [
        "endocrine", "hormone", "pituitary", "thyroid", "parathyroid",
        "adrenal", "gonad"
    ],
    "immune": [
        "immunodeficiency", "agammaglobulinemia", "hypogammaglobulinemia",
        "angioedema", "anaphylaxis", "mastocytosis", "allergic",
        "complement deficiency", "scid"
    ],
    "rare_genetic": [
        "achondroplasia", "alkaptonuria", "amyloidosis", "wilson disease",
        "hemochromatosis", "charcot-marie", "ehlers-danlos", "marfan",
        "neurofibromatosis", "tuberous sclerosis", "huntington"
    ]
}

# Categories where TxGNN excels (R@30 > 20% based on GPU evaluation)
# Storage: 83.3%, Psychiatric: 28.6%, Dermatological: 25.0%,
# Autoimmune: 22.2%, Metabolic: 21.7%
TXGNN_PREFERRED = {"storage", "psychiatric", "dermatological", "autoimmune", "metabolic"}

# Categories where TxGNN performs poorly (R@30 < 12%) - use best_rank
# Respiratory: 7.3%, Renal: 7.1%, Gastrointestinal: 8.7%, Cancer: 11.7%, Hematological: 10.8%
BEST_RANK_PREFERRED = {"respiratory", "renal", "gastrointestinal", "cancer", "hematological"}

# GB model not used - best_rank ensemble is better
GB_PREFERRED: set[str] = set()  # Empty - best_rank is always better than pure GB


def categorize_disease(disease_name: str) -> Optional[str]:
    """
    Categorize a disease based on name patterns.
    Returns the most specific matching category.
    """
    disease_lower = disease_name.lower()

    matches = []
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if pattern in disease_lower:
                matches.append((category, len(pattern)))
                break  # One match per category is enough

    if not matches:
        return None

    # Return category with longest matching pattern (most specific)
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0]


def get_model_preference(category: Optional[str]) -> str:
    """
    Get model preference for a disease category.
    Returns: "txgnn" or "best_rank"

    Strategy:
    - Use TxGNN for categories where it excels (R@30 > 20%)
    - Use best_rank ensemble for everything else
    """
    if category is None:
        return "best_rank"
    if category in TXGNN_PREFERRED:
        return "txgnn"
    # Everything else uses best_rank (including BEST_RANK_PREFERRED and unknown)
    return "best_rank"


def categorize_all_diseases(disease_names: List[str]) -> Dict[str, Dict]:
    """
    Categorize all diseases and return summary.
    """
    results = {}
    category_counts: Dict[str, int] = {}
    uncategorized = []

    for disease in disease_names:
        category = categorize_disease(disease)
        model = get_model_preference(category)

        results[disease] = {
            "category": category,
            "model_preference": model
        }

        if category:
            category_counts[category] = category_counts.get(category, 0) + 1
        else:
            uncategorized.append(disease)

    return {
        "results": results,
        "category_counts": category_counts,
        "uncategorized": uncategorized,
        "coverage": len(disease_names) - len(uncategorized),
        "total": len(disease_names)
    }


if __name__ == "__main__":
    import json

    # Load GT diseases
    with open('data/reference/everycure_gt_for_txgnn.json', 'r') as f:
        gt = json.load(f)

    disease_names = list(gt.keys())
    print(f"Categorizing {len(disease_names)} diseases...")

    result = categorize_all_diseases(disease_names)

    print(f"\nCoverage: {result['coverage']}/{result['total']} ({result['coverage']/result['total']*100:.1f}%)")
    print(f"\nCategory counts:")
    for cat, count in sorted(result['category_counts'].items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    print(f"\nUncategorized diseases ({len(result['uncategorized'])}):")
    for d in sorted(result['uncategorized'])[:20]:
        print(f"  - {d}")

    # Model routing summary
    txgnn_count = sum(1 for r in result['results'].values() if r['model_preference'] == 'txgnn')
    gb_count = sum(1 for r in result['results'].values() if r['model_preference'] == 'gb')
    best_rank_count = sum(1 for r in result['results'].values() if r['model_preference'] == 'best_rank')

    print(f"\nModel routing:")
    print(f"  TxGNN preferred: {txgnn_count}")
    print(f"  GB preferred: {gb_count}")
    print(f"  Best rank fallback: {best_rank_count}")

    # Save results
    output_file = 'data/analysis/keyword_disease_categories.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {output_file}")
