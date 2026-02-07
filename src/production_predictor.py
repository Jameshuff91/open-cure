#!/usr/bin/env python3
"""
Production Drug Repurposing Predictor

Unified pipeline integrating validated research findings:
- h39: kNN collaborative filtering with Node2Vec embeddings (best method)
- h135: Production tiered confidence system (GOLDEN/HIGH/MEDIUM/LOW/FILTER)
- h136: Category-specific filters for Tier 2/3 rescue (infectious, cardiovascular, respiratory)
- h144: Metabolic statin rescue - statins + rank<=10 = 60% precision
- h150: Drug class rescue criteria (validated):
  - Cancer: taxane + rank<=5 = 40%, alkylating + rank<=10 = 36.4%
  - Ophthalmic: antibiotic + rank<=15 = 62.5%, steroid + rank<=15 = 48%
  - Dermatological: topical_steroid + rank<=5 = 63.6%
- h197: Colorectal cancer mAb rescue - colorectal + mAb = 50-60% precision (GOLDEN)
- h201: Disease-specific kinase rules (from h198):
  - CML/ALL + BCR-ABL inhibitors (imatinib/nilotinib/dasatinib) = 22% precision (HIGH)
  - CLL/lymphoma + BTK inhibitors (ibrutinib/acalabrutinib/zanubrutinib) = 22% precision (HIGH)
- h154/h266: Cardiovascular beta_blocker + rank<=10 = 42.1% precision
- h157: Autoimmune DMARD + rank<=10 = 75.4% precision
- h170: Selective category boosting - +2.40pp R@30 (p=0.009) for isolated categories
  - Boosts same-category neighbors 1.5x for: neurological, respiratory, metabolic, renal, hematological, immunological
- h187: Neurological anticonvulsant rescue - anticonvulsant + rank<=10 + mech = 58.8% precision
- h189: ATC L4-based rescue criteria (more systematic drug class identification)
  - Autoimmune: H02AB (glucocorticoids) 77%, L04AX (methotrexate/azathioprine) 82%
  - Excludes biologics (L04AB, L04AC, L04AF) which have low precision (8-17%)
- h309/h310: ATC coherence boost - drugs matching expected ATC for disease category
  - Coherent: 35.5% precision vs Incoherent: 18.7% (+16.8 pp gap)
  - Boosts LOW→MEDIUM for coherent predictions with rank<=10 + evidence
- h311: ATC incoherence demotion - demote incoherent high-tier predictions
  - Incoherent GOLDEN: 22.5% precision vs Coherent: 30.3% (+7.8 pp improvement)
  - Demotes GOLDEN→HIGH and HIGH→MEDIUM when drug ATC doesn't match category
- h314/h316: Zero-precision ATC mismatch FILTER
  - 16 ATC→category pairs with <3% precision (e.g., A/J/N→cancer, B→other)
  - Filter removes 1,319 predictions with 1.21% precision (+0.78 pp overall)
  - NOTE: High-precision mismatches (10-27%) still demote to HIGH (not GOLDEN)
- h318: Antibiotic FILTER for non-infectious diseases
  - J drugs have 0% precision for: hematological, gastrointestinal, metabolic, immune, rare_genetic
  - Exception: J→respiratory = 17.8% (kept as HIGH_PRECISION_MISMATCH)
  - Filters additional 180 predictions with 0% precision (0 hits lost)
- h319: Comprehensive Low-Precision ATC Filter (Batch 2)
  - Added 22 additional 0% precision ATC→category pairs
  - Includes: C/N drugs for non-CV/non-neuro, L for non-cancer/non-autoimmune, etc.
  - Filters additional 703 predictions with 0% precision (0 hits lost)
- h317: Refine HIGH_PRECISION_MISMATCHES with 5 missing patterns
  - B→respiratory (30%), N→autoimmune (29%), C→autoimmune (27%)
  - N→dermatological (14%), C→genetic (12%)
  - These "incoherent" pairs have HIGHER precision than coherent baseline (11.7%)
- h326: Broad Class Isolation Demotion (from h307 Lidocaine analysis)
  - Drugs from "broad" classes (anesthetics, steroids, TNFi, NSAIDs) predicted alone have 1.9% precision
  - vs 12.7% when classmates are also predicted for same disease
  - HIGH tier has 0% precision for isolated broad-class drugs → demote to LOW
  - 212 predictions affected, 4 hits (only MEDIUM tier, no HIGH hits lost)
- h328: Class Cohesion Analysis - Added IL inhibitors to broad class list
  - IL inhibitors have strongest cohesion effect: 3% alone vs 50% with classmates (+47 pp!)
  - Additional 62 predictions affected (7.4% HIGH, 0% MEDIUM precision)
- h346: Cancer-Only Drug Filter (generalizes h340 MEK inhibitor filter)
  - 69 cancer-only drugs (BRAF, PD-1, BCL2, PARP, ALK, EGFR, etc.) have 0% precision for non-cancer
  - 115 non-cancer predictions, 0 GT hits → FILTER tier (zero recall loss)
  - Excludes drugs with non-cancer uses: mTOR inhibitors, imatinib, ranibizumab, aflibercept
- h353: Complication-Specific Drug Class Filter
  - Complication diseases (nephropathy, retinopathy, cardiomyopathy, neuropathy) only respond to specific drug classes
  - Validated drugs: 12.5-69.2% precision vs Non-validated: 0.0% precision
  - 214 predictions filtered, 0 GT hits lost (100% filter accuracy)
  - Preserves steroids+immunosuppressants for nephrotic, anti-VEGF for retinopathy, etc.
- h354: CV Pathway-Comprehensive Boost (from h351 analysis)
  - Drugs with GT for BOTH CV base (hypertension/lipids) AND CV complications boost to HIGH
  - Pathway-comprehensive: 48.8% precision vs non-pathway: 7.6% (+41.2 pp, 6.4x lift)
  - 109 pathway-comprehensive drugs (statins, ACEi, ARBs, beta-blockers, SGLT2i, etc.)
  - CV complications: heart failure, stroke, MI, angina, peripheral vascular, cardiomyopathy

USAGE:
    # Get predictions for a disease
    from production_predictor import DrugRepurposingPredictor
    predictor = DrugRepurposingPredictor()
    results = predictor.predict("rheumatoid arthritis")

    # CLI usage
    python -m src.production_predictor "rheumatoid arthritis"
    python -m src.production_predictor --disease "type 2 diabetes" --top-k 30

TIER SYSTEM (h478 holdout-validated, use holdout as authoritative):
- GOLDEN (~67% holdout): Hierarchy match + high-freq/mechanism rescue rules
- HIGH (~61% holdout):   freq>=15 + mechanism OR rank<=5 + freq>=10 + mechanism
- MEDIUM (~31% holdout): freq>=5 + mechanism OR freq>=10 OR ATC coherent
- LOW (~15% holdout):    All else passing filter + demoted categories
- FILTER (~10% holdout): rank>20 OR no_targets OR harmful patterns
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

# h189: ATC-based drug classification (lazy import to avoid circular dependency)
_atc_mapper: Optional["ATCMapper"] = None  # type: ignore[name-defined]


def _get_atc_mapper() -> "ATCMapper":  # type: ignore[name-defined]
    """Lazy load ATCMapper to avoid import overhead when not needed."""
    global _atc_mapper
    if _atc_mapper is None:
        from src.atc_features import ATCMapper
        _atc_mapper = ATCMapper()
    return _atc_mapper


class ConfidenceTier(Enum):
    """Confidence tiers from h135 (validated 9.1x precision separation)."""
    GOLDEN = "GOLDEN"    # ~57.7% precision
    HIGH = "HIGH"        # ~20.9% precision
    MEDIUM = "MEDIUM"    # ~14.3% precision
    LOW = "LOW"          # ~6.4% precision
    FILTER = "FILTER"    # ~3.2% precision (excluded)


# h165/h167: Category-specific precision calibration
# Key finding: Overall tier precision is massively miscalibrated by category
# Psychiatric MEDIUM = 85% vs Other MEDIUM = 17%
CATEGORY_PRECISION = {
    # h398: All values measured with current code on full dataset (455 diseases)
    # Replaces stale h165/h136/h150 estimates. Only entries with n>=10 included.
    # Format: (category, tier) -> precision percentage
    #
    # Autoimmune
    ("autoimmune", "GOLDEN"): 72.0,   # n=93
    ("autoimmune", "HIGH"): 70.0,     # n=120
    ("autoimmune", "MEDIUM"): 28.7,   # n=240
    ("autoimmune", "LOW"): 9.9,       # n=81
    ("autoimmune", "FILTER"): 16.6,   # n=362
    # Cancer (GOLDEN/HIGH have n<10, use tier defaults)
    ("cancer", "MEDIUM"): 23.4,       # n=1321 (includes cancer_same_type)
    ("cancer", "LOW"): 7.2,           # n=209
    ("cancer", "FILTER"): 5.7,        # n=653
    # Cardiovascular
    ("cardiovascular", "GOLDEN"): 58.6,  # n=29
    ("cardiovascular", "HIGH"): 43.2,    # n=229
    ("cardiovascular", "MEDIUM"): 25.0,  # n=200
    ("cardiovascular", "LOW"): 15.2,     # n=145
    ("cardiovascular", "FILTER"): 16.0,  # n=567
    # Dermatological
    ("dermatological", "GOLDEN"): 45.2,  # n=42
    ("dermatological", "MEDIUM"): 29.4,  # n=255
    ("dermatological", "LOW"): 9.4,      # n=117
    ("dermatological", "FILTER"): 8.5,   # n=484
    # Endocrine
    ("endocrine", "MEDIUM"): 24.0,    # n=25
    ("endocrine", "LOW"): 25.0,       # n=12 (inversion, small n)
    ("endocrine", "FILTER"): 19.6,    # n=51
    # Gastrointestinal (h462/h463: MEDIUM demoted to LOW)
    ("gastrointestinal", "HIGH"): 31.4,   # n=35
    ("gastrointestinal", "LOW"): 9.3,     # n=54 (includes former MEDIUM)
    ("gastrointestinal", "FILTER"): 10.8, # n=509
    # Hematological (h553: MEDIUM demoted to LOW; 10.0% ± 20.0% holdout, n=8/seed)
    ("hematological", "LOW"): 6.5,      # n=93 (includes former MEDIUM)
    ("hematological", "FILTER"): 5.4,   # n=349
    # Immunological (h462: MEDIUM demoted to LOW; 2.5% holdout = massive overfitting)
    ("immunological", "LOW"): 9.1,      # n=22 (includes former MEDIUM)
    ("immunological", "FILTER"): 16.0,  # n=94
    # Infectious
    ("infectious", "HIGH"): 53.7,     # n=95
    ("infectious", "MEDIUM"): 27.4,   # n=797
    ("infectious", "LOW"): 14.0,      # n=463
    ("infectious", "FILTER"): 11.9,   # n=1156
    # Metabolic
    ("metabolic", "GOLDEN"): 42.7,    # n=157
    ("metabolic", "HIGH"): 17.6,      # n=51
    ("metabolic", "MEDIUM"): 15.1,    # n=86
    ("metabolic", "LOW"): 6.1,        # n=196
    ("metabolic", "FILTER"): 9.1,     # n=822
    # Musculoskeletal
    ("musculoskeletal", "MEDIUM"): 38.9,  # n=54
    ("musculoskeletal", "LOW"): 9.7,      # n=31
    ("musculoskeletal", "FILTER"): 0.9,   # n=117
    # Neurological (h462: MEDIUM demoted to LOW; 10.2% holdout)
    ("neurological", "GOLDEN"): 50.0,  # n=20
    ("neurological", "LOW"): 9.2,      # n=76 (includes former MEDIUM)
    ("neurological", "FILTER"): 5.9,   # n=576
    # Ophthalmic
    ("ophthalmic", "GOLDEN"): 50.0,   # n=12
    ("ophthalmic", "MEDIUM"): 29.0,   # n=138
    ("ophthalmic", "LOW"): 5.6,       # n=54
    ("ophthalmic", "FILTER"): 7.2,    # n=305
    # Other
    ("other", "MEDIUM"): 30.0,        # n=50
    ("other", "LOW"): 8.5,            # n=59
    ("other", "FILTER"): 11.6,        # n=249
    # Psychiatric
    ("psychiatric", "MEDIUM"): 52.1,   # n=117
    ("psychiatric", "LOW"): 7.1,       # n=14
    ("psychiatric", "FILTER"): 26.3,   # n=167
    # Renal
    ("renal", "MEDIUM"): 23.1,        # n=39
    ("renal", "LOW"): 15.2,           # n=46
    ("renal", "FILTER"): 17.0,        # n=212
    # Reproductive (h462: MEDIUM demoted to LOW; 0.0% holdout)
    ("reproductive", "LOW"): 17.4,     # n=23 (includes former MEDIUM; inversion, small n)
    ("reproductive", "FILTER"): 4.0,   # n=101
    # Respiratory
    ("respiratory", "HIGH"): 48.8,    # n=41
    ("respiratory", "MEDIUM"): 17.9,  # n=123
    ("respiratory", "LOW"): 5.6,      # n=107
    ("respiratory", "FILTER"): 16.3,  # n=203
}

# Default tier-only precision (fallback)
# h402: Updated from h402 holdout validation after rule demotions
# Previous: h396 values from h393 holdout
DEFAULT_TIER_PRECISION = {
    # h498: Updated to holdout-validated numbers (h478 GT sync + subsequent fixes)
    # NOTE: These are 5-seed HOLDOUT precisions, the authoritative metric.
    # Full-data precision is inflated by GT self-prediction and should not be used.
    "GOLDEN": 67.0,   # h478: 67.0% ± 20.6% holdout (full-data 89.2% is inflated)
    "HIGH": 60.8,     # h478: 60.8% ± 7.2% holdout (full-data 76.8% is inflated)
    "MEDIUM": 30.8,   # h478: 30.8% ± 3.4% holdout (full-data 40.5% is inflated)
    "LOW": 14.8,      # h478: 14.8% ± 2.3% holdout (full-data 21.4% is inflated)
    "FILTER": 10.3,   # h478: 10.3% ± 1.1% holdout (full-data 16.8% is inflated)
}


def get_category_precision(category: str, tier: str) -> float:
    """Get precision estimate for a (category, tier) pair.

    Returns category-specific precision if available (h165),
    otherwise falls back to default tier precision (h135).
    """
    key = (category.lower(), tier)
    if key in CATEGORY_PRECISION and CATEGORY_PRECISION[key] is not None:
        return CATEGORY_PRECISION[key]
    return DEFAULT_TIER_PRECISION.get(tier, 10.0)


@dataclass
class DrugPrediction:
    """A single drug prediction with confidence metadata."""
    drug_name: str
    drug_id: str
    rank: int
    knn_score: float
    norm_score: float
    confidence_tier: ConfidenceTier
    train_frequency: int
    mechanism_support: bool
    has_targets: bool
    category: str
    disease_tier: int

    # Category-specific rescue criteria (h136)
    category_rescue_applied: bool = False
    category_specific_tier: Optional[str] = None

    # h405/h439: TransE consilience flag
    transe_consilience: bool = False

    # h444: Rank-bucket precision (holdout-validated)
    rank_bucket_precision: float = 0.0

    # h466: Category-specific holdout precision
    category_holdout_precision: float = 0.0

    # h481: Literature status annotation
    literature_status: str = 'NOVEL'
    soc_drug_class: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'drug': self.drug_name,
            'drug_id': self.drug_id,
            'rank': self.rank,
            'score': float(self.knn_score),
            'norm_score': float(self.norm_score),
            'confidence_tier': self.confidence_tier.value,
            'train_frequency': self.train_frequency,
            'mechanism_support': self.mechanism_support,
            'has_targets': self.has_targets,
            'category': self.category,
            'category_rescue_applied': self.category_rescue_applied,
            'transe_consilience': self.transe_consilience,
            'rank_bucket_precision': self.rank_bucket_precision,
            'category_holdout_precision': self.category_holdout_precision,
            'literature_status': self.literature_status,
            'soc_drug_class': self.soc_drug_class,
        }


@dataclass
class PredictionResult:
    """Complete prediction result for a disease."""
    disease_name: str
    disease_id: Optional[str]
    category: str
    disease_tier: int
    predictions: List[DrugPrediction]
    neighbors_used: int
    coverage_warning: Optional[str] = None

    def get_by_tier(self, tier: ConfidenceTier) -> List[DrugPrediction]:
        """Get predictions filtered by confidence tier."""
        return [p for p in self.predictions if p.confidence_tier == tier]

    def summary(self) -> Dict:
        """Get summary statistics."""
        tier_counts = defaultdict(int)
        for p in self.predictions:
            tier_counts[p.confidence_tier.value] += 1

        # h167: Add category-specific precision estimates
        precision_by_tier = {}
        for tier_name in tier_counts.keys():
            precision_by_tier[tier_name] = get_category_precision(self.category, tier_name)

        return {
            'disease': self.disease_name,
            'category': self.category,
            'tier': self.disease_tier,
            'total_predictions': len(self.predictions),
            'by_tier': dict(tier_counts),
            'category_precision_by_tier': precision_by_tier,  # h167
            'coverage_warning': self.coverage_warning,
        }


# Category definitions (from h71/h135)
TIER_1_CATEGORIES = {'autoimmune', 'dermatological', 'psychiatric', 'ophthalmic'}
TIER_2_CATEGORIES = {'cardiovascular', 'other', 'cancer'}

# h144: Statins achieve 60% precision for metabolic diseases (vs 6% baseline)
STATIN_DRUGS = {
    'atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin', 'lovastatin',
    'fluvastatin', 'pitavastatin', 'cerivastatin',
}

# h150: Corticosteroids achieve 48.6% precision for hematological diseases
# Mechanism support NOT required (works through immunosuppression)
CORTICOSTEROID_DRUGS = {
    'prednisone', 'prednisolone', 'methylprednisolone', 'dexamethasone', 'hydrocortisone',
    'betamethasone', 'triamcinolone', 'fluticasone', 'budesonide', 'beclomethasone',
    'cortisone', 'fludrocortisone', 'mometasone',  # h476: added missing corticosteroids
}

# h520: Corticosteroid SOC promotion constants
# Non-hematological categories have 50.1% holdout precision, comparable to HIGH (51.5%)
# Hematological excluded: 19.1% holdout, below MEDIUM avg
_CORTICOSTEROID_LOWER = {d.lower() for d in CORTICOSTEROID_DRUGS}
_CORTICOSTEROID_SOC_PROMOTE_CATEGORIES = {
    'autoimmune', 'dermatological', 'respiratory', 'ophthalmic',
}

# h150: Drug class rescue criteria
# Cancer drugs
TAXANE_DRUGS = {'paclitaxel', 'docetaxel', 'cabazitaxel', 'nab-paclitaxel'}  # 40% precision rank<=5
ALKYLATING_DRUGS = {'cyclophosphamide', 'ifosfamide', 'melphalan', 'chlorambucil', 'busulfan'}  # 36.4% rank<=10

# h197: Colorectal cancer mAb detection
# colorectal_cancer + monoclonal_antibody = 50-60% precision (validated h160)
COLORECTAL_KEYWORDS = {'colorectal', 'colon cancer', 'rectal cancer', 'bowel cancer'}
# Known colorectal mAbs: bevacizumab (Avastin), cetuximab (Erbitux), panitumumab (Vectibix)
COLORECTAL_MABS = {'bevacizumab', 'cetuximab', 'panitumumab', 'ramucirumab'}  # 50-60% precision

# h215: CDK4/6 inhibitors for breast cancer (100% precision)
BREAST_CANCER_KEYWORDS = {'breast cancer', 'breast carcinoma', 'mammary cancer'}
CDK_INHIBITORS = {'palbociclib', 'ribociclib', 'abemaciclib'}  # 100% precision for breast cancer

# h201: Disease-specific kinase inhibitor rules (from h198 analysis)
# BCR-ABL inhibitors for CML/ALL (22% precision in h198)
CML_KEYWORDS = {'chronic myeloid leukemia', 'cml', 'chronic myelogenous'}
ALL_KEYWORDS = {'acute lymphoblastic leukemia', 'all', 'acute lymphocytic'}
BCR_ABL_INHIBITORS = {'imatinib', 'nilotinib', 'dasatinib', 'ponatinib', 'bosutinib'}

# BTK inhibitors for CLL/lymphoma (22% precision in h198)
CLL_KEYWORDS = {'chronic lymphocytic leukemia', 'cll', 'small lymphocytic lymphoma', 'sll'}
LYMPHOMA_BTK_KEYWORDS = {'lymphoplasmacytic', 'mantle cell lymphoma', 'marginal zone lymphoma',
                         'waldenstrom', 'macroglobulinemia'}
BTK_INHIBITORS = {'ibrutinib', 'acalabrutinib', 'zanubrutinib', 'pirtobrutinib'}

# h340: MEK inhibitors - cancer-only drugs (0% precision for non-cancer)
# These target the RAS/RAF/MEK/ERK pathway which is only relevant for cancer
# Unlike VEGF inhibitors, MEK inhibitors have no plausible non-cancer mechanism
MEK_INHIBITORS = {'trametinib', 'cobimetinib', 'binimetinib', 'selumetinib',
                  'mirdametinib', 'pimasertib', 'refametinib'}

# h346: Cancer-only drugs - 0% precision for non-cancer predictions (115 preds, 0 GT hits)
# These drugs have NO approved non-cancer uses and no plausible non-cancer mechanism
# Filter: cancer-only drug + non-cancer disease = FILTER tier (prevents false positives)
# Note: Excludes drugs with known non-cancer uses (mTOR inhibitors, imatinib, ranibizumab, etc.)
CANCER_ONLY_DRUGS = (
    # BRAF inhibitors (melanoma, lung cancer)
    {'vemurafenib', 'dabrafenib', 'encorafenib'}
    # Immunotherapy (PD-1/PD-L1/CTLA-4)
    | {'pembrolizumab', 'nivolumab', 'ipilimumab', 'atezolizumab',
       'durvalumab', 'avelumab', 'cemiplimab'}
    # BCL2 inhibitors (CLL, AML)
    | {'venetoclax', 'navitoclax'}
    # PARP inhibitors (BRCA cancers)
    | {'olaparib', 'rucaparib', 'niraparib', 'talazoparib'}
    # Proteasome inhibitors (myeloma)
    | {'bortezomib', 'carfilzomib', 'ixazomib'}
    # HDAC inhibitors (CTCL, lymphoma)
    | {'vorinostat', 'romidepsin', 'belinostat', 'panobinostat'}
    # CDK inhibitors (breast cancer)
    | {'palbociclib', 'ribociclib', 'abemaciclib'}
    # BCR-ABL inhibitors (CML) - NOT imatinib (GIST use)
    | {'nilotinib', 'dasatinib', 'ponatinib', 'bosutinib'}
    # ALK inhibitors (lung cancer)
    | {'crizotinib', 'ceritinib', 'alectinib', 'brigatinib', 'lorlatinib'}
    # EGFR inhibitors (lung cancer)
    | {'erlotinib', 'gefitinib', 'afatinib', 'osimertinib'}
    # HER2 inhibitors (breast cancer)
    | {'lapatinib', 'neratinib', 'tucatinib'}
    # VEGFR TKIs (multiple cancers) - NOT ranibizumab/aflibercept (ophthalmic)
    | {'sunitinib', 'sorafenib', 'pazopanib', 'axitinib', 'regorafenib',
       'lenvatinib', 'cabozantinib', 'vandetanib', 'nintedanib'}
    # Taxanes (multiple cancers)
    | {'paclitaxel', 'docetaxel', 'cabazitaxel'}
    # Platinum agents (multiple cancers)
    | {'cisplatin', 'carboplatin', 'oxaliplatin'}
    # Anthracyclines (multiple cancers)
    | {'doxorubicin', 'daunorubicin', 'epirubicin', 'idarubicin'}
    # MEK inhibitors (already in MEK_INHIBITORS set)
    | MEK_INHIBITORS
)

# h538: Cancer targeted therapy drugs (kinase inhibitors + immunotherapy)
# These are mutation-specific and DON'T transfer across cancer subtypes via kNN.
# Holdout precision: kinase 9.6%, immunotherapy 18.3% vs cytotoxic 53%.
# cancer_same_type predictions with these drugs should be demoted MEDIUM → LOW.
CANCER_TARGETED_THERAPY = (
    # Kinase inhibitors (mutation-specific, don't generalize across cancer subtypes)
    {'imatinib', 'dasatinib', 'nilotinib', 'sunitinib', 'sorafenib',
     'erlotinib', 'gefitinib', 'lapatinib', 'crizotinib', 'ruxolitinib',
     'ibrutinib', 'palbociclib', 'ribociclib', 'lenvatinib', 'regorafenib',
     'axitinib', 'pazopanib', 'vemurafenib', 'dabrafenib', 'trametinib',
     'cobimetinib', 'osimertinib', 'afatinib', 'cabozantinib', 'ponatinib',
     'bosutinib', 'vandetanib', 'fedratinib', 'gilteritinib', 'midostaurin',
     'entrectinib', 'larotrectinib', 'capmatinib', 'tepotinib', 'tucatinib',
     'neratinib', 'lorlatinib', 'alectinib', 'brigatinib', 'ceritinib',
     'encorafenib', 'binimetinib', 'futibatinib', 'infigratinib',
     'pemigatinib', 'erdafitinib', 'abemaciclib',
     'ivosidenib', 'everolimus'}
    # Immunotherapy (PD-1/PD-L1/CTLA-4 — tumor-specific immune activation)
    | {'nivolumab', 'pembrolizumab', 'atezolizumab', 'ipilimumab',
       'durvalumab', 'avelumab', 'tremelimumab', 'cemiplimab'}
    # h598: Anti-HER2/EGFR/VEGFR mAbs (target-specific, biomarker-dependent)
    | {'trastuzumab', 'pertuzumab', 'cetuximab', 'ramucirumab'}
    # h598: PARP inhibitors (BRCA/HRD-specific)
    | {'olaparib', 'niraparib', 'rucaparib'}
    # h598: BTK inhibitors (B-cell malignancy specific, same class as ibrutinib)
    | {'tirabrutinib', 'acalabrutinib', 'zanubrutinib'}
    # h598: Other mechanism-specific (6.1% holdout vs 40.2% existing cancer_same_type)
    | {'trabectedin', 'eribulin', 'lanreotide'}
)

# Ophthalmic drugs
OPHTHALMIC_ANTIBIOTICS = {'ciprofloxacin', 'moxifloxacin', 'ofloxacin', 'tobramycin', 'gentamicin',
                          'gatifloxacin', 'levofloxacin', 'besifloxacin', 'neomycin', 'polymyxin'}  # 62.5% rank<=15
OPHTHALMIC_STEROIDS = {'dexamethasone', 'prednisolone', 'fluorometholone', 'loteprednol',
                       'difluprednate', 'rimexolone', 'triamcinolone'}  # 48% rank<=15

# Dermatological drugs
TOPICAL_STEROIDS = {'hydrocortisone', 'betamethasone', 'triamcinolone', 'clobetasol', 'fluocinolone',
                    'fluocinonide', 'mometasone', 'desonide', 'halobetasol', 'desoximetasone'}  # 63.6% rank<=5

# h154: Cardiovascular beta-blockers achieve 33.3% precision at rank<=5
BETA_BLOCKERS = {'metoprolol', 'atenolol', 'carvedilol', 'bisoprolol', 'propranolol',
                 'labetalol', 'nebivolol', 'nadolol', 'timolol', 'esmolol', 'sotalol'}  # 33.3% rank<=5

# h217: Heart failure drug classes (from h212 analysis)
# Loop diuretics: 75% precision for heart failure (GOLDEN)
LOOP_DIURETICS = {'furosemide', 'bumetanide', 'torsemide', 'torasemide', 'ethacrynic acid'}

# Aldosterone antagonists: 50% precision for heart failure (GOLDEN)
ALDOSTERONE_ANTAGONISTS = {'spironolactone', 'eplerenone', 'finerenone'}

# ARBs: 27% precision for heart failure, 20% for hypertension (HIGH)
ARB_DRUGS = {'losartan', 'valsartan', 'irbesartan', 'candesartan', 'telmisartan',
             'olmesartan', 'azilsartan', 'eprosartan'}

# Heart failure disease keywords
HF_KEYWORDS = {'heart failure', 'cardiac failure', 'cardiomyopathy', 'ventricular dysfunction'}

# h230: Additional CV drug classes (from h229 validation: +21.8pp vs kNN)
# AFib: 65.6% drug-class coverage vs 23.9% kNN (+41.7pp)
AFIB_KEYWORDS = {'atrial fibrillation', 'atrial flutter', 'afib', 'a-fib'}
ANTICOAGULANT_DRUGS = {'warfarin', 'rivaroxaban', 'apixaban', 'dabigatran', 'edoxaban',
                       'heparin', 'enoxaparin', 'dalteparin', 'fondaparinux'}  # DOACs + warfarin + heparin
RATE_CONTROL_DRUGS = {'diltiazem', 'verapamil', 'digoxin'}  # Non-beta-blocker rate control

# MI/CAD: 47.9%/65.8% drug-class coverage vs 10.2%/31.7% kNN
MI_KEYWORDS = {'myocardial infarction', 'heart attack', 'stemi', 'nstemi', 'mi '}
CAD_KEYWORDS = {'coronary artery disease', 'coronary heart', 'ischemic heart', 'angina'}
ANTIPLATELET_DRUGS = {'aspirin', 'clopidogrel', 'prasugrel', 'ticagrelor', 'dipyridamole',
                      'ticlopidine', 'vorapaxar', 'cangrelor'}  # P2Y12 inhibitors + aspirin
NITRATE_DRUGS = {'nitroglycerin', 'isosorbide', 'nitrate'}  # Vasodilators

# h157: DMARDs achieve 75.4% precision for autoimmune diseases
DMARD_DRUGS = {'methotrexate', 'sulfasalazine', 'hydroxychloroquine', 'leflunomide',
               'azathioprine', 'mycophenolate', 'cyclosporine', 'tacrolimus'}  # 75.4% rank<=10

# h189: ATC L4 codes for rescue criteria (from h152 analysis)
# High-precision drug subclasses (use for GOLDEN rescue)
ATC_HIGH_PRECISION_AUTOIMMUNE = {'H02AB', 'L04AX'}  # Glucocorticoids (77%), Traditional immunosuppressants (82%)
# h397: ATC_HIGH_PRECISION_DERMATOLOGICAL removed (was dead code, never referenced)

# Low-precision drug subclasses (use for FILTER/demotion)
# h152 finding: Biologics have 8-17% precision vs 77-82% for traditional drugs
ATC_BIOLOGIC_CODES = {'L04AB', 'L04AC', 'L04AF'}  # TNF (17%), IL (8.7%), JAK (18.8%) inhibitors

# h265: Additional drug class tier modifiers (from h163 precision analysis)
# SGLT2 inhibitors: 71.4% precision for cardiovascular diseases (GOLDEN)
SGLT2_INHIBITORS = {'canagliflozin', 'dapagliflozin', 'empagliflozin', 'ertugliflozin', 'sotagliflozin'}

# Thiazolidinediones: 66.7% precision for metabolic diseases (GOLDEN)
THIAZOLIDINEDIONES = {'pioglitazone', 'rosiglitazone'}

# NSAIDs: 50% precision for autoimmune diseases (HIGH)
NSAID_DRUGS = {'ibuprofen', 'naproxen', 'diclofenac', 'indomethacin', 'celecoxib',
               'meloxicam', 'piroxicam', 'ketorolac', 'aspirin'}

# Fluoroquinolones: 44.4% precision for respiratory diseases (HIGH)
FLUOROQUINOLONE_DRUGS = {'ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'ofloxacin',
                          'gatifloxacin', 'norfloxacin', 'gemifloxacin'}

# h265: Low-precision drug class × category combinations (demote/warn)
# mAbs for cancer have 6.2% precision despite seeming appropriate (sparse GT)
# Kinase inhibitors for cancer have 2.8% precision (sparse GT)

# h326: Broad therapeutic class isolation detection
# When a drug from a "broad" class (treats many conditions) is predicted ALONE
# (no other drugs from same class predicted for the disease), precision is very low.
# h307 findings:
#   - Local anesthetics: 1.8% alone vs 15.0% with classmates (+13.2 pp)
#   - Corticosteroids: 0.0% alone vs 12.6% with classmates (+12.6 pp)
#   - TNF Inhibitors: 3.4% alone vs 27.3% with classmates (+23.8 pp)
#   - NSAIDs: 2.4% alone vs 7.1% with classmates (+4.7 pp)
# EXCEPTION: Statins - alone is BETTER (37.5% vs 29.3%)
LOCAL_ANESTHETICS = {'lidocaine', 'bupivacaine', 'ropivacaine', 'prilocaine',
                     'tetracaine', 'mepivacaine', 'articaine', 'levobupivacaine'}
TNF_INHIBITORS = {'adalimumab', 'infliximab', 'etanercept', 'golimumab',
                  'certolizumab', 'certolizumab pegol'}
# h328: IL inhibitors have 3% isolated vs 50% with classmates (+47 pp!)
IL_INHIBITORS = {'tocilizumab', 'secukinumab', 'ixekizumab', 'ustekinumab', 'guselkumab',
                 'risankizumab', 'tildrakizumab', 'anakinra', 'canakinumab', 'brodalumab'}

# h332: mTOR inhibitors - 7.4 pp gap, isolated 1.4% vs non-isolated 8.8%
# Used in both immunosuppression (transplant) and cancer - cohesion is positive signal
MTOR_INHIBITORS = {'sirolimus', 'temsirolimus', 'everolimus', 'rapamycin'}

# h332: Alkylating agents - 5.6 pp gap, isolated 0% vs non-isolated 5.6%
# Cyclophosphamide used in cancer + autoimmune (lupus, vasculitis) - cohesion is positive signal
ALKYLATING_AGENTS = {'cyclophosphamide', 'ifosfamide', 'melphalan', 'chlorambucil',
                     'busulfan', 'carmustine', 'lomustine', 'dacarbazine',
                     'temozolomide', 'bendamustine'}

# h183: Hormone drugs for reproductive diseases (26.3% precision vs 3.1% non-hormone)
REPRODUCTIVE_HORMONE_DRUGS = {'estradiol', 'estrogen', 'estropipate', 'conjugated estrogen',
                              'progesterone', 'progestin', 'medroxyprogesterone',
                              'testosterone', 'follitropin', 'corifollitropin', 'lutropin',
                              'gonadotropin', 'gonadorelin', 'clomiphene', 'letrozole'}

# Broad therapeutic classes where ISOLATION = bad signal (1.9% precision overall)
# These are classes that treat many conditions; if kNN only recommends ONE,
# it's likely noise rather than a real signal.
# h328: Added IL inhibitors (3% alone vs 50% with classmates)
BROAD_THERAPEUTIC_CLASSES: Dict[str, Set[str]] = {
    'local_anesthetics': LOCAL_ANESTHETICS,
    'corticosteroids': CORTICOSTEROID_DRUGS,
    'tnf_inhibitors': TNF_INHIBITORS,
    'nsaids': NSAID_DRUGS,
    'il_inhibitors': IL_INHIBITORS,  # h328: +47 pp cohesion effect
    'mtor_inhibitors': MTOR_INHIBITORS,  # h332: isolated 0% HIGH, 1.4% overall
    'alkylating_agents': ALKYLATING_AGENTS,  # h332: isolated 0% HIGH, 0% MEDIUM
}

# h353: Complication-Specific Drug Class Filter
# Validated finding: For complication diseases (nephropathy, retinopathy, etc.),
# only specific drug classes have non-zero precision.
# Non-validated classes have 0% precision across 214 predictions (0 GT hits lost).
#
# Key results:
#   - Nephrotic syndrome: Validated 69.2% vs Non-validated 0.0% (+69.2 pp)
#   - Retinopathy: Validated 33.3% vs Non-validated 0.0%
#   - Cardiomyopathy: Validated 12.5% vs Non-validated 0.0%
#   - Neuropathy: Validated 0.0% vs Non-validated 0.0% (both bad, but filter anyway)
#
# This filter removes 214 predictions with 100% accuracy (zero GT loss).
COMPLICATION_VALIDATED_DRUGS: Dict[str, Set[str]] = {
    # Nephrotic syndrome: Steroids + immunosuppressants + anesthetics (for procedures)
    'nephrotic': {
        'prednisolone', 'methylprednisolone', 'dexamethasone', 'hydrocortisone',
        'prednisone', 'betamethasone', 'cortisone', 'corticotropin', 'triamcinolone',
        'budesonide', 'lidocaine', 'bupivacaine', 'rituximab', 'tacrolimus',
        'cyclosporine', 'mycophenolate', 'cyclophosphamide', 'azathioprine'
    },
    # Retinopathy: Anti-VEGF + steroids only
    'retinopathy': {
        'ranibizumab', 'aflibercept', 'bevacizumab', 'brolucizumab', 'faricimab',
        'dexamethasone', 'triamcinolone', 'fluocinolone'
    },
    # Cardiomyopathy: Beta blockers + ACEi/ARBs + SGLT2i
    'cardiomyopathy': {
        'metoprolol', 'bisoprolol', 'carvedilol', 'enalapril', 'lisinopril',
        'ramipril', 'losartan', 'valsartan', 'candesartan', 'sacubitril',
        'spironolactone', 'eplerenone', 'dapagliflozin', 'empagliflozin'
    },
    # Neuropathy: SNRIs + anticonvulsants + TCAs + topicals
    'neuropathy': {
        'duloxetine', 'pregabalin', 'gabapentin', 'amitriptyline', 'nortriptyline',
        'carbamazepine', 'oxcarbazepine', 'lidocaine', 'capsaicin'
    },
}

# h354/h356: CV Pathway-Comprehensive Drugs (from h351 analysis)
# Criteria: drugs with GT for BOTH CV base AND CV complication, OR drugs with 2+ CV complication types
# h356 expanded criteria to include antiplatelets/anticoagulants that specialize in CV complications
#
# For CV complication predictions (with expanded criteria):
#   - Pathway-comprehensive: 28.9% precision (26/90)
#   - Non-pathway-comprehensive: 1.1% precision (1/90)
#   - GAP: +27.8 pp
#
# CV Base: hypertension, coronary artery disease, hyperlipidemia, dyslipidemia, atherosclerosis
# CV Complications: heart failure, stroke, myocardial infarction, angina, peripheral vascular
#
# 129 drugs identified as pathway-comprehensive for CV (109 original + 20 multi-complication drugs)
# Use for boost (pathway-comp → HIGH) and demotion (non-pathway → LOW for CV complications)
CV_PATHWAY_COMPREHENSIVE_DRUGS = {
    # Original 109 drugs (base + complication)
    'alirocumab', 'aliskiren', 'aliskiren mixture with hydrochlorothiazide',
    'amiloride', 'amiloride / hydrochlorothiazide oral tablet', 'amiodarone',
    'amlodipine', 'amlodipine / hydrochlorothiazide / olmesartan',
    'amlodipine besylate; olmesartan medoxomil', 'amlodipine mixture with valsartan',
    'amlodipine, atorvastatin drug combination', 'apixaban', 'aprocitentan',
    'atenolol', 'atenolol; chlorthalidone', 'atorvastatin',
    'azilsartan kamedoxomil; chlorthalidone', 'bempedoic acid', 'benazepril-amlodipine',
    'bendroflumethiazide / nadolol pill', 'bisoprolol', 'bms 747158-02', 'caduet',
    'candesartan', 'candesartan / hydrochlorothiazide pill', 'cangrelor', 'captopril',
    'carvedilol', 'chlorothiazide', 'chlorthalidone', 'cholestyramine', 'colestipol',
    'dabigatran etexilate', 'digoxin', 'dobutamine', 'doxazosin', 'dronedarone',
    'edoxaban', 'enalapril', 'enalaprilat', 'eplerenone', 'evolocumab',
    'ezetimibe / simvastatin oral tablet', 'felodipine', 'fosinoprilat', 'furosemide',
    'hydrochlorothiazide', 'hydrochlorothiazide / losartan pill',
    'hydrochlorothiazide / metoprolol pill',
    'hydrochlorothiazide 12.5 mg / lisinopril 10 mg oral tablet',
    'hydrochlorothiazide 12.5 mg / olmesartan medoxomil 40 mg oral tablet',
    'hydrochlorothiazide; irbesartan', 'hydrochlorothiazide; spironolactone',
    'hydrochlorothiazide; telmisartan', 'hydrochlorothiazide; valsartan',
    'hydroflumethiazide', 'indapamide', 'irbesartan', 'isosorbide', 'isosorbide dinitrate',
    'isosorbide mononitrate', 'labetalol', 'lactose, anhydrous', 'levamlodipine',
    'lidocaine', 'liraglutide', 'lisinopril', 'losartan',
    'losartan potassium and hydrochlorothiazide', 'lovastatin', 'metolazone', 'metoprolol',
    'nadolol', 'nebivolol', 'niacin', 'nicardipine', 'nifedipine', 'nitroglycerin',
    'nitroglycerin lactose', 'olmesartan', 'perindoprilat', 'phenobarbital', 'plavix',
    'polythiazide', 'potassium cation k-40', 'pravastatin', 'prazosin', 'propranolol',
    'quinaprilat', 'ramipril', 'ramiprilat', 'rivaroxaban', 'rosuvastatin',
    'sacubitril and valsartan sodium hydrate drug combination', 'semaglutide', 'simvastatin',
    'spironolactone', 'technetium tc-99m sestamibi', 'telmisartan', 'telmisartan/amlodipine',
    'terazosin', 'ticagrelor', 'timolol', 'torasemide', 'trandolaprilat', 'valsartan',
    'valsartan, amlodipine, hct', 'verapamil', 'warfarin',
    # h356: 20 additional drugs with 2+ CV complication types (antiplatelets, thrombolytics, etc.)
    'alteplase', 'aspirin', 'aspirin; omeprazole', 'celecoxib', 'clopidogrel',
    'clopidogrel aspirin', 'colchicine', 'dalteparin', 'dopamine', 'dulaglutide',
    'enoxaparin', 'eptifibatide', 'finerenone', 'fondaparinux', 'icosapent',
    'papaverine', 'prasugrel', 'stannous cation', 'tenecteplase', 'vorapaxar'
}

# CV complication keywords (heart failure, stroke, MI, angina, peripheral vascular)
CV_COMPLICATION_KEYWORDS = {'heart failure', 'stroke', 'myocardial infarction', 'angina',
                            'peripheral vascular', 'cardiomyopathy', 'cardiac failure'}

# h384: Drugs to EXCLUDE from cv_pathway_comprehensive HIGH tier
# These have 0% precision despite being in CV_PATHWAY_COMPREHENSIVE_DRUGS
# Antiplatelets: 0% precision (prevent events, don't treat conditions)
# Others: various reasons for 0% precision in evaluation
# h397: CV_PATHWAY_EXCLUDE removed (was dead code from h384, never referenced in tier logic)

# h669/h670: False GT entries in indicationList.xlsx (Every Cure data quality issue)
# These drugs are NOT treatments for the listed diseases. The FDA label mentions
# the disease as a differential diagnosis/exclusion criterion, NOT an indication.
# h686: Drug name aliases for Every Cure → DrugBank name mismatches.
# EC uses INN names, brand names, salt forms, and spelling variants that differ
# from DrugBank's canonical names. These aliases expand internal GT by ~12%.
# Format: {ec_name_lower: drugbank_name_lower}
_DRUG_NAME_ALIASES = {
    # INN vs brand name
    'acyclovir': 'aciclovir',
    'aspirin': 'acetylsalicylic acid',
    'co-trimoxazole': 'sulfamethoxazole',
    'augmentin': 'amoxicillin',
    'plavix': 'clopidogrel',
    'rifampin': 'rifampicin',
    'adcetris': 'brentuximab vedotin',
    'dysport': 'botulinum toxin type a',
    # Spelling variants
    'etacrynic acid': 'ethacrynic acid',
    'alendronic acid': 'alendronate',
    'benztropine': 'benzatropine',
    '(+)-hyoscyamine': 'hyoscyamine',
    'levodopa': 'l-dopa',
    # Salt form / format differences
    'medroxyprogesterone': 'medroxyprogesterone acetate',
    'megestrol': 'megestrol acetate',
    'certolizumab': 'certolizumab pegol',
    'asparaginase': 'asparaginase escherichia coli',
    'tacrolimus anhydrous 19-epimer': 'tacrolimus',
    '2-phenylbutyric acid': 'phenylbutyric acid',
    'leuprorelin acetate': 'leuprolide',
    '4-hydroxycyclophosphamide': 'cyclophosphamide',
    'sodium nitroprusside': 'nitroprusside',
    'lithium cation': 'lithium',
    'potassium(1+)': 'potassium',
    'potassium cation k-40': 'potassium cation',
    'arginine': 'l-arginine',
    'pamidronic acid': 'pamidronate',
    'dexamfetamine': 'dextroamphetamine',
    'gemtuzumab': 'gemtuzumab ozogamicin',
    'pitavastatin(1-)': 'pitavastatin',
    'insulin-glargine': 'insulin glargine',
    'chlortheophylline': '8-chlorotheophylline',
    # Diagnostic agents (format differences)
    'ioflupane i-123': 'ioflupane i 123',
    'rose bengal at': 'rose bengal',
    'technetium tc 99m sulfur colloid': 'technetium tc-99m sulfur colloid',
    'florbetaben f18': 'florbetaben (18f)',
}

# Example: "secondary causes such as hypothyroidism should be excluded before starting
# lipid therapy" → NLP incorrectly extracts hypothyroidism as an indication.
# Format: {drug_name_lower: {disease_name_lower, ...}}
FALSE_GT_PAIRS = {
    # h669: Lipid drugs mention hypothyroidism as a secondary cause to rule out, not treat
    # h670: Extended to ALL secondary cause diseases from standard lipid drug FDA label boilerplate:
    #   "secondary causes for hypercholesterolemia (e.g., poorly controlled diabetes mellitus,
    #    hypothyroidism, nephrotic syndrome, dysproteinemias, obstructive liver disease,
    #    other drug therapy, alcoholism) should be excluded"
    # These diseases CAUSE hyperlipidemia; lipid drugs do NOT treat them.
    'fenofibrate': {'hypothyroidism', 'diabetes mellitus'},
    'gemfibrozil': {'hypothyroidism', 'diabetes mellitus', 'nephrotic syndrome'},
    'lovastatin': {'hypothyroidism', 'diabetes mellitus', 'nephrotic syndrome'},
    'cholestyramine': {
        'hypothyroidism',  # Also an inverse: reduces T4 absorption
        'diabetes mellitus', 'nephrotic syndrome',
        # NOTE: cholestyramine → obstructive liver disease is TRUE (treats biliary pruritus)
    },
    'lomitapide': {'hypothyroidism', 'nephrotic syndrome'},
    'omega-3 fatty acids': {'hypothyroidism', 'diabetes mellitus'},
    'simvastatin': {'hypothyroidism', 'diabetes mellitus', 'nephrotic syndrome'},
    'pravastatin': {'diabetes mellitus', 'nephrotic syndrome'},
    'pitavastatin': {'diabetes mellitus'},
    'colestipol': {'diabetes mellitus'},
    'niacin': {'diabetes mellitus'},  # Niacin worsens glycemic control
    # h670: Nafarelin label says "other causes of sexual precocity such as congenital
    # adrenal hyperplasia, testotoxicosis must be excluded" — CAH is a differential,
    # not an indication for GnRH agonist therapy
    'nafarelin': {'congenital adrenal hyperplasia'},
    # h677/h681: B12 supplements list diseases that CAUSE B12 deficiency, not diseases treated BY B12.
    # Label text: "conditions associated with B12 deficiency: hypothyroidism, multiple sclerosis,
    # iron deficiency" — these are differential diagnoses, not indications.
    # Also from multi-vitamin combo products (iron+B12+folate+fluoride) where iron/folate/fluoride
    # indications are attributed to the B12 component.
    'cyanocobalamin': {
        'multiple sclerosis', 'iron deficiency', 'thyrotoxicosis',
        'hemolytic anemia', 'hepatic disease', 'renal disease',  # h681: B12 deficiency CAUSES, not indications
        'hypochromic anemia', 'iron deficiency anemia',  # h681: from iron+B12 combo products
        'vitamin c deficiency', 'dental caries', 'vitamin deficiency',  # h681: from multi-vitamin combos
    },
    'hydroxocobalamin': {
        'folate deficiency', 'multiple sclerosis', 'iron deficiency',
    },
}

# h480: Inverse-indication FILTER
# Drugs that treat condition A should NOT be predicted for the OPPOSITE of A
# These are HARMFUL predictions (drug causes/worsens the predicted disease)
# Found via literature validation of HIGH novel predictions
INVERSE_INDICATION_PAIRS = {
    # Thyroid: anti-thyroid drugs → hypothyroidism (they CAUSE hypothyroidism)
    'propylthiouracil': {'hypothyroidism', 'congenital hypothyroidism'},
    'methimazole': {'hypothyroidism', 'congenital hypothyroidism', 'primary hyperparathyroidism', 'hypoparathyroidism'},
    # Thyroid: thyroid hormones → hyperthyroidism (they CAUSE/worsen hyperthyroidism)
    'levothyroxine': {'hyperthyroidism'},
    'liothyronine': {'hyperthyroidism'},
    # Diabetes: hyperglycemia-causing drugs → hyperglycemia/DKA
    'diazoxide': {'hyperglycemia', 'diabetic ketoacidosis', 'central diabetes insipidus'},
    # Diabetes: glucagon → hyperglycemia (raises blood sugar)
    'glucagon': {'hyperglycemia'},
    # Diabetes: vasopressin → hyperglycemia (not a glucose-lowering drug)
    'vasopressin': {'hyperglycemia'},
    # h482: Sulfonylureas/insulin → hypoglycemia (they CAUSE hypoglycemia)
    # These drugs lower blood sugar; predicting them for hypoglycemia is inverse
    # h675: Also → T1D/DKA: Sulfonylureas and meglitinides require functioning beta cells
    #   to stimulate insulin release. T1D has autoimmune beta cell destruction → zero efficacy.
    #   FDA labels explicitly state "should not be used for type 1 diabetes or DKA."
    #   NLP extraction error: EC indicationList extracts T1D/DKA from "limitations of use" text.
    'glipizide': {'hypoglycemia', 'hyperinsulinemic hypoglycemia',
                  'type 1 diabetes mellitus', 'diabetic ketoacidosis'},
    'glimepiride': {'hypoglycemia', 'hyperinsulinemic hypoglycemia',
                    'type 1 diabetes mellitus', 'diabetic ketoacidosis'},
    'glyburide': {'hypoglycemia', 'hyperinsulinemic hypoglycemia',
                  'type 1 diabetes mellitus', 'diabetic ketoacidosis'},
    'tolazamide': {'hypoglycemia',
                   'type 1 diabetes mellitus', 'diabetic ketoacidosis'},
    'nateglinide': {'hypoglycemia', 'hyperinsulinemic hypoglycemia',
                    'type 1 diabetes mellitus', 'diabetic ketoacidosis'},
    'repaglinide': {'hypoglycemia', 'hyperinsulinemic hypoglycemia',
                    'type 1 diabetes mellitus', 'diabetic ketoacidosis'},
    'insulin lispro': {'hypoglycemia'},
    'insulin human': {'hypoglycemia', 'hyperinsulinemic hypoglycemia'},
    # h482: Insulin sensitizers → hypoglycemia (can cause hypoglycemia)
    # h675: TZDs → T1D/DKA: Require endogenous insulin production to work.
    #   T1D has no beta cells → TZDs are ineffective. FDA labels explicit.
    'rosiglitazone': {'hypoglycemia', 'hyperinsulinemic hypoglycemia',
                      'type 1 diabetes mellitus', 'diabetic ketoacidosis'},
    'pioglitazone': {'hypoglycemia', 'hyperinsulinemic hypoglycemia',
                     'type 1 diabetes mellitus', 'diabetic ketoacidosis'},
    # h483: Metronidazole → myopia (CAUSES transient myopia as adverse effect, JAMA case report)
    'metronidazole': {'myopia'},
    # h483: Verapamil → cardiac arrest (CAN PRECIPITATE cardiac arrest in WPW/VT; not a treatment)
    # IV verapamil is CONTRAINDICATED in ventricular tachycardia per ACLS guidelines
    # h484: Non-DHP CCBs contraindicated in HF (ACC/AHA 2022 Class III - harm)
    # Negative inotropes that can cause acute decompensation in HFrEF
    'verapamil': {
        'cardiac arrest',
        'chronic heart failure', 'heart failure', 'congestive heart failure',
        'systolic heart failure', 'acute heart failure',
    },
    # h484: Diltiazem — same non-DHP CCB class contraindications as verapamil
    # Cardiac arrest: can precipitate arrest in WPW/structural heart disease
    # Heart failure: negative inotrope, ACC/AHA 2022 Class III (harm)
    # Ventricular tachycardia: NOT indicated (unlike verapamil for fascicular VT)
    #   Can cause hemodynamic collapse in VT with structural heart disease
    'diltiazem': {
        'cardiac arrest',
        'chronic heart failure', 'heart failure', 'congestive heart failure',
        'systolic heart failure', 'acute heart failure',
        'ventricular tachycardia',
        'myocardial infarction',  # h486: SIDER AE, CCB in acute MI is harmful
    },
    # h484: Short-acting nifedipine → ACS (HINT study: excess mortality)
    # Extended-release may be acceptable but we cannot distinguish formulations
    # Furberg 1995 meta-analysis: dose-dependent mortality increase with short-acting nifedipine
    'nifedipine': {'acute coronary syndrome'},
    # h495: Class Ic antiarrhythmics + structural heart disease (CAST trial: 2.5x mortality)
    # Flecainide and propafenone are FDA-approved for SVT/atrial fibrillation
    # but CONTRAINDICATED with structural heart (post-MI, HF, VT, VF)
    'flecainide': {
        'myocardial infarction', 'ventricular tachycardia', 'ventricular fibrillation',
        'heart failure', 'chronic heart failure', 'cardiomyopathy',
        'dilated cardiomyopathy', 'cardiac arrest',
    },
    'propafenone': {
        'myocardial infarction', 'ventricular tachycardia', 'ventricular fibrillation',
        'heart failure', 'chronic heart failure', 'cardiomyopathy',
        'dilated cardiomyopathy', 'cardiac arrest',
    },
    # h495: SGLT2 inhibitors → hypoglycemia (they CAUSE hypoglycemia as adverse effect)
    # SGLT2i are glucose-lowering drugs; predicting them for hypoglycemia is inverse
    'empagliflozin': {'hypoglycemia', 'hyperinsulinemic hypoglycemia'},
    'dapagliflozin': {'hypoglycemia', 'hyperinsulinemic hypoglycemia'},
    'canagliflozin': {'hypoglycemia', 'hyperinsulinemic hypoglycemia'},
    # h479: Minocycline CAUSES urticaria as adverse effect (case reports: acute severe urticaria
    # developing 3-28 days after initiation, can recur >12 months after discontinuation)
    'minocycline': {'urticaria'},
    # h493: Corticosteroids contraindicated in stable IPF (ATS/ERS/JRS/ALAT 2022)
    # PANTHER-IPF trial (NEJM 2012): prednisone+azathioprine+NAC INCREASED mortality
    # Standard of care is pirfenidone/nintedanib, NOT corticosteroids
    # h486: Added TB reactivation, steroid-induced MG crisis, glaucoma, osteoporosis, pancreatitis
    # h542/h673: Merged CS inverse indications (h542 HPA suppression + h486/h493 original + h673 safety audit)
    # BUG FIX: Previously had duplicate dict keys for prednisolone/prednisone/methylprednisolone,
    # causing h542 entries to silently overwrite h486/h493 entries (IPF, glaucoma, osteoporosis LOST)
    # h673: Added TEN (no proven efficacy, infection risk in skin-barrier-compromised patients),
    #   autoimmune PAP (74% deteriorate on CS, suppresses already-impaired alveolar macrophages),
    #   OSA (CS increase OSA risk HR 1.40, weight gain + airway fat deposition)
    'prednisolone': {
        'idiopathic pulmonary fibrosis',
        'neovascular glaucoma', 'osteoporosis', 'pancreatitis',
        'secondary adrenocortical insufficiency',  # h542: long-acting CS causes HPA suppression
        'toxic epidermal necrolysis',  # h673: no proven efficacy, infection risk, 40% mortality on CS
        'obstructive sleep apnea',  # h673: CS increase OSA risk (HR 1.40), weight gain worsens OSA
        'pulmonary alveolar proteinosis',  # h673: 74% deteriorate on CS, macrophage suppression
    },
    'prednisone': {
        'idiopathic pulmonary fibrosis', 'extrapulmonary tuberculosis',
        'neovascular glaucoma', 'osteoporosis', 'pancreatitis',
        'secondary adrenocortical insufficiency',  # h542: long-acting CS causes HPA suppression
        'toxic epidermal necrolysis',  # h673
        'obstructive sleep apnea',  # h673
        'pulmonary alveolar proteinosis',  # h673
    },
    'methylprednisolone': {
        'idiopathic pulmonary fibrosis',
        'neovascular glaucoma', 'osteoporosis',
        'secondary adrenocortical insufficiency',  # h542: long-acting CS causes HPA suppression
        'toxic epidermal necrolysis',  # h673
        'obstructive sleep apnea',  # h673
        'pulmonary alveolar proteinosis',  # h673
    },
    'dexamethasone': {
        'idiopathic pulmonary fibrosis', 'extrapulmonary tuberculosis',
        'systemic myasthenia gravis',
        'neovascular glaucoma', 'osteoporosis',
        'secondary adrenocortical insufficiency',  # h542: long-acting CS causes HPA suppression
        'toxic epidermal necrolysis',  # h673
        'obstructive sleep apnea',  # h673
        'pulmonary alveolar proteinosis',  # h673
    },
    'hydrocortisone': {
        'idiopathic pulmonary fibrosis',
        'toxic epidermal necrolysis',  # h673
        'pulmonary alveolar proteinosis',  # h673
    },
    'mometasone': {'idiopathic pulmonary fibrosis'},
    'fluticasone': {'idiopathic pulmonary fibrosis'},
    'cortisone': {
        'idiopathic pulmonary fibrosis', 'neovascular glaucoma',
        'toxic epidermal necrolysis',  # h673
        'pulmonary alveolar proteinosis',  # h673
    },
    # Note: hydrocortisone, cortisone, fludrocortisone, corticotropin are legitimate
    # replacement therapy for adrenocortical insufficiency — NOT inverse indications
    'betamethasone': {
        'secondary adrenocortical insufficiency',  # h542: long-acting CS causes HPA suppression
        'toxic epidermal necrolysis',  # h673
        'obstructive sleep apnea',  # h673
        'pulmonary alveolar proteinosis',  # h673
    },
    'triamcinolone': {
        'extrapulmonary tuberculosis',
        'neovascular glaucoma', 'osteoporosis',
        'secondary adrenocortical insufficiency',  # h542: long-acting CS causes HPA suppression
        'toxic epidermal necrolysis',  # h673: controversial, no RCT evidence, infection risk
        'obstructive sleep apnea',  # h673
        'pulmonary alveolar proteinosis',  # h673
    },
    'budesonide': {
        'neovascular glaucoma',
        'secondary adrenocortical insufficiency',
        'toxic epidermal necrolysis',  # h673
        'pulmonary alveolar proteinosis',  # h673: autoimmune PAP, 74% deteriorate on CS
    },
    # h486: Azathioprine causes TEN, erythema multiforme, hepatitis B reactivation,
    # interstitial pneumonia, cholestasis (well-documented immunosuppressant AEs)
    'azathioprine': {
        'toxic epidermal necrolysis', 'severe erythema multiforme',
        'hepatitis b', 'hereditary chronic cholestasis',
        # Note: interstitial pneumonia omitted — azathioprine treats underlying myositis
    },
    # h486: NSAIDs cause TEN (Stevens-Johnson spectrum), drug-induced SLE, peptic ulcer,
    # lichen planus, cerebrovascular events (COX-2 class effect)
    'naproxen': {
        'systemic lupus erythematosus', 'toxic epidermal necrolysis', 'lichen planus',
    },
    'celecoxib': {
        'toxic epidermal necrolysis', 'ischemic cerebrovascular disorder',
        'ischemic stroke',  # h528: COX-2 inhibitors increase stroke risk
    },
    'diclofenac': {
        'toxic epidermal necrolysis', 'chronic peptic ulcer disease',
    },
    'indomethacin': {'toxic epidermal necrolysis'},
    # h486: Estradiol causes endometrial/uterine cancer (classic), exacerbates hereditary angioedema
    'estradiol': {'endometrial cancer', 'uterine cancer', 'hereditary angioedema'},
    # h486: Paroxetine triggers mania in bipolar (all SSRIs can, paroxetine well-documented)
    'paroxetine': {'bipolar disorder'},
    # h526: SSRI/SNRI class effect — all trigger mania in bipolar patients
    # APA guidelines: antidepressant monotherapy contraindicated in bipolar
    # Risk: tricyclics > SNRIs > SSRIs, but ALL carry clinically significant risk
    'fluoxetine': {'bipolar disorder'},
    'sertraline': {'bipolar disorder'},
    'escitalopram': {'bipolar disorder'},
    # h526: SNRIs — higher mania induction risk than SSRIs
    'venlafaxine': {'bipolar disorder'},
    'duloxetine': {'bipolar disorder'},
    # h486: Paricalcitol (vitamin D analog) suppresses PTH → hypoparathyroidism
    'paricalcitol': {'hypoparathyroidism'},
    # h486: Erythromycin causes erythema multiforme
    'erythromycin': {'severe erythema multiforme'},
    # h486: Proarrhythmic drugs → ventricular tachycardia (torsades de pointes risk)
    # Note: lidocaine omitted — it's a Class Ib antiarrhythmic that TREATS VT
    'ibutilide': {'ventricular tachycardia'},
    'dofetilide': {'ventricular tachycardia'},
    'milrinone': {'ventricular tachycardia'},
    # h486: Methotrexate is gonadotoxic → female infertility
    'methotrexate': {'female infertility'},
    # h486: Imatinib causes interstitial lung disease
    'imatinib': {'systemic sclerosis associated interstitial lung disease'},
    # h486: Carbamazepine causes drug-induced dyskinesia
    'carbamazepine': {'dyskinesia'},
    # h486: Sulfadiazine nephrotoxicity → nephrotic syndrome
    'sulfadiazine': {'nephrotic syndrome'},
    # h486: GnRH agonists cause ovarian hyperstimulation syndrome
    'nafarelin': {'ovarian hyperstimulation syndrome'},
    # h486: Everolimus causes acute pancreatitis
    'everolimus': {'pancreatitis'},
    # h526: Conjugated estrogens — same carcinogenic mechanism as estradiol
    # WHI trial: HR 1.24 for breast cancer, 2-10x endometrial cancer risk
    # IARC Group 1 carcinogen (estrogen-progestogen combinations)
    'conjugated estrogens': {'breast cancer', 'endometrial cancer'},
    # h526: ACE inhibitors → angioedema (bradykinin accumulation class effect)
    # ACEi block bradykinin degradation → can cause life-threatening angioedema
    # Especially dangerous in hereditary angioedema (already bradykinin-mediated)
    'benazepril': {'angioedema', 'hereditary angioedema'},
    'quinapril': {'angioedema'},
    # h408+h544: Anti-TNF biologics INDUCE paradoxical autoimmune conditions
    # CLASS EFFECTS (all anti-TNF agents):
    #   SLE: 12,080 FAERS reports, >90% serious, median onset 7+ months
    #   MS/demyelination: Paradoxical demyelination, FDA warning
    #   Autoimmune hepatitis: 389 cases in VigiBase, infliximab > adalimumab > etanercept
    #   Sarcoidosis: 90+ cases, predominantly etanercept, 71/90 resolve on discontinuation
    #   Vasculitis: 113 cases (leukocytoclastic), mean 36 months on treatment
    # DRUG-SPECIFIC:
    #   Adalimumab → MG: Case reports of adalimumab-induced myasthenia gravis
    #   Adalimumab → polymyositis: 20 cases, 91% ANA+, safety precaution
    #   Adalimumab → lichen planus: 21 cases lichenoid reactions + 11 oral LP
    'adalimumab': {
        'systemic lupus erythematosus',
        'systemic myasthenia gravis',
        'multiple sclerosis',
        'autoimmune hepatitis',  # h544: 389 VigiBase cases
        'symptomatic sarcoidosis',  # h544: 90+ cases paradoxical sarcoidosis
        'vasculitis',  # h544: 113 cases leukocytoclastic vasculitis
        'polymyositis',  # h544: 20 cases, safety precaution
        'lichen planus',  # h544: 21 cases lichenoid reactions
    },
    'etanercept': {
        'systemic lupus erythematosus',
        'multiple sclerosis',
        'autoimmune hepatitis',  # h544: VigiBase class effect
        'symptomatic sarcoidosis',  # h544: most common anti-TNF for sarcoidosis
        'vasculitis',  # h544: 59/113 cases were etanercept
    },
    'infliximab': {
        'systemic lupus erythematosus',
        'multiple sclerosis',
        'autoimmune hepatitis',  # h544: 50.1% of 389 cases were infliximab
        'symptomatic sarcoidosis',  # h544: class effect
        'vasculitis',  # h544: 47/113 cases were infliximab
    },
    'golimumab': {
        'systemic lupus erythematosus',  # h544: class effect
        'multiple sclerosis',  # h544: class effect
        'autoimmune hepatitis',  # h544: class effect
        'symptomatic sarcoidosis',  # h544: class effect
    },
    # h537: Statins CAUSE diabetes (2024 Lancet IPD meta-analysis: 10-36% increase in new-onset
    # diabetes, dose-dependent). Predicting statins as diabetes TREATMENT is inverse indication.
    # Note: diabetic complications (nephropathy, neuropathy) excluded — statins may help those.
    # Note: diabetes insipidus excluded — unrelated (ADH pathway, not glucose).
    'lovastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
    'simvastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
    'atorvastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
    'rosuvastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
    'pravastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
    'fluvastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
    'pitavastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
    # h674: Defensive coverage for withdrawn/research statins (no current predictions but prevents future leakage)
    'cerivastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
    'mevastatin': {'type 2 diabetes mellitus', 'diabetes mellitus', 'hyperglycemia'},
}

# h481: Drug class → disease category standard-of-care mappings
# Used to classify predictions as LIKELY_GT_GAP (drug class is standard for disease category)
# Validated: SOC predictions have +28.4pp higher full-data precision vs NOVEL in HIGH tier
DRUG_CLASS_SOC_MAPPINGS: Dict[str, Dict[str, Any]] = {
    'corticosteroids': {
        'drugs': CORTICOSTEROID_DRUGS,
        'categories': {'autoimmune', 'hematological', 'dermatological', 'respiratory',
                       'ophthalmic', 'immunological'},
    },
    'statins': {
        'drugs': STATIN_DRUGS,
        'categories': {'cardiovascular', 'metabolic'},
    },
    'nsaids': {
        'drugs': NSAID_DRUGS,
        'categories': {'autoimmune', 'musculoskeletal'},
    },
    'dmards': {
        'drugs': DMARD_DRUGS,
        'categories': {'autoimmune'},
    },
    'beta_blockers': {
        'drugs': BETA_BLOCKERS,
        'categories': {'cardiovascular'},
    },
    'loop_diuretics': {
        'drugs': LOOP_DIURETICS,
        'categories': {'cardiovascular', 'renal'},
    },
    'arbs': {
        'drugs': ARB_DRUGS,
        'categories': {'cardiovascular', 'renal'},
    },
    'anticoagulants': {
        'drugs': ANTICOAGULANT_DRUGS,
        'categories': {'cardiovascular', 'hematological'},
    },
    'antiplatelets': {
        'drugs': ANTIPLATELET_DRUGS,
        'categories': {'cardiovascular'},
    },
    'sglt2_inhibitors': {
        'drugs': SGLT2_INHIBITORS,
        'categories': {'metabolic', 'cardiovascular', 'renal'},
    },
    'thiazolidinediones': {
        'drugs': THIAZOLIDINEDIONES,
        'categories': {'metabolic'},
    },
    'fluoroquinolones': {
        'drugs': FLUOROQUINOLONE_DRUGS,
        'categories': {'infectious', 'respiratory'},
    },
    'topical_steroids': {
        'drugs': TOPICAL_STEROIDS,
        'categories': {'dermatological'},
    },
    'cancer_drugs': {
        'drugs': CANCER_ONLY_DRUGS | TAXANE_DRUGS | ALKYLATING_DRUGS,
        'categories': {'cancer'},
    },
    'ophthalmic_antibiotics': {
        'drugs': OPHTHALMIC_ANTIBIOTICS,
        'categories': {'ophthalmic'},
    },
    'ophthalmic_steroids': {
        'drugs': OPHTHALMIC_STEROIDS,
        'categories': {'ophthalmic'},
    },
    'aldosterone_antagonists': {
        'drugs': ALDOSTERONE_ANTAGONISTS,
        'categories': {'cardiovascular', 'renal'},
    },
}

# Build reverse lookup for SOC classification
_DRUG_TO_SOC: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
for _cls_name, _cls_info in DRUG_CLASS_SOC_MAPPINGS.items():
    for _drug in _cls_info['drugs']:
        _DRUG_TO_SOC[_drug.lower()].append({
            'class': _cls_name,
            'categories': _cls_info['categories']
        })


def classify_literature_status(
    drug_name: str,
    disease_name: str,
    category: str,
    is_known_indication: bool,
) -> Tuple[str, Optional[str]]:
    """Classify a prediction's literature status.

    Returns:
        (status, soc_drug_class) where status is one of:
        - KNOWN_INDICATION: Already in ground truth
        - LIKELY_GT_GAP: Drug class is standard-of-care for disease category
        - NOVEL: No automated classification
    """
    if is_known_indication:
        return 'KNOWN_INDICATION', None

    drug_lower = drug_name.lower()

    # Check SOC match
    soc_matches = _DRUG_TO_SOC.get(drug_lower, [])
    for soc in soc_matches:
        if category in soc['categories']:
            return 'LIKELY_GT_GAP', soc['class']

    return 'NOVEL', None


# h280/h281: Complication vs Subtype relationship mapping
# Complications are CAUSED BY the base disease (different treatment expected)
# Subtypes are IS_A relationships (same treatment expected)
# h280 finding: Base→Complication predictions have 0% precision (FILTER these)
# h281 finding: Complication→Base predictions have 36.2% precision (acceptable)
#
# Format: {base_disease_term: [complication_terms]}
# If drug treats base and prediction is for complication → FILTER (0% precision)
BASE_TO_COMPLICATIONS = {
    'diabetes': [
        'diabetic nephropathy', 'diabetic neuropathy', 'diabetic retinopathy',
        'diabetic macular edema', 'diabetic foot', 'diabetic ketoacidosis',
        'diabetic peripheral neuropathy', 'diabetic peripheral angiopathy',
        'diabetic kidney disease', 'proliferative diabetic retinopathy'
    ],
    'diabetes mellitus': [
        'diabetic nephropathy', 'diabetic neuropathy', 'diabetic retinopathy',
        'diabetic macular edema', 'diabetic ketoacidosis', 'diabetic kidney disease'
    ],
    'type 2 diabetes mellitus': [
        'diabetic nephropathy', 'diabetic neuropathy', 'diabetic retinopathy',
        'diabetic macular edema', 'diabetic ketoacidosis', 'diabetic kidney disease'
    ],
    'type 1 diabetes mellitus': [
        'diabetic nephropathy', 'diabetic neuropathy', 'diabetic retinopathy',
        'diabetic ketoacidosis'
    ],
    'hypertension': [
        'hypertensive heart disease', 'hypertensive nephropathy', 'hypertensive retinopathy',
        'hypertensive crisis', 'hypertensive emergency'
    ],
    'atherosclerosis': [
        # h292/h296: CV events as complications of atherosclerosis
        # Statins achieve 100% precision, non-statins 0% (handled by statin rule)
        'myocardial infarction', 'ischemic stroke', 'stroke',
        'transient ischemic attack', 'acute coronary syndrome',
        'unstable angina', 'angina pectoris',
    ],
    'alcoholism': [
        'alcoholic liver disease', 'alcoholic hepatitis', 'alcoholic cirrhosis',
        'wernicke encephalopathy', 'korsakoff syndrome'
    ],
    'obesity': [
        'obstructive sleep apnea', 'nonalcoholic fatty liver', 'metabolic syndrome',
        'obesity hypoventilation syndrome'
    ],
    'heart failure': [
        'pulmonary edema', 'cardiorenal syndrome', 'cardiac cachexia'
    ],
    'chronic kidney disease': [
        'anemia of chronic kidney disease', 'uremic pruritus', 'secondary hyperparathyroidism',
        'renal osteodystrophy', 'uremia'
    ],
}

# h284/h291: Complication Transferability for confidence boosting
# Transferability = % of drugs treating complication that also treat base
# HIGH transferability complications: comp→base predictions work well (62.5% precision)
# LOW transferability complications: drugs are complication-specific
#
# Format: {complication_term: transferability_percentage}
COMPLICATION_TRANSFERABILITY = {
    # HIGH transferability (≥50%) - comp→base predictions reliable
    'acute heart failure': 100.0,
    'chronic heart failure': 94.8,
    'diabetic ketoacidosis': 86.7,
    'diabetic nephropathy': 80.0,
    'pulmonary edema': 75.0,
    'diabetic kidney disease': 68.8,
    'anemia of chronic kidney disease': 62.1,
    'stroke': 72.2,
    'transient ischemic attack': 50.0,
    'deep vein thrombosis': 100.0,
    # MEDIUM transferability (20-50%)
    'pulmonary embolism': 46.2,
    'ischemic stroke': 41.7,
    'angina pectoris': 40.0,
    'alcoholic cirrhosis': 33.3,
    'cardiac arrest': 33.3,
    'secondary hyperparathyroidism': 25.0,
    # LOW transferability (<20%) - drugs are complication-specific
    'myocardial infarction': 18.5,  # h292: statins only, see h296
    'diabetic neuropathy': 0.0,
    'diabetic retinopathy': 0.0,
    'diabetic macular edema': 5.6,
    'diabetic peripheral neuropathy': 0.0,
    'unstable angina': 12.5,
    'stable angina': 12.5,
    'obstructive sleep apnea': 16.7,
    'uremic pruritus': 10.0,
}

# h296: Statins achieve 100% precision for CV event→atherosclerosis
# Non-statins achieve 0% precision for same predictions
STATIN_NAMES = [
    'atorvastatin', 'rosuvastatin', 'simvastatin', 'pravastatin',
    'lovastatin', 'fluvastatin', 'pitavastatin', 'cerivastatin',
]

# CV events where statin rule applies (from h292)
CV_EVENTS_FOR_STATIN_RULE = [
    'myocardial infarction', 'stroke', 'ischemic stroke', 'transient ischemic attack',
    'acute coronary syndrome', 'unstable angina',
]

# h297: Mechanism-Specific Diseases (kNN will fail - set LOW confidence)
# These diseases have breadth < 3 and mono_pct > 70%
# Only 6.6% of their GT drugs are repurposable (vs 87.9% for other diseases)
# Drugs for these diseases are mechanism-specific and don't transfer to neighbors
MECHANISM_SPECIFIC_DISEASES = {
    'allergy', 'bleeding episodes', 'covid-19', 'chagas disease', 'cough',
    'fabry disease', 'familial immunoglobulin a nephropathy', 'global progressive disease',
    'hiv-1 infection', 'human immunodeficiency virus positive', 'hunter syndrome',
    'huntington disease', 'hypozincaemia', 'invasive meningococcal disease',
    'japanese encephalitis', 'muscular headache', 'relapse multiple myeloma',
    'situational hypoactive sexual desire disorder', 'viremia', 'wilson disease',
    'acute bacterial conjunctivitis', 'allergic respiratory disease',
    'amyotrophic lateral sclerosis', 'castration-resistant prostate carcinoma',
    'chronic pain syndrome', 'coagulation protein disease', 'epidermolysis bullosa dystrophica',
    'familial dupuytren contracture', 'familial chylomicronemia syndrome',
    'glycogen storage disease ii', 'hemophilia a', 'hemophilia b',
    'heparin-induced thrombocytopenia', 'hepatitis a', 'hereditary angioedema',
    'interstitial cystitis', 'junctional epidermolysis bullosa', 'myelofibrosis',
    'non-invasive bladder urothelial carcinoma', 'opioid-induced constipation',
    'osteoarthritis, knee', 'primary biliary cholangitis 1', 'primary hyperoxaluria type 1',
    'psoriasis vulgaris', 'schizoaffective disorder', 'skin carcinoma in situ',
    'tardive dyskinesia', 'tenosynovial giant cell tumor', 'uterine fibroid',
    # Additional mechanism-specific diseases (rare genetic, targeted therapies)
    'gaucher disease', 'niemann-pick disease', 'pompe disease', 'tay-sachs disease',
}

# h297: Highly Repurposable Diseases (kNN works well - boost to HIGH confidence)
# These diseases have breadth >= 5 and mono_pct < 40%
# Their drugs are widely used across multiple diseases (good kNN candidates)
HIGHLY_REPURPOSABLE_DISEASES = {
    'aids', 'abdominal infection', 'acute pain', 'acute bacterial sinusitis',
    'acute bronchitis', 'acute bursitis', 'acute glomerulonephritis', 'acute otitis media',
    'acute rheumatic carditis', 'addison disease', 'advanced breast cancer',
    'advanced renal cell carcinoma', 'angina pectoris', 'arrhythmia', 'atopic rhinitis',
    'axial spondyloarthritis', 'b-cell chronic lymphocytic leukemia', 'back pain',
    'bacterial sepsis', 'bone and joint infections', 'bronchospasm',
    'carcinoma breast stage iv', 'chronic anterior uveitis', 'chronic pain',
    'common migraine', 'corneal injury', 'crohn disease', 'cyclitis',
    'deep venous thrombosis', 'depressed mood', 'diabetes', 'diamond-blackfan anemia',
    'dyslipidemia', 'dysmenorrhea', 'edema', 'erosive esophagitis',
    'erosive gastroesophageal reflux disease', 'erythroderma', 'lupus erythematosus',
    'idiopathic nephrotic syndrome', 'pure red-cell aplasia', 'aspiration pneumonitis',
    'trichinellosis', 'ocular cicatricial pemphigoid', 'idiopathic eosinophilic pneumonia',
}

# h274: Cancer type matching for confidence tiering (from h270 analysis)
# Same cancer type (subtype refinement): 100% precision
# Different cancer type (cross-repurposing): 30.6% precision
# No cancer GT: 0% precision - FILTER these
CANCER_TYPE_KEYWORDS = {
    'lymphoma': ['lymphoma', 'lymphomas', 'hodgkin', 'non-hodgkin', 'dlbcl', 'follicular lymphoma',
                 'mantle cell', 'burkitt', 'marginal zone', 'lymphoblastic'],
    'leukemia': ['leukemia', 'leukaemia', 'cll', 'aml', 'all', 'cml', 'myeloid leukemia',
                 'lymphocytic leukemia', 'acute leukemia', 'chronic leukemia'],
    'carcinoma': ['carcinoma', 'adenocarcinoma', 'squamous cell', 'hepatocellular', 'renal cell',
                  'transitional cell', 'small cell', 'non-small cell', 'nsclc', 'sclc'],
    'melanoma': ['melanoma', 'melanotic'],
    'sarcoma': ['sarcoma', 'leiomyosarcoma', 'osteosarcoma', 'ewing', 'rhabdomyosarcoma',
                'liposarcoma', 'fibrosarcoma', 'angiosarcoma'],
    'myeloma': ['myeloma', 'multiple myeloma', 'plasma cell myeloma'],
    # h274: Add broad 'cancer' category for generic cancer terms
    # This captures "breast cancer", "lung cancer", etc. that don't specify histology
    'solid_tumor': ['cancer', 'tumor', 'tumour', 'neoplasm', 'malignant', 'metastatic',
                    'oncology', 'glioma', 'glioblastoma', 'neuroblastoma', 'blastoma'],
    # h710: Benign/intermediate proliferative disorders treated with mTOR inhibitors.
    # These are categorized as cancer but extract_cancer_types didn't recognize them,
    # causing cancer_no_gt filter to block sirolimus (standard of care).
    'vascular_proliferative': ['hemangioendothelioma', 'lymphangioma', 'lymphangioleiomyomatosis'],
}

# h273: Disease hierarchy groups for cross-category matching
# When prediction disease doesn't exactly match GT but is in same disease group,
# this indicates the drug treats a related condition (2.9x precision improvement)
#
# Key findings from h273:
# - Autoimmune: 18.4% exact → 37.2% hierarchy (+18.8pp)
# - Metabolic: 14.4% exact → 49.6% hierarchy (+35.2pp)
# - Infectious: 4.3% exact → 20.4% hierarchy (+16.1pp)
# - Cardiovascular: 3.1% exact → 18.0% hierarchy (+14.9pp)
#
DISEASE_HIERARCHY_GROUPS = {
    'autoimmune': {
        'psoriasis': ['psoriasis', 'plaque psoriasis', 'chronic plaque psoriasis', 'scalp psoriasis',
                      'erythrodermic psoriasis', 'pustular psoriasis', 'guttate psoriasis', 'psoriasis vulgaris'],
        'rheumatoid_arthritis': ['rheumatoid arthritis', 'juvenile rheumatoid arthritis', 'juvenile idiopathic arthritis',
                                  'polyarticular juvenile idiopathic arthritis', 'systemic juvenile idiopathic arthritis',
                                  'osteoarthritis', 'arthritis', 'psoriatic arthritis'],
        'multiple_sclerosis': ['multiple sclerosis', 'relapsing multiple sclerosis', 'remitting multiple sclerosis',
                               'progressive multiple sclerosis', 'primary progressive multiple sclerosis',
                               'secondary progressive multiple sclerosis', 'relapsing-remitting multiple sclerosis'],
        # h410: Removed 'sle' - too short, matches 'sleep', 'sleepiness' (false positives)
        'lupus': ['lupus', 'systemic lupus erythematosus', 'discoid lupus', 'lupus nephritis',
                  'membranous lupus nephritis', 'cutaneous lupus'],
        'colitis': ['colitis', 'ulcerative colitis', 'chronic ulcerative colitis', 'pediatric ulcerative colitis',
                    'crohns disease', 'crohn disease', 'crohn colitis', 'inflammatory bowel disease'],
        'scleroderma': ['scleroderma', 'systemic sclerosis', 'systemic sclerosis associated interstitial lung disease',
                        'diffuse scleroderma', 'limited scleroderma'],
        'spondylitis': ['ankylosing spondylitis', 'axial spondyloarthritis', 'non-radiographic axial spondyloarthritis'],
    },
    # h387: Removed hepatitis and HIV groups - viral diseases have 0% hierarchy precision
    # Kept: UTI (75%), tuberculosis (45.5%), pneumonia, sepsis, skin_infection, respiratory_infection
    'infectious': {
        'pneumonia': ['pneumonia', 'bronchopneumonia', 'community-acquired pneumonia', 'hospital-acquired pneumonia',
                      'streptococcal pneumonia', 'pneumococcal pneumonia', 'bacterial pneumonia', 'aspiration pneumonia'],
        # h387: hepatitis REMOVED - 0% precision (viral diseases don't work with hierarchy)
        # h410: Keep 'cystitis' but use HIERARCHY_EXCLUSIONS to block cholecystitis/dacryocystitis/interstitial cystitis
        'uti': ['urinary tract infection', 'uti', 'complicated urinary tract infection', 'uncomplicated uti',
                'recurrent uti', 'chronic urinary tract infection', 'pyelonephritis', 'cystitis'],
        'sepsis': ['sepsis', 'bacterial sepsis', 'septicemia', 'blood stream infection', 'severe sepsis', 'septic shock'],
        'skin_infection': ['skin infection', 'cellulitis', 'impetigo', 'wound infection', 'burn infection',
                           'skin and soft tissue infection', 'abscess'],
        'respiratory_infection': ['respiratory infection', 'bronchitis', 'acute bronchitis', 'chronic bronchitis',
                                  'respiratory tract infection', 'upper respiratory infection', 'lower respiratory infection'],
        'tuberculosis': ['tuberculosis', 'tb', 'pulmonary tuberculosis', 'latent tuberculosis', 'multidrug-resistant tuberculosis'],
        # h387: HIV REMOVED - 0% precision (viral diseases don't work with hierarchy)
    },
    'neurological': {
        'epilepsy': ['epilepsy', 'seizure', 'seizure disorder', 'partial seizure', 'generalized seizure',
                     'focal seizure', 'absence seizure', 'tonic-clonic seizure', 'status epilepticus'],
        'parkinsons': ['parkinson', "parkinson's disease", 'parkinsons disease', 'parkinsonism', 'tremor'],
        'alzheimers': ['alzheimer', "alzheimer's disease", 'dementia', 'cognitive impairment', 'memory loss'],
        'migraine': ['migraine', 'headache', 'chronic migraine', 'episodic migraine', 'cluster headache', 'tension headache'],
        'neuropathy': ['neuropathy', 'peripheral neuropathy', 'diabetic neuropathy', 'polyneuropathy', 'nerve damage'],
        # h467: Removed 'tia' - too short, matches 'interstitial' (4 false matches)
        'stroke': ['stroke', 'cerebrovascular', 'ischemic stroke', 'hemorrhagic stroke', 'transient ischemic attack'],
    },
    'cardiovascular': {
        'heart_failure': ['heart failure', 'congestive heart failure', 'chf', 'left ventricular failure',
                          'right heart failure', 'cardiomyopathy', 'dilated cardiomyopathy'],
        'hypertension': ['hypertension', 'high blood pressure', 'essential hypertension', 'pulmonary hypertension',
                         'resistant hypertension', 'renovascular hypertension'],
        'arrhythmia': ['arrhythmia', 'atrial fibrillation', 'afib', 'ventricular arrhythmia', 'tachycardia',
                       'bradycardia', 'supraventricular tachycardia', 'ventricular tachycardia'],
        'coronary': ['coronary', 'angina', 'coronary artery disease', 'ischemic heart disease',
                     'acute coronary syndrome', 'myocardial infarction', 'heart attack'],
        'atherosclerosis': ['atherosclerosis', 'arteriosclerosis', 'plaque', 'arterial disease'],
    },
    'metabolic': {
        'diabetes': ['diabetes', 'type 2 diabetes', 'type 1 diabetes', 'diabetic', 'hyperglycemia',
                     'hypoglycemia', 'diabetic nephropathy', 'diabetic neuropathy', 'diabetic retinopathy',
                     'diabetes mellitus', 'diabetes insipidus'],
        'obesity': ['obesity', 'overweight', 'weight management', 'bmi'],
        'lipid': ['hyperlipidemia', 'dyslipidemia', 'hypercholesterolemia', 'hypertriglyceridemia', 'cholesterol',
                  'familial hypercholesterolemia'],
        'thyroid': ['thyroid', 'hypothyroidism', 'hyperthyroidism', 'goiter', 'thyroiditis'],
        'gout': ['gout', 'hyperuricemia', 'uric acid'],
    },
    'respiratory': {
        'asthma': ['asthma', 'asthmatic', 'bronchospasm', 'reactive airway', 'exercise-induced asthma'],
        'copd': ['copd', 'chronic obstructive', 'emphysema'],
        # h410: Removed bare 'fibrosis' - matches 'cystic fibrosis' (CF is not pulmonary fibrosis)
        'pulmonary_fibrosis': ['pulmonary fibrosis', 'idiopathic pulmonary fibrosis', 'interstitial lung'],
    },
}

# h410: Known false-match exclusions for hierarchy substring matching.
# These disease name substrings should NEVER match the specified hierarchy group,
# even though they contain a matching variant substring.
HIERARCHY_EXCLUSIONS: Dict[Tuple[str, str], list[str]] = {
    # h410: 'cystitis' in UTI group matches cholecystitis (gallbladder), dacryocystitis (tear duct),
    # interstitial cystitis (bladder pain syndrome - not infectious)
    ('infectious', 'uti'): ['cholecystitis', 'dacryocystitis', 'interstitial cystitis'],
    # h410: 'bronchitis' in respiratory_infection matches chronic bronchitis (which is COPD, not infection)
    ('infectious', 'respiratory_infection'): ['chronic bronchitis'],
    # h410: 'cystic fibrosis' is not pulmonary fibrosis
    ('respiratory', 'pulmonary_fibrosis'): ['cystic fibrosis'],
    # h467: 'pneumonia' matches autoimmune interstitial pneumonia (not infectious)
    ('infectious', 'pneumonia'): ['interstitial pneumonia'],
    # h467: 'diabetic' matches 'nondiabetic' and 'diabetic foot infections' (infection, not metabolic)
    # h482: 'diabetes' matches 'diabetes insipidus' (completely different disease - ADH deficiency)
    ('metabolic', 'diabetes'): ['nondiabetic', 'diabetic foot', 'diabetes insipidus'],
    # h467: 'thyroid' matches 'thyroid cancer' (cancer, not metabolic)
    # h469: 'thyroid' matches 'parathyroid' (parathyroid gland ≠ thyroid gland, different organ)
    ('metabolic', 'thyroid'): ['thyroid cancer', 'parathyroid'],
}

# h171: Neurological drug class mappings (60.4% coverage vs 18% kNN baseline)
# Maps disease subtypes to appropriate drug classes
NEUROLOGICAL_DISEASE_DRUG_CLASSES = {
    # Epilepsy, seizure -> anticonvulsants (65% precision)
    'epilepsy': ['anticonvulsant'],
    'seizure': ['anticonvulsant'],
    # Parkinson's -> dopaminergic (58% precision)
    'parkinson': ['dopaminergic'],
    # Alzheimer's, dementia -> ChEI/NMDA (50% precision)
    'alzheimer': ['cholinesterase_inhibitor', 'nmda_antagonist'],
    'dementia': ['cholinesterase_inhibitor', 'nmda_antagonist'],
    # Migraine -> triptans (78% precision)
    'migraine': ['triptan', 'cgrp_inhibitor'],
    'headache': ['triptan'],
    # Neuropathy -> gabapentinoids (100% precision for generic neuropathy)
    'neuropathy': ['gabapentinoid', 'tricyclic'],
    'neuralgia': ['anticonvulsant', 'tricyclic'],
    # Movement disorders
    'dyskinesia': ['dopaminergic', 'anticholinergic'],
    'dystonia': ['anticholinergic'],
    # Sleep disorders
    'narcolepsy': ['stimulant', 'wake_promoting'],
}

# h171: Drug class members
NEUROLOGICAL_DRUG_CLASS_MEMBERS = {
    'anticonvulsant': [
        'carbamazepine', 'valproic acid', 'phenytoin', 'lamotrigine',
        'topiramate', 'levetiracetam', 'gabapentin', 'pregabalin',
        'oxcarbazepine', 'zonisamide', 'lacosamide', 'perampanel',
        'clobazam', 'clonazepam', 'brivaracetam', 'eslicarbazepine'
    ],
    'dopaminergic': [
        'l-dopa', 'levodopa', 'carbidopa', 'pramipexole', 'ropinirole',
        'bromocriptine', 'apomorphine', 'rotigotine', 'amantadine',
        'entacapone', 'rasagiline', 'selegiline', 'safinamide'
    ],
    'cholinesterase_inhibitor': ['donepezil', 'rivastigmine', 'galantamine'],
    'nmda_antagonist': ['memantine'],
    'triptan': [
        'sumatriptan', 'rizatriptan', 'zolmitriptan', 'eletriptan',
        'naratriptan', 'almotriptan', 'frovatriptan', 'lasmiditan'
    ],
    'cgrp_inhibitor': ['erenumab', 'fremanezumab', 'galcanezumab', 'ubrogepant', 'rimegepant'],
    'gabapentinoid': ['gabapentin', 'pregabalin'],
    'tricyclic': ['amitriptyline', 'nortriptyline', 'desipramine'],
    'anticholinergic': ['trihexyphenidyl', 'benztropine', 'biperiden'],
    'stimulant': ['amphetamine', 'methylphenidate', 'modafinil', 'armodafinil', 'solriamfetol'],
    'wake_promoting': ['modafinil', 'armodafinil', 'pitolisant', 'solriamfetol'],
}

# h380: GI Disease Drug Classes (from h377 analysis)
# GI category is worst performing (42.9% R@30) because kNN finds neighbors
# from different categories (infectious, neurological) with different drug needs.
GI_DISEASE_DRUG_CLASSES = {
    # Constipation diseases -> laxatives (0/4 hits without rescue)
    'constipation': ['laxative', 'opioid_antagonist'],
    'opioid-induced': ['opioid_antagonist', 'laxative'],
    # Liver/hepatic diseases -> bile acid agents (0/3 hits without rescue)
    'cholangitis': ['bile_acid_agent'],
    'cholestasis': ['bile_acid_agent'],
    'hepatic encephalopathy': ['ammonia_reducer'],
    # Ulcer diseases -> PPIs (0/1 hits without rescue)
    'ulcer': ['ppi', 'h2_blocker', 'cytoprotective'],
    'reflux': ['ppi', 'h2_blocker'],
    'gerd': ['ppi', 'h2_blocker'],
    # IBS -> antispasmodics
    'irritable bowel': ['antispasmodic', 'laxative'],
    # Inflammatory (but check autoimmune category first)
    'pancreatitis': ['pancreatic_enzyme', 'ppi'],
}

# h380: GI Drug Class Members
GI_DRUG_CLASS_MEMBERS = {
    'laxative': [
        'lactulose', 'lubiprostone', 'prucalopride', 'plecanatide',
        'linaclotide', 'macrogol', 'sennosides', 'bisacodyl',
        'polyethylene glycol', 'docusate', 'elobixibat', 'tegaserod'
    ],
    'opioid_antagonist': [
        'naldemedine', 'naloxegol', 'methylnaltrexone', 'alvimopan'
    ],
    'bile_acid_agent': [
        'cholestyramine', 'obeticholic acid', 'ursodeoxycholic acid',
        'maralixibat', 'odevixibat', 'seladelpar', 'colestipol'
    ],
    'ammonia_reducer': ['lactulose', 'rifaximin', 'neomycin'],
    'ppi': [
        'omeprazole', 'esomeprazole', 'lansoprazole', 'pantoprazole',
        'rabeprazole', 'dexlansoprazole', 'vonoprazan'
    ],
    'h2_blocker': ['ranitidine', 'famotidine', 'cimetidine', 'nizatidine'],
    'cytoprotective': ['sucralfate', 'misoprostol', 'bismuth'],
    'antispasmodic': ['dicyclomine', 'hyoscyamine', 'peppermint oil'],
    'pancreatic_enzyme': ['pancrelipase', 'pancreatin'],
}

# h170: Selective category boosting (VALIDATED: +2.40pp, p=0.009)
# Only boost same-category neighbors for isolated categories where it helps
# Improves neurological +14.3pp, respiratory +16.8pp, metabolic +13.9pp
# Without hurting infectious (-11.2pp) or other (-4.8pp) that would be hurt by universal boost
SELECTIVE_BOOST_CATEGORIES = {
    'neurological', 'respiratory', 'metabolic', 'renal', 'hematological', 'immunological'
}
SELECTIVE_BOOST_ALPHA = 0.5  # Similarity multiplier: sim * (1 + alpha) for same-category neighbors

# h374: MinRank Ensemble Categories (from h369/h370 validation)
# NOTE: h374 INVALIDATED - MinRank ensemble does not help in production context
# The production predictor already has h274 cancer_same_type and other rules that
# capture the same target overlap signal. MinRank was validated in isolation but
# is redundant when combined with production tier assignment rules.
# R@30: MinRank 83.7% vs kNN 83.9% (-0.2%) - no improvement
# Keeping constants for reference but set to empty (disabled)
MINRANK_ENSEMBLE_CATEGORIES: set[str] = set()  # Disabled - was {'cancer', 'neurological', 'metabolic'}

# h374: Categories where Target-only is better (gap >10% from h370)
# NOTE: Also disabled as MinRank is not used
# h397: TARGET_DOMINANT_CATEGORIES removed (was dead code, set to empty set)

# h388: Target overlap tier promotion thresholds
# Use drug-disease target overlap to promote tier WITHOUT changing rankings.
# HIGH + overlap>=3 + eligible rule → GOLDEN (69.5-91.3% precision vs 38.4% baseline)
# LOW + overlap>=1 → MEDIUM (37.9% precision vs 19.9% baseline)
# Guard: HIGH→GOLDEN only for rules with demonstrated good precision.
# 'default' HIGH = 0% precision when promoted (broad-spectrum drugs like corticosteroids).
TARGET_OVERLAP_PROMOTE_HIGH_TO_GOLDEN = 3  # Minimum overlap for HIGH→GOLDEN
TARGET_OVERLAP_PROMOTE_LOW_TO_MEDIUM = 1   # Minimum overlap for LOW→MEDIUM
# Rules eligible for HIGH→GOLDEN promotion (demonstrated >33% precision with overlap>=3)
TARGET_OVERLAP_GOLDEN_ELIGIBLE_RULES: set[str] = {
    # h402: cv_pathway_comprehensive removed (demoted from HIGH to MEDIUM)
    'cardiovascular_hierarchy_hypertension',
    'cardiovascular_hierarchy_arrhythmia',
    'cardiovascular_hierarchy_coronary',
    'comp_to_base_high_87',
    'autoimmune_hierarchy_rheumatoid_arthritis',
    'autoimmune_hierarchy_multiple_sclerosis',
    'autoimmune_hierarchy_spondylitis',
    'autoimmune_hierarchy_lupus',
    'autoimmune_hierarchy_colitis',
    'respiratory_hierarchy_asthma',
    'respiratory_hierarchy_copd',
    'infectious_hierarchy_tuberculosis',
    'metabolic_hierarchy_thyroid',
}

# h444: Holdout-validated rank-bucket precision (5-seed mean)
# Used for clinical reporting: expected precision given tier + rank bucket
# Format: (tier, rank_lo, rank_hi) -> holdout_precision
RANK_BUCKET_PRECISION: Dict[Tuple[str, int, int], float] = {
    # h457: Updated with 5-seed holdout validation. Reliability varies by tier:
    # GOLDEN: reliable R1-15, R16-20 unreliable (29.5pp full-to-holdout gap)
    ('GOLDEN', 1, 5): 61.5, ('GOLDEN', 6, 10): 65.8,
    ('GOLDEN', 11, 15): 50.9, ('GOLDEN', 16, 20): 32.3,
    # HIGH: UNRELIABLE calibration - hierarchy rescue creates non-monotonic rank patterns
    # Full-to-holdout gaps: R1-5=+15pp, R6-10=+21pp, R11-15=+38pp, R16-20=+34pp
    ('HIGH', 1, 5): 34.6, ('HIGH', 6, 10): 30.8,
    ('HIGH', 11, 15): 15.8, ('HIGH', 16, 20): 20.0,
    # MEDIUM: MOST RELIABLE - monotonic on holdout, clean gradient
    ('MEDIUM', 1, 5): 20.5, ('MEDIUM', 6, 10): 16.3,
    ('MEDIUM', 11, 15): 10.8, ('MEDIUM', 16, 20): 9.2,
    # LOW: rank calibration REVERSES on holdout (collider effect, h453)
    ('LOW', 1, 5): 3.8, ('LOW', 6, 10): 2.0,
    ('LOW', 11, 15): 6.5, ('LOW', 16, 20): 5.2,
    # FILTER: approximately monotonic on holdout
    ('FILTER', 1, 5): 6.0, ('FILTER', 6, 10): 7.8,
    ('FILTER', 11, 15): 3.2, ('FILTER', 16, 20): 3.0,
}


def get_rank_bucket_precision(tier: str, rank: int) -> float:
    """Get holdout-validated precision for a tier + rank combination."""
    if rank <= 5:
        lo, hi = 1, 5
    elif rank <= 10:
        lo, hi = 6, 10
    elif rank <= 15:
        lo, hi = 11, 15
    elif rank <= 20:
        lo, hi = 16, 20
    else:
        # rank > 20: use tier's 16-20 bucket as conservative estimate
        lo, hi = 16, 20
    return RANK_BUCKET_PRECISION.get((tier, lo, hi), 0.0)


# h462: Category-specific MEDIUM holdout precision (5-seed mean)
# Used for clinical reporting: expected precision given category within MEDIUM tier
# Format: category -> holdout_precision (%)
# Categories not listed use the default MEDIUM holdout precision (30.8% after h478 GT sync)
CATEGORY_MEDIUM_HOLDOUT_PRECISION: Dict[str, float] = {
    # h498: Updated with h499 corrected GT holdout values (post h478 GT sync)
    'musculoskeletal': 55.6,   # ±29.8 (HIGH VARIANCE, small n)
    'dermatological': 48.4,    # ±14.5 (biggest beneficiary of GT sync, +19.9pp)
    'autoimmune': 36.4,        # ±7.1
    'psychiatric': 33.3,       # ±5.2 (dropped from 45.7%, seed distribution change)
    'cardiovascular': 19.5,    # h490: standard+ATC demoted to LOW, remaining MEDIUM = PC(21.4%)+overlap(16.2%)
    'respiratory': 30.5,       # ±6.5 (+14.3pp from GT sync)
    'infectious': 27.9,        # ±7.0
    'cancer': 24.5,            # ±3.7 (MOST RELIABLE, largest n)
    # Categories below not updated by h499 (small n, using h462 values):
    'renal': 43.5,             # ±25.2 (HIGH VARIANCE, n=12/seed)
    'endocrine': 23.4,         # ±5.2
    'ophthalmic': 23.3,        # ±16.2
    'metabolic': 19.4,         # ±10.4
    'other': 17.2,             # ±14.7
    'hematological': 10.0,     # h553: ±20.0, n=8/seed → demoted to LOW
    # Demoted categories (now LOW): holdout-validated at demotion
    'neurological': 5.8,       # h499: ±? → demoted to LOW, confirmed justified
    'immunological': 8.3,      # h499: → demoted to LOW, confirmed justified
    'gastrointestinal': 5.0,   # h499: → demoted to LOW, confirmed justified
    'reproductive': 2.9,       # h499: → demoted to LOW, confirmed justified
}


def get_category_holdout_precision(category: str, tier: str) -> float:
    """Get holdout-validated precision for a category+tier combination.

    For MEDIUM tier, returns category-specific holdout precision from h462.
    For other tiers, returns the default tier holdout precision.
    """
    if tier == 'MEDIUM' and category.lower() in CATEGORY_MEDIUM_HOLDOUT_PRECISION:
        return CATEGORY_MEDIUM_HOLDOUT_PRECISION[category.lower()]
    # h498: Default tier holdout precisions (from h478 GT sync + subsequent fixes)
    tier_defaults = {
        'GOLDEN': 67.0,
        'HIGH': 60.8,
        'MEDIUM': 30.8,
        'LOW': 14.8,
        'FILTER': 10.3,
    }
    return tier_defaults.get(tier, 0.0)


# h169/h148: Expanded category keywords to reduce 'other' bucket
# h148 reduced 'other' from 44.9% to ~25% with comprehensive keyword expansion
CATEGORY_KEYWORDS = {
    'autoimmune': ['autoimmune', 'lupus', 'rheumatoid', 'arthritis', 'scleroderma', 'myasthenia',
                   'multiple sclerosis', 'crohn', 'colitis', 'psoriasis', 'sjögren', 'behcet',
                   'spondylitis', 'vasculitis', 'dermatomyositis', 'polymyositis', 'still disease',
                   # h148: Additional autoimmune patterns
                   'graft versus host', 'eosinophilic granulomatosis', 'autoinflammatory',
                   'microscopic polyangiitis', 'lichen planus', 'familial mediterranean fever',
                   # h188: Additional from h186 analysis
                   'polyarteritis nodosa', 'takayasu', 'temporal arteritis', 'pyoderma gangrenosum',
                   'rheumatic fever', 'sarcoidosis'],
    'infectious': ['infection', 'bacterial', 'viral', 'fungal', 'hiv', 'aids', 'hepatitis',
                   'tuberculosis', 'malaria', 'pneumonia', 'sepsis', 'meningitis', 'amebiasis',
                   'aspergillosis', 'brucellosis', 'actinomycosis', 'bartonellosis', 'burkholderia',
                   'parasitic', 'otitis media', 'endocarditis', 'osteomyelitis', 'cellulitis',
                   # h148: Additional infectious patterns
                   'cholera', 'chagas', 'trypanosomiasis', 'leishmaniasis', 'cryptococcosis',
                   'coccidioidomycosis', 'histoplasmosis', 'blastomycosis', 'candidiasis',
                   'chromomycosis', 'common cold', 'yellow fever', 'rabies', 'leptospirosis',
                   'listeriosis', 'botulism', 'tetanus', 'anthrax', 'plague', 'typhoid',
                   'diphtheria', 'pertussis', 'polio', 'measles', 'mumps', 'rubella',
                   'toxoplasmosis', 'herpes zoster', 'shingles', 'fusariosis', 'zygomycosis',
                   'schistosomiasis', 'strongyloides', 'giardia', 'dysentery', 'gastroenteritis',
                   'influenza', 'japanese encephalitis', 'leprosy', 'mycetoma', 'nocardiosis',
                   'onchocerciasis', 'gonococcal',
                   # h188: Additional from h186 analysis
                   'q fever', 'relapsing fever', 'tularemia', 'syphilis', 'yaws', 'trichomoniasis',
                   'smallpox', 'encephalitis', 'shigellosis', 'sporotrichosis', 'scabies',
                   'ringworm', 'tinea', 'tonsillitis', 'vulvovaginitis', 'proctitis',
                   'staphylococcus', 'pseudomonas'],
    'cancer': ['cancer', 'carcinoma', 'tumor', 'leukemia', 'lymphoma', 'melanoma',
               'neoplasm', 'oncology', 'sarcoma', 'myeloma', 'glioma', 'adenocarcinoma',
               'neuroblastoma', 'mastocytosis',
               # h148: Additional cancer patterns
               'carcinoid', 'ependymoma', 'hemangioendothelioma', 'mole', 'neoplasia',
               'langerhans cell histiocytosis', 'lymphangioleiomyomatosis', 'lymphangioma',
               'medulloblastoma',
               # h621: Cancer diseases miscategorized as other specialties
               'mesothelioma', 'retinoblastoma'],
    'cardiovascular': ['cardiac', 'heart', 'coronary', 'hypertension', 'arrhythmia',
                       'atherosclerosis', 'stroke', 'vascular', 'myocardial', 'angina',
                       'tachycardia', 'aneurysm', 'aorta', 'thrombosis', 'embolism',
                       'cardiomyopathy', 'ischemia',
                       # h148: Additional cardiovascular patterns
                       'cerebral infarction', 'deep vein thrombosis', 'dvt', 'pulmonary embolism',
                       'raynaud', 'claudication', 'peripheral arterial', 'varicose', 'phlebitis',
                       'patent ductus arteriosus', 'orthostatic hypotension',
                       # h188: Additional from h186 analysis
                       'torsades de pointes', 'tetralogy of fallot', 'edema', 'lymphedema',
                       # h230: AFib and flutter (previously missing)
                       'atrial fibrillation', 'atrial flutter', 'fibrillation'],
    'neurological': ['neurological', 'alzheimer', 'parkinson', 'epilepsy', 'neuropathy',
                     'dementia', 'huntington', 'brain', 'seizure', 'ataxia', 'dystonia',
                     'dyskinesia', 'narcolepsy', 'migraine', 'neuralgia', 'headache',
                     # h148: Additional neurological patterns
                     'als', 'amyotrophic lateral sclerosis', 'cerebral', 'motor neuron',
                     'spinal cord', 'spinal muscular', 'chorea', 'cerebral palsy',
                     'hydrocephalus', 'myelitis', 'lennox gastaut', 'multiple system atrophy',
                     'neurofibromatosis', 'neuromuscular', 'neuronal ceroid lipofuscinosis',
                     'motion sickness',
                     # h188: Additional from h186 analysis
                     'tuberous sclerosis', 'periodic paralysis', 'pure autonomic failure',
                     'achondroplasia'],
    'metabolic': ['diabetes', 'metabolic', 'obesity', 'thyroid', 'hyperlipidemia',
                  'hypercholesterolemia', 'gout', 'porphyria', 'glycogen storage',
                  'hyperuricemia', 'acromegaly', 'addison', 'adrenal', 'hypoglycemia',
                  'hyperglycemia', 'cushing',
                  # h148: Additional metabolic patterns
                  'hyperammonemia', 'acidemia', 'alkaptonuria', 'cystinosis', 'cystinuria',
                  'phenylketonuria', 'homocystinuria', 'galactosemia', 'tyrosinemia',
                  'maple syrup', 'fatty acid oxidation', 'urea cycle', 'diabetic',
                  'dyslipidemia', 'lipodystrophy', 'gaucher', 'fabry', 'niemann-pick',
                  'chylomicronemia', 'mucopolysaccharidosis', 'sphingolipidosis',
                  'lipid storage', 'lysosomal storage', 'multinodular goiter', 'myxedema',
                  'hypophosphatasia', 'lysosomal acid lipase', 'mevalonate kinase',
                  'ornithine carbamoyltransferase',
                  # h188: Additional from h186 analysis
                  'thyrotoxicosis', 'zollinger ellison', 'lactic acidosis',
                  'hyperlipoproteinemia', 'hyperphenylalaninemia', 'scurvy'],
    'psychiatric': ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'psychiatric',
                    'ptsd', 'ocd', 'adhd', 'psychosis', 'agoraphobia', 'bulimia', 'anorexia',
                    'alcohol withdrawal', 'insomnia', 'sleep disorder', 'panic disorder',
                    # h148: Additional psychiatric patterns
                    'hyperactive', 'major depressive', 'obsessive compulsive'],
    'respiratory': ['respiratory', 'asthma', 'copd', 'pulmonary', 'lung', 'bronchitis',
                    'pneumonitis', 'fibrosis', 'bronchiectasis', 'emphysema', 'pleurisy',
                    # h148: Additional respiratory patterns
                    'rhinitis', 'sinusitis', 'pharyngitis', 'laryngitis', 'bronchiolitis',
                    'croup', 'whooping cough', 'pleural', 'pneumothorax', 'tracheitis',
                    'empyema', 'pneumoconiosis', 'silicosis', 'asbestosis',
                    'obstructive sleep apnea'],
    'gastrointestinal': ['gastrointestinal', 'gastric', 'intestinal', 'bowel', 'liver',
                         'hepatic', 'cirrhosis', 'pancreatitis', 'celiac', 'dysphagia',
                         'cholecystitis', 'cholangitis', 'esophageal', 'dyspepsia', 'gerd',
                         # h148: Additional gastrointestinal patterns
                         'peptic ulcer', 'gastroparesis', 'diverticulitis', 'appendicitis',
                         'peritonitis', 'ascites', 'duodenal ulcer', 'esophagitis',
                         'pancreat', 'gallbladder', 'diarrhea', 'constipation', 'ileus',
                         'volvulus', 'intussusception', 'malabsorption', 'chronic cholestasis',
                         # h188: Additional from h186 analysis
                         'gingivitis', 'dental caries', 'stomatitis'],
    'dermatological': ['skin', 'dermatitis', 'eczema', 'dermatological',
                       'acne', 'urticaria', 'vitiligo', 'alopecia', 'pruritus', 'rosacea',
                       'angioedema', 'blepharitis',
                       # h148: Additional dermatological patterns
                       'actinic keratosis', 'ichthyosis', 'keratosis', 'seborrheic',
                       'pemphigus', 'pemphigoid', 'erythema', 'impetigo',
                       'folliculitis', 'furuncle', 'carbuncle', 'paronychia',
                       'hidradenitis', 'hordeolum', 'stye', 'dermatosis', 'lichenoid',
                       'hyperhidrosis', 'hirsutism', 'keloid', 'psoriatic', 'otitis externa',
                       # h188: Additional from h186 analysis
                       'pityriasis', 'toxic epidermal necrolysis', 'pyoderma'],
    'ophthalmic': ['eye', 'retinal', 'glaucoma', 'macular', 'ophthalmic', 'uveitis',
                   'conjunctivitis', 'keratitis', 'blepharoconjunctivitis', 'cataract',
                   # h148: Additional ophthalmic patterns
                   'corneal', 'choroiditis', 'dacryocystitis', 'iritis', 'retinitis',
                   'scleritis', 'optic', 'blindness', 'keratoconus', 'pterygium',
                   'hyperopia', 'myopia',
                   # h188: Additional from h186 analysis
                   'retinopathy'],  # h621: retinoblastoma moved to cancer
    'hematological': ['anemia', 'hemophilia', 'thrombocytopenia',
                      'neutropenia', 'hematological', 'myelodysplastic', 'polycythemia',
                      'agranulocytosis', 'coagulation', 'thalassemia', 'sickle cell',
                      # h148: Additional hematological patterns
                      'thrombotic thrombocytopenic', 'hemolytic uremic', 'purpura',
                      'coagulopathy', 'hemorrhagic', 'von willebrand', 'factor viii',
                      'prothrombin deficiency', 'protein c deficiency', 'protein s deficiency',
                      'factor deficiency', 'hemostasis', 'leukopenia',
                      'paroxysmal nocturnal hemoglobinuria', 'hypereosinophilic',
                      # h188: Additional from h186 analysis
                      'red cell aplasia', 'osteopetrosis', 'thrombophilia'],
    # h169: New categories to reduce 'other' bucket
    'renal': ['kidney', 'renal', 'nephropathy', 'nephritis', 'uremia', 'glomerular',
              'nephrotic', 'dialysis',
              # h148: Additional renal patterns
              'cystitis', 'pyelonephritis', 'urolithiasis', 'nephrolithiasis',
              'hydronephrosis', 'urinary incontinence', 'overactive bladder',
              # h188: Additional from h186 analysis
              'vesicoureteral reflux', 'proteinuria'],
    'musculoskeletal': ['bone', 'osteoporosis', 'osteomalacia', 'fracture', 'bursitis',
                        'tendonitis', 'fibromyalgia', 'osteogenesis', 'paget',
                        # h148: Additional musculoskeletal patterns
                        'muscular dystrophy', 'myotonic', 'rhabdomyolysis', 'fasciitis', 'myositis',
                        # h188: Additional from h186 analysis
                        'tendinitis', 'tenosynovitis', 'spina bifida', 'paraplegia'],
    'immunological': ['immunodeficiency', 'agammaglobulinemia', 'complement deficiency',
                      'amyloidosis', 'hypersensitivity', 'allergy', 'immunological',
                      # h148: Additional immunological patterns
                      'anaphylaxis', 'granulomatous disease', 'hyperimmunoglobulin', 'mast cell'],
    # h148: New category for endocrine diseases
    'endocrine': ['precocious puberty', 'hypogonadotropic', 'hypopituitarism', 'pituitary',
                  'growth hormone', 'prolactinoma', 'hormone deficiency',
                  # h188: Additional from h186 analysis
                  'adrenocortical insufficiency'],
    # h188: New category for reproductive diseases
    'reproductive': ['female infertility', 'ovarian hyperstimulation', 'hypoestrogenism',
                     'endometritis', 'toxemia of pregnancy', 'primary ovarian failure'],
}

# h309/h310: Refined ATC-Category Coherence Map for confidence boosting
# Maps disease categories to expected ATC L1 codes for that category
# h309 finding: Coherent predictions (drug ATC matches category) have 35.5% precision
# vs 18.7% for incoherent (gap: +16.8 pp)
#
# Refinement: Added H (systemic hormonal, includes corticosteroids) and A (alimentary,
# where some corticosteroid formulations are classified) to inflammatory categories
# because corticosteroids are used broadly for inflammation across many disease categories
DISEASE_CATEGORY_ATC_MAP: Dict[str, Set[str]] = {
    'autoimmune': {'L', 'M', 'H', 'A'},  # Added A (corticosteroid formulations)
    'cancer': {'L'},
    'cardiovascular': {'C', 'B'},
    'dermatological': {'D', 'L', 'H', 'A'},  # Added H, A (corticosteroids)
    'infectious': {'J', 'P'},
    'metabolic': {'A', 'H'},
    'neurological': {'N'},
    'ophthalmic': {'S', 'H', 'A'},  # Added H, A (corticosteroids for inflammation)
    'ophthalmological': {'S', 'H', 'A'},  # Same for alternate spelling
    'psychiatric': {'N'},
    'respiratory': {'R', 'H', 'A'},  # Added H, A (corticosteroids for asthma/inflammation)
    'gastrointestinal': {'A'},
    'hematological': {'B', 'L', 'H', 'A'},  # Added H, A (corticosteroids for blood disorders)
    'renal': {'C', 'A', 'H'},  # h334: Added A, H for corticosteroids (nephrotic, CAH)
    'musculoskeletal': {'M', 'H', 'A'},  # Added H, A (corticosteroids for inflammation)
    'genetic': {'H', 'A'},  # For genetic disorders treated with corticosteroids
    'immunological': {'L', 'H', 'A'},  # Immunomodulators and corticosteroids
    'endocrine': {'H', 'A'},  # Hormones
    'reproductive': {'G', 'H'},  # Genitourinary hormones
    'other': set(),  # No ATC coherence for uncategorized
}

# h314/h316: ATC Mismatch-Specific Rules
# Some "incoherent" predictions actually have HIGHER precision than coherent baseline (11.7%)
# HIGH-PRECISION MISMATCHES - don't demote these (precision > coherent baseline):
# h317: Added 5 missing high-precision patterns
HIGH_PRECISION_MISMATCHES: Dict[Tuple[str, str], float] = {
    # h317: New additions (>coherent baseline)
    ('B', 'respiratory'): 30.0,      # Blood drugs for respiratory (30%!)
    ('N', 'autoimmune'): 29.2,       # Nervous system drugs for autoimmune
    ('C', 'autoimmune'): 26.7,       # Cardiovascular drugs for autoimmune
    ('N', 'dermatological'): 14.3,   # Nervous system drugs for dermatological
    ('C', 'genetic'): 12.0,          # Cardiovascular drugs for genetic
    # Original h314 set
    ('D', 'respiratory'): 27.5,      # Dermatological drugs for respiratory
    ('A', 'ophthalmological'): 26.5, # Alimentary drugs for ophthalmic
    ('A', 'ophthalmic'): 26.5,       # Same for alternate spelling
    ('D', 'autoimmune'): 20.0,       # Dermatological drugs for autoimmune
    ('J', 'respiratory'): 17.8,      # Antiinfectives for respiratory
    ('A', 'infectious'): 17.3,       # Alimentary drugs for infectious
    ('L', 'gastrointestinal'): 17.2, # Antineoplastic for GI
    ('A', 'other'): 13.3,            # Alimentary drugs for other
    ('C', 'other'): 11.7,            # Cardiovascular for other
    ('D', 'infectious'): 11.2,       # Dermatological for infectious
    ('D', 'other'): 10.3,            # Dermatological for other
}

# ZERO-PRECISION MISMATCHES - always FILTER these (precision < 3%):
# h316: Initial set from h314 analysis
# h318: Added comprehensive J (antibiotic) filter for non-infectious diseases
# h319: Batch 2 - all remaining 0% precision ATC→category pairs
ZERO_PRECISION_MISMATCHES: Set[Tuple[str, str]] = {
    # h316: Original set
    ('A', 'cancer'),          # 6.1% - Alimentary for cancer (h415: some valid, e.g. H2 blockers for mastocytosis)
    ('B', 'other'),           # 0.0% - Blood drugs for other
    ('J', 'dermatological'),  # 8.4% - Antibiotics for skin (h415: valid for impetigo/otitis)
    ('N', 'cancer'),          # 0.0% - Nervous system drugs for cancer
    ('A', 'immune'),          # 0.0% - Alimentary for immune disorders
    ('A', 'immunological'),   # 0.0% - Same for alternate name
    # h415 REMOVED: ('R', 'other') - 100% precision (codeine/hydrocodone for pain are valid analgesics)
    ('J', 'cancer'),          # 0.0% - Antibiotics for cancer
    ('L', 'ophthalmological'), # 1.3% - Antineoplastic for ophthalmic
    ('L', 'ophthalmic'),      # 8.1% - h415: cyclosporine for keratitis, adalimumab for uveitis are valid
    ('G', 'other'),           # 12.5% - h415: naproxen for pain is valid
    ('D', 'cancer'),          # 8.8% - Dermatological for cancer (h415: some valid e.g. imiquimod for BCC)
    ('J', 'musculoskeletal'), # 1.6% - Antibiotics for musculoskeletal
    ('J', 'genetic'),         # 1.6% - Antibiotics for genetic diseases
    ('L', 'genetic'),         # 1.7% - Antineoplastic for genetic
    ('L', 'other'),           # 3.4% - Antineoplastic for other
    # h318: Antibiotic FILTER for non-infectious diseases
    ('J', 'hematological'),   # 0.0% - Antibiotics for blood disorders
    ('J', 'gastrointestinal'),# 10.1% - h415: valid for H. pylori ulcers, peritonitis
    ('J', 'metabolic'),       # 5.9% - Antibiotics for metabolic diseases
    ('J', 'immune'),          # 0.0% - Antibiotics for immune disorders
    ('J', 'rare_genetic'),    # 0.0% - Antibiotics for rare genetic diseases
    # h319: Batch 2
    # h415 REMOVED: ('C', 'neurological') - 14.3% (droxidopa for Parkinson's, lidocaine for neuropathy)
    ('L', 'infectious'),      # 1.9% - Antineoplastic for infectious
    # h415 REMOVED: ('M', 'other') - 58.3% (dantrolene for malignant hyperthermia, NSAIDs for pain)
    ('N', 'cardiovascular'),  # 7.0% - Nervous system for cardiovascular
    ('L', 'storage'),         # 0.0% - Antineoplastic for storage diseases
    ('C', 'cancer'),          # 9.1% - Cardiovascular for cancer
    ('P', 'other'),           # 0.0% - Antiparasitic for other
    # h415 REMOVED: ('H', 'cancer') - 10.8% (levothyroxine for thyroid cancer, lanreotide for NETs)
    # h415 REMOVED: ('L', 'neurological') - 14.8% (rituximab for NMOSD, celecoxib for migraine)
    # h415 REMOVED: ('A', 'renal') - 18.7% (corticosteroids ARE first-line for nephrotic syndrome)
    # h415 REMOVED: ('L', 'endocrine') - 20.0% (GnRH analogs for precocious puberty are standard)
    # h415 REMOVED: ('B', 'renal') - 23.1% (EPO IS standard treatment for CKD anemia)
    ('D', 'gastrointestinal'),# 9.5% - Dermatological for GI
    ('R', 'cancer'),          # 0.0% - Respiratory for cancer
    ('L', 'cardiovascular'),  # 6.1% - Antineoplastic for cardiovascular
    # h415 REMOVED: ('V', 'other') - 30.0% (antidotes for poisoning are correct treatments)
    ('C', 'musculoskeletal'), # 7.1% - Cardiovascular for musculoskeletal
    ('A', 'rare_genetic'),    # 0.0% - Alimentary for rare genetic
    ('N', 'gastrointestinal'),# 5.8% - Nervous system for GI
    ('R', 'autoimmune'),      # 6.5% - Respiratory for autoimmune
    ('N', 'metabolic'),       # 12.5% - Nervous system for metabolic
    ('N', 'rare_genetic'),    # 0.0% - Nervous system for rare genetic
}


def extract_cancer_types(disease_name: str) -> Set[str]:
    """
    h274: Extract cancer types from a disease name.
    h562: Fixed word boundary matching for short abbreviations (ALL, CLL, AML, CML, SCLC)
    to prevent false matches like "small" → "all", "fallopian" → "all".

    Returns set of cancer types (e.g., {'lymphoma', 'leukemia'}).
    """
    import re
    disease_lower = disease_name.lower()
    cancer_types = set()

    for cancer_type, keywords in CANCER_TYPE_KEYWORDS.items():
        for kw in keywords:
            if len(kw) <= 4:
                # Short abbreviations need word boundary matching
                if re.search(r'\b' + re.escape(kw) + r'\b', disease_lower):
                    cancer_types.add(cancer_type)
                    break
            else:
                if kw in disease_lower:
                    cancer_types.add(cancer_type)
                    break

    return cancer_types


class DrugRepurposingPredictor:
    """
    Production drug repurposing predictor using kNN collaborative filtering.

    Based on validated research:
    - h39: kNN with k=20 achieves 37.04% R@30 (best method)
    - h135: Tiered confidence with 9.1x precision separation
    - h136: Category-specific filters rescue Tier 2/3 categories
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize the predictor by loading required data."""
        if data_dir is None:
            data_dir = Path(__file__).parent.parent

        self.data_dir = data_dir
        self.reference_dir = data_dir / "data" / "reference"
        self.embeddings_dir = data_dir / "data" / "embeddings"

        self._load_data()

    def _load_data(self) -> None:
        """Load all required data files."""
        # Load Node2Vec embeddings (h179: use NPY for 110x speedup)
        entities_path = self.embeddings_dir / "node2vec_256_entities.npy"
        embeddings_path = self.embeddings_dir / "node2vec_256_embeddings.npy"

        if entities_path.exists() and embeddings_path.exists():
            # Fast path: load from NPY
            entities = np.load(entities_path, allow_pickle=True)
            embeddings_arr = np.load(embeddings_path)
            self.embeddings: Dict[str, np.ndarray] = {
                f"drkg:{entity}": embeddings_arr[i]
                for i, entity in enumerate(entities)
            }
        else:
            # Fallback: load from CSV (slower)
            csv_path = self.embeddings_dir / "node2vec_256_named.csv"
            df = pd.read_csv(csv_path)
            dim_cols = [c for c in df.columns if c.startswith("dim_")]
            self.embeddings = {}
            for _, row in df.iterrows():
                entity = f"drkg:{row['entity']}"
                self.embeddings[entity] = row[dim_cols].values.astype(np.float32)

        # Load DrugBank lookup
        with open(self.reference_dir / "drugbank_lookup.json") as f:
            id_to_name = json.load(f)
        self.name_to_drug_id = {
            name.lower(): f"drkg:Compound::{db_id}"
            for db_id, name in id_to_name.items()
        }
        self.drug_id_to_name = {
            f"drkg:Compound::{db_id}": name
            for db_id, name in id_to_name.items()
        }

        # h686: Add drug name aliases for Every Cure → DrugBank name mismatches.
        # EC uses different naming conventions (INN vs brand, salt forms, spelling
        # variants) that prevent matching to DrugBank entries. These aliases
        # expand internal GT by ~12% with legitimate drug-disease pairs.
        for ec_name, db_name in _DRUG_NAME_ALIASES.items():
            db_name_lower = db_name.lower()
            if db_name_lower in self.name_to_drug_id and ec_name not in self.name_to_drug_id:
                self.name_to_drug_id[ec_name] = self.name_to_drug_id[db_name_lower]

        # h686: Salt form suffix stripping — EC often uses full salt names
        # (e.g., "caspofungin acetate") while DrugBank uses base names
        # (e.g., "caspofungin"). Strip common suffixes to find matches.
        _SALT_SUFFIXES = [
            ' hydrochloride', ' hcl', ' citrate', ' acetate', ' sodium',
            ' sulfate', ' phosphate', ' maleate', ' fumarate', ' besylate',
            ' tartrate', ' bromide', ' mesylate', ' nitrate', ' succinate',
            ' potassium', ' calcium', ' magnesium', ' monohydrate',
            ' dihydrate', ' anhydrous, (e)-', ' anhydrous',
        ]
        # Build a copy of current unmapped EC drug names for suffix check
        # (done during GT loading, not here — but we need the lookup ready)
        self._salt_suffixes = _SALT_SUFFIXES

        # Load MESH mappings
        mesh_path = self.reference_dir / "mesh_mappings_from_agents.json"
        self.mesh_mappings: Dict[str, str] = {}
        if mesh_path.exists():
            with open(mesh_path) as f:
                mesh_data = json.load(f)
            for batch_data in mesh_data.values():
                if isinstance(batch_data, dict):
                    for disease_name, mesh_id in batch_data.items():
                        if mesh_id:
                            mesh_str = str(mesh_id)
                            if mesh_str.startswith("D") or mesh_str.startswith("C"):
                                self.mesh_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_str}"

        # Load ground truth (for training drug frequencies)
        self._load_ground_truth()

        # Load drug targets
        self.drug_targets: Dict[str, Set[str]] = {}
        targets_path = self.reference_dir / "drug_targets.json"
        if targets_path.exists():
            with open(targets_path) as f:
                drug_targets = json.load(f)
            self.drug_targets = {
                f"drkg:Compound::{k}": set(v)
                for k, v in drug_targets.items()
            }

        # Load disease genes
        self.disease_genes: Dict[str, Set[str]] = {}
        genes_path = self.reference_dir / "disease_genes.json"
        if genes_path.exists():
            with open(genes_path) as f:
                disease_genes = json.load(f)
            for k, v in disease_genes.items():
                gene_set = set(v)
                self.disease_genes[k] = gene_set
                if k.startswith('MESH:'):
                    self.disease_genes[f"drkg:Disease::{k}"] = gene_set

        # Build disease lists for kNN
        self._build_knn_index()

        # h274: Build drug→cancer type GT mapping
        self._build_cancer_type_mapping()

        # h273: Build drug→disease group hierarchy mapping
        self._build_disease_hierarchy_mapping()

        # h271: Build domain-isolated drug mapping
        self._build_domain_isolation_mapping()

        # h405/h439: Load TransE model for consilience scoring
        self._load_transe_model()

    def _load_transe_model(self) -> None:
        """Load TransE model for consilience scoring (h405/h439).

        TransE agreement is a strong holdout-validated signal:
        MEDIUM + TransE top-30 = 34.7% holdout (+13.6pp over MEDIUM avg).
        """
        transe_path = self.data_dir / "models" / "transe.pt"
        self.transe_entity_emb: Optional[np.ndarray] = None
        self.transe_entity2id: Optional[Dict[str, int]] = None
        self.transe_treat_vec: Optional[np.ndarray] = None

        if not transe_path.exists():
            return

        try:
            data = torch.load(transe_path, map_location="cpu", weights_only=False)
            self.transe_entity_emb = data["model_state_dict"][
                "entity_embeddings.weight"
            ].numpy()
            self.transe_entity2id = data["entity2id"]
            relation2id = data["relation2id"]
            treat_rel_id = relation2id.get("DRUGBANK::treats::Compound:Disease")
            if treat_rel_id is not None:
                self.transe_treat_vec = data["model_state_dict"][
                    "relation_embeddings.weight"
                ].numpy()[treat_rel_id]
        except Exception:
            # TransE loading is optional; continue without it
            self.transe_entity_emb = None
            self.transe_entity2id = None
            self.transe_treat_vec = None

    def _get_transe_top_n(
        self, disease_id: str, candidate_drugs: Set[str], n: int = 30
    ) -> Set[str]:
        """Get top-N drugs by TransE scoring for a disease (h405/h439).

        TransE score: -||drug_emb + treat_vec - disease_emb||
        Higher (less negative) is better.
        """
        if (
            self.transe_entity_emb is None
            or self.transe_entity2id is None
            or self.transe_treat_vec is None
        ):
            return set()

        if disease_id not in self.transe_entity2id:
            return set()

        disease_emb = self.transe_entity_emb[self.transe_entity2id[disease_id]]
        scores: List[Tuple[str, float]] = []

        for drug_id in candidate_drugs:
            if drug_id in self.transe_entity2id:
                drug_emb = self.transe_entity_emb[self.transe_entity2id[drug_id]]
                score = -float(
                    np.linalg.norm(drug_emb + self.transe_treat_vec - disease_emb)
                )
                scores.append((drug_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return {drug_id for drug_id, _ in scores[:n]}

    def _get_cache_key(self) -> str:
        """Generate a cache key based on source file modification times and content hash."""
        # h176: Cache invalidation based on source files
        source_files = [
            self.reference_dir / "everycure" / "indicationList.xlsx",
            self.reference_dir / "mesh_mappings_from_agents.json",
            self.reference_dir / "mondo_to_mesh.json",
            self.reference_dir / "drugbank_lookup.json",
            self.data_dir / "src" / "disease_name_matcher.py",  # Include matcher code
        ]

        mtimes = []
        for f in source_files:
            if f.exists():
                mtimes.append(f"{f.name}:{os.path.getmtime(f):.0f}")

        return hashlib.md5("|".join(mtimes).encode()).hexdigest()[:16]

    def _load_ground_truth(self) -> None:
        """Load ground truth for training drug frequencies.

        h176: Uses caching to speed up initialization from ~210s to <10s.
        Cache is invalidated when source files change.
        """
        cache_dir = self.data_dir / "data" / "cache"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / "ground_truth_cache.json"

        # Check if cache is valid
        current_key = self._get_cache_key()
        cache_valid = False

        if cache_path.exists():
            try:
                with open(cache_path) as f:
                    cached_data = json.load(f)
                if cached_data.get("cache_key") == current_key:
                    # Cache hit - load from cache
                    self.ground_truth = {
                        k: set(v) for k, v in cached_data["ground_truth"].items()
                    }
                    self.disease_names = cached_data["disease_names"]
                    # h669: Apply false GT removal even on cached data
                    self._remove_false_gt_pairs()
                    cache_valid = True
            except (json.JSONDecodeError, KeyError):
                pass  # Invalid cache, regenerate

        if cache_valid:
            return  # Fast path complete

        # Cache miss - regenerate (slow path, ~200s)
        sys.path.insert(0, str(self.data_dir / "src"))
        from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

        df = pd.read_excel(self.reference_dir / "everycure" / "indicationList.xlsx")
        fuzzy_mappings = load_mesh_mappings()
        matcher = DiseaseMatcher(fuzzy_mappings)

        self.ground_truth: Dict[str, Set[str]] = defaultdict(set)
        self.disease_names: Dict[str, str] = {}

        for _, row in df.iterrows():
            disease = str(row.get("disease name", "")).strip()
            drug = str(row.get("final normalized drug label", "")).strip()
            if not disease or not drug:
                continue

            disease_id = matcher.get_mesh_id(disease)
            if not disease_id:
                disease_id = self.mesh_mappings.get(disease.lower())
            if not disease_id:
                continue

            self.disease_names[disease_id] = disease
            drug_lower = drug.lower()
            drug_id = self.name_to_drug_id.get(drug_lower)
            # h686: Salt form suffix stripping fallback
            if not drug_id:
                for suffix in self._salt_suffixes:
                    base = drug_lower.replace(suffix, '').strip()
                    if base != drug_lower and base in self.name_to_drug_id:
                        drug_id = self.name_to_drug_id[base]
                        break
            if drug_id:
                self.ground_truth[disease_id].add(drug_id)

        self.ground_truth = dict(self.ground_truth)

        # h669: Remove false GT entries (NLP extraction errors in indicationList.xlsx)
        self._remove_false_gt_pairs()

        # Save to cache
        cache_data = {
            "cache_key": current_key,
            "ground_truth": {k: list(v) for k, v in self.ground_truth.items()},
            "disease_names": self.disease_names,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

    def _remove_false_gt_pairs(self) -> None:
        """h669: Remove known false GT entries from ground truth.

        These are NLP extraction errors where FDA labels mention a disease as a
        differential diagnosis/exclusion criterion, not as an indication.
        """
        # Build reverse lookup: drug_name_lower → drug_id
        name_to_id = {}
        for drug_id, name in self.drug_id_to_name.items():
            name_to_id[name.lower()] = drug_id

        removed = 0
        for drug_name_lower, false_diseases in FALSE_GT_PAIRS.items():
            drug_id = name_to_id.get(drug_name_lower)
            if not drug_id:
                continue
            for disease_id, drug_ids in self.ground_truth.items():
                disease_name = self.disease_names.get(disease_id, '').lower()
                if disease_name in false_diseases and drug_id in drug_ids:
                    drug_ids.discard(drug_id)
                    removed += 1

        # h680: Remove non-therapeutic compounds from internal GT entirely.
        # These are diagnostic imaging agents, not treatments. Their presence
        # in DRKG treatment edges is from co-occurrence with diseases in clinical
        # contexts (e.g., FDG PET scans for cancer staging).
        # h685: Extended to include Tc-99m sestamibi, Ioflupane I-123,
        # Florbetaben, Tc-99m sulfur colloid (all purely diagnostic agents).
        _NON_THERAPEUTIC_GT_DRUGS = {
            'fludeoxyglucose (18f)',  # DB09502 — PET imaging tracer, not a treatment
            'fludeoxyglucose f-18',  # Alternate name
            'technetium tc-99m sestamibi',  # DB09161 — myocardial perfusion imaging
            'ioflupane i 123',  # DB08824 — DaTscan dopamine transporter imaging
            'florbetaben (18f)',  # DB09148 — amyloid PET imaging
            'technetium tc-99m sulfur colloid',  # DB09397 — sentinel lymph node mapping
            # h689: Additional diagnostic agents found in census
            'flortaucipir f-18',  # DB14914 — tau PET imaging (Tauvid)
            'fluciclovine (18f)',  # DB13146 — amino acid PET imaging (Axumin)
            'pentagastrin',  # DB00183 — gastric acid stimulation test only
        }
        for drug_name_lower in _NON_THERAPEUTIC_GT_DRUGS:
            drug_id = name_to_id.get(drug_name_lower)
            if not drug_id:
                continue
            for disease_id, drug_ids in self.ground_truth.items():
                if drug_id in drug_ids:
                    drug_ids.discard(drug_id)
                    removed += 1

        # h685: Iobenguane (DB06704) is DUAL-USE — therapeutic for neuroblastoma
        # and paraganglioma (I-131 MIBG therapy, FDA-approved 2018 as Azedra),
        # but DIAGNOSTIC ONLY for Parkinson's and heart failure (cardiac MIBG
        # scintigraphy). Remove only from non-therapeutic disease associations.
        _IOBENGUANE_DIAGNOSTIC_ONLY = {
            'parkinson', 'heart failure', 'cardiomyopath',
        }
        iobenguane_id = name_to_id.get('iobenguane')
        if iobenguane_id:
            for disease_id, drug_ids in self.ground_truth.items():
                if iobenguane_id in drug_ids:
                    dname = self.disease_names.get(disease_id, '').lower()
                    if any(kw in dname for kw in _IOBENGUANE_DIAGNOSTIC_ONLY):
                        drug_ids.discard(iobenguane_id)
                        removed += 1

        # h685: Remove specific false GT entries found in single-drug disease audit.
        # These are DRKG co-occurrence artifacts, not genuine treatments.
        _FALSE_GT_SPECIFIC = {
            # Diazoxide is for hyperinsulinemic hypoglycemia, not carcinoma
            'diazoxide': {'carcinoma'},
            # Insulin has no role in protein C deficiency (coagulation disorder)
            'insulin human': {'congenital protein c deficiency'},
            # Isoniazid treats TB, not these diseases (co-occurrence artifacts)
            'isoniazid': {
                'chronic peptic ulcer disease',  # INH can CAUSE GI issues
                'silicosis',  # INH is for TB prophylaxis in silicosis, not silicosis itself
                'malabsorption syndrome',  # False co-occurrence
            },
            # h690: Pyrazinamide treats TB, not immunodeficiency
            # (co-occurrence from HIV/TB co-infection treatment context)
            'pyrazinamide': {
                'immunodeficiency',
            },
            # Chlorhexidine is a topical antiseptic, not for coagulation disorders
            'chlorhexidine': {
                'purpura fulminans',  # Requires heparin/protein C/plasma
                'thrombophilia',  # Requires anticoagulants
            },
        }
        for drug_name_lower, false_diseases in _FALSE_GT_SPECIFIC.items():
            drug_id = name_to_id.get(drug_name_lower)
            if not drug_id:
                continue
            for disease_id, drug_ids in self.ground_truth.items():
                if drug_id in drug_ids:
                    dname = self.disease_names.get(disease_id, '').lower()
                    if dname in false_diseases:
                        drug_ids.discard(drug_id)
                        removed += 1

        # h677/h680: Allowlist-based cleanup for lidocaine and bupivacaine
        # These drugs have 78/61 false GT entries from combo product NLP mismatch
        # (corticosteroid indication text assigned to LA component).
        # Instead of listing all false diseases, we keep ONLY legitimate indications.
        _LA_ALLOWLIST = {
            'lidocaine': {
                'pain', 'arrhythmi', 'ventricular', 'pruritus', 'itch', 'neuropath',
                'neuralgia', 'tachycardia', 'hemorrhoid', 'herpes zoster', 'herpes simplex',
                'postherpetic', 'fibromyalgia', 'cardiac arrest', 'dermatit', 'eczema',
                'psoriasis', 'urticaria', 'lichen', 'keratosis', 'seborrheic', 'pemphig',
                'erythema multiforme', 'bullous', 'alopecia', 'vitiligo', 'acne',
                'depression', 'depressive', 'migraine', 'headache',
                'epilepsy', 'seizure', 'status epilepticus',
                'stomatitis', 'aphthous', 'mouth', 'dental', 'toothache',
                'cystitis', 'interstitial cystitis', 'hidradenitis',
                'osteoarthritis', 'tenosynovitis', 'torsades', 'myocardial infarction',
            },
            'bupivacaine': {
                'pain', 'anesthesia', 'labor', 'surgical', 'dental', 'epidural', 'spinal',
                'postoperative', 'nerve block', 'obstetric', 'caesarean', 'cesarean',
                'herpes zoster', 'tendinitis',
            },
        }
        for drug_name_lower, legit_keywords in _LA_ALLOWLIST.items():
            drug_id = name_to_id.get(drug_name_lower)
            if not drug_id:
                continue
            for disease_id, drug_ids in self.ground_truth.items():
                if drug_id in drug_ids:
                    dname = self.disease_names.get(disease_id, '').lower()
                    if dname and not any(kw in dname for kw in legit_keywords):
                        drug_ids.discard(drug_id)
                        removed += 1

        if removed > 0:
            # Remove empty disease entries
            self.ground_truth = {k: v for k, v in self.ground_truth.items() if v}

    def _build_knn_index(self) -> None:
        """Build index for kNN lookups."""
        # Training diseases (all diseases in ground truth)
        self.train_diseases = [d for d in self.ground_truth if d in self.embeddings]
        self.train_embeddings = np.array(
            [self.embeddings[d] for d in self.train_diseases],
            dtype=np.float32
        )

        # h170: Pre-compute categories for training diseases (for selective boosting)
        self.train_disease_categories: Dict[str, str] = {}
        for d in self.train_diseases:
            name = self.disease_names.get(d, d)
            self.train_disease_categories[d] = self.categorize_disease(name)

        # Drug training frequency
        self.drug_train_freq: Dict[str, int] = defaultdict(int)
        for _, drugs in self.ground_truth.items():
            for drug_id in drugs:
                self.drug_train_freq[drug_id] += 1

        # h405/h439: Pre-compute set of all GT drugs with embeddings for TransE scoring
        self.all_gt_drugs_with_embeddings: Set[str] = {
            d for drugs in self.ground_truth.values() for d in drugs
            if d in self.embeddings
        }

        # h280/h281: Reverse mapping - drug_id → set of disease names (for complication check)
        self.drug_to_diseases: Dict[str, Set[str]] = defaultdict(set)
        for disease_id, drug_ids in self.ground_truth.items():
            disease_name = self.disease_names.get(disease_id, disease_id)
            for drug_id in drug_ids:
                self.drug_to_diseases[drug_id].add(disease_name.lower())
        self.drug_to_diseases = dict(self.drug_to_diseases)

    def _build_cancer_type_mapping(self) -> None:
        """
        h274: Build mapping of drug_id → set of cancer types in GT.

        From h270 analysis:
        - If drug treats same cancer type: 100% precision (subtype refinement)
        - If drug treats different cancer type: 30.6% precision (cross-repurposing)
        - If drug has no cancer GT: 0% precision (FILTER these for cancer)
        """
        self.drug_cancer_types: Dict[str, Set[str]] = defaultdict(set)

        for disease_id, drug_ids in self.ground_truth.items():
            disease_name = self.disease_names.get(disease_id, disease_id)
            cancer_types = extract_cancer_types(disease_name)

            if cancer_types:  # Only process cancer diseases
                for drug_id in drug_ids:
                    self.drug_cancer_types[drug_id].update(cancer_types)

        self.drug_cancer_types = dict(self.drug_cancer_types)

    def _check_cancer_type_match(
        self, drug_id: str, disease_name: str
    ) -> Tuple[bool, bool, Set[str]]:
        """
        h274: Check if drug's cancer GT matches the target disease's cancer type.

        Returns:
            (has_cancer_gt, same_type_match, overlapping_types)
            - has_cancer_gt: Drug has ANY cancer indication in GT
            - same_type_match: Drug treats the SAME cancer type (e.g., lymphoma → DLBCL)
            - overlapping_types: Set of matching cancer types
        """
        drug_cancer_types = self.drug_cancer_types.get(drug_id, set())
        has_cancer_gt = len(drug_cancer_types) > 0

        if not has_cancer_gt:
            return False, False, set()

        disease_cancer_types = extract_cancer_types(disease_name)
        overlapping_types = drug_cancer_types & disease_cancer_types
        same_type_match = len(overlapping_types) > 0

        return has_cancer_gt, same_type_match, overlapping_types

    def _build_disease_hierarchy_mapping(self) -> None:
        """
        h273: Build mapping of drug_id → set of (category, disease_group) pairs.

        From h273 analysis:
        - Hierarchy matching increases precision 2.9x overall (8.5% → 24.7%)
        - Metabolic: +35.2 pp, Autoimmune: +18.8 pp, Infectious: +16.1 pp
        """
        self.drug_disease_groups: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)

        for disease_id, drug_ids in self.ground_truth.items():
            disease_name = self.disease_names.get(disease_id, disease_id)
            disease_lower = disease_name.lower()

            # Check each category's disease groups
            for category, groups in DISEASE_HIERARCHY_GROUPS.items():
                for group_name, variants in groups.items():
                    # h410: Check exclusion list before matching
                    exclusions = HIERARCHY_EXCLUSIONS.get((category, group_name), [])
                    if any(excl in disease_lower for excl in exclusions):
                        continue
                    if any(variant in disease_lower or disease_lower in variant
                           for variant in variants):
                        for drug_id in drug_ids:
                            self.drug_disease_groups[drug_id].add((category, group_name))

        self.drug_disease_groups = dict(self.drug_disease_groups)

    def _check_disease_hierarchy_match(
        self, drug_id: str, disease_name: str, category: str
    ) -> Tuple[bool, bool, Optional[str]]:
        """
        h273: Check if drug's GT matches the target disease through hierarchy.

        Returns:
            (has_category_gt, same_group_match, matching_group)
            - has_category_gt: Drug has ANY GT in this category's groups
            - same_group_match: Drug treats the SAME disease group (e.g., psoriasis → plaque psoriasis)
            - matching_group: Name of the matching disease group
        """
        drug_groups = self.drug_disease_groups.get(drug_id, set())
        # Filter to this category's groups only
        category_groups = {g for (c, g) in drug_groups if c == category}
        has_category_gt = len(category_groups) > 0

        if not has_category_gt or category not in DISEASE_HIERARCHY_GROUPS:
            return False, False, None

        # Find what disease group the prediction disease belongs to
        disease_lower = disease_name.lower()
        pred_disease_group = None
        for group_name, variants in DISEASE_HIERARCHY_GROUPS[category].items():
            # h410: Check exclusion list before matching
            exclusions = HIERARCHY_EXCLUSIONS.get((category, group_name), [])
            if any(excl in disease_lower for excl in exclusions):
                continue
            if any(variant in disease_lower or disease_lower in variant
                   for variant in variants):
                pred_disease_group = group_name
                break

        if pred_disease_group is None:
            return has_category_gt, False, None

        # Check if drug has GT in the same disease group
        same_group_match = pred_disease_group in category_groups

        return has_category_gt, same_group_match, pred_disease_group if same_group_match else None

    def _build_domain_isolation_mapping(self) -> None:
        """
        h271: Build mapping of domain-isolated drugs and their single category.

        From h271 analysis:
        - 1,312 drugs only have GT in a single disease category
        - Cross-domain predictions for these drugs have 0% precision
        - Same-domain predictions have 17.2% precision
        - FILTER cross-domain predictions involving domain-isolated drugs
        """
        # Map drug_id → set of categories in GT
        drug_categories: Dict[str, Set[str]] = defaultdict(set)

        for disease_id, drug_ids in self.ground_truth.items():
            disease_name = self.disease_names.get(disease_id, disease_id)
            category = self.categorize_disease(disease_name)
            for drug_id in drug_ids:
                drug_categories[drug_id].add(category)

        # Find domain-isolated drugs (only 1 category)
        # Store as dict: drug_id → their single category
        self.domain_isolated_drugs: Dict[str, str] = {
            drug_id: list(cats)[0]
            for drug_id, cats in drug_categories.items()
            if len(cats) == 1
        }

    def _is_cross_domain_isolated(self, drug_id: str, target_category: str) -> bool:
        """
        h271: Check if drug is domain-isolated and predicting for wrong category.

        Returns True if this is a cross-domain prediction for an isolated drug
        (0% precision - should be filtered).
        """
        if drug_id not in self.domain_isolated_drugs:
            return False  # Not isolated, allow prediction

        drug_category = self.domain_isolated_drugs[drug_id]
        return drug_category != target_category

    def _is_broad_class_isolated(
        self,
        drug_name: str,
        disease_name: str,
        all_predictions_for_disease: Optional[Set[str]] = None,
    ) -> bool:
        """
        h326: Check if drug is from broad therapeutic class but predicted alone.

        When a drug from a broad class (anesthetics, steroids, TNFi, NSAIDs) is
        predicted alone (no classmates), precision is only 1.9% (vs 12.7% with classmates).

        Args:
            drug_name: The drug being predicted
            disease_name: The disease being predicted for (used to get co-predictions)
            all_predictions_for_disease: Optional set of all drug names predicted for this disease

        Returns:
            True if drug is from broad class AND has no classmates predicted (should demote/filter)
        """
        drug_lower = drug_name.lower()

        # Find which broad class this drug belongs to (if any)
        drug_class = None
        for class_name, class_drugs in BROAD_THERAPEUTIC_CLASSES.items():
            if any(cd in drug_lower for cd in class_drugs):
                drug_class = class_name
                break

        if drug_class is None:
            return False  # Not in a broad class, allow prediction

        # If we don't have co-predictions, we can't check isolation
        # This happens during single-drug calls; will be checked at batch level
        if all_predictions_for_disease is None:
            return False

        # Check if any classmates are also predicted for this disease
        class_drugs = BROAD_THERAPEUTIC_CLASSES[drug_class]
        for other_drug in all_predictions_for_disease:
            other_lower = other_drug.lower()
            if other_lower == drug_lower:
                continue  # Skip self
            if any(cd in other_lower for cd in class_drugs):
                return False  # Found a classmate, not isolated

        # Drug is from broad class but isolated (no classmates) → bad signal
        return True

    def _is_base_to_complication(self, drug_id: str, predicted_disease: str) -> bool:
        """
        h280/h281: Check if prediction is a base→complication pattern (0% precision).

        Returns True if:
        1. Drug treats a base disease (e.g., diabetes) in GT
        2. Prediction is for a complication of that base disease (e.g., diabetic nephropathy)

        These predictions have 0% precision and should be filtered.
        """
        if not drug_id or not predicted_disease:
            return False

        drug_gt = self.drug_to_diseases.get(drug_id, set())
        if not drug_gt:
            return False

        pred_lower = predicted_disease.lower()

        # Check each base disease the drug treats
        for base_disease, complications in BASE_TO_COMPLICATIONS.items():
            base_lower = base_disease.lower()
            # Does drug treat this base disease?
            drug_treats_base = any(base_lower in gt or gt == base_lower
                                   for gt in drug_gt)
            if drug_treats_base:
                # Is prediction for one of its complications?
                # Use strict matching: prediction must BE the complication
                # e.g., "diabetic nephropathy" matches but "chronic kidney disease"
                # doesn't match "anemia of chronic kidney disease"
                for comp in complications:
                    comp_lower = comp.lower()
                    # Match if prediction equals complication exactly, or starts with it
                    if pred_lower == comp_lower or pred_lower.startswith(comp_lower + ' '):
                        return True

        return False

    def _is_comp_to_base(self, drug_id: str, predicted_disease: str) -> Tuple[bool, float, bool]:
        """
        h284/h291: Check if prediction is a comp→base pattern and get transferability.

        Returns: (is_comp_to_base, transferability_score, is_statin_cv)

        HIGH transferability (≥50%): comp→base predictions work well (62.5% precision)
        h296: For CV events, statins achieve 100% precision, non-statins 0%
        """
        if not drug_id or not predicted_disease:
            return False, 0.0, False

        drug_gt = self.drug_to_diseases.get(drug_id, set())
        if not drug_gt:
            return False, 0.0, False

        pred_lower = predicted_disease.lower()
        drug_name = self._get_drug_name(drug_id).lower()

        # Check if drug treats a complication and prediction is for a base
        # h669: Exclusions for base disease matching — diseases that contain
        # the base term but are medically unrelated
        COMP_TO_BASE_EXCLUSIONS = {
            'diabetes': ['diabetes insipidus', 'central diabetes insipidus',
                         'nephrogenic diabetes insipidus'],
        }
        for comp, transferability in COMPLICATION_TRANSFERABILITY.items():
            comp_lower = comp.lower()
            # Does drug treat this complication?
            drug_treats_comp = any(comp_lower in gt.lower() for gt in drug_gt)
            if drug_treats_comp:
                # Is prediction for the base disease?
                # Check if prediction is a base disease for this complication
                for base_disease, complications in BASE_TO_COMPLICATIONS.items():
                    if comp_lower in [c.lower() for c in complications]:
                        # comp is a complication of base_disease
                        base_lower = base_disease.lower()
                        # h669: Check exclusions before matching
                        exclusions = COMP_TO_BASE_EXCLUSIONS.get(base_lower, [])
                        if any(excl in pred_lower for excl in exclusions):
                            continue
                        if base_lower in pred_lower or pred_lower in base_lower:
                            # This is a comp→base prediction
                            # h296: Check statin rule for CV events
                            is_statin = any(s in drug_name for s in STATIN_NAMES)
                            is_cv_event = any(cv in comp_lower for cv in CV_EVENTS_FOR_STATIN_RULE)
                            is_statin_cv = is_statin and is_cv_event
                            return True, transferability, is_statin_cv

        return False, 0.0, False

    def _get_drug_name(self, drug_id: str) -> str:
        """Get drug name from ID using drug_id_to_name lookup."""
        return self.drug_id_to_name.get(drug_id, drug_id)

    @staticmethod
    def _is_mechanism_specific_disease(disease_name: str) -> bool:
        """
        h297: Check if disease is mechanism-specific (kNN will fail).

        Mechanism-specific diseases have drugs that don't transfer from
        similar diseases - only 6.6% of their GT drugs are repurposable.

        Returns True if disease is in MECHANISM_SPECIFIC_DISEASES list.
        """
        disease_lower = disease_name.lower()
        return disease_lower in MECHANISM_SPECIFIC_DISEASES

    @staticmethod
    def _is_cancer_drug(drug_name: str) -> bool:
        """
        h346: Check if drug is a cancer-only drug.

        These drugs have 0% precision for non-cancer predictions (115 preds, 0 GT hits).
        They have NO approved non-cancer uses and no plausible non-cancer mechanism.
        """
        return drug_name.lower() in CANCER_ONLY_DRUGS

    @staticmethod
    def _is_cancer_disease(disease_name: str) -> bool:
        """Check if disease is a cancer-related condition."""
        cancer_keywords = ['cancer', 'carcinoma', 'tumor', 'melanoma', 'leukemia',
                          'lymphoma', 'neoplasm', 'neurofibroma', 'glioma', 'sarcoma',
                          'myeloma', 'blastoma', 'adenocarcinoma']
        return any(kw in disease_name.lower() for kw in cancer_keywords)

    @staticmethod
    def _is_cancer_only_drug_non_cancer(drug_name: str, disease_name: str) -> bool:
        """
        h346: Check if this is a cancer-only drug predicted for non-cancer disease.

        Cancer-only drugs have 0% precision for non-cancer predictions because:
        1. They target cancer-specific pathways (BRAF, PD-1, BCL2, PARP, etc.)
        2. All GT indications are cancer
        3. No plausible non-cancer mechanism exists

        Returns True if drug is cancer-only AND disease is NOT cancer.
        """
        if not DrugRepurposingPredictor._is_cancer_drug(drug_name):
            return False

        return not DrugRepurposingPredictor._is_cancer_disease(disease_name)

    @staticmethod
    def _is_mek_inhibitor_non_cancer(drug_name: str, disease_name: str) -> bool:
        """
        h340: Check if this is a MEK inhibitor predicted for non-cancer disease.
        Note: This is now redundant with h346 but kept for explicit documentation.

        MEK inhibitors have 0% precision for non-cancer predictions because:
        1. They target the RAS/RAF/MEK/ERK pathway specific to cancer
        2. All GT indications are cancer (100%)
        3. No plausible non-cancer mechanism exists

        Returns True if drug is MEK inhibitor AND disease is NOT cancer.
        """
        # h346 now covers MEK inhibitors via CANCER_ONLY_DRUGS
        return DrugRepurposingPredictor._is_cancer_only_drug_non_cancer(drug_name, disease_name)

    @staticmethod
    def _is_highly_repurposable_disease(disease_name: str) -> bool:
        """
        h297: Check if disease is highly repurposable (kNN works well).

        Highly repurposable diseases have drugs that are used across many
        diseases - 87.9% of their GT drugs are repurposable.

        Returns True if disease is in HIGHLY_REPURPOSABLE_DISEASES list.
        """
        disease_lower = disease_name.lower()
        return disease_lower in HIGHLY_REPURPOSABLE_DISEASES

    @staticmethod
    def _is_complication_non_validated_class(drug_name: str, disease_name: str) -> bool:
        """
        h353: Check if this is a complication disease with non-validated drug class.

        For specific complication diseases (nephropathy, retinopathy, cardiomyopathy,
        neuropathy), only certain validated drug classes have non-zero precision.
        Non-validated classes have 0% precision across 214 predictions.

        Results:
            - Nephrotic syndrome: Validated 69.2% vs Non-validated 0.0%
            - Retinopathy: Validated 33.3% vs Non-validated 0.0%
            - Cardiomyopathy: Validated 12.5% vs Non-validated 0.0%
            - Neuropathy: Validated 0.0% vs Non-validated 0.0%

        Returns True if disease is a complication AND drug is NOT in validated class.
        """
        disease_lower = disease_name.lower()
        drug_lower = drug_name.lower()

        for complication_term, validated_drugs in COMPLICATION_VALIDATED_DRUGS.items():
            if complication_term in disease_lower:
                # Check if drug is in validated class
                for validated in validated_drugs:
                    if validated in drug_lower:
                        return False  # Drug IS validated for this complication
                return True  # Disease IS complication, drug NOT in validated class

        return False  # Not a complication disease, no filter needed

    @staticmethod
    def _is_cv_complication(disease_name: str) -> bool:
        """
        h354: Check if disease is a CV complication.

        CV complications include: heart failure, stroke, myocardial infarction,
        angina, peripheral vascular disease, cardiomyopathy.

        Returns True if disease contains any CV complication keyword.
        """
        disease_lower = disease_name.lower()
        return any(kw in disease_lower for kw in CV_COMPLICATION_KEYWORDS)

    @staticmethod
    def _is_cv_pathway_comprehensive(drug_name: str) -> bool:
        """
        h354: Check if drug is CV pathway-comprehensive.

        Pathway-comprehensive drugs have GT for BOTH a CV base disease
        (hypertension, CAD, hyperlipidemia) AND a CV complication
        (heart failure, stroke, MI).

        For CV complication predictions:
        - Pathway-comprehensive: 48.8% precision (20/41)
        - Non-pathway-comprehensive: 7.6% precision (6/79)
        - GAP: +41.2 pp (6.4x lift)

        Returns True if drug is in CV_PATHWAY_COMPREHENSIVE_DRUGS set.
        """
        drug_lower = drug_name.lower().strip()
        # Check exact match and partial match (for combo drugs)
        if drug_lower in CV_PATHWAY_COMPREHENSIVE_DRUGS:
            return True
        # Check if any comprehensive drug name is contained in this drug name
        for comp_drug in CV_PATHWAY_COMPREHENSIVE_DRUGS:
            if comp_drug in drug_lower:
                return True
        return False

    @staticmethod
    def _is_established_cv_drug(drug_name: str) -> bool:
        """
        h618: Check if drug belongs to an established cardiovascular drug class.

        CV drug classes with MEDIUM-level holdout precision (expanded GT, 5-seed):
        - Anticoagulants/antiplatelets: 32.6% ± 23.4% (n=14.2/seed)
        - CCBs: 49.7% ± 34.6% (n=3.4/seed)
        - Diuretics: 33.8% ± 32.4% (n=3.2/seed)
        - ARBs: 30.0% ± 40.0% (n=3.0/seed)
        - Statins, beta-blockers, ACE inhibitors, antiarrhythmics: high holdout but small-n

        Non-CV drugs (antibiotics, biologics, corticosteroids) in CV demotion: <3-18% holdout.
        Corticosteroid CS→CV already handled separately by h557/h520.

        Returns True if drug is a genuine cardiovascular pharmacotherapy.
        """
        drug_lower = drug_name.lower().strip()

        # Anticoagulants/antiplatelets (32.6% holdout, n=14.2/seed)
        anticoag = ['warfarin', 'heparin', 'enoxaparin', 'rivaroxaban', 'apixaban',
                    'dabigatran', 'edoxaban', 'clopidogrel', 'ticagrelor', 'prasugrel',
                    'dipyridamole', 'cilostazol', 'fondaparinux', 'dalteparin',
                    'tinzaparin', 'eptifibatide', 'vorapaxar']

        # CCBs (49.7% holdout)
        ccbs = ['amlodipine', 'nifedipine', 'diltiazem', 'verapamil', 'felodipine',
                'nicardipine', 'nimodipine', 'isradipine', 'clevidipine']

        # Diuretics (33.8% holdout)
        diuretics = ['hydrochlorothiazide', 'furosemide', 'spironolactone', 'chlorthalidone',
                     'bumetanide', 'torsemide', 'torasemide', 'amiloride', 'triamterene',
                     'indapamide', 'metolazone', 'eplerenone']

        # ARBs (30.0% holdout)
        arbs = ['losartan', 'valsartan', 'irbesartan', 'candesartan', 'telmisartan',
                'olmesartan', 'eprosartan', 'azilsartan']

        # Statins (56.7% holdout, small-n but consistently high)
        statins = ['atorvastatin', 'simvastatin', 'rosuvastatin', 'pravastatin',
                   'lovastatin', 'fluvastatin', 'pitavastatin']

        # Beta-blockers (50.0% holdout, small-n)
        betas = ['metoprolol', 'atenolol', 'propranolol', 'carvedilol', 'bisoprolol',
                 'nebivolol', 'labetalol', 'nadolol', 'sotalol', 'timolol',
                 'acebutolol', 'pindolol', 'esmolol', 'landiolol']

        # ACE inhibitors (18.7% holdout, small-n but genuine CV)
        ace_inhibitors = ['enalapril', 'lisinopril', 'ramipril', 'captopril', 'perindopril',
                          'quinapril', 'benazepril', 'fosinopril', 'trandolapril', 'moexipril']

        # Antiarrhythmics (60.0% holdout, small-n)
        antiarrhythmics = ['amiodarone', 'flecainide', 'propafenone', 'dronedarone',
                           'dofetilide', 'ibutilide', 'mexiletine', 'procainamide',
                           'quinidine', 'disopyramide']

        # Nitrates/vasodilators (40.0% holdout, very small-n)
        nitrates = ['nitroglycerin', 'isosorbide', 'hydralazine', 'minoxidil']

        # Other established CV drugs
        other_cv = ['digoxin', 'digitoxin', 'milrinone', 'ranolazine', 'ivabradine',
                    'bosentan', 'ambrisentan', 'macitentan', 'treprostinil', 'epoprostenol',
                    'iloprost', 'sildenafil', 'tadalafil',  # PAH-indicated PDE5i
                    'fenofibrate', 'gemfibrozil', 'bezafibrate',  # fibrates
                    'doxazosin', 'prazosin', 'terazosin',  # alpha blockers
                    'clonidine', 'methyldopa', 'guanfacine',  # centrally acting
                    'aliskiren', 'alirocumab', 'evolocumab',  # renin/PCSK9
                    'cholestyramine', 'colestipol', 'colesevelam',  # bile acid sequestrants
                    'niacin', 'icosapent', 'papaverine', 'adenosine',
                    'alteplase', 'tenecteplase', 'reteplase',  # thrombolytics
                    'pentoxifylline']  # peripheral vascular

        all_cv_drugs = (anticoag + ccbs + diuretics + arbs + statins + betas +
                        ace_inhibitors + antiarrhythmics + nitrates + other_cv)

        for drug in all_cv_drugs:
            if drug in drug_lower:
                return True
        return False

    @staticmethod
    def _is_immune_mediated_hematological(disease_name: str) -> bool:
        """
        h625: Check if a hematological disease is immune-mediated.

        Corticosteroids are genuine treatments for immune-mediated cytopenias
        (autoimmune destruction of blood cells) but NOT for genetic/structural
        hematological disorders (hemoglobinopathies, coagulation factor deficiencies).

        5-seed holdout (expanded GT):
        - Immune-mediated CS: 48.4% ± 28.8% (n=16.2/seed) → MEDIUM quality
        - Non-immune CS: 3.8% ± 4.0% (n=9.0/seed) → correctly LOW

        Returns True if disease is immune-mediated hematological.
        """
        dl = disease_name.lower()

        immune_keywords = [
            # Autoimmune cytopenias
            'autoimmune hemolytic', 'warm autoimmune', 'cold autoimmune',
            'immune thrombocytopeni', 'idiopathic thrombocytopeni',
            'pure red cell aplasia', 'aplastic anemia',
            'evans syndrome',
            # Immune-mediated conditions
            'heparin-induced thrombocytopeni', 'heparininduced thrombocytopeni',
            'thrombotic thrombocytopenic purpura',
            'hemolytic uremic',
            'hypereosinophilic',
            'hemolytic anemia',
            # Transplant-related
            'graft versus host', 'gvhd',
            # Acquired immune-mediated
            'acquired hemophilia',
        ]

        for kw in immune_keywords:
            if kw in dl:
                return True

        # "anemia" without genetic qualifiers is often immune-mediated/multifactorial
        if 'anemia' in dl and 'sickle' not in dl and 'thalassemia' not in dl:
            return True

        return False

    def _is_atc_coherent(self, drug_name: str, category: str) -> bool:
        """
        h309/h310: Check if drug's ATC code is coherent with disease category.

        Coherent predictions (ATC L1 matches expected codes for category)
        have 35.5% precision vs 18.7% for incoherent (+16.8 pp gap).

        Args:
            drug_name: Name of the drug
            category: Disease category (e.g., 'autoimmune', 'infectious')

        Returns:
            True if drug's ATC L1 code is in expected codes for category,
            False otherwise. Returns False if no ATC mapping exists.
        """
        expected_atc = DISEASE_CATEGORY_ATC_MAP.get(category, set())
        if not expected_atc:
            return False  # No coherence check for uncategorized

        try:
            mapper = _get_atc_mapper()
            atc_codes = mapper.get_atc_codes(drug_name)
            if not atc_codes:
                return False  # No ATC code found

            # Check if any of the drug's ATC L1 codes match expected
            for code in atc_codes:
                if code and code[0] in expected_atc:
                    return True
            return False
        except Exception:
            return False

    def _check_atc_mismatch_rules(self, drug_name: str, category: str) -> Tuple[bool, bool, Optional[float]]:
        """
        h314/h316: Check ATC mismatch-specific rules.

        Returns: (is_high_precision_mismatch, is_zero_precision_mismatch, precision)
        - is_high_precision_mismatch: True if this ATC→category pair has >10% precision
        - is_zero_precision_mismatch: True if this ATC→category pair has <3% precision
        - precision: The precision percentage if known, else None
        """
        try:
            mapper = _get_atc_mapper()
            atc_codes = mapper.get_atc_codes(drug_name)
            if not atc_codes:
                return False, False, None

            # Get primary ATC L1 code
            atc_l1 = atc_codes[0][0] if atc_codes and atc_codes[0] else None
            if not atc_l1:
                return False, False, None

            # Check high-precision mismatches
            key = (atc_l1, category)
            if key in HIGH_PRECISION_MISMATCHES:
                return True, False, HIGH_PRECISION_MISMATCHES[key]

            # Check zero-precision mismatches
            if key in ZERO_PRECISION_MISMATCHES:
                return False, True, 0.0

            return False, False, None
        except Exception:
            return False, False, None

    @staticmethod
    def categorize_disease(disease_name: str) -> str:
        """Categorize a disease by name."""
        name_lower = disease_name.lower()
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                if kw in name_lower:
                    return category
        return 'other'

    @staticmethod
    def get_category_tier(category: str) -> int:
        """Get the tier (1-3) for a disease category."""
        if category in TIER_1_CATEGORIES:
            return 1
        elif category in TIER_2_CATEGORIES:
            return 2
        else:
            return 3

    def _compute_mechanism_support(self, drug_id: str, disease_id: str) -> bool:
        """Check if drug targets overlap with disease genes."""
        drug_genes = self.drug_targets.get(drug_id, set())
        dis_genes = self.disease_genes.get(disease_id, set())
        return len(drug_genes & dis_genes) > 0

    def _get_target_overlap_count(self, drug_id: str, disease_id: str) -> int:
        """Get count of overlapping genes between drug targets and disease genes.

        h374: Used for target-based scoring in MinRank ensemble.
        """
        drug_genes = self.drug_targets.get(drug_id, set())
        dis_genes = self.disease_genes.get(disease_id, set())
        return len(drug_genes & dis_genes)

    def _get_target_scores(self, disease_id: str) -> Dict[str, int]:
        """Get target overlap scores for all drugs with targets.

        h374: Returns dict of drug_id -> overlap count for target-based ranking.
        """
        dis_genes = self.disease_genes.get(disease_id, set())
        if not dis_genes:
            return {}

        scores = {}
        for drug_id, drug_genes in self.drug_targets.items():
            overlap = len(drug_genes & dis_genes)
            if overlap > 0:
                scores[drug_id] = overlap
        return scores

    def _minrank_fusion(
        self,
        knn_scores: Dict[str, float],
        target_scores: Dict[str, int],
    ) -> Dict[str, Tuple[float, int]]:
        """Combine kNN and target scores using MinRank fusion.

        h374: For each drug, take the minimum rank from either method.
        Returns dict of drug_id -> (combined_score, min_rank) sorted by min_rank.

        The combined score is used to break ties within same min_rank.
        """
        # Rank drugs by each method (1-indexed)
        knn_sorted = sorted(knn_scores.items(), key=lambda x: -x[1])
        knn_ranks = {drug_id: rank for rank, (drug_id, _) in enumerate(knn_sorted, 1)}

        target_sorted = sorted(target_scores.items(), key=lambda x: -x[1])
        target_ranks = {drug_id: rank for rank, (drug_id, _) in enumerate(target_sorted, 1)}

        # Get all drugs from both methods
        all_drugs = set(knn_scores.keys()) | set(target_scores.keys())

        # Default rank for drugs not in a method is very high (won't be selected)
        max_rank = len(all_drugs) + 100

        # Compute min rank and combined score for each drug
        results = {}
        for drug_id in all_drugs:
            knn_rank = knn_ranks.get(drug_id, max_rank)
            tgt_rank = target_ranks.get(drug_id, max_rank)
            min_rank = min(knn_rank, tgt_rank)

            # Combined score for tie-breaking: normalize and take max
            knn_max = max(knn_scores.values()) if knn_scores else 1.0
            tgt_max = max(target_scores.values()) if target_scores else 1
            knn_norm = knn_scores.get(drug_id, 0) / knn_max if knn_max > 0 else 0
            tgt_norm = target_scores.get(drug_id, 0) / tgt_max if tgt_max > 0 else 0
            combined_score = max(knn_norm, tgt_norm)

            results[drug_id] = (combined_score, min_rank)

        return results

    def _assign_confidence_tier(
        self,
        rank: int,
        train_frequency: int,
        mechanism_support: bool,
        has_targets: bool,
        disease_tier: int,
        category: str,
        drug_name: str = "",
        disease_name: str = "",
        drug_id: str = "",
    ) -> Tuple[ConfidenceTier, bool, Optional[str]]:
        """
        Assign confidence tier based on h135 criteria.
        Also applies h136/h144/h171/h274 category-specific rescue for Tier 2/3 diseases.

        Returns: (tier, rescue_applied, category_specific_tier)
        """
        # h526: Check inverse indications FIRST (safety-critical)
        # Must precede all positive tier assignments (cancer_same_type, hierarchy, etc.)
        drug_lower = drug_name.lower()
        disease_lower = disease_name.lower()
        for inv_drug, inv_diseases in INVERSE_INDICATION_PAIRS.items():
            if inv_drug in drug_lower:
                if any(inv_d in disease_lower for inv_d in inv_diseases):
                    return ConfidenceTier.FILTER, False, 'inverse_indication'

        # h542/h552/h685/h689: Non-therapeutic compounds → FILTER
        # These are diagnostic/imaging agents in DRKG, not therapeutic drugs.
        # All predictions are artifacts of diagnostic co-occurrence.
        # h689: Comprehensive census of diagnostic agents. Dual-use agents
        # (methylene blue, gallium nitrate, I-131, iobenguane, metyrapone,
        # tetracosactide) are NOT included — they have genuine therapeutic uses.
        NON_THERAPEUTIC_COMPOUNDS = (
            'fludeoxyglucose', 'indocyanine green',
            'technetium tc-99m sestamibi', 'ioflupane',
            'florbetaben', 'technetium tc-99m sulfur colloid',
            'flortaucipir', 'fluciclovine', 'pentagastrin',
            'florbetapir', 'flutemetamol', 'fluoroestradiol',
        )
        if any(ntc in drug_lower for ntc in NON_THERAPEUTIC_COMPOUNDS):
            return ConfidenceTier.FILTER, False, 'non_therapeutic_compound'

        # h540: Local anesthetic procedural artifact demotion
        # Lidocaine/bupivacaine appear in disease contexts due to PROCEDURAL use
        # (joint injections, biopsies, nerve blocks) not therapeutic treatment.
        # KG edges from co-occurrence inflate kNN scores for non-pain diseases.
        # Bupivacaine: demote to LOW for ALL categories (no systemic therapeutic use)
        # Lidocaine: demote to LOW except neurological (neuropathic pain),
        #   cardiovascular (Class Ib antiarrhythmic), dermatological (topical),
        #   psychiatric (IV lidocaine for depression has some evidence)
        if any(la in drug_lower for la in LOCAL_ANESTHETICS):
            la_therapeutic_categories = {'neurological', 'cardiovascular', 'dermatological', 'psychiatric'}
            if 'bupivacaine' in drug_lower:
                # Bupivacaine has NO systemic therapeutic use — always procedural
                return ConfidenceTier.LOW, False, 'local_anesthetic_procedural'
            elif category not in la_therapeutic_categories:
                # Lidocaine in non-therapeutic category → procedural artifact
                return ConfidenceTier.LOW, False, 'local_anesthetic_procedural'

        # h556/h560: Antimicrobial-pathogen mismatch demotion
        # Antimicrobial drugs predicted for wrong pathogen type have 0% holdout (h560).
        # h556 original: antibiotic→viral only. h560 expanded to all cross-pathogen mismatches.
        # Dual-activity drugs excluded (e.g., metronidazole treats bacteria AND parasites).
        # Holdout: mismatch 0.0% ± 0.0% (n=26.4/seed) vs matched 27.8% ± 4.8%.
        if category == 'infectious':
            # Drug antimicrobial activity sets (what pathogen types they can treat)
            _DUAL_ACTIVITY = {
                'metronidazole': {'antibacterial', 'antiparasitic'},
                'sulfadiazine': {'antibacterial', 'antiparasitic'},
                'doxycycline': {'antibacterial', 'antiparasitic'},
                'tetracycline': {'antibacterial', 'antiparasitic'},
                # h671: Amphotericin B is primarily antifungal with NARROW antiparasitic
                # activity (Leishmania only). Removed blanket 'antiparasitic' tag.
                # Leishmania exceptions added to _LEGITIMATE_CROSS_PAIRS below.
                # Does NOT work against: schistosomes, T. cruzi, Toxoplasma, malaria
                'amphotericin b': {'antifungal'},
            }
            _ANTIBACTERIAL_DRUGS = {
                'erythromycin', 'azithromycin', 'clarithromycin',
                'ciprofloxacin', 'levofloxacin', 'moxifloxacin', 'ofloxacin', 'norfloxacin', 'gemifloxacin',
                'gatifloxacin', 'garenoxacin', 'sitafloxacin',
                'gentamicin', 'tobramycin', 'amikacin', 'streptomycin',
                'vancomycin', 'clindamycin', 'trimethoprim', 'sulfamethoxazole', 'nitrofurantoin',
                'linezolid', 'daptomycin',
                'amoxicillin', 'ampicillin', 'penicillin', 'piperacillin',
                'cephalexin', 'ceftriaxone', 'cefazolin', 'cefepime', 'ceftazidime',
                'cefuroxime', 'cefaclor', 'cefadroxil', 'cefdinir', 'cefditoren',
                'cefotaxime', 'ceftizoxime',
                'meropenem', 'imipenem', 'ertapenem', 'aztreonam',
                'colistin', 'polymyxin', 'fosfomycin', 'chloramphenicol',
                'cycloserine', 'bedaquiline', 'tigecycline',
                'demeclocycline', 'oxytetracycline', 'minocycline',
            }
            _ANTIFUNGAL_DRUGS = {
                'fluconazole', 'itraconazole', 'ketoconazole', 'voriconazole',
                'posaconazole', 'caspofungin', 'micafungin', 'anidulafungin',
                'clotrimazole', 'miconazole', 'econazole', 'terbinafine', 'nystatin',
                'griseofulvin', 'flucytosine',
            }
            _ANTIPARASITIC_DRUGS = {
                'chloroquine', 'hydroxychloroquine', 'mefloquine', 'primaquine', 'tafenoquine',
                'quinine', 'artesunate', 'artemether', 'lumefantrine',
                'ivermectin', 'albendazole', 'mebendazole', 'praziquantel',
                'pyrimethamine', 'atovaquone', 'pentamidine',
                'miltefosine', 'suramin', 'nifurtimox', 'benznidazole',
                'diethylcarbamazine', 'trimetrexate',
            }
            # Legitimate cross-pathogen pairs (drug has known activity against this pathogen)
            _LEGITIMATE_CROSS_PAIRS = {
                ('pyrimethamine', 'acquired immunodeficiency syndrome aids'),
                ('trimetrexate', 'acquired immunodeficiency syndrome aids'),
                ('azithromycin', 'congenital toxoplasmosis'),
                ('itraconazole', 'chagas disease american trypanosomiasis'),
                ('itraconazole', 'cutaneous leishmaniasis caused by leishmania braziliensis'),
                ('ketoconazole', 'chagas disease american trypanosomiasis'),
                ('ketoconazole', 'cutaneous leishmaniasis caused by leishmania braziliensis'),
                ('ketoconazole', 'visceral leishmaniasis caused by leishmania donovani'),
                ('demeclocycline', 'malaria'),
                ('oxytetracycline', 'malaria'),
                # h671: Amphotericin B has genuine activity against Leishmania only
                # (FDA-approved for visceral leishmaniasis, guideline-recommended for cutaneous)
                ('amphotericin b', 'cutaneous leishmaniasis caused by leishmania braziliensis'),
                ('amphotericin b', 'visceral leishmaniasis caused by leishmania donovani'),
                ('amphotericin b', 'mycetoma'),  # Some fungal mycetoma responds to AmB
            }
            # Disease → pathogen type
            _VIRAL_KEYWORDS = [
                'influenza', 'respiratory syncytial', 'rsv', 'covid', 'sars',
                'herpes simplex', 'hsv', 'varicella', 'zoster', 'shingles',
                'cytomegalovirus', 'cmv', 'hiv', 'aids', 'hepatitis',
                'measles', 'mumps', 'rubella', 'dengue', 'ebola', 'rabies',
                'viral', 'smallpox', 'adenovirus', 'norovirus', 'rotavirus',
                'poliomyelitis', 'japanese encephalitis',
            ]
            _FUNGAL_KEYWORDS = [
                'aspergillosis', 'candidiasis', 'candida', 'cryptococcal', 'cryptococcosis',
                'coccidioidomycosis', 'histoplasmosis', 'fusariosis', 'zygomycosis',
                'chromomycosis', 'sporotrichosis', 'mycetoma', 'ringworm', 'tinea',
                'fungal',
            ]
            _PARASITIC_KEYWORDS = [
                'malaria', 'leishmaniasis', 'leishmania', 'chagas', 'trypanosomiasis',
                'toxoplasmosis', 'schistosomiasis', 'onchocerciasis', 'trichomoniasis',
                'amebiasis', 'scabies',
            ]

            # Determine drug's antimicrobial activities
            drug_activities = _DUAL_ACTIVITY.get(drug_lower)
            if not drug_activities:
                if drug_lower in _ANTIBACTERIAL_DRUGS:
                    drug_activities = {'antibacterial'}
                elif drug_lower in _ANTIFUNGAL_DRUGS:
                    drug_activities = {'antifungal'}
                elif drug_lower in _ANTIPARASITIC_DRUGS:
                    drug_activities = {'antiparasitic'}

            if drug_activities:
                # Determine disease pathogen type
                is_viral = any(vk in disease_lower for vk in _VIRAL_KEYWORDS)
                is_fungal = any(fk in disease_lower for fk in _FUNGAL_KEYWORDS)
                is_parasitic = any(pk in disease_lower for pk in _PARASITIC_KEYWORDS)

                # Check for mismatch (drug doesn't cover the pathogen type)
                needed_activity = None
                if is_viral:
                    needed_activity = 'antiviral'
                elif is_fungal:
                    needed_activity = 'antifungal'
                elif is_parasitic:
                    needed_activity = 'antiparasitic'

                if needed_activity and needed_activity not in drug_activities:
                    # Check legitimate exceptions
                    if (drug_lower, disease_lower) not in _LEGITIMATE_CROSS_PAIRS:
                        return ConfidenceTier.LOW, False, 'antimicrobial_pathogen_mismatch'

        # h274/h396: For cancer, check cancer type match BEFORE applying rank filter
        # h393 holdout validation: cancer_same_type has 24.5% full-data, 19.2% holdout precision
        # This is MEDIUM-level, not GOLDEN. Demoted from GOLDEN→MEDIUM by h396.
        if category == 'cancer' and drug_id:
            has_cancer_gt, same_type_match, _ = self._check_cancer_type_match(drug_id, disease_name)
            if same_type_match:
                # h538: Targeted therapy (kinase inhibitors + immunotherapy) has 12.6% holdout
                # vs cytotoxic 53%. They don't transfer across cancer subtypes via kNN.
                # Demote targeted therapy cancer_same_type MEDIUM → LOW.
                if any(t in drug_lower for t in CANCER_TARGETED_THERAPY):
                    return ConfidenceTier.LOW, False, 'cancer_targeted_therapy'
                # h633: cancer_same_type + mechanism + rank<=10 = 56.6% ± 9.7% holdout (expanded GT)
                # Promoted MEDIUM → HIGH. Reopened CLOSED direction #4 with expanded GT.
                # Non-circular: mechanism is drug-target/disease-gene overlap, rank is kNN score.
                if mechanism_support and rank <= 10:
                    return ConfidenceTier.HIGH, True, 'cancer_same_type_mech_rank10'
                # h634: cancer_same_type without mechanism = 17.9% ± 4.2% holdout (below MEDIUM)
                # Demote to LOW. With mechanism but rank>10 stays MEDIUM (33.8%).
                if not mechanism_support:
                    return ConfidenceTier.LOW, False, 'cancer_same_type_no_mechanism'
                # h648: cancer_same_type + mechanism + rank 21+ = 25.5% ± 15.2% holdout
                # Well below MEDIUM avg (41.5%). Rank 11-20 = 42.3% (above MEDIUM).
                # Demote rank 21+ to LOW.
                if rank >= 21:
                    return ConfidenceTier.LOW, False, 'cancer_same_type_high_rank'
                # h396: Demoted from GOLDEN to MEDIUM (24.5% full, 19.2% holdout)
                # cancer_same_type was 57% of GOLDEN predictions, dragging GOLDEN below HIGH
                return ConfidenceTier.MEDIUM, True, 'cancer_same_type'
            if not has_cancer_gt:
                # No cancer GT → FILTER (0% precision)
                return ConfidenceTier.FILTER, False, 'cancer_no_gt'
            # Cross-type: continue to standard filtering, will get MEDIUM in _apply_category_rescue

        # h399/h418/h423: Multiple attempts to rescue rank 21-30 predictions have FAILED holdout:
        # - h399/h418: Hierarchy-before-rank: full GOLDEN +2.1pp, holdout HIGH -6.2pp → REVERTED
        # - h423: Category+mechanism rescue: full GOLDEN +1.6pp, holdout HIGH -8.7pp → REVERTED
        # Root cause: drugs at rank 21-30 appear high-quality on full data (high freq, mechanism)
        # but lose these signals on holdout (80% GT → lower freq, weaker mechanism evidence).
        # The rank>20 filter is a robust, validated boundary. DO NOT attempt further rescues
        # without first solving the underlying freq/mechanism inflation at full data.

        # FILTER tier (h123 negative signals)
        if rank > 20:
            return ConfidenceTier.FILTER, False, None
        if not has_targets:
            return ConfidenceTier.FILTER, False, None

        # h708: Validated complication drugs bypass freq<=2 filter.
        # Anti-VEGF drugs (ranibizumab, aflibercept) have small GT footprint (2-3 diseases)
        # → low kNN frequency, but are standard of care for retinal diseases.
        # Only applies to rank<=20 drugs matching COMPLICATION_VALIDATED_DRUGS.
        # Rescues: ranibizumab→PDR (R2), ranibizumab→ROP (R1), aflibercept→ROP (R4).
        is_validated_complication_drug = False
        for comp_term, validated_drugs in COMPLICATION_VALIDATED_DRUGS.items():
            if comp_term in disease_lower:
                if any(v in drug_lower for v in validated_drugs):
                    is_validated_complication_drug = True
                    break

        if train_frequency <= 2 and not mechanism_support:
            if not is_validated_complication_drug:
                return ConfidenceTier.FILTER, False, None

        # h153/h476: Corticosteroids for iatrogenic conditions = FILTER
        # Corticosteroids CAUSE these conditions (inverse indications):
        # - Metabolic: hyperglycemia, diabetes (h153)
        # - Dermatological: steroid rosacea (h476)
        # - Musculoskeletal: osteoporosis, avascular necrosis (h476)
        # - Ophthalmic: glaucoma, cataracts (h476)
        # - GI: pancreatitis (h476 - steroids cause drug-induced pancreatitis; they
        #   treat autoimmune pancreatitis but generic "pancreatitis" is not autoimmune)
        # - Endocrine: Cushing syndrome (h476 - exogenous steroids ARE the cause)
        if any(steroid in drug_lower for steroid in CORTICOSTEROID_DRUGS):
            if category == 'metabolic':
                return ConfidenceTier.FILTER, False, None
            steroid_iatrogenic = ['rosacea', 'osteoporosis', 'avascular necrosis',
                                  'glaucoma', 'cataract', 'pancreatitis', 'cushing']
            if any(iatrogen in disease_lower for iatrogen in steroid_iatrogenic):
                return ConfidenceTier.FILTER, False, 'corticosteroid_iatrogenic'

        # h480/h526: Inverse indication check moved to top of function (before cancer_same_type)
        # to ensure safety-critical filters take precedence over all positive tier assignments

        # h271: Filter cross-domain predictions for domain-isolated drugs (0% precision)
        # Domain-isolated drugs only treat one category, cross-domain predictions are always wrong
        if drug_id and self._is_cross_domain_isolated(drug_id, category):
            return ConfidenceTier.FILTER, False, 'cross_domain_isolated'

        # h280/h281: Filter base→complication predictions (0% precision)
        # When drug treats base disease (diabetes) and prediction is for complication (diabetic nephropathy)
        # h280 finding: 0% precision for these predictions
        if drug_id and self._is_base_to_complication(drug_id, disease_name):
            return ConfidenceTier.FILTER, False, 'base_to_complication'

        # h284/h291: Boost comp→base predictions based on transferability
        # HIGH transferability (≥50%) = 62.5% precision → HIGH tier
        # h296: Statins for CV events achieve 100% precision → GOLDEN tier
        #       Non-statins for CV events achieve 0% precision → NO BOOST
        if drug_id:
            is_comp_to_base, transferability, is_statin_cv = self._is_comp_to_base(drug_id, disease_name)
            if is_comp_to_base:
                if is_statin_cv:
                    # h296: Statins treating CV events and predicting atherosclerosis = 100% precision
                    return ConfidenceTier.GOLDEN, True, 'statin_cv_event'
                # h296: Check if this is a non-statin CV event (0% precision - no boost)
                drug_name_lower = self._get_drug_name(drug_id).lower()
                is_statin = any(s in drug_name_lower for s in STATIN_NAMES)
                is_cv_pred = 'athero' in disease_name.lower()
                if is_cv_pred and not is_statin:
                    # Non-statin predicting atherosclerosis from CV event - don't boost (0% precision)
                    pass  # Continue to standard tier assignment
                elif transferability >= 50:
                    # h284: HIGH transferability comp→base = 62.5% precision
                    return ConfidenceTier.HIGH, True, f'comp_to_base_high_{transferability:.0f}'
                elif transferability >= 20:
                    # MEDIUM transferability
                    return ConfidenceTier.MEDIUM, True, f'comp_to_base_med_{transferability:.0f}'
                # LOW transferability - no boost, continue to standard tier assignment

        # h297: Mechanism-specific diseases should get LOW confidence (kNN will fail)
        # These diseases have drugs that don't transfer from similar diseases
        # Only 6.6% of their GT drugs are repurposable (vs 87.9% for others)
        if self._is_mechanism_specific_disease(disease_name):
            # For mechanism-specific diseases, cap confidence at LOW
            # This overrides other tier boosts since kNN fundamentally won't work
            return ConfidenceTier.LOW, False, 'mechanism_specific'

        # h346: Cancer-only drugs for non-cancer diseases have 0% precision
        # (115 predictions, 0 GT hits across ALL tiers)
        # These drugs have NO approved non-cancer uses (BRAF, PD-1, BCL2, PARP, etc.)
        # This generalizes h340 (MEK inhibitors) to all cancer-only drug classes
        if self._is_cancer_only_drug_non_cancer(drug_name, disease_name):
            return ConfidenceTier.FILTER, False, 'cancer_only_non_cancer'

        # h353: Complication diseases with non-validated drug classes have 0% precision
        # For nephropathy/retinopathy/cardiomyopathy/neuropathy, only specific drug classes work
        # Non-validated classes have 0% precision across 214 predictions (zero GT loss)
        if self._is_complication_non_validated_class(drug_name, disease_name):
            return ConfidenceTier.FILTER, False, 'complication_non_validated'

        # h354: CV pathway-comprehensive boost
        # Drugs with GT for BOTH CV base (hypertension/lipids) AND CV complications
        # have 48.8% precision for CV complication predictions (vs 7.6% non-pathway)
        # Boost pathway-comprehensive to HIGH for CV complications
        if self._is_cv_complication(disease_name):
            if self._is_cv_pathway_comprehensive(drug_name):
                # h402: 26.0% ± 4.9% holdout precision (n=105) vs HIGH avg 44.1%
                # Demoted from HIGH to MEDIUM
                return ConfidenceTier.MEDIUM, True, 'cv_pathway_comprehensive'
            # Non-pathway-comprehensive CV complication: 7.6% precision
            # Don't filter (some GT hits exist), but don't boost either
            # Standard tier assignment will apply (likely MEDIUM/LOW)

        # h297: Highly repurposable diseases can get a confidence boost
        # These diseases have drugs widely used across many conditions
        # 87.9% of their GT drugs are repurposable
        is_highly_repurposable = self._is_highly_repurposable_disease(disease_name)

        # h273/h276/h278: Disease hierarchy matching - boost tier for subtype refinements
        # This indicates the prediction is a subtype refinement (e.g., "psoriasis" → "plaque psoriasis")
        # 2.9x precision improvement overall (8.5% → 24.7%)
        #
        # h278: Category-specific tier assignment based on FULL evaluation (all diseases):
        # - Metabolic: 65.2% → GOLDEN (>50% threshold)
        # - Neurological: 63.3% → GOLDEN (>50% threshold)
        # - Autoimmune: 44.7% → HIGH (below 50% threshold)
        # - Respiratory: 40.4% → HIGH
        # - Cardiovascular: 22.6% → HIGH
        # - Infectious: 22.1% → HIGH
        HIERARCHY_GOLDEN_CATEGORIES = {'metabolic', 'neurological'}
        # h615: Group-level GOLDEN promotions validated with expanded GT holdout
        # coronary: 65.5% ± 1.2% holdout (n=13/seed), arrhythmia: 72.9% ± 1.5% (n=11/seed)
        # rheumatoid_arthritis: 86.4% ± 8.7% (n=23/seed), colitis: 85.7% ± 0.0% (n=7/seed)
        HIERARCHY_PROMOTE_TO_GOLDEN = {
            'coronary', 'arrhythmia', 'rheumatoid_arthritis', 'colitis',
        }
        # h385: Thyroid hierarchy has 20.6% precision vs 35.8% GOLDEN avg - demote to HIGH
        # h402: Diabetes hierarchy 31.5% holdout ± 13.8% (n=72) vs GOLDEN avg 46.3% - demote to HIGH
        # h430: Attempted T2D rescue back to GOLDEN — FAILED holdout (42.1%, GOLDEN dropped -5pp)
        HIERARCHY_DEMOTE_TO_HIGH = {'thyroid', 'diabetes'}
        # h396: These hierarchy groups have 0% precision (n>=2) - demote to MEDIUM
        HIERARCHY_DEMOTE_TO_MEDIUM = {'parkinsons', 'migraine'}
        # h649: pneumonia demoted MEDIUM→LOW (16.7% ± 0.0% holdout, n=6/seed)
        # h402 originally demoted to MEDIUM (6.7% holdout), but holdout is near LOW (14.8%)
        # 80% full-data vs 16.7% holdout = most overfitted rule (Δ=-63pp)
        HIERARCHY_DEMOTE_TO_LOW = {'pneumonia'}

        if category in DISEASE_HIERARCHY_GROUPS and drug_id:
            has_category_gt, same_group_match, matching_group = self._check_disease_hierarchy_match(
                drug_id, disease_name, category
            )
            if same_group_match:
                # h649: Demote to LOW for groups with near-LOW holdout
                if matching_group in HIERARCHY_DEMOTE_TO_LOW:
                    return ConfidenceTier.LOW, True, f'{category}_hierarchy_{matching_group}'
                # h396: Demote 0% precision groups to MEDIUM
                if matching_group in HIERARCHY_DEMOTE_TO_MEDIUM:
                    return ConfidenceTier.MEDIUM, True, f'{category}_hierarchy_{matching_group}'
                # h385/h402: Check if this specific group should be demoted to HIGH
                # h430: Attempted T2D→GOLDEN rescue, failed holdout (42.1%, caused GOLDEN<HIGH)
                if matching_group in HIERARCHY_DEMOTE_TO_HIGH:
                    return ConfidenceTier.HIGH, True, f'{category}_hierarchy_{matching_group}'
                # h615: Group-level GOLDEN promotion (holdout-validated with expanded GT)
                if matching_group in HIERARCHY_PROMOTE_TO_GOLDEN:
                    return ConfidenceTier.GOLDEN, True, f'{category}_hierarchy_{matching_group}'
                # h276: Use GOLDEN for high-precision categories (>70%), HIGH otherwise
                if category in HIERARCHY_GOLDEN_CATEGORIES:
                    return ConfidenceTier.GOLDEN, True, f'{category}_hierarchy_{matching_group}'
                else:
                    return ConfidenceTier.HIGH, True, f'{category}_hierarchy_{matching_group}'

        # Apply h136/h144/h171/h274 category-specific rescue for Tier 2/3
        if disease_tier > 1:
            rescued_tier = self._apply_category_rescue(
                rank, train_frequency, mechanism_support, category, drug_name, disease_name, drug_id
            )
            if rescued_tier is not None:
                return rescued_tier, True, category

        # Standard h135 tier assignment for Tier 1
        # h311/h314/h316: Check ATC coherence and mismatch rules for tier adjustment
        # h311: Incoherent GOLDEN has 22.5% precision vs 30.3% coherent (+7.8 pp gap)
        # h316: Zero-precision mismatches should always be FILTERED
        # Note: High-precision mismatches (10-27%) are BELOW GOLDEN (30%) so demotion is correct
        #       They are ABOVE HIGH baseline (20%) so regular demotion path is optimal
        is_coherent = drug_name and category and self._is_atc_coherent(drug_name, category)

        # h314/h316: Check for zero-precision ATC mismatch rules
        _, is_zero_prec_mismatch, _ = (False, False, None)
        if drug_name and category:
            _, is_zero_prec_mismatch, _ = self._check_atc_mismatch_rules(drug_name, category)

        # h316: Zero-precision mismatches are always FILTER (0-3% precision)
        # These are patterns that NEVER work: A→cancer, J→cancer, N→cancer, etc.
        # h708: Exception for validated complication drugs (e.g., aflibercept ATC=L for ophthalmic)
        # Some drugs have dual ATC codes (L + S) but only the first is checked.
        if is_zero_prec_mismatch and not is_validated_complication_drug:
            return ConfidenceTier.FILTER, False, 'zero_precision_mismatch'

        # GOLDEN tier (Tier1 + freq>=10 + mechanism)
        if disease_tier == 1 and train_frequency >= 10 and mechanism_support:
            # h311: Demote incoherent GOLDEN to HIGH
            if not is_coherent:
                return ConfidenceTier.HIGH, False, 'incoherent_demotion'
            return ConfidenceTier.GOLDEN, False, None

        # HIGH tier
        if train_frequency >= 15 and mechanism_support:
            # h311/h488: Demote incoherent HIGH to LOW (3.6% holdout, 6.0% full-data, n=50)
            if not is_coherent:
                return ConfidenceTier.LOW, False, 'incoherent_demotion'
            return ConfidenceTier.HIGH, False, None
        if rank <= 5 and train_frequency >= 10 and mechanism_support:
            # h311/h488: Demote incoherent HIGH to LOW (3.6% holdout, 6.0% full-data, n=50)
            if not is_coherent:
                return ConfidenceTier.LOW, False, 'incoherent_demotion'
            return ConfidenceTier.HIGH, False, None

        # MEDIUM tier
        # h462/h463: Category-specific MEDIUM demotions (holdout-validated)
        # Categories where MEDIUM holdout precision is at or below LOW tier (14.8%):
        # - GI: 10.9% full-data as LOW; kNN finds wrong drug classes
        # - Immunological: 2.5% ± 3.5% holdout (38.9% full-data = massive overfitting, n=5 diseases)
        # - Reproductive: 0.0% holdout (4.5% full-data, n=5 diseases)
        # - Neurological: 10.2% ± 11.1% holdout (15.7% full-data, n=24 diseases)
        # - Cardiovascular: h490 standard 2.0% ± 4.0%, ATC coherent 8.4% ± 10.4% holdout
        #   cv_pathway_comprehensive (21.4%) and target_overlap (16.2%) return BEFORE this check
        # - Hematological: h553 10.0% ± 20.0% holdout (27.6% full-data, n=8/seed)
        #   default sub-reason: 0% holdout (n=6/seed × 4 seeds), target_overlap: 25% ± 43% (n=1.2/seed)
        # - Metabolic: h603 standard holdout 8.3% ± 14.4% (n=3.8/seed), total demotion 10.3% ± 4.7%
        #   holdout (n=39 across 5 seeds). Statin/TZD rescued to LOW by _apply_category_rescue;
        #   hierarchy rules (diabetes/thyroid) return GOLDEN/HIGH before this check.
        #   Already excluded from ATC coherent rescue (h395).
        # h603: Respiratory (22.3% holdout) and endocrine (24.5% holdout) NOT demoted — precision
        #   is below MEDIUM avg but above LOW avg, making demotion marginal. Small n = unreliable.
        # Category-specific rescue rules (h380 GI, hierarchy) still promote valid drugs to HIGH.
        MEDIUM_DEMOTED_CATEGORIES = {'gastrointestinal', 'immunological', 'reproductive', 'neurological', 'cardiovascular', 'hematological', 'metabolic'}
        if category in MEDIUM_DEMOTED_CATEGORIES:
            # h618: CV drug-class rescue — established CV drug classes have 32-50% holdout
            # (expanded GT, 5-seed) while non-CV drugs (antibiotics, biologics, corticosteroids)
            # have <3-18%. Rescue anticoagulants/antiplatelets (32.6%, n=14.2/seed) and
            # other established CV classes to MEDIUM; keep non-CV drugs as LOW.
            # h643: Mechanism gate — CV rescue with mechanism: 40.4% ± 20.5% (n=14/seed).
            # CV rescue without mechanism: 22.5% ± 14.7% (n=29/seed, 3/5 seeds at LOW-level).
            # Require mechanism for rescue; no-mech CV drugs stay as LOW.
            if category == 'cardiovascular' and self._is_established_cv_drug(drug_name) and mechanism_support:
                return ConfidenceTier.MEDIUM, False, 'cv_established_drug_rescue'
            return ConfidenceTier.LOW, False, f'{category}_medium_demotion'

        if train_frequency >= 5 and mechanism_support:
            return ConfidenceTier.MEDIUM, False, 'default_freq5_mechanism'
        if train_frequency >= 10:
            # h555: Frequency-only MEDIUM (no mechanism) at rank 11-20 has 18.7-20.7% holdout
            # — near LOW level (16.2%). Demote to LOW. Rank 1-10 stays MEDIUM (24-38.7%).
            if rank >= 11 and not mechanism_support:
                return ConfidenceTier.LOW, False, 'default_no_mech_high_rank'
            # h657 INVALIDATED: With expanded GT, NoMech R6-10 = 40.5% ± 9.4% holdout
            # (n=14.6/seed). This is MEDIUM-quality (z=-0.4 vs MEDIUM avg). Original 30.0%
            # was from internal GT. Expanded GT lifts signal-rich predictions. Keep as MEDIUM.
            # h662: Named reasons for holdout tracking
            if mechanism_support:
                return ConfidenceTier.MEDIUM, False, 'default_freq10_mechanism'
            elif rank <= 5:
                return ConfidenceTier.MEDIUM, False, 'default_freq10_nomech_r1_5'
            else:
                return ConfidenceTier.MEDIUM, False, 'default_freq10_nomech_r6_10'

        # h297: Highly repurposable diseases get MEDIUM instead of LOW
        # These diseases have drugs widely used across many conditions
        if is_highly_repurposable and (mechanism_support or train_frequency >= 5):
            return ConfidenceTier.MEDIUM, False, 'highly_repurposable'

        # h309/h310: ATC coherence boost for LOW tier predictions
        # Coherent predictions have 35.5% precision vs 18.7% for incoherent
        # Boost coherent LOW→MEDIUM when drug has good rank and some support
        # h395: Exclude metabolic (4.3%) and neurological (10.8%) — below MEDIUM avg
        # h487: Exclude hematological (5.0% holdout, 18.2% full-data, n=44)
        # h606: Exclude psychiatric (17.2% ± 5.8% holdout, n=13.4/seed, p=0.0006 < MEDIUM avg)
        # h651: Exclude endocrine (0%), musculoskeletal (0%), respiratory (19.4%), renal (11.1%)
        #       All n<5/seed but consistently below MEDIUM avg (42.9%). +0.7pp MEDIUM.
        ATC_COHERENT_EXCLUDED = {'metabolic', 'neurological', 'hematological', 'psychiatric',
                                 'endocrine', 'musculoskeletal', 'respiratory', 'renal'}
        if drug_name and category and category not in ATC_COHERENT_EXCLUDED and self._is_atc_coherent(drug_name, category):
            # Only boost if there's some additional evidence
            if rank <= 10 and (mechanism_support or train_frequency >= 3):
                return ConfidenceTier.MEDIUM, True, f'atc_coherent_{category}'

        # LOW tier
        return ConfidenceTier.LOW, False, None

    def _apply_category_rescue(
        self,
        rank: int,
        train_frequency: int,
        mechanism_support: bool,
        category: str,
        drug_name: str = "",
        disease_name: str = "",
        drug_id: str = "",
    ) -> Optional[ConfidenceTier]:
        """
        Apply h136/h144/h171/h274 category-specific rescue filters.

        Returns the rescued tier or None if no rescue criteria met.

        h136 findings (REVISED by h386/h387):
        - Infectious: REMOVED - h386 found 5.3% precision (not 55.6%)
        - Cardiovascular: rank<=5 + mech = 38.2% precision (HIGH)
        - Respiratory: rank<=10 + freq>=15 + mech = 35.0% precision (HIGH)

        h144 findings:
        - Metabolic + statin + rank<=10 = 60.0% precision (GOLDEN!)

        h387: Removed infectious GOLDEN/HIGH rules - precision was 5.3% not 55.6%.
        The infectious_hierarchy rules (UTI 75%, TB 45.5%) in _assign_confidence_tier
        still apply for bacterial diseases with specific matches.
        """
        # h387: Infectious rules removed - were 5.3% precision
        # Specific hierarchy rules (UTI, TB) in _assign_confidence_tier still apply

        if category == 'cardiovascular':
            drug_lower = drug_name.lower()
            disease_lower = disease_name.lower()

            # h265: SGLT2 inhibitors achieve 71.4% precision for cardiovascular (GOLDEN)
            if any(sglt2 in drug_lower for sglt2 in SGLT2_INHIBITORS):
                return ConfidenceTier.GOLDEN  # 71.4% precision (h265/h163)

            # h217: Heart failure specific GOLDEN rescue rules (from h212)
            is_heart_failure = any(kw in disease_lower for kw in HF_KEYWORDS)
            if is_heart_failure:
                # Loop diuretics: 75% precision for heart failure (GOLDEN)
                if any(diuretic in drug_lower for diuretic in LOOP_DIURETICS):
                    return ConfidenceTier.GOLDEN  # 75% precision (h212)
                # Aldosterone antagonists: 50% precision for heart failure (GOLDEN)
                if any(aa in drug_lower for aa in ALDOSTERONE_ANTAGONISTS):
                    return ConfidenceTier.GOLDEN  # 50% precision (h212)
                # ARBs: 27% precision for heart failure (HIGH)
                if any(arb in drug_lower for arb in ARB_DRUGS):
                    return ConfidenceTier.HIGH  # 27% precision (h212)
                # Beta-blockers: 21% precision for heart failure (HIGH)
                if any(bb in drug_lower for bb in BETA_BLOCKERS):
                    return ConfidenceTier.HIGH  # 21% precision (h212)

            # h218: Hypertension + ARBs = 20% precision (HIGH)
            if 'hypertension' in disease_lower or 'hypertensive' in disease_lower:
                if any(arb in drug_lower for arb in ARB_DRUGS):
                    return ConfidenceTier.HIGH  # 20% precision (h212)

            # h230: AFib specific rescue rules (from h229 validation: +41.7pp vs kNN)
            # AFib: 65.6% drug-class coverage vs 23.9% kNN
            is_afib = any(kw in disease_lower for kw in AFIB_KEYWORDS)
            if is_afib:
                # Anticoagulants for AFib stroke prevention (HIGH tier)
                if any(ac in drug_lower for ac in ANTICOAGULANT_DRUGS):
                    return ConfidenceTier.HIGH  # Stroke prevention
                # Rate control drugs (beta-blockers already handled above, add non-BB rate control)
                if any(rc in drug_lower for rc in RATE_CONTROL_DRUGS):
                    return ConfidenceTier.HIGH
                # Antiarrhythmics (amiodarone, sotalol already in beta-blockers, add others)
                if 'amiodarone' in drug_lower or 'dronedarone' in drug_lower:
                    return ConfidenceTier.HIGH

            # h230: MI/ACS specific rescue rules (from h229 validation: +37.8pp vs kNN)
            # MI: 47.9% drug-class coverage vs 10.2% kNN
            is_mi = any(kw in disease_lower for kw in MI_KEYWORDS)
            if is_mi:
                # Antiplatelet drugs for MI (HIGH tier)
                if any(ap in drug_lower for ap in ANTIPLATELET_DRUGS):
                    return ConfidenceTier.HIGH
                # Statins for MI (already covered by metabolic statin rule, but add here for clarity)
                if any(st in drug_lower for st in STATIN_DRUGS):
                    return ConfidenceTier.HIGH
                # Nitrates for acute MI
                if any(ni in drug_lower for ni in NITRATE_DRUGS):
                    return ConfidenceTier.HIGH

            # h230: CAD/Angina specific rescue rules (from h229 validation: +34.1pp vs kNN)
            # CAD: 65.8% drug-class coverage vs 31.7% kNN
            is_cad = any(kw in disease_lower for kw in CAD_KEYWORDS)
            if is_cad:
                # Antiplatelet drugs for CAD
                if any(ap in drug_lower for ap in ANTIPLATELET_DRUGS):
                    return ConfidenceTier.HIGH
                # Nitrates for angina
                if any(ni in drug_lower for ni in NITRATE_DRUGS):
                    return ConfidenceTier.HIGH
                # Statins for CAD
                if any(st in drug_lower for st in STATIN_DRUGS):
                    return ConfidenceTier.HIGH

            # h136/h395: Generic CV rescue → MEDIUM (was HIGH)
            # h395 found 26.5% precision (n=34) vs HIGH avg 47.1%
            if rank <= 5 and mechanism_support:
                return ConfidenceTier.MEDIUM  # h395: demoted from HIGH (26.5% precision)
            # h154/h266: Beta-blockers achieve 42.1% precision at rank<=10
            # h266 found extending from rank<=5 to rank<=10 captures more predictions
            if rank <= 10 and any(bb in drug_lower for bb in BETA_BLOCKERS):
                return ConfidenceTier.HIGH  # 42.1% precision (h266)

        elif category == 'respiratory':
            drug_lower = drug_name.lower()

            # h265: Fluoroquinolones achieve 44.4% precision for respiratory (HIGH)
            if any(fq in drug_lower for fq in FLUOROQUINOLONE_DRUGS):
                return ConfidenceTier.HIGH  # 44.4% precision (h265/h163)

            # h395: Respiratory generic rescue → MEDIUM (was HIGH)
            # h395 found 14.3% precision (n=21) vs HIGH avg 47.1%
            if rank <= 10 and train_frequency >= 15 and mechanism_support:
                return ConfidenceTier.MEDIUM  # h395: demoted from HIGH (14.3% precision)

        elif category == 'metabolic':
            drug_lower = drug_name.lower()

            # h265/h395: TZDs for metabolic → was MEDIUM (was GOLDEN)
            # h395 found 6.9% precision — TZDs only work for diabetes, not thyroid/rare metabolic
            # h553: Demoted to LOW (8.3% ± 14.4% holdout, n=4.2/seed — below LOW avg 16.2%)
            if any(tzd in drug_lower for tzd in THIAZOLIDINEDIONES):
                return ConfidenceTier.LOW  # h553: demoted from MEDIUM (8.3% holdout)

            # h144/h395: Statins for metabolic → was MEDIUM (was GOLDEN)
            # h395 found 6.9% overall — statins get boosted for non-lipid metabolic diseases
            # h553: Demoted to LOW (8.3% ± 14.4% holdout, n=4.2/seed — below LOW avg 16.2%)
            if rank <= 10 and any(statin in drug_lower for statin in STATIN_DRUGS):
                return ConfidenceTier.LOW  # h553: demoted from MEDIUM (8.3% holdout)

        elif category == 'cancer':
            # h150/h274: Drug class rescue for cancer
            # Note: h274 same-type and no-GT checks are now in _assign_confidence_tier
            # This code only runs for cross-type repurposing (30.6% precision)
            drug_lower = drug_name.lower()
            disease_lower = disease_name.lower()

            # h215: CDK4/6 inhibitors for breast cancer = 100% precision (GOLDEN)
            is_breast_cancer = any(kw in disease_lower for kw in BREAST_CANCER_KEYWORDS)
            is_cdk_inhibitor = any(cdk in drug_lower for cdk in CDK_INHIBITORS)
            if is_breast_cancer and is_cdk_inhibitor:
                return ConfidenceTier.GOLDEN  # 100% precision (h215)

            # h197: Colorectal cancer + monoclonal antibody = 50-60% precision (GOLDEN)
            is_colorectal = any(kw in disease_lower for kw in COLORECTAL_KEYWORDS)
            is_mab = any(mab in drug_lower for mab in COLORECTAL_MABS) or drug_lower.endswith('mab')
            if is_colorectal and is_mab:
                return ConfidenceTier.GOLDEN  # 50-60% precision (h160)

            # h201: BCR-ABL inhibitors for CML/ALL (22% precision in h198)
            is_cml = any(kw in disease_lower for kw in CML_KEYWORDS)
            is_all = any(kw in disease_lower for kw in ALL_KEYWORDS)
            is_bcr_abl_inhibitor = any(drug in drug_lower for drug in BCR_ABL_INHIBITORS)
            if (is_cml or is_all) and is_bcr_abl_inhibitor:
                return ConfidenceTier.HIGH  # 22% precision (h198)

            # h201: BTK inhibitors for CLL/lymphoma (22% precision in h198)
            is_cll = any(kw in disease_lower for kw in CLL_KEYWORDS)
            is_lymphoma_btk = any(kw in disease_lower for kw in LYMPHOMA_BTK_KEYWORDS)
            is_btk_inhibitor = any(drug in drug_lower for drug in BTK_INHIBITORS)
            if (is_cll or is_lymphoma_btk) and is_btk_inhibitor:
                return ConfidenceTier.HIGH  # 22% precision (h198)

            if rank <= 5 and any(taxane in drug_lower for taxane in TAXANE_DRUGS):
                return ConfidenceTier.HIGH  # 40.0% precision
            if rank <= 10 and any(alk in drug_lower for alk in ALKYLATING_DRUGS):
                return ConfidenceTier.HIGH  # 36.4% precision

            # h274/h395: Cross-type cancer drugs (has cancer GT but different type)
            # h395 found 0.9% precision (n=349) — much lower than MEDIUM avg 21.2%
            # Original h274 measured 30.6% but that was before other tier rules reduced the pool
            return ConfidenceTier.LOW  # h395: demoted from MEDIUM (0.9% precision)

        elif category == 'ophthalmic':
            # h150: Drug class rescue for ophthalmic (62.5%/48% precision)
            drug_lower = drug_name.lower()
            if rank <= 15 and any(ab in drug_lower for ab in OPHTHALMIC_ANTIBIOTICS):
                return ConfidenceTier.GOLDEN  # 62.5% precision
            if rank <= 15 and any(st in drug_lower for st in OPHTHALMIC_STEROIDS):
                return ConfidenceTier.HIGH  # 48.0% precision

        elif category == 'dermatological':
            # h150: Topical steroids achieve 63.6% precision at rank<=5
            drug_lower = drug_name.lower()
            if rank <= 5 and any(ts in drug_lower for ts in TOPICAL_STEROIDS):
                return ConfidenceTier.GOLDEN  # 63.6% precision

        elif category == 'hematological':
            # h150/h395: Corticosteroids for hematological → MEDIUM (was HIGH)
            # h395 found 19.1% precision (n=94) vs HIGH avg 47.1%
            # The h150 precision (48.6%) was before other tier rules reduced the pool
            drug_lower = drug_name.lower()
            if rank <= 10 and any(steroid in drug_lower for steroid in CORTICOSTEROID_DRUGS):
                return ConfidenceTier.MEDIUM  # h395: demoted from HIGH (19.1% precision)

        elif category == 'autoimmune':
            drug_lower = drug_name.lower()

            # h157: DMARDs achieve 75.4% precision for autoimmune diseases
            if rank <= 10 and any(dmard in drug_lower for dmard in DMARD_DRUGS):
                return ConfidenceTier.GOLDEN  # 75.4% precision

            # h265: NSAIDs achieve 50% precision for autoimmune (HIGH)
            if any(nsaid in drug_lower for nsaid in NSAID_DRUGS):
                return ConfidenceTier.HIGH  # 50.0% precision (h265/h163)

            # h189: ATC L4-based rescue for autoimmune
            # H02AB (glucocorticoids) = 77%, L04AX (traditional immunosuppressants) = 82%
            atc_l4 = self._get_atc_level4(drug_name)
            if atc_l4:
                # Check for high-precision autoimmune ATC codes
                if atc_l4 & ATC_HIGH_PRECISION_AUTOIMMUNE:
                    # Exclude biologics even if they have other codes
                    if not (atc_l4 & ATC_BIOLOGIC_CODES):
                        if rank <= 10:
                            return ConfidenceTier.GOLDEN  # 77-82% precision
                # Demote biologics (TNF, IL, JAK inhibitors) - only 8-17% precision
                if atc_l4 & ATC_BIOLOGIC_CODES:
                    # Don't rescue biologics to GOLDEN, let them get normal tier
                    pass  # Explicit no-op to document this is intentional

        elif category == 'neurological':
            # h187: Anticonvulsant + rank<=10 + mechanism = 58.8% precision (GOLDEN)
            # BUT only for seizure/epilepsy diseases, not all neurological
            # h171: Drug class-based prediction (60.4% coverage vs 18% kNN)
            drug_lower = drug_name.lower()
            disease_lower = disease_name.lower()

            # h187: GOLDEN tier for anticonvulsants - ONLY for seizure-related diseases
            is_seizure_disease = 'epilepsy' in disease_lower or 'seizure' in disease_lower
            if is_seizure_disease:
                anticonvulsants = NEUROLOGICAL_DRUG_CLASS_MEMBERS.get('anticonvulsant', [])
                is_anticonvulsant = any(ac.lower() in drug_lower for ac in anticonvulsants)
                if rank <= 10 and mechanism_support and is_anticonvulsant:
                    return ConfidenceTier.GOLDEN  # 58.8% precision (h187)

            # h171: HIGH tier for drug class matches
            if self._is_neurological_class_match(drug_lower, disease_name):
                return ConfidenceTier.HIGH  # ~60% coverage

        elif category == 'reproductive':
            # h183: Hormone drugs achieve 26.3% precision for reproductive diseases
            # vs 3.1% for non-hormone drugs - significant enough for HIGH tier
            drug_lower = drug_name.lower()
            if any(h in drug_lower for h in REPRODUCTIVE_HORMONE_DRUGS):
                return ConfidenceTier.HIGH  # 26.3% precision (h183)

        elif category == 'gastrointestinal':
            # h380: GI category is worst (42.9% R@30) because kNN finds wrong drug classes
            # Apply drug class rescue similar to h171 for neurological
            drug_lower = drug_name.lower()
            disease_lower = disease_name.lower()

            # Constipation diseases (4 failures) -> laxatives/opioid antagonists
            if 'constipation' in disease_lower:
                laxatives = GI_DRUG_CLASS_MEMBERS.get('laxative', [])
                opioid_antag = GI_DRUG_CLASS_MEMBERS.get('opioid_antagonist', [])
                if any(lax.lower() in drug_lower for lax in laxatives):
                    return ConfidenceTier.HIGH  # Drug class rescue
                if any(oa.lower() in drug_lower for oa in opioid_antag):
                    return ConfidenceTier.HIGH  # Drug class rescue
                # Opioid-induced specifically needs opioid antagonists
                if 'opioid' in disease_lower:
                    if any(oa.lower() in drug_lower for oa in opioid_antag):
                        return ConfidenceTier.GOLDEN  # Higher precision for specific match

            # Liver/hepatic diseases (3 failures) -> bile acid agents
            if any(kw in disease_lower for kw in ['cholangitis', 'cholestasis', 'hepatic']):
                bile_acid = GI_DRUG_CLASS_MEMBERS.get('bile_acid_agent', [])
                ammonia = GI_DRUG_CLASS_MEMBERS.get('ammonia_reducer', [])
                if any(ba.lower() in drug_lower for ba in bile_acid):
                    return ConfidenceTier.HIGH  # Drug class rescue
                # Hepatic encephalopathy specifically
                if 'encephalopathy' in disease_lower:
                    if any(am.lower() in drug_lower for am in ammonia):
                        return ConfidenceTier.GOLDEN  # Specific match

            # Ulcer/reflux diseases (1 failure) -> PPIs
            if any(kw in disease_lower for kw in ['ulcer', 'reflux', 'gerd']):
                ppis = GI_DRUG_CLASS_MEMBERS.get('ppi', [])
                h2_blockers = GI_DRUG_CLASS_MEMBERS.get('h2_blocker', [])
                cytoprotective = GI_DRUG_CLASS_MEMBERS.get('cytoprotective', [])
                if any(ppi.lower() in drug_lower for ppi in ppis):
                    return ConfidenceTier.HIGH  # Drug class rescue
                if any(h2.lower() in drug_lower for h2 in h2_blockers):
                    return ConfidenceTier.HIGH  # Drug class rescue
                if any(cp.lower() in drug_lower for cp in cytoprotective):
                    return ConfidenceTier.HIGH  # Drug class rescue

            # IBS -> antispasmodics + laxatives
            if 'irritable bowel' in disease_lower:
                antispasmodics = GI_DRUG_CLASS_MEMBERS.get('antispasmodic', [])
                laxatives = GI_DRUG_CLASS_MEMBERS.get('laxative', [])
                if any(asp.lower() in drug_lower for asp in antispasmodics):
                    return ConfidenceTier.HIGH  # Drug class rescue
                if any(lax.lower() in drug_lower for lax in laxatives):
                    return ConfidenceTier.HIGH  # Drug class rescue

        return None

    def _get_atc_level4(self, drug_name: str) -> Set[str]:
        """Get ATC level 4 codes (5 characters) for a drug name.

        h189: Used for systematic drug class identification.
        Returns set of ATC L4 codes like {'H02AB', 'L04AX'}.
        """
        try:
            mapper = _get_atc_mapper()
            full_codes = mapper.get_atc_codes(drug_name)
            return set(code[:5] for code in full_codes if len(code) >= 5)
        except Exception:
            return set()

    def _get_neurological_drug_classes(self, disease_name: str) -> List[str]:
        """Get appropriate drug classes for a neurological disease subtype."""
        disease_lower = disease_name.lower()
        matching_classes = []

        for disease_key, drug_classes in NEUROLOGICAL_DISEASE_DRUG_CLASSES.items():
            if disease_key in disease_lower:
                matching_classes.extend(drug_classes)

        return list(set(matching_classes))

    def _is_neurological_class_match(self, drug_name_lower: str, disease_name: str) -> bool:
        """Check if a drug matches appropriate class for a neurological disease."""
        drug_classes = self._get_neurological_drug_classes(disease_name)

        for drug_class in drug_classes:
            members = NEUROLOGICAL_DRUG_CLASS_MEMBERS.get(drug_class, [])
            for member in members:
                if member.lower() in drug_name_lower:
                    return True
        return False

    def _get_class_matched_drugs(self, disease_name: str) -> List[Tuple[str, str, str]]:
        """
        Get drugs from appropriate classes for a neurological disease.

        h173: Drug-class prediction for neurological diseases.
        Returns list of (drug_id, drug_name, drug_class) tuples sorted by training frequency.
        """
        drug_classes = self._get_neurological_drug_classes(disease_name)
        if not drug_classes:
            return []

        matched_drugs: List[Tuple[str, str, str, int]] = []  # (id, name, class, freq)

        for drug_class in drug_classes:
            members = NEUROLOGICAL_DRUG_CLASS_MEMBERS.get(drug_class, [])
            for member in members:
                # Find drug ID for this drug name
                member_lower = member.lower()
                if member_lower in self.name_to_drug_id:
                    drug_id = self.name_to_drug_id[member_lower]
                    if drug_id in self.embeddings:  # Only include drugs in DRKG
                        freq = self.drug_train_freq.get(drug_id, 0)
                        matched_drugs.append((drug_id, member, drug_class, freq))

        # Sort by training frequency (higher frequency = more evidence)
        matched_drugs.sort(key=lambda x: x[3], reverse=True)

        # Return (drug_id, drug_name, drug_class) without freq
        return [(d[0], d[1], d[2]) for d in matched_drugs]

    def _supplement_neurological_predictions(
        self,
        disease_name: str,
        disease_id: str,
        disease_tier: int,
        category: str,
        existing_predictions: List[DrugPrediction],
        max_knn_score: float,
        top_n: int,
        include_filtered: bool,
    ) -> List[DrugPrediction]:
        """
        h173: Supplement kNN predictions with drug-class-matched drugs for neurological diseases.

        h171 showed drug-class prediction achieves 60.4% coverage vs 18% for kNN on neurological.
        This method injects drugs from appropriate classes (anticonvulsants for epilepsy,
        dopaminergics for Parkinson's, etc.) that aren't already in kNN results.

        Args:
            disease_name: Name of the neurological disease
            disease_id: DRKG disease ID
            disease_tier: Disease tier (always 3 for neurological)
            category: Disease category (always 'neurological')
            existing_predictions: Predictions already generated by kNN
            max_knn_score: Maximum kNN score (for normalization)
            top_n: Maximum predictions to return
            include_filtered: Whether to include FILTER tier predictions

        Returns:
            List of predictions with class-matched drugs injected
        """
        # Get drugs already predicted by kNN
        existing_drug_ids = {p.drug_id for p in existing_predictions}

        # Get class-matched drugs not in kNN results
        class_matched = self._get_class_matched_drugs(disease_name)
        missing_drugs = [(d_id, d_name, d_class) for d_id, d_name, d_class in class_matched
                         if d_id not in existing_drug_ids]

        if not missing_drugs:
            return existing_predictions

        # Calculate starting rank for injected drugs
        # They go after HIGH tier drugs but before MEDIUM tier from kNN
        # Find the position after the last HIGH/GOLDEN tier prediction
        high_tier_count = sum(1 for p in existing_predictions
                             if p.confidence_tier in [ConfidenceTier.GOLDEN, ConfidenceTier.HIGH])

        # Inject at position after HIGH tier (but use original rank in display)
        supplemented = existing_predictions.copy()

        for drug_id, drug_name, drug_class in missing_drugs:
            # Stop if we've reached top_n
            if len(supplemented) >= top_n:
                break

            train_freq = self.drug_train_freq.get(drug_id, 0)
            mech_support = self._compute_mechanism_support(drug_id, disease_id)
            has_targets = drug_id in self.drug_targets and len(self.drug_targets[drug_id]) > 0

            # Assign tier - these are class-matched so they get HIGH tier (60% coverage)
            # Use existing tier assignment but mark as rescued
            tier, _, _ = self._assign_confidence_tier(
                rank=high_tier_count + 1,  # Conservative rank estimate
                train_frequency=train_freq,
                mechanism_support=mech_support,
                has_targets=has_targets,
                disease_tier=disease_tier,
                category=category,
                drug_name=drug_name,
                disease_name=disease_name,
                drug_id=drug_id,
            )

            # h395: Cap class-injected drugs at MEDIUM (was HIGH+)
            # h395 found 0% precision for class_injected at HIGH tier
            # Class-matched drugs get at most MEDIUM tier (26.1% precision at MEDIUM)
            if tier in [ConfidenceTier.GOLDEN, ConfidenceTier.HIGH]:
                tier = ConfidenceTier.MEDIUM
            if tier in [ConfidenceTier.LOW, ConfidenceTier.FILTER]:
                tier = ConfidenceTier.MEDIUM

            if not include_filtered and tier == ConfidenceTier.FILTER:
                continue

            # Use a synthetic score based on training frequency
            # (lower than kNN max since these weren't found by kNN)
            synthetic_score = max_knn_score * 0.5 * (1 + train_freq / 100)
            norm_score = synthetic_score / max_knn_score if max_knn_score > 0 else 0.5

            pred = DrugPrediction(
                drug_name=drug_name,
                drug_id=drug_id,
                rank=len(supplemented) + 1,  # Append to end
                knn_score=synthetic_score,
                norm_score=norm_score,
                confidence_tier=tier,
                train_frequency=train_freq,
                mechanism_support=mech_support,
                has_targets=has_targets,
                category=category,
                disease_tier=disease_tier,
                category_rescue_applied=True,  # Mark as class-based prediction
                category_specific_tier="class_injected",  # Special marker
                rank_bucket_precision=get_rank_bucket_precision(tier.value, len(supplemented) + 1),
                category_holdout_precision=get_category_holdout_precision(category, tier.value),
            )
            supplemented.append(pred)

        # Re-sort by tier priority, then by score within tier
        tier_priority = {
            ConfidenceTier.GOLDEN: 0,
            ConfidenceTier.HIGH: 1,
            ConfidenceTier.MEDIUM: 2,
            ConfidenceTier.LOW: 3,
            ConfidenceTier.FILTER: 4,
        }
        # h501: Added drug_id as tiebreaker for deterministic ordering
        supplemented.sort(key=lambda p: (tier_priority.get(p.confidence_tier, 5), -p.knn_score, p.drug_id))

        # Re-assign ranks
        for i, pred in enumerate(supplemented, 1):
            pred.rank = i

        return supplemented[:top_n]

    def _get_gi_drug_classes(self, disease_name: str) -> List[str]:
        """Get appropriate drug classes for a GI disease subtype.

        h380: Maps GI diseases to appropriate drug classes.
        """
        disease_lower = disease_name.lower()
        matching_classes = []

        for disease_key, drug_classes in GI_DISEASE_DRUG_CLASSES.items():
            if disease_key in disease_lower:
                matching_classes.extend(drug_classes)

        return list(set(matching_classes))

    def _get_gi_class_matched_drugs(self, disease_name: str) -> List[Tuple[str, str, str]]:
        """Get drugs from appropriate classes for a GI disease.

        h380: Drug-class prediction for gastrointestinal diseases.
        Returns list of (drug_id, drug_name, drug_class) tuples sorted by training frequency.
        """
        drug_classes = self._get_gi_drug_classes(disease_name)
        if not drug_classes:
            return []

        matched_drugs: List[Tuple[str, str, str, int]] = []  # (id, name, class, freq)

        for drug_class in drug_classes:
            members = GI_DRUG_CLASS_MEMBERS.get(drug_class, [])
            for member in members:
                # Find drug ID for this drug name
                member_lower = member.lower()
                if member_lower in self.name_to_drug_id:
                    drug_id = self.name_to_drug_id[member_lower]
                    if drug_id in self.embeddings:  # Only include drugs in DRKG
                        freq = self.drug_train_freq.get(drug_id, 0)
                        matched_drugs.append((drug_id, member, drug_class, freq))

        # Sort by training frequency (higher frequency = more evidence)
        matched_drugs.sort(key=lambda x: x[3], reverse=True)

        # Return (drug_id, drug_name, drug_class) without freq
        return [(d[0], d[1], d[2]) for d in matched_drugs]

    def _supplement_gi_predictions(
        self,
        disease_name: str,
        disease_id: str,
        disease_tier: int,
        category: str,
        existing_predictions: List[DrugPrediction],
        max_knn_score: float,
        top_n: int,
        include_filtered: bool,
    ) -> List[DrugPrediction]:
        """
        h380: Supplement kNN predictions with drug-class-matched drugs for GI diseases.

        GI category has 42.9% R@30 because kNN finds neighbors from different categories
        (infectious, neurological) that have different drug needs. This method injects
        drugs from appropriate classes (laxatives for constipation, PPIs for ulcer, etc.)
        that aren't already in kNN results.
        """
        # Get drugs already predicted by kNN
        existing_drug_ids = {p.drug_id for p in existing_predictions}

        # Get class-matched drugs not in kNN results
        class_matched = self._get_gi_class_matched_drugs(disease_name)
        missing_drugs = [(d_id, d_name, d_class) for d_id, d_name, d_class in class_matched
                         if d_id not in existing_drug_ids]

        if not missing_drugs:
            return existing_predictions

        # Calculate starting position for injected drugs
        high_tier_count = sum(1 for p in existing_predictions
                             if p.confidence_tier in [ConfidenceTier.GOLDEN, ConfidenceTier.HIGH])

        supplemented = existing_predictions.copy()

        for drug_id, drug_name, drug_class in missing_drugs:
            # Stop if we've reached top_n
            if len(supplemented) >= top_n:
                break

            train_freq = self.drug_train_freq.get(drug_id, 0)
            mech_support = self._compute_mechanism_support(drug_id, disease_id)
            has_targets = drug_id in self.drug_targets and len(self.drug_targets[drug_id]) > 0

            # Assign tier through normal process but mark as rescued
            tier, _, _ = self._assign_confidence_tier(
                rank=high_tier_count + 1,
                train_frequency=train_freq,
                mechanism_support=mech_support,
                has_targets=has_targets,
                disease_tier=disease_tier,
                category=category,
                drug_name=drug_name,
                disease_name=disease_name,
                drug_id=drug_id,
            )

            # h395: Cap GI class-injected drugs at MEDIUM (was HIGH+)
            # h395 found 0% precision for gi_class_injected at HIGH tier
            if tier in [ConfidenceTier.GOLDEN, ConfidenceTier.HIGH]:
                tier = ConfidenceTier.MEDIUM
            if tier in [ConfidenceTier.LOW, ConfidenceTier.FILTER]:
                tier = ConfidenceTier.MEDIUM

            if not include_filtered and tier == ConfidenceTier.FILTER:
                continue

            # Create synthetic score (lower than kNN max since these weren't found by kNN)
            synthetic_score = max_knn_score * 0.5 * (1 + train_freq / 100)
            norm_score = synthetic_score / max_knn_score if max_knn_score > 0 else 0.5

            pred = DrugPrediction(
                drug_name=drug_name,
                drug_id=drug_id,
                rank=len(supplemented) + 1,
                knn_score=synthetic_score,
                norm_score=norm_score,
                confidence_tier=tier,
                train_frequency=train_freq,
                mechanism_support=mech_support,
                has_targets=has_targets,
                category=category,
                disease_tier=disease_tier,
                category_rescue_applied=True,
                category_specific_tier="gi_class_injected",
                rank_bucket_precision=get_rank_bucket_precision(tier.value, len(supplemented) + 1),
                category_holdout_precision=get_category_holdout_precision(category, tier.value),
            )
            supplemented.append(pred)

        # Re-sort by tier priority, then by score within tier
        tier_priority = {
            ConfidenceTier.GOLDEN: 0,
            ConfidenceTier.HIGH: 1,
            ConfidenceTier.MEDIUM: 2,
            ConfidenceTier.LOW: 3,
            ConfidenceTier.FILTER: 4,
        }
        # h501: Added drug_id as tiebreaker for deterministic ordering
        supplemented.sort(key=lambda p: (tier_priority.get(p.confidence_tier, 5), -p.knn_score, p.drug_id))

        # Re-assign ranks
        for i, pred in enumerate(supplemented, 1):
            pred.rank = i

        return supplemented[:top_n]

    def find_disease_id(self, disease_name: str) -> Optional[str]:
        """Find the DRKG disease ID for a disease name."""
        # Try exact match first
        disease_lower = disease_name.lower()
        if disease_lower in self.mesh_mappings:
            return self.mesh_mappings[disease_lower]

        # Try fuzzy matching
        sys.path.insert(0, str(self.data_dir / "src"))
        from disease_name_matcher import DiseaseMatcher, load_mesh_mappings

        fuzzy_mappings = load_mesh_mappings()
        matcher = DiseaseMatcher(fuzzy_mappings)
        return matcher.get_mesh_id(disease_name)

    def predict(
        self,
        disease_name: str,
        k: int = 20,
        top_n: int = 30,
        include_filtered: bool = False,
    ) -> PredictionResult:
        """
        Generate drug predictions for a disease.

        Args:
            disease_name: Name of the disease
            k: Number of nearest neighbors for kNN (default 20 from h39)
            top_n: Number of top predictions to return
            include_filtered: If True, include FILTER tier predictions

        Returns:
            PredictionResult with ranked predictions and confidence tiers
        """
        # Find disease ID
        disease_id = self.find_disease_id(disease_name)

        # Categorize disease
        category = self.categorize_disease(disease_name)
        disease_tier = self.get_category_tier(category)

        # Check if disease is in embeddings
        coverage_warning = None
        if disease_id is None or disease_id not in self.embeddings:
            coverage_warning = f"Disease '{disease_name}' not found in DRKG. Using name-based matching only."
            # Fall back to finding similar diseases by name
            disease_id = None

        predictions = []

        if disease_id and disease_id in self.embeddings:
            # Run kNN (h39 method)
            test_emb = self.embeddings[disease_id].reshape(1, -1)
            sims = cosine_similarity(test_emb, self.train_embeddings)[0]

            # h170: Apply selective category boost for isolated categories
            # Boosts same-category neighbors by (1 + alpha) for categories that benefit
            if category in SELECTIVE_BOOST_CATEGORIES:
                boosted_sims = sims.copy()
                for i, train_d in enumerate(self.train_diseases):
                    if self.train_disease_categories.get(train_d) == category:
                        boosted_sims[i] *= (1 + SELECTIVE_BOOST_ALPHA)
                top_k_idx = np.argsort(boosted_sims)[-k:]
                working_sims = boosted_sims
            else:
                top_k_idx = np.argsort(sims)[-k:]
                working_sims = sims

            # Aggregate drug scores from neighbors
            drug_scores: Dict[str, float] = defaultdict(float)
            for idx in top_k_idx:
                neighbor_disease = self.train_diseases[idx]
                neighbor_sim = working_sims[idx]
                for drug_id in self.ground_truth[neighbor_disease]:
                    if drug_id in self.embeddings:
                        drug_scores[drug_id] += neighbor_sim

            if not drug_scores:
                coverage_warning = "No drugs found in kNN neighborhood."
            else:
                # h374: Apply MinRank ensemble for cancer/neuro/metabolic categories
                # For these categories, combine kNN and target overlap via min-rank fusion
                use_minrank = category in MINRANK_ENSEMBLE_CATEGORIES

                if use_minrank:
                    # Get target scores for MinRank fusion
                    target_scores = self._get_target_scores(disease_id)

                    if target_scores:
                        # Apply MinRank fusion
                        fused = self._minrank_fusion(drug_scores, target_scores)
                        # Sort by min_rank first, then by combined_score for ties
                        # h501: Added drug_id as final tiebreaker for determinism
                        sorted_drugs = sorted(
                            fused.items(),
                            key=lambda x: (x[1][1], -x[1][0], x[0])  # (min_rank asc, score desc, drug_id asc)
                        )
                        # Extract (drug_id, combined_score) for top_n
                        sorted_drugs = [(drug_id, info[0]) for drug_id, info in sorted_drugs[:top_n]]
                        max_score = 1.0  # Already normalized
                    else:
                        # No target data - fall back to kNN only
                        # h501: Use drug_id as tiebreaker for deterministic ranking
                        sorted_drugs = sorted(drug_scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]
                        max_score = sorted_drugs[0][1] if sorted_drugs else 1.0
                else:
                    # Standard kNN ranking
                    # h501: Use drug_id as tiebreaker for deterministic ranking across processes
                    sorted_drugs = sorted(drug_scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]
                    max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

                # h405/h439: Compute TransE top-30 once per disease for
                # consilience annotation. TransE agreement = +13.6pp
                # holdout-validated lift for MEDIUM tier.
                transe_top30 = self._get_transe_top_n(
                    disease_id, self.all_gt_drugs_with_embeddings, n=30
                )

                for rank, (drug_id, score) in enumerate(sorted_drugs, 1):
                    if use_minrank:
                        # For MinRank: score is already normalized combined score
                        norm_score = score
                        # Get original kNN score for knn_score field
                        knn_score = drug_scores.get(drug_id, 0.0)
                    else:
                        norm_score = score / max_score if max_score > 0 else 0
                        knn_score = score

                    train_freq = self.drug_train_freq.get(drug_id, 0)
                    mech_support = self._compute_mechanism_support(drug_id, disease_id)
                    has_targets = drug_id in self.drug_targets and len(self.drug_targets[drug_id]) > 0
                    drug_name = self.drug_id_to_name.get(drug_id, drug_id)

                    tier, rescue_applied, cat_specific = self._assign_confidence_tier(
                        rank, train_freq, mech_support, has_targets, disease_tier, category,
                        drug_name, disease_name, drug_id
                    )

                    # h388: Target overlap tier promotion (no rank change)
                    # Guard: HIGH→GOLDEN only for rules with demonstrated precision
                    target_overlap = self._get_target_overlap_count(drug_id, disease_id)
                    if (tier == ConfidenceTier.HIGH
                            and target_overlap >= TARGET_OVERLAP_PROMOTE_HIGH_TO_GOLDEN
                            and cat_specific in TARGET_OVERLAP_GOLDEN_ELIGIBLE_RULES):
                        tier = ConfidenceTier.GOLDEN
                    elif (tier == ConfidenceTier.LOW
                            and target_overlap >= TARGET_OVERLAP_PROMOTE_LOW_TO_MEDIUM
                            # h462: Block LOW→MEDIUM for categories with poor MEDIUM holdout
                            # h485: Block cancer (cross-type overlap=0.3% holdout, n=197)
                            # h505: Block cardiovascular (13.6% holdout < LOW avg 14.8%, n=13/seed)
                            # h553: Block hematological (25% ± 43% holdout, n=1.2/seed — too tiny, default=0%)
                            # h647: Block metabolic (22.3% ± 21% holdout, n=6.4/seed — 37 preds leaking)
                            and category not in {'gastrointestinal', 'immunological', 'reproductive', 'neurological', 'cancer', 'cardiovascular', 'hematological', 'metabolic'}
                            # h488: Block rescue of incoherent demotions (3.6% holdout)
                            and cat_specific != 'incoherent_demotion'
                            # h560: Block rescue of antimicrobial-pathogen mismatches (0% holdout)
                            and cat_specific != 'antimicrobial_pathogen_mismatch'
                            # h649: Block rescue of hierarchy-demoted pneumonia (16.7% holdout)
                            and cat_specific != 'infectious_hierarchy_pneumonia'
                            # h677: Block rescue of LA procedural demotions (bupivacaine/lidocaine)
                            # LA drugs demoted by h540 should not be rescued via target overlap
                            and cat_specific != 'local_anesthetic_procedural'):
                        tier = ConfidenceTier.MEDIUM
                        cat_specific = cat_specific or 'target_overlap_promotion'

                    # h405/h439: TransE consilience flag
                    # MEDIUM + TransE top-30 = 34.7% holdout (+13.6pp)
                    # Not promoted to HIGH (37.4% full-data < HIGH 50.8%)
                    # but flagged for downstream prioritization.
                    in_transe_top30 = drug_id in transe_top30

                    # h520: Corticosteroid SOC promotion for non-hematological categories
                    # Corticosteroid MEDIUM predictions in autoimmune/dermatological/respiratory/ophthalmic
                    # have 50.1% holdout precision (p=0.0065), comparable to HIGH (51.5%).
                    # Hematological excluded (19.1% holdout, below MEDIUM avg).
                    if (tier == ConfidenceTier.MEDIUM
                            and category in _CORTICOSTEROID_SOC_PROMOTE_CATEGORIES
                            and drug_name.lower() in _CORTICOSTEROID_LOWER):
                        tier = ConfidenceTier.HIGH
                        cat_specific = 'corticosteroid_soc_promotion'

                    # h522: Hematological corticosteroid demotion MEDIUM→LOW
                    # Hematological corticosteroid MEDIUM = 19.1% holdout (below MEDIUM avg 31.1%)
                    # h625: Immune-mediated diseases exempted (48.4% ± 28.8% holdout)
                    # Non-immune (genetic/structural): 3.8% ± 4.0% → stays LOW
                    if (tier == ConfidenceTier.MEDIUM
                            and category == 'hematological'
                            and drug_name.lower() in _CORTICOSTEROID_LOWER
                            and not self._is_immune_mediated_hematological(disease_name)):
                        tier = ConfidenceTier.LOW
                        cat_specific = 'hematological_corticosteroid_demotion'

                    # h559: Corticosteroid→TB hierarchy demotion HIGH→MEDIUM
                    # CS→TB HIGH = 33.3% full-data (6/18) vs non-CS infectious HIGH = 76.9%
                    # Dexamethasone for TB meningitis is WHO-recommended, but other CS→TB are weak
                    # 33.3% is at MEDIUM level, not HIGH. n=18 too small for holdout.
                    if (tier == ConfidenceTier.HIGH
                            and category == 'infectious'
                            and drug_name.lower() in _CORTICOSTEROID_LOWER
                            and cat_specific and 'tuberculosis' in cat_specific):
                        tier = ConfidenceTier.MEDIUM
                        cat_specific = 'infectious_cs_tb_demotion'

                    # h557: Corticosteroid→infectious demotion MEDIUM→LOW
                    # CS→infectious MEDIUM = 2.1% ± 2.5% holdout (5-seed, 11.6/seed)
                    # Even medically valid CS uses (ABPA, zoster, leprosy) = 2.9% holdout
                    # Non-CS infectious MEDIUM = 18.7% ± 5.2% — 16.6pp gap
                    # CS predicted for infections due to KG co-occurrence, not therapeutic use
                    # h559: Exclude CS→TB hierarchy demotions (stay at MEDIUM — strongest evidence)
                    if (tier == ConfidenceTier.MEDIUM
                            and category == 'infectious'
                            and drug_name.lower() in _CORTICOSTEROID_LOWER
                            and cat_specific != 'infectious_cs_tb_demotion'):
                        tier = ConfidenceTier.LOW
                        cat_specific = 'infectious_corticosteroid_demotion'

                    # h630: TransE MEDIUM → HIGH promotion (non-CS, strict criteria)
                    # h629 validated: TransE+mechanism = 59.4% ± 13.9% holdout,
                    # TransE+rank<=5 = 64.9% ± 12.4% — both above HIGH (52.8%).
                    # Non-CS TransE MEDIUM = 49.1% holdout (per-seed: 50,50,42,57,50%).
                    # Expanded GT resolves h439 blocker (34.7% → 56.5%).
                    # Strict criteria: require mechanism OR rank<=5 to ensure above HIGH.
                    if (tier == ConfidenceTier.MEDIUM
                            and in_transe_top30
                            and drug_name.lower() not in _CORTICOSTEROID_LOWER
                            and (mech_support or rank <= 5)):
                        tier = ConfidenceTier.HIGH
                        cat_specific = 'transe_medium_promotion'

                    # h374: Mark predictions from MinRank ensemble
                    if use_minrank and cat_specific is None:
                        cat_specific = 'minrank_ensemble'

                    pred = DrugPrediction(
                        drug_name=drug_name,
                        drug_id=drug_id,
                        rank=rank,
                        knn_score=knn_score,
                        norm_score=norm_score,
                        confidence_tier=tier,
                        train_frequency=train_freq,
                        mechanism_support=mech_support,
                        has_targets=has_targets,
                        category=category,
                        disease_tier=disease_tier,
                        category_rescue_applied=rescue_applied,
                        category_specific_tier=cat_specific,
                        transe_consilience=in_transe_top30,
                        rank_bucket_precision=get_rank_bucket_precision(tier.value, rank),
                        category_holdout_precision=get_category_holdout_precision(category, tier.value),
                    )

                    if include_filtered or tier != ConfidenceTier.FILTER:
                        predictions.append(pred)

            # h173: Supplement with drug-class predictions for neurological diseases
            # When kNN has low coverage, inject drugs from appropriate classes
            if category == 'neurological':
                # Get max kNN score for normalization (use 1.0 as fallback)
                knn_max = max(drug_scores.values()) if drug_scores else 1.0
                predictions = self._supplement_neurological_predictions(
                    disease_name, disease_id, disease_tier, category,
                    predictions, knn_max, top_n, include_filtered
                )

            # h380: Supplement with drug-class predictions for GI diseases
            # GI has 42.9% R@30 because kNN finds wrong drug classes
            if category == 'gastrointestinal':
                knn_max = max(drug_scores.values()) if drug_scores else 1.0
                predictions = self._supplement_gi_predictions(
                    disease_name, disease_id, disease_tier, category,
                    predictions, knn_max, top_n, include_filtered
                )

            # h326: Demote broad-class-isolated predictions (1.9% precision when alone vs 12.7% with classmates)
            # For drugs from broad classes (anesthetics, steroids, TNFi, NSAIDs), check if any classmates
            # are also predicted. If not, demote HIGH→LOW, MEDIUM→LOW (FILTER would drop too many)
            all_drug_names = {p.drug_name for p in predictions}
            for pred in predictions:
                if self._is_broad_class_isolated(pred.drug_name, disease_name, all_drug_names):
                    # Demote by 2 tiers (HIGH→LOW, MEDIUM→LOW) - h326 finding: 0% precision for HIGH
                    if pred.confidence_tier == ConfidenceTier.HIGH:
                        pred.confidence_tier = ConfidenceTier.LOW
                        pred.category_specific_tier = 'broad_class_isolated'
                    elif pred.confidence_tier == ConfidenceTier.MEDIUM:
                        pred.confidence_tier = ConfidenceTier.LOW
                        pred.category_specific_tier = 'broad_class_isolated'

        return PredictionResult(
            disease_name=disease_name,
            disease_id=disease_id,
            category=category,
            disease_tier=disease_tier,
            predictions=predictions,
            neighbors_used=k,
            coverage_warning=coverage_warning,
        )

    def batch_predict(
        self,
        diseases: List[str],
        **kwargs,
    ) -> Dict[str, PredictionResult]:
        """Generate predictions for multiple diseases."""
        results = {}
        for disease in diseases:
            results[disease] = self.predict(disease, **kwargs)
        return results


def print_predictions(result: PredictionResult) -> None:
    """Pretty-print prediction results."""
    print("=" * 80)
    print(f"PREDICTIONS FOR: {result.disease_name}")
    print("=" * 80)
    print(f"Category: {result.category} (Tier {result.disease_tier})")
    print(f"Disease ID: {result.disease_id or 'Not found'}")
    if result.coverage_warning:
        print(f"Warning: {result.coverage_warning}")
    print()

    # Print by tier
    tier_order = [ConfidenceTier.GOLDEN, ConfidenceTier.HIGH, ConfidenceTier.MEDIUM, ConfidenceTier.LOW]
    tier_emoji = {
        ConfidenceTier.GOLDEN: "[GOLD]",
        ConfidenceTier.HIGH: "[HIGH]",
        ConfidenceTier.MEDIUM: "[MED] ",
        ConfidenceTier.LOW: "[LOW] ",
    }

    for tier in tier_order:
        preds = result.get_by_tier(tier)
        if preds:
            # h167: Show category-specific precision instead of generic tier precision
            cat_precision = get_category_precision(result.category, tier.value)
            print(f"\n{tier_emoji[tier]} {tier.value} CONFIDENCE (precision for {result.category}: ~{cat_precision:.0f}%)")
            print("-" * 70)
            print(f"{'Rank':<6} {'Drug':<35} {'Score':<8} {'Freq':<6} {'Mech':<6}")
            for p in preds[:10]:  # Show top 10 per tier
                mech = "Y" if p.mechanism_support else "-"
                rescue = " [rescued]" if p.category_rescue_applied else ""
                print(f"{p.rank:<6} {p.drug_name[:33]:<35} {p.norm_score:<8.3f} {p.train_frequency:<6} {mech:<6}{rescue}")

    # Summary
    print("\n" + "=" * 80)
    summary = result.summary()
    print(f"Total predictions: {summary['total_predictions']}")
    print(f"By tier: {summary['by_tier']}")
    # h167: Show category-specific precision summary
    print(f"Category precision (for {result.category}): {summary['category_precision_by_tier']}")


def main():
    """CLI interface for the predictor."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Drug Repurposing Predictions using kNN Collaborative Filtering"
    )
    parser.add_argument("disease", nargs="?", help="Disease name to predict for")
    parser.add_argument("--disease", "-d", dest="disease_flag", help="Disease name (alternative)")
    parser.add_argument("--top-k", "-k", type=int, default=30, help="Number of predictions (default: 30)")
    parser.add_argument("--neighbors", "-n", type=int, default=20, help="kNN neighbors (default: 20)")
    parser.add_argument("--include-filtered", action="store_true", help="Include FILTER tier predictions")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    disease = args.disease or args.disease_flag
    if not disease:
        parser.print_help()
        print("\nExample: python -m src.production_predictor 'rheumatoid arthritis'")
        sys.exit(1)

    print("Loading predictor...")
    predictor = DrugRepurposingPredictor()

    print(f"Generating predictions for: {disease}")
    result = predictor.predict(
        disease,
        k=args.neighbors,
        top_n=args.top_k,
        include_filtered=args.include_filtered,
    )

    if args.json:
        output = {
            'summary': result.summary(),
            'predictions': [p.to_dict() for p in result.predictions],
        }
        print(json.dumps(output, indent=2))
    else:
        print_predictions(result)


if __name__ == "__main__":
    main()
