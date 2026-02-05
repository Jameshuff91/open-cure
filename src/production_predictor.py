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

USAGE:
    # Get predictions for a disease
    from production_predictor import DrugRepurposingPredictor
    predictor = DrugRepurposingPredictor()
    results = predictor.predict("rheumatoid arthritis")

    # CLI usage
    python -m src.production_predictor "rheumatoid arthritis"
    python -m src.production_predictor --disease "type 2 diabetes" --top-k 30

TIER SYSTEM (h135 validated, 9.1x separation):
- GOLDEN (57.7%): Tier1 category + freq>=10 + mechanism
- HIGH (20.9%):   freq>=15 + mechanism OR rank<=5 + freq>=10 + mechanism
- MEDIUM (14.3%): freq>=5 + mechanism OR freq>=10
- LOW (6.4%):     All else passing filter
- FILTER (3.2%):  rank>20 OR no_targets OR (freq<=2 AND no_mechanism)
"""

import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
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
    # Format: (category, tier) -> precision percentage
    # From h165 analysis: 5 seeds, 101,939 predictions, 2,455 diseases
    # GOLDEN/HIGH values from h136/h144/h150/h154/h157 rescue criteria validation
    #
    # GOLDEN tier values (from rescue criteria validation):
    ("autoimmune", "GOLDEN"): 75.4,   # h157: DMARD + rank<=10
    ("infectious", "GOLDEN"): 55.6,   # h136: rank<=10 + freq>=15 + mech
    ("metabolic", "GOLDEN"): 60.0,    # h144: statin + rank<=10
    ("ophthalmic", "GOLDEN"): 62.5,   # h150: antibiotic + rank<=15
    ("dermatological", "GOLDEN"): 63.6,  # h150: topical_steroid + rank<=5
    # HIGH tier values (from rescue criteria validation):
    ("cardiovascular", "HIGH"): 38.2, # h136: rank<=5 + mech OR h154/h266: beta_blocker + rank<=10 (42.1%)
    ("respiratory", "HIGH"): 35.0,    # h136: rank<=10 + freq>=15 + mech
    ("cancer", "GOLDEN"): 55.0,       # h197: colorectal + mAb = 50-60% precision
    ("cancer", "HIGH"): 40.0,         # h150: taxane + rank<=5
    ("ophthalmic", "HIGH"): 48.0,     # h150: steroid + rank<=15
    ("hematological", "HIGH"): 48.6,  # h150: corticosteroid + rank<=10
    #
    # MEDIUM tier values (from h165):
    ("psychiatric", "MEDIUM"): 85.0,
    ("psychiatric", "LOW"): None,  # No data
    ("psychiatric", "FILTER"): 90.0,  # Very high!
    ("autoimmune", "MEDIUM"): 77.8,
    ("autoimmune", "LOW"): 46.2,
    ("autoimmune", "FILTER"): 45.9,
    ("respiratory", "MEDIUM"): 54.2,
    ("respiratory", "LOW"): 11.1,
    ("respiratory", "FILTER"): 35.7,
    ("dermatological", "MEDIUM"): 49.0,
    ("dermatological", "LOW"): 28.6,
    ("dermatological", "FILTER"): 17.5,
    ("metabolic", "MEDIUM"): 47.6,
    ("metabolic", "LOW"): 35.9,
    ("metabolic", "FILTER"): 21.6,
    ("cancer", "MEDIUM"): 45.7,
    ("cancer", "LOW"): 12.3,
    ("cancer", "FILTER"): 30.6,
    ("gastrointestinal", "MEDIUM"): 41.3,
    ("gastrointestinal", "LOW"): 22.2,
    ("gastrointestinal", "FILTER"): 18.3,
    ("hematological", "MEDIUM"): 37.5,
    ("hematological", "LOW"): 0.0,
    ("hematological", "FILTER"): 13.6,
    ("infectious", "MEDIUM"): 38.4,
    ("infectious", "LOW"): 6.2,
    ("infectious", "FILTER"): 19.0,
    ("ophthalmic", "MEDIUM"): 36.1,
    ("ophthalmic", "LOW"): 67.9,  # Higher than MEDIUM!
    ("ophthalmic", "FILTER"): 19.9,
    ("cardiovascular", "MEDIUM"): 36.4,
    ("cardiovascular", "LOW"): 17.6,
    ("cardiovascular", "FILTER"): 26.4,
    ("neurological", "GOLDEN"): 58.8,  # h187: anticonvulsant + rank<=10 + mech
    ("neurological", "MEDIUM"): 26.1,
    ("neurological", "LOW"): 15.0,
    ("neurological", "FILTER"): 12.5,
    ("other", "MEDIUM"): 17.3,
    ("other", "LOW"): 4.9,
    ("other", "FILTER"): 6.2,
    # h169: New categories - using conservative estimates (similar to gastrointestinal)
    # until we have calibration data
    ("renal", "MEDIUM"): 40.0,  # Similar to gastrointestinal
    ("renal", "LOW"): 20.0,
    ("renal", "FILTER"): 15.0,
    ("musculoskeletal", "MEDIUM"): 35.0,  # Conservative estimate
    ("musculoskeletal", "LOW"): 15.0,
    ("musculoskeletal", "FILTER"): 10.0,
    ("immunological", "MEDIUM"): 45.0,  # Similar to autoimmune-related
    ("immunological", "LOW"): 25.0,
    ("immunological", "FILTER"): 20.0,
}

# Default tier-only precision (fallback)
DEFAULT_TIER_PRECISION = {
    "GOLDEN": 57.7,
    "HIGH": 20.9,
    "MEDIUM": 19.3,  # Overall average from h165
    "LOW": 5.7,
    "FILTER": 7.6,
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
ATC_HIGH_PRECISION_DERMATOLOGICAL = {'D07AA', 'D07XA', 'D07AB', 'D07AC', 'D07XB', 'D07XC', 'H02AB'}  # Topical + systemic steroids (66-79%)

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
        'lupus': ['lupus', 'systemic lupus erythematosus', 'discoid lupus', 'lupus nephritis',
                  'membranous lupus nephritis', 'cutaneous lupus', 'sle'],
        'colitis': ['colitis', 'ulcerative colitis', 'chronic ulcerative colitis', 'pediatric ulcerative colitis',
                    'crohns disease', 'crohn disease', 'crohn colitis', 'inflammatory bowel disease'],
        'scleroderma': ['scleroderma', 'systemic sclerosis', 'systemic sclerosis associated interstitial lung disease',
                        'diffuse scleroderma', 'limited scleroderma'],
        'spondylitis': ['ankylosing spondylitis', 'axial spondyloarthritis', 'non-radiographic axial spondyloarthritis'],
    },
    'infectious': {
        'pneumonia': ['pneumonia', 'bronchopneumonia', 'community-acquired pneumonia', 'hospital-acquired pneumonia',
                      'streptococcal pneumonia', 'pneumococcal pneumonia', 'bacterial pneumonia', 'aspiration pneumonia'],
        'hepatitis': ['hepatitis', 'hepatitis b', 'hepatitis c', 'chronic hepatitis', 'viral hepatitis',
                      'hepatitis c genotype 1', 'hepatitis c genotype 2', 'hepatitis c genotype 3'],
        'uti': ['urinary tract infection', 'uti', 'complicated urinary tract infection', 'uncomplicated uti',
                'recurrent uti', 'chronic urinary tract infection', 'pyelonephritis', 'cystitis'],
        'sepsis': ['sepsis', 'bacterial sepsis', 'septicemia', 'blood stream infection', 'severe sepsis', 'septic shock'],
        'skin_infection': ['skin infection', 'cellulitis', 'impetigo', 'wound infection', 'burn infection',
                           'skin and soft tissue infection', 'abscess'],
        'respiratory_infection': ['respiratory infection', 'bronchitis', 'acute bronchitis', 'chronic bronchitis',
                                  'respiratory tract infection', 'upper respiratory infection', 'lower respiratory infection'],
        'tuberculosis': ['tuberculosis', 'tb', 'pulmonary tuberculosis', 'latent tuberculosis', 'multidrug-resistant tuberculosis'],
        'hiv': ['hiv', 'human immunodeficiency virus', 'aids', 'hiv infection', 'hiv-1 infection'],
    },
    'neurological': {
        'epilepsy': ['epilepsy', 'seizure', 'seizure disorder', 'partial seizure', 'generalized seizure',
                     'focal seizure', 'absence seizure', 'tonic-clonic seizure', 'status epilepticus'],
        'parkinsons': ['parkinson', "parkinson's disease", 'parkinsons disease', 'parkinsonism', 'tremor'],
        'alzheimers': ['alzheimer', "alzheimer's disease", 'dementia', 'cognitive impairment', 'memory loss'],
        'migraine': ['migraine', 'headache', 'chronic migraine', 'episodic migraine', 'cluster headache', 'tension headache'],
        'neuropathy': ['neuropathy', 'peripheral neuropathy', 'diabetic neuropathy', 'polyneuropathy', 'nerve damage'],
        'stroke': ['stroke', 'cerebrovascular', 'ischemic stroke', 'hemorrhagic stroke', 'tia', 'transient ischemic attack'],
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
        'pulmonary_fibrosis': ['fibrosis', 'pulmonary fibrosis', 'idiopathic pulmonary fibrosis', 'interstitial lung'],
    },
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

# h170: Selective category boosting (VALIDATED: +2.40pp, p=0.009)
# Only boost same-category neighbors for isolated categories where it helps
# Improves neurological +14.3pp, respiratory +16.8pp, metabolic +13.9pp
# Without hurting infectious (-11.2pp) or other (-4.8pp) that would be hurt by universal boost
SELECTIVE_BOOST_CATEGORIES = {
    'neurological', 'respiratory', 'metabolic', 'renal', 'hematological', 'immunological'
}
SELECTIVE_BOOST_ALPHA = 0.5  # Similarity multiplier: sim * (1 + alpha) for same-category neighbors

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
               'medulloblastoma'],
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
                   'retinoblastoma', 'retinopathy'],
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


def extract_cancer_types(disease_name: str) -> Set[str]:
    """
    h274: Extract cancer types from a disease name.

    Returns set of cancer types (e.g., {'lymphoma', 'leukemia'}).
    """
    disease_lower = disease_name.lower()
    cancer_types = set()

    for cancer_type, keywords in CANCER_TYPE_KEYWORDS.items():
        if any(kw in disease_lower for kw in keywords):
            cancer_types.add(cancer_type)

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
        # Load Node2Vec embeddings
        embeddings_path = self.embeddings_dir / "node2vec_256_named.csv"
        df = pd.read_csv(embeddings_path)
        dim_cols = [c for c in df.columns if c.startswith("dim_")]
        self.embeddings: Dict[str, np.ndarray] = {}
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
            drug_id = self.name_to_drug_id.get(drug.lower())
            if drug_id:
                self.ground_truth[disease_id].add(drug_id)

        self.ground_truth = dict(self.ground_truth)

        # Save to cache
        cache_data = {
            "cache_key": current_key,
            "ground_truth": {k: list(v) for k, v in self.ground_truth.items()},
            "disease_names": self.disease_names,
        }
        with open(cache_path, "w") as f:
            json.dump(cache_data, f)

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
            if any(variant in disease_lower or disease_lower in variant
                   for variant in variants):
                pred_disease_group = group_name
                break

        if pred_disease_group is None:
            return has_category_gt, False, None

        # Check if drug has GT in the same disease group
        same_group_match = pred_disease_group in category_groups

        return has_category_gt, same_group_match, pred_disease_group if same_group_match else None

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
        # h274: For cancer, check cancer type match BEFORE applying rank filter
        # Same-type cancer drugs are highly predictive regardless of rank
        if category == 'cancer' and drug_id:
            has_cancer_gt, same_type_match, _ = self._check_cancer_type_match(drug_id, disease_name)
            if same_type_match:
                # Same cancer type → GOLDEN regardless of rank (100% precision)
                return ConfidenceTier.GOLDEN, True, 'cancer_same_type'
            if not has_cancer_gt:
                # No cancer GT → FILTER (0% precision)
                return ConfidenceTier.FILTER, False, 'cancer_no_gt'
            # Cross-type: continue to standard filtering, will get MEDIUM in _apply_category_rescue

        # FILTER tier (h123 negative signals)
        if rank > 20:
            return ConfidenceTier.FILTER, False, None
        if not has_targets:
            return ConfidenceTier.FILTER, False, None
        if train_frequency <= 2 and not mechanism_support:
            return ConfidenceTier.FILTER, False, None

        # h153: Corticosteroids for metabolic diseases = FILTER (contraindicated)
        # Corticosteroids cause hyperglycemia and worsen diabetes
        if category == 'metabolic':
            drug_lower = drug_name.lower()
            if any(steroid in drug_lower for steroid in CORTICOSTEROID_DRUGS):
                return ConfidenceTier.FILTER, False, None

        # h273/h276: Disease hierarchy matching - if drug treats same disease group, boost tier
        # This indicates the prediction is a subtype refinement (e.g., "psoriasis" → "plaque psoriasis")
        # 2.9x precision improvement overall (8.5% → 24.7%)
        #
        # h276: Category-specific tier assignment based on validated precision:
        # - Autoimmune: 75.9% → GOLDEN
        # - Metabolic: 72.1% → GOLDEN
        # - Neurological: 81.8% → GOLDEN
        # - Cardiovascular: 38.3% → HIGH
        # - Respiratory: 28.6% → HIGH
        HIERARCHY_GOLDEN_CATEGORIES = {'autoimmune', 'metabolic', 'neurological'}

        if category in DISEASE_HIERARCHY_GROUPS and drug_id:
            has_category_gt, same_group_match, matching_group = self._check_disease_hierarchy_match(
                drug_id, disease_name, category
            )
            if same_group_match:
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
        # GOLDEN tier (Tier1 + freq>=10 + mechanism)
        if disease_tier == 1 and train_frequency >= 10 and mechanism_support:
            return ConfidenceTier.GOLDEN, False, None

        # HIGH tier
        if train_frequency >= 15 and mechanism_support:
            return ConfidenceTier.HIGH, False, None
        if rank <= 5 and train_frequency >= 10 and mechanism_support:
            return ConfidenceTier.HIGH, False, None

        # MEDIUM tier
        if train_frequency >= 5 and mechanism_support:
            return ConfidenceTier.MEDIUM, False, None
        if train_frequency >= 10:
            return ConfidenceTier.MEDIUM, False, None

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

        h136 findings:
        - Infectious: rank<=10 + freq>=15 + mech = 55.6% precision (GOLDEN!)
        - Cardiovascular: rank<=5 + mech = 38.2% precision (HIGH)
        - Respiratory: rank<=10 + freq>=15 + mech = 35.0% precision (HIGH)

        h144 findings:
        - Metabolic + statin + rank<=10 = 60.0% precision (GOLDEN!)
        """
        if category == 'infectious':
            if rank <= 10 and train_frequency >= 15 and mechanism_support:
                return ConfidenceTier.GOLDEN  # 55.6% precision
            if rank <= 10 and train_frequency >= 10 and mechanism_support:
                return ConfidenceTier.HIGH

        elif category == 'cardiovascular':
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

            # h136 generic rescue
            if rank <= 5 and mechanism_support:
                return ConfidenceTier.HIGH  # 38.2% precision
            # h154/h266: Beta-blockers achieve 42.1% precision at rank<=10
            # h266 found extending from rank<=5 to rank<=10 captures more predictions
            if rank <= 10 and any(bb in drug_lower for bb in BETA_BLOCKERS):
                return ConfidenceTier.HIGH  # 42.1% precision (h266)

        elif category == 'respiratory':
            drug_lower = drug_name.lower()

            # h265: Fluoroquinolones achieve 44.4% precision for respiratory (HIGH)
            if any(fq in drug_lower for fq in FLUOROQUINOLONE_DRUGS):
                return ConfidenceTier.HIGH  # 44.4% precision (h265/h163)

            if rank <= 10 and train_frequency >= 15 and mechanism_support:
                return ConfidenceTier.HIGH  # 35.0% precision

        elif category == 'metabolic':
            drug_lower = drug_name.lower()

            # h265: Thiazolidinediones achieve 66.7% precision for metabolic (GOLDEN)
            if any(tzd in drug_lower for tzd in THIAZOLIDINEDIONES):
                return ConfidenceTier.GOLDEN  # 66.7% precision (h265/h163)

            # h144: Statin drugs achieve 60% precision for metabolic diseases
            if rank <= 10 and any(statin in drug_lower for statin in STATIN_DRUGS):
                return ConfidenceTier.GOLDEN  # 60.0% precision

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

            # h274: Cross-type cancer drugs (has cancer GT but different type)
            # 30.6% precision - give them MEDIUM tier (don't FILTER them)
            # They're not subtype refinements, but they're still valid cancer drug predictions
            return ConfidenceTier.MEDIUM  # Cross-type repurposing: 30.6% precision

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
            # h150: Corticosteroids achieve 48.6% precision for hematological diseases
            # NOTE: Mechanism support NOT required - works through immunosuppression
            drug_lower = drug_name.lower()
            if rank <= 10 and any(steroid in drug_lower for steroid in CORTICOSTEROID_DRUGS):
                return ConfidenceTier.HIGH  # 48.6% precision

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

            # Class-matched neurological drugs get at least MEDIUM tier
            # (they have strong class-based evidence even if kNN didn't find them)
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
        supplemented.sort(key=lambda p: (tier_priority.get(p.confidence_tier, 5), -p.knn_score))

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
                # Rank drugs
                sorted_drugs = sorted(drug_scores.items(), key=lambda x: x[1], reverse=True)
                max_score = sorted_drugs[0][1] if sorted_drugs else 1.0

                for rank, (drug_id, score) in enumerate(sorted_drugs[:top_n], 1):
                    norm_score = score / max_score if max_score > 0 else 0
                    train_freq = self.drug_train_freq.get(drug_id, 0)
                    mech_support = self._compute_mechanism_support(drug_id, disease_id)
                    has_targets = drug_id in self.drug_targets and len(self.drug_targets[drug_id]) > 0
                    drug_name = self.drug_id_to_name.get(drug_id, drug_id)

                    tier, rescue_applied, cat_specific = self._assign_confidence_tier(
                        rank, train_freq, mech_support, has_targets, disease_tier, category,
                        drug_name, disease_name, drug_id
                    )

                    pred = DrugPrediction(
                        drug_name=drug_name,
                        drug_id=drug_id,
                        rank=rank,
                        knn_score=score,
                        norm_score=norm_score,
                        confidence_tier=tier,
                        train_frequency=train_freq,
                        mechanism_support=mech_support,
                        has_targets=has_targets,
                        category=category,
                        disease_tier=disease_tier,
                        category_rescue_applied=rescue_applied,
                        category_specific_tier=cat_specific,
                    )

                    if include_filtered or tier != ConfidenceTier.FILTER:
                        predictions.append(pred)

            # h173: Supplement with drug-class predictions for neurological diseases
            # When kNN has low coverage, inject drugs from appropriate classes
            if category == 'neurological':
                predictions = self._supplement_neurological_predictions(
                    disease_name, disease_id, disease_tier, category,
                    predictions, max_score if drug_scores else 1.0, top_n, include_filtered
                )

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
