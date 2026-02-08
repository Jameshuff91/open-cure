#!/usr/bin/env python3
"""
Improved disease name matching with fuzzy matching and synonym handling.

This module fixes the critical data mapping issue where 77.9% of drug-disease
pairs were lost during training due to exact string matching failures.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Set, Tuple
from collections import defaultdict

# Synonym groups - map variants to canonical name
DISEASE_SYNONYMS: Dict[str, str] = {
    # Atopic dermatitis variants - D003876 is the specific MESH code
    "atopic dermatitis": "atopic dermatitis",
    "atopic dermatitis ": "atopic dermatitis",  # trailing space
    "atopic eczema": "atopic dermatitis",
    # "eczema" alone is too generic, don't map it

    # Psoriasis variants
    "plaque psoriasis": "psoriasis",
    "chronic plaque psoriasis": "psoriasis",
    "moderate to severe plaque psoriasis": "psoriasis",
    "moderate-to-severe plaque psoriasis": "psoriasis",

    # Diabetes variants
    "type 2 diabetes": "type 2 diabetes mellitus",
    "type 2 diabetes mellitus": "type 2 diabetes mellitus",
    "type 2 diabetes mellitus ": "type 2 diabetes mellitus",  # trailing space
    "type ii diabetes": "type 2 diabetes mellitus",
    "type ii diabetes mellitus": "type 2 diabetes mellitus",
    "diabetes mellitus type 2": "type 2 diabetes mellitus",
    "t2dm": "type 2 diabetes mellitus",
    "diabetes mellitus": "type 2 diabetes mellitus",
    "noninsulin dependent diabetes mellitus type ii": "type 2 diabetes mellitus",
    "non-insulin dependent diabetes mellitus": "type 2 diabetes mellitus",

    # HIV variants
    "hiv1 infection": "hiv infection",
    "hiv-1 infection": "hiv infection",
    "hiv infection": "hiv infection",
    "human immunodeficiency virus infection": "hiv infection",

    # Parkinson's variants
    "parkinsons disease": "parkinson disease",
    "parkinson's disease": "parkinson disease",
    "parkinson disease": "parkinson disease",

    # Alzheimer's variants
    "alzheimers disease": "alzheimer disease",
    "alzheimer's disease": "alzheimer disease",
    "alzheimer disease": "alzheimer disease",

    # Lung cancer variants
    "non-small cell lung cancer": "lung cancer",
    "non small cell lung cancer": "lung cancer",
    "nonsmall cell lung cancer": "lung cancer",
    "nsclc": "lung cancer",
    "small cell lung cancer": "lung cancer",

    # Heart conditions
    "congestive heart failure": "heart failure",
    "chronic heart failure": "heart failure",
    "cardiac failure": "heart failure",

    # MS variants
    "relapsing multiple sclerosis": "multiple sclerosis",
    "relapsing-remitting multiple sclerosis": "multiple sclerosis",

    # Rheumatoid arthritis
    "ra": "rheumatoid arthritis",

    # Asthma variants
    "bronchial asthma": "asthma",
    "allergic asthma": "asthma",

    # COPD variants
    "chronic obstructive pulmonary disease": "copd",
    "chronic bronchitis": "copd",

    # Allergic rhinitis variants
    "seasonal allergic rhinitis": "allergic rhinitis",
    "perennial allergic rhinitis": "allergic rhinitis",
    "hay fever": "allergic rhinitis",

    # Hypertension variants
    "high blood pressure": "hypertension",
    "essential hypertension": "hypertension",
    "arterial hypertension": "hypertension",

    # Cancer variants (keep specific when MESH mapping exists)
    "breast neoplasm": "breast cancer",
    "breast neoplasms": "breast cancer",
    "mammary cancer": "breast cancer",
    "metastatic breast cancer": "breast cancer",

    # Hepatitis variants
    "chronic hepatitis c": "hepatitis c",
    "hepatitis c virus infection": "hepatitis c",
    "hcv infection": "hepatitis c",

    # Ankylosing spondylitis
    "ankylosing spondylitis": "ankylosing spondylitis",
    "as": "ankylosing spondylitis",

    # Multiple myeloma
    "multiple myeloma": "multiple myeloma",
    "mm": "multiple myeloma",

    # Inflammatory bowel disease
    "ulcerative colitis": "ulcerative colitis",
    "uc": "ulcerative colitis",
    "crohn disease": "crohn disease",
    "crohn's disease": "crohn disease",
    "crohns disease": "crohn disease",
    "inflammatory bowel disease": "inflammatory bowel disease",
    "ibd": "inflammatory bowel disease",

    # h712: Synonym mappings for top unmapped EC diseases
    "diabetic kidney disease": "diabetic nephropathy",
    "diabetic renal disease": "diabetic nephropathy",
    "septicaemia": "sepsis",
    "septicemia": "sepsis",
    "renal cell carcinoma": "renal carcinoma",
    "regional enteritis": "crohn disease",
    "hodgkins disease": "hodgkin lymphoma",
    "severe psoriasis": "psoriasis",
    "openangle glaucoma": "open angle glaucoma",
    "acute gouty arthritis": "gout",
    "erosive esophagitis": "esophagitis",
    "gastroesophageal reflux disease": "gerd",
    "idiopathic thrombocytopenic purpura": "itp",
    "acne rosacea": "rosacea",
    "primary dysmenorrhea": "dysmenorrhea",
    "chronic anterior uveitis": "uveitis",
    "postherpetic neuralgia": "postherpetic neuralgia",
    "agerelated macular degeneration": "macular degeneration",
    "age related macular degeneration": "macular degeneration",

    # h712 (continued): Additional synonym mappings for top unmapped EC diseases
    # These map EC disease names to canonical names that already have MESH IDs
    "bipolar i disorder": "bipolar disorder",
    "bipolar 1 disorder": "bipolar disorder",
    "bipolar disorder": "bipolar disorder",
    "partial seizures": "epilepsy",
    "focal seizures": "epilepsy",
    "migraine without aura": "migraine disorders",
    "migraine with aura": "migraine disorders",
    "major depressive episode": "major depressive disorder",
    "malignant lymphoma": "lymphoma",
    "lymphomas": "lymphoma",
    "febrile neutropenia": "neutropenia",
    "hemophilia a": "hemophilia",
    "hepatic cirrhosis": "liver cirrhosis",
    "liver fibrosis": "liver cirrhosis",
    "compensated cirrhosis type c": "hepatitis c",
    "hypercalcemia associated with cancer": "hypercalcemia",
    "acute rheumatic carditis": "rheumatic heart disease",
    "stevensjohnson syndrome": "stevens johnson syndrome",
    "active secondary progressive disease": "multiple sclerosis",
    "secondary progressive multiple sclerosis": "multiple sclerosis",
    "primary progressive multiple sclerosis": "multiple sclerosis",
    "lower respiratory tract infections": "respiratory tract infections",
    "upper respiratory tract infections": "respiratory tract infections",
    "infective conjunctivitides": "conjunctivitis",
    "bacterial conjunctivitis": "conjunctivitis",
    "allergic conjunctivitis": "conjunctivitis",
    "allergic disease": "hypersensitivity",
    "allergen specific allergic disease": "hypersensitivity",
    "allergy": "hypersensitivity",
    "allergic corneal marginal ulcers": "corneal ulcer",
    "anterior segment inflammation": "uveitis",
    "corticosteroid responsive dermatoses": "dermatitis",
    "nonsuppurative thyroiditis": "thyroiditis subacute",
    "subacute thyroiditis": "thyroiditis subacute",
    "hiv 1 infection": "hiv infection",
    "bone infections": "osteomyelitis",
    "anxiety": "anxiety disorder",
    "generalized anxiety disorder": "anxiety disorder",
    "social anxiety disorder": "anxiety disorder",
    "corticosteroid responsive dermatoses": "dermatitis",

    # h336/h347: Additional synonyms for GT matching (+25 GT hits)
    # These were found by analyzing prediction-GT mismatches
    "acquired hemolytic anemia": "anemia, hemolytic, acquired",  # Comma reordering
    "pure red cell aplasia": "pure red-cell aplasia",  # Hyphen variant
    "zollinger ellison syndrome": "zollinger-ellison syndrome",  # Hyphen variant
    "graft versus host disease gvhd": "graft versus host disease",  # Abbreviation removal
    "diffuse large b cell lymphoma dlbcl": "diffuse large b-cell lymphoma",  # Hyphen + abbrev

    # Additional common abbreviations (discovered during h336 analysis)
    "attention deficit hyperactivity disorder adhd": "attention deficit-hyperactivity disorder",
    "atypical hemolytic uremic syndrome": "atypical hemolytic-uremic syndrome",
    "autosomal dominant polycystic kidney disease adpkd": "autosomal dominant polycystic kidney disease",
    "basal cell carcinoma bcc": "basal cell carcinoma",
    "central precocious puberty cpp": "central precocious puberty",
    "common variable immunodeficiency cvid": "common variable immunodeficiency",
    "disseminated intravascular coagulation dic": "disseminated intravascular coagulation",

    # h712 session 2: Additional synonym mappings for residual unmapped EC diseases
    # These are EC disease names that need mapping to canonical names (which have MESH IDs)
    # Curated with medical validation — symptoms and overly vague groupings excluded

    # Eye diseases — subtypes to parent
    "iridocyclitis": "anterior uveitis",  # Inflammation of iris + ciliary body = anterior uveitis
    "cyclitis": "anterior uveitis",       # Ciliary body inflammation = anterior uveitis
    "chorioretinitis": "uveitis",         # Posterior segment inflammation
    "ocular inflammatory conditions": "uveitis",  # Generic → map to uveitis

    # Infections — specifics to DRKG-mapped parents
    "nosocomial pneumonia": "pneumonia",
    "hospital acquired pneumonia": "pneumonia",
    "community acquired pneumonia": "pneumonia",
    "acute bronchitis": "bronchitis",
    "acute bacterial sinusitis": "sinusitis",
    "purulent meningitis": "meningitis",
    "acute intestinal amebiasis": "amebiasis",
    "intestinal amebiasis": "amebiasis",
    "nongonococcal urethritis": "urethritis",

    # Musculoskeletal — subtypes to parent
    "acute bursitis": "bursitis",
    "subacute bursitis": "bursitis",

    # Dermatology
    "exfoliative dermatitis": "exfoliative dermatitis",  # Has own MESH: D003877
    "actinic keratoses": "actinic keratosis",
    "actinic keratosis": "actinic keratosis",

    # Migraine subtypes
    "migraine attacks without aura": "migraine disorders",
    "common migraine": "migraine disorders",

    # Cancer subtypes
    "early breast cancer": "breast cancer",
    "colon cancer": "colon cancer",  # Has own MESH: D003110
    "primary peritoneal cancer": "peritoneal neoplasms",

    # Constipation
    "chronic constipation": "constipation",

    # Reflux
    "reflux esophagitis": "esophagitis",

    # Leukemia
    "acute leukemia": "leukemia",
    "acute leukemia of childhood": "leukemia",

    # Parkinsonism
    "parkinsonism": "parkinsonism",  # Has own MESH: D020734 (distinct from Parkinson disease)

    # Headache types
    "tension headache": "tension type headache",
    "muscle contraction headache": "tension type headache",

    # Back pain
    "backache": "back pain",
    "low back pain": "back pain",

    # Obesity
    "exogenous obesity": "obesity",

    # Anemia
    "renal anemia": "anemia",
    "renal anaemia": "anemia",
    "anemias of childhood": "anemia",

    # Nausea/vomiting
    "nausea and vomiting": "nausea",
    "postoperative nausea and vomiting": "ponv",

    # Sprains
    "sprains": "sprains and strains",
    "strains": "sprains and strains",

    # Muscle soreness
    "muscle soreness": "myalgia",

    # Addison's disease
    "addisons disease": "addison disease",

    # Bipolar/manic
    "manic episodes": "bipolar disorder",
    "manic episode": "bipolar disorder",

    # Seizures
    "primary generalized tonicclonic seizures": "epilepsy",
    "generalized tonic clonic seizures": "epilepsy",
    "secondary generalisation": "epilepsy",

    # Influenza
    "influenza a virus infection": "influenza",
    "influenza a": "influenza",

    # Atrophy
    "vulvar and vaginal atrophy": "vulvovaginal atrophy",

    # Paget's disease
    "pagets disease of bone": "paget disease of bone",

    # Goiter/thyroid
    "euthyroid goiters": "thyroid nodule",
    "thyroid nodules": "thyroid nodule",

    # CIS (clinically isolated syndrome = first MS episode)
    "clinically isolated syndrome": "multiple sclerosis",

    # Bone infections
    "bone and joint infections": "osteomyelitis",

    # Skin infections (SSSI terms)
    "skin and skin structure infections": "skin diseases infectious",
    "complicated skin and skin structure infections": "skin diseases infectious",
    "uncomplicated skin and skin structure infections": "skin diseases infectious",
    "acute bacterial skin and skin structure infections": "skin diseases infectious",

    # Onychomycosis
    "tinea unguium": "onychomycosis",
    "onychomycosis": "onychomycosis",

    # Organ rejection
    "organ rejection": "graft rejection",
    "transplant rejection": "graft rejection",

    # Idiopathic eosinophilic pneumonia
    "idiopathic eosinophilic pneumonias": "pulmonary eosinophilia",
    "idiopathic eosinophilic pneumonia": "pulmonary eosinophilia",
    "eosinophilic pneumonia": "pulmonary eosinophilia",

    # h712 session 3: Additional synonym mappings for EC name variants
    # Map EC names to canonical forms that have MESH mappings above
    "cancer pain ": "cancer pain",
    "covid19 ": "covid19",
    "renal anemia ": "renal anemia",
    "legionella infection ": "legionella infection",
    "peripheral neuropathic pain ": "peripheral neuropathic pain",
    "influenza b virus infection ": "influenza b virus infection",
    "palmoplantar pustulosis ": "palmoplantar pustulosis",
    "malignant melanoma ": "malignant melanoma",
    "cognitive impairment ": "dementia of the alzheimers type",
    "all ": "acute lymphocytic leukemia",  # ALL = Acute Lymphoblastic Leukemia
    "haemophilia b": "hemophilia b",
    "pseudomonas aeruginosa infection ": "pseudomonas infection",
    "pseudomonas aeruginosa infection": "pseudomonas infection",
    "infections caused by pseudomonas aeruginosa": "pseudomonas infection",
    "wilsons disease ": "wilsons disease",
    "her2 overexpression ": "breast cancer",
    "pituitary gigantism ": "growth failure",
    "hypozincemia ": "hypocalcemia",  # Zinc/calcium deficiency parent
    "unresectable melanoma ": "malignant melanoma",
    "dementia with lewy bodies ": "dementia of the alzheimers type",
    "deep mycosis ": "invasive candidiasis",
    "testicular tumor": "solid tumors",
    "extragonadal tumors ": "solid tumors",
    "germ cell tumors ": "solid tumors",
    "secondary generalized seizures ": "epilepsy",
    "chronic pyoderma": "skin infections",
    "surgical wounds": "wound sepsis",
    "gastric malt lymphoma": "lymphoma",
    "meibomianitis": "conjunctivitis",
    "cin 2": "cervicitis",
    "peritonsillitis": "laryngopharyngitis",
    "peritonsillar abscess": "abscesses",
    "cystic fibrosis cf": "cystic fibrosis",
    "nonradiographic axial spondyloarthritis": "ankylosing spondylitis",
    "metastasis": "solid tumors",
    "primary mediastinal large bcell lymphoma": "diffuse large bcell lymphoma",
    "acute myeloid leukaemia": "leukemia",
    "erythroblastopenia rbc anemia": "erythroblastopenia",
    "vanishing testis syndrome": "cryptorchidism",
    "anemias of nutritional origin": "folate deficiency",
    "end stage renal failure": "endstage kidney disease",
    "nephritic syndrome": "glomerulonephritis",
    "diarrheal states": "irritable bowel syndrome",
    "xerosis": "dermatitis",
    "acute subacromial bursitissupraspinatus tendinitis": "bursitis",
    "potassiumlosing nephropathy": "hypokalemia",
    "psychomotor seizures": "epilepsy",
    "corneal ulcers": "corneal ulcer",
    "anaphylactic reactions": "hypersensitivity",
    "acute enterocolitis": "irritable bowel syndrome",
    "biliary colic": "heartburn",
    "obstructive jaundice": "hepatic disease",
    "histiocytic lymphoma": "diffuse large bcell lymphoma",
    "refractory anemia with ringed sideroblasts": "anemia",
    "refractory anemia with excess blasts": "anemia",
    "localized hypertrophic inflammatory lesions of granuloma annulare": "dermatitis",
    "cystic tumors of a tendon ganglia": "tendonitis",
    "acute nonlymphocytic leukemia": "leukemia",
    "nonfatal stroke": "stroke",
    "extensive burns": "wound sepsis",
    "vasodilatory shock": "septic shock",
    "candida infections": "invasive candidiasis",
    "burn infections": "wound sepsis",
    "distributive shock": "septic shock",
    "neurogenic detrusor overactivity": "urinary urgency",
    "vitamin c deficiency": "folate deficiency",
    "soft tissue infections": "skin infections",
    "hodgkins lymphoma": "hodgkin lymphoma",
    "hematologic malignancies": "leukemia",
    "igan": "glomerulonephritis",
    "adult periodontitis": "cellulitis",
    "fractures": "osteoporotic fracture",
    "abrasions": "wound sepsis",
    "stiffness": "muscle stiffness",
    "bruises": "wound sepsis",
    "gynecological infections": "cervicitis",
    "acute lymphoblastic leukaemia": "acute lymphocytic leukemia",
    "molluscum contagiosum": "measles",
    # h712s4: High-value unmapped EC disease synonyms (56 entries, ~565 GT pairs)
    # --- Respiratory/Infectious (33+8=41 pairs) ---
    "respiratory tract infectious disorder": "respiratory tract infections",
    "upper respiratory tract disorder": "respiratory tract infections",
    # --- Cardiovascular (28+15+5=48 pairs) ---
    "coronary artery disorder": "coronary artery disease",
    "angina unstable": "unstable angina",
    "variant angina": "angina",
    # --- HIV/AIDS (23+9=32 pairs) ---
    "human immunodeficiency virus positive": "hiv infection",
    "aids": "hiv infection",
    # --- Allergy (20 pairs) ---
    "atopic rhinitis": "allergic rhinitis",
    # --- Cancer subtypes (18+17+16+12+6+6+5+7=87 pairs) ---
    "carcinoma breast stage iv": "breast cancer",
    "hodgkin disease": "hodgkin lymphoma",
    "chronic myelogenous leukemia, bcr abl1 positive": "chronic myeloid leukemia",
    "liver carcinoma": "hepatocellular carcinoma",
    "pancreatic carcinoma non-resectable": "pancreatic cancer",
    "blast phase": "chronic myeloid leukemia",
    "primary peritoneal carcinoma": "ovarian cancer",
    "myelodysplasia": "myelodysplastic syndromes",
    "myelodysplastic syndrome with ring sideroblasts": "myelodysplastic syndromes",
    # --- Psychiatric (18+10+6=34 pairs) ---
    "depression, ctcae": "major depressive disorder",
    "depressed mood": "major depressive disorder",
    "manic disorder": "bipolar disorder",
    # --- Pulmonary (18+10=28 pairs) ---
    "reactive airway disease": "asthma",
    "copd, severe early onset": "copd",
    # --- Infectious (16+10+8+8+8+7+7+6+6+6+6+6=104 pairs) ---
    "abdominal infection": "intraabdominal infections",
    "miliary tuberculosis": "tuberculosis",
    "anthrax disease": "anthrax",
    "mycobacterium avium complex disease": "nontuberculous mycobacteriosis",
    "mycobacterium infections, nontuberculous": "nontuberculous mycobacteriosis",
    "helicobacter pylori infectious disease": "h pylori infection",
    "mycoses": "systemic mycosis",
    "bacterial sepsis": "septicemia",
    "plasmodium falciparum malaria": "malaria",
    "malaria, vivax": "malaria",
    "acute bronchitis": "bronchitis",
    "bacteriemia": "bacteremia",
    # --- Rheumatic/Hematologic (14+8=22 pairs) ---
    # "primary gout" handled via hardcoded MESH mapping (canonical chaining issue with arthritis→gout)
    "autoimmune thrombocytopenic purpura": "itp",  # = idiopathic thrombocytopenic purpura
    # --- Endocrine/Metabolic (10+9+5=24 pairs) ---
    "secondary adrenal insufficiency": "adrenocortical insufficiency",
    "vasomotor menopausal symptoms": "vasomotor symptoms associated with menopause",
    "simple obesity": "obesity",
    # --- Ophthalmologic (10+5=15 pairs) ---
    "superficial keratitis": "keratitis",
    "central retinal vein occlusion with macular edema": "retinal vein occlusion",
    # --- Neurological (8+6+5+5+5=29 pairs) ---
    "generalized onset seizure": "epilepsy",
    "generalized myasthenia": "myasthenia gravis",
    "complex partial epilepsy": "epilepsy",
    "childhood absence epilepsy": "epilepsy",
    "restless legs": "restless legs syndrome",
    # --- GI (9+7=16 pairs) ---
    "crohns aggravated": "crohn disease",
    # "peptic esophagitis" handled via direct MESH mapping (canonical chain issue)
    # "post-operative pain" handled via direct MESH mapping (canonical chain issue)
    # --- Other (9+8+7+7+5+5+5+5=51 pairs) ---
    "anemia due to chronic disorder": "anemia",
    "brain edema": "cerebral edema associated with primary brain tumor",
    "deep venous thrombosis": "deep vein thrombosis",
    # "post-operative pain" handled via direct MESH mapping
    "alcohol dependence": "alcoholism",
    "pulmonary edema": "pulmonary edema",
    "folic acid deficiency anemia": "megaloblastic anemias",
    # "muscular headache" handled via direct MESH mapping
    # --- Urological (9+6+5=20 pairs) ---
    "pollakisuria": "urinary urgency",
    "genital warts": "genital warts",
    "menorrhagia": "menorrhagia",
    # --- Renal (11 pairs) ---
    "kidney disorder": "kidney diseases",
    # h712s5: Synonym mappings for new MESH-mapped diseases
    # Map EC name variants to canonical names that now have direct MESH mappings
    "upper respiratory infections": "respiratory tract infections",
    "uncontrolled diabetes": "type 2 diabetes mellitus",
    "hot flashes": "hot flushes",
    "alzheimers type dementia": "dementia of the alzheimers type",
    "primary progressive ms": "multiple sclerosis",
    "infections caused by klebsiella spp": "klebsiella infections",
    "infections caused by klebsiella spp ": "klebsiella infections",
    "infections caused by haemophilus influenzae type b": "haemophilus influenzae infections",
    "infections caused by haemophilus influenzae type b ": "haemophilus influenzae infections",
    "infections caused by acinetobacter spp": "acinetobacter infections",
    "infections caused by acinetobacter spp ": "acinetobacter infections",

    # h712s6 (h731 session): Final batch of unmapped EC diseases
    # Map remaining EC names to canonical names with MESH mappings
    "all": "acute lymphoblastic leukemia",  # ALL abbreviation
    "schizoaffective disorder": "psychotic disorders",
    "polycythemia vera": "acquired polycythemia vera",
    "nmosd": "neuromyelitis optica",  # NMO spectrum disorder
    "post herpetic neuralgia": "postherpetic neuralgia",  # space variant → D051474 via MESH mapping
    "methyl alcohol poisoning": "methanol poisoning",
    "ais": "acute ischemic stroke",  # AIS abbreviation
    "ventricular fibrillation": "ventricular fibrillation",  # Needs direct MESH mapping
    "absence seizures": "epilepsy, absence",
    "grand mal seizures": "epilepsy, generalized",
    "acute myelogenous leukemia": "leukemia, myeloid, acute",

    # h712s7: Comprehensive unmapped EC disease synonym expansion
    # Maps EC disease name variants to canonical names that have MESH mappings
    "covid19": "covid-19",
    "covid 19": "covid-19",
    "coronavirus disease 2019": "covid-19",
    "sars cov 2 infection": "covid-19",
    "varicella": "varicella zoster infection",
    "chickenpox": "varicella zoster infection",
    "softtissue sarcoma": "soft tissue sarcoma",
    "soft tissue sarcomas": "soft tissue sarcoma",
    "lumbago": "low back pain",
    "pruritus cutaneous": "pruritus",
    "pruritus scroti": "pruritus",
    "skin pruritus": "pruritus",
    "migraine headaches": "migraine disorders",
    "chronic asthma": "asthma",
    "reversible obstructive airways disease": "asthma",
    "regional ileitis": "crohn disease",
    "dermographism": "urticaria",
    "dermatographism": "urticaria",
    "sjogrens syndrome": "sjogren syndrome",
    "sjgrens syndrome": "sjogren syndrome",
    "prostatic cancer": "prostate cancer",
    "prostatic carcinoma": "prostate cancer",
    "smoking addiction": "tobacco use disorder",
    "smoking cessation": "tobacco use disorder",
    "autistic disorder": "autistic spectrum disorder",
    "gallbladder stones": "cholelithiasis",
    "ulcerative proctitis": "proctitis",
    "ulcerative proctosigmoiditis": "proctocolitis",
    "gonococcal infections": "gonorrhea",
    "uncomplicated cervical gonorrhea": "gonorrhea",
    "uncomplicated gonococcal urethritis": "gonorrhea",
    "pneumocystis carinii pneumonia": "pneumocystis pneumonia",
    "pneumocystis carinii pneumonia pcp": "pneumocystis pneumonia",
    "pcp": "pneumocystis pneumonia",
    "pneumocystis pneumonia": "pneumocystis pneumonia",
    "pneumonia due to pneumocystis carinii": "pneumocystis pneumonia",
    "endomyometritis": "endometritis",
    "postpartum endomyometritis": "endometritis",
    "attention deficit disorder with hyperactivity": "attention deficit hyperactivity disorder",
    "brain tumor": "brain neoplasms",
    "primary brain tumor": "brain neoplasms",
    "primary malignant brain tumor": "brain neoplasms",
    "secondary amenorrhea": "amenorrhea",
    "secondary amenorrhea of undetermined etiology": "amenorrhea",
    "postoralcontraceptive amenorrhea": "amenorrhea",
    "insulin shock": "hypoglycemia",
    "delirium tremens": "alcohol withdrawal delirium",
    "withdrawal symptoms of acute alcoholism": "alcohol withdrawal delirium",
    "tinea capitis": "tinea",
    "tinea barbae": "tinea",
    "ringworm infections": "tinea",
    "bleeding episodes": "hemorrhage",
    "hemorrhage": "hemorrhage",
    "thermal burns": "burns",
    "thirddegree burns": "burns",
    "radiation burns": "burns",
    "acute renal failure": "acute kidney injury",
    "acute cholecystitis": "cholecystitis",
    "acute prostatitis": "prostatitis",
    "fungal infections": "mycoses",
    "presumed fungal infections": "mycoses",
    "serious fungal infections caused by fusarium species": "mycoses",
    "hepatitis d": "hepatitis d",
    "pathologic myopia": "myopia",
    "pathological myopia": "myopia",
    "systemic embolism": "embolism",
    "peripheral arterial embolism": "embolism",
    "peripheral embolism": "embolism",
    "mild cognitive impairment": "cognitive dysfunction",
    "invasive meningococcal disease": "meningococcal infections",
    "meningococcal infection": "meningococcal infections",
    "chronic liver disease": "liver diseases",
    "severe liver disease": "liver diseases",
    "hepatic disease": "liver diseases",
    "overactive bladder": "urinary bladder, overactive",
    "overactive bladder syndrome": "urinary bladder, overactive",
    "bullous pemphigoid": "pemphigoid, bullous",
    "absssi": "skin diseases, infectious",
    "adult growth hormone deficiency": "dwarfism, pituitary",
    "cerebrovascular disease": "cerebrovascular disorders",
    "nasal polyps": "nasal polyps",
    "endometrial hyperplasia": "endometrial hyperplasia",
    "cerebral edema": "brain edema",
    "sinus tachycardia": "tachycardia, sinus",
    "paroxysmal tachycardia": "tachycardia, paroxysmal",
    "reciprocating tachycardias": "tachycardia, supraventricular",
    "tachyarrhythmias": "tachycardia",
    "tremor": "tremor",
    "intermittent claudication": "intermittent claudication",
    "bacterial infections": "bacterial infections",
    "secondary bacterial infections": "bacterial infections",
    "serious infections of bones and joints": "bacterial infections",
    "liver abscess": "liver abscess",
    "apnea of prematurity": "apnea",
    "primary hypogonadism": "hypogonadism",
    "testicular failure": "hypogonadism",
    "synovitis": "synovitis",
    "potassium deficiency states": "hypokalemia",
    "parathyroid carcinoma": "parathyroid neoplasms",
    "miosis": "miosis",
    "digitalis intoxication": "digitalis glycoside poisoning",
    "digitalis intoxications": "digitalis glycoside poisoning",
    "dysproteinemias": "paraproteinemias",
    "obstructive liver disease": "cholestasis",
    "partial biliary obstruction": "cholestasis",
    "symptomatic carotid artery disease": "carotid artery diseases",
    "transient ischemia of the brain": "ischemic attack, transient",
    "transient ischaemic attack": "ischemic attack, transient",
    "hypoestrogenism": "menopause",
    "sodium retention": "edema",
    "acute nonspecific diarrhea": "diarrhea",
    "travelers diarrhea": "diarrhea",
    "clostridium species infections": "clostridium infections",
    "haemophilus influenzae respiratory infections": "haemophilus infections",
    "idiopathic steatorrhea": "malabsorption syndromes",
    "mixed infections": "coinfection",
    "vasomotor symptoms associated with the menopause": "hot flashes",
    "vasomotor symptoms": "hot flashes",
    "blastomycosis": "blastomycosis",
    "uncomplicated endocervical infections": "cervicitis",
    "uncomplicated endocervical infections caused by chlamydia trachomatis": "cervicitis",

    # Additional 1-drug disease synonyms for common conditions
    "shingles": "herpes zoster",
    "scarlet fever": "scarlatina",
    "tachycardia": "tachycardia",
    "subarachnoid hemorrhage": "subarachnoid hemorrhage",
    "peripheral artery disease": "peripheral arterial disease",
    "peripheral vessel disease": "peripheral vascular diseases",
    "pouchitis": "pouchitis",
    "vaginal candidiasis": "candidiasis, vulvovaginal",
    "vaginitis": "vaginitis",
    "oral thrush": "candidiasis, oral",
    "seasonal affective disorder": "seasonal affective disorder",
    "panic attacks": "panic disorder",
    "recurrent depression": "depressive disorder, major",
    "psychotic depression": "depressive disorder, major",
    "psychotic depressive disorders": "depressive disorder, major",
    "psychoneurosis": "neurotic disorders",
    "phobic disorders": "phobic disorders",
    "stiffman syndrome": "stiff person syndrome",
    "shift work sleep disorder": "sleep wake disorders",
    "transient insomnia": "sleep initiation and maintenance disorders",
    "severe spasticity": "muscle spasticity",
    "spasms": "muscle spasticity",
    "spasms due to spinal cord trauma": "muscle spasticity",
    "spasticity due to traumatic brain injury": "muscle spasticity",
    "spasticity of cerebral origin": "muscle spasticity",
    "spasticity of spinal cord origin": "muscle spasticity",
    "skeletal muscle spasm": "muscle spasticity",
    "overweight": "overweight",
    "preeclampsia": "pre-eclampsia",
    "preterm labour": "obstetric labor, premature",
    "polycystic ovarian disease": "polycystic ovary syndrome",
    "polycystic ovarian syndrome": "polycystic ovary syndrome",
    "polycystic ovary syndrome": "polycystic ovary syndrome",
    "varicose veins": "varicose veins",
    "varicose veins of the lower extremities": "varicose veins",
    "splenomegaly": "splenomegaly",
    "strabismus": "strabismus",
    "photophobia": "photophobia",
    "presbyopia": "presbyopia",
    "panuveitis": "panuveitis",
    "rhinorrhea": "rhinorrhea",
    "perennial rhinitis": "rhinitis",
    "seasonal rhinitis": "rhinitis",
    "nonallergic rhinitis": "rhinitis",
    "sialorrhea": "sialorrhea",
    "sialorrhoea": "sialorrhea",
    "polydipsia": "polydipsia",
    "thyroid carcinoma": "thyroid neoplasms",
    "thyroid eye disease": "graves ophthalmopathy",
    "testicular carcinoma": "testicular neoplasms",
    "testicular carcinoma embryonal cell": "testicular neoplasms",
    "penis cancer": "penile neoplasms",
    "urothelial cancer": "urologic neoplasms",
    "upper tract urothelial cancer": "urologic neoplasms",
    "renal cancer": "kidney neoplasms",
    "renalcell carcinoma": "carcinoma, renal cell",
    "transitional cell bladder cancer": "urinary bladder neoplasms",
    "transitional cell bladder carcinoma": "urinary bladder neoplasms",
    "superficial bladder cancer": "urinary bladder neoplasms",
    "uveal melanoma": "uveal neoplasms",
    "severe malaria": "malaria",
    "vivax malaria": "malaria, vivax",
    "plasmodium vivax malaria": "malaria, vivax",
    "uncomplicated malaria": "malaria",
    "salmonella infections": "salmonella infections",
    "rickettsial infections": "rickettsia infections",
    "paratyphoid": "paratyphoid fever",
    "typhoid": "typhoid fever",
    "scarlet fever": "scarlatina",
    "spinal cord injuries": "spinal cord injuries",
    "spinal cord injury": "spinal cord injuries",
    "primary syphilis": "syphilis",
    "severe nodular acne": "acne vulgaris",
    "pustular acne": "acne vulgaris",
    "solar keratoses": "keratosis, actinic",
    "pediculus humanus capitis infection": "lice infestations",
    "scabies infestations": "scabies",
    "staphylococcus infections": "staphylococcal infections",
    "staphylococci infection": "staphylococcal infections",
    "staphylococci infections": "staphylococcal infections",
    "staphylococcal disease": "staphylococcal infections",
    "streptococcus infections": "streptococcal infections",
    "streptococcal disease": "streptococcal infections",
    "streptococci infection": "streptococcal infections",
    "throat infections": "pharyngitis",
    "petit mal": "epilepsy, absence",
    "severe myoclonic epilepsy in infancy": "epilepsies, myoclonic",
    "refractory generalized tonicclonic seizures": "epilepsy, generalized",
    "tonic seizures": "epilepsy, generalized",
    "secondarily generalized seizure": "epilepsy, generalized",
    "secondarily generalised seizures": "epilepsy, generalized",
    "seizures with secondary generalization": "epilepsy, generalized",
    "severe recurrent convulsive seizures": "seizures",
    "seizures focal partial onset": "epilepsies, partial",
    "premature ventricular extrasystoles": "ventricular premature complexes",
    "ventricular premature beats": "ventricular premature complexes",
    "ventricular premature contraction": "ventricular premature complexes",
    "tumor lysis syndrome": "tumor lysis syndrome",
    "pneumothorax": "pneumothorax",
    "short stature": "dwarfism",
    "pituitary dwarfism": "dwarfism, pituitary",
    "stills disease": "still disease, adult onset",
    "wegener granulomatosis": "granulomatosis with polyangiitis",
    "wegeners granulomatosis": "granulomatosis with polyangiitis",
    "uterine leiomyomas fibroids": "leiomyoma",
    "uterine hemorrhage": "uterine hemorrhage",
    "uterine adenomyosis": "adenomyosis",
    "vulvar atrophy": "atrophy",
    "postmenopausal atrophy of the vagina": "atrophy",
    "endometriosis": "endometriosis",
    "recurrent pericarditis": "pericarditis",
    "progesterone deficiency": "progesterone",
    "severe heart failure": "heart failure",
    "refractory congestive failure": "heart failure",
    "reduced ejection fraction": "heart failure",
    "right ventricular outflow tract obstruction": "ventricular outflow obstruction",
    "shock": "shock",
    "shock syndrome": "shock",
    "primary osteoporosis": "osteoporosis",
    "refractory rickets": "rickets",
    "primary biliary cirrhosis": "liver cirrhosis, biliary",
    "primary biliary cirrhosis pbc": "liver cirrhosis, biliary",
    "pseudomembranous colitis": "enterocolitis, pseudomembranous",
    "staphylococcal enterocolitis": "enterocolitis, pseudomembranous",
    "recurrent glioblastoma": "glioblastoma",
    "refractory anaplastic astrocytoma": "astrocytoma",
    "periarthritis of scapulohumerous": "periarthritis",
    "periarthritis scapulohumeralis": "periarthritis",
    "scapulohumeral periarthritis": "periarthritis",
    "posttraumatic stress disorder ptsd": "stress disorders, post-traumatic",
    "recurrent aphthous stomatitis canker sores": "stomatitis, aphthous",
    "painful musculoskeletal conditions": "musculoskeletal pain",
    "premenstrual cramps": "dysmenorrhea",
    "osteoarthrosis": "osteoarthritis",
    "tophi": "gout",
    "renal colic": "renal colic",
    "renal diseases": "kidney diseases",
    "renal bone disease": "renal osteodystrophy",
    "renal tubular acidosis": "acidosis, renal tubular",
    "renal tubular acidosis rta with calcium stones": "acidosis, renal tubular",
    "thiamine deficiency": "thiamine deficiency",
    "vitamin a deficiency": "vitamin a deficiency",
    "vitamin b 12 deficiency": "vitamin b 12 deficiency",
    "vitamin b 12 deficiencies": "vitamin b 12 deficiency",
    "selenium deficiency": "selenium",
    "zinc deficiency": "zinc",
    "sodium deficiency": "hyponatremia",
    "iron deficiency": "anemia, iron-deficiency",
    "transfusional iron overload": "iron overload",
    "transfusional iron overload in patients with chronic anemia": "iron overload",
    "prolactinsecreting adenomas": "prolactinoma",
    "pituitary adenomas": "pituitary neoplasms",
    "pituitary tumors": "pituitary neoplasms",
    "sheehans syndrome": "sheehan syndrome",
    "pruritus associated with partial biliary obstruction": "pruritus",
    "pneumonia community acquired": "community-acquired infections",
    "pneumonia nosocomial": "cross infection",
    "strongly suspected bacterial pneumonia": "pneumonia, bacterial",
    "pneumonia of infancy": "pneumonia",
    "upper gi bleeding": "gastrointestinal hemorrhage",
    "zollingerellison syndrome zes": "zollinger-ellison syndrome",
    "peptic ulcer refractory to histamine h2 receptor antagonist therapy": "peptic ulcer",
    "postoperative stomal ulcer": "peptic ulcer",
    "stomal ulcer": "peptic ulcer",
    # h712: Disease name synonym expansion — 85 new mappings
    # Exact synonyms (same disease, different name form)
    "posttraumatic stress disorder": "stress disorders, post-traumatic",
    "h pylori": "h pylori infection",
    "burkitts lymphoma": "burkitt lymphoma",
    "urea cycle disorders": "urea cycle disorder",
    "carcinoma in situ": "in situ carcinoma",
    "bipolar ii disorder": "bipolar disorder",
    "bipolar depression": "bipolar disorder",
    "tardive dyskinesia": "dyskinesia",
    "oligodendroglioma": "glioma",
    "mantlecell lymphoma": "mantle cell lymphoma",
    "lennoxgastaut syndrome lgs": "lennox-gastaut syndrome",
    "kawasaki disease": "mucocutaneous lymph node syndrome",
    "dyslipidaemia": "mixed dyslipidemia",
    "hyperlipidaemia": "hyperlipidemia",
    "superficial punctate keratits": "superficial punctate keratitis",
    "gastric carcinoma": "gastric cancer",
    "stomach cancer": "gastric cancer",
    "vulvar cancer": "vulvar squamous cell carcinoma",
    "neuroendocrine tumors": "neuroendocrine neoplasm",
    "neuroendocrine tumors of the pancreas": "pancreatic neuroendocrine tumor",
    "neuroendocrine tumors of the gastrointestinal tract": "digestive system neuroendocrine neoplasm",
    # Subtype → parent (severity/acuity/site modifiers)
    "scalp psoriasis": "psoriasis",
    "severe hypoglycemia": "hypoglycemia",
    "active tuberculosis": "tuberculosis",
    "systemic candidiasis": "candidiasis",
    "rectal gonorrhea": "gonorrhea",
    "acute rhinitis": "rhinitis",
    "caries": "dental caries",
    "status asthmaticus": "asthma",
    "severe diarrhea": "diarrhea",
    "acute poliomyelitis": "poliomyelitis",
    "knee osteoarthritis": "osteoarthritis",
    "chronic urticaria": "urticaria",
    "systemic vasculitis": "vasculitis",
    "early lyme disease": "lyme disease",
    "acute cystitis": "cystitis",
    "major depression": "depression",
    "chronic diarrhea": "diarrhea",
    "hemiplegic migraine": "migraine",
    "severe chronic pain": "chronic pain",
    "pruritic eczemas": "eczema",
    "hypertensive emergencies": "hypertension",
    "depressive symptoms": "depression",
    "painful urethritis": "urethritis",
    # Infection genus/species variants
    "klebsiella respiratory infections": "klebsiella infections",
    "klebsiella species infections": "klebsiella infections",
    "shigella species infections": "shigella infections",
    "acinetobacter species infections": "acinetobacter infections",
    "infections due to campylobacter fetus": "campylobacter fetus infections",
    "central nervous system infections": "central nervous system infectious disorder",
    "fungal infection": "mycoses",
    "invasive fungal infections": "mycoses",
    "ocular infections": "superficial ocular infections",
    # Plural/singular fixes
    "astrocytomas": "astrocytoma",
    "bone sarcomas": "bone sarcoma",
    "cerebral infarctions": "cerebral infarction",
    "craniopharyngiomas": "craniopharyngioma",
    "duodenal ulcers": "duodenal ulcer",
    "dysenteries": "dysentery",
    "dyskinesias": "dyskinesia",
    "ependymomas": "ependymoma",
    "glioblastomas": "glioblastoma",
    "gliomas": "glioma",
    "malignant gliomas": "malignant glioma",
    "medulloblastomas": "medulloblastoma",
    "neuromas": "neuroma",
    "pains": "pain",
    "pyodermas": "pyoderma",
    "solid tumor": "solid tumors",
    "toothaches": "toothache",
    "pressure ulcers": "pressure ulcer",
    # Additional modifier removals (1-drug diseases)
    "acute heart failure": "heart failure",
    "acute lead poisoning": "lead poisoning",
    "acute mania": "mania",
    "acute urticaria": "urticaria",
    "chronic angina": "angina",
    "chronic cancer pain": "cancer pain",
    "chronic emphysema": "emphysema",
    "chronic gout": "gout",
    "chronic insomnia": "insomnia",
    "chronic migraine": "migraine",
    "mild hyperthyroidism": "hyperthyroidism",
}


def normalize_disease_name(name: str) -> str:
    """Normalize a disease name for matching."""
    if not name:
        return ""

    # Lowercase and strip whitespace
    name = name.lower().strip()

    # Remove possessive apostrophes
    name = name.replace("'s", "s")
    name = name.replace("'", "")

    # Normalize hyphens and spaces
    name = re.sub(r'[-_]', ' ', name)
    name = re.sub(r'\s+', ' ', name)

    # Remove trailing punctuation
    name = name.rstrip('.,;:')

    return name


def get_canonical_name(disease_name: str) -> str:
    """Get the canonical disease name for mapping."""
    normalized = normalize_disease_name(disease_name)

    # Check exact match in synonyms
    if normalized in DISEASE_SYNONYMS:
        return DISEASE_SYNONYMS[normalized]

    # Only check for longer synonyms (>=6 chars) to avoid false substring matches
    # Short synonyms like "ra", "as", "mm" would match inside unrelated words
    # Also require the query to be >=6 chars when checking if query is IN synonym
    # to prevent "mi" matching inside "relapsing-remitting multiple sclerosis" (h712 bug fix)
    for synonym, canonical in DISEASE_SYNONYMS.items():
        if len(synonym) >= 6:
            if synonym in normalized:
                return canonical
            if len(normalized) >= 6 and normalized in synonym:
                return canonical

    return normalized


class DiseaseMatcher:
    """Match disease names to MESH IDs with fuzzy matching."""

    def __init__(self, mesh_mappings: Dict[str, str]):
        """
        Initialize with MESH mappings.

        Args:
            mesh_mappings: Dict mapping disease names to MESH IDs
        """
        self.mesh_mappings = mesh_mappings
        self.normalized_mappings: Dict[str, str] = {}
        self.canonical_to_mesh: Dict[str, str] = {}

        self._build_index()

    def _build_index(self):
        """Build normalized index for fast lookup."""
        for disease_name, mesh_id in self.mesh_mappings.items():
            normalized = normalize_disease_name(disease_name)
            canonical = get_canonical_name(disease_name)

            self.normalized_mappings[normalized] = mesh_id
            self.canonical_to_mesh[canonical] = mesh_id

    def get_mesh_id(self, disease_name: str) -> Optional[str]:
        """
        Get MESH ID for a disease name.

        Args:
            disease_name: The disease name to look up

        Returns:
            MESH ID if found, None otherwise
        """
        normalized = normalize_disease_name(disease_name)

        # 1. Try exact normalized match
        if normalized in self.normalized_mappings:
            return self.normalized_mappings[normalized]

        # 2. Try canonical name match
        canonical = get_canonical_name(disease_name)
        if canonical in self.canonical_to_mesh:
            return self.canonical_to_mesh[canonical]

        # 3. Try finding canonical in mappings
        for mapped_name, mesh_id in self.normalized_mappings.items():
            mapped_canonical = get_canonical_name(mapped_name)
            if canonical == mapped_canonical:
                return mesh_id

        # 4. Try substring matching for very long disease names (>20 chars)
        # and only if both strings are long enough to avoid false matches
        if len(normalized) > 20:
            for mapped_name, mesh_id in self.normalized_mappings.items():
                if len(mapped_name) > 10:  # Only match against longer disease names
                    if mapped_name in normalized or normalized in mapped_name:
                        return mesh_id

        return None

    def get_all_mappings(self, disease_names: Set[str]) -> Dict[str, str]:
        """
        Get MESH mappings for a set of disease names.

        Args:
            disease_names: Set of disease names to map

        Returns:
            Dict mapping disease names to MESH IDs
        """
        result = {}
        for name in disease_names:
            mesh_id = self.get_mesh_id(name)
            if mesh_id:
                result[name] = mesh_id
        return result

    def coverage_report(self, disease_names: Set[str]) -> Dict:
        """Generate coverage report for disease names."""
        mapped = 0
        unmapped = []

        for name in disease_names:
            if self.get_mesh_id(name):
                mapped += 1
            else:
                unmapped.append(name)

        return {
            "total": len(disease_names),
            "mapped": mapped,
            "unmapped": len(unmapped),
            "coverage": mapped / len(disease_names) if disease_names else 0,
            "unmapped_diseases": unmapped[:50],  # First 50 unmapped
        }


def load_mesh_mappings() -> Dict[str, str]:
    """Load MESH mappings from hardcoded + agent sources."""
    PROJECT_ROOT = Path(__file__).parent.parent
    REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"

    # Hardcoded mappings (from train_gb_enhanced.py)
    hardcoded = {
        "hiv infection": "drkg:Disease::MESH:D015658",
        "hepatitis c": "drkg:Disease::MESH:D006526",
        "tuberculosis": "drkg:Disease::MESH:D014376",
        "breast cancer": "drkg:Disease::MESH:D001943",
        "lung cancer": "drkg:Disease::MESH:D008175",
        "colorectal cancer": "drkg:Disease::MESH:D015179",
        "hypertension": "drkg:Disease::MESH:D006973",
        "heart failure": "drkg:Disease::MESH:D006333",
        "atrial fibrillation": "drkg:Disease::MESH:D001281",
        "epilepsy": "drkg:Disease::MESH:D004827",
        "parkinson disease": "drkg:Disease::MESH:D010300",
        "alzheimer disease": "drkg:Disease::MESH:D000544",
        "rheumatoid arthritis": "drkg:Disease::MESH:D001172",
        "multiple sclerosis": "drkg:Disease::MESH:D009103",
        "psoriasis": "drkg:Disease::MESH:D011565",
        "type 2 diabetes mellitus": "drkg:Disease::MESH:D003924",
        "obesity": "drkg:Disease::MESH:D009765",
        "asthma": "drkg:Disease::MESH:D001249",
        "copd": "drkg:Disease::MESH:D029424",
        "osteoporosis": "drkg:Disease::MESH:D010024",
        "myocardial infarction": "drkg:Disease::MESH:D009203",
        "stroke": "drkg:Disease::MESH:D020521",
        "ulcerative colitis": "drkg:Disease::MESH:D003093",
        "schizophrenia": "drkg:Disease::MESH:D012559",
        "major depressive disorder": "drkg:Disease::MESH:D003865",
        "osteoarthritis": "drkg:Disease::MESH:D010003",
        "atopic eczema": "drkg:Disease::MESH:D003876",  # Added for atopic dermatitis
        "atopic dermatitis": "drkg:Disease::MESH:D003876",  # Explicit mapping
        "ankylosing spondylitis": "drkg:Disease::MESH:D013167",
        "multiple myeloma": "drkg:Disease::MESH:D009101",
        "crohn disease": "drkg:Disease::MESH:D003424",
        "allergic rhinitis": "drkg:Disease::MESH:D065631",
        "acne vulgaris": "drkg:Disease::MESH:D000152",
        # h712: Top unmapped EC disease names (by GT drug count)
        # Group 1: Diseases already in predictor under different names (synonyms)
        "diabetic kidney disease": "drkg:Disease::MESH:D003928",  # = diabetic nephropathy
        "depression": "drkg:Disease::MESH:D003865",  # = major depressive disorder
        "septicemia": "drkg:Disease::MESH:D018805",  # = sepsis
        "renal cell carcinoma": "drkg:Disease::MESH:D002292",  # = renal carcinoma
        "regional enteritis": "drkg:Disease::MESH:D003424",  # = Crohn disease
        "hodgkins disease": "drkg:Disease::MESH:D006689",  # = Hodgkin Disease
        "hodgkin lymphoma": "drkg:Disease::MESH:D006689",
        "severe psoriasis": "drkg:Disease::MESH:D011565",  # = psoriasis
        "openangle glaucoma": "drkg:Disease::MESH:D005902",  # = open-angle glaucoma
        "open angle glaucoma": "drkg:Disease::MESH:D005902",
        # Group 2: New diseases to add to predictor
        "emphysema": "drkg:Disease::MESH:D004646",
        "pulmonary emphysema": "drkg:Disease::MESH:D004646",
        "ocular hypertension": "drkg:Disease::MESH:D009798",
        "meningitis": "drkg:Disease::MESH:D008581",
        "mycosis fungoides": "drkg:Disease::MESH:D009182",
        "insomnia": "drkg:Disease::MESH:D007319",
        "hepatocellular carcinoma": "drkg:Disease::MESH:D006528",
        "migraine": "drkg:Disease::MESH:D008881",
        "migraine disorders": "drkg:Disease::MESH:D008881",
        "migraine without aura": "drkg:Disease::MESH:D008881",
        "migraine with aura": "drkg:Disease::MESH:D008881",  # Map to parent for GT purposes
        "erosive esophagitis": "drkg:Disease::MESH:D004942",
        "esophagitis": "drkg:Disease::MESH:D004942",
        "acute gouty arthritis": "drkg:Disease::MESH:D006073",
        "gout": "drkg:Disease::MESH:D006073",
        "chronic renal failure": "drkg:Disease::MESH:D051436",
        "chronic kidney disease": "drkg:Disease::MESH:D051436",
        "gastroesophageal reflux disease": "drkg:Disease::MESH:D005764",
        "gerd": "drkg:Disease::MESH:D005764",
        "serum sickness": "drkg:Disease::MESH:D012713",
        "iritis": "drkg:Disease::MESH:D007500",
        "idiopathic thrombocytopenic purpura": "drkg:Disease::MESH:D016553",
        "itp": "drkg:Disease::MESH:D016553",
        "leukemia": "drkg:Disease::MESH:D007938",
        "leukemias": "drkg:Disease::MESH:D007938",
        "arthritis": "drkg:Disease::MESH:D001168",
        "benign prostatic hyperplasia": "drkg:Disease::MESH:D011470",
        "bph": "drkg:Disease::MESH:D011470",
        "pheochromocytoma": "drkg:Disease::MESH:D010673",
        "mixed dyslipidemia": "drkg:Disease::MESH:D050171",
        "hypertriglyceridemia": "drkg:Disease::MESH:D015228",
        "iron deficiency anemia": "drkg:Disease::MESH:D018798",
        "status epilepticus": "drkg:Disease::MESH:D013226",
        "acne rosacea": "drkg:Disease::MESH:D012393",
        "rosacea": "drkg:Disease::MESH:D012393",
        "h pylori infection": "drkg:Disease::MESH:D016481",
        "helicobacter pylori infection": "drkg:Disease::MESH:D016481",
        "hyperphosphatemia": "drkg:Disease::MESH:D054559",
        "primary dysmenorrhea": "drkg:Disease::MESH:D004412",
        "dysmenorrhea": "drkg:Disease::MESH:D004412",
        "chancroid": "drkg:Disease::MESH:D002602",
        "anthrax": "drkg:Disease::MESH:D000881",
        "tinea cruris": "drkg:Disease::MESH:D014005",
        "tinea pedis": "drkg:Disease::MESH:D014006",
        "uveitis": "drkg:Disease::MESH:D014606",
        "anterior uveitis": "drkg:Disease::MESH:D014606",
        "chronic anterior uveitis": "drkg:Disease::MESH:D014606",
        "urinary incontinence": "drkg:Disease::MESH:D014549",
        "postherpetic neuralgia": "drkg:Disease::MESH:D051474",
        "age related macular degeneration": "drkg:Disease::MESH:D008268",
        "agerelated macular degeneration": "drkg:Disease::MESH:D008268",
        "macular degeneration": "drkg:Disease::MESH:D008268",
        # h712 (continued): New disease MESH mappings for previously unmapped EC diseases
        # Symptoms that are treated as diseases in DRKG
        "nausea": "drkg:Disease::MESH:D009325",
        "vomiting": "drkg:Disease::MESH:D014839",
        "acute pain": "drkg:Disease::MESH:D059787",
        "postoperative pain": "drkg:Disease::MESH:D010149",
        "bronchospasm": "drkg:Disease::MESH:D001986",
        # Infections
        "intraabdominal infections": "drkg:Disease::MESH:D059413",
        "intra abdominal infections": "drkg:Disease::MESH:D059413",
        "respiratory tract infections": "drkg:Disease::MESH:D012141",
        "osteomyelitis": "drkg:Disease::MESH:D010019",
        "bone diseases infectious": "drkg:Disease::MESH:D001170",
        "tuberculous meningitis": "drkg:Disease::MESH:D014390",
        "tuberculosis meningeal": "drkg:Disease::MESH:D014390",
        "trachoma": "drkg:Disease::MESH:D014141",
        "trichinosis": "drkg:Disease::MESH:D014235",
        "conjunctivitis": "drkg:Disease::MESH:D003234",
        # Allergy/Immune
        "hypersensitivity": "drkg:Disease::MESH:D006967",
        "drug hypersensitivity": "drkg:Disease::MESH:D004342",
        "drug hypersensitivity reactions": "drkg:Disease::MESH:D004342",
        "stevens johnson syndrome": "drkg:Disease::MESH:D013262",
        "berylliosis": "drkg:Disease::MESH:D001607",
        # Psychiatric/Neurological
        "bipolar disorder": "drkg:Disease::MESH:D001714",
        "anxiety disorder": "drkg:Disease::MESH:D001008",
        "anxiety disorders": "drkg:Disease::MESH:D001008",
        # GI
        "gastric ulcer": "drkg:Disease::MESH:D013276",
        "stomach ulcer": "drkg:Disease::MESH:D013276",
        "liver cirrhosis": "drkg:Disease::MESH:D008103",
        # Cancer/Hematological
        "lymphoma": "drkg:Disease::MESH:D008223",
        "non hodgkins lymphoma": "drkg:Disease::MESH:D008228",
        "precursor cell lymphoblastic leukemia": "drkg:Disease::MESH:D054198",
        "acute lymphoblastic leukemia": "drkg:Disease::MESH:D054198",
        "hemophilia": "drkg:Disease::MESH:D006467",
        "neutropenia": "drkg:Disease::MESH:D009503",
        # Eye
        "corneal injury": "drkg:Disease::MESH:D003316",
        "corneal injuries": "drkg:Disease::MESH:D003316",
        "sympathetic ophthalmia": "drkg:Disease::MESH:D013577",
        "corneal ulcer": "drkg:Disease::MESH:D003320",
        # Metabolic/Endocrine
        "hypokalemia": "drkg:Disease::MESH:D007008",
        "hypercalcemia": "drkg:Disease::MESH:D006934",
        "thyroiditis subacute": "drkg:Disease::MESH:D013967",
        # Rheumatology
        "epicondylitis": "drkg:Disease::MESH:D013716",
        "tennis elbow": "drkg:Disease::MESH:D013716",
        "bursitis": "drkg:Disease::MESH:D002062",
        "rheumatic heart disease": "drkg:Disease::MESH:D012214",
        # Reproductive
        "endometriosis": "drkg:Disease::MESH:D004715",
        "infertility": "drkg:Disease::MESH:D007246",
        "female infertility": "drkg:Disease::MESH:D007247",
        "male infertility": "drkg:Disease::MESH:D007248",
        # Other
        "dermatitis": "drkg:Disease::MESH:D003872",
        "corticosteroid responsive dermatoses": "drkg:Disease::MESH:D003872",
        "fanconi anemia": "drkg:Disease::MESH:D029503",
        # h712 session 2: New MESH mappings for previously unmapped canonical names
        # All verified present in DRKG with embeddings
        "hyperkalemia": "drkg:Disease::MESH:D006947",
        "hypotension": "drkg:Disease::MESH:D007022",
        "panic disorder": "drkg:Disease::MESH:D016584",
        "erectile dysfunction": "drkg:Disease::MESH:D007172",
        "impotence": "drkg:Disease::MESH:D007172",
        "pertussis": "drkg:Disease::MESH:D014917",
        "whooping cough": "drkg:Disease::MESH:D014917",
        "peptic ulcer": "drkg:Disease::MESH:D010437",
        "peptic ulcer disease": "drkg:Disease::MESH:D010437",
        "hemophilia b": "drkg:Disease::MESH:D002836",
        "christmas disease": "drkg:Disease::MESH:D002836",
        "gastrointestinal stromal tumor": "drkg:Disease::MESH:D024821",
        "gist": "drkg:Disease::MESH:D024821",
        "tinea versicolor": "drkg:Disease::MESH:D014009",
        "pityriasis versicolor": "drkg:Disease::MESH:D014009",
        "lung abscess": "drkg:Disease::MESH:D008171",
        "back pain": "drkg:Disease::MESH:D001416",
        "low back pain": "drkg:Disease::MESH:D017116",
        "tension type headache": "drkg:Disease::MESH:D018781",
        "headache": "drkg:Disease::MESH:D006261",
        "sprains and strains": "drkg:Disease::MESH:D013180",
        "myalgia": "drkg:Disease::MESH:D063806",
        "muscle pain": "drkg:Disease::MESH:D063806",
        "ponv": "drkg:Disease::MESH:D020250",
        "postoperative nausea and vomiting": "drkg:Disease::MESH:D020250",
        "actinic keratosis": "drkg:Disease::MESH:D055623",
        "colon cancer": "drkg:Disease::MESH:D003110",
        "colonic neoplasms": "drkg:Disease::MESH:D003110",
        "peritoneal neoplasms": "drkg:Disease::MESH:D010538",
        "parkinsonism": "drkg:Disease::MESH:D020734",
        "secondary parkinsonism": "drkg:Disease::MESH:D020734",
        "graft rejection": "drkg:Disease::MESH:D006086",
        "transplant rejection": "drkg:Disease::MESH:D006086",
        "metabolic alkalosis": "drkg:Disease::MESH:D001259",
        "alkalosis": "drkg:Disease::MESH:D001259",
        "pulmonary eosinophilia": "drkg:Disease::MESH:D011657",
        "eosinophilic pneumonia": "drkg:Disease::MESH:D015518",
        "vulvovaginal atrophy": "drkg:Disease::MESH:D064129",
        "vaginal atrophy": "drkg:Disease::MESH:D064129",
        "thyroid nodule": "drkg:Disease::MESH:D016606",
        "paget disease of bone": "drkg:Disease::MESH:D010001",
        "pagets disease": "drkg:Disease::MESH:D010001",
        "rocky mountain spotted fever": "drkg:Disease::MESH:D012373",
        "lymphogranuloma venereum": "drkg:Disease::MESH:D008219",
        "bacteremia": "drkg:Disease::MESH:D016470",
        "bacteraemia": "drkg:Disease::MESH:D016470",
        "septicaemia": "drkg:Disease::MESH:D018805",
        "influenza": "drkg:Disease::MESH:D007251",
        "exfoliative dermatitis": "drkg:Disease::MESH:D003877",
        "erythroderma": "drkg:Disease::MESH:D003877",
        "skin diseases infectious": "drkg:Disease::MESH:D001424",  # Bacterial infections (SSSI)
        "candidemia": "drkg:Disease::MESH:D058387",
        "acute glomerulonephritis": "drkg:Disease::MESH:D005921",
        "glomerulonephritis": "drkg:Disease::MESH:D005921",
        "iron deficiency": "drkg:Disease::MESH:D018798",  # Iron deficiency anemia (D007501 not in DRKG)
        "anemia": "drkg:Disease::MESH:D000740",
        "prostatitis": "drkg:Disease::MESH:D011472",
        "urethritis": "drkg:Disease::MESH:D014526",
        "onychomycosis": "drkg:Disease::MESH:D014006",  # Map to tinea pedis (dermatophyte; D014007 not in DRKG)
        "addison disease": "drkg:Disease::MESH:D000224",
        # h712 session 2 fixes: Direct mappings for synonym-chain failures
        # (canonical resolution can break when intermediate canonicals get further resolved)
        "iridocyclitis": "drkg:Disease::MESH:D014606",  # anterior uveitis
        "cyclitis": "drkg:Disease::MESH:D014606",        # ciliary body inflammation
        "chorioretinitis": "drkg:Disease::MESH:D014605",  # posterior uveitis
        "lupus nephritis": "drkg:Disease::MESH:D008181",
        "myelofibrosis": "drkg:Disease::MESH:D055728",
        "primary myelofibrosis": "drkg:Disease::MESH:D055728",
        "vasomotor rhinitis": "drkg:Disease::MESH:D012223",
        # h712 session 3: Comprehensive unmapped EC disease MESH mappings
        # 163 new mappings covering diseases with >= 2 drugs in EC GT
        # All verified present in DRKG
        # --- Symptoms treated as diseases ---
        "cancer pain": "drkg:Disease::MESH:D010146",
        "severe pain": "drkg:Disease::MESH:D010146",
        "ocular pain": "drkg:Disease::MESH:D058447",
        "cough": "drkg:Disease::MESH:D003371",
        "fever": "drkg:Disease::MESH:D005334",
        "toothache": "drkg:Disease::MESH:D014098",
        "heartburn": "drkg:Disease::MESH:D006356",
        "anorexia": "drkg:Disease::MESH:D000855",
        "dysuria": "drkg:Disease::MESH:D053159",
        "headaches": "drkg:Disease::MESH:D006261",
        "muscle stiffness": "drkg:Disease::MESH:D009128",
        "spasticity": "drkg:Disease::MESH:D009128",
        "respiratory depression": "drkg:Disease::MESH:D012131",
        # --- Urinary ---
        "urinary frequency": "drkg:Disease::MESH:D014555",
        "urinary urgency": "drkg:Disease::MESH:D053201",
        "urge incontinence": "drkg:Disease::MESH:D014549",
        "bacteriuria": "drkg:Disease::MESH:D001437",
        # --- Hematological ---
        "erythroblastopenia": "drkg:Disease::MESH:D012010",
        "rbc anemia": "drkg:Disease::MESH:D000740",
        "congenital hypoplastic anemia": "drkg:Disease::MESH:D029503",
        "anemias of pregnancy": "drkg:Disease::MESH:D000740",
        "megaloblastic anemias": "drkg:Disease::MESH:D000740",
        "diamondblackfan anemia": "drkg:Disease::MESH:D029503",
        "sickle cell disease": "drkg:Disease::MESH:D000755",
        "transfusion reactions": "drkg:Disease::MESH:D065227",
        "haemophilia a": "drkg:Disease::MESH:D006467",
        # --- Infections ---
        "pelvic cellulitis": "drkg:Disease::MESH:D034161",
        "streptococcus pneumoniae": "drkg:Disease::MESH:D011018",
        "streptococcus pneumoniae infections": "drkg:Disease::MESH:D011018",
        "abscesses": "drkg:Disease::MESH:D000038",
        "chronic hepatitis b": "drkg:Disease::MESH:D006509",
        "legionella infection": "drkg:Disease::MESH:D007877",
        "typhus fever": "drkg:Disease::MESH:D014438",
        "tick fevers": "drkg:Disease::MESH:D013984",
        "enterobacter aerogenes infections": "drkg:Disease::MESH:D004756",
        "shigella infections": "drkg:Disease::MESH:D004405",
        "haemophilus influenzae infections": "drkg:Disease::MESH:D006192",
        "bacterial vaginosis": "drkg:Disease::MESH:D016585",
        "genital herpes": "drkg:Disease::MESH:D006558",
        "septic shock": "drkg:Disease::MESH:D012772",
        "postoperative infections": "drkg:Disease::MESH:D013530",
        "cmv retinitis": "drkg:Disease::MESH:D017726",
        "pneumonic plague": "drkg:Disease::MESH:D010930",
        "septicemic plague": "drkg:Disease::MESH:D010930",
        "invasive candidiasis": "drkg:Disease::MESH:D058387",
        "psittacosis": "drkg:Disease::MESH:D009956",
        "campylobacter fetus infections": "drkg:Disease::MESH:D002169",
        "skin infections": "drkg:Disease::MESH:D017192",
        "uncomplicated gonorrhea": "drkg:Disease::MESH:D006069",
        "herpes labialis": "drkg:Disease::MESH:D006560",
        "klebsiella infections": "drkg:Disease::MESH:D007710",
        "acute pyelonephritis": "drkg:Disease::MESH:D011704",
        "cellulitis": "drkg:Disease::MESH:D002481",
        "clostridioides difficile infection": "drkg:Disease::MESH:D003015",
        "measles": "drkg:Disease::MESH:D008457",
        "nontuberculous mycobacteriosis": "drkg:Disease::MESH:D009165",
        "superficial ocular infections": "drkg:Disease::MESH:D015818",
        "wound sepsis": "drkg:Disease::MESH:D014946",
        "skin and skinstructure infections": "drkg:Disease::MESH:D001424",
        # --- Neurological/Psychiatric ---
        "dementia of the alzheimers type": "drkg:Disease::MESH:D000544",
        "pd": "drkg:Disease::MESH:D010300",
        "paralysis agitans": "drkg:Disease::MESH:D010300",
        "partial onset seizures": "drkg:Disease::MESH:D004827",
        "partialonset seizures": "drkg:Disease::MESH:D004827",
        "endogenous depression": "drkg:Disease::MESH:D003866",
        "treatment resistant depression": "drkg:Disease::MESH:D061218",
        "adhd": "drkg:Disease::MESH:D001289",
        "tourettes disorder": "drkg:Disease::MESH:D005879",
        "social phobia": "drkg:Disease::MESH:D010698",
        "psychotic disorders": "drkg:Disease::MESH:D011618",
        "opioid dependence": "drkg:Disease::MESH:D009293",
        "opioid overdose": "drkg:Disease::MESH:D009293",
        "chorea associated with huntingtons disease": "drkg:Disease::MESH:D006816",
        "migraine attacks with aura": "drkg:Disease::MESH:D008881",
        "migraine headache": "drkg:Disease::MESH:D008881",
        "cluster headache": "drkg:Disease::MESH:D003027",
        "restless legs syndrome": "drkg:Disease::MESH:D012148",
        "dravet syndrome": "drkg:Disease::MESH:D004831",
        "mania": "drkg:Disease::MESH:D001714",
        # --- Cardiovascular ---
        "atherosclerotic vascular disease": "drkg:Disease::MESH:D050197",
        "chd": "drkg:Disease::MESH:D003327",
        "transient ischemic attack": "drkg:Disease::MESH:D002546",
        "hypokalemic familial periodic paralysis": "drkg:Disease::MESH:D020514",
        "wolffparkinsonwhite syndrome": "drkg:Disease::MESH:D014927",
        # --- GI ---
        "irritable bowel syndrome": "drkg:Disease::MESH:D043183",
        "gastric ulcers": "drkg:Disease::MESH:D013276",
        "benign gastric ulcer": "drkg:Disease::MESH:D013276",
        "cirrhosis of the liver": "drkg:Disease::MESH:D008103",
        "hepatic disease": "drkg:Disease::MESH:D008107",
        "tropical sprue": "drkg:Disease::MESH:D013182",
        "nontropical sprue": "drkg:Disease::MESH:D002446",
        "hemorrhoids": "drkg:Disease::MESH:D006484",
        "anal fissures": "drkg:Disease::MESH:D005401",
        # --- Musculoskeletal ---
        "fibromyalgia": "drkg:Disease::MESH:D005356",
        "tendonitis": "drkg:Disease::MESH:D052256",
        "osteoporotic fracture": "drkg:Disease::MESH:D058866",
        # --- Cancer ---
        "malignant melanoma": "drkg:Disease::MESH:D008545",
        "cutaneous tcell lymphoma": "drkg:Disease::MESH:D016410",
        "nonhodgkins lymphoma": "drkg:Disease::MESH:D008228",
        "nonhodgkin lymphoma": "drkg:Disease::MESH:D008228",
        "lymphocytic lymphoma": "drkg:Disease::MESH:D008228",
        "acute lymphocytic leukemia": "drkg:Disease::MESH:D054198",
        "meningeal leukemia": "drkg:Disease::MESH:D007938",
        "aidsrelated kaposis sarcoma": "drkg:Disease::MESH:D012514",
        "kaposis sarcoma": "drkg:Disease::MESH:D012514",
        "diffuse large bcell lymphoma": "drkg:Disease::MESH:D016403",
        "waldenstrms macroglobulinemia": "drkg:Disease::MESH:D008258",
        "endometrial carcinoma": "drkg:Disease::MESH:D016889",
        "carcinoma of the breast": "drkg:Disease::MESH:D001943",
        "metastatic nsclc": "drkg:Disease::MESH:D008175",
        "nonsquamous nsclc": "drkg:Disease::MESH:D008175",
        "metastatic hcc": "drkg:Disease::MESH:D006528",
        "anal cancer": "drkg:Disease::MESH:D001005",
        "solid tumors": "drkg:Disease::MESH:D009369",
        # --- Metabolic/Endocrine ---
        "cretinism": "drkg:Disease::MESH:D003409",
        "cushings syndrome": "drkg:Disease::MESH:D003480",
        "hyponatremia": "drkg:Disease::MESH:D007010",
        "hypocalcemia": "drkg:Disease::MESH:D006996",
        "metabolic acidosis": "drkg:Disease::MESH:D000138",
        "folate deficiency": "drkg:Disease::MESH:D005494",
        "hyperuricemia": "drkg:Disease::MESH:D033461",
        "hypophosphatemia": "drkg:Disease::MESH:D017674",
        "myxedema coma": "drkg:Disease::MESH:D009230",
        "multiple endocrine adenomas": "drkg:Disease::MESH:D009377",
        "pompe disease": "drkg:Disease::MESH:D006008",
        "wilsons disease": "drkg:Disease::MESH:D006527",
        "growth failure": "drkg:Disease::MESH:D006130",
        # --- Renal ---
        "endstage kidney disease": "drkg:Disease::MESH:D007676",
        "end stage renal failure": "drkg:Disease::MESH:D007676",
        "renal anemia": "drkg:Disease::MESH:D000740",
        # --- Dermatological ---
        "keloids": "drkg:Disease::MESH:D007627",
        "necrobiosis lipoidica diabeticorum": "drkg:Disease::MESH:D009335",
        "insect bites": "drkg:Disease::MESH:D007299",
        # --- Respiratory ---
        "severe asthma": "drkg:Disease::MESH:D001249",
        "acute bronchospasm": "drkg:Disease::MESH:D001986",
        # --- Reproductive ---
        "premenstrual dysphoric disorder": "drkg:Disease::MESH:D065446",
        "abnormal uterine bleeding": "drkg:Disease::MESH:D008796",
        "anovulation": "drkg:Disease::MESH:D000858",
        "amenorrhea": "drkg:Disease::MESH:D000568",
        "cryptorchidism": "drkg:Disease::MESH:D003456",
        "orchitis": "drkg:Disease::MESH:D009920",
        "eclampsia": "drkg:Disease::MESH:D004461",
        # --- Eye ---
        "dry eye disease": "drkg:Disease::MESH:D015352",
        "superficial punctate keratitis": "drkg:Disease::MESH:D007634",
        "diabetic macular edema": "drkg:Disease::MESH:D008269",
        # --- Rheumatic ---
        "juvenile arthritis": "drkg:Disease::MESH:D001171",
        "gout flares": "drkg:Disease::MESH:D006073",
        # --- Neurological Pain ---
        "peripheral neuropathic pain": "drkg:Disease::MESH:D009437",
        "diabetic peripheral neuropathy": "drkg:Disease::MESH:D003929",
        "diabetic peripheral neuropathic pain": "drkg:Disease::MESH:D003929",
        # --- Other ---
        "cerebral edema associated with primary brain tumor": "drkg:Disease::MESH:D001929",
        "generalized edema": "drkg:Disease::MESH:D004487",
        "edema due to pathologic causes": "drkg:Disease::MESH:D004487",
        "vasomotor symptoms associated with menopause": "drkg:Disease::MESH:D019584",
        "influenza b virus infection": "drkg:Disease::MESH:D007251",
        "laryngopharyngitis": "drkg:Disease::MESH:D010612",
        "palmoplantar pustulosis": "drkg:Disease::MESH:D011565",
        "acute sinusitis": "drkg:Disease::MESH:D012852",
        "acute maxillary sinusitis": "drkg:Disease::MESH:D015523",
        "acute otitis externa": "drkg:Disease::MESH:D010032",
        "cervicitis": "drkg:Disease::MESH:D002575",
        "hiv1": "drkg:Disease::MESH:D015658",
        "upper respiratory allergies": "drkg:Disease::MESH:D012220",
        # h712s4: New MESH mappings for high-value unmapped EC diseases
        "alcoholism": "drkg:Disease::MESH:D000437",
        "pulmonary edema": "drkg:Disease::MESH:D011654",
        "menorrhagia": "drkg:Disease::MESH:D008595",
        "retinal vein occlusion": "drkg:Disease::MESH:D012170",
        "genital warts": "drkg:Disease::MESH:D003218",  # condylomata acuminata
        "kidney diseases": "drkg:Disease::MESH:D007674",
        "myelodysplastic syndromes": "drkg:Disease::MESH:D009190",
        "primary gout": "drkg:Disease::MESH:D006073",  # = gout (synonym chain bypassed)
        # h712s4: Direct MESH mappings for diseases with canonical chaining issues
        "peptic esophagitis": "drkg:Disease::MESH:D005764",  # = GERD
        "post operative pain": "drkg:Disease::MESH:D010149",  # = postoperative pain (specific)
        "post-operative pain": "drkg:Disease::MESH:D010149",
        "acute bronchitis": "drkg:Disease::MESH:D001991",  # = bronchitis
        "pancreatic carcinoma non resectable": "drkg:Disease::MESH:D010190",  # = pancreatic neoplasms
        "variant angina": "drkg:Disease::MESH:D000788",  # = angina pectoris, variant (specific)
        "hodgkin disease": "drkg:Disease::MESH:D006689",  # = Hodgkin disease
        "muscular headache": "drkg:Disease::MESH:D018781",  # = tension-type headache
        "primary peritoneal carcinoma": "drkg:Disease::MESH:D010534",  # = peritoneal neoplasms
        # h712s5: Additional MESH mappings for remaining unmapped EC diseases (47 entries)
        # All verified present in DRKG as of 2026-02-07
        # --- Infections ---
        "rickettsialpox": "drkg:Disease::MESH:D012282",
        "mycoplasma pneumoniae": "drkg:Disease::MESH:D011019",
        "acinetobacter infections": "drkg:Disease::MESH:D000151",
        "rubella": "drkg:Disease::MESH:D012409",
        "rubella ": "drkg:Disease::MESH:D012409",
        "mycobacterium avium complex mac": "drkg:Disease::MESH:D015270",
        "mycobacterium avium complex mac ": "drkg:Disease::MESH:D015270",
        "respiratory syncytial virus lower respiratory tract disease": "drkg:Disease::MESH:D018357",
        "rsv infection": "drkg:Disease::MESH:D018357",
        "amebic liver abscess": "drkg:Disease::MESH:D008101",
        "anaerobic vaginosis": "drkg:Disease::MESH:D016585",
        "septic abortion": "drkg:Disease::MESH:D000033",
        # --- Renal ---
        "renal impairment": "drkg:Disease::MESH:D051437",
        "esrd": "drkg:Disease::MESH:D007676",
        "neurogenic bladder": "drkg:Disease::MESH:D001750",
        "neurogenic bladder ": "drkg:Disease::MESH:D001750",
        # --- Oncology ---
        "rhabdomyosarcoma": "drkg:Disease::MESH:D012208",
        "alveolar soft part sarcoma": "drkg:Disease::MESH:D018234",
        "early gastric cancer": "drkg:Disease::MESH:D013274",
        "early gastric cancer ": "drkg:Disease::MESH:D013274",
        "perianal warts": "drkg:Disease::MESH:D003218",
        # --- Neurological ---
        "essential tremor": "drkg:Disease::MESH:D020329",
        "excessive daytime sleepiness": "drkg:Disease::MESH:D006970",
        "myoclonic seizures": "drkg:Disease::MESH:D004831",
        "infantile spasms": "drkg:Disease::MESH:D013036",
        "extrapyramidal disorders": "drkg:Disease::MESH:D001480",
        "alzheimers type dementia": "drkg:Disease::MESH:D000544",
        # --- Cardiovascular ---
        "effortassociated angina": "drkg:Disease::MESH:D000787",
        "effort associated angina": "drkg:Disease::MESH:D000787",
        "aortitis syndrome": "drkg:Disease::MESH:D013625",
        # --- Metabolic ---
        "primary dysbetalipoproteinemia fredrickson type iii": "drkg:Disease::MESH:D006952",
        "homozygous familial hypercholesterolaemia": "drkg:Disease::MESH:D006938",
        "niemannpick disease type c npc": "drkg:Disease::MESH:D052556",
        "niemann pick disease type c": "drkg:Disease::MESH:D052556",
        "alagille syndrome": "drkg:Disease::MESH:D016738",
        # --- GI ---
        "ascites": "drkg:Disease::MESH:D001201",
        # --- Rheumatic ---
        "hereditary transthyretinmediated amyloidosis": "drkg:Disease::MESH:D028227",
        "hereditary transthyretin mediated amyloidosis": "drkg:Disease::MESH:D028227",
        # --- Dermatological ---
        "warts": "drkg:Disease::MESH:D014860",
        "dystrophic epidermolysis bullosa": "drkg:Disease::MESH:D016109",
        # --- Allergy/Immune ---
        "allergic conditions": "drkg:Disease::MESH:D006967",
        "eye infections": "drkg:Disease::MESH:D015818",
        # --- Reproductive ---
        "hot flushes": "drkg:Disease::MESH:D019584",
        "female hypogonadism": "drkg:Disease::MESH:D007006",
        "dyspareunia": "drkg:Disease::MESH:D004414",
        # --- Respiratory ---
        "reversible obstructive airway disease": "drkg:Disease::MESH:D001249",
        "heritable pah": "drkg:Disease::MESH:D006976",
        # --- Glomerular ---
        "primary immunoglobulin a nephropathy": "drkg:Disease::MESH:D005922",
        "immunoglobulin a nephropathy": "drkg:Disease::MESH:D005922",
        "iga nephropathy": "drkg:Disease::MESH:D005922",
        # --- Glaucoma ---
        "secondary glaucoma": "drkg:Disease::MESH:D005901",
        # h712s6 (h731 session): Final batch of direct MESH mappings
        "ventricular fibrillation": "drkg:Disease::MESH:D014693",
        "epilepsy, absence": "drkg:Disease::MESH:D004832",
        "absence epilepsy": "drkg:Disease::MESH:D004832",
        "epilepsy, generalized": "drkg:Disease::MESH:D004829",
        "generalized epilepsy": "drkg:Disease::MESH:D004829",
        "leukemia, myeloid, acute": "drkg:Disease::MESH:D015470",
        "acute myeloid leukemia": "drkg:Disease::MESH:D015470",
        "acute myelogenous leukemia": "drkg:Disease::MESH:D015470",
        "polycythemia vera": "drkg:Disease::MESH:D011087",
        "acquired polycythemia vera": "drkg:Disease::MESH:D011087",
        "schizoaffective disorder": "drkg:Disease::MESH:D011618",
        "psychotic disorders": "drkg:Disease::MESH:D011618",
        "neuromyelitis optica": "drkg:Disease::MESH:D009471",
        "methanol poisoning": "drkg:Disease::MESH:D000432",
        "acute ischemic stroke": "drkg:Disease::MESH:D020521",
        # Fix: "post herpetic neuralgia" synonym resolves incorrectly to generic neuralgia
        "post herpetic neuralgia": "drkg:Disease::MESH:D051474",

        # h712s7: Direct MESH mappings for unmapped EC disease synonyms
        # Varicella/Chickenpox
        "varicella zoster infection": "drkg:Disease::MESH:D002644",
        # Burns
        "burns": "drkg:Disease::MESH:D002056",
        # Embolism
        "embolism": "drkg:Disease::MESH:D004617",
        # Acute Kidney Injury
        "acute kidney injury": "drkg:Disease::MESH:D058186",
        # Hemorrhage
        "hemorrhage": "drkg:Disease::MESH:D006470",
        # Sjogren Syndrome
        "sjogren syndrome": "drkg:Disease::MESH:D012859",
        # Brain Neoplasms
        "brain neoplasms": "drkg:Disease::MESH:D001932",
        # Cerebrovascular Disorders
        "cerebrovascular disorders": "drkg:Disease::MESH:D002561",
        # Nasal Polyps
        "nasal polyps": "drkg:Disease::MESH:D009298",
        # Endometrial Hyperplasia
        "endometrial hyperplasia": "drkg:Disease::MESH:D004714",
        # Amenorrhea
        "amenorrhea": "drkg:Disease::MESH:D000568",
        # Cholelithiasis / Gallstones
        "cholelithiasis": "drkg:Disease::MESH:D002769",
        # Brain Edema
        "brain edema": "drkg:Disease::MESH:D001929",
        # Alcohol Withdrawal Delirium
        "alcohol withdrawal delirium": "drkg:Disease::MESH:D000430",
        # Tachycardia, Sinus
        "tachycardia, sinus": "drkg:Disease::MESH:D013616",
        "tachycardia": "drkg:Disease::MESH:D013610",
        "tachycardia, paroxysmal": "drkg:Disease::MESH:D013610",
        "tachycardia, supraventricular": "drkg:Disease::MESH:D013617",
        # Tinea
        "tinea": "drkg:Disease::MESH:D014005",
        # Mycoses
        "mycoses": "drkg:Disease::MESH:D009181",
        # Proctitis
        "proctitis": "drkg:Disease::MESH:D011349",
        # Gonorrhea
        "gonorrhea": "drkg:Disease::MESH:D006069",
        # Cholecystitis
        "cholecystitis": "drkg:Disease::MESH:D041881",
        # Prostatitis
        "prostatitis": "drkg:Disease::MESH:D011472",
        # Pneumocystis Pneumonia
        "pneumocystis pneumonia": "drkg:Disease::MESH:D011020",
        # Endometritis
        "endometritis": "drkg:Disease::MESH:D004716",
        # Attention Deficit Hyperactivity Disorder
        "attention deficit hyperactivity disorder": "drkg:Disease::MESH:D001289",
        # Migraine Disorders
        "migraine disorders": "drkg:Disease::MESH:D008881",
        # Tobacco Use Disorder
        "tobacco use disorder": "drkg:Disease::MESH:D014029",
        # Autistic Spectrum Disorder
        "autistic spectrum disorder": "drkg:Disease::MESH:D001321",
        # Liver Diseases
        "liver diseases": "drkg:Disease::MESH:D008107",
        # Urinary Bladder, Overactive
        "urinary bladder, overactive": "drkg:Disease::MESH:D053201",
        # Pemphigoid, Bullous
        "pemphigoid, bullous": "drkg:Disease::MESH:D010391",
        # Skin Diseases, Infectious
        "skin diseases, infectious": "drkg:Disease::MESH:D017192",
        # Dwarfism, Pituitary
        "dwarfism, pituitary": "drkg:Disease::MESH:D004393",
        # Cognitive Dysfunction
        "cognitive dysfunction": "drkg:Disease::MESH:D060825",
        # Meningococcal Infections
        "meningococcal infections": "drkg:Disease::MESH:D008585",
        # Myopia
        "myopia": "drkg:Disease::MESH:D009216",
        # Tremor
        "tremor": "drkg:Disease::MESH:D014202",
        # Intermittent Claudication
        "intermittent claudication": "drkg:Disease::MESH:D007383",
        # Bacterial Infections
        "bacterial infections": "drkg:Disease::MESH:D001424",
        # Liver Abscess
        "liver abscess": "drkg:Disease::MESH:D008100",
        # Apnea
        "apnea": "drkg:Disease::MESH:D001049",
        # Hypogonadism
        "hypogonadism": "drkg:Disease::MESH:D007006",
        # Synovitis
        "synovitis": "drkg:Disease::MESH:D013585",
        # Hypokalemia
        "hypokalemia": "drkg:Disease::MESH:D007008",
        # Parathyroid Neoplasms
        "parathyroid neoplasms": "drkg:Disease::MESH:D010282",
        # Miosis
        "miosis": "drkg:Disease::MESH:D015877",
        # Paraproteinemias
        "paraproteinemias": "drkg:Disease::MESH:D054219",
        # Cholestasis
        "cholestasis": "drkg:Disease::MESH:D002779",
        # Carotid Artery Diseases
        "carotid artery diseases": "drkg:Disease::MESH:D002340",
        # Ischemic Attack, Transient
        "ischemic attack, transient": "drkg:Disease::MESH:D002546",
        # Hot Flashes
        "hot flashes": "drkg:Disease::MESH:D019584",
        # Blastomycosis (not in DRKG but add mapping anyway)
        "blastomycosis": "drkg:Disease::MESH:D001760",
        # Cervicitis
        "cervicitis": "drkg:Disease::MESH:D002575",
        # Malabsorption Syndromes
        "malabsorption syndromes": "drkg:Disease::MESH:D008286",
        # Coinfection
        "coinfection": "drkg:Disease::MESH:D060085",
        # Clostridium Infections
        "clostridium infections": "drkg:Disease::MESH:D003015",
        # Haemophilus Infections
        "haemophilus infections": "drkg:Disease::MESH:D006192",
        # Digitalis Glycoside Poisoning (not in DRKG)
        "digitalis glycoside poisoning": "drkg:Disease::MESH:D004071",
        # Hepatitis D (not in DRKG)
        "hepatitis d": "drkg:Disease::MESH:D003698",
        # Diarrhea
        "diarrhea": "drkg:Disease::MESH:D003967",
        # Herpes Zoster (shingles)
        "herpes zoster": "drkg:Disease::MESH:D006562",
        # Candidiasis, vulvovaginal
        "candidiasis, vulvovaginal": "drkg:Disease::MESH:D002181",
        # Vaginitis
        "vaginitis": "drkg:Disease::MESH:D014627",
        # Candidiasis, oral
        "candidiasis, oral": "drkg:Disease::MESH:D002180",
        # Seasonal Affective Disorder
        "seasonal affective disorder": "drkg:Disease::MESH:D016574",
        # Panic Disorder
        "panic disorder": "drkg:Disease::MESH:D016584",
        # Stiff Person Syndrome
        "stiff person syndrome": "drkg:Disease::MESH:D016750",
        # Muscle Spasticity
        "muscle spasticity": "drkg:Disease::MESH:D009128",
        # Overweight
        "overweight": "drkg:Disease::MESH:D050177",
        # Pre-eclampsia
        "pre-eclampsia": "drkg:Disease::MESH:D011225",
        # Varicose Veins
        "varicose veins": "drkg:Disease::MESH:D014648",
        # Splenomegaly
        "splenomegaly": "drkg:Disease::MESH:D013163",
        # Strabismus
        "strabismus": "drkg:Disease::MESH:D013285",
        # Photophobia
        "photophobia": "drkg:Disease::MESH:D020795",
        # Rhinorrhea
        "rhinorrhea": "drkg:Disease::MESH:D012220",
        # Rhinitis
        "rhinitis": "drkg:Disease::MESH:D012220",
        # Sialorrhea
        "sialorrhea": "drkg:Disease::MESH:D012798",
        # Thyroid Neoplasms
        "thyroid neoplasms": "drkg:Disease::MESH:D013964",
        # Graves Ophthalmopathy
        "graves ophthalmopathy": "drkg:Disease::MESH:D049970",
        # Testicular Neoplasms
        "testicular neoplasms": "drkg:Disease::MESH:D013736",
        # Penile Neoplasms
        "penile neoplasms": "drkg:Disease::MESH:D010412",
        # Urologic Neoplasms
        "urologic neoplasms": "drkg:Disease::MESH:D014571",
        # Kidney Neoplasms
        "kidney neoplasms": "drkg:Disease::MESH:D007680",
        # Carcinoma, Renal Cell
        "carcinoma, renal cell": "drkg:Disease::MESH:D002292",
        # Uveal Neoplasms
        "uveal neoplasms": "drkg:Disease::MESH:D014604",
        # Malaria, Vivax
        "malaria, vivax": "drkg:Disease::MESH:D016780",
        # Salmonella Infections
        "salmonella infections": "drkg:Disease::MESH:D012480",
        # Rickettsia Infections
        "rickettsia infections": "drkg:Disease::MESH:D012282",
        # Paratyphoid Fever
        "paratyphoid fever": "drkg:Disease::MESH:D010284",
        # Typhoid Fever
        "typhoid fever": "drkg:Disease::MESH:D014435",
        # Scarlatina
        "scarlatina": "drkg:Disease::MESH:D012541",
        # Spinal Cord Injuries
        "spinal cord injuries": "drkg:Disease::MESH:D013119",
        # Keratosis, Actinic
        "keratosis, actinic": "drkg:Disease::MESH:D055623",
        # Lice Infestations
        "lice infestations": "drkg:Disease::MESH:D010373",
        # Scabies
        "scabies": "drkg:Disease::MESH:D012532",
        # Staphylococcal Infections
        "staphylococcal infections": "drkg:Disease::MESH:D013203",
        # Streptococcal Infections
        "streptococcal infections": "drkg:Disease::MESH:D013290",
        # Pharyngitis
        "pharyngitis": "drkg:Disease::MESH:D010612",
        # Epilepsies, Myoclonic
        "epilepsies, myoclonic": "drkg:Disease::MESH:D020191",
        # Ventricular Premature Complexes
        "ventricular premature complexes": "drkg:Disease::MESH:D018879",
        # Tumor Lysis Syndrome
        "tumor lysis syndrome": "drkg:Disease::MESH:D015275",
        # Pneumothorax
        "pneumothorax": "drkg:Disease::MESH:D011030",
        # Dwarfism
        "dwarfism": "drkg:Disease::MESH:D004392",
        # Still Disease, Adult Onset
        "still disease, adult onset": "drkg:Disease::MESH:D016706",
        # Granulomatosis with Polyangiitis
        "granulomatosis with polyangiitis": "drkg:Disease::MESH:D014890",
        # Leiomyoma
        "leiomyoma": "drkg:Disease::MESH:D007889",
        # Uterine Hemorrhage
        "uterine hemorrhage": "drkg:Disease::MESH:D014592",
        # Adenomyosis
        "adenomyosis": "drkg:Disease::MESH:D062788",
        # Pericarditis
        "pericarditis": "drkg:Disease::MESH:D010493",
        # Shock
        "shock": "drkg:Disease::MESH:D012769",
        # Ventricular Outflow Obstruction
        "ventricular outflow obstruction": "drkg:Disease::MESH:D014694",
        # Liver Cirrhosis, Biliary
        "liver cirrhosis, biliary": "drkg:Disease::MESH:D008105",
        # Enterocolitis, Pseudomembranous
        "enterocolitis, pseudomembranous": "drkg:Disease::MESH:D004761",
        # Glioblastoma
        "glioblastoma": "drkg:Disease::MESH:D005909",
        # Astrocytoma
        "astrocytoma": "drkg:Disease::MESH:D001254",
        # Periarthritis
        "periarthritis": "drkg:Disease::MESH:D010489",
        # Stress Disorders, Post-Traumatic
        "stress disorders, post-traumatic": "drkg:Disease::MESH:D013313",
        # Stomatitis, Aphthous
        "stomatitis, aphthous": "drkg:Disease::MESH:D013281",
        # Musculoskeletal Pain
        "musculoskeletal pain": "drkg:Disease::MESH:D059352",
        # Dysmenorrhea
        "dysmenorrhea": "drkg:Disease::MESH:D004412",
        # Renal Colic
        "renal colic": "drkg:Disease::MESH:D056844",
        # Kidney Diseases
        "kidney diseases": "drkg:Disease::MESH:D007674",
        # Renal Osteodystrophy
        "renal osteodystrophy": "drkg:Disease::MESH:D012080",
        # Acidosis, Renal Tubular
        "acidosis, renal tubular": "drkg:Disease::MESH:D000141",
        # Thiamine Deficiency
        "thiamine deficiency": "drkg:Disease::MESH:D013832",
        # Vitamin A Deficiency
        "vitamin a deficiency": "drkg:Disease::MESH:D014802",
        # Vitamin B 12 Deficiency
        "vitamin b 12 deficiency": "drkg:Disease::MESH:D014806",
        # Hyponatremia
        "hyponatremia": "drkg:Disease::MESH:D007010",
        # Iron Overload
        "iron overload": "drkg:Disease::MESH:D019190",
        # Prolactinoma
        "prolactinoma": "drkg:Disease::MESH:D015175",
        # Pituitary Neoplasms
        "pituitary neoplasms": "drkg:Disease::MESH:D010911",
        # Sheehan Syndrome
        "sheehan syndrome": "drkg:Disease::MESH:D007018",
        # Community-acquired infections
        "community-acquired infections": "drkg:Disease::MESH:D017714",
        # Pneumonia, Bacterial
        "pneumonia, bacterial": "drkg:Disease::MESH:D018410",
        # Gastrointestinal Hemorrhage
        "gastrointestinal hemorrhage": "drkg:Disease::MESH:D006471",
        # Zollinger-Ellison Syndrome
        "zollinger-ellison syndrome": "drkg:Disease::MESH:D015043",
        # Phobic Disorders
        "phobic disorders": "drkg:Disease::MESH:D010698",
        # Neurotic Disorders
        "neurotic disorders": "drkg:Disease::MESH:D009497",
        # Subarachnoid Hemorrhage
        "subarachnoid hemorrhage": "drkg:Disease::MESH:D013345",
        # Peripheral Arterial Disease
        "peripheral arterial disease": "drkg:Disease::MESH:D058729",
        # Peripheral Vascular Diseases
        "peripheral vascular diseases": "drkg:Disease::MESH:D016491",
        # Pouchitis
        "pouchitis": "drkg:Disease::MESH:D019449",
        # Obstetric Labor, Premature
        "obstetric labor, premature": "drkg:Disease::MESH:D007752",
        # Polycystic Ovary Syndrome
        "polycystic ovary syndrome": "drkg:Disease::MESH:D011085",
        # Presbyopia (may not be in DRKG)
        "presbyopia": "drkg:Disease::MESH:D011305",
        # Panuveitis
        "panuveitis": "drkg:Disease::MESH:D015864",
        # Polydipsia
        "polydipsia": "drkg:Disease::MESH:D059606",
        # Syphilis
        "syphilis": "drkg:Disease::MESH:D013587",
        # Anemia, Iron-Deficiency
        "anemia, iron-deficiency": "drkg:Disease::MESH:D018798",
        # Epilepsies, Partial
        "epilepsies, partial": "drkg:Disease::MESH:D004828",
        # Seizures
        "seizures": "drkg:Disease::MESH:D012640",
        # Proctocolitis
        "proctocolitis": "drkg:Disease::MESH:D011349",
        # Sleep Initiation and Maintenance Disorders
        "sleep initiation and maintenance disorders": "drkg:Disease::MESH:D007319",
        # Sleep Wake Disorders
        "sleep wake disorders": "drkg:Disease::MESH:D012893",
        # Depressive Disorder, Major
        "depressive disorder, major": "drkg:Disease::MESH:D003865",
        # Edema
        "edema": "drkg:Disease::MESH:D004487",
        # Hypoglycemia
        "hypoglycemia": "drkg:Disease::MESH:D007003",
        # Cross Infection (nosocomial)
        "cross infection": "drkg:Disease::MESH:D003428",
    }

    # Agent mappings
    agent_path = REFERENCE_DIR / "mesh_mappings_from_agents.json"
    agent_mappings = {}

    if agent_path.exists():
        with open(agent_path) as f:
            agent_data = json.load(f)

        for batch_name, batch_data in agent_data.items():
            if batch_name == "metadata" or not isinstance(batch_data, dict):
                continue
            for disease_name, mesh_id in batch_data.items():
                if mesh_id is not None and mesh_id.startswith("D"):
                    agent_mappings[disease_name.lower()] = f"drkg:Disease::MESH:{mesh_id}"

    # MONDO to MESH mappings (adds ~8000 disease ID mappings)
    # Coverage: 44.7% of Every Cure diseases (up from 17.2%)
    mondo_path = REFERENCE_DIR / "mondo_to_mesh.json"
    mondo_mappings = {}

    if mondo_path.exists():
        with open(mondo_path) as f:
            mondo_data = json.load(f)

        # Store MONDO ID -> DRKG disease ID mapping for lookup
        # Format: "drkg:Disease::MESH:D..." to match GT format
        for mondo_id, mesh_id in mondo_data.items():
            if mesh_id and mesh_id.startswith("MESH:"):
                # Store as lowercase MONDO ID for lookup
                mondo_mappings[mondo_id.lower()] = f"drkg:Disease::{mesh_id}"

    # Merge (agent takes priority over hardcoded, mondo is separate lookup)
    name_mappings = {**hardcoded, **agent_mappings}

    # Return combined dict - name_mappings + mondo_mappings
    return {**name_mappings, **mondo_mappings}


def test_matcher():
    """Test the disease matcher."""
    mesh_mappings = load_mesh_mappings()
    matcher = DiseaseMatcher(mesh_mappings)

    # Test cases
    test_diseases = [
        "atopic dermatitis",
        "atopic dermatitis ",  # trailing space
        "plaque psoriasis",
        "type 2 diabetes mellitus ",  # trailing space
        "hiv1 infection",
        "parkinsons disease",
        "Parkinson's Disease",  # mixed case
        "non-small cell lung cancer",
        "seasonal allergic rhinitis",
        "Rheumatoid Arthritis",
    ]

    print("=== Disease Matcher Test ===\n")
    for disease in test_diseases:
        mesh_id = matcher.get_mesh_id(disease)
        status = "✓" if mesh_id else "✗"
        print(f"{status} {disease:40} -> {mesh_id or 'NOT FOUND'}")

    # Coverage report
    import pandas as pd
    gt = pd.read_excel("data/reference/everycure/indicationList.xlsx")
    ec_diseases = set(gt['disease name'].unique())

    report = matcher.coverage_report(ec_diseases)
    print(f"\n=== Coverage Report ===")
    print(f"Total diseases: {report['total']}")
    print(f"Mapped: {report['mapped']} ({report['coverage']*100:.1f}%)")
    print(f"Unmapped: {report['unmapped']}")


if __name__ == "__main__":
    test_matcher()
