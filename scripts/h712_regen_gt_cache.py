#!/usr/bin/env python3
"""h712: Regenerate GT cache after disease name synonym expansion.

This script regenerates the ground truth cache without loading full predictor
(which requires Node2Vec embeddings, TransE, etc.). It replicates the
_load_ground_truth logic from production_predictor.py.
"""

import sys
import time
import json
import hashlib
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd

t0 = time.time()
data_dir = Path('.')
reference_dir = data_dir / 'data' / 'reference'

# Load DrugBank lookup
with open(reference_dir / 'drugbank_lookup.json') as f:
    id_to_name = json.load(f)
name_to_drug_id = {
    name.lower(): f'drkg:Compound::{db_id}'
    for db_id, name in id_to_name.items()
}
print(f'Loaded {len(name_to_drug_id)} drug names')

# h686: Add drug name aliases
sys.path.insert(0, 'src')
from production_predictor import _DRUG_NAME_ALIASES
for ec_name, db_name in _DRUG_NAME_ALIASES.items():
    db_name_lower = db_name.lower()
    if db_name_lower in name_to_drug_id and ec_name not in name_to_drug_id:
        name_to_drug_id[ec_name] = name_to_drug_id[db_name_lower]
print(f'After aliases: {len(name_to_drug_id)} drug names')

# Salt suffixes
_SALT_SUFFIXES = [
    ' hydrochloride', ' hcl', ' citrate', ' acetate', ' sodium',
    ' sulfate', ' phosphate', ' maleate', ' fumarate', ' besylate',
    ' tartrate', ' bromide', ' mesylate', ' nitrate', ' succinate',
    ' potassium', ' calcium', ' magnesium', ' monohydrate',
    ' dihydrate', ' anhydrous, (e)-', ' anhydrous',
]

# Load MESH mappings
mesh_path = reference_dir / 'mesh_mappings_from_agents.json'
mesh_mappings = {}
if mesh_path.exists():
    with open(mesh_path) as f:
        mesh_data = json.load(f)
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id:
                    mesh_str = str(mesh_id)
                    if mesh_str.startswith('D') or mesh_str.startswith('C'):
                        mesh_mappings[disease_name.lower()] = f'drkg:Disease::MESH:{mesh_str}'
print(f'Loaded {len(mesh_mappings)} MESH mappings')

# Load disease name matcher
from disease_name_matcher import DiseaseMatcher, load_mesh_mappings
fuzzy_mappings = load_mesh_mappings()
matcher = DiseaseMatcher(fuzzy_mappings)
print(f'Loaded matcher with {len(fuzzy_mappings)} fuzzy mappings')

# Now build GT
df = pd.read_excel(reference_dir / 'everycure' / 'indicationList.xlsx')
print(f'Loaded {len(df)} EC rows')

ground_truth = defaultdict(set)
disease_names = {}
unmapped_diseases = defaultdict(int)
unmapped_drugs = defaultdict(int)

for _, row in df.iterrows():
    disease = str(row.get('disease name', '')).strip()
    drug = str(row.get('final normalized drug label', '')).strip()
    if not disease or not drug:
        continue

    disease_id = matcher.get_mesh_id(disease)
    if not disease_id:
        disease_id = mesh_mappings.get(disease.lower())
    # h712: Fallback to EC disease ID column
    if not disease_id:
        ec_disease_id = str(row.get('final normalized disease id', '')).strip()
        if ec_disease_id:
            disease_id = fuzzy_mappings.get(ec_disease_id.lower())
    if not disease_id:
        unmapped_diseases[disease] += 1
        continue

    disease_names[disease_id] = disease
    drug_lower = drug.lower()
    drug_id = name_to_drug_id.get(drug_lower)
    # h686: Salt form suffix stripping fallback
    if not drug_id:
        for suffix in _SALT_SUFFIXES:
            base = drug_lower.replace(suffix, '').strip()
            if base != drug_lower and base in name_to_drug_id:
                drug_id = name_to_drug_id[base]
                break
    if drug_id:
        ground_truth[disease_id].add(drug_id)
    else:
        unmapped_drugs[drug] += 1

ground_truth = dict(ground_truth)
total_pairs = sum(len(v) for v in ground_truth.values())
print(f'\nPost-h712 GT: {len(ground_truth)} diseases, {total_pairs} drug-disease pairs')
print(f'Unmapped diseases: {len(unmapped_diseases)} (total mentions: {sum(unmapped_diseases.values())})')
print(f'Unmapped drugs: {len(unmapped_drugs)} (total mentions: {sum(unmapped_drugs.values())})')

# Compare with pre-h712 cache
pre_cache = data_dir / 'data' / 'cache' / 'ground_truth_cache.json.pre_h712'
if pre_cache.exists():
    with open(pre_cache) as f:
        pre_data = json.load(f)
    pre_gt = pre_data['ground_truth']
    pre_pairs = sum(len(v) for v in pre_gt.values())
    print(f'\nPre-h712 GT: {len(pre_gt)} diseases, {pre_pairs} drug-disease pairs')
    print(f'Delta: +{len(ground_truth) - len(pre_gt)} diseases, +{total_pairs - pre_pairs} pairs')

    # Find new diseases
    new_diseases = set(ground_truth.keys()) - set(pre_gt.keys())
    print(f'New diseases: {len(new_diseases)}')
    for d in sorted(new_diseases):
        name = disease_names.get(d, 'unknown')
        n_drugs = len(ground_truth[d])
        print(f'  {d}: {name} ({n_drugs} drugs)')

    # Find diseases with expanded GT
    expanded = 0
    for d in ground_truth:
        if d in pre_gt:
            old_set = set(pre_gt[d])
            new_set = ground_truth[d]
            if len(new_set) > len(old_set):
                expanded += 1
    print(f'Diseases with expanded GT: {expanded}')

# Save cache
cache_dir = data_dir / 'data' / 'cache'
cache_dir.mkdir(exist_ok=True)
cache_path = cache_dir / 'ground_truth_cache.json'

# Compute cache key
source_files = [
    reference_dir / 'everycure' / 'indicationList.xlsx',
    reference_dir / 'mesh_mappings_from_agents.json',
    reference_dir / 'mondo_to_mesh.json',
    reference_dir / 'drugbank_lookup.json',
    data_dir / 'src' / 'disease_name_matcher.py',
]
mtimes = []
for f in source_files:
    if f.exists():
        mtimes.append(f'{f.name}:{os.path.getmtime(f):.0f}')
cache_key = hashlib.md5('|'.join(mtimes).encode()).hexdigest()[:16]

cache_data = {
    'cache_key': cache_key,
    'ground_truth': {k: list(v) for k, v in ground_truth.items()},
    'disease_names': disease_names,
}
with open(cache_path, 'w') as f:
    json.dump(cache_data, f)
print(f'\nCache saved to {cache_path}')

# Top unmapped diseases
print(f'\nTop 30 unmapped diseases (by drug count):')
for d, n in sorted(unmapped_diseases.items(), key=lambda x: -x[1])[:30]:
    print(f'  {d}: {n} drugs')

# Top unmapped drugs
print(f'\nTop 20 unmapped drugs:')
for d, n in sorted(unmapped_drugs.items(), key=lambda x: -x[1])[:20]:
    print(f'  {d}: {n} diseases')

print(f'\nTime: {time.time()-t0:.1f}s')
