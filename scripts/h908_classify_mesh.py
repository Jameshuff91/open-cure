"""
h908: Classify h901-added MeSH mappings by tree number to identify
symptoms/findings/procedures that should not generate drug predictions.

Queries NCBI MeSH API at https://id.nlm.nih.gov/mesh/<id>.json
Caches results to data/cache/mesh_tree_cache.json
"""

import json
import time
from pathlib import Path

import requests

CACHE_PATH = Path('data/cache/mesh_tree_cache.json')
INPUT_PATH = Path('data/analysis/h908_h901_added_mappings.json')
OUTPUT_PATH = Path('data/analysis/h908_classified.json')


def load_cache():
    if CACHE_PATH.exists():
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)


def fetch_tree_numbers(mesh_id, cache):
    if mesh_id in cache:
        return cache[mesh_id]

    url = f'https://id.nlm.nih.gov/mesh/{mesh_id}.json'
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            data = r.json()
            trees = data.get('treeNumber', [])
            if isinstance(trees, str):
                trees = [trees]
            elif isinstance(trees, list):
                trees = [t if isinstance(t, str) else t.get('@id', t.get('value', str(t))) for t in trees]
            label = data.get('label', {})
            if isinstance(label, dict):
                label_text = label.get('@value', label.get('value', ''))
            else:
                label_text = str(label)
            result = {'tree_numbers': trees, 'label': label_text, 'status': 'ok'}
        elif r.status_code == 404:
            result = {'tree_numbers': [], 'label': '', 'status': 'not_found'}
        else:
            result = {'tree_numbers': [], 'label': '', 'status': f'http_{r.status_code}'}
    except Exception as e:
        result = {'tree_numbers': [], 'label': '', 'status': f'error: {e}'}

    cache[mesh_id] = result
    return result


def _strip(t):
    """Tree numbers come back as 'http://id.nlm.nih.gov/mesh/C20.673.480.040'."""
    if isinstance(t, str) and t.startswith('http'):
        return t.rsplit('/', 1)[-1]
    return t


def classify_tree_numbers(trees):
    """Classify MeSH descriptor by its tree numbers.

    Tree categories (https://www.nlm.nih.gov/mesh/intro_trees.html):
      A = Anatomy
      B = Organisms (incl. infectious agents)
      C = Diseases (real disease entities)
      C23 = Pathological Conditions, Signs and Symptoms (mostly findings)
      D = Chemicals and Drugs
      E = Analytical, Diagnostic and Therapeutic Techniques and Equipment (procedures)
      F = Psychiatry and Psychology
        F01 = Behavior and Behavior Mechanisms
        F02 = Psychological Phenomena
        F03 = Mental Disorders (legitimate diseases)
      G = Phenomena and Processes
      H = Disciplines
      ...
    """
    if not trees:
        return 'no_tree'

    trees = [_strip(t) for t in trees]
    # If any tree is C* (other than C23), it's a real disease
    has_disease = any(t.startswith('C') and not t.startswith('C23') for t in trees)
    has_mental_disorder = any(t.startswith('F03') for t in trees)
    has_c23 = any(t.startswith('C23') for t in trees)
    has_procedure = any(t.startswith('E') for t in trees)
    has_phenomenon = any(t.startswith('G') for t in trees)
    has_organism = any(t.startswith('B') for t in trees)
    has_anatomy = any(t.startswith('A') for t in trees)
    has_chemical = any(t.startswith('D') for t in trees)
    has_behavior = any(t.startswith('F01') or t.startswith('F02') for t in trees)

    # Real disease takes precedence
    if has_disease or has_mental_disorder:
        return 'disease'
    if has_c23:
        return 'symptom_finding'
    if has_procedure:
        return 'procedure'
    if has_organism:
        return 'organism'
    if has_anatomy:
        return 'anatomy'
    if has_chemical:
        return 'chemical'
    if has_phenomenon:
        return 'phenomenon'
    if has_behavior:
        return 'behavior'
    return 'other'


def main():
    with open(INPUT_PATH) as f:
        mappings = json.load(f)

    cache = load_cache()
    unique_ids = sorted({v['mesh_id'] for v in mappings.values() if v['mesh_id']})
    print(f"Total mappings: {len(mappings)}")
    print(f"Unique MeSH IDs to query: {len(unique_ids)}")
    print(f"Already cached: {sum(1 for i in unique_ids if i in cache)}")

    new_queries = 0
    for i, mid in enumerate(unique_ids):
        if mid not in cache:
            fetch_tree_numbers(mid, cache)
            new_queries += 1
            time.sleep(0.35)  # ~3 req/sec
            if new_queries % 25 == 0:
                save_cache(cache)
                print(f"  {i+1}/{len(unique_ids)} processed ({new_queries} new queries)")

    save_cache(cache)

    # Classify each mapping
    results = {}
    class_counts = {}
    for name, info in mappings.items():
        mid = info['mesh_id']
        if not mid:
            cls = 'no_mesh_id'
            trees = []
            label = ''
        else:
            entry = cache.get(mid, {})
            trees = entry.get('tree_numbers', [])
            label = entry.get('label', '')
            cls = classify_tree_numbers(trees)
        results[name] = {
            'mesh_id': mid,
            'mesh_label': label,
            'tree_numbers': trees,
            'class': cls,
            'batch': info['batch'],
        }
        class_counts[cls] = class_counts.get(cls, 0) + 1

    with open(OUTPUT_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Classification Summary ({len(results)} mappings) ===")
    for cls, n in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls}: {n} ({100*n/len(results):.1f}%)")

    # Print samples of each non-disease class
    print("\n=== Samples by Class ===")
    for cls in ['symptom_finding', 'procedure', 'phenomenon', 'organism',
                'chemical', 'anatomy', 'behavior', 'no_tree', 'no_mesh_id', 'other']:
        samples = [(n, r) for n, r in results.items() if r['class'] == cls][:8]
        if samples:
            print(f"\n  {cls} (n={class_counts.get(cls, 0)}):")
            for name, r in samples:
                tn = r['tree_numbers'][:2] if r['tree_numbers'] else []
                print(f"    {name!r} ({r['mesh_id']}, {tn}, label={r['mesh_label']!r})")


if __name__ == '__main__':
    main()
