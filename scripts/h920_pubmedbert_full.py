#!/usr/bin/env python3
"""
h920: Full PubMedBERT retrieval pipeline with MeSH scope notes.

Upgrade from h956 (names-only, 500 drugs × 200 diseases) to:
- Full evaluable drug pool (~1,300 drugs used anywhere in expanded GT)
- Full evaluable disease pool (~1,070 diseases with embeddings)
- Disease inputs = name + MeSH scope note (definition)
- Drug inputs = name (DrugBank descriptions require licensed XML; queued
  separately as h920-v2)

Data flow:
1. Fetch MeSH scope notes for all MeSH IDs in disease pool (cached to
   data/reference/mesh_scope_notes.json; incremental).
2. Embed drugs (names) and diseases (name + scope note).
3. Cosine-similarity ranking; per-disease top-30 R@30 on expanded GT.
4. Compare to h956 name-only and random baseline.

Run via:
    /tmp/h956_venv/bin/python scripts/h920_pubmedbert_full.py

Outputs:
    data/analysis/h920_pubmedbert_full.json
    data/reference/mesh_scope_notes.json  (incremental cache)
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
SCOPE_CACHE = REFERENCE_DIR / "mesh_scope_notes.json"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
TOP_N = 30


def fetch_scope_note(mesh_id: str) -> str:
    """Fetch MeSH scope note for a single ID. Returns "" if unavailable."""
    url = f"https://id.nlm.nih.gov/mesh/{mesh_id}.json"
    try:
        req = urllib.request.Request(url, headers={"Accept": "application/ld+json"})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return ""
    # The response is JSON-LD; scope note is typically at top level.
    # preferredConcept -> scopeNote (English) or preferredTerm -> label
    if isinstance(data, list):
        data = data[0] if data else {}
    # Walk common paths
    for key in ("scopeNote", "prefLabel", "label"):
        v = data.get(key)
        if isinstance(v, str):
            return v
        if isinstance(v, dict):
            s = v.get("@value") or v.get("value")
            if isinstance(s, str):
                return s
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    s = item.get("@value") or item.get("value")
                    if isinstance(s, str) and (
                        item.get("@language", "en") == "en"
                        or key == "scopeNote"
                    ):
                        return s
                if isinstance(item, str):
                    return item
    return ""


def load_scope_cache() -> Dict[str, str]:
    if SCOPE_CACHE.exists():
        with open(SCOPE_CACHE) as f:
            return json.load(f)
    return {}


def save_scope_cache(cache: Dict[str, str]) -> None:
    with open(SCOPE_CACHE, "w") as f:
        json.dump(cache, f, indent=2)


def mesh_id_from_drkg(drkg_id: str) -> str:
    """drkg:Disease::MESH:D009369 -> D009369"""
    return drkg_id.rsplit("MESH:", 1)[-1]


def load_pools() -> tuple[Dict[str, str], Dict[str, str]]:
    """Return (drug_id -> name, disease_id -> name) for the evaluation pool."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        dbl = json.load(f)
    id_to_name = {f"drkg:Compound::{db_id}": name for db_id, name in dbl.items()}

    with open(REFERENCE_DIR / "expanded_ground_truth.json") as f:
        exp_gt = json.load(f)

    # Drugs: include any that appear in expanded GT
    drug_ids_used = set()
    for drugs in exp_gt.values():
        if isinstance(drugs, list):
            drug_ids_used.update(drugs)
    drugs: Dict[str, str] = {}
    for did in sorted(drug_ids_used):
        if did in id_to_name:
            drugs[did] = id_to_name[did]

    # Diseases: from mesh_mappings + h961 aliases, evaluable = with GT + name
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)
    id_to_dname: Dict[str, str] = {}
    for batch in mesh_data.values():
        if isinstance(batch, dict):
            for name, mid in batch.items():
                if mid and str(mid).startswith(("D", "C")):
                    did = f"drkg:Disease::MESH:{mid}"
                    id_to_dname.setdefault(did, name)
    alias_path = REFERENCE_DIR / "h961_disease_name_aliases.json"
    if alias_path.exists():
        with open(alias_path) as f:
            ad = json.load(f)
        for name, did in ad.get("disease_names_backfill", {}).items():
            id_to_dname.setdefault(did, name)

    diseases: Dict[str, str] = {}
    for did, drugs_in_gt in exp_gt.items():
        if isinstance(drugs_in_gt, list) and len(drugs_in_gt) >= 1 and did in id_to_dname:
            diseases[did] = id_to_dname[did]

    return drugs, diseases


def fetch_all_scope_notes(
    disease_ids: List[str], rate_per_sec: float = 2.0
) -> Dict[str, str]:
    cache = load_scope_cache()
    missing = [did for did in disease_ids if mesh_id_from_drkg(did) not in cache]
    print(f"Scope notes: {len(cache)} cached, {len(missing)} to fetch")
    if not missing:
        return cache
    interval = 1.0 / rate_per_sec
    t_last = 0.0
    hits = 0
    for i, did in enumerate(missing):
        mid = mesh_id_from_drkg(did)
        # Rate limit
        dt = time.time() - t_last
        if dt < interval:
            time.sleep(interval - dt)
        t_last = time.time()
        note = fetch_scope_note(mid)
        cache[mid] = note
        if note:
            hits += 1
        if (i + 1) % 50 == 0:
            save_scope_cache(cache)
            print(f"  {i+1}/{len(missing)} fetched, {hits} with notes")
    save_scope_cache(cache)
    print(f"  Done: {hits}/{len(missing)} missing IDs had scope notes")
    return cache


def embed(texts: List[str], batch: int = 16, max_len: int = 256) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embs: List[np.ndarray] = []
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        enc = tok(
            chunk, padding=True, truncation=True, max_length=max_len, return_tensors="pt"
        )
        with torch.no_grad():
            out = model(**enc)
        last = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs.append(pooled.numpy())
        if (i // batch) % 20 == 0:
            print(f"  embedded {i + len(chunk)}/{len(texts)}")
    return np.vstack(embs).astype(np.float32)


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    drugs, diseases = load_pools()
    print(f"Drugs: {len(drugs)}, Diseases: {len(diseases)}")

    # Fetch scope notes (incremental)
    cache = fetch_all_scope_notes(list(diseases.keys()))

    # Build inputs
    drug_ids = list(drugs.keys())
    drug_texts = [drugs[d] for d in drug_ids]
    disease_ids = list(diseases.keys())
    disease_texts: List[str] = []
    has_note = 0
    for did in disease_ids:
        name = diseases[did]
        note = cache.get(mesh_id_from_drkg(did), "") or ""
        if note:
            has_note += 1
            disease_texts.append(f"{name}. {note}")
        else:
            disease_texts.append(name)
    print(f"Disease inputs: {has_note}/{len(disease_ids)} have scope notes")

    with open(REFERENCE_DIR / "expanded_ground_truth.json") as f:
        exp_gt = json.load(f)

    print("\nEmbedding drugs ...")
    de = l2n(embed(drug_texts, max_len=64))
    print(f"  shape={de.shape}")

    print("\nEmbedding diseases ...")
    dise = l2n(embed(disease_texts, max_len=256))
    print(f"  shape={dise.shape}")

    sims = dise @ de.T

    pm_r30: List[float] = []
    rand_r30: List[float] = []
    by_has_note: Dict[str, List[float]] = {"with_note": [], "name_only": []}
    rng = np.random.default_rng(42)
    drug_pool = set(drug_ids)

    for i, did in enumerate(disease_ids):
        gt = set(exp_gt.get(did, [])) & drug_pool
        if not gt:
            continue
        top30 = set(drug_ids[j] for j in np.argsort(-sims[i])[:TOP_N])
        pm = len(top30 & gt) / min(len(gt), TOP_N)
        pm_r30.append(pm)
        r_idx = rng.choice(len(drug_ids), size=TOP_N, replace=False)
        r_ids = {drug_ids[j] for j in r_idx}
        rand_r30.append(len(r_ids & gt) / min(len(gt), TOP_N))
        note = cache.get(mesh_id_from_drkg(did), "")
        by_has_note["with_note" if note else "name_only"].append(pm)

    pm_mean = float(np.mean(pm_r30))
    rand_mean = float(np.mean(rand_r30))
    lift = pm_mean - rand_mean
    ratio = pm_mean / max(rand_mean, 1e-6)

    print("\n" + "=" * 60)
    print("  h920 PubMedBERT (CPU) full-pool results")
    print("=" * 60)
    print(f"  Diseases:         {len(pm_r30)}")
    print(f"  Drug pool:        {len(drug_ids)}")
    print(f"  PubMedBERT R@30:  {pm_mean:.4f} ± {np.std(pm_r30):.4f}")
    print(f"  Random R@30:      {rand_mean:.4f} ± {np.std(rand_r30):.4f}")
    print(f"  Lift:             +{lift:.4f}  ({ratio:.2f}x)")
    if by_has_note["with_note"]:
        wn = np.mean(by_has_note["with_note"])
        no = np.mean(by_has_note["name_only"]) if by_has_note["name_only"] else 0.0
        print(f"  R@30 with scope note:  {wn:.4f} (n={len(by_has_note['with_note'])})")
        print(f"  R@30 name-only:        {no:.4f} (n={len(by_has_note['name_only'])})")
    print(f"  h956 reference (names, smaller pool): 0.2278 ± 0.1847")
    print(f"  Wall time: {time.time()-t0:.0f}s")

    if ratio >= 3.0 and pm_mean >= 0.15:
        decision = "PRODUCE_FUSION_LAYER"
    elif ratio >= 2.0 and pm_mean >= 0.10:
        decision = "FUSION_FEATURE_ONLY"
    else:
        decision = "DEQUEUE"
    print(f"\n  Decision: {decision}")

    out = {
        "pm_r30_mean": pm_mean,
        "pm_r30_std": float(np.std(pm_r30)),
        "random_r30_mean": rand_mean,
        "random_r30_std": float(np.std(rand_r30)),
        "lift_ratio": ratio,
        "n_diseases": len(pm_r30),
        "n_drugs_pool": len(drug_ids),
        "r30_with_scope_note": float(np.mean(by_has_note["with_note"])) if by_has_note["with_note"] else None,
        "r30_name_only": float(np.mean(by_has_note["name_only"])) if by_has_note["name_only"] else None,
        "n_with_scope_note": len(by_has_note["with_note"]),
        "n_name_only": len(by_has_note["name_only"]),
        "decision": decision,
        "wall_time_seconds": float(time.time() - t0),
    }
    with open(ANALYSIS_DIR / "h920_pubmedbert_full.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to data/analysis/h920_pubmedbert_full.json")


if __name__ == "__main__":
    main()
