#!/usr/bin/env python3
"""
h956: PubMedBERT CPU-only smoke test.

Goal: cheaply check whether PubMedBERT text embeddings of drug + disease
names produce a usable ranking signal. If yes, green-light h920 full
GPU pipeline (with disease *definitions* and drug *descriptions*, not
just names). If no, dequeue h920.

Skips the production_predictor comparison to keep venv deps minimal —
compared against expanded_ground_truth directly, plus a literature-sanity
baseline (random top-30 from the candidate pool).

Run via the dedicated venv:
    /tmp/h956_venv/bin/python scripts/h956_pubmedbert_smoke.py

Outputs:
    data/analysis/h956_pubmedbert_smoke.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

N_DRUGS = 500
N_DISEASES = 200
TOP_N = 30


def load_drugs() -> Dict[str, str]:
    """drug_id -> name, restricted to drugs that appear in GT."""
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        dbl = json.load(f)
    id_to_name = {f"drkg:Compound::{db_id}": name for db_id, name in dbl.items()}

    with open(REFERENCE_DIR / "expanded_ground_truth.json") as f:
        exp_gt = json.load(f)
    usage: Dict[str, int] = {}
    for _, drugs in exp_gt.items():
        if isinstance(drugs, list):
            for d in drugs:
                usage[d] = usage.get(d, 0) + 1
    top = sorted(usage.keys(), key=lambda x: -usage[x])
    out: Dict[str, str] = {}
    for did in top:
        if len(out) >= N_DRUGS:
            break
        name = id_to_name.get(did)
        if name:
            out[did] = name
    return out


def load_diseases(drug_sampled: Dict[str, str]) -> Dict[str, str]:
    """disease_id -> name, restricted to diseases with >=2 sampled-GT drugs."""
    # disease names come from the merged mesh_mappings (id -> name is inverse)
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)
    id_to_name: Dict[str, str] = {}
    for batch in mesh_data.values():
        if isinstance(batch, dict):
            for name, mid in batch.items():
                if mid and str(mid).startswith(("D", "C")):
                    did = f"drkg:Disease::MESH:{mid}"
                    id_to_name.setdefault(did, name)

    alias_path = REFERENCE_DIR / "h961_disease_name_aliases.json"
    if alias_path.exists():
        with open(alias_path) as f:
            ad = json.load(f)
        for name, did in ad.get("disease_names_backfill", {}).items():
            id_to_name.setdefault(did, name)

    with open(REFERENCE_DIR / "expanded_ground_truth.json") as f:
        exp_gt = json.load(f)

    drug_set = set(drug_sampled.keys())
    out: Dict[str, str] = {}
    for did, drugs in exp_gt.items():
        if not isinstance(drugs, list):
            continue
        hits = [d for d in drugs if d in drug_set]
        if len(hits) >= 2 and did in id_to_name:
            out[did] = id_to_name[did]
        if len(out) >= N_DISEASES:
            break
    return out


def embed(texts: List[str]) -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    embs: List[np.ndarray] = []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        enc = tok(
            chunk, padding=True, truncation=True, max_length=64, return_tensors="pt"
        )
        with torch.no_grad():
            out = model(**enc)
        last = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs.append(pooled.numpy())
        if (i // batch) % 10 == 0:
            print(f"  embedded {i + len(chunk)}/{len(texts)}")
    return np.vstack(embs).astype(np.float32)


def l2n(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    drugs = load_drugs()
    diseases = load_diseases(drugs)
    print(f"Drugs: {len(drugs)}, Diseases: {len(diseases)}")

    drug_ids = list(drugs.keys())
    drug_texts = [drugs[d] for d in drug_ids]
    dis_ids = list(diseases.keys())
    dis_texts = [diseases[d] for d in dis_ids]

    with open(REFERENCE_DIR / "expanded_ground_truth.json") as f:
        exp_gt = json.load(f)

    print("\nEmbedding drugs ...")
    de = l2n(embed(drug_texts))
    print(f"  shape={de.shape}")

    print("\nEmbedding diseases ...")
    dise = l2n(embed(dis_texts))
    print(f"  shape={dise.shape}")

    sims = dise @ de.T  # (N_dis, N_drugs)

    pubmed_r30: List[float] = []
    random_r30: List[float] = []
    rng = np.random.default_rng(42)
    drug_pool_size = len(drug_ids)

    for i, did in enumerate(dis_ids):
        gt = set(exp_gt.get(did, []))
        gt_in_pool = gt & set(drug_ids)
        if not gt_in_pool:
            continue

        # PubMedBERT top-30
        top30_idx = np.argsort(-sims[i])[:TOP_N]
        top30_ids = {drug_ids[j] for j in top30_idx}
        hits = len(top30_ids & gt_in_pool)
        pubmed_r30.append(hits / min(len(gt_in_pool), TOP_N))

        # Random top-30 baseline (expected recall = TOP_N / N_DRUGS)
        r_idx = rng.choice(drug_pool_size, size=TOP_N, replace=False)
        r_ids = {drug_ids[j] for j in r_idx}
        r_hits = len(r_ids & gt_in_pool)
        random_r30.append(r_hits / min(len(gt_in_pool), TOP_N))

    print("\n" + "=" * 60)
    print("  h956 PubMedBERT CPU smoke test")
    print("=" * 60)
    print(f"  Diseases evaluated: {len(pubmed_r30)}")
    print(f"  PubMedBERT R@30:    {np.mean(pubmed_r30):.4f} ± {np.std(pubmed_r30):.4f}")
    print(f"  Random R@30:        {np.mean(random_r30):.4f} ± {np.std(random_r30):.4f}")
    lift = np.mean(pubmed_r30) - np.mean(random_r30)
    lift_ratio = np.mean(pubmed_r30) / max(np.mean(random_r30), 1e-6)
    print(f"  Lift over random:   {lift:+.4f}  ({lift_ratio:.2f}x)")
    print(f"  Wall time: {time.time()-t0:.0f}s")

    # Reference: current production R@30 on similar pool is ~20-30%.
    # Decision rule (conservative, CPU-only names-only test):
    #  - PubMedBERT >= 0.10 AND 3x random: interesting, merits GPU experiment
    #    with proper descriptions and fusion
    #  - 0.05-0.10 or 2-3x random: fusion-only, not standalone
    #  - < 0.05 or < 2x random: dequeue h920
    pm = float(np.mean(pubmed_r30))
    if pm >= 0.10 and lift_ratio >= 3.0:
        decision = "GO_GPU_FULL"
    elif pm >= 0.05 and lift_ratio >= 2.0:
        decision = "FUSION_ONLY"
    else:
        decision = "DEQUEUE_h920"
    print(f"\n  Decision for h920: {decision}")

    out = {
        "pubmed_r30_mean": pm,
        "pubmed_r30_std": float(np.std(pubmed_r30)),
        "random_r30_mean": float(np.mean(random_r30)),
        "random_r30_std": float(np.std(random_r30)),
        "lift_over_random": float(lift),
        "lift_ratio": float(lift_ratio),
        "n_diseases": int(len(pubmed_r30)),
        "n_drugs_sampled": len(drugs),
        "n_diseases_sampled": len(diseases),
        "decision_for_h920": decision,
        "wall_time_seconds": float(time.time() - t0),
        "notes": (
            "Name-only smoke test. Real h920 would use DrugBank drug descriptions "
            "and MeSH disease definitions, both longer and richer — expect the "
            "full run to exceed this signal."
        ),
    }
    with open(ANALYSIS_DIR / "h956_pubmedbert_smoke.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to data/analysis/h956_pubmedbert_smoke.json")


if __name__ == "__main__":
    main()
