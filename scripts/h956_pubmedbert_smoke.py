#!/usr/bin/env python3
"""
h956: PubMedBERT CPU-only smoke test.

Goal: cheaply estimate whether text embeddings can add precision signal
before committing to a full GPU h920 run. We:

1. Embed ~500 drug names (DrugBank) + ~500 disease names (MeSH) with
   PubMedBERT. CPU-only via transformers.
2. For each of N held-out diseases, rank all drugs by PubMedBERT cosine
   similarity. Compute per-drug R@30 on expanded GT.
3. Compare to the current production kNN top-30 on the same diseases.

Decision rule:
- If PubMedBERT mean R@30 >= 0.6 * kNN R@30 on biologic subset, green-
  light full h920 GPU pipeline.
- If between 0.3-0.6: marginal; consider as a fusion feature, not a
  standalone ranker.
- If < 0.3: de-queue h920 entirely.

Run via the dedicated venv that has transformers installed:
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
sys.path.insert(0, str(PROJECT_ROOT / "src"))

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def load_drug_and_disease_text() -> tuple[Dict[str, str], Dict[str, str], object]:
    """Return (drug_id -> name, disease_id -> name, predictor) for embedding.

    Uses drugbank_lookup.json and mesh_mappings_from_agents.json (with h961
    aliases) as canonical name sources. Keeps to 500 + 500 for CPU budget.
    """
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        dbl = json.load(f)
    drug_id_to_name: Dict[str, str] = {
        f"drkg:Compound::{db_id}": name for db_id, name in dbl.items()
    }

    import sys as _sys
    _sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from production_predictor import DrugRepurposingPredictor

    p = DrugRepurposingPredictor()

    # Disease: take the canonical disease_names values (already de-duplicated
    # by id). Keep only diseases with >=2 GT drugs so we have labels to score.
    disease_id_to_name: Dict[str, str] = {}
    for did, dname in p.disease_names.items():
        gt = p.ground_truth.get(did, set())
        if len(gt) >= 2:
            disease_id_to_name[did] = dname

    # Bound to 500 diseases and 500 drugs for the smoke test
    disease_items = sorted(disease_id_to_name.items())[:500]
    disease_sampled = dict(disease_items)

    # Drugs: keep only drugs that appear in at least one GT set
    drug_usage = {}
    for did, drugs in p.ground_truth.items():
        for d in drugs:
            drug_usage[d] = drug_usage.get(d, 0) + 1
    drug_ids_sorted = sorted(drug_usage.keys(), key=lambda x: -drug_usage[x])
    drug_sampled: Dict[str, str] = {}
    for drug_id in drug_ids_sorted[:500]:
        name = drug_id_to_name.get(drug_id)
        if name:
            drug_sampled[drug_id] = name

    return drug_sampled, disease_sampled, p


def embed_texts(texts: List[str], model_name: str = MODEL_NAME) -> np.ndarray:
    """Mean-pooled PubMedBERT embeddings, CPU."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    embs: List[np.ndarray] = []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        enc = tok(chunk, padding=True, truncation=True, max_length=64, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
        # Mean-pool excluding padding
        last = out.last_hidden_state  # (B, L, H)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        embs.append(pooled.numpy())
        if (i // batch) % 10 == 0:
            print(f"  embedded {i + len(chunk)}/{len(texts)}")
    return np.vstack(embs).astype(np.float32)


def l2_normalize(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-8, None)


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    drug_sampled, disease_sampled, predictor = load_drug_and_disease_text()
    print(f"Drugs: {len(drug_sampled)}, Diseases: {len(disease_sampled)}")

    drug_ids = list(drug_sampled.keys())
    drug_texts = [drug_sampled[d] for d in drug_ids]
    disease_ids = list(disease_sampled.keys())
    disease_texts = [disease_sampled[d] for d in disease_ids]

    # Expanded GT for recall labels
    exp_gt_path = REFERENCE_DIR / "expanded_ground_truth.json"
    with open(exp_gt_path) as f:
        exp_gt = json.load(f)

    print("Embedding drugs ...")
    drug_emb = l2_normalize(embed_texts(drug_texts))
    print(f"Drug emb shape: {drug_emb.shape}")

    print("Embedding diseases ...")
    disease_emb = l2_normalize(embed_texts(disease_texts))
    print(f"Disease emb shape: {disease_emb.shape}")

    # Score: cosine similarity => rank drugs per disease
    sims = disease_emb @ drug_emb.T  # (D, N_drug)

    pubmed_r30: List[float] = []
    production_r30: List[float] = []
    bio_pubmed_r30: List[float] = []
    bio_prod_r30: List[float] = []

    for i, did in enumerate(disease_ids):
        gt_drugs = exp_gt.get(did, [])
        if not isinstance(gt_drugs, list):
            continue
        gt_in_sample = [d for d in gt_drugs if d in drug_sampled]
        if not gt_in_sample:
            continue
        gt_set = set(gt_in_sample)

        # PubMedBERT ranking
        top30_idx = np.argsort(-sims[i])[:30]
        top30_ids = {drug_ids[j] for j in top30_idx}
        pm_hits = len(top30_ids & gt_set)
        pm_r30 = pm_hits / min(len(gt_set), 30) if gt_set else 0.0
        pubmed_r30.append(pm_r30)

        # Production ranking on same drug pool
        try:
            result = predictor.predict(did, top_n=30, include_filtered=False)
        except Exception:
            result = None
        if result:
            prod_top30_ids = {p.drug_id for p in result.predictions[:30]}
            prod_top30_in_sample = prod_top30_ids & set(drug_sampled.keys())
            prod_hits = len(prod_top30_in_sample & gt_set)
            prod_r30 = prod_hits / min(len(gt_set), 30) if gt_set else 0.0
        else:
            prod_r30 = 0.0
        production_r30.append(prod_r30)

        # Biologic-only bucket: if most GT drugs for this disease are biologics
        bio_gt = sum(
            1 for d in gt_in_sample
            if "mab" in drug_sampled[d].lower() or "cept" in drug_sampled[d].lower()
        )
        if bio_gt >= 1:
            bio_pubmed_r30.append(pm_r30)
            bio_prod_r30.append(prod_r30)

    print("\n" + "=" * 60)
    print("  h956 PubMedBERT smoke test results")
    print("=" * 60)
    print(f"  Diseases evaluated: {len(pubmed_r30)}")
    print(f"  PubMedBERT R@30:    {np.mean(pubmed_r30):.4f} ± {np.std(pubmed_r30):.4f}")
    print(f"  Production  R@30:   {np.mean(production_r30):.4f} ± {np.std(production_r30):.4f}")
    ratio = np.mean(pubmed_r30) / max(np.mean(production_r30), 1e-6)
    print(f"  Ratio (pubmed/prod): {ratio:.2%}")
    if bio_pubmed_r30:
        print(f"  Biologic subset (n={len(bio_pubmed_r30)}):")
        print(f"    PubMedBERT: {np.mean(bio_pubmed_r30):.4f}")
        print(f"    Production: {np.mean(bio_prod_r30):.4f}")

    decision = (
        "GO_GPU_FULL" if ratio >= 0.60
        else ("FUSION_ONLY" if ratio >= 0.30 else "DEQUEUE_h920")
    )
    print(f"\n  Decision for h920: {decision}")

    out = {
        "pubmed_r30_mean": float(np.mean(pubmed_r30)),
        "pubmed_r30_std": float(np.std(pubmed_r30)),
        "production_r30_mean": float(np.mean(production_r30)),
        "production_r30_std": float(np.std(production_r30)),
        "ratio": float(ratio),
        "n_diseases": int(len(pubmed_r30)),
        "bio_n": int(len(bio_pubmed_r30)),
        "bio_pubmed_r30": float(np.mean(bio_pubmed_r30)) if bio_pubmed_r30 else None,
        "bio_prod_r30": float(np.mean(bio_prod_r30)) if bio_prod_r30 else None,
        "decision_for_h920": decision,
        "wall_time_seconds": float(time.time() - t0),
        "n_drugs_sampled": len(drug_sampled),
        "n_diseases_sampled": len(disease_sampled),
    }
    with open(ANALYSIS_DIR / "h956_pubmedbert_smoke.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to data/analysis/h956_pubmedbert_smoke.json")


if __name__ == "__main__":
    main()
