#!/usr/bin/env python3
"""
h927: Train a 3-layer MLP calibrator and compare against the current logistic-
regression classical calibrator on identical features. No GPU required — the
training set is ~1.5k samples over 15 features.

Usage:
    python3 scripts/h927_train_mlp_calibrator.py

Outputs:
    data/analysis/h927_calibrator_comparison.json — AUROC/AUPRC/ECE for both
    models/confidence_calibrator_mlp.pt — MLP weights (only if it beats LR)
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from confidence_calibration import (  # type: ignore[import-not-found]
    PredictionFeatures,
    classify_drug_type,
    classify_disease_category,
)
from pathway_features import PathwayEnrichment  # type: ignore[import-not-found]
from chemical_features import (  # type: ignore[import-not-found]
    DrugFingerprinter,
    compute_tanimoto_similarity,
)
from atc_features import ATCMapper  # type: ignore[import-not-found]

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
CACHE_PATH = ANALYSIS_DIR / "h927_training_features.npz"


def collect_features() -> tuple[np.ndarray, np.ndarray, List[dict]]:
    """Run the same feature pipeline as scripts/train_confidence_model.py."""
    print("Loading GB model...")
    with open(MODELS_DIR / "drug_repurposing_gb_enhanced.pkl", "rb") as f:
        model = pickle.load(f)

    print("Loading TransE embeddings...")
    checkpoint = torch.load(
        MODELS_DIR / "transe.pt", map_location="cpu", weights_only=False
    )
    embeddings = None
    if "entity_embeddings" in checkpoint:
        embeddings = checkpoint["entity_embeddings"].numpy()
    elif "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        for key in ["entity_embeddings.weight", "ent_embeddings.weight"]:
            if key in state:
                embeddings = state[key].numpy()
                break
    entity2id = checkpoint.get("entity2id", {})

    print("Loading targets / genes / mappings / GT...")
    with open(REFERENCE_DIR / "drug_targets.json") as f:
        drug_targets = {k: set(v) for k, v in json.load(f).items()}
    with open(REFERENCE_DIR / "disease_genes.json") as f:
        disease_genes = {k: set(v) for k, v in json.load(f).items()}
    with open(REFERENCE_DIR / "mesh_mappings_from_agents.json") as f:
        mesh_data = json.load(f)
    mesh_mappings: Dict[str, str] = {}
    for batch_data in mesh_data.values():
        if isinstance(batch_data, dict):
            for disease_name, mesh_id in batch_data.items():
                if mesh_id and str(mesh_id).startswith(("D", "C")):
                    mesh_mappings[disease_name.lower()] = (
                        f"drkg:Disease::MESH:{mesh_id}"
                    )
    with open(REFERENCE_DIR / "drugbank_lookup.json") as f:
        id_to_name = json.load(f)
    name_to_id = {
        name.lower(): f"drkg:Compound::{db_id}" for db_id, name in id_to_name.items()
    }
    id_to_drug_name = {
        f"drkg:Compound::{db_id}": name for db_id, name in id_to_name.items()
    }
    with open(REFERENCE_DIR / "everycure_gt_for_txgnn.json") as f:
        gt_raw = json.load(f)

    print("Loading feature modules...")
    pe = PathwayEnrichment()
    fingerprinter = DrugFingerprinter(use_cache=True)
    atc_mapper = ATCMapper()

    all_drug_ids = [eid for eid in entity2id.keys() if "Compound" in eid]
    valid_drug_ids: List[str] = []
    valid_drug_indices: List[int] = []
    for drug_id in all_drug_ids:
        drug_idx = entity2id.get(drug_id)
        if drug_idx is not None:
            valid_drug_ids.append(drug_id)
            valid_drug_indices.append(drug_idx)
    drug_embs = embeddings[valid_drug_indices]
    drug_id_to_local_idx = {did: i for i, did in enumerate(valid_drug_ids)}
    print(f"  Drugs with embeddings: {len(valid_drug_ids)}")

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    all_metadata: List[dict] = []

    for disease_name, disease_data in tqdm(gt_raw.items(), desc="Diseases"):
        mesh_id = mesh_mappings.get(disease_name.lower())
        if not mesh_id:
            continue
        disease_idx = entity2id.get(mesh_id)
        if disease_idx is None:
            continue
        mesh_short = mesh_id.split("MESH:")[-1]
        disease_emb = embeddings[disease_idx]
        dis_genes = disease_genes.get(f"MESH:{mesh_short}", set())
        dis_cats = classify_disease_category(disease_name)

        gt_drug_ids = set()
        gt_drug_names: List[str] = []
        for drug_info in disease_data.get("drugs", []):
            drug_name = drug_info["name"]
            drug_id = name_to_id.get(drug_name.lower())
            if drug_id and drug_id in drug_id_to_local_idx:
                gt_drug_ids.add(drug_id)
                gt_drug_names.append(drug_name)
        if not gt_drug_ids:
            continue

        gt_fps = []
        for dn in gt_drug_names:
            fp = fingerprinter.get_fingerprint(dn, fetch_if_missing=False)
            if fp is not None:
                gt_fps.append(fp)

        n_drugs = len(drug_embs)
        disease_emb_tiled = np.tile(disease_emb, (n_drugs, 1))
        concat_feats = np.hstack([drug_embs, disease_emb_tiled])
        product_feats = drug_embs * disease_emb_tiled
        diff_feats = drug_embs - disease_emb_tiled
        base_features = np.hstack([concat_feats, product_feats, diff_feats])
        base_scores = model.predict_proba(base_features)[:, 1]

        drug_features: List[PredictionFeatures] = []
        for i, drug_id in enumerate(valid_drug_ids):
            db_id = drug_id.split("::")[-1]
            drug_name = id_to_drug_name.get(drug_id, "")
            drug_types = classify_drug_type(drug_name)
            drug_genes = drug_targets.get(db_id, set())
            target_overlap = len(drug_genes & dis_genes)
            atc_score = atc_mapper.get_mechanism_score(drug_name, disease_name)
            po, _, _ = pe.get_pathway_overlap(db_id, f"MESH:{mesh_short}")
            chem_sim = 0.0
            query_fp = fingerprinter.get_fingerprint(drug_name, fetch_if_missing=False)
            if query_fp is not None and gt_fps:
                for gt_fp in gt_fps:
                    chem_sim = max(
                        chem_sim, compute_tanimoto_similarity(query_fp, gt_fp)
                    )
            boost = (
                1 + 0.01 * min(target_overlap, 10) + 0.05 * atc_score + 0.01 * min(po, 10)
            )
            if chem_sim > 0.7:
                boost *= 1.2
            boosted_score = float(base_scores[i] * boost)
            feats = PredictionFeatures(
                base_score=float(base_scores[i]),
                target_overlap=int(target_overlap),
                atc_score=float(atc_score),
                chemical_sim=float(chem_sim),
                pathway_overlap=int(po),
                is_biologic=drug_types["is_biologic"],
                is_kinase_inhibitor=drug_types["is_kinase_inhibitor"],
                is_antibiotic=drug_types["is_antibiotic"],
                is_cancer=dis_cats["is_cancer"],
                is_infectious=dis_cats["is_infectious"],
                is_autoimmune=dis_cats["is_autoimmune"],
                has_fingerprint=query_fp is not None,
                has_targets=len(drug_genes) > 0,
                has_atc=len(atc_mapper.get_atc_codes(drug_name)) > 0,
                boosted_score=boosted_score,
            )
            drug_features.append(feats)

        boosted_scores = np.array([f.boosted_score for f in drug_features])
        rankings = np.argsort(boosted_scores)[::-1]
        top_30_set = set(rankings[:30])

        for drug_id in gt_drug_ids:
            local_idx = drug_id_to_local_idx[drug_id]
            feats = drug_features[local_idx]
            label = 1 if local_idx in top_30_set else 0
            all_features.append(feats.to_array())
            all_labels.append(label)
            all_metadata.append(
                {
                    "disease": disease_name,
                    "drug": id_to_drug_name.get(drug_id, ""),
                    "label": label,
                }
            )

    X = np.asarray(all_features, dtype=np.float32)
    y = np.asarray(all_labels, dtype=np.int64)
    print(f"\nCollected {len(X)} samples, {y.sum()} positive ({100*y.mean():.1f}%)")
    return X, y, all_metadata


class MLPCalibrator(nn.Module):
    def __init__(self, in_dim: int, hidden: tuple[int, ...] = (64, 32, 16)):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.2)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _evaluate(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    brier = brier_score_loss(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = auc(recall, precision)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ece = float(np.mean(np.abs(prob_true - prob_pred)))
    return {"brier": float(brier), "auroc": float(auroc), "auprc": float(auprc), "ece": ece}


def train_mlp_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, epochs: int = 200) -> np.ndarray:
    """Cross-validated MLP probabilities. Returns CV predictions aligned with X."""
    probs = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    # Standardize features per fold
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        mu, sd = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        Xte = (X[te] - mu) / sd
        ytr = y[tr]
        # Class balancing via weighted BCE
        pos_frac = ytr.mean()
        pos_weight = torch.tensor((1 - pos_frac) / max(pos_frac, 1e-6), dtype=torch.float32)
        model = MLPCalibrator(in_dim=X.shape[1])
        opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        Xtr_t = torch.from_numpy(Xtr.astype(np.float32))
        ytr_t = torch.from_numpy(ytr.astype(np.float32))
        Xte_t = torch.from_numpy(Xte.astype(np.float32))
        for ep in range(epochs):
            model.train()
            opt.zero_grad()
            logits = model(Xtr_t)
            loss = loss_fn(logits, ytr_t)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            p = torch.sigmoid(model(Xte_t)).numpy()
        probs[te] = p
        print(f"  MLP fold {fold}/{n_splits}: train_loss={loss.item():.4f}")
    return probs


def train_lr_cv(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Match the classical calibrator: LogisticRegression class_weight=balanced."""
    probs = np.zeros(len(y), dtype=np.float32)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in skf.split(X, y):
        lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
        lr.fit(X[tr], y[tr])
        probs[te] = lr.predict_proba(X[te])[:, 1]
    return probs


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists():
        print(f"Loading cached features from {CACHE_PATH}")
        cache = np.load(CACHE_PATH, allow_pickle=True)
        X = cache["X"]
        y = cache["y"]
    else:
        t0 = time.time()
        X, y, meta = collect_features()
        np.savez(CACHE_PATH, X=X, y=y)
        print(f"Collected in {time.time()-t0:.0f}s, cached to {CACHE_PATH}")

    print(f"\nFeatures: {X.shape}, labels: {y.shape}, positive rate: {100*y.mean():.1f}%")

    print("\n1. Classical LR (5-fold CV) ...")
    lr_probs = train_lr_cv(X, y)
    lr_metrics = _evaluate(y, lr_probs)

    print("\n2. MLP (5-fold CV) ...")
    mlp_probs = train_mlp_cv(X, y)
    mlp_metrics = _evaluate(y, mlp_probs)

    print("\n" + "=" * 70)
    print("  CALIBRATOR COMPARISON (h927)")
    print("=" * 70)
    print(f"  Metric          LR        MLP       Δ")
    for k in ("auroc", "auprc", "brier", "ece"):
        lv, mv = lr_metrics[k], mlp_metrics[k]
        delta = mv - lv
        marker = "better" if (delta > 0 if k in ("auroc", "auprc") else delta < 0) else "worse "
        print(f"  {k:<12}  {lv:>6.4f}   {mv:>6.4f}   {delta:+.4f}  ({marker})")

    auroc_gain = mlp_metrics["auroc"] - lr_metrics["auroc"]
    decision = "SWAP-IN" if auroc_gain >= 0.03 else (
        "MARGINAL" if auroc_gain >= 0.005 else "KEEP-LR"
    )
    print(f"\n  Decision: {decision}  (AUROC gain: {auroc_gain:+.4f})")

    out = {
        "lr": lr_metrics,
        "mlp": mlp_metrics,
        "decision": decision,
        "auroc_gain": float(auroc_gain),
        "n_samples": int(len(y)),
        "positive_rate": float(y.mean()),
        "n_features": int(X.shape[1]),
    }
    with open(ANALYSIS_DIR / "h927_calibrator_comparison.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {ANALYSIS_DIR / 'h927_calibrator_comparison.json'}")

    if decision in ("SWAP-IN", "MARGINAL"):
        # Retrain final MLP on all data and save
        mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-8
        Xn = (X - mu) / sd
        final = MLPCalibrator(in_dim=X.shape[1])
        opt = torch.optim.AdamW(final.parameters(), lr=2e-3, weight_decay=1e-4)
        pos_weight = torch.tensor(
            (1 - y.mean()) / max(y.mean(), 1e-6), dtype=torch.float32
        )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        Xt = torch.from_numpy(Xn.astype(np.float32))
        yt = torch.from_numpy(y.astype(np.float32))
        for ep in range(200):
            final.train()
            opt.zero_grad()
            loss = loss_fn(final(Xt), yt)
            loss.backward()
            opt.step()
        torch.save(
            {
                "state_dict": final.state_dict(),
                "in_dim": X.shape[1],
                "hidden": (64, 32, 16),
                "feat_mean": mu,
                "feat_std": sd,
                "decision": decision,
                "cv_metrics": mlp_metrics,
            },
            MODELS_DIR / "confidence_calibrator_mlp.pt",
        )
        print(f"Saved MLP weights to {MODELS_DIR / 'confidence_calibrator_mlp.pt'}")


if __name__ == "__main__":
    main()
