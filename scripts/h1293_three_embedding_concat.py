#!/usr/bin/env python3
"""h1293: Three-embedding concat_l2 — N2V + FastRP + TransE.

h1287 exhausted the linear-weight axis on 2-embedding soft-blend (R@30 plateau
21.54-21.71% across w ∈ [0.25, 0.45]). The next axis to open is embedding
diversity: does adding a translational embedding (TransE) sample a different
structural prior than random-walk (N2V) and random-projection (FastRP) and
lift the DRKG ceiling?

`models/transe.pt` is a PyKEEN/DGL-style checkpoint (273,581 entities × 128
dim) trained on DRKG + PrimeKG + Hetionet union with the `drkg:` namespace
prefix. After stripping `drkg:` from keys, 100% of Node2Vec's 49,616
entities are covered.

Compared modes (20 seeds, SUBSET_D_GLOBAL):

    concat_l2_2way       — anchor: L2(concat(l2(n2v) | l2(fastrp)))  (512-d)
    concat_l2_3way       — candidate: L2(concat(l2(n2v) | l2(fastrp) | l2(transe)))  (640-d)
    soft_blend_w050_2way — h1269 canonical on 2-way: 0.5·z(n2v) + 0.5·z(concat_2way)
    soft_blend_w050_3way — extend canonical to 3-way: 0.5·z(n2v) + 0.5·z(concat_3way)

Promotion gate (preregistered):
    Δ R@30 > +0.3pp AND Δ per-dis AUPRC > +0.001 at p < 0.05 vs concat_l2_2way.

If h1293 FAILS the gate, the linear embedding-stacking axis joins the linear
weight axis as exhausted, and future recall lift requires non-linear
combiners (h1266 RRF) or external signals (h1202 LINCS-as-feature).

Outputs:
    data/analysis/h1293_three_embedding_concat.{json,md}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clean_embedding_benchmark import (  # noqa: E402
    load_embeddings,
    mean_std,
    split_diseases,
)
from h1215_fusion_benchmark import (  # noqa: E402
    build_concat_lookup,
    l2_normalise,
    score_disease_single,
)
from h1249_entropy_routed_benchmark import paired_t  # noqa: E402
from h1255_soft_blend_fusion import z_normalise  # noqa: E402

OUT_JSON = PROJECT_ROOT / "data/analysis/h1293_three_embedding_concat.json"
OUT_MD = PROJECT_ROOT / "data/analysis/h1293_three_embedding_concat.md"


def load_transe(checkpoint_path: Path) -> Dict[str, np.ndarray]:
    """Load TransE entity embeddings keyed by full `drkg:`-prefixed ID.

    `clean_embedding_benchmark.load_embeddings` prepends `drkg:` to every
    Node2Vec/FastRP entity ID, and knn_gt/expanded_gt caches are stored with
    the same prefix. TransE's native keys already start with `drkg:`, so we
    keep them as-is for a direct dict intersection.
    """
    m = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    entity2id = m["entity2id"]
    emb = m["model_state_dict"]["entity_embeddings.weight"].detach().cpu().numpy().astype(np.float32)
    lu: Dict[str, np.ndarray] = {}
    for key, idx in entity2id.items():
        if key.startswith("drkg:"):
            lu[key] = emb[idx]
    return lu


def build_concat3_lookup(
    lu_a: Dict[str, np.ndarray],
    lu_b: Dict[str, np.ndarray],
    lu_c: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """L2-normalise each sub-embedding then concatenate, keyed by 3-way intersection."""
    keys = sorted(set(lu_a) & set(lu_b) & set(lu_c))
    mat_a = l2_normalise(np.stack([lu_a[k] for k in keys]))
    mat_b = l2_normalise(np.stack([lu_b[k] for k in keys]))
    mat_c = l2_normalise(np.stack([lu_c[k] for k in keys]))
    concat = np.concatenate([mat_a, mat_b, mat_c], axis=1).astype(np.float32)
    return {k: concat[i] for i, k in enumerate(keys)}


MODES = (
    "concat_l2_2way",
    "concat_l2_3way",
    "soft_blend_w050_2way",
    "soft_blend_w050_3way",
)
BLEND_W = 0.5


def evaluate_seed(
    *,
    seed: int,
    lu_n2v: Dict[str, np.ndarray],
    lu_concat_2: Dict[str, np.ndarray],
    lu_concat_3: Dict[str, np.ndarray],
    all_diseases: List[str],
    knn_gt: Dict[str, Set[str]],
    eval_gt: Dict[str, Set[str]],
    k: int,
) -> Dict:
    train_ids, holdout_ids = split_diseases(all_diseases, seed)
    common = set(lu_n2v) & set(lu_concat_2) & set(lu_concat_3)
    train_ids_ordered = [d for d in train_ids if d in common]

    def build_emb_mat(lu: Dict[str, np.ndarray]) -> np.ndarray:
        return np.stack([lu[d] for d in train_ids_ordered])

    train_emb_n2v = build_emb_mat(lu_n2v)
    train_emb_c2 = build_emb_mat(lu_concat_2)
    train_emb_c3 = build_emb_mat(lu_concat_3)

    train_gt = {
        d: (knn_gt[d] & common) for d in train_ids_ordered if d in knn_gt
    }

    cand_drugs: Set[str] = set()
    for d in train_ids_ordered:
        if d in knn_gt:
            cand_drugs |= (knn_gt[d] & common)
    all_hold_drugs: Set[str] = set()
    for did in holdout_ids:
        all_hold_drugs |= eval_gt.get(did, set())
    universe = (cand_drugs | all_hold_drugs) & common
    cand_list = sorted(universe)
    cand_index = {d: i for i, d in enumerate(cand_list)}
    n_cands = len(cand_list)

    per_drug_r30: Dict[str, List[float]] = {m: [] for m in MODES}
    per_disease_auprc: Dict[str, List[float]] = {m: [] for m in MODES}
    per_disease_auroc: Dict[str, List[float]] = {m: [] for m in MODES}
    pool_score: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
    pool_label: Dict[str, List[np.ndarray]] = {m: [] for m in MODES}
    n_eval = 0

    for did in holdout_ids:
        if did not in common:
            continue
        gt_drugs = eval_gt.get(did, set()) & universe
        if not gt_drugs:
            continue
        n_eval += 1

        # Raw scores
        def raw_scores(lu: Dict[str, np.ndarray], train_emb: np.ndarray) -> np.ndarray:
            ds = score_disease_single(lu[did], train_emb, train_ids_ordered, train_gt, k)
            sv = np.zeros(n_cands, dtype=np.float32)
            for drug, sc in ds.items():
                if drug in cand_index:
                    sv[cand_index[drug]] = sc
            return sv

        sv_n2v = raw_scores(lu_n2v, train_emb_n2v)
        sv_c2 = raw_scores(lu_concat_2, train_emb_c2)
        sv_c3 = raw_scores(lu_concat_3, train_emb_c3)

        z_n2v = z_normalise(sv_n2v)
        z_c2 = z_normalise(sv_c2)
        z_c3 = z_normalise(sv_c3)

        score_per_mode: Dict[str, np.ndarray] = {
            "concat_l2_2way": sv_c2,
            "concat_l2_3way": sv_c3,
            "soft_blend_w050_2way": BLEND_W * z_n2v + (1.0 - BLEND_W) * z_c2,
            "soft_blend_w050_3way": BLEND_W * z_n2v + (1.0 - BLEND_W) * z_c3,
        }

        label_vec = np.zeros(n_cands, dtype=np.int8)
        for d in gt_drugs:
            idx = cand_index.get(d)
            if idx is not None:
                label_vec[idx] = 1
        n_pos = int(label_vec.sum())
        if n_pos == 0:
            continue

        n_gt = len(gt_drugs)
        for mode, score_vec in score_per_mode.items():
            order = np.argsort(-score_vec, kind="stable")
            rank_of_idx = {int(idx): r + 1 for r, idx in enumerate(order)}
            hits_30 = sum(
                1 for d in gt_drugs
                if cand_index.get(d) is not None and rank_of_idx.get(cand_index[d], 10**9) <= 30
            )
            per_drug_r30[mode].append(hits_30 / n_gt)
            if 0 < n_pos < n_cands:
                per_disease_auprc[mode].append(float(average_precision_score(label_vec, score_vec)))
                per_disease_auroc[mode].append(float(roc_auc_score(label_vec, score_vec)))
            elif n_pos == n_cands:
                per_disease_auprc[mode].append(1.0)
            pool_score[mode].append(score_vec.copy())
            pool_label[mode].append(label_vec.copy())

    out: Dict[str, Dict] = {}
    for m in MODES:
        sf = np.concatenate(pool_score[m]) if pool_score[m] else np.zeros(0)
        lf = np.concatenate(pool_label[m]) if pool_label[m] else np.zeros(0)
        pa = float(average_precision_score(lf, sf)) if lf.sum() > 0 else 0.0
        pr = float(roc_auc_score(lf, sf)) if 0 < lf.sum() < len(lf) else 0.0
        out[m] = {
            "per_drug_r30": float(np.mean(per_drug_r30[m])) if per_drug_r30[m] else 0.0,
            "per_disease_auprc_mean": float(np.mean(per_disease_auprc[m])) if per_disease_auprc[m] else 0.0,
            "per_disease_auroc_mean": float(np.mean(per_disease_auroc[m])) if per_disease_auroc[m] else 0.0,
            "pooled_auprc": pa,
            "pooled_auroc": pr,
            "n_test_diseases": len(per_drug_r30[m]),
        }
    return {
        "seed": seed,
        "n_eval": n_eval,
        "modes": out,
    }


def paired_t_vs(results: List[Dict], mode: str, ref: str) -> Dict[str, Dict]:
    d_r30 = [r["modes"][mode]["per_drug_r30"] - r["modes"][ref]["per_drug_r30"] for r in results]
    d_pd_ap = [r["modes"][mode]["per_disease_auprc_mean"] - r["modes"][ref]["per_disease_auprc_mean"] for r in results]
    d_pd_auc = [r["modes"][mode]["per_disease_auroc_mean"] - r["modes"][ref]["per_disease_auroc_mean"] for r in results]
    return {
        "R@30": {"mean": paired_t(d_r30)["mean"], "p": paired_t(d_r30)["p_two_sided"]},
        "per_disease_AUPRC": {"mean": paired_t(d_pd_ap)["mean"], "p": paired_t(d_pd_ap)["p_two_sided"]},
        "per_disease_AUROC": {"mean": paired_t(d_pd_auc)["mean"], "p": paired_t(d_pd_auc)["p_two_sided"]},
    }


DEFAULT_SEEDS = "42,123,456,789,2024,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix-a", default="node2vec_256")
    ap.add_argument("--prefix-b", default="fastrp_256")
    ap.add_argument("--transe", default="models/transe.pt")
    ap.add_argument("--seeds", type=str, default=DEFAULT_SEEDS)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--eval-gt", default="data/reference/expanded_ground_truth.json")
    ap.add_argument("--knn-gt", default="data/cache/ground_truth_cache.json")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    print("=" * 72)
    print(f"h1293: Three-embedding concat_l2 — N2V + FastRP + TransE")
    print(f"  seeds={len(seeds)}")
    print("=" * 72)

    t0 = time.time()
    print("Loading embeddings...")
    lu_a, _, _ = load_embeddings(args.prefix_a)
    lu_b, _, _ = load_embeddings(args.prefix_b)
    lu_transe = load_transe(PROJECT_ROOT / args.transe)
    print(
        f"  N2V={len(lu_a):,}  FastRP={len(lu_b):,}  "
        f"TransE_raw={len(lu_transe):,}  "
        f"3-way intersection={len(set(lu_a) & set(lu_b) & set(lu_transe)):,}"
    )

    lu_concat_2 = build_concat_lookup(lu_a, lu_b)
    lu_concat_3 = build_concat3_lookup(lu_a, lu_b, lu_transe)
    print(f"  concat_2way={len(lu_concat_2):,}  concat_3way={len(lu_concat_3):,}")

    with open(PROJECT_ROOT / args.eval_gt) as f:
        eval_gt = {d: set(v) for d, v in json.load(f).items()}
    with open(PROJECT_ROOT / args.knn_gt) as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "ground_truth" in raw:
        raw = raw["ground_truth"]
    knn_gt = {d: set(v) for d, v in raw.items()}

    intersection_keys = set(lu_a) & set(lu_b) & set(lu_transe)
    all_diseases = [d for d in knn_gt if d in intersection_keys]
    print(f"Universe: {len(all_diseases):,} diseases  | setup {time.time()-t0:.1f}s")

    per_seed_results: List[Dict] = []
    for seed in seeds:
        ts = time.time()
        seed_out = evaluate_seed(
            seed=seed, lu_n2v=lu_a, lu_concat_2=lu_concat_2, lu_concat_3=lu_concat_3,
            all_diseases=all_diseases, knn_gt=knn_gt, eval_gt=eval_gt, k=args.k,
        )
        per_seed_results.append(seed_out)
        print(f"\n  seed {seed} ({time.time()-ts:.1f}s) n_eval={seed_out['n_eval']}")
        for m in MODES:
            mm = seed_out["modes"][m]
            print(
                f"    {m:24s} R@30={mm['per_drug_r30']*100:.2f}%  "
                f"per-dis-AUPRC={mm['per_disease_auprc_mean']:.4f}  "
                f"per-dis-AUROC={mm['per_disease_auroc_mean']:.4f}"
            )

    print("\n" + "=" * 72)
    print(f"AGGREGATE (mean ± std, {len(seeds)} seeds)")
    print("=" * 72)
    agg: Dict[str, Dict] = {}
    for m in MODES:
        rows = [r["modes"][m] for r in per_seed_results]
        agg[m] = {
            "r30": mean_std([r["per_drug_r30"] for r in rows]),
            "per_disease_auprc": mean_std([r["per_disease_auprc_mean"] for r in rows]),
            "per_disease_auroc": mean_std([r["per_disease_auroc_mean"] for r in rows]),
            "pooled_auprc": mean_std([r["pooled_auprc"] for r in rows]),
            "pooled_auroc": mean_std([r["pooled_auroc"] for r in rows]),
        }
        print(
            f"  {m:24s}  R@30={agg[m]['r30'][0]*100:5.2f}%±{agg[m]['r30'][1]*100:.2f}%  "
            f"per-dis-AUPRC={agg[m]['per_disease_auprc'][0]:.4f}±{agg[m]['per_disease_auprc'][1]:.4f}  "
            f"per-dis-AUROC={agg[m]['per_disease_auroc'][0]:.4f}±{agg[m]['per_disease_auroc'][1]:.4f}"
        )

    # Paired-t comparisons
    print("\n" + "=" * 72)
    print("PAIRED-T COMPARISONS")
    print("=" * 72)
    comparisons = [
        ("concat_l2_3way", "concat_l2_2way", "3-way concat vs 2-way anchor"),
        ("soft_blend_w050_3way", "soft_blend_w050_2way", "3-way soft-blend vs 2-way canonical"),
        ("soft_blend_w050_3way", "concat_l2_2way", "3-way soft-blend vs 2-way anchor"),
        ("soft_blend_w050_3way", "concat_l2_3way", "3-way soft-blend vs 3-way raw"),
    ]
    paired: Dict[str, Dict] = {}
    for mode, ref, label in comparisons:
        rt = paired_t_vs(per_seed_results, mode, ref)
        paired[f"{mode}_vs_{ref}"] = rt
        print(
            f"\n  {label}:"
            f"\n    R@30           Δ={rt['R@30']['mean']*100:+.4f}pp  p={rt['R@30']['p']:.4g}"
            f"\n    per-dis-AUPRC  Δ={rt['per_disease_AUPRC']['mean']:+.5f}  p={rt['per_disease_AUPRC']['p']:.4g}"
            f"\n    per-dis-AUROC  Δ={rt['per_disease_AUROC']['mean']:+.5f}  p={rt['per_disease_AUROC']['p']:.4g}"
        )

    # Promotion gate (3-way concat vs 2-way concat)
    print("\n" + "=" * 72)
    print("PROMOTION GATE: 3-way beats 2-way at ΔR@30 > +0.3pp AND Δper-dis AUPRC > +0.001 at p<0.05")
    print("=" * 72)
    gate_3vs2 = paired[f"concat_l2_3way_vs_concat_l2_2way"]
    r30_pass = gate_3vs2["R@30"]["mean"] * 100 > 0.3 and gate_3vs2["R@30"]["p"] < 0.05
    ap_pass = gate_3vs2["per_disease_AUPRC"]["mean"] > 0.001 and gate_3vs2["per_disease_AUPRC"]["p"] < 0.05
    passes = r30_pass and ap_pass
    decision = "PROMOTE 3-way concat as canonical" if passes else "STAY with 2-way concat"
    print(
        f"  concat_l2_3way vs concat_l2_2way:"
        f"\n    ΔR@30={gate_3vs2['R@30']['mean']*100:+.4f}pp p={gate_3vs2['R@30']['p']:.4g}  R@30_gate={'PASS' if r30_pass else 'FAIL'}"
        f"\n    ΔAUPRC={gate_3vs2['per_disease_AUPRC']['mean']:+.5f} p={gate_3vs2['per_disease_AUPRC']['p']:.4g}  AUPRC_gate={'PASS' if ap_pass else 'FAIL'}"
        f"\n    → {decision}"
    )

    # JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump({
            "hypothesis": "h1293",
            "seeds": seeds,
            "n_diseases_universe": len(all_diseases),
            "aggregate": {m: {k: list(v) if isinstance(v, tuple) else v for k, v in d.items()} for m, d in agg.items()},
            "paired_t": paired,
            "promotion_gate": {
                "r30_pass": bool(r30_pass),
                "auprc_pass": bool(ap_pass),
                "passes_gate": bool(passes),
                "decision": decision,
                "delta_r30_pp": gate_3vs2["R@30"]["mean"] * 100,
                "r30_p": gate_3vs2["R@30"]["p"],
                "delta_auprc": gate_3vs2["per_disease_AUPRC"]["mean"],
                "auprc_p": gate_3vs2["per_disease_AUPRC"]["p"],
            },
            "per_seed_summaries": per_seed_results,
        }, f, indent=2)
    print(f"\nWrote {OUT_JSON.relative_to(PROJECT_ROOT)}")

    # Markdown
    md: List[str] = []
    md.append("# h1293 — Three-embedding concat_l2 (N2V + FastRP + TransE) (20-seed SUBSET_D_GLOBAL)\n\n")
    md.append(f"**Universe:** {len(all_diseases):,} diseases (3-way embedding intersection ∩ knn_gt)\n\n")
    md.append(f"## Aggregate (mean ± std across {len(seeds)} seeds)\n\n")
    md.append("| Mode | R@30 | per-dis-AUPRC | per-dis-AUROC | pooled-AUPRC | pooled-AUROC |\n")
    md.append("|---|---|---|---|---|---|\n")
    for m in MODES:
        a = agg[m]
        md.append(
            f"| `{m}` | "
            f"{a['r30'][0]*100:.2f}%±{a['r30'][1]*100:.2f}% | "
            f"{a['per_disease_auprc'][0]:.4f}±{a['per_disease_auprc'][1]:.4f} | "
            f"{a['per_disease_auroc'][0]:.4f}±{a['per_disease_auroc'][1]:.4f} | "
            f"{a['pooled_auprc'][0]:.4f}±{a['pooled_auprc'][1]:.4f} | "
            f"{a['pooled_auroc'][0]:.4f}±{a['pooled_auroc'][1]:.4f} |\n"
        )

    md.append("\n## Paired-t comparisons\n\n")
    md.append("| Comparison | ΔR@30 (p) | Δper-dis-AUPRC (p) | Δper-dis-AUROC (p) |\n|---|---|---|---|\n")
    for mode, ref, label in comparisons:
        rt = paired[f"{mode}_vs_{ref}"]
        md.append(
            f"| {label} | "
            f"{rt['R@30']['mean']*100:+.4f}pp ({rt['R@30']['p']:.3g}) | "
            f"{rt['per_disease_AUPRC']['mean']:+.5f} ({rt['per_disease_AUPRC']['p']:.3g}) | "
            f"{rt['per_disease_AUROC']['mean']:+.5f} ({rt['per_disease_AUROC']['p']:.3g}) |\n"
        )

    md.append("\n## Promotion gate\n\n")
    md.append("3-way concat_l2 vs 2-way concat_l2 anchor; pass if ΔR@30>+0.3pp AND ΔAUPRC>+0.001 both at p<0.05.\n\n")
    md.append(
        f"- ΔR@30 = {gate_3vs2['R@30']['mean']*100:+.4f}pp (p={gate_3vs2['R@30']['p']:.3g})  "
        f"→ R@30 gate: **{'PASS' if r30_pass else 'FAIL'}**\n"
        f"- ΔAUPRC = {gate_3vs2['per_disease_AUPRC']['mean']:+.5f} (p={gate_3vs2['per_disease_AUPRC']['p']:.3g})  "
        f"→ AUPRC gate: **{'PASS' if ap_pass else 'FAIL'}**\n"
        f"- **Decision: {decision}**\n"
    )

    with open(OUT_MD, "w") as f:
        f.write("".join(md))
    print(f"Wrote {OUT_MD.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
