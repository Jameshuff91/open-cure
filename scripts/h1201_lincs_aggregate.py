#!/usr/bin/env python3
"""h1201 Phase B — aggregate LINCS Level 5 signatures to one per compound.

Reads GSE92742 Level 5 COMPZ.MODZ gctx, filters to trt_cp 24h top-dose
signatures, takes mean across cell lines per compound. Maps LINCS pert_id
to DrugBank ID via InChIKey or PubChem CID (Phase A bridge).

Output:
  data/embeddings/lincs_signatures.npy    — (n_compounds, n_genes) float32
  data/embeddings/lincs_drug_ids.npy      — (n_compounds,) DrugBank IDs
  data/embeddings/lincs_gene_ids.npy      — (n_genes,) Entrez gene IDs
  data/embeddings/lincs_meta.json         — provenance

Runtime: ~5 min on a laptop. Pure CPU.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

TREATMENT_TYPE = "trt_cp"
TIME_HOURS = "24"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--lincs-dir", default="data/external/lincs")
    p.add_argument("--drugbank-vocab",
                   default="data/external/drugbank/drugbank vocabulary.csv")
    p.add_argument("--drug-mappings",
                   default="data/external/drug-mappings.tsv")
    p.add_argument("--out-dir", default="data/embeddings")
    p.add_argument("--dose-threshold", type=float, default=9.0,
                   help="Keep signatures with pert_dose >= this (µM). Default 9.0 captures 10µM nominal.")
    p.add_argument("--landmark-only", action="store_true",
                   help="Keep only the 978 landmark genes (pr_is_lm=1) instead of all 12,328")
    return p.parse_args()


def load_drugbank_bridge(vocab_path: Path, mappings_path: Path) -> tuple[Dict[str, str], Dict[str, str]]:
    db_inchi: Dict[str, str] = {}
    with open(vocab_path) as f:
        for row in csv.DictReader(f):
            ik = row.get("Standard InChI Key", "").strip()
            if ik:
                db_inchi[ik] = row["DrugBank ID"]
    db_pubchem: Dict[str, str] = {}
    with open(mappings_path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 4 and parts[3] and parts[3] != "null":
                db_pubchem[parts[3]] = parts[0]
    return db_inchi, db_pubchem


def map_lincs_perts_to_drugbank(
    pert_info_gz: Path, db_inchi: Dict[str, str], db_pubchem: Dict[str, str]
) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with gzip.open(pert_info_gz, "rt") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[2] != TREATMENT_TYPE:
                continue
            inchi, pcid = p[5], p[7]
            db = None
            if inchi != "-666" and inchi in db_inchi:
                db = db_inchi[inchi]
            elif pcid != "-666" and pcid in db_pubchem:
                db = db_pubchem[pcid]
            if db:
                mapping[p[0]] = db
    return mapping


def select_signatures(
    sig_info_gz: Path, pert_to_db: Dict[str, str], dose_threshold: float
) -> Dict[str, List[str]]:
    """Return {pert_id: [sig_id, ...]} for trt_cp 24h high-dose only."""
    keep: Dict[str, List[str]] = {}
    with gzip.open(sig_info_gz, "rt") as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 11 or p[3] != TREATMENT_TYPE:
                continue
            pert_id = p[1]
            if pert_id not in pert_to_db:
                continue
            if p[8] != TIME_HOURS or p[9] != "h":
                continue
            try:
                dose = float(p[5])
            except ValueError:
                continue
            if dose < dose_threshold:
                continue
            keep.setdefault(pert_id, []).append(p[0])
    return keep


def load_landmark_mask(gene_info_gz: Path) -> tuple[np.ndarray, np.ndarray]:
    gene_ids: List[str] = []
    is_lm: List[int] = []
    with gzip.open(gene_info_gz, "rt") as f:
        header = next(f).rstrip("\n").split("\t")
        lm_idx = header.index("pr_is_lm")
        gid_idx = header.index("pr_gene_id")
        for line in f:
            p = line.rstrip("\n").split("\t")
            gene_ids.append(p[gid_idx])
            is_lm.append(int(p[lm_idx]))
    return np.asarray(gene_ids), np.asarray(is_lm, dtype=bool)


def main() -> None:
    args = parse_args()
    lincs_dir = Path(args.lincs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    import h5py

    gctx_gz = lincs_dir / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
    gctx = lincs_dir / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"
    if not gctx.exists():
        if not gctx_gz.exists():
            raise SystemExit(f"Level 5 file missing: {gctx_gz}")
        print(f"Decompressing {gctx_gz.name} ...")
        import subprocess
        subprocess.run(["gunzip", "-k", str(gctx_gz)], check=True)
    print(f"Level 5 gctx: {gctx}")

    print("Mapping LINCS → DrugBank via Phase A bridge ...")
    db_inchi, db_pubchem = load_drugbank_bridge(
        Path(args.drugbank_vocab), Path(args.drug_mappings)
    )
    pert_to_db = map_lincs_perts_to_drugbank(
        lincs_dir / "GSE92742_Broad_LINCS_pert_info.txt.gz", db_inchi, db_pubchem
    )
    print(f"  {len(pert_to_db):,} LINCS compounds map to DrugBank")

    print(f"Selecting trt_cp 24h dose>={args.dose_threshold}µM signatures ...")
    pert_sigs = select_signatures(
        lincs_dir / "GSE92742_Broad_LINCS_sig_info.txt.gz",
        pert_to_db,
        args.dose_threshold,
    )
    total_sigs = sum(len(v) for v in pert_sigs.values())
    print(f"  {len(pert_sigs):,} compounds, {total_sigs:,} signatures to aggregate")

    gene_ids, is_lm = load_landmark_mask(
        lincs_dir / "GSE92742_Broad_LINCS_gene_info.txt.gz"
    )
    gene_filter = is_lm if args.landmark_only else np.ones(len(gene_ids), dtype=bool)
    n_genes_out = int(gene_filter.sum())
    print(f"  Gene filter: {'landmark 978' if args.landmark_only else 'all 12,328'} → {n_genes_out} genes kept")

    print(f"Opening {gctx.name} ...")
    with h5py.File(gctx, "r") as f:
        sig_id_arr = f["/0/META/COL/id"][:].astype(str)
        gene_id_arr = f["/0/META/ROW/id"][:].astype(str)
        matrix_ds = f["/0/DATA/0/matrix"]
        print(f"  matrix shape: {matrix_ds.shape}  (sig_rows x gene_cols)")

        sig_idx = {sid: i for i, sid in enumerate(sig_id_arr)}
        if not np.array_equal(gene_id_arr, gene_ids):
            gid_to_gctx_col = {g: i for i, g in enumerate(gene_id_arr)}
            col_reorder = np.asarray(
                [gid_to_gctx_col[g] for g in gene_ids if g in gid_to_gctx_col],
                dtype=np.int64,
            )
            gene_filter_gctx = gene_filter[: len(col_reorder)]
            out_gene_ids = gene_ids[gene_filter]
        else:
            col_reorder = np.arange(len(gene_ids))
            gene_filter_gctx = gene_filter
            out_gene_ids = gene_ids[gene_filter]

        final_col_idx = col_reorder[gene_filter_gctx]

        db_to_sum: Dict[str, np.ndarray] = {}
        db_to_count: Dict[str, int] = {}
        dropped_no_match = 0
        processed = 0
        total = len(pert_sigs)
        for pert_id, sig_ids in sorted(pert_sigs.items()):
            processed += 1
            rows = [sig_idx.get(s) for s in sig_ids]
            rows = [r for r in rows if r is not None]
            if not rows:
                dropped_no_match += 1
                continue
            rows_sorted = sorted(set(rows))
            sub = matrix_ds[rows_sorted, :]
            sub = sub[:, final_col_idx]
            sig = sub.mean(axis=0).astype(np.float32)
            db_id = pert_to_db[pert_id]
            if db_id in db_to_sum:
                db_to_sum[db_id] = db_to_sum[db_id] + sig
                db_to_count[db_id] += 1
            else:
                db_to_sum[db_id] = sig
                db_to_count[db_id] = 1
            if processed % 200 == 0:
                print(f"  {processed}/{total} compounds processed")

        drugs_ordered = sorted(db_to_sum.keys())
        signatures = [db_to_sum[d] / db_to_count[d] for d in drugs_ordered]

    sig_matrix = np.stack(signatures, axis=0)
    print(f"Aggregated {sig_matrix.shape[0]:,} unique DrugBank compounds × {sig_matrix.shape[1]:,} genes")
    if dropped_no_match:
        print(f"  dropped {dropped_no_match} compounds with no matching sigs in gctx")

    np.save(out_dir / "lincs_signatures.npy", sig_matrix)
    np.save(out_dir / "lincs_drug_ids.npy",
            np.asarray(drugs_ordered, dtype=object), allow_pickle=True)
    np.save(out_dir / "lincs_gene_ids.npy", out_gene_ids)

    meta = {
        "hypothesis": "h1201",
        "source": "GSE92742 LINCS Phase I Level 5 COMPZ.MODZ",
        "n_compounds": int(sig_matrix.shape[0]),
        "n_genes": int(sig_matrix.shape[1]),
        "time_hours": TIME_HOURS,
        "dose_threshold_uM": args.dose_threshold,
        "landmark_only": args.landmark_only,
        "aggregation": "mean across cell lines",
    }
    (out_dir / "lincs_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {out_dir}/lincs_{{signatures,drug_ids,gene_ids,meta}}.npy/.json")


if __name__ == "__main__":
    main()
