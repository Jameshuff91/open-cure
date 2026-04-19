#!/usr/bin/env python3
"""
h922-v2 evaluation: swap the new GraphSAGE embeddings into production_predictor
and re-run h393 5-seed tier holdout for side-by-side comparison with Node2Vec.

Assumes h922_v2_graphsage_retrain.py has already produced:
    data/embeddings/graphsage_256_entities.npy
    data/embeddings/graphsage_256_embeddings.npy

Usage:
    python3 scripts/h922_v2_evaluate.py

Outputs:
    data/analysis/h922_v2_vs_node2vec.json
    data/analysis/h922_v2_h393_graphsage.txt
    data/analysis/h393_holdout_validation.json is OVERWRITTEN twice during
        the run — we snapshot it before/after and restore the Node2Vec
        version at the end so production tooling is unaffected.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
ANALYSIS_DIR = PROJECT_ROOT / "data" / "analysis"
EMB_DIR = PROJECT_ROOT / "data" / "embeddings"
H393 = PROJECT_ROOT / "scripts" / "h393_holdout_tier_validation.py"
H393_OUT = ANALYSIS_DIR / "h393_holdout_validation.json"


def run_h393(prefix: str, log_path: Path) -> dict:
    env = os.environ.copy()
    env["OPEN_CURE_EMBEDDINGS_PREFIX"] = prefix
    with open(log_path, "w") as f:
        subprocess.run(
            [sys.executable, "-u", str(H393)],
            env=env, stdout=f, stderr=subprocess.STDOUT, check=True,
        )
    with open(H393_OUT) as f:
        return json.load(f)


def summarize(name: str, d: dict) -> dict:
    tc = d["tier_comparison"]
    out = {}
    for t in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        info = tc[t]
        out[t] = {
            "full_precision": info["full_precision"],
            "holdout_mean": info["holdout_mean"],
            "holdout_std": info["holdout_std"],
        }
    # Pull per-drug R@30 if the script reports it (h393 reports tier precision
    # but per-drug R@30 may be elsewhere). Extract from rule_comparison stats.
    return {
        "name": name,
        "tiers": out,
    }


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    ents = EMB_DIR / "graphsage_256_entities.npy"
    embs = EMB_DIR / "graphsage_256_embeddings.npy"
    if not ents.exists() or not embs.exists():
        print(f"ERROR: missing GraphSAGE files at {ents} / {embs}.")
        print("Run scripts/h922_v2_graphsage_retrain.py on GPU first.")
        sys.exit(1)

    # Snapshot the existing h393 output (assumed Node2Vec baseline)
    backup = ANALYSIS_DIR / "h393_holdout_validation.node2vec_snapshot.json"
    if H393_OUT.exists():
        shutil.copy(H393_OUT, backup)

    # 1. Run h393 on Node2Vec (baseline, for side-by-side freshness)
    print("Running h393 on Node2Vec baseline ...")
    t0 = time.time()
    n2v = run_h393(
        "node2vec_256", ANALYSIS_DIR / "h922_v2_h393_node2vec.txt"
    )
    print(f"  Node2Vec h393 done ({time.time()-t0:.0f}s)")

    # 2. Run h393 on GraphSAGE
    print("Running h393 on GraphSAGE ...")
    t0 = time.time()
    gsg = run_h393(
        "graphsage_256", ANALYSIS_DIR / "h922_v2_h393_graphsage.txt"
    )
    print(f"  GraphSAGE h393 done ({time.time()-t0:.0f}s)")

    # 3. Compare
    n2v_s = summarize("node2vec", n2v)
    gsg_s = summarize("graphsage", gsg)

    print("\n" + "=" * 70)
    print("  h922-v2: GraphSAGE vs Node2Vec — 5-seed holdout")
    print("=" * 70)
    print(f"  {'Tier':<8} {'Node2Vec':>16}  {'GraphSAGE':>16}  {'Δpp':>8}")
    all_tiers_ok = True
    for t in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]:
        a = n2v_s["tiers"][t]
        b = gsg_s["tiers"][t]
        delta = b["holdout_mean"] - a["holdout_mean"]
        if delta < -1.0:
            all_tiers_ok = False
        marker = "OK" if delta >= -0.5 else ("DROP" if delta < -1.0 else "small-drop")
        print(
            f"  {t:<8} {a['holdout_mean']:>6.1f}%±{a['holdout_std']:>4.1f}  "
            f"{b['holdout_mean']:>6.1f}%±{b['holdout_std']:>4.1f}  "
            f"{delta:>+6.1f}  {marker}"
        )

    # Decision: GOLDEN/HIGH must not regress >1pp; MEDIUM R@30 proxy should rise
    golden_delta = gsg_s["tiers"]["GOLDEN"]["holdout_mean"] - n2v_s["tiers"]["GOLDEN"]["holdout_mean"]
    medium_delta = gsg_s["tiers"]["MEDIUM"]["holdout_mean"] - n2v_s["tiers"]["MEDIUM"]["holdout_mean"]
    if golden_delta >= -0.5 and medium_delta >= 2.0:
        decision = "PRODUCTIONIZE"
    elif golden_delta >= -1.0 and medium_delta >= 1.0:
        decision = "PROMISING_NEEDS_TUNING"
    elif all_tiers_ok:
        decision = "NEUTRAL"
    else:
        decision = "REGRESSION"
    print(f"\n  Decision: {decision}")

    # 4. Restore Node2Vec snapshot so default tooling is unaffected
    if backup.exists():
        shutil.copy(backup, H393_OUT)

    out = {
        "node2vec": n2v_s["tiers"],
        "graphsage": gsg_s["tiers"],
        "tier_deltas_pp": {
            t: gsg_s["tiers"][t]["holdout_mean"] - n2v_s["tiers"][t]["holdout_mean"]
            for t in ["GOLDEN", "HIGH", "MEDIUM", "LOW", "FILTER"]
        },
        "decision": decision,
    }
    with open(ANALYSIS_DIR / "h922_v2_vs_node2vec.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved to data/analysis/h922_v2_vs_node2vec.json")


if __name__ == "__main__":
    main()
