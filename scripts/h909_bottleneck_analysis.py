"""h909: Identify real bottleneck — missing DRKG embeddings vs missing GT drugs.

For each of the 1,555 MeSH disease mappings produced by h901, build a 2x2 table:
  - embedding present (MeSH ID has DRKG node with TransE vector) — yes/no
  - GT drugs present (MeSH ID keys an entry in expanded_ground_truth.json) — yes/no

Decide whether the ~569 non-evaluable mappings are blocked by missing embeddings
(LINCS/h905 fix) or missing GT (DrugBank/h906 fix).
"""

import json
from collections import Counter
from pathlib import Path


def load_mesh_mappings() -> dict[str, str]:
    """Return a flat disease_name -> MeSH-D-id mapping (1,555 entries after h901).

    Skips batch-level metadata fields (source, date) and any value that is
    not a string starting with 'D' followed by digits (MeSH D-ids).
    """
    with open("data/reference/mesh_mappings_from_agents.json") as f:
        batches = json.load(f)
    flat: dict[str, str] = {}
    for batch in batches.values():
        if not isinstance(batch, dict):
            continue
        for name, mesh in batch.items():
            if not isinstance(mesh, str):
                continue
            if not (mesh.startswith("D") and any(c.isdigit() for c in mesh)):
                continue
            flat[name] = mesh
    return flat


def load_drkg_disease_mesh_ids() -> set[str]:
    """Return set of MeSH IDs (bare 'D003920' form) present as Disease nodes in DRKG entity list."""
    mesh_ids = set()
    with open("data/raw/drkg/embed/entities.tsv") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if not parts:
                continue
            ent = parts[0]
            if ent.startswith("Disease::MESH:"):
                mesh_id = ent.split("::MESH:", 1)[1]
                mesh_ids.add(mesh_id)
    return mesh_ids


def load_gt_disease_mesh_ids() -> set[str]:
    """Return set of MeSH IDs that have >=1 drug pair in expanded_ground_truth.json."""
    with open("data/reference/expanded_ground_truth.json") as f:
        egt = json.load(f)
    mesh_ids = set()
    for key, drugs in egt.items():
        # keys look like 'drkg:Disease::MESH:D003550'
        if "::MESH:" in key and drugs:
            mesh_id = key.split("::MESH:", 1)[1]
            mesh_ids.add(mesh_id)
    return mesh_ids


def main() -> None:
    mappings = load_mesh_mappings()
    drkg_emb = load_drkg_disease_mesh_ids()
    gt_ids = load_gt_disease_mesh_ids()

    print(f"MeSH mappings (h901 total):            {len(mappings):>6}")
    print(f"Unique MeSH IDs in mappings:           {len(set(mappings.values())):>6}")
    print(f"Disease MeSH IDs with DRKG embedding:  {len(drkg_emb):>6}")
    print(f"MeSH IDs with expanded-GT drugs:       {len(gt_ids):>6}")
    print()

    # Build 2x2 at the (disease-name, mesh-id) pair granularity because
    # h902 reports "evaluable DISEASES" which is per disease-name.
    cells = Counter()
    non_evaluable_examples: dict[str, list[tuple[str, str]]] = {
        "no_embed_no_gt": [],
        "embed_only": [],
        "gt_only": [],
        "both": [],
    }
    for name, mesh in mappings.items():
        has_emb = mesh in drkg_emb
        has_gt = mesh in gt_ids
        if has_emb and has_gt:
            cell = "both"
        elif has_emb and not has_gt:
            cell = "embed_only"
        elif not has_emb and has_gt:
            cell = "gt_only"
        else:
            cell = "no_embed_no_gt"
        cells[cell] += 1
        if len(non_evaluable_examples[cell]) < 10:
            non_evaluable_examples[cell].append((name, mesh))

    total = sum(cells.values())
    print("2x2 coverage table (disease-name granularity, N=1,555):")
    print(f"{'':<20} {'GT yes':>10} {'GT no':>10} {'row total':>12}")
    row_emb_yes = cells['both'] + cells['embed_only']
    row_emb_no = cells['gt_only'] + cells['no_embed_no_gt']
    col_gt_yes = cells['both'] + cells['gt_only']
    col_gt_no = cells['embed_only'] + cells['no_embed_no_gt']
    print(f"{'embed yes':<20} {cells['both']:>10} {cells['embed_only']:>10} {row_emb_yes:>12}")
    print(f"{'embed no':<20} {cells['gt_only']:>10} {cells['no_embed_no_gt']:>10} {row_emb_no:>12}")
    print(f"{'col total':<20} {col_gt_yes:>10} {col_gt_no:>10} {total:>12}")
    print()

    print("Percentages of 1,555 mappings:")
    for cell, label in [
        ("both", "EVALUABLE (embed + GT)"),
        ("embed_only", "embedding only (no GT drugs)"),
        ("gt_only", "GT only (no DRKG embedding)"),
        ("no_embed_no_gt", "neither"),
    ]:
        pct = 100.0 * cells[cell] / total if total else 0.0
        print(f"  {label:<35} {cells[cell]:>6} ({pct:5.1f}%)")
    print()

    print("Sample non-evaluable disease names per cell (first 10):")
    for cell in ("embed_only", "gt_only", "no_embed_no_gt"):
        print(f"  [{cell}]")
        for name, mesh in non_evaluable_examples[cell]:
            print(f"    {name:<45} -> {mesh}")

    # Derive the h902 'non-evaluable' class: anything NOT in "both"
    non_evaluable_total = total - cells['both']
    print(f"\nNon-evaluable total: {non_evaluable_total}  (h902 reported ~569 missing from '+78% coverage' target)")
    print()

    # ROI recommendation: what fraction of non-evaluable could each external source unblock?
    # LINCS/h905 would supply an embedding for diseases in {no_embed_no_gt} ∪ {gt_only}
    # DrugBank/h906 would supply GT drugs for diseases in {embed_only} ∪ {no_embed_no_gt}
    lincs_unblockable = cells['gt_only']           # has GT already -> new embed makes it evaluable
    drugbank_unblockable = cells['embed_only']     # has embed already -> new GT drugs makes it evaluable
    both_needed = cells['no_embed_no_gt']          # both gaps -> need both external sources, or skip
    print("ROI under optimistic assumptions (1 external source = 1 missing axis filled):")
    print(f"  LINCS/h905 alone could unblock (gt_only cell):       {lincs_unblockable}  ({100*lincs_unblockable/total:.1f}%)")
    print(f"  DrugBank/h906 alone could unblock (embed_only cell): {drugbank_unblockable}  ({100*drugbank_unblockable/total:.1f}%)")
    print(f"  Blocked on both axes (no_embed_no_gt):               {both_needed}  ({100*both_needed/total:.1f}%)")

    # --- Second pass: deeper gap analysis ---------------------------------
    # The h902 gap is not explained by 'missing embedding or missing GT' alone
    # (those total only ~91 diseases here). So count the 1,464 'both' cell by
    # GT-drug depth (GT drug in DRKG compound pool or not) — that is what the
    # evaluator actually requires.
    drkg_compounds: set[str] = set()
    with open("data/raw/drkg/embed/entities.tsv") as f:
        for line in f:
            ent = line.split("\t", 1)[0]
            if ent.startswith("Compound::"):
                drkg_compounds.add(ent)  # e.g. 'Compound::DB00001' or 'Compound::MESH:C065382'
    # Translate 'drkg:Compound::DB00001' style keys used in expanded GT to DRKG-native form
    def to_drkg_compound(key: str) -> str:
        return key.split("drkg:", 1)[-1]

    with open("data/reference/expanded_ground_truth.json") as f:
        egt = json.load(f)

    drkg_drug_per_mesh: dict[str, int] = {}
    for key, drugs in egt.items():
        if "::MESH:" not in key:
            continue
        mesh = key.split("::MESH:", 1)[1]
        if not drugs:
            continue
        n = sum(1 for d in drugs if to_drkg_compound(d) in drkg_compounds)
        drkg_drug_per_mesh[mesh] = n

    both_with_drkg_drug = 0
    both_without_drkg_drug = 0
    for name, mesh in mappings.items():
        if mesh in drkg_emb and mesh in gt_ids:
            n = drkg_drug_per_mesh.get(mesh, 0)
            if n >= 1:
                both_with_drkg_drug += 1
            else:
                both_without_drkg_drug += 1

    print("\nDeeper view inside the 'both' cell (has embed + has GT):")
    print(f"  >=1 GT drug is in DRKG (evaluable by kNN):      {both_with_drkg_drug}")
    print(f"  all GT drugs lie outside DRKG embedding pool:   {both_without_drkg_drug}")
    print("  -> the second group would need DrugBank/LINCS-drug expansion (h906-adjacent).")

    # Save JSON result
    out = {
        "summary": {
            "total_mappings": total,
            "evaluable_both": cells['both'],
            "embed_only_no_gt": cells['embed_only'],
            "gt_only_no_embed": cells['gt_only'],
            "no_embed_no_gt": cells['no_embed_no_gt'],
        },
        "roi": {
            "lincs_h905_unblockable_diseases": lincs_unblockable,
            "drugbank_h906_unblockable_diseases": drugbank_unblockable,
            "both_needed": both_needed,
        },
        "deeper_both_cell": {
            "at_least_one_gt_drug_in_drkg": both_with_drkg_drug,
            "all_gt_drugs_outside_drkg": both_without_drkg_drug,
        },
        "examples": non_evaluable_examples,
    }
    Path("data/analysis").mkdir(exist_ok=True, parents=True)
    with open("data/analysis/h909_bottleneck_2x2.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: data/analysis/h909_bottleneck_2x2.json")


if __name__ == "__main__":
    main()
