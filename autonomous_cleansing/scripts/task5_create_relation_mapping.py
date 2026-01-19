#!/usr/bin/env python3
"""
Task 5: Create Relation Mapping Table

Creates a mapping from 153+ relation types to ~30-40 standardized relations.
Removes source prefixes and groups semantically equivalent relations.

The mapping is saved as relation_mappings.json for use in Task 6.
"""

import json
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
REF_DIR = PROJECT_ROOT / "data" / "reference"
MAPPING_DIR = DATA_DIR / "entity_mappings"

# Relation mapping: original → standardized
# Organized by semantic category
RELATION_MAPPING = {
    # ============================================
    # GENE-GENE INTERACTIONS
    # ============================================
    # Physical interactions
    "protein_protein": "INTERACTS_PROTEIN",
    "Hetionet::GiG::Gene:Gene": "INTERACTS_PROTEIN",
    "interacts": "INTERACTS_PROTEIN",
    "INTACT::PHYSICAL ASSOCIATION::Gene:Gene": "INTERACTS_PROTEIN",
    "INTACT::ASSOCIATION::Gene:Gene": "INTERACTS_PROTEIN",
    "INTACT::DIRECT INTERACTION::Gene:Gene": "INTERACTS_PROTEIN",
    "bioarx::HumGenHumGen:Gene:Gene": "INTERACTS_PROTEIN",
    "GNBR::Q::Gene:Gene": "INTERACTS_PROTEIN",
    "GNBR::H::Gene:Gene": "INTERACTS_PROTEIN",
    "STRING::BINDING::Gene:Gene": "INTERACTS_PROTEIN",
    "STRING::OTHER::Gene:Gene": "INTERACTS_PROTEIN",
    "INTACT::COLOCALIZATION::Gene:Gene": "COLOCALIZED_WITH",

    # Regulatory interactions
    "Hetionet::Gr>G::Gene:Gene": "REGULATES_GENE",
    "regulates": "REGULATES_GENE",
    "Hetionet::GcG::Gene:Gene": "COVARIES_WITH",
    "covaries": "COVARIES_WITH",
    "GNBR::Rg::Gene:Gene": "REGULATES_GENE",
    "GNBR::E+::Gene:Gene": "INCREASES_EXPRESSION",
    "GNBR::E-::Gene:Gene": "DECREASES_EXPRESSION",
    "GNBR::E::Gene:Gene": "AFFECTS_EXPRESSION",
    "STRING::ACTIVATION::Gene:Gene": "ACTIVATES_GENE",
    "STRING::INHIBITION::Gene:Gene": "INHIBITS_GENE",
    "STRING::EXPRESSION::Gene:Gene": "AFFECTS_EXPRESSION",

    # Biochemical reactions
    "STRING::REACTION::Gene:Gene": "REACTS_WITH",
    "STRING::CATALYSIS::Gene:Gene": "CATALYZES",
    "STRING::PTMOD::Gene:Gene": "POST_TRANS_MODIFIES",
    "INTACT::PHOSPHORYLATION REACTION::Gene:Gene": "PHOSPHORYLATES",
    "INTACT::DEPHOSPHORYLATION REACTION::Gene:Gene": "DEPHOSPHORYLATES",
    "INTACT::UBIQUITINATION REACTION::Gene:Gene": "UBIQUITINATES",
    "INTACT::CLEAVAGE REACTION::Gene:Gene": "CLEAVES",
    "INTACT::PROTEIN CLEAVAGE::Gene:Gene": "CLEAVES",
    "INTACT::ADP RIBOSYLATION REACTION::Gene:Gene": "ADP_RIBOSYLATES",

    # Other gene relationships
    "GNBR::B::Gene:Gene": "BINDS_GENE",
    "GNBR::I::Gene:Gene": "INHIBITS_GENE",
    "GNBR::V+::Gene:Gene": "AFFECTS_ACTIVITY",
    "GNBR::W::Gene:Gene": "ENHANCES_GENE",
    "bioarx::VirGenHumGen:Gene:Gene": "INTERACTS_VIRAL",

    # ============================================
    # DRUG-DRUG INTERACTIONS
    # ============================================
    "drug_drug": "INTERACTS_DRUG",
    "DRUGBANK::ddi-interactor-in::Compound:Compound": "INTERACTS_DRUG",
    "Hetionet::CrC::Compound:Compound": "RESEMBLES_DRUG",
    "resembles": "RESEMBLES_DRUG",

    # ============================================
    # DRUG-GENE/PROTEIN INTERACTIONS
    # ============================================
    "drug_protein": "TARGETS",
    "DRUGBANK::target::Compound:Gene": "TARGETS",
    "Hetionet::CdG::Compound:Gene": "DOWNREGULATES_GENE",
    "Hetionet::CuG::Compound:Gene": "UPREGULATES_GENE",
    "Hetionet::CbG::Compound:Gene": "BINDS_GENE",
    "binds": "BINDS_GENE",
    "DRUGBANK::enzyme::Compound:Gene": "METABOLIZED_BY",
    "DRUGBANK::carrier::Compound:Gene": "CARRIED_BY",
    "bioarx::DrugHumGen:Compound:Gene": "AFFECTS_GENE",
    "bioarx::DrugVirGen:Compound:Gene": "TARGETS_VIRAL",

    # GNBR Drug-Gene relations
    "GNBR::E::Compound:Gene": "AFFECTS_GENE_EXPRESSION",
    "GNBR::E+::Compound:Gene": "INCREASES_EXPRESSION",
    "GNBR::E-::Compound:Gene": "DECREASES_EXPRESSION",
    "GNBR::N::Compound:Gene": "INHIBITS_GENE",
    "GNBR::B::Compound:Gene": "BINDS_GENE",
    "GNBR::K::Compound:Gene": "METABOLISM_RELATED",
    "GNBR::O::Compound:Gene": "TRANSPORT_RELATED",
    "GNBR::Z::Compound:Gene": "AFFECTS_PROTEIN",
    "GNBR::A+::Compound:Gene": "ACTIVATES_GENE",
    "GNBR::A-::Compound:Gene": "INACTIVATES_GENE",

    # DGIDB Drug-Gene relations
    "DGIDB::OTHER::Gene:Compound": "INTERACTS_GENE",
    "DGIDB::INHIBITOR::Gene:Compound": "INHIBITS_GENE",
    "DGIDB::AGONIST::Gene:Compound": "AGONIST_OF",
    "DGIDB::ANTAGONIST::Gene:Compound": "ANTAGONIST_OF",
    "DGIDB::BLOCKER::Gene:Compound": "BLOCKS",
    "DGIDB::CHANNEL BLOCKER::Gene:Compound": "BLOCKS_CHANNEL",
    "DGIDB::POSITIVE ALLOSTERIC MODULATOR::Gene:Compound": "ALLOSTERIC_MODULATOR",
    "DGIDB::ALLOSTERIC MODULATOR::Gene:Compound": "ALLOSTERIC_MODULATOR",
    "DGIDB::ACTIVATOR::Gene:Compound": "ACTIVATES_GENE",
    "DGIDB::MODULATOR::Gene:Compound": "MODULATES",
    "DGIDB::ANTIBODY::Gene:Compound": "ANTIBODY_TARGET",
    "DGIDB::BINDER::Gene:Compound": "BINDS_GENE",
    "DGIDB::PARTIAL AGONIST::Gene:Compound": "PARTIAL_AGONIST",

    # INTACT Drug-Gene
    "INTACT::ASSOCIATION::Compound:Gene": "ASSOCIATES_WITH",
    "INTACT::PHYSICAL ASSOCIATION::Compound:Gene": "BINDS_GENE",
    "INTACT::DIRECT INTERACTION::Compound:Gene": "INTERACTS_GENE",

    # ============================================
    # DRUG-DISEASE INTERACTIONS
    # ============================================
    "indication": "INDICATED_FOR",
    "DRUGBANK::treats::Compound:Disease": "INDICATED_FOR",
    "Hetionet::CtD::Compound:Disease": "INDICATED_FOR",
    "treats": "INDICATED_FOR",
    "contraindication": "CONTRAINDICATED_FOR",
    "Hetionet::CpD::Compound:Disease": "PALLIATES",
    "palliates": "PALLIATES",
    "off-label use": "OFF_LABEL_USE",

    # GNBR Drug-Disease
    "GNBR::T::Compound:Disease": "TREATS",
    "GNBR::Sa::Compound:Disease": "SIDE_EFFECT_OF",
    "GNBR::C::Compound:Disease": "CAUSES_DISEASE",
    "GNBR::Pa::Compound:Disease": "ALLEVIATES",
    "GNBR::J::Compound:Disease": "ROLE_IN_DISEASE",
    "GNBR::Pr::Compound:Disease": "PREVENTS",
    "GNBR::Mp::Compound:Disease": "BIOMARKER_FOR",

    # ============================================
    # DRUG-SIDE EFFECT INTERACTIONS
    # ============================================
    "drug_effect": "CAUSES_EFFECT",
    "Hetionet::CcSE::Compound:Side Effect": "CAUSES_SIDE_EFFECT",
    "causes": "CAUSES_SIDE_EFFECT",

    # ============================================
    # GENE-DISEASE ASSOCIATIONS
    # ============================================
    "disease_protein": "ASSOCIATED_GENE",
    "Hetionet::DaG::Disease:Gene": "ASSOCIATED_GENE",
    "associates": "ASSOCIATED_GENE",
    "Hetionet::DuG::Disease:Gene": "UPREGULATED_IN",
    "Hetionet::DdG::Disease:Gene": "DOWNREGULATED_IN",
    "bioarx::Covid2_acc_host_gene::Disease:Gene": "ASSOCIATED_GENE",
    "bioarx::Coronavirus_ass_host_gene::Disease:Gene": "ASSOCIATED_GENE",

    # GNBR Gene-Disease
    "GNBR::L::Gene:Disease": "GENETIC_VARIANT",
    "GNBR::J::Gene:Disease": "ROLE_IN_DISEASE",
    "GNBR::U::Gene:Disease": "DRUG_TARGET_DISEASE",
    "GNBR::Te::Gene:Disease": "THERAPEUTIC_TARGET",
    "GNBR::G::Gene:Disease": "MARKER_IN",
    "GNBR::Y::Gene:Disease": "PATHOGENESIS",
    "GNBR::X::Gene:Disease": "OVEREXPRESSED_IN",
    "GNBR::Md::Gene:Disease": "BIOMARKER_FOR",
    "GNBR::Ud::Gene:Disease": "UNDEREXPRESSED_IN",
    "GNBR::D::Gene:Disease": "DRUG_TARGET_CANDIDATE",

    # ============================================
    # GENE-BIOLOGICAL PROCESS
    # ============================================
    "bioprocess_protein": "PARTICIPATES_IN_BP",
    "participates": "PARTICIPATES_IN_BP",
    "Hetionet::GpBP::Gene:Biological Process": "PARTICIPATES_IN_BP",

    # ============================================
    # GENE-MOLECULAR FUNCTION
    # ============================================
    "molfunc_protein": "HAS_FUNCTION",
    "Hetionet::GpMF::Gene:Molecular Function": "HAS_FUNCTION",

    # ============================================
    # GENE-CELLULAR COMPONENT
    # ============================================
    "cellcomp_protein": "LOCATED_IN",
    "Hetionet::GpCC::Gene:Cellular Component": "LOCATED_IN",

    # ============================================
    # GENE-PATHWAY
    # ============================================
    "pathway_protein": "IN_PATHWAY",
    "Hetionet::GpPW::Gene:Pathway": "IN_PATHWAY",

    # ============================================
    # GENE-ANATOMY (Expression)
    # ============================================
    "anatomy_protein_present": "EXPRESSED_IN",
    "Hetionet::AeG::Anatomy:Gene": "EXPRESSED_IN",
    "expresses": "EXPRESSED_IN",
    "Hetionet::AdG::Anatomy:Gene": "DOWNREGULATED_IN",
    "Hetionet::AuG::Anatomy:Gene": "UPREGULATED_IN",
    "anatomy_protein_absent": "NOT_EXPRESSED_IN",

    # ============================================
    # GENE-TAXONOMY
    # ============================================
    "GNBR::in_tax::Gene:Tax": "IN_TAXONOMY",

    # ============================================
    # DISEASE-DISEASE
    # ============================================
    "disease_disease": "RELATED_DISEASE",
    "Hetionet::DrD::Disease:Disease": "RESEMBLES_DISEASE",

    # ============================================
    # DISEASE-ANATOMY
    # ============================================
    "Hetionet::DlA::Disease:Anatomy": "LOCALIZES_TO",
    "localizes": "LOCALIZES_TO",

    # ============================================
    # DISEASE-SYMPTOM
    # ============================================
    "Hetionet::DpS::Disease:Symptom": "PRESENTS_SYMPTOM",
    "presents": "PRESENTS_SYMPTOM",

    # ============================================
    # DISEASE-PHENOTYPE
    # ============================================
    "disease_phenotype_positive": "HAS_PHENOTYPE",
    "disease_phenotype_negative": "LACKS_PHENOTYPE",
    "phenotype_protein": "ASSOCIATED_WITH",

    # ============================================
    # ONTOLOGY RELATIONSHIPS
    # ============================================
    "bioprocess_bioprocess": "RELATED_PROCESS",
    "molfunc_molfunc": "RELATED_FUNCTION",
    "cellcomp_cellcomp": "RELATED_COMPONENT",
    "pathway_pathway": "RELATED_PATHWAY",
    "anatomy_anatomy": "RELATED_ANATOMY",
    "phenotype_phenotype": "RELATED_PHENOTYPE",

    # ============================================
    # EXPOSURE RELATIONS
    # ============================================
    "exposure_disease": "EXPOSURE_DISEASE",
    "exposure_exposure": "RELATED_EXPOSURE",
    "exposure_bioprocess": "EXPOSURE_PROCESS",
    "exposure_protein": "EXPOSURE_PROTEIN",
    "exposure_molfunc": "EXPOSURE_FUNCTION",
    "exposure_cellcomp": "EXPOSURE_COMPONENT",

    # ============================================
    # REGULATORY/EXPRESSION
    # ============================================
    "upregulates": "UPREGULATES",
    "downregulates": "DOWNREGULATES",

    # ============================================
    # DRUG CLASS
    # ============================================
    "DRUGBANK::x-atc::Compound:Atc": "HAS_ATC_CODE",
    "Hetionet::PCiC::Pharmacologic Class:Compound": "IN_PHARM_CLASS",
    "includes": "IN_PHARM_CLASS",
}


def main():
    print("=" * 60)
    print("Task 5: Create Relation Mapping Table")
    print("=" * 60)

    # Create output directory
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    # Save mapping
    output_file = MAPPING_DIR / "relation_mappings.json"

    # Count statistics
    unique_targets = set(RELATION_MAPPING.values())
    print(f"\nMapping statistics:")
    print(f"  Original relation types: {len(RELATION_MAPPING)}")
    print(f"  Standardized relation types: {len(unique_targets)}")

    # Group by target relation
    target_groups = {}
    for orig, target in RELATION_MAPPING.items():
        if target not in target_groups:
            target_groups[target] = []
        target_groups[target].append(orig)

    print(f"\n--- Standardized Relations ---")
    for target in sorted(unique_targets):
        sources = target_groups[target]
        print(f"\n  {target}:")
        for src in sources[:3]:
            print(f"    ← {src}")
        if len(sources) > 3:
            print(f"    ... and {len(sources) - 3} more")

    # Save as JSON with metadata
    mapping_data = {
        "version": "1.0",
        "created": datetime.now().isoformat(),
        "description": "Maps 153 original relation types to ~46 standardized relations",
        "statistics": {
            "original_relations": len(RELATION_MAPPING),
            "standardized_relations": len(unique_targets)
        },
        "mapping": RELATION_MAPPING,
        "target_relations": sorted(unique_targets)
    }

    with open(output_file, 'w') as f:
        json.dump(mapping_data, f, indent=2)

    print(f"\n✓ Saved to {output_file}")

    # Check for any relations not in mapping
    # (We'll verify this when we apply the mapping in Task 6)

    print("\n" + "=" * 60)
    print("Task 5 COMPLETE")
    print("=" * 60)

    return True


if __name__ == "__main__":
    main()
