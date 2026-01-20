#!/usr/bin/env python3
"""Update research state with new Atrial Fibrillation drug classifications."""

import json
from datetime import datetime
from pathlib import Path

# New classifications for Atrial Fibrillation drugs
new_classifications = [
    {
        "drug_id": "drkg:Compound::DB11852",
        "name": "Tegobuvir",
        "classification": "NOVEL",
        "evidence": "Tegobuvir - HCV NS5B polymerase inhibitor for hepatitis C. No atrial fibrillation indication. Development discontinued. Source: DrugBank, PMC3165336"
    },
    {
        "drug_id": "drkg:Compound::DB12567",
        "name": "Camicinal",
        "classification": "NOVEL",
        "evidence": "Camicinal (GSK962040) - Motilin receptor agonist investigated for gastroparesis. Development discontinued. No AF connection. Source: DrugBank, Wikipedia"
    },
    {
        "drug_id": "drkg:Compound::DB06868",
        "name": "N-(3-chlorobenzyl)-1-(4-methylpentanoyl)-L-prolinamide",
        "classification": "NOVEL",
        "evidence": "Experimental small molecule compound (DB06868). No clinical trials, no indicated conditions, no AF connection. Source: DrugBank"
    },
    {
        "drug_id": "drkg:Compound::DB09422",
        "name": "Soybean oil",
        "classification": "NOVEL",
        "evidence": "Soybean oil - Nutritional supplement and pharmaceutical excipient. Omega-3 studies show mixed/increased AF risk with supplementation. Not an AF treatment. Source: PMC10175873"
    },
    {
        "drug_id": "drkg:Compound::DB05345",
        "name": "SO-101",
        "classification": "NOVEL",
        "evidence": "SO-101 (SCO-101) - Efflux pump inhibitor for cancer drug resistance in colorectal and pancreatic cancer. Phase II trials for chemotherapy-resistant cancer. No cardiac/AF indication. Source: PMC7539971, Scandion Oncology"
    },
    {
        "drug_id": "drkg:Compound::DB08010",
        "name": "(3Z)-1-[(6-fluoro-4H-1,3-benzodioxin-8-yl)methyl]-4-[(E)-2-phenylethenyl]-1H-indole-2,3-dione 3-oxime",
        "classification": "NOVEL",
        "evidence": "Experimental benzodioxin-indole compound. 0 clinical trials, no indicated conditions, no AF connection. Source: DrugBank"
    },
    {
        "drug_id": "drkg:Compound::DB08836",
        "name": "Temocapril",
        "classification": "EXPERIMENTAL",
        "evidence": "Temocapril - ACE inhibitor approved in Japan/South Korea for hypertension and heart failure. ACE inhibitors as class reduce AF risk by 28% in meta-analysis, greatest benefit in HF (44% RRR). Not specifically studied for AF but class effect supports benefit. Source: DrugBank, ScienceDirect"
    },
    {
        "drug_id": "drkg:Compound::DB11732",
        "name": "Lasmiditan",
        "classification": "NOVEL",
        "evidence": "Lasmiditan - Selective 5-HT1F receptor agonist for acute migraine. FDA approved 2019. Cardiac arrhythmia reports (palpitations, tachycardia) in trials but no AF indication. No vasoconstriction, cardiovascularly safe. Source: PMC6734241, PMC6787764"
    }
]

def main():
    base_dir = Path(__file__).parent
    state_file = base_dir / ".research_state.json"

    # Load current state
    with open(state_file, 'r') as f:
        state = json.load(f)

    # Add new classifications
    timestamp = datetime.now().isoformat()

    for drug_data in new_classifications:
        drug_id = drug_data["drug_id"]

        # Add to candidates_researched
        state["candidates_researched"][drug_id] = {
            "classification": drug_data["classification"],
            "evidence": drug_data["evidence"],
            "timestamp": timestamp
        }

        # Add to research_log
        state["research_log"].append({
            "disease": "Atrial fibrillation",
            "drug": drug_data["name"],
            "classification": drug_data["classification"],
            "timestamp": timestamp
        })

    # Save updated state
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)

    print(f"✓ Updated research state with {len(new_classifications)} new AF drug classifications")

    # Count classifications by type
    af_drugs = [log for log in state["research_log"] if log["disease"] == "Atrial fibrillation"]
    confirmed = len([d for d in af_drugs if d["classification"] == "CONFIRMED"])
    experimental = len([d for d in af_drugs if d["classification"] == "EXPERIMENTAL"])
    novel = len([d for d in af_drugs if d["classification"] == "NOVEL"])

    print(f"  Total AF drugs researched: {len(af_drugs)}")
    print(f"  - CONFIRMED: {confirmed}")
    print(f"  - EXPERIMENTAL: {experimental}")
    print(f"  - NOVEL: {novel}")

    # Check consecutive novel count
    consecutive_novel = 0
    for log in reversed(af_drugs):
        if log["classification"] == "NOVEL":
            consecutive_novel += 1
        else:
            break

    print(f"  Consecutive NOVEL at end: {consecutive_novel}")
    if consecutive_novel >= 5:
        print(f"  ⚠️  Reached early stopping threshold (5+ consecutive NOVEL)")

if __name__ == "__main__":
    main()
