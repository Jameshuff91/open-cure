# Practical Application: Model vs LLM, World vs Personal

Two complementary use cases for Open-Cure. They are not alternatives — they are different tools for different jobs. Knowing which one to reach for matters.

## The two modes

### Mode A — The trained model (world-facing, scale)

**Shape:** Run `predict()` against the current production pipeline over all ~1,070 evaluable diseases × ~24k drugs. Output is the 31,926-row deliverable (post-h952-fix, post-h962 regeneration) with calibrated confidence tiers.

**When to reach for it:**
- Bulk screening across many diseases.
- Research output you want to cite, publish, or ship to Every Cure.
- Any claim that depends on **calibrated probability**: GOLDEN 82.6% holdout, HIGH 80.4%, MEDIUM 39.9%, LOW 10.0%, FILTER 6.8%. The tier label means the same thing for Huntington's as it does for UTI.
- Reproducibility and audit: output is deterministic given code + data.
- **Novel-signal recovery that literature misses** (Dantrolene → HF is the canonical case — the DRKG-path signal was there before the confirmatory RCT).

**Strengths:**
- Cost: ~$0.0001 per drug-disease pair, sub-second per disease.
- Uniform methodology across 35M+ potential pairs.
- Edge-grounded: tied to an identifiable reasoning chain in the graph.
- Calibrated confidence is a real methodological contribution an LLM case-by-case cannot replicate.

**Limits:**
- Coverage is gated by DRKG structural support (~95% of MeSH diseases covered post-h961 aliasing; the 71 absent-from-DRKG rare diseases are unreachable without external signal).
- Ceiling around 37% per-drug R@30 for kNN-on-DRKG, ~60% for best-possible model on DRKG.
- Tier rules require periodic recalibration (see `confidence_system_history.md`).

---

### Mode B — LLM + retrieval (personal, one-off, same-day)

**Shape:** Give a capable LLM (Claude, GPT) access to PubMed, DrugBank, DisGeNET, ClinicalTrials.gov, OMIM, and `drug_repurposing_predictions_with_confidence.xlsx`. Ask it to produce the top-30 candidate drugs for one specific disease, with reasoning for each.

**When to reach for it:**
- Someone you love gets sick. You need the best available candidate list **today**.
- A rare disease absent from DRKG (LLM can reason from OMIM genetics + literature even when the graph doesn't encode the disease).
- A condition where the literature is rich enough that expert-style reading captures most of the signal (anything well-studied, especially common diseases).
- Cross-checking a Mode-A prediction before acting on it.

**Strengths:**
- Can read the latest literature, including papers published after the model's training data.
- Reasons over mechanisms, not just graph edges — captures textual specificity (especially antibody target binding) that the DRKG embedding can't express.
- Per-query ~60% R@30 for well-studied diseases, often higher than the model on that specific disease.
- Cost per query: ~$0.10–$1.

**Limits:**
- Not calibrated. "Confidence: high" from an LLM is not comparable across queries.
- Stochastic and prompt-dependent — two runs of the same query may differ.
- Literature-anchored, so **systematically biased against novel repurposing** — if nobody has written about the indication yet, the LLM probably won't surface it. This is the exact failure mode Mode A avoids with Dantrolene → HF.
- Does not scale: $3.5M–$35M to run over the full 35M-pair space at LLM pricing.

---

## Workflow by scenario

### Publishing / research / Every Cure delivery
**Mode A.** Use the model. The whole argument of the bioRxiv preprint is calibrated, auditable, scale-class output. If you're writing the Nature paper, compare against an LLM-per-disease baseline and expect to match or slightly underperform on recall while winning on calibration, reproducibility, and novel-signal recovery.

### "My mom has X — what should we look at?"
**Mode B first, then Mode A as a cross-check.**

1. Ask the LLM (with retrieval) for top-30 candidates for disease X. ~5 min, ~$0.50.
2. Look up X in the shipped deliverable. Find the GOLDEN and HIGH predictions the model produced.
3. Compare. High-agreement candidates are the strongest leads. Model-only candidates are novel-signal hypotheses worth investigating. LLM-only candidates are usually literature-anchored indications the model missed.
4. For the shortlist (≤5 drugs), ask the LLM to pull: recent trial data, safety profile, off-label precedent, interaction profile with current medications.
5. Send the shortlist + reasoning to the relevant specialist for their take before acting.

**Critical:** neither mode is a substitute for clinical judgment. Both produce *hypotheses* that still require a physician to evaluate against the specific patient's history, comorbidities, and risk profile.

### "Is this rare disease reachable at all?"
- If it's in `data/reference/disease_ontology_mapping.json` or resolves via `data/reference/h961_disease_name_aliases.json`: Mode A has coverage. Check the deliverable.
- If not: Mode A will have 0 predictions for it. Use Mode B with literature + OMIM genetics.
- If the disease is in DRKG but has zero kNN neighbours: Mode A returns nothing useful (h903 invalidated the mechanism-only fallback). Use Mode B.

### "I want to find something genuinely novel"
**Mode A.** Literature-anchored reasoning won't surface anything that isn't already in a paper. The model's GOLDEN-tier novel predictions (with `is_known_indication=False`) are where repurposing signal lives that no expert has written up yet. These are also where Ryland's blinded review adds the most value — expert evaluation of model-novel candidates is how you confirm whether the graph-signal is real (the Dantrolene → HF path).

---

## Design principle

The two modes are not competitive; they're complementary. Mode A says "across 35M pairs, here are the 13,000 I'm confident about, uniformly calibrated." Mode B says "for this one disease, here's the best I can assemble right now with everything the world knows."

For the world: Mode A is the artifact.
For one person: Mode B is the triage tool, with Mode A as the reproducible check.

---

## Current state (2026-04-19)

- **Mode A deliverable**: `data/deliverables/drug_repurposing_predictions_with_confidence.xlsx` — 31,926 rows, 1,070 diseases, 1,328 drugs, tier precisions GOLDEN 82.6% / HIGH 80.4% / MEDIUM 39.9% / LOW 10.0% / FILTER 6.8% (post-h986).
- **Mode B readiness**: No dedicated tooling yet. Natural next step: a small `scripts/query_disease.py` that prints (a) the deliverable's rows for a disease and (b) a ready-to-paste LLM prompt template with retrieval hooks for PubMed / DrugBank / ClinicalTrials.gov.
- **Audit trail**: every prediction has `disease_id`, `drug_id`, `rank`, `knn_score`, `normalized_score`, `confidence_tier`, `tier_rule`, `mechanism_support`, `gene_overlap_count`, `literature_evidence_level`, `literature_evidence_score`. Enough to reconstruct the reasoning behind any single prediction.
- **Known failure mode reminder (h951/h952)**: always rule out infrastructure bugs before declaring a ceiling physical. A 4pp recall gap looked like a biologic modeling problem for months; it turned out to be a name-resolution bug in `find_disease_id`.

## Open questions

- Does an LLM-retrieval baseline beat Mode A on calibrated novel-signal recovery? Probably not — this is the Nature-paper claim to defend.
- Should Mode B be wrapped into a CLI (`query_disease.py`) or left as a prompt template? Depends on frequency of personal use. If used more than monthly, script it.
- Hybrid: can LLM outputs be fed back into the model's calibration layer? In principle yes (treat LLM agreement as a feature for the calibrator). h927 MLP calibrator is a place to try this after it lands.
