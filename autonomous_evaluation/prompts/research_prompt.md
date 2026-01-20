# Evaluation Enhancement Agent - Research Session

You are continuing the Open Cure Evaluation Enhancement research.

## Stopping Criteria

1. **Max 30 candidates per disease** (matches R@30 metric)
2. **Stop early if 5+ consecutive NOVEL** (diminishing returns)
3. **Auto-classify by name patterns first** (saves time)
4. **Require evidence for CONFIRMED** (quality over quantity)

## Current Status

Check `autonomous_evaluation/evaluation_tasks.json` for:
- Which diseases still need research
- How many drugs have been researched per disease
- confirmed_found and novel_found counts

Check `autonomous_evaluation/.research_state.json` for:
- Drugs already classified
- Current research progress

## Your Task This Session

1. Load the current state
2. Find the next disease with incomplete research
3. For that disease, research the next 5-10 unclassified drugs
4. **AUTO-CLASSIFY FIRST** using name patterns:
   - Antiviral: vir, avir, navir, buvir, previr, gravir
   - Nucleoside: abine, udine, osine, deoxy, dideoxy
   - If name matches pattern for the disease type → likely CONFIRMED
5. **ONLY web search for ambiguous cases**
6. For each drug classify as CONFIRMED, EXPERIMENTAL, or NOVEL
7. **Check early stopping**: if 5+ consecutive NOVEL, mark disease complete
8. Save results and update all state files

## Research Workflow

For each drug candidate:

```
1. Search: "Amdoxovir HIV treatment mechanism of action"
2. Evaluate results:
   - Is there evidence it treats/treated the disease?
   - Was it in clinical trials?
   - Is it FDA approved or was it ever?
3. Classify:
   - CONFIRMED if: FDA approved, was approved but discontinued, Phase III+ trials
   - EXPERIMENTAL if: Phase I-II trials, preclinical with strong rationale
   - NOVEL if: No evidence of treating this specific disease
4. Record with evidence
```

## Quality Standards

- Only mark CONFIRMED if there's clear evidence
- Include source (NIH, PubMed ID, FDA, Wikipedia reference)
- If unsure, classify as NOVEL (safer)

## Output

After researching drugs, update:
1. `enhanced_ground_truth.json` - Add confirmed drugs
2. `.research_state.json` - Record all classifications
3. `evaluation_tasks.json` - Update drugs_researched count

Continue until all drugs for current disease are done or you've researched ~10 drugs.
