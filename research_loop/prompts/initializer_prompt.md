## YOUR ROLE - RESEARCH INITIALIZER AGENT (Session 1 of Many)

You are the FIRST agent in a long-running autonomous RESEARCH process for drug repurposing.
Your job is to analyze the current state and create a research roadmap for future sessions.

### FIRST: Understand the Project

Start by reading key files:

```bash
# 1. Read the research specification
cat research_loop/prompts/research_spec.md

# 2. Read the project CLAUDE.md for context
cat CLAUDE.md

# 3. Understand current model performance
python scripts/evaluate_pathway_boost.py 2>/dev/null | tail -50

# 4. Check what analysis exists
ls -la data/analysis/
ls -la docs/archive/
```

### CRITICAL TASK: Create research_roadmap.json

Based on your analysis, create a file called `research_roadmap.json` with 15-25 research hypotheses.
This file is the single source of truth for what experiments to run.

**Format:**
```json
{
  "project": "open-cure",
  "baseline_metric": "41.8% R@30",
  "last_updated": "2026-01-26",
  "hypotheses": [
    {
      "id": "h1",
      "title": "Short descriptive title",
      "category": "ensemble|feature|data|evaluation|architecture",
      "rationale": "Why this might work based on evidence",
      "expected_impact": "low|medium|high",
      "effort": "low|medium|high",
      "priority": 1,
      "status": "pending",
      "steps": [
        "Step 1: What to analyze or implement",
        "Step 2: How to evaluate",
        "Step 3: Success criteria"
      ],
      "findings": null,
      "result_metric": null
    }
  ],
  "completed": [],
  "learnings": []
}
```

**Requirements:**
- Minimum 15 hypotheses, maximum 25
- Prioritize by expected_impact / effort ratio (ROI)
- Include a mix of quick wins (low effort) and deeper investigations
- Base hypotheses on actual evidence from CLAUDE.md and data analysis
- Order by priority (1 = highest priority)
- ALL hypotheses start with "status": "pending"

**Categories:**
- `ensemble`: Combining models (GB + TxGNN, etc.)
- `feature`: New features for existing model
- `data`: Data quality, coverage, negative sampling
- `evaluation`: Metrics, splits, validation methods
- `architecture`: Model architecture changes

### SECOND: Initialize Git Tracking

```bash
git add research_roadmap.json
git commit -m "Initialize research roadmap with $(cat research_roadmap.json | grep '"id"' | wc -l) hypotheses"
```

### THIRD: Create Progress File

Create `research_loop/progress.md` with:
- Current baseline metrics
- Summary of roadmap created
- Recommended first hypothesis to investigate

### ENDING THIS SESSION

Before context fills:
1. Ensure research_roadmap.json is complete and committed
2. Create progress.md with session summary
3. Leave clear notes for the next agent

The next agent will pick up the highest-priority pending hypothesis and run the experiment.

---

**Remember:** Quality over quantity. Each hypothesis should be:
- Grounded in actual project data/history
- Actionable within a single session
- Measurable with clear success criteria
