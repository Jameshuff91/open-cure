# Data Cleansing Initializer

You are starting a data cleansing project for the Open Cure drug repurposing knowledge graph.

## YOUR MISSION

Transform messy, merged knowledge graph data into clean, standardized data suitable for training high-quality ML models. The goal is to achieve data quality comparable to PrimeKG (used by TxGNN).

## STEP 1: UNDERSTAND THE PROJECT

```bash
# 1. See your working directory
pwd

# 2. Check project structure
ls -la /Users/jimhuff/github/open-cure/

# 3. Read the data cleansing specification
cat /Users/jimhuff/github/open-cure/autonomous_cleansing/prompts/data_cleansing_spec.md

# 4. Read the task list
cat /Users/jimhuff/github/open-cure/autonomous_cleansing/cleansing_tasks.json

# 5. Check current data state
wc -l /Users/jimhuff/github/open-cure/data/processed/unified_nodes.csv
wc -l /Users/jimhuff/github/open-cure/data/processed/unified_edges.csv

# 6. Sample the data
head -10 /Users/jimhuff/github/open-cure/data/processed/unified_nodes.csv
head -10 /Users/jimhuff/github/open-cure/data/processed/unified_edges.csv
```

## STEP 2: SET UP CLEANSING INFRASTRUCTURE

Create the necessary directories and files:

```bash
mkdir -p /Users/jimhuff/github/open-cure/data/reference
mkdir -p /Users/jimhuff/github/open-cure/data/processed/validation
mkdir -p /Users/jimhuff/github/open-cure/autonomous_cleansing/scripts
```

## STEP 3: CREATE BASELINE STATISTICS

Before any cleansing, document the current state:

1. Count unique entity types
2. Count unique relation types
3. Sample entities to understand naming patterns
4. Save as `before_stats.json`

## STEP 4: BEGIN TASK 1 - Normalize Entity Types

This is the highest priority task. Start implementing it:

1. Create a Python script: `scripts/normalize_entity_types.py`
2. The script should:
   - Read unified_nodes.csv
   - Apply type normalization (see spec for mappings)
   - Write to unified_nodes_clean.csv
   - Log all transformations
3. Run the script
4. Validate results

## STEP 5: UPDATE PROGRESS

After completing Task 1:

1. Update `cleansing_tasks.json` - set Task 1 `"passes": true`
2. Update `claude-progress.txt` with what you accomplished
3. Commit your changes

## IMPORTANT NOTES

- **Create backups** before modifying any data files
- **Validate counts** - node/edge counts should not change during normalization
- **Log everything** - all transformations should be auditable
- **Test scripts** on small samples first before running on full data
- **Use pandas** for CSV processing - it handles the 16M edge file efficiently

## FILES TO CREATE THIS SESSION

1. `scripts/normalize_entity_types.py` - Entity type normalization
2. `scripts/generate_stats.py` - Generate before/after statistics
3. `data/processed/before_stats.json` - Baseline statistics
4. `claude-progress.txt` - Progress notes

Begin by running Step 1 to understand the project.
