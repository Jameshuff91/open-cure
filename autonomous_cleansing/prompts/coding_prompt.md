# Data Cleansing Agent - Continuation Session

You are continuing work on the Open Cure data cleansing project.
This is a FRESH context window - you have no memory of previous sessions.

## STEP 1: GET YOUR BEARINGS (MANDATORY)

```bash
# 1. See your working directory
cd /Users/jimhuff/github/open-cure
pwd

# 2. Read progress from previous sessions
cat autonomous_cleansing/claude-progress.txt

# 3. Check task status
cat autonomous_cleansing/cleansing_tasks.json | python3 -c "import json,sys; d=json.load(sys.stdin); done=[t for t in d['tasks'] if t['passes']]; todo=[t for t in d['tasks'] if not t['passes']]; print(f'Completed: {len(done)}/{len(d[\"tasks\"])}'); print('Next tasks:'); [print(f'  [{t[\"id\"]}] {t[\"name\"]}') for t in todo[:3]]"

# 4. Check what files exist
ls -la data/processed/
ls -la autonomous_cleansing/scripts/ 2>/dev/null || echo "No scripts yet"

# 5. Recent git history
git log --oneline -10

# 6. Read the spec if needed
cat autonomous_cleansing/prompts/data_cleansing_spec.md | head -100
```

## STEP 2: IDENTIFY NEXT TASK

Look at `cleansing_tasks.json` and find the highest-priority task with `"passes": false`.

Priority order:
1. Tasks with priority=1 that aren't done
2. Tasks with priority=2 that aren't done
3. Validation and documentation tasks

## STEP 3: IMPLEMENT THE TASK

For each task:

1. **Understand requirements** - Read the task description and acceptance criteria
2. **Create/update script** - Write Python code in `autonomous_cleansing/scripts/`
3. **Test on sample** - Run on first 1000 rows to verify logic
4. **Run on full data** - Process all 273K nodes or 16M edges
5. **Validate results** - Check counts, spot-check samples
6. **Log transformations** - Document what changed

### Script Guidelines

```python
#!/usr/bin/env python3
"""
Script description here.
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# Always define paths relative to project root
PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
DATA_DIR = PROJECT_ROOT / "data" / "processed"
SCRIPTS_DIR = PROJECT_ROOT / "autonomous_cleansing" / "scripts"

def main():
    # Your implementation
    pass

if __name__ == "__main__":
    main()
```

### Processing Large Files

For the 16M edge file, use chunked processing:

```python
# Process in chunks to avoid memory issues
chunk_size = 100_000
chunks = pd.read_csv(input_file, chunksize=chunk_size)
for i, chunk in enumerate(chunks):
    # Process chunk
    # Append to output file
```

## STEP 4: VALIDATE RESULTS

After running a cleansing script:

1. **Count check** - Row counts should match (unless deduplicating)
2. **Sample check** - Manually inspect 10-20 random rows
3. **Integrity check** - No null values in required fields
4. **Referential check** - Edges still reference valid nodes

```bash
# Example validation
wc -l data/processed/unified_nodes.csv
wc -l data/processed/unified_nodes_clean.csv

# Should match (unless deduplicating)
```

## STEP 5: UPDATE TASK STATUS

After thorough validation:

1. Edit `cleansing_tasks.json`
2. Change `"passes": false` to `"passes": true` for completed task
3. **Only mark as passing if ALL acceptance criteria are met**

## STEP 6: UPDATE PROGRESS NOTES

Update `autonomous_cleansing/claude-progress.txt`:

```
## Session [DATE]

### Completed
- Task X: [description of what was done]

### Results
- Before: [stats]
- After: [stats]

### Issues Encountered
- [Any problems and how they were resolved]

### Next Steps
- Task Y should be done next because...

### Current Status
- Tasks completed: X/12
- Data quality: [brief assessment]
```

## STEP 7: COMMIT CHANGES

```bash
git add .
git commit -m "Data cleansing: [task name]

- [Specific changes made]
- Before: [key stat]
- After: [key stat]
- Validation: [passed/issues]
"
```

## STEP 8: CHECK IF ALL TASKS COMPLETE

```bash
# Count remaining tasks
cat autonomous_cleansing/cleansing_tasks.json | python3 -c "
import json,sys
d=json.load(sys.stdin)
todo=[t for t in d['tasks'] if not t['passes']]
print(f'Remaining tasks: {len(todo)}')
if len(todo) == 0:
    print('ALL TASKS COMPLETE!')
"
```

If all tasks are complete:
1. Generate final data quality report
2. Update progress notes with final summary
3. The agent will enter maintenance mode (checking hourly for any issues)

## IMPORTANT REMINDERS

**Quality over speed** - Take time to validate thoroughly. Bad data cleaning is worse than no cleaning.

**Preserve originals** - Never overwrite original files. Always write to new files (_clean suffix).

**Log everything** - Every transformation should be logged to `cleansing_log.json`.

**Test first** - Always test scripts on small samples before processing full data.

**Idempotent scripts** - Scripts should be safe to re-run.

## REFERENCE DATA SOURCES

If you need to download reference data:

```bash
# DrugBank vocabulary (open data)
# https://go.drugbank.com/releases/latest#open-data

# NCBI Gene Info
wget -P data/reference/ https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz
gunzip data/reference/Homo_sapiens.gene_info.gz
```

---

Begin by running Step 1 (Get Your Bearings).
