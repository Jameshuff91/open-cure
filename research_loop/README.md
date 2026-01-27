# Open-Cure Autonomous Research Loop

An autonomous research agent that continuously runs experiments to improve drug repurposing predictions.

## How It Works

1. **Initializer Session**: Creates `research_roadmap.json` with 15-25 prioritized hypotheses
2. **Research Sessions**: Each session picks the highest-priority pending hypothesis, runs the experiment, records findings, and proposes new hypotheses
3. **Self-Improving**: Findings from one experiment inform new hypotheses

## Usage

```bash
# Start the research loop (runs until complete or paused)
python research_loop/research_agent_cli.py

# Limit iterations for testing
python research_loop/research_agent_cli.py --max-iterations 3

# Use sonnet model (faster, cheaper)
python research_loop/research_agent_cli.py --model sonnet
```

## Key Files

| File | Purpose |
|------|---------|
| `research_roadmap.json` | Source of truth for all hypotheses |
| `research_loop/progress.md` | Session notes and recommendations |
| `prompts/research_spec.md` | Project context and constraints |
| `prompts/research_prompt.md` | Main loop prompt |

## Hypothesis Status

- `pending` - Not yet started
- `in_progress` - Currently being investigated
- `validated` - Hypothesis improved metrics or revealed actionable insight
- `invalidated` - Hypothesis did not work (still valuable learning)
- `inconclusive` - Needs more investigation or was blocked

## Completion Criteria

The loop pauses when:
- All hypotheses exhausted and no new high-priority ones generated
- Major breakthrough requiring human review
- Blocked on external resources

## Slack Notifications

Set `SLACK_WEBHOOK_URL` environment variable for notifications when:
- Agent is stuck (no progress for 3 sessions)
- Research complete

## Crash Recovery

State is saved to `.research_agent_state.json`. If the process is interrupted, just run again to resume.
