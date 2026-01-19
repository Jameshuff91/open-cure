#!/usr/bin/env python3 -u
"""
Open Cure Data Cleansing Agent
==============================

Autonomous agent that cleanses the unified knowledge graph data.
Uses Claude Code CLI with your Max subscription (no API billing).

Usage:
    python data_cleansing_agent.py
    python data_cleansing_agent.py --max-iterations 5
    python data_cleansing_agent.py --model sonnet  # Faster, less quota
"""

import argparse
import fcntl
import json
import os
import re
import select
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Configuration
DEFAULT_MODEL = "opus"
AUTO_CONTINUE_DELAY_SECONDS = 5
DEFAULT_COOLDOWN_HOURS = 5
STATE_FILE_NAME = ".cleansing_state.json"
MAX_SLEEP_SECONDS = 300
COMPLETION_SLEEP_SECONDS = 3600  # 1 hour sleep when all tasks complete
SLACK_WEBHOOK_ENV_VAR = "SLACK_WEBHOOK_URL"

# Paths
PROJECT_ROOT = Path("/Users/jimhuff/github/open-cure")
CLEANSING_DIR = PROJECT_ROOT / "autonomous_cleansing"
PROMPTS_DIR = CLEANSING_DIR / "prompts"
TASKS_FILE = CLEANSING_DIR / "cleansing_tasks.json"
PROGRESS_FILE = CLEANSING_DIR / "claude-progress.txt"


def parse_cooldown_time(response_text: str) -> Optional[int]:
    """Parse cooldown/reset time from rate limit messages."""
    response_lower = response_text.lower()

    rate_limit_indicators = [
        'rate limit', 'usage limit', 'limit reached', 'cooldown',
        'try again', 'resets in', 'quota', 'too many requests', 'plan usage limit',
    ]

    if not any(indicator in response_lower for indicator in rate_limit_indicators):
        return None

    # Try to parse reset time
    hr_min_pattern = r'(\d+)\s*(?:hr|hour)s?\s*(?:(\d+)\s*(?:min|minute)s?)?'
    match = re.search(hr_min_pattern, response_lower)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 3600 + minutes * 60

    min_pattern = r'(\d+)\s*(?:min|minute)s?'
    match = re.search(min_pattern, response_lower)
    if match:
        return int(match.group(1)) * 60

    hours_pattern = r'(\d+)\s*(?:hour)s?'
    match = re.search(hours_pattern, response_lower)
    if match:
        return int(match.group(1)) * 3600

    return DEFAULT_COOLDOWN_HOURS * 3600


def load_state() -> dict:
    """Load agent state from disk."""
    state_file = CLEANSING_DIR / STATE_FILE_NAME
    if state_file.exists():
        try:
            return json.loads(state_file.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_state(state: dict) -> None:
    """Save agent state to disk."""
    state_file = CLEANSING_DIR / STATE_FILE_NAME
    state["updated_at"] = datetime.now().isoformat()
    state_file.write_text(json.dumps(state, indent=2))


def get_task_status() -> tuple[int, int, list]:
    """Get current task completion status."""
    if not TASKS_FILE.exists():
        return 0, 0, []

    data = json.loads(TASKS_FILE.read_text())
    tasks = data.get("tasks", [])
    completed = [t for t in tasks if t.get("passes", False)]
    remaining = [t for t in tasks if not t.get("passes", False)]

    return len(completed), len(tasks), remaining


def check_cooldown_state() -> Optional[datetime]:
    """Check if we're in a cooldown period."""
    state = load_state()
    cooldown_until_str = state.get("cooldown_until")
    if cooldown_until_str:
        try:
            cooldown_until = datetime.fromisoformat(cooldown_until_str)
            if datetime.now() < cooldown_until:
                return cooldown_until
        except ValueError:
            pass
    return None


def set_cooldown_state(seconds: int, reason: str = "rate_limit") -> datetime:
    """Set cooldown state."""
    resume_time = datetime.now() + timedelta(seconds=seconds)
    state = load_state()
    state["cooldown_until"] = resume_time.isoformat()
    state["cooldown_reason"] = reason
    save_state(state)
    return resume_time


def clear_cooldown_state() -> None:
    """Clear cooldown state."""
    state = load_state()
    state.pop("cooldown_until", None)
    state.pop("cooldown_reason", None)
    save_state(state)


def wait_for_cooldown(seconds: int, reason: str = "rate_limit") -> None:
    """Wait for cooldown period with countdown."""
    resume_time = set_cooldown_state(seconds, reason)

    if reason == "completion":
        print("\n" + "=" * 70)
        print("  ALL TASKS COMPLETE - MAINTENANCE MODE")
        print("=" * 70)
        print(f"\nSleeping for 1 hour before next check...")
    else:
        print("\n" + "=" * 70)
        print("  RATE LIMIT DETECTED - PAUSING")
        print("=" * 70)

    print(f"Wait time: {seconds // 3600}h {(seconds % 3600) // 60}m")
    print(f"Resume at: {resume_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    last_status = datetime.now()
    status_interval = timedelta(minutes=10)

    while datetime.now() < resume_time:
        remaining = (resume_time - datetime.now()).total_seconds()
        if remaining <= 0:
            break

        sleep_time = min(MAX_SLEEP_SECONDS, remaining)
        time.sleep(sleep_time)

        if datetime.now() - last_status >= status_interval:
            remaining = (resume_time - datetime.now()).total_seconds()
            if remaining > 0:
                hrs = int(remaining // 3600)
                mins = int((remaining % 3600) // 60)
                print(f"  {hrs}h {mins}m remaining...")
                last_status = datetime.now()

    clear_cooldown_state()
    print("\n" + "=" * 70)
    print("  RESUMING")
    print("=" * 70 + "\n")


def send_slack_notification(message: str) -> bool:
    """Send Slack notification if webhook configured."""
    import urllib.request
    import urllib.error

    url = os.environ.get(SLACK_WEBHOOK_ENV_VAR)
    if not url:
        return False

    try:
        payload = json.dumps({"text": message}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status == 200
    except Exception as e:
        print(f"  Failed to send Slack notification: {e}")
        return False


def load_prompt(name: str) -> str:
    """Load a prompt template."""
    prompt_file = PROMPTS_DIR / f"{name}.md"
    return prompt_file.read_text()


def run_claude_session(prompt: str, model: str = DEFAULT_MODEL) -> tuple[str, str]:
    """Run a single Claude CLI session."""
    print("Starting Claude session...\n")

    cmd = [
        "claude",
        "-p",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--add-dir", str(PROJECT_ROOT),
    ]

    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        process.stdin.write(prompt)
        process.stdin.close()

        response_text = ""

        fd = process.stdout.fileno()
        fl = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)

        while True:
            ret = process.poll()
            readable, _, _ = select.select([process.stdout], [], [], 0.5)

            if readable:
                try:
                    chunk = process.stdout.read(4096)
                    if chunk:
                        sys.stdout.write(chunk)
                        sys.stdout.flush()
                        response_text += chunk
                except (IOError, BlockingIOError):
                    pass

            if ret is not None:
                try:
                    remaining = process.stdout.read()
                    if remaining:
                        sys.stdout.write(remaining)
                        sys.stdout.flush()
                        response_text += remaining
                except Exception:
                    pass
                break

        process.wait(timeout=1800)

        print("\n" + "-" * 70 + "\n")

        cooldown_seconds = parse_cooldown_time(response_text)
        if cooldown_seconds is not None:
            return "cooldown", str(cooldown_seconds)

        if process.returncode == 0:
            return "continue", response_text
        else:
            if len(response_text.strip()) < 100:
                return "cooldown", str(DEFAULT_COOLDOWN_HOURS * 3600)
            return "error", f"Exit code: {process.returncode}"

    except subprocess.TimeoutExpired:
        process.kill()
        return "error", "Session timed out"
    except Exception as e:
        error_str = str(e)
        cooldown_seconds = parse_cooldown_time(error_str)
        if cooldown_seconds is not None:
            return "cooldown", str(cooldown_seconds)
        return "error", error_str


def print_status():
    """Print current cleansing status."""
    completed, total, remaining = get_task_status()

    print("\n" + "=" * 70)
    print(f"  DATA CLEANSING STATUS: {completed}/{total} tasks complete")
    print("=" * 70)

    if remaining:
        print("\nNext tasks:")
        for task in remaining[:3]:
            print(f"  [{task['id']}] {task['name']} (priority {task.get('priority', 2)})")
    else:
        print("\n  ALL TASKS COMPLETE!")

    print("=" * 70 + "\n")


def run_agent(model: str = DEFAULT_MODEL, max_iterations: Optional[int] = None):
    """Run the autonomous cleansing agent."""
    print("\n" + "=" * 70)
    print("  OPEN CURE DATA CLEANSING AGENT")
    print("  Using Claude Code subscription (not API key)")
    print("=" * 70)
    print(f"\nProject: {PROJECT_ROOT}")
    print(f"Model: {model}")
    if max_iterations:
        print(f"Max iterations: {max_iterations}")
    print()

    # Check for pending cooldown
    cooldown_until = check_cooldown_state()
    if cooldown_until:
        remaining = (cooldown_until - datetime.now()).total_seconds()
        if remaining > 0:
            print("Resuming from saved cooldown state...")
            wait_for_cooldown(int(remaining))

    # Load state
    state = load_state()
    iteration = state.get("session_count", 0)

    # Check if first run
    is_first_run = not PROGRESS_FILE.exists()

    if is_first_run:
        print("First run - using initializer prompt")
        PROGRESS_FILE.write_text(f"# Data Cleansing Progress\n\nStarted: {datetime.now().isoformat()}\n\n")
    else:
        print("Continuing existing cleansing project")
        print_status()

    # Main loop
    while True:
        iteration += 1

        # Save session count
        state = load_state()
        state["session_count"] = iteration
        save_state(state)

        # Check max iterations
        if max_iterations and iteration > max_iterations:
            print(f"\nReached max iterations ({max_iterations})")
            break

        # Check task completion
        completed, total, remaining = get_task_status()

        if total > 0 and completed >= total:
            print("\n" + "=" * 70)
            print("  ALL TASKS COMPLETE!")
            print("=" * 70)

            send_slack_notification(
                f":white_check_mark: *Open Cure Data Cleansing Complete!*\n"
                f"All {total} tasks finished. Data is ready for model training."
            )

            # Sleep for 1 hour then check again (maintenance mode)
            wait_for_cooldown(COMPLETION_SLEEP_SECONDS, reason="completion")
            continue

        # Print session header
        print("\n" + "=" * 70)
        print(f"  SESSION {iteration}")
        print(f"  Tasks: {completed}/{total} complete")
        print("=" * 70 + "\n")

        # Choose prompt
        if is_first_run:
            prompt = load_prompt("initializer_prompt")
            is_first_run = False
        else:
            prompt = load_prompt("coding_prompt")

        # Run session
        status, response = run_claude_session(prompt, model)

        # Handle status
        if status == "continue":
            print(f"\nAuto-continuing in {AUTO_CONTINUE_DELAY_SECONDS}s...")
            print_status()
            time.sleep(AUTO_CONTINUE_DELAY_SECONDS)

        elif status == "cooldown":
            try:
                cooldown_seconds = int(response)
            except ValueError:
                cooldown_seconds = DEFAULT_COOLDOWN_HOURS * 3600

            cooldown_seconds += 300  # 5 min buffer
            wait_for_cooldown(cooldown_seconds)

        elif status == "error":
            print(f"\nSession error: {response}")
            print("Retrying in 30 seconds...")
            time.sleep(30)

    # Final summary
    print("\n" + "=" * 70)
    print("  AGENT STOPPED")
    print("=" * 70)
    print_status()


def main():
    parser = argparse.ArgumentParser(
        description="Open Cure Data Cleansing Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python data_cleansing_agent.py                    # Run with Opus
    python data_cleansing_agent.py --model sonnet    # Use Sonnet (faster)
    python data_cleansing_agent.py --max-iterations 3  # Limit sessions
        """
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum sessions (default: unlimited)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model: 'opus', 'sonnet' (default: {DEFAULT_MODEL})"
    )

    args = parser.parse_args()

    # Check Claude CLI
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: Claude CLI not found")
            return
        print(f"Using {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Claude CLI not found")
        print("Install: npm install -g @anthropic-ai/claude-code")
        return

    # Run agent
    try:
        run_agent(model=args.model, max_iterations=args.max_iterations)
    except KeyboardInterrupt:
        print("\n\nInterrupted. Run again to resume.")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise


if __name__ == "__main__":
    main()
