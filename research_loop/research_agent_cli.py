#!/usr/bin/env python3 -u
"""
Autonomous Research Agent - CLI Version
========================================

Runs continuous research experiments using Claude Code CLI.
Uses your Claude Code subscription (Max plan) instead of direct API billing.

Example Usage:
    python research_loop/research_agent_cli.py
    python research_loop/research_agent_cli.py --max-iterations 5
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
import urllib.request
import urllib.error
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from prompts import get_initializer_prompt, get_research_prompt  # type: ignore[import-not-found]
from progress import print_session_header, print_progress_summary, count_hypothesis_status  # type: ignore[import-not-found]


# Configuration
DEFAULT_MODEL = "opus"  # Uses Opus 4.5 for best research quality
AUTO_CONTINUE_DELAY_SECONDS = 3
DEFAULT_COOLDOWN_HOURS = 5  # Fallback if we can't parse the reset time
STATE_FILE_NAME = ".research_agent_state.json"
MAX_SLEEP_SECONDS = 300  # Max 5 min sleep at a time to survive system sleep/wake
STUCK_THRESHOLD_SESSIONS = 3  # Sessions without progress before considered stuck
SLACK_WEBHOOK_ENV_VAR = "SLACK_WEBHOOK_URL"
MAX_CONSECUTIVE_ERRORS = 3  # Stop after this many consecutive errors


def parse_cooldown_time(response_text: str) -> Optional[int]:
    """
    Parse the cooldown/reset time from rate limit messages.
    Returns seconds to wait, or None if no rate limit detected.
    """
    response_lower = response_text.lower()

    rate_limit_indicators = [
        'rate limit', 'usage limit', 'limit reached', 'cooldown',
        'try again', 'resets in', 'quota', 'too many requests', 'plan usage limit',
    ]

    is_rate_limit = any(indicator in response_lower for indicator in rate_limit_indicators)
    if not is_rate_limit:
        return None

    # Try to parse the reset time
    # Pattern: "X hr Y min"
    hr_min_pattern = r'(\d+)\s*(?:hr|hour)s?\s*(?:(\d+)\s*(?:min|minute)s?)?'
    match = re.search(hr_min_pattern, response_lower)
    if match:
        hours = int(match.group(1))
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 3600 + minutes * 60

    # Pattern: "X min"
    min_pattern = r'(\d+)\s*(?:min|minute)s?'
    match = re.search(min_pattern, response_lower)
    if match:
        return int(match.group(1)) * 60

    # Pattern: "X hours"
    hours_pattern = r'(\d+)\s*(?:hour)s?'
    match = re.search(hours_pattern, response_lower)
    if match:
        return int(match.group(1)) * 3600

    return DEFAULT_COOLDOWN_HOURS * 3600


def get_state_file_path(project_dir: Path) -> Path:
    """Get the path to the state file for a project."""
    return project_dir / STATE_FILE_NAME


def load_agent_state(project_dir: Path) -> dict:
    """Load agent state from disk."""
    state_file = get_state_file_path(project_dir)
    if state_file.exists():
        try:
            with open(state_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_agent_state(project_dir: Path, state: dict) -> None:
    """Save agent state to disk."""
    state_file = get_state_file_path(project_dir)
    state["updated_at"] = datetime.now().isoformat()
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2)


def clear_cooldown_state(project_dir: Path) -> None:
    """Clear any cooldown state."""
    state = load_agent_state(project_dir)
    state.pop("cooldown_until", None)
    state.pop("cooldown_reason", None)
    save_agent_state(project_dir, state)


def check_cooldown_state(project_dir: Path) -> Optional[datetime]:
    """Check if we're in a cooldown period."""
    state = load_agent_state(project_dir)
    cooldown_until_str = state.get("cooldown_until")
    if cooldown_until_str:
        try:
            cooldown_until = datetime.fromisoformat(cooldown_until_str)
            if datetime.now() < cooldown_until:
                return cooldown_until
            else:
                clear_cooldown_state(project_dir)
        except ValueError:
            pass
    return None


def set_cooldown_state(project_dir: Path, seconds: int, reason: str = "rate_limit") -> datetime:
    """Set cooldown state. Returns the resume time."""
    resume_time = datetime.now() + timedelta(seconds=seconds)
    state = load_agent_state(project_dir)
    state["cooldown_until"] = resume_time.isoformat()
    state["cooldown_reason"] = reason
    state["cooldown_seconds"] = seconds
    save_agent_state(project_dir, state)
    return resume_time


def send_slack_notification(message: str, webhook_url: Optional[str] = None) -> bool:
    """Send a notification to Slack via webhook."""
    url = webhook_url or os.environ.get(SLACK_WEBHOOK_ENV_VAR)
    if not url:
        print("  ℹ️  No Slack webhook configured (set SLACK_WEBHOOK_URL env var)")
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
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
        print(f"  ⚠️  Failed to send Slack notification: {e}")
        return False


def check_if_stuck(project_dir: Path) -> tuple[bool, str]:
    """Check if the agent is stuck (no progress for N sessions)."""
    state = load_agent_state(project_dir)
    counts = count_hypothesis_status(project_dir)

    completed = counts["validated"] + counts["invalidated"] + counts["inconclusive"]
    progress_history = state.get("progress_history", [])

    if progress_history:
        last_completed = progress_history[-1].get("completed", 0)
        sessions_without_progress = state.get("sessions_without_progress", 0)

        if completed <= last_completed:
            sessions_without_progress += 1
        else:
            sessions_without_progress = 0

        state["sessions_without_progress"] = sessions_without_progress

        if sessions_without_progress >= STUCK_THRESHOLD_SESSIONS:
            remaining = counts["pending"]
            reason = (f"No progress for {sessions_without_progress} sessions. "
                      f"{completed}/{counts['total']} hypotheses complete, {remaining} pending.")
            return True, reason

    progress_history.append({
        "session": state.get("session_count", 0),
        "completed": completed,
        "total": counts["total"],
        "timestamp": datetime.now().isoformat()
    })

    state["progress_history"] = progress_history[-20:]
    save_agent_state(project_dir, state)

    return False, ""


def notify_stuck(project_dir: Path, reason: str) -> None:
    """Notify the user that the agent is stuck."""
    project_name = project_dir.name
    counts = count_hypothesis_status(project_dir)
    completed = counts["validated"] + counts["invalidated"] + counts["inconclusive"]

    message = f"""🚨 *Research Agent Needs Help*

*Project:* {project_name}
*Progress:* {completed}/{counts['total']} hypotheses

*Issue:* {reason}

The remaining hypotheses may require:
• Human judgment on approach
• External data or resources
• Manual intervention

Please check the agent and provide guidance."""

    print("\n" + "=" * 70)
    print("  🚨 AGENT STUCK - NEEDS HUMAN HELP")
    print("=" * 70)
    print(f"\n{reason}\n")

    if send_slack_notification(message):
        print("  ✅ Slack notification sent")
    print("=" * 70)


def notify_completion(project_dir: Path) -> None:
    """Notify the user that research is complete."""
    project_name = project_dir.name
    counts = count_hypothesis_status(project_dir)

    message = f"""🎉 *Research Loop Complete!*

*Project:* {project_name}
*Validated:* {counts['validated']}
*Invalidated:* {counts['invalidated']}
*Inconclusive:* {counts['inconclusive']}

All hypotheses have been investigated!"""

    print("\n" + "=" * 70)
    print("  🎉 RESEARCH COMPLETE!")
    print("=" * 70)

    if send_slack_notification(message):
        print("  ✅ Slack notification sent")


def wait_for_cooldown(seconds: int, project_dir: Optional[Path] = None) -> None:
    """Wait for the specified cooldown period with countdown display."""
    resume_time = datetime.now() + timedelta(seconds=seconds)

    if project_dir:
        set_cooldown_state(project_dir, seconds, "rate_limit")

    print("\n" + "=" * 70)
    print("  ⏸️  RATE LIMIT DETECTED - PAUSING")
    print("=" * 70)
    print(f"\nWait time: {seconds // 3600}h {(seconds % 3600) // 60}m")
    print(f"Resume at: {resume_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThe script will automatically resume when the cooldown expires.")
    print("State saved to disk - will recover if process is interrupted.")
    print("=" * 70)

    last_status_update = datetime.now()
    status_interval = timedelta(minutes=5)

    while datetime.now() < resume_time:
        remaining = (resume_time - datetime.now()).total_seconds()
        if remaining <= 0:
            break

        sleep_time = min(MAX_SLEEP_SECONDS, remaining)
        time.sleep(sleep_time)

        if datetime.now() - last_status_update >= status_interval:
            remaining = (resume_time - datetime.now()).total_seconds()
            if remaining > 0:
                hours_left = int(remaining // 3600)
                mins_left = int((remaining % 3600) // 60)
                print(f"  ⏳ {hours_left}h {mins_left}m remaining until resume...")
                last_status_update = datetime.now()

    if project_dir:
        clear_cooldown_state(project_dir)

    print("\n" + "=" * 70)
    print("  ▶️  COOLDOWN COMPLETE - RESUMING")
    print("=" * 70 + "\n")


def run_claude_session(
    prompt: str,
    project_dir: Path,
    model: str = DEFAULT_MODEL,
) -> tuple[str, str]:
    """Run a single Claude CLI session."""
    print("Sending prompt to Claude CLI...\n")

    cmd = [
        "claude",
        "-p",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--add-dir", str(project_dir.resolve()),
    ]

    cwd = project_dir.resolve()

    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(cwd),
        )

        assert process.stdin is not None
        assert process.stdout is not None

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

        # Check for research pause signal
        if "<promise>RESEARCH PAUSED</promise>" in response_text:
            return "paused", response_text

        cooldown_seconds = parse_cooldown_time(response_text)
        if cooldown_seconds is not None:
            return "cooldown", str(cooldown_seconds)

        if process.returncode == 0:
            return "continue", response_text
        else:
            error_msg = f"Exit code: {process.returncode}"
            cooldown_seconds = parse_cooldown_time(error_msg + " " + response_text)
            if cooldown_seconds is not None:
                return "cooldown", str(cooldown_seconds)

            # Check for "No messages returned" error - likely nested session conflict
            if "No messages returned" in response_text:
                print("\n⚠️  'No messages returned' error detected.")
                print("    This often happens when running from within another Claude session.")
                print("    Try running from a fresh terminal instead.")
                return "error", "No messages returned - try running from fresh terminal"

            if len(response_text.strip()) < 100 and process.returncode != 0:
                print(f"WARNING: Short/empty response with exit code {process.returncode}")
                return "cooldown", str(DEFAULT_COOLDOWN_HOURS * 3600)

            return "error", error_msg

    except subprocess.TimeoutExpired:
        if process is not None:
            process.kill()
        return "error", "Session timed out after 30 minutes"
    except Exception as e:
        error_str = str(e)
        cooldown_seconds = parse_cooldown_time(error_str)
        if cooldown_seconds is not None:
            return "cooldown", str(cooldown_seconds)
        return "error", error_str


def run_research_agent(
    project_dir: Path,
    model: str = DEFAULT_MODEL,
    max_iterations: Optional[int] = None,
) -> None:
    """Run the autonomous research agent loop."""
    print("\n" + "=" * 70)
    print("  AUTONOMOUS RESEARCH AGENT (CLI Version)")
    print("  Using Claude Code subscription")
    print("=" * 70)
    print(f"\nProject directory: {project_dir}")
    print(f"Model: {model}")
    if max_iterations:
        print(f"Max iterations: {max_iterations}")
    else:
        print("Max iterations: Unlimited (will run until completion or pause)")
    print()

    # Check for pending cooldown from previous run
    cooldown_until = check_cooldown_state(project_dir)
    if cooldown_until:
        remaining = (cooldown_until - datetime.now()).total_seconds()
        if remaining > 0:
            print(f"\n⏸️  Resuming from saved cooldown state...")
            wait_for_cooldown(int(remaining), project_dir)

    # Load session count
    state = load_agent_state(project_dir)
    starting_iteration = state.get("session_count", 0)

    # Check if this is a fresh start or continuation
    roadmap_file = project_dir / "research_roadmap.json"
    is_first_run = not roadmap_file.exists()

    if is_first_run:
        print("Fresh start - will use initializer agent to create research roadmap")
        print()
    else:
        print("Continuing existing research")
        print_progress_summary(project_dir)

    # Main loop
    iteration = starting_iteration
    consecutive_errors = 0

    while True:
        iteration += 1

        state = load_agent_state(project_dir)
        state["session_count"] = iteration
        save_agent_state(project_dir, state)

        if max_iterations and iteration > max_iterations:
            print(f"\nReached max iterations ({max_iterations})")
            break

        print_session_header(iteration, is_first_run)

        if is_first_run:
            prompt = get_initializer_prompt()
            is_first_run = False
        else:
            prompt = get_research_prompt()

        status, response = run_claude_session(prompt, project_dir, model)

        if status == "continue":
            consecutive_errors = 0  # Reset error counter on success
            print(f"\nAgent will auto-continue in {AUTO_CONTINUE_DELAY_SECONDS}s...")
            print_progress_summary(project_dir)
            time.sleep(AUTO_CONTINUE_DELAY_SECONDS)

        elif status == "paused":
            print("\n" + "=" * 70)
            print("  🔬 RESEARCH PAUSED BY AGENT")
            print("=" * 70)
            print("\nThe agent has requested a pause. Check progress.md for details.")
            break

        elif status == "cooldown":
            try:
                cooldown_seconds = int(response)
            except ValueError:
                cooldown_seconds = DEFAULT_COOLDOWN_HOURS * 3600
            cooldown_seconds += 300
            wait_for_cooldown(cooldown_seconds, project_dir)

        elif status == "error":
            consecutive_errors += 1
            print(f"\nSession encountered an error ({consecutive_errors}/{MAX_CONSECUTIVE_ERRORS}): {response}")

            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print("\n" + "=" * 70)
                print("  ❌ TOO MANY CONSECUTIVE ERRORS - STOPPING")
                print("=" * 70)
                print("\nTry running from a fresh terminal (not from within Claude Code):")
                print(f"  cd {project_dir}")
                print("  python3 research_loop/research_agent_cli.py")
                break

            print("Will retry with a fresh session...")
            time.sleep(AUTO_CONTINUE_DELAY_SECONDS)

        # Check for completion
        counts = count_hypothesis_status(project_dir)
        if counts["total"] > 0 and counts["pending"] == 0 and counts["in_progress"] == 0:
            notify_completion(project_dir)
            break

        # Check if stuck
        if status == "continue":
            is_stuck, stuck_reason = check_if_stuck(project_dir)
            if is_stuck:
                notify_stuck(project_dir, stuck_reason)
                state = load_agent_state(project_dir)
                state["sessions_without_progress"] = 0
                state["last_stuck_notification"] = datetime.now().isoformat()
                save_agent_state(project_dir, state)

        if max_iterations is None or iteration < max_iterations:
            print("\nPreparing next session...\n")
            time.sleep(1)

    # Final summary
    print("\n" + "=" * 70)
    print("  SESSION COMPLETE")
    print("=" * 70)
    print(f"\nProject directory: {project_dir}")
    print_progress_summary(project_dir)
    print("\nDone!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Autonomous Research Agent - Uses your Claude Code subscription",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start research loop
  python research_loop/research_agent_cli.py

  # Limit iterations for testing
  python research_loop/research_agent_cli.py --max-iterations 3

  # Use sonnet model (faster, uses less quota)
  python research_loop/research_agent_cli.py --model sonnet

Environment Variables:
  SLACK_WEBHOOK_URL  - Slack webhook for notifications
        """,
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of agent iterations (default: unlimited)",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Claude model to use (default: {DEFAULT_MODEL})",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Check that claude CLI is available
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("Error: Claude CLI not found or not working")
            print("Install it with: npm install -g @anthropic-ai/claude-code")
            return
        print(f"Using Claude CLI: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Error: Claude CLI not found")
        return

    # Project directory is the open-cure root
    project_dir = Path(__file__).parent.parent.resolve()

    try:
        run_research_agent(
            project_dir=project_dir,
            model=args.model,
            max_iterations=args.max_iterations,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        print("To resume, run the same command again")
    except Exception as e:
        print(f"\nFatal error: {e}")
        raise


if __name__ == "__main__":
    main()
