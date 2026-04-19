#!/bin/bash
# Watches research_agent_cli.py and fires one iMessage when it dies, then exits.
# Checks every 60s. Log: research_loop/logs/death_monitor.log

set -u
PHONE="+17247472408"
PATTERN="research_agent_cli.py"
LOG="/Users/jimhuff/github/open-cure/research_loop/logs/death_monitor.log"

echo "$(date '+%Y-%m-%d %H:%M:%S'): monitor PID=$$ starting, watching $PATTERN" >> "$LOG"

while pgrep -f "$PATTERN" > /dev/null 2>&1; do
  sleep 60
done

DIED_AT=$(date '+%Y-%m-%d %H:%M:%S')
TAIL=$(tail -3 /Users/jimhuff/github/open-cure/research_loop/logs/agent.log 2>/dev/null | tr '\n' ' ')
MSG="Open-Cure research loop STOPPED at ${DIED_AT}. Last log: ${TAIL}"

echo "$(date '+%Y-%m-%d %H:%M:%S'): process death detected, sending iMessage" >> "$LOG"

osascript <<EOF >> "$LOG" 2>&1
tell application "Messages"
  set svc to 1st account whose service type = iMessage
  send "${MSG//\"/\\\"}" to participant "$PHONE" of svc
end tell
EOF

echo "$(date '+%Y-%m-%d %H:%M:%S'): notification attempted, exiting" >> "$LOG"
