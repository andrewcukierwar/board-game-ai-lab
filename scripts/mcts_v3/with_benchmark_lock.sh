#!/bin/sh
# Run one command while holding the shared laptop benchmark lock.
#
#   scripts/mcts_v3/with_benchmark_lock.sh <command> [args...]
#
# The lock is an atomically created directory shared with the Negamax agent.
# This script never removes a lock it did not create: the cleanup trap is
# installed only after our own mkdir succeeds, and the command runs inside this
# same shell process so the lock is released whenever it exits or is signalled.
# Set BGAI_LOCK_WAIT_SECONDS to poll for a busy lock instead of failing at once.
LOCK="$HOME/.cache/bgai-laptop-benchmark.lock"
WAIT="${BGAI_LOCK_WAIT_SECONDS:-0}"
waited=0
until mkdir "$LOCK" 2>/dev/null; do
    if [ "$waited" -ge "$WAIT" ]; then
        echo "benchmark lock busy after ${waited}s; owner:" >&2
        cat "$LOCK/owner" >&2 2>/dev/null || echo "(no owner file)" >&2
        exit 75
    fi
    sleep 5
    waited=$((waited + 5))
done
release() { rm -rf "$LOCK"; }
trap release EXIT
trap 'release; trap - EXIT; exit 130' INT TERM HUP
{
    echo "pid=$$"
    echo "branch=$(git branch --show-current 2>/dev/null)"
    echo "start=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "worktree=$(pwd)"
    echo "command=$*"
} > "$LOCK/owner"
"$@"
status=$?
exit $status
