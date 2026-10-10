#!/bin/sh
# Shared with the simultaneous MCTS researcher. Never steal an active lock.
set -eu
lock_dir="$HOME/.cache/bgai-laptop-benchmark.lock"
mkdir -p "$HOME/.cache"
if ! mkdir "$lock_dir" 2>/dev/null; then
    echo "Benchmark deferred: shared lock busy ($lock_dir)" >&2
    if test -f "$lock_dir/owner"; then cat "$lock_dir/owner" >&2; fi
    exit 75
fi
cleanup() {
    # Only delete our own record/directory, after foreground command returns.
    if test -f "$lock_dir/owner" && test "$(sed -n 's/^pid=//p' "$lock_dir/owner")" = "$$"; then
        rm "$lock_dir/owner"
        rmdir "$lock_dir"
    fi
}
trap cleanup EXIT
# Defer cleanup on signals until synchronous foreground command/children finish.
trap ':' INT TERM HUP
branch=$(git branch --show-current)
printf 'pid=%s\nbranch=%s\nstart_utc=%s\nworktree=%s\n' "$$" "$branch" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$PWD" > "$lock_dir/owner"
"$@"
