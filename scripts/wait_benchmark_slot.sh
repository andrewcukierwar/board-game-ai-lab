#!/bin/sh
# Wait fairly for normal release; atomic acquisition remains in the shared wrapper.
set -u
lock_dir="$HOME/.cache/bgai-laptop-benchmark.lock"
started=$(date +%s)
announced=0
while :; do
    if test -d "$lock_dir"; then
        if test "$announced" = 0; then
            echo 'Waiting for shared benchmark slot; no owner will be removed.' >&2
            announced=1
        fi
        if test "$(( $(date +%s) - started ))" -ge 1200; then
            echo 'Wait budget exhausted; benchmark was not run.' >&2
            exit 75
        fi
        sleep 1
        continue
    fi
    scripts/with_benchmark_lock.sh "$@"
    result=$?
    if test "$result" != 75; then exit "$result"; fi
    sleep 1
 done
