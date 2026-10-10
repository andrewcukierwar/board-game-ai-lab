# Negamax v3 progress and recovery journal

Date: 2026-10-10 (America/New_York). No merge or deployment authorized.

- Starting main SHA: `60b99b0d42145da149f433907585b809060c946d`.
- Branch: `research/negamax-v3`.
- Worktree: `/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab-negamax-v3`.
- Verified clean initial tree, origin `https://github.com/andrewcukierwar/board-game-ai-lab.git`, starting SHA ancestor; other worktrees untouched.
- Baseline: current optimized direct alpha-beta, immediate-win-first + same-depth hints, exact incremental evaluation, packed lossless TT. Earlier iterative/cross-depth variants remain rejected.
- Phase A: research-only mirror/full and mirror/depth>=3 implemented; 68 focused tests passed (9.19s); frozen manifest and sources ready; timing batches 0:8 and 8:16 complete; 16:24 underway; 24:32 pending. Interim evidence suggests asymmetry overhead; no acceptance decision yet.
- Phase B: PVS pending independent declaration after A checkpoint.
- Phase C: up to two profile-guided bounded experiments pending measurement.
- Phase D: experimental depths 10/12 pending; public caps stay fixed.
- Hypotheses/criteria: see each phase DESIGN.md; never revise after measurements.
- Current implementation/tests: unchanged production; worktree-local CPython 3.11 venv installed with requirements-dev plus pillow/tqdm (no Torch). PVS implemented independently but not performance-measured.
- Measurements completed: none.
- Last validated Git SHA: `60b99b0d42145da149f433907585b809060c946d` (inherited validated baseline).
- Last pushed Git SHA: `e45cc917c5dc6901820fa6016d0fd11f92eefa06` (verified origin; frozen primary design/source).
- Known blockers: none; lock directory outside workspace requires authorized sandbox access. Never remove an active owner's lock.
- Exact next action: commit/push frozen mirror manifest + sources; run condition blocks 0:8,8:16,16:24,24:32 through shared-lock wrapper; audit and summarize, checkpoint rejection/acceptance before Phase B.

All measured CPU-intensive runs, profiling, memory and full suite use
`scripts/with_benchmark_lock.sh COMMAND...`. It atomically creates
`$HOME/.cache/bgai-laptop-benchmark.lock`; busy exit 75 is a deferral, not failure.
Evidence writes must be exclusive. Use new directories to reproduce.

WIP checkpoint: partial mirror condition files preserved; separate orientation/profile diagnostic declaration frozen before its own measurement. Expanded bound/hint/PVS re-search tests await run after measured batch. Do not rerun completed condition files (exclusive writes); resume only missing conditions. Primary source hashes remain frozen.
