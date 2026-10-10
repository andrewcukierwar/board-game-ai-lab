# Negamax v3 progress and recovery journal

Date: 2026-10-10 (America/New_York). No merge or deployment authorized.

- Starting main SHA: `60b99b0d42145da149f433907585b809060c946d`.
- Branch: `research/negamax-v3`.
- Worktree: `/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab-negamax-v3`.
- Verified clean initial tree, origin `https://github.com/andrewcukierwar/board-game-ai-lab.git`, starting SHA ancestor; other worktrees untouched.
- Baseline: current optimized direct alpha-beta, immediate-win-first + same-depth hints, exact incremental evaluation, packed lossless TT. Earlier iterative/cross-depth variants remain rejected.
- Phase A: DESIGN declared; implementation and correctness pending; no measurements yet.
- Phase B: PVS pending independent declaration after A checkpoint.
- Phase C: up to two profile-guided bounded experiments pending measurement.
- Phase D: experimental depths 10/12 pending; public caps stay fixed.
- Hypotheses/criteria: see each phase DESIGN.md; never revise after measurements.
- Current implementation/tests: unchanged production; local environment absent, setup pending.
- Measurements completed: none.
- Last validated Git SHA: `60b99b0d42145da149f433907585b809060c946d` (inherited validated baseline).
- Last pushed Git SHA: initial branch verification pending first push.
- Known blockers: none; lock directory outside workspace requires authorized sandbox access. Never remove an active owner's lock.
- Exact next action: checkpoint and push this design, establish local .venv, implement research-only mirror variants and focused tests, freeze source manifest and push before acquiring shared lock for measurement.

All measured CPU-intensive runs, profiling, memory and full suite use
`scripts/with_benchmark_lock.sh COMMAND...`. It atomically creates
`$HOME/.cache/bgai-laptop-benchmark.lock`; busy exit 75 is a deferral, not failure.
Evidence writes must be exclusive. Use new directories to reproduce.
