# Negamax v3 current progress and recovery

2026-10-10, America/New_York. Work only on this branch; never merge/deploy.

## Verified scope and baseline

- Starting main SHA: `60b99b0d42145da149f433907585b809060c946d`.
- Branch: `research/negamax-v3`.
- Worktree/root: `/Users/andrewcukierwar/Documents/GitHub/board-game-ai-lab-negamax-v3`.
- Origin: `https://github.com/andrewcukierwar/board-game-ai-lab.git`.
- Initial clean HEAD exactly starting main; starting main is ancestor.
- Production engine: unchanged original optimized direct alpha-beta.
- Public caps, API, live services, other worktrees and canonical v2 evidence untouched.
- No paid API calls. Local .venv: CPython 3.11.17, requirements-dev + Pillow/tqdm; Torch absent.

## Phases, hypotheses and fixed gates

A/mirror COMPLETE, both REJECTED. Full and depth>=3 selective symmetry keys
preserve exact results, but broad/aggregate/held-out latency gates fail.
See mirror/DESIGN.md and mirror/REPORT.md. 128 conditions, 4224 audited decisions,
111 independent oracle vectors, 112 inherited vectors. Full geometric 0.884x,
wall sum/A 1.124; selective 0.966x, wall sum/A 1.010. Memory gates pass.
Actual opposite-orientation reuse appears on symmetric empty board; zero on
three asymmetric D10 diagnostic histories. No production integration.

B/PVS DESIGN/SOURCES FROZEN, correctness passed, measurements NOT YET STARTED.
See pvs/DESIGN.md. Scout unit windows, mandatory qualifying full re-search,
exact original-window TT semantics and all root scores. Same 32 histories,
D4/6/8/10, seven warmed interleaved samples and two memory decisions. Fixed
gates cannot change. Combination with mirror excluded because A failed.

C/profile-guided PENDING: profile strongest validated engine after B checkpoint;
select at most two bounded experiments from measured hotspots. During lock
wait, prepared provisional terminal-parent-proof/trusted-TT transformations
and independent small oracle/bound tests; not selected or performance-measured.

D/deeper feasibility PENDING: D10/12 complete decisions and memory, no public cap
change or strength inference. Additional research only if justified by evidence.

## Validation and checkpoints

- A design `244b0c5117a03b154b79db5ecf69a721d8121ca1`.
- A frozen sources `e45cc917c5dc6901820fa6016d0fd11f92eefa06`.
- A WIP evidence `5372bff7406a0d3f53cdd79159e376114cfb386c`.
- A audited rejection `b13340326490ea606f9586197027f272d079017e`.
- B design `59b996fbeb762dd1f3c74c31d407ff2680841665`.
- B diagnostic design `923fbfd48aab5edd8ae4627316958e31e054cf42`.
- Latest validation/preparation `a676a2697bd6d42a917696f39fd124bfda4674bf`.
- Last validated Git SHA: `a676a2697bd6d42a917696f39fd124bfda4674bf` (focused/research validation; production still starting main).
- Last pushed Git SHA: `e7723700da7b28b6cff952af28b7101c061e8558`, verified origin.
- Full focused suite: 89 passed in 7.69s, under shared lock (pvs/focused-tests.txt).
- Provisional tuning correctness: 14 passed in 1.01s (preparation/tuning-correctness.txt).
- Earlier expanded PVS fixture precondition failed (empty D4 needed no re-search);
  failure preserved in mirror/expanded-tests.txt, corrected legal tactical fixture
  passed and full 89-test rerun passed. No hidden score mismatch.
- Full backend suite pending any accepted production integration.

## Shared lock and exact next action

Shared lock: `$HOME/.cache/bgai-laptop-benchmark.lock`. Acquire only with
`scripts/with_benchmark_lock.sh COMMAND...` (atomic mkdir; owner PID/branch/time;
same-process wait/trap; signal interruption cannot release a running child).
Busy exit75 means defer. Never delete another owner's lock. A full batches held
lock for 55.6/101.9/143.1/79.6s, released between blocks. MCTS then ran backend,
throughput/preflight and pilot batches. All ownership respected.

Current temporary blocker: MCTS owns lock (last observed pid79054, pilot1,
max-seconds560). A waiting process is queued via scripts/wait_benchmark_slot.sh
for PVS block0:8, at most1200s wait. It polls for normal release, then delegates
atomic acquisition to the same wrapper. It never removes a lock.

Exact next action: poll existing queued session if still active; otherwise
inspect saved `pvs/condition-*.json` and run only missing whole blocks:

```sh
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 run --phase pvs --start 0 --stop 8
# Then independent blocks8:16,16:24,24:32; never overwrite completed conditions.
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 audit --phase pvs
.venv/bin/python -m scripts.benchmark_negamax_v3 summarize --phase pvs
.venv/bin/python -m scripts.negamax_v3_stability --phase pvs
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.diagnose_negamax_pvs_v3
```

Then report, commit/push completed B, verify remote, profile strongest engine
under lock using predeclared profile manifest, select C experiments. All sources
and runner hashes are frozen; do not edit frozen runner during primary runs.
Exclusive evidence writes fail rather than overwrite. Reproduction uses a fresh
--directory copied DESIGN.md and frozen checkpoint. Checkpoint WIP if interrupted.

Lock correctness tests (isolated temporary path; production wrapper still mandatory shared path) 4 passed in0.21s: successful/failing cleanup, busy lock owner unchanged/child never launched, SIGTERM leaves ownership until foreground child finishes. Provisional expanded checks 22 passed in1.18s; no CPU-intensive workload run while MCTS holds lock.
