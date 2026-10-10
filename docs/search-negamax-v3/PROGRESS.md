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

B/PVS COMPLETE, REJECTED. 128conditions,2816exact decisions,111oracle vectors,112inherited vectors; geometric1.011x, broad1.086x, wall sum/A0.886. Failed aggregate/broad geometric and immediate-win/D8 regression gates. Memory gates pass. Strongest engine remains original direct.
See pvs/DESIGN.md. Scout unit windows, mandatory qualifying full re-search,
exact original-window TT semantics and all root scores. Same 32 histories,
D4/6/8/10, seven warmed interleaved samples and two memory decisions. Fixed
gates cannot change. Combination with mirror excluded because A failed.

C/profile-guided DESIGN FROZEN at ee9ec57; strongest direct profile complete and audited at92a272f. Selected independent terminal-parent-proof and trusted-TT hypotheses; tuning/DESIGN.md has fixed gates. Payload audit declarationf48401d. First two eight-position blocks queued in serial exec session18421; no candidate timing yet (MCTS pilot3c owns shared lock).

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
- Last pushed Git SHA: `f48401d4bd77e5aba9f5ea309980fcb287dc9de4`, verified origin.
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

MCTS pilot released normally at16:58:33 UTC and queued PVS acquired atomically. First PVS block complete with exact saved vectors; each next block waits fairly for normal release if needed. No current correctness blocker. Wait helper never removes an owner.

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

Latest pushed checkpoint d85ee212c0cfd6a257b89d5b70a4caf0067d72cd (lock correctness). PVS first block32conditions exact, further acceptance pending complete held-out workload.

PVS blocks0:8 and8:16 complete; next serial command sequence handles remaining blocks then audit/summarize/stability/scouts. No acceptance decision until entire workload. Profile utility now checks its own source hash and excludes game-construction overhead from profiling (not used for timing acceptance).

Phase B complete and audited: preserve rejection then commit/push before Phase C. Exact next action: freeze profile design/source/script hashes for original direct; profile four declared costly D10 histories under shared lock; choose up to two hotspot-backed bounded experiments and push their final designs before timing.

B completed checkpoint1e515a9; original direct strongest. Profile design/source/script hashes frozen; no profile measurements yet. Exact next action push profile declaration, then profile under shared lock and compare counters/vectors to frozen primary A.

Profile COMPLETE: 8decisions match A counters/vectors,7.479s, declared92a272f. Selected two independent bounded per-node hypotheses: terminal-parent proof and trusted TT fast path. Final tuning DESIGN+manifest/source/test/generator hashes frozen; prepared22tests pass, no candidate performance timing yet. Next commit/push profile result+tuning declaration, then measured blocks under shared lock.

Current exact next action: resume queued tuning session18421 (blocks0:8 then8:16). Then queue16:24 and24:32, audit/summarize/stability, scripts.audit_negamax_v3_payloads under lock. Pick eligible fastest only after all fixed gates and exact TT/tree audit. If eligible, integrate measured source in a separate production-path commit only after full backend suite. No candidate performance evidence yet; MCTS pid6236 pilot3c540s owns lock as of17:17UTC. Main REPORT updated through B and profile.

C first64conditions complete, batch runtimes59.7s/87.1s, allscore/counter parity. Remaining16:24/24:32 plus audits queued serially; eligibility pending. Resume interrupted partial batches using scripts/resume_negamax_v3.py under shared wrapper with --phase/--start/--stop: skips only parsed complete existing conditions; exclusive files remain untouched; source/manifest hashes validated, original sample counts/criteria preserved.

C COMPLETE: terminal-proof ELIGIBLE allgates, geometric1.135x,broad1.138x, wall/CPU0.878*A,heldout1.130x,retained1.0; trusted-TT REJECTED1.019x. 4224exact decisions111oracle112inherited; payload/tree audit128conditions exactincluding leaf/terminal/probe counts. Next commit/push research result; apply exact measuredterminal source toproduction, fullbackend suiteunder sharedlock; then separate validatedintegration commit before PhaseD.

Accepted production-path integration validated: exact terminal-proof snapshot copied to negamax_agent.py; full backend2867passed/15Torchskips126.21s under sharedlock; canonicalv2unchanged. Researchresult116c3e26fb79c9662aa444102d3fa4338da82b19 pushed/verified. Next separate productionintegrationcommit+push; then final depth10/12 DESIGN/source/manifest before measuring.

Production terminal-proof integration51bd80e pushed/verified, 2867passed15skipped. PhaseD DESIGN+manifest/source digests frozen: originaldirect vs strongestterminal proof, sixpositions D10/12, sevenpaired+two memory, budget900s, two3-position lockbatches. NoD measurements yet. Exactnextaction pushdeeperdeclaration thenbounded measured/audit/payload commands in deeper/DESIGN.md.

D queued serial measurement/audit session54360, noDresults yet; MCTS confirm_primary pid22308 currentlyowner. Standalone hash-pinned baseline-loader prototype3tests pass0.04s; not wired intofrozen toolsuntil Dcomplete. Latest validatedproduction51bd80ed65413fe0d14dc6e6b4d04d60a3381fb6; deeperdesign5e79a7d79f28b58473ae5878acc0476a0446a299 pushed/verified.

D first3-position blockcomplete198.8s, original/accepted D10and12 exact; emptycandidateD12~0.592s,near~0.548s,unchangedTTmemory. Secondblock/auditsongoing serialsession54360. Compatibilityprototype4tests pass0.05s: onlyenumeratedmetadatachanges recognized, measured/gate ASTunchanged, alteredthresholdwithforgednewhashrejected. Frozen runner/variant files remainunediteduntilDcomplete.

D COMPLETE: 264exactdecisions33oracle12fullTT/treecomparisons;4historicalD12vectors/countersmatch; measured391.7s plusauditswithin900. AcceptedD12empty0.592s,near0.548s,seeded1.766s,tail2.907s; largestretained54.52MiB/tracedpeak56.49MiB. Exactnextaction pushcompletedD, wirevalidatedhash-pinnedloaderwitharchivedmetadata-onlycompatibilityguard, thenpredeclareinitial-root-symmetrypolicy asjustifiedadditionalresearch. Strongproduction remains51bd80e.
