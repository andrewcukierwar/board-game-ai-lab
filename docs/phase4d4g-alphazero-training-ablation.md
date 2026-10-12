# Phase 4D.4G — controlled AlphaZero exploration training ablation

This phase prepares and launches the explicitly authorized comparison of eight
versus sixteen exploratory self-play plies. **Experimental results are pending.**
Preparation, independently managed execution, and final analysis are separate
stages at the user's request. This document is the prospective report shell;
it must not be cited as completed learning, tactical or strength evidence.

## Prospective design and source freeze

Branch `phase4d4g-alphazero-training-ablation`, separate worktree
`experiment-output/phase4d4g-worktree`, starts exactly at
`9325f0a19e6b0106ea1dcd265316b57323f8eb4f`. Historical checkout remains at
`1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`.

The complete [prospective protocol](../research/alphazero_training_ablation/DESIGN.md)
defines fixtures, endpoints, four separate decisions, bootstrap rules, execution
order, budgets, deadline/recovery policy and publication. The supervisor records
the exact pushed prospective design commit SHA in its durable launch ledger,
`experiment-output/phase4d4g-primary/execution/status.json`, before collection.
`results/freeze.json` binds all original execution-source files, every new
ablation Python source, design, fixtures and runtime. No original package source
or strict validation is changed; supplemental enforcement extends the freeze to
the research tooling. Subsequent execution changes require a new documented fork.

Both generation-20 boundaries come from archived summary manifests, validated
by the pinned historical checkout's unchanged strict reader. Trusted parent SHA:

| Seed | Parent boundary SHA-256 |
|---|---|
| 42 | 2866722b5e55726b75ae27d4bdb70e98d9b01b9ca503b0925fe9aa8b17d210b1 |
| 314159 | 328022e5f4c45c11650b7fd2b76fbafbbe4734f1161a75d91b5d0b4cb274f9d6 |

Readiness records four strict generation-20 forks, matched computational-state
identities per seed, model/optimizer/replay/RNG/counter hashes, full config and
runtime. Binary boundaries remain in the new ignored output tree; absolute paths
and file hashes are retained in tracked readiness and later completion sidecars.

## Training and evaluation scope

Four arms: seeds42/314159 x control/treatment. Each runs generation21 and22:
64 complete games,256 simulations/move,pool8,64 fixed AdamW updates,batch128.
Totals512 primary games/updates. Control tau1 plies0–7 then0; treatment tau1
plies0–15 then0. All other historical CNN, optimizer, noise, augmentation,
replay, search and game RNG settings remain matched. No ResNet, proof search,
replay sampling changes, checkpoint promotion or main merge.

Frozen development suites: solved300 with complete exact legal-action WDL
values, tactical200,128 extra full-vector4E rows, and32 opening base families.
Union clustering protects mirrors and shared opening ancestry:102 solved,
71 tactical and28 additional deep clusters. Unknown generator trajectory
identities and reused development inputs limit generalization.

Each parent is evaluated once per seed and reused for the initially identical
arms. Every generation21/22 checkpoint receives raw policy and raw PUCT512
evaluations with independent, arm-matched streams, no root noise or tactical
guard. Generation22 strength uses64 games/arm against its own frozen parent,
32 openings x both colors,256 matches total. Policy CE/accuracy/value MSE,
optimal-set mass, calibration, tactical/deep errors and strength remain separate.
10,000 seeded cluster-bootstrap replicates use paired data and equal seed
weight; one-sided bounds differ from two-sided intervals. Exact criteria and
FAIL versus INCONCLUSIVE interpretation are frozen in DESIGN.md.

## Managed execution, integrity and resources

One complete generation per child, fresh strict load, SHA-verified atomic save
and complete state reload comparison. Alternating order is prospective. Attempt
ceiling20min; total primary ceiling3h including evaluations. The supervisor
owns subprocess-group deadlines and cooperative cancellation. It retains
failure logs and partial counts, stops without automatic retries and rejects
duplicate launch. Interrupted generations are never completed data. Failed
progress counters are lower bounds, with64-game/64-update attempt upper bounds.
Thirty seconds of the total budget are reserved for final preservation/logging.

Each completed generation exports compact complete games, histories and visits,
per-game RNG identities, all state hashes, training/search/replay metrics,
sampling exposure and resource sidecars. Positions/time are measured rather
than forced equal. Historical replay starts with68,856 examples/4,957 steps
(42) and64,514 examples/4,890 steps(314159),2,048 games across generations13–20.
The report will quantify actual new-position share and optimizer sampling
exposure; a short diluted pilot cannot establish exploration's irrelevance.

Preparation includes bounded search/optimization throughput probes, zero
primary games/updates and independent disposable scratch updates. Measured
readiness and launch state are recorded below after validation. Final primary
CPU/wall/RSS accounting remains pending.

Readiness completed in132.656s, under its300s ceiling. It used48 independent
batched256 search roots,24 raw512 probes and8 disposable128-example AdamW
updates, with **zero primary collection or optimization**. Both parent pairs
strictly roundtrip and match initial computational state:

| Seed | Matched initial-state SHA-256 |
|---|---|
| 42 | 2dd15035fd4e10654a80a78cf5dec16425d4ad6e23f21323adcc219ca8bee5e4 |
| 314159 | 5aa9916bc689710aa8ed4e6a5d268cb94d913cb8375e5d14542280587be29c77 |

Conservative projected seconds: collection858.095, optimization24.748,
evaluation3357.662; twofold contingency plus600s I/O gives9081.010s
(151.35min), below10,800s. This is feasibility evidence, not guaranteed duration.
Actual execution will still stop at its declared ceilings.

Frozen environment: AppleM4 CPU, CPython3.11.17, Torch2.10.0, NumPy1.26.4,
one intra/inter-op thread, all five library thread controls one, deterministic
algorithms enabled. Full runtime fields and Torch build digest remain in
freeze/readiness and every boundary. Original enforced training source digest
remains `b8aae0d6e14292dad2d2bcdb86f6048fad93a5eeaa5d342d7b5c340e9fc2bde5`.
Frozen fixture SHA-256:
`80f6851838b18172d6f818ebd306ce0cd36e4796cac1e3a71df29694feb3ca2c`.
Readiness inventories each contain150 entries and equal the previous4F evidence.

## Validation and independent review

New ablation checks:6 passed in1.99s. Experimental boundary wrapper plus initial
ablation checks:6 outer tests passed in44.02s; wrapper executes56 additional
strict-fork/resume/interruption cases. Relevant neural/evaluation/bitboard/
batching/exploration regressions:341 passed in135.71s.

One preparation command referenced nonexistent test filenames and collected no
tests; the corrected suite above passed. The first readiness invocation failed
at import because the new compact-game exporter imported `game_record` from
the historical generation module rather than experimental.py. It created no
fork, collected no games and made no updates. The failure log remains tracked,
the import was corrected before source freeze, and an entrypoint import check
was added. No training failure or unfavorable observation was discarded.

A separate reviewer inspected protocol, new sources and fixture/statistics
contracts, independently tested the owned watchdog including SIGKILL escalation,
and validated exact vectors, mirrors and clusters. Findings on clustering,
supervisor termination, JSON durability, calibration/one-ply reporting and
partial-count interpretation were resolved. See
[the independent review](../research/alphazero_training_ablation/independent-review.md).
This is review on the same host, not external replication. The final experimental
evidence and interpretation require independent postexecution review.

All150 official files are verified byte-identical to the4F inventory. Declaration
SHA remains `c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716`.
The historical checkout, original runner/checkpoints/boundaries, previous reports,
canonical benchmark, public application/API/deployment and unrelated worktrees
remain unchanged. Only new ablation research/test/report files are added.

## Status and separate-session completion

| Conclusion | Current status |
|---|---|
| Experimental execution | PENDING; no completed primary result claimed |
| Neural learning per seed/arm | PENDING |
| Tactical/deep reliability | PENDING |
| Playing strength | PENDING |
| Longer exploration recommendation | Deferred until evidence exists |

After managed execution ends, the separate final-analysis session must inspect
all frozen sources/checkpoint identities, completed games/boundaries and150
official files; run `analyze` and `publish_evidence`; re-review uncertainty,
failure/recovery/resource accounting; finish these four decisions and per-seed
tables; recommend one evidence-supported next phase; review diff and commit/push
compact evidence on this dedicated branch. Do not automatically add generations
or a tournament. A small ResNet remains a later architecture experiment.

Progress/recovery commands and local paths are in
[README.md](../research/alphazero_training_ablation/README.md).
