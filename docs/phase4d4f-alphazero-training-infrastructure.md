# Phase 4D.4F — experimental AlphaZero training infrastructure

The experimental runner supports validated generation-20 forks, pool-eight
collection, mixed-schedule replay and strict generation-boundary recovery. It
preserves the historical runner and campaign contract. Both development seeds
are ready for a separately approved exploration/training pilot. This phase
provides engineering evidence, not evidence that either schedule learns better.

## Implementation and guarantees

Development starts at `a1639463ef5d66f0f12ab090674d5b861612a11c`, in the separate
`phase4d4f-alphazero-training-infrastructure` worktree/branch. Only new experimental
modules, tests and research/documentation files are added. The Phase 4D.4A–4D.4E
reports and the requested eight implementation modules were reviewed. Existing
launch-control, campaign, source-closure and resume tests remain unchanged.

`experimental.py` provides a small lifecycle around existing model, trainer,
reflection augmentation, replay, diagnostics and atomic artifact utilities.
Collection uses the existing `collect_batched`, capacity eight, and bitboard
PUCT. The model is copied, frozen and checked before training; all completed
examples are reconstructed and compared before replay insertion. Optimization
uses an explicit fixed update count, persistent AdamW and independent replay/
reflection streams. A generation succeeds only after atomic boundary persistence.

The training configuration retains the original optimizer, batch and reflection
settings. Its existing `samples_per_new_position` value is preserved as historical
configuration but the experimental update budget explicitly supersedes the
historical position-dependent `updates_for` rule. Configured `max_generations=22`
permits two new fork generations after historical ID 20. IDs 21/22 denote matched
new experimental games, never official continuations.

`experimental_models.py` is an explicit CNN adapter for construction, architecture/
output identity, model state validation, hashing, freezing, inference, trainer
construction and optimizer restoration. Unsupported IDs are rejected. A future
ResNet adapter can reuse collection/replay/lifecycle/evaluation while supplying
its own model and optimizer/output contracts. No ResNet or new search mathematics
was implemented.

### Historical fork versus experimental continuation

`experimental_import.py` invokes a read-only subprocess importing the unchanged
historical checkout at `1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`, with bytecode writes
disabled. That reader calls the original `load_resume_boundary` with its default
**strict source and strict runtime checks**, the expected manifest file hash and
no global-RNG installation. It does not run the historical generation lifecycle.
No non-strict loading or altered source manifest is used.

The original reader verifies architecture, weights/state digest, persistent
optimizer, trainer counters, legal complete replay histories, historical eight-ply
temperature schedule, source identity and every runtime field. Explicit additional
checks require the designated seed, generation 20, generations 13–20, 2,048 games
and 325,896 parameters. Conversion again verifies complete ordered AdamW groups,
all options, finite moment shapes/types, steps, network hash and replay digest.

Each arm owns independent deep copies of the verified model, optimizer, replay,
training counters, sampling/reflection state and process-global Python/NumPy/Torch
state. Parent file/state/weights/replay hashes, source/runtime/configuration,
original history/RNGs, conversion method, changed budgets and a **fork guarantee**
are retained. Historical search/action/noise states are archived as provenance;
new games use the Phase 4D.4C independent-game protocol. They do not reproduce
historical sequential-RNG trajectories.

The matched initial-state hash covers all computational state, parent provenance
and training configuration, excluding only arm name, prospective current schedule,
Git reporting and measured resources. Full arm artifact hashes differ because
arm names and schedule declarations differ. Equality means identical initialization;
it does not mean equal games or learned weights after schedules diverge.

### Mixed schedules, recovery and accounting

Every retained generation stores its full canonical `TemperatureSchedule`, visit
budget, RNG protocol and game count. Historical records use a verified legacy
schedule derived from the parent configuration. New records use their declared
schedule. Loading verifies legal moves, terminal winners, visit accounting,
zero-temperature maximum actions, visited legal actions, normalized visit targets
and actor-relative final labels. Replay content and contiguous window identities
are checked; original generations expire as whole generations. Neither missing
metadata nor a global reinterpretation of old temperatures is permitted.

Experimental boundary version 1 includes model/optimizer/config, counters,
complete replay, schedules, all persistent RNG state, augmentation counts,
historical parent, experimental identity, runtime/source content, deterministic
summaries, resource records and content integrity. Ordered optimizer compatibility,
retained position counts and cumulative trainer/model history are checked.
Strict experimental load has no relaxed runtime/source switches. A supplied
trusted file SHA also detects byte corruption; hashes are not signatures against
an attacker capable of changing both artifacts and manifests.

The original atomic primitive writes/fsyncs a sibling temporary file, strictly
reloads it, publishes a no-replace hard link and fsyncs the directory. No boundary
is published before complete collection/validation/replay/optimization. Failures
poison the attempted runner and require reload. A directory-fsync error after
publication can leave a complete validated file with uncertain crash durability;
inspect/reload it before selecting it. This is tested and documented. Cooperative
checks run between batches/simulations, updates and validation. No mid-search
checkpointing is introduced. Campaign-ancestor/symlink output guards prevent
experimental publication under an existing campaign declaration.

`resources` persists finite collection/training/wall accounting before checkpoint
persistence; returned/sidecar timing includes persistence completion. Timings are
separated from deterministic history/computational equality. Position counts and
wall time can differ by schedule. Primary matching is **games and optimizer
updates**. A later compute-matched sensitivity analysis requires its own declaration.
`evaluate_boundary` supplies an independent frozen model to a read-only evaluation
hook and restores global RNG state; it neither selects an arm nor promotes a model.

## Exact validation results

The focused suite has **56 passing cases** (independent run: 43.99 seconds;
post-isolation run: 40.96 seconds). The wrapper plus the historical fresh-process
regression passed together: **2 tests in 44.07 seconds**. An initial combined
regression run reported 663 passes, two skips and one failure: the new fixture
changed native Torch inter-op/deterministic settings in the shared test process,
so the historical child correctly rejected that different runtime. Moving the
56 cases to a dedicated child interpreter resolves the interference without
altering any historical test or loader. The original failure log is retained.

The final complete AlphaZero sweep passed: **609 passed, 2 skipped in
337.36 seconds**. That includes the isolation wrapper, which separately executes
all 56 focused cases. The existing neural-MCTS, import-isolation, MCTS, MCTS API
and engine dependency checks passed: **307 passed in 3.54 seconds**. The two
AlphaZero skips are the existing optional retained-v1-checkpoint checks.
CLI inspection of both seeds and scratch one-generation launch also passed.

Final test and real-parent validation records are in
`research/alphazero_training/results/`. The focused suite compares full game records,
replay digests, network byte hashes, ordered optimizer state, counters, sampling/
reflection/global RNG states and deterministic generation summaries. Resource
measurements and archive byte hashes are deliberately excluded from deterministic
trajectory equality.

The tests inject failures during batched search, game finalization, collection
validation, replay insertion, optimization, serialization, boundary validation
and fsync. Reloaded/retried scratch generations match uninterrupted execution,
including in a fresh interpreter. Adversarial tests cover replay temperature,
schedule, legal move, visit budget, outcome, missing metadata, window/protocol,
optimizer options/moments/steps, architecture, runtime/source, model/checkpoint
corruption, RNGs, augmentation and jointly changed history counters. Atomic
no-overwrite and campaign/symlink isolation are tested.

The one-generation disposable smoke uses a fresh random CNN, seed 44046,
eight games, two simulations/decision, batch two and exactly one optimizer update.
It collects 116 positions and reloads identical state. Collection took
0.08114 seconds; total collection/training/persistence took 0.29636 seconds.
The CLI independently inspected both real fork boundaries and repeated the same
eight scratch game identities from the generation-zero fixture; its weights and
replay also matched exactly. That repeat took 0.29409 seconds, bringing total
new measured scratch collection/training/persistence to **0.59044 seconds**.
These tiny-budget timings validate execution only and are not throughput claims.
No archived trained model receives an optimizer update. Ordinary bounded scratch
regression tests are separate from that measured smoke; no 64/256-game training
generation, real ablation or playing-strength tournament was launched.

The real read-only readiness screen validates both original generation-20 files
against their archived `summary.json` file hashes, converts both matched arms,
atomically saves each fork and strictly reloads each. It verifies model, full
optimizer, counters and replay equal the parent before any collection. Seed 42
retains 68,856 positions and trainer step 4,957; seed 314159 retains 64,514 positions
and trainer step 4,890. Each retains 2,048 games across generations 13–20. Both
arms in each seed have identical initial-state identities. Readiness collection
and optimizer-update counts are **zero**.

## Preservation and provenance

All 150 official campaign files were hashed before and after this phase and
compared to the previous Phase 4D.4E inventory. The declaration hash remains
`c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716`.
The historical checkout, eight historical implementation files and all prior
research evidence remain unchanged. The canonical benchmark, public deployments,
API services, main and unrelated worktrees were not modified. Official sealed
artifacts were hashed as bytes only; they were not used as new evaluation inputs.

The environment is the existing
`/Users/andrewcukierwar/.venvs/board-game-alphazero-m4`: Apple M4 CPU, CPython
3.11.17, Torch 2.10.0 and NumPy 1.26.4. All five library thread variables are one;
intra/inter-op threads are one and deterministic algorithms are enabled. Full
runtime fields and the complete execution file map/digest are stored in readiness,
smoke and each boundary. The final enforced execution digest is
`b8aae0d6e14292dad2d2bcdb86f6048fad93a5eeaa5d342d7b5c340e9fc2bde5`.
Git provenance at validation records the starting commit
and newly authored files; execution-content identity is the enforced guarantee,
and committing documentation does not change it.

Prior external methodology informs the design: the pinned eugeneyp coach saves
model/optimizer and batches independent games; kuotunyu saves model/optimizer/
replay/history with fallback; AlphaZero.jl documents saved-iteration resume and
inference batching. Our complete RNG/schedule/integrity and interrupted-resume
checks extend what those reviewed sources establish. Their saved checkpoints do
not prove our required exact stochastic resume. No external code or performance
claims were adopted. See the previous audit's
`research/alphazero_baseline/external_review.json` and
`research/alphazero_batching/external_sources.json` for pinned primary-source URLs
and hashes.

## Independent critical review

A separate reviewer inspected the implementation, historical import, tests,
research reports and real-parent readiness evidence. Four findings were resolved:
assertion-independent reader eligibility, adapter-owned optimizer restoration,
durable resource accounting and semantic history-counter/model-chain validation.
The reviewer independently ran all 56 focused tests successfully (43.99 seconds)
and found no unresolved material issue. This is independent review on this host,
not replication by an external laboratory. The full review is preserved in
`research/alphazero_training/independent-review.md`.

## Requirement status

| User requirement | Status | Evidence |
|---|---|---|
| 1. Repository/research context and isolated branch | PASS | Exact starting revision; new worktree; all reports/modules reviewed |
| 2. Separate experimental system; no disabled checks | PASS | New format/lifecycle; original strict reader subprocess |
| 3. Immutable official campaign and unrelated assets | PASS | 150-file before/after/previous inventories; only added files |
| 4. Complete pool-eight collection/training/persistence lifecycle | PASS | Frozen scratch smoke and regression tests |
| 5. Safe seed-42/314159 warm starts and identical independent arms | PASS | Both real parent files, full states, four fork reloads |
| 6. Explicit original/extended and mixed-generation replay schedules | PASS | Per-generation metadata, adversarial reconstruction, expiration tests |
| 7. Reproducible independent-game and persistent training RNGs | PASS | Matched identities; sampling/reflection/global/Torch restoration |
| 8. Versioned atomic boundary with full provenance and strict resume | PASS | Schema and strict integrity/config/runtime/source/model/optimizer tests |
| 9. Collection/finalization/replay/update/serialization/validation interruptions | PASS | Failure matrix, poisoned runners, deterministic retry, no partial publication |
| 10. Fair two-arm support and measured unequal positions/time | PASS | Games/update budgets; independent directories; resources; evaluation hook |
| 11. Future architecture interface, CNN only | PASS | Factory owns all model/optimizer contracts; unsupported IDs rejected |
| 12. All requested validation cases and existing regressions | PASS | Focused suite plus final regression record; fresh-process equality |
| 13. Bounded engineering scope | PASS | Two tiny one-generation scratch probes; no actual training experiment |
| 14. Runner/schema/import/tests/review/provenance/docs and dedicated push | PASS | Package, phase report, commit/push on dedicated branch |
| 15. Definition of done and proposed next-phase decision | PASS | Both seeds ready; gates/design below; launch awaits approval |

## Proposed Phase 4D.4G — approve separately before launch

1. **Prospective declaration.** Commit the exact experiment config, parent hashes,
   initial state equality records, source/runtime identity, development evaluation
   inputs/family hashes, evaluation RNG seeds, endpoints and decision rules before
   collection. Use fresh independent output directories. Never use sealed official
   inputs or resume the official campaign.
2. **Primary pilot.** Both seeds 42/314159, control and treatment, two complete
   generations per arm (fork IDs 21 and 22): **64 games/generation**, pool eight,
   256 simulations/decision and **64 AdamW updates/generation**, batch128. This is
   512 new games and 512 optimizer updates in total. Retain the eight-generation
   window, historical replay initialization, noise .25/alpha1, LR3e-4, weight decay
   1e-4, clip5 and reflection .5. Control has tau1 at plies0–7 then0; treatment
   tau1 at plies0–15 then0. Match seed/generation/game identities and initialize
   both arms from the same verified parent. Record positions, root simulations,
   inference batches, collection/training/evaluation/persistence CPU/wall time and
   replay historical/new sampling shares. They are measured, not assumed equal.
3. **Bounded execution/recovery.** Run one arm generation at a time, alternating
   arm order by seed/generation. Stop at every persisted boundary. Each attempt
   has a 20-minute wall ceiling enforced cooperatively; the entire primary run
   has a three-hour wall ceiling including evaluation. Exceeding either ends the
   attempt/pilot without counting partial generations; any budget expansion needs
   a new declaration. No automatic winning-arm selection or deployment.
4. **Learning evaluations at parent, ID21 and ID22.** Freeze/hash the same existing
   nonsealed development solved300 and tactical200 suites prospectively. Analyze
   their mirrored/trajectory families as clusters; these are reused development
   diagnostics, not unseen generalization or 300 independent seeds. Primary
   network-only learning endpoint: optimal-action-set policy CE
   `-log(sum policy[a] for exact-optimal a)` on solved300. At ID22, a promising
   learning signal requires mean treatment-minus-control CE <= **-0.02 nats**,
   with each seed <=0; mean raw optimal-action accuracy >= **+2 percentage points**,
   each seed >=0; neither seed value-MSE regression may exceed **+0.02**. Report
   paired family-bootstrap 95% intervals with 10,000 fixed-seed draws and all
   per-seed changes. If any improvement interval includes zero, label the pilot
   signal preliminary, not confirmed. Training loss alone never passes this gate.
5. **Separate tactical gates.** At both new checkpoints, evaluate raw neural
   policy and raw PUCT (512 simulations, no noise or tactical guard) on tactical200.
   For each seed/method/category, immediate-win and safe-response accuracy may
   decline by at most **2 pp** versus both its parent and matched control. On the
   fixed development solved300 oracle table and a prospectively frozen subset
   of Phase 4D.4E position families with **complete legal-action exact values**,
   record deeper oracle move-value losses separately; a tactical noninferiority
   pass additionally requires each seed's upper one-sided 95% paired family
   confidence bound for treatment-minus-control deep mistake rate <= **+5 pp**.
   Freeze the complete action-value proof table before training; existing
   selected-action proofs alone are insufficient for new networks' arbitrary
   actions. Require **100% exact oracle coverage** for included positions, with omissions
   or inadequate confidence reported as **INCONCLUSIVE**, never safety. Do not
   adapt position selection, search budgets or thresholds after seeing outcomes.
6. **Strength support at ID22 only.** Use 32 prospectively frozen nonsealed
   development opening families, both colors (64 games/checkpoint), raw512 PUCT
   against its own frozen parent. Control and treatment face identical parent,
   openings/color and evaluation identities. Report score and paired family
   intervals; a supporting positive signal requires treatment score >50% in both
   seeds and an equal-seed-weight pooled lower95% bound >=50%. Prospectively
   define the separate **strength regression gate**: for each seed, the one-sided
   95% paired opening-family lower bound on treatment-minus-control parent-match
   score must be >= **-5 pp**. If that lower bound misses the margin while the
   upper bound overlaps it, report INCONCLUSIVE; an upper one-sided95% bound
   below -5 pp is FAIL. Bootstrap the same paired32 families within each seed,
   10,000 fixed-seed draws. This is separate from the learning and tactical gates,
   with no model promotion. It can remain inconclusive in a small
   pilot. No canonical benchmark changes or broad tournament.
7. **Decision after the pilot.** A further, larger confirmation is recommended
   only if learning and tactical gates pass, the prospectively defined strength
   regression gate passes, both seeds complete the matched budgets and provenance/recovery
   checks pass. Otherwise report FAIL/INCONCLUSIVE and inspect the mechanism
   before extending training. A compute-matched sensitivity analysis, replay/
   optimization changes, ResNet and tactical search changes each require a
   separate prospective design; none is bundled into the primary schedule test.

**Recommendation:** launch the bounded Phase 4D.4G pilot after explicit approval
and its prospective declaration. The infrastructure is ready; the scientific
benefit and tactical acceptability of extended exploration remain unproved.
This phase does not launch that pilot.
