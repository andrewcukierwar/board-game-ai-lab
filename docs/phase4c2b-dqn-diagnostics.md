# Phase 4C.2b — DQN diagnostics and scaled-experiment preparation

**Status: stopped at the verification gate; no scaled training experiment was launched.** The first test run failed one newly added inference-summary assertion. The assertion was corrected, and all 174 DQN tests now pass. The user's instruction was: “If the new fixture validation, inference or evaluation tests fail, stop without launching training.” That stop condition was honored even after fixing the test. There is no new trained candidate or three-model experimental comparison to report.

Work was performed October 3, 2026 (America/New_York), on local branch `phase4c2b-dqn-diagnostics`. No commits, pushes, deployments, paid API calls, or changes to production dependencies, hosting, credentials, API/UI, neural MCTS, or VictorAgent were made.

## Preflight and preservation

Fetched `origin/main`; it exactly matched the supplied reference `43e656dfd2f7e0df8402101afb351b8c54329d35`. The initial working tree was clean. Created the requested branch from that commit. Read the Phase 4C.1 and 4C.2 handoffs and existing implementation/tests.

The original `experiment-output/phase4c2-seed42/` was left untouched. All eight files, including its checkpoint, configuration, full logs, report, patch and source snapshots, were copied into the new Git-ignored directory:

```text
experiment-output/phase4c2b-preservation-20261003/
```

The copy also includes the Phase 4C.2 Markdown handoff and `original-manifest.json`. Every original and copied file was checked against the manifest after implementation. The archived candidate's SHA-256 remains:

```text
1be2836c252575e65ce6c46891dcca1ab5b426e08bffb552cee0b3ec570267a9
```

It contains inference weights/provenance only. No exact training continuation is possible from this artifact, and none was attempted.

## Frozen evaluation suite

[diagnostic_positions.json](../games/connect4/dqn/diagnostic_positions.json) freezes 60 explicit legal move sequences. Tactical candidates were discovered using seeded legal random trajectories (construction seed 42002), without consulting any neural network, Q-value, or diagnostic prediction. Accepted positions and mirrors are literal data; future evaluation does not regenerate them. The fixture file's version is 1 and SHA-256 is:

```text
ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a
```

| Coverage | Positions |
|---|---:|
| Unique immediate wins | 24 |
| Unique defenses against an immediate opposing win | 24 |
| Opening probes | 4 |
| Middlegame probes | 4 |
| Explicit full-column probes | 4 |
| X / O to move | 30 / 30 |
| Original / mirrored positions | 30 / 30 |
| Horizontal / vertical / diagonal tactics | 16 / 16 / 16 |

Each of the 12 tactical strata (win/block × horizontal/vertical/diagonal × X/O) has four fixtures. Mirroring covers both diagonal slopes. Tactical correct-action counts for physical zero-based columns 0 through 6 are **[4, 3, 12, 10, 12, 3, 4]**. Coverage spans every column, including both edges, but is not column-balanced. The 60 rows include correlated mirror pairs, so they are not 60 independent observations. Several tactical fixtures incidentally have full columns in addition to the four explicit masking probes.

Every sequence is replayed from the empty engine board with checks for valid moves and premature termination. Runtime validation enumerates all legal actions using the actual engine. Win fixtures require exactly one immediate winning action. Block fixtures require no own immediate win, exactly one opposing immediate threat, and exactly one move after which the opponent has no immediate winning reply. A separate four-in-a-row scan verifies the labeled orientation. Tests independently enumerate engine moves/replies, verify actual mirrored boards and physical action indices, and pin the suite hash.

“Safe block” means safe against the **next reply**, not a proof that the entire game is drawn or won. There are no labels for longer forced sequences, strategic opening quality, or middlegame strength. Non-tactical probes have no asserted best move. Labels are never passed to replay or optimization; no supervised targets, symmetry augmentation, reward shaping, architecture changes, or replay sampling changes were introduced.

## Diagnostics and evaluation implementation

[diagnostics.py](../games/connect4/dqn/diagnostics.py) records all seven raw Q-values, the legal mask, greedy action, legal Q minimum/maximum, and tactical correctness. Tactical margin is `Q(correct) - max(Q(other legal action))`; illegal columns cannot affect it. Summaries include accuracy and margin extrema/means by win/block category, orientation, player, and expected column.

Greedy action summaries include all seven counts, dominant column/share, and entropy in bits. Each mirror pair records whether actions transform as `c → 6-c`, the mean absolute difference between aligned legal Q-values, and the maximum absolute difference. Lowest-physical-column tie breaking remains unchanged; ties can themselves break mirror action agreement and should be considered when interpreting it.

Random evaluation now permits **100 games per model**, alternating sides for 50 X and 50 O games, and predeclares the full `(index, opponent seed, side)` protocol. With experiment seed 42, opponent seeds are 42 through 141. Each game gets its own private RNG. Initial, archived and final models receive the same protocol. Different model actions still induce different trajectories despite matching seeds. Reports include transcripts, aggregate and side-specific wins/losses/draws, and greedy action distributions. Partial games are excluded from outcome totals; their actions are explicitly included in action distributions. Per-model evaluation remains bounded to at most 60 seconds.

The driver keeps the existing `--evaluation-games` meaning of combined initial + final games: `200` requests 100 each. An optional `--archived-checkpoint` adds another 100, separately capped at 60 seconds. Configuration records the additional archive budget and exact protocol. A missing archive is explicitly reported as unavailable. Evaluation is inference-only and never consumes exploration or replay RNG state. No stronger-opponent evaluation was added.

[training.py](../games/connect4/dqn/training.py) adds only a read-only replay-composition snapshot: terminal/nonterminal counts, reward counts, and actor counts. It consumes no RNG and changes no sampling. The driver logs retained replay composition per completed episode and at exit, plus cumulative collected terminal/nonterminal and reward counts. These distinguish all collected experience from the bounded replay window; they do not claim to measure actual sampled minibatch frequencies. Existing per-ply reward/terminal/epsilon logs remain complete. Training-time actions include epsilon exploration and must not be interpreted as greedy policy diversity.

[experiment.py](../games/connect4/dqn/experiment.py) validates fixtures before creating output or training, supports archive evaluation, attaches summaries and suite identity, and retains the existing output-overwrite refusal, source snapshot, bounded loop, numerical checks, and exact checkpoint weight/prediction/metadata reload checks. Checkpoint contract v1, canonical encoding, one-ply signed Bellman target, optimizer, and target synchronization are unchanged. Report schema is now version 2.

## Verification and stop condition

Initial gate results:

- DQN suite: **173 passed, 1 failed** in 3.70 seconds.
- Backend suite: **353 passed, 3 optional DQN modules skipped** in 6.38 seconds, including import isolation and the live-HTTPS prohibition.
- `git diff --check`: passed.

The failed test, `test_legal_q_margin_mirror_and_concentration_metrics`, assumed that only the explicit full-column probes could mask column 0. Other legal tactical fixtures also contain full columns. The synthetic fixed-Q policy correctly respected those masks, so its action histogram differed from the hardcoded test expectation. No tactical label or production inference defect was found. The test now calculates its expected action for each fixture from engine availability and the synthetic policy's known column preference, then independently checks aggregated counts and mirror agreement.

After that correction:

```sh
/tmp/board-game-phase4c1-venv/bin/python -m pytest -q \
  tests/test_connect4_dqn.py tests/test_dqn_experiment.py tests/test_dqn_diagnostics.py
# 174 passed in 3.46 seconds

.venv/bin/python -m pytest -q tests
# 353 passed, 3 skipped in 6.38 seconds (initial gate; backend code unchanged)

git diff --check
# Passed
```

The new tests cover all 60 fixtures, frozen identity/coverage, rejected labels before training, legal Q/margin and mirror summaries, 100-game deterministic balanced evaluation, invalid budgets, replay eviction/composition/RNG preservation, and three-model orchestration with training explicitly stubbed out. Existing tiny synthetic optimizer tests were also run; these are correctness checks, not self-play training experiments. Test checkpoint round trips use temporary synthetic/untrained artifacts.

The initial failed verification gate remains decisive: **zero scaled training invocations, zero scaled-run self-play games/plies/updates, and no new trained checkpoint**. The fixed initial/archived experimental evaluation was not launched separately. Local verification records live in `experiment-output/phase4c2b-verification-20261003/`, separate from the preserved original experiment.

## Experiment configuration prepared, not executed

The requested run remains pending review following the failed gate. Its intended configuration is seed 42; fresh initialization; CPU/float32; one intra-op and one inter-op thread; deterministic algorithms; limits of 5,000 completed games, 110,000 collected plies, 110,000 optimizer updates, or 300 training seconds, whichever occurs first. All Phase 4C.1 defaults are retained except epsilon start 1.0, floor 0.10, decay 0.99997 per successful update. Replay capacity is 10,000, batch size 64, gamma 0.99, Adam learning rate 0.001, and hard target synchronization every 100 updates. Existing CLI defaults remain conservative; these larger bounds and slower decay must be supplied explicitly for the pending run.

No command launching this experiment was executed. No automatic retry, extension, sweep, or resumption occurred. A subsequent authorized run must use a new unique Git-ignored directory, save one versioned inference candidate, verify safe-loader weight/prediction equality, and record its SHA-256, source identity, configuration and actual budget.

## Comparison and conclusions available now

The original Phase 4C.2 measurements are preserved, not replaced by new evidence:

| Measurement | Archived Phase 4C.2 | Phase 4C.2b scaled run |
|---|---:|---|
| Completed self-play games | 200 | Not launched |
| Plies / optimizer updates | 3,779 / 3,716 | Not launched |
| Training seconds | 4.078934833 | Not measured |
| Original tactical suite | 0/4 initial and final | Not evaluated experimentally |
| Original Random suite | Initial 8/12; candidate 11/12 | No new comparison |
| Original greedy concentration | Column 4 on 9/10 final probes | Not measured |
| New 60-position suite / 100-game Random protocol | No experimental evaluation yet | No final candidate |

There is **no new evidence that tactical learning is emerging**. The archived Random improvement alone still does not establish it. Whether more training under the same algorithm is justified remains unanswered by this phase because the larger baseline experiment did not run. The diagnostic work is ready for review, but the test failure triggered the user's explicit stop condition. No evidence here selects symmetry augmentation, replay sampling changes, or Double DQN as the next modification; choosing among them should await the pending comparison. None was implemented.
