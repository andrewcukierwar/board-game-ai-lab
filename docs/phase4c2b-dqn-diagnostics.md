# Phase 4C.2b — DQN diagnostics and scaled learning experiment

**Completed exactly one fresh scaled run on October 3, 2026 (America/New_York).** It stopped at 5,000 games in 118.695 training seconds without an invariant failure. The result is **better Random performance than the untrained network, with insufficient tactical learning**: the final model won 75/100 against Random, equal to the archived 200-game model, and solved only 15/48 tactical fixtures. Policy diversity improved, but diagonal accuracy and Q-value symmetry regressed. No further experiment or algorithm modification was launched.

## Preflight, authorization, and verification

The user reviewed the earlier corrected test assertion and explicitly authorized the pending experiment. Fetched `origin/main`; both the existing local branch `phase4c2b-dqn-diagnostics` and origin were already at `0e4004bef0bd6387da8b2bcb189784cb92dd3699`. The tree was clean, so no reconciliation edits, merge, or local-change replacement was needed. No implementation or fixture changes were made during this resumed task. The only tracked change afterward is this handoff.

Confirmed that the corrected action-histogram test derives expected actions from `is_valid_move` on each actual engine position. The final pre-training gate passed:

```sh
/tmp/board-game-phase4c1-venv/bin/python -m pytest -q \
  tests/test_connect4_dqn.py tests/test_dqn_experiment.py tests/test_dqn_diagnostics.py
# 174 passed in 4.66 seconds

.venv/bin/python -m pytest -q tests
# 353 passed, 3 optional DQN modules skipped in 6.83 seconds
# Includes import isolation and the existing prohibition on live HTTPS calls.

git diff --check
# Passed before training and after the report update.
```

The earlier gate had stopped on a test assumption that only explicit full-column probes could mask column 0. Tactical fixtures also contain full columns. That assertion was corrected and reviewed before this renewed authorization. The earlier handoff is preserved in `pre-run-handoff.md` beside the new artifacts; the historical failure was not a failure of this final gate.

All eight original files under `experiment-output/phase4c2-seed42/` and their copies under `experiment-output/phase4c2b-preservation-20261003/` still match the preservation manifest. The archived candidate remains available with SHA-256:

```text
1be2836c252575e65ce6c46891dcca1ab5b426e08bffb552cee0b3ec570267a9
```

It was used for inference comparison only. The new trainer started from scratch; neither checkpoint is an exact training-resume artifact.

## Unchanged frozen suite and measurement definitions

The existing [60-position suite](../games/connect4/dqn/diagnostic_positions.json) remains version 1 with unchanged SHA-256:

```text
ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a
```

It contains 24 unique immediate wins, 24 uniquely safe immediate blocks, four opening probes, four middlegame probes, and four explicit full-column probes. There are 30 X and 30 O positions and 30 original/mirror pairs. Each win/block × horizontal/vertical/diagonal × player stratum contains four tactical fixtures. Both diagonal slopes and all seven physical action columns are covered. Expected-column counts for columns 0–6 are `[4, 3, 12, 10, 12, 3, 4]`, so column coverage is not balanced.

Sequences are frozen legal engine replays, originally constructed without model predictions. Win labels require a unique immediate winning action; block labels require no immediate own win and a uniquely safe response against the opponent's next winning reply. “Safe” is a one-reply property, not a solved-game claim. Tests independently verify labels, mirrored boards/physical columns, and masks. Opening/middlegame probes have no purported optimal action. No labels entered training or model selection.

The already implemented [diagnostics](../games/connect4/dqn/diagnostics.py) retain raw Q-vectors, legal masks, selected actions, legal Q ranges, category/player/column summaries, action concentration, and mirror comparisons. Tactical margin is `Q(correct) - max(Q(other legal action))`; positive margins favor the expected move. Mirror Q differences align column `c` with physical column `6-c` and use legal actions only. Deterministic lowest-column tie breaking can itself break mirror action agreement. Replay summaries are read-only terminal/nonterminal, reward, and actor counts; they do not measure sampled minibatch frequencies or alter sampling.

## Actual invocation and provenance

The following command was executed **once**. It is an execution record, not authorization to repeat it:

```sh
/tmp/board-game-phase4c1-venv/bin/python -m games.connect4.dqn.experiment \
  --output experiment-output/phase4c2b-seed42-20261003-2059 \
  --seed 42 \
  --max-games 5000 --max-plies 110000 --max-updates 110000 --max-seconds 300 \
  --epsilon-start 1.0 --epsilon-min 0.10 --epsilon-decay 0.99997 \
  --threads 1 --interop-threads 1 \
  --evaluation-games 200 --evaluation-seconds 120 \
  --archived-checkpoint experiment-output/phase4c2-seed42/candidate.pt
```

Console output was redirected to the new ignored gate directory and copied into the final output directory. The driver refuses an existing output directory or candidate path. `--evaluation-games 200` means 100 initial + 100 final games; archive evaluation adds 100 games separately. All 300 completed, with no partial games. No stronger-opponent evaluation was performed.

| Configuration | Value |
|---|---|
| Seed / initialization | 42 / fresh network, target, Adam, replay and RNGs |
| Replay capacity / batch size | 10,000 / 64 |
| Gamma / Adam learning rate | 0.99 / 0.001 |
| Epsilon start / floor / decay per successful update | 1.0 / 0.10 / 0.99997 |
| Hard target synchronization | Every 100 successful updates |
| Limits, first reached stops | 5,000 games; 110,000 plies; 110,000 updates; 300 training seconds |
| Device / dtype / threads | CPU / float32 / 1 intra-op, 1 inter-op |
| Deterministic algorithms | Enabled |
| Runtime | Python 3.11.17, PyTorch 2.10.0, macOS 26.6.2 ARM64 |
| Start | October 3, 2026, 20:59:16 EDT (`2026-10-04T00:59:16.502858+00:00`) |
| Source commit, clean at launch | `0e4004bef0bd6387da8b2bcb189784cb92dd3699` |
| Source identity SHA-256 | `86594df8b37b4c316e1d3675cc982f8002b5bcb9b07043f54fa927fb564682ba` |

The source patch is empty and there were no untracked source files at launch. This report was updated after the experiment. Existing canonical encoding, corrected one-ply signed Bellman target, terminal rewards, legal masking, network architecture, optimizer and checkpoint contract were unchanged. No reward shaping, symmetry augmentation, prioritized replay, or Double DQN was introduced.

## Completed budget and numerical behavior

| Measurement | Actual |
|---|---:|
| Stop reason | `max_games` |
| Started / completed games | 5,000 / 5,000 |
| Partial games | 0 |
| Collected plies | 94,846 |
| Successful optimizer updates | 94,783 |
| Training seconds | 118.694633916 |
| Total driver seconds after trainer setup | 119.913818041 |
| Self-play X wins / O wins / draws | 2,760 / 2,233 / 7 |
| Final epsilon | 0.10 |
| Final replay size | 10,000 |
| Scheduled target synchronizations | 947, at updates 100 through 94,700 |
| Nonfinite invariant failures / artifact errors | 0 / 0 |

The first 63 plies warmed replay; every subsequent ply received one successful update. The time budget covers the training loop and its logging, not initial/final evaluation or checkpoint serialization. Limits are cooperative, checked before collection and optimization; this run finished well below the wall-clock limit.

| Completed games | Updates | Epsilon | Training seconds |
|---:|---:|---:|---:|
| 200 | 4,218 | 0.881137 | 4.415 |
| 1,000 | 19,051 | 0.564656 | 21.638 |
| 2,000 | 36,762 | 0.331915 | 43.519 |
| 3,000 | 55,991 | 0.186420 | 67.982 |
| 4,000 | 75,638 | 0.103398 | 93.604 |
| 5,000 | 94,783 | 0.100000 | 118.693 |

All 94,783 losses were finite. Mean Smooth L1 loss was **0.017256305**, median **0.016192380**, and minimum/maximum **0.000271199 / 0.089825243**. First/last losses were **0.027449012 / 0.014986956**. The first/last 100-update means were **0.004362511 / 0.020817912**.

| Update interval | Mean loss |
|---|---:|
| 1–10,000 | 0.009260157 |
| 10,001–20,000 | 0.017608000 |
| 20,001–30,000 | 0.017203886 |
| 30,001–40,000 | 0.018321145 |
| 40,001–50,000 | 0.017443148 |
| 50,001–60,000 | 0.017169775 |
| 60,001–70,000 | 0.018578266 |
| 70,001–80,000 | 0.017635006 |
| 80,001–90,000 | 0.021138338 |
| 90,001–94,783 | 0.019240473 |

There was no numerical explosion, but no sustained loss reduction. Replay contents, exploration and targets change over time, so this is not a fixed validation loss. Pre-action training Q-values across all seven columns ranged from **−4.847270012 to +4.878247738**; maximum absolute gradient component was **0.175783634**. Such Q magnitudes exceed terminal reward magnitude and raise calibration concerns; finiteness is not correctness of learned values.

Collected transitions were **5,000 terminal / 89,846 nonterminal** (5.27% terminal). Rewards were **4,993 at +1 / 89,853 at 0**; seven terminal draws also have reward zero. No negative immediate rewards are expected under the existing actor-relative one-ply convention. The final replay window had **535 terminal / 9,465 nonterminal**, rewards **533 at +1 / 9,467 at 0**, and actors **5,150 X / 4,850 O**. Thus positive terminal targets comprise only 5.33% of the final uniform replay population. This motivates investigation of replay exposure but does not establish a sampling defect; actual minibatch composition was not instrumented.

Training action counts were `[8387, 11018, 16371, 18619, 13270, 15654, 11527]`. These mix greedy decisions with epsilon exploration, including a final exploration floor of 10%, and are **not** evidence of greedy policy diversity. Greedy-only measurements follow.

## Three-model tactical comparison

A = fresh seed-42 network; B = archived Phase 4C.2 candidate; C = final scaled candidate. Fractions are correct/total. Each margin cell is the mean correct-action margin, rounded to six decimals; full precision and per-position values remain in `report.json`.

| Model | Overall accuracy | Win accuracy | Block accuracy | Mean margin | Margin min / max | Legal Q min / max |
|---|---:|---:|---:|---:|---:|---:|
| A: Initial | 10/48 (20.83%) | 6/24 | 4/24 | -0.047792 | -0.182887 / 0.062967 | -0.141177 / 0.137868 |
| B: Archived | 11/48 (22.92%) | 5/24 | 6/24 | -0.226721 | -1.264318 / 0.413529 | -1.406198 / 1.001102 |
| C: Final | 15/48 (31.25%) | 7/24 | 8/24 | -0.248615 | -1.150934 / 0.422447 | -2.648116 / 1.876006 |

| Win/block category | A accuracy; mean margin | B accuracy; mean margin | C accuracy; mean margin |
|---|---:|---:|---:|
| block | 4/24; -0.053466 | 6/24; -0.202236 | 8/24; -0.248196 |
| win | 6/24; -0.042117 | 5/24; -0.251206 | 7/24; -0.249034 |

| Player (0 = X, 1 = O) | A accuracy; mean margin | B accuracy; mean margin | C accuracy; mean margin |
|---|---:|---:|---:|
| 0 | 5/24; -0.046603 | 5/24; -0.274178 | 6/24; -0.253374 |
| 1 | 5/24; -0.048980 | 6/24; -0.179264 | 9/24; -0.243856 |

| Threat direction | A accuracy; mean margin | B accuracy; mean margin | C accuracy; mean margin |
|---|---:|---:|---:|
| diagonal | 3/16; -0.045608 | 4/16; -0.203474 | 2/16; -0.262150 |
| horizontal | 2/16; -0.061168 | 5/16; -0.137209 | 6/16; -0.261935 |
| vertical | 5/16; -0.036599 | 2/16; -0.339479 | 7/16; -0.221760 |

| Correct physical column | A accuracy; mean margin | B accuracy; mean margin | C accuracy; mean margin |
|---|---:|---:|---:|
| 0 | 1/4; -0.051613 | 0/4; -0.345318 | 1/4; -0.191570 |
| 1 | 0/3; -0.117005 | 0/3; -0.548555 | 0/3; -0.633204 |
| 2 | 3/12; -0.030170 | 1/12; -0.288525 | 5/12; -0.067349 |
| 3 | 0/10; -0.075207 | 1/10; -0.358631 | 2/10; -0.329225 |
| 4 | 5/12; -0.015333 | 5/12; -0.066050 | 4/12; -0.330538 |
| 5 | 0/3; -0.103489 | 2/3; 0.211216 | 2/3; -0.068106 |
| 6 | 1/4; -0.031991 | 2/4; -0.162023 | 1/4; -0.249103 |

C corrects 11 tactical errors but regresses on six formerly correct positions versus A; versus B it corrects 11 and regresses on seven. Net gains are modest: +5/48 versus untrained and +4/48 versus archived. C still misses **17/24 immediate wins and 16/24 blocks**. Diagonal accuracy falls to 2/16, below both A (3/16) and B (4/16). The mean margin is more negative than B's despite higher binary accuracy, indicating some remaining errors are strongly preferred. Absolute margin comparisons also reflect changed Q scale.

For known immediate-winning actions, C's mean predicted Q is **0.053413**, with range **−1.823957 to +1.426230**, and **10/24 are negative**, despite their exact terminal target being +1. This is concrete evidence that obvious wins still fail to generalize reliably.

## Greedy diversity and mirrored behavior

Histograms below are ordered physical columns 0–6 and use all 60 diagnostic positions.

| Model | Greedy histogram | Dominant column / share | Entropy, bits |
|---|---|---|---:|
| initial | `[1, 0, 19, 0, 23, 3, 14]` | 4 / 38.33% | 1.860051 |
| archived | `[0, 2, 8, 10, 26, 8, 6]` | 4 / 43.33% | 2.224549 |
| final | `[5, 2, 14, 12, 10, 11, 6]` | 2 / 23.33% | 2.628308 |

| Model | Mirror action matches | Aligned legal-Q mean absolute difference¹ | Maximum difference | Tactical pairs: both / one / neither correct |
|---|---:|---:|---:|---:|
| initial | 8/30 | 0.049140 | 0.206188 | 3 / 4 / 17 |
| archived | 6/30 | 0.406838 | 1.484432 | 1 / 9 / 14 |
| final | 10/30 | 0.576036 | 2.536337 | 5 / 5 / 14 |

¹ Mean of the 30 pair-level means, giving each pair equal weight.

The final policy uses every column and its dominant share falls to 23.33%, compared with 43.33% for B on this suite. There is no evidence of single-column collapse in these probes. Mirror action agreement improves to 10/30, but aligned Q asymmetry grows substantially. These are different properties: more varied or occasionally consistent argmax choices do not imply symmetric value predictions. Absolute Q discrepancies also depend on Q scale; no normalized symmetry score or causal attribution is claimed.

The 48 tactical rows contain only **24 original/mirror pairs**, with unequal column counts. They must not be treated as 48 independent trials. Only five final pairs are correct on both orientations; 14 pairs remain wrong on both. No independence-based significance claim is made. This fixed one-ply suite measures a limited set of tactical behaviors, not general playing strength or longer forcing sequences.

## Fixed Random evaluation

All three models played exactly 100 greedy games, alternating sides (50 X / 50 O). Game index `i` uses a private `random.Random(42+i)`, so the predeclared opponent seeds are 42–141. The opponent samples uniformly from legal moves. Identical seeds control the random stream, but different model choices induce different states and trajectories. No evaluation games update replay, optimizer, or exploration/replay RNG state.

| Model | Wins / losses / draws | Win rate | As X: W / L / D | As O: W / L / D | Evaluation seconds |
|---|---:|---:|---:|---:|---:|
| initial | 64 / 36 / 0 | 64% | 31 / 19 / 0 | 33 / 17 / 0 | 0.126738333 |
| archived | 75 / 25 / 0 | 75% | 41 / 9 / 0 | 34 / 16 / 0 | 0.123752167 |
| final | 75 / 25 / 0 | 75% | 40 / 10 / 0 | 35 / 15 / 0 | 0.130915291 |

| Model | Random-game greedy histogram, columns 0–6 | Dominant column / share | Entropy, bits |
|---|---|---|---:|
| initial | `[56, 2, 143, 60, 309, 21, 183]` | 4 / 39.92% | 2.194406 |
| archived | `[29, 36, 177, 127, 241, 121, 38]` | 4 / 31.34% | 2.460783 |
| final | `[128, 66, 189, 177, 80, 132, 37]` | 2 / 23.36% | 2.645998 |

C equals B's aggregate 75% Random win rate, with one fewer X win and one more O win. Its greedy actions are less concentrated in these games too. This is improved performance over A, but **no Random gain over the archived candidate despite 25 times as many completed training games**. It is exploratory evidence against a weak opponent, not a formal strength benchmark, and cannot establish tactical learning by itself.

The historical Phase 4C.2 run had 200 games, 3,779 plies, 3,716 updates and 4.078935 training seconds, with epsilon decay 0.9997 rather than this run's 0.99997. Its old 12-game Random score was 11/12 (initial 8/12), and its old four tactical probes all failed. Those historical numbers remain preserved. The new 75/100 archived result and 11/48 archived tactical result use larger, different evaluation suites; comparing their percentages directly with the old tiny suites would be misleading. This is also not a controlled training-budget-only ablation because the exploration schedule changed intentionally.

## Checkpoint, integrity, and retained artifacts

New local-only, Git-ignored output:

```text
experiment-output/phase4c2b-seed42-20261003-2059/
```

The one final `candidate.pt` is **3,800,941 bytes**, using the unchanged version-1 inference contract. Its SHA-256 is:

```text
29c1b6c3288811b449b4d90744607e572fca17bc631d5cff4334fc2b4efaf56e
```

The driver saved it only after training ended without an invariant failure, then used the existing `weights_only=True`, CPU safe loader to verify exact equality of every online weight tensor, metadata, and all 60 diagnostic Q-vectors/actions. A separate post-run audit rechecked the SHA-256, safe-loaded predictions and metadata, all **94,846 training moves/rewards**, all **300 evaluation transcripts/outcomes**, replay composition, loss/update/synchronization counts, and the frozen suite identity. All checks passed. Numerical Q output and legal-action checks passed for all three models on all probes and games. Training Q-values, losses, gradients and weights remained finite under the existing runtime safeguards.

Files include `candidate.pt`, `configuration.json`, `report.json` (all three-model evaluations and diagnostics), `events.jsonl` (every ply and completed episode), `source.patch`, `command.txt`, `verification.json`, `run-console.txt`, `audit_results.py`, `audit.json`, and `pre-run-handoff.md`. The checkpoint includes configuration, source/runtime identity, actual budget and evaluation provenance. Its embedded metadata reflects the pre-save report (`checkpoint: null`); the external report adds the checkpoint hash/reload confirmation and total driver time. The checkpoint omits optimizer, target network, replay and RNG states, so it remains **inference-only, not exactly resumable**. Ignored artifacts must be backed up separately if the workspace is removed.

## Conclusion and next Phase 4C action

The best-supported classification is **better Random performance than initialization without sufficient tactical learning**. There is a small emerging tactical signal in horizontal/vertical accuracy and the total solved count, but not enough to call the learning meaningful or robust: 33/48 tactical probes still fail, diagonal accuracy regresses, most mirror pairs remain wrong on both orientations, and immediate-winning Q-values are poorly calibrated. There is no broad single-column collapse; greedy diversity improved. There are nevertheless specific regressions in diagonal accuracy and value symmetry.

A larger run under the unchanged algorithm is **not the recommended next action solely on this evidence**. It produced no gain over the archived model's Random score and only four additional correct tactical decisions. This does not prove that more training could never help; one seed and two endpoint comparisons cannot establish a learning curve.

Recommend that the next separately approved Phase 4C experiment isolate **horizontal symmetry augmentation in replay**, with the same baseline retained and an unchanged evaluation protocol. The strongest direct motivation is the growing mirrored-Q discrepancy and only 10/30 mirrored action matches in a symmetric game. This is a testable next hypothesis, not a demonstrated remedy, and should not be combined with other changes in the same comparison. Replay terminal/sample exposure deserves measurement as another hypothesis because positive terminal rewards are sparse; present counts do not prove that prioritized or stratified sampling is needed. Q overestimation and calibration also warrant attention, but these measurements alone do not identify Double DQN as the preferred first change.

Limitations remain one training seed, one configuration, correlated and column-imbalanced tactical fixtures, no multi-ply tactical/strong-opponent benchmark, no sampled-minibatch composition log, and no cross-runtime determinism guarantee. No diagnostic-label tuning, extra training, automatic retry, sweep, extension, API/UI integration, dependency/hosting/credential changes, neural MCTS/VictorAgent changes, paid calls, commits, pushes, merges, or deployments occurred. **Stop here for review; the proposed next action has not been launched.**
