# Canonical Connect 4 benchmark — observed results

**Verification PASS — 112/112 games.** `canonical-season-v1` evaluates eight public agent configurations on the frozen backend source `c7d0ce65e0a2a23dc6f398164429bd1f2162228f`. Negamax depth 6 leads the Season Lab standings; depth 8 has the highest final Elo. They tie at 71.4% observed score rate and draw all four direct games.

**This is a reproducible balanced evaluation from the standard starting position, not 112 independent statistical trials.** Deterministic competitors repeat trajectories when given the same colors and empty board. Results describe this field, schedule, starting position, and implementations; they do not establish universal agent strength.

## Setup and entrant configurations

- Season seed: **20261008**; deterministic schedule v1.
- Eight entrants, 28 unordered pairs, **four games per pair**, **112 games** total.
- Every game starts on the standard empty 6×7 board; Red moves first.
- Every pair plays twice in each orientation. Each entrant plays **28 games: 14 Red, 14 Yellow**.
- Wins/draws/losses score **1 / 0.5 / 0**. Draws stand without rematches.
- One live server game; sequential execution, maximum concurrent mutation POSTs **1**.
- No neural agents, checkpoints, LLM/provider calls, or randomized opening suite.

| Slot / ID | Agent | Exact config |
| --- | --- | --- |
| 1 / `entrant-1` | Random | `{"type":"random"}` |
| 2 / `entrant-2` | Negamax depth 2 | `{"type":"negamax","depth":2}` |
| 3 / `entrant-3` | Negamax depth 4 | `{"type":"negamax","depth":4}` |
| 4 / `entrant-4` | Negamax depth 6 | `{"type":"negamax","depth":6}` |
| 5 / `entrant-5` | Negamax depth 8 | `{"type":"negamax","depth":8}` |
| 6 / `entrant-6` | MCTS 100 | `{"type":"mcts","simulations":100}` |
| 7 / `entrant-7` | MCTS 400 | `{"type":"mcts","simulations":400}` |
| 8 / `entrant-8` | MCTS 800 | `{"type":"mcts","simulations":800}` |

The [frozen configuration](../../canonical-season-v1.json) and [methodology](../../canonical-season-v1.md) specify seed derivation, pairing order, local stochastic RNGs, Elo, and bootstrap algorithms. The artifact retains the exact ordered fixtures and game seeds.

## Standings, Elo, and observed-score intervals

The following is in **Season Lab standings order**. Score rate = `(wins + 0.5 × draws) / played`. Elo is rounded to the nearest integer here; [evaluation JSON](evaluation.json) and [summary CSV](summary.csv) retain full precision.

| Rank | Agent | W–D–L | Points / 28 | Score rate | Final Elo | Observed score-rate bootstrap interval |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | Negamax depth 6 | 17–6–5 | 20 | 71.4% | 1586 | 55.4–83.9% |
| 2 | Negamax depth 8 | 18–4–6 | 20 | 71.4% | 1593 | 55.4–85.7% |
| 3 | Negamax depth 4 | 18–2–8 | 19 | 67.9% | 1575 | 51.8–83.9% |
| 4 | Negamax depth 2 | 18–0–10 | 18 | 64.3% | 1564 | 46.4–82.1% |
| 5 | MCTS 400 | 15–0–13 | 15 | 53.6% | 1531 | 35.7–71.4% |
| 6 | MCTS 800 | 13–0–15 | 13 | 46.4% | 1471 | 28.6–64.3% |
| 7 | MCTS 100 | 7–0–21 | 7 | 25.0% | 1395 | 10.7–39.4% |
| 8 | Random | 0–0–28 | 0 | 0.0% | 1285 | 0–0% |

Standings sort points descending, score rate descending, then slot seed ascending. Depth 6 and depth 8 tie on both performance measures used by standings, so slot 4 precedes slot 5. **Depth 6 is the standings leader under this ordering; depth 8 is the Elo leader** (1593.0522801720003 versus 1586.3544441568047). These are distinct summaries, not a universal winner designation.

Elo begins at **1500**, K=**24**, scale=**400**, and updates at full precision in completion order. It is **pool-relative and order-dependent**, not a universal Connect 4 rating.

The intervals are **deterministic bootstrap intervals for observed score rate**: 1,000 with-replacement resamples of each entrant's 28 scores, with 2.5th/97.5th percentiles using linear interpolation and a seeded Mulberry32 PRNG. This is descriptive IID resampling of the observed vector. It does not model deterministic repetitions, repeated opponents, or uncertainty in the opponent pool. These are **not Elo confidence intervals, population confidence intervals, or formal agent-strength confidence intervals**. Random's collapsed interval reflects an all-loss sample, not proof of zero strength in all settings.

## Algorithm findings and pairwise checks

All highlights below were recomputed from the completed game evidence, including terminal winner indices and ordered move columns, and agree with the verified analytics.

| Pair / baseline | Observed outcome |
| --- | --- |
| Negamax 6 vs Negamax 8 | Four draws |
| MCTS 400 vs MCTS 800 | Two wins each, no draws |
| MCTS 400 vs Negamax 2 | MCTS 400 wins 3–1 |
| MCTS 800 vs Negamax 2 | MCTS 800 loses 1–3 |
| MCTS 800 vs Negamax 4 | MCTS 800 loses 1–3 |
| MCTS 800 vs Negamax 6 | MCTS 800 loses 1–3 |
| MCTS 800 vs Negamax 8 | MCTS 800 loses 1–3 |
| Random across the field | 0 wins, 0 draws, 28 losses |

**Negamax was the strongest algorithm family in this field:** all four depths finished above all MCTS variants. Depths 6 and 8 tied at 71.4%. Increasing depth did not give a simple monotonic improvement in the observed benchmark; the standings tiebreak does not establish that depth 6 is inherently stronger than depth 8.

**MCTS budget and performance:** increasing from 100 to 400 simulations improved observed score rate from **25.0% to 53.6%**. At 800 it was **46.4%**. Since 400 and 800 split their direct matchup 2–2, this benchmark does not establish intrinsic superiority of 400. Configured simulation budgets scale 1:4:8, but tactical root guards can bypass search and games differ in length and positions. Metadata records whole-game times, not per-agent search time or realized simulations. It does not support an isolated compute-efficiency ranking or machine-independent latency claim.

## Red / Yellow results

| Outcome | Games |
| --- | ---: |
| Red wins | 59 |
| Yellow wins | 47 |
| Draws | 6 |

Red score rate = `(59 + 0.5 × 6) / 112` = **55.357142857…%**, approximately **55.4%**. Yellow's is 44.6%. Every entrant has exact 14/14 color balance, and every pair has exact 2/2 orientation balance. The observed side difference supports that design choice. It does not precisely estimate the universal first-player advantage of Connect 4; this is a fixed agent pool from one starting position.

## Deterministic repeated trajectories

Negamax does not use stochastic seeds. For a fixed deterministic pair, empty board, and color assignment, repeated games follow the same ordered columns. Direct inspection finds **two distinct color-oriented trajectories among four games for each of the six Negamax-vs-Negamax pairs**, with each trajectory repeated twice. Those 24 games therefore contain 12 distinct oriented trajectories.

The four Negamax 6 vs 8 draws likewise comprise two repeated orientations. Exact repeatability is valuable evidence of deterministic behavior, but repetition does not add independent samples. Stochastic agents receive deterministic per-ply seeds for reproduction; those games also share opponents and a common starting position. Neither the 112-game count nor the bootstrap computation removes these dependencies.

## Runtime and machine context

Recomputed from unchanged [run metadata](run-metadata.json):

| Measurement | Observed value |
| --- | ---: |
| Total wall time | **293,562.797750 ms** = 293.562798 s ≈ **4m 54s** |
| Sum of per-game times | 289,881.079121 ms |
| Mean game time | **2,588.223921 ms** ≈ 2.59 s |
| Median game time | **1,030.802625 ms** ≈ 1.03 s |
| Maximum game time | **10,823.412000 ms** ≈ 10.82 s |
| Game plies | **3,057** |
| Mutation requests | **3,169** = 112 game starts + 3,057 plies |
| Maximum concurrent POSTs | **1** |

The recorded host is **Apple M5, 10 CPUs reported**, `darwin` / `25.6.0` / `arm64`; runner Node is **v25.9.0**, API origin `http://127.0.0.1:8000`. Start/finish metadata is `2026-10-08T03:38:38.504Z` / `2026-10-08T03:43:31.978Z` (UTC). Total uses the runner's elapsed timer rather than subtracting rounded ISO timestamps.

Per-game time includes HTTP/history/manifest checks and excludes subsequent checkpoint export; total also includes checkpoint work. Measurements are specific to this local execution. **`backend_runtime` is null:** Python/backend dependency versions, memory, power state, and background contention were not captured in this file. Do not infer them from the packaging environment or present the timing as portable across machines.

## Provenance and preservation

Original local source directory:

```text
benchmarks/connect4/runs/canonical-season-v1-2026-10-08T03-38-38-499Z-82b7a4b5/
```

Stable public directory:

```text
benchmarks/connect4/results/canonical-season-v1/
```

The four raw files were copied **byte-for-byte**, retaining timestamps within their contents, JSON formatting, CSV CRLF records, metadata, and hashes. The transient `runs/` directory remains ignored; the stable result directory is trackable without changing `.gitignore`.

All **112/112** captured backend manifests and `run-metadata.json` name source commit `c7d0ce65e0a2a23dc6f398164429bd1f2162228f`. The evaluation's top-level `provenance.source_commit` is **null**, as expected for the CLI artifact generator; it is separate from each game's recorded backend source. This field was preserved, not retroactively filled.

Version registry: engine **1**, Random/Negamax/MCTS **2**, API contract **2**, API provenance **1**, stochastic seed **1**, schedule **1**, evaluation methodology **1**, export schema **1**. Version/commit declarations are reference identifiers, not authenticated build attestations.

Canonical evidence SHA-256:

```text
e5845905b29c158cebed8579cd0a858b3a873a289d2b2e754c29601b7c9e407d
```

This hashes canonical `{format, schema_version, provenance, methodology, evidence}`. It excludes export timestamp, derived analytics, and separate runtime metadata. The verifier recomputes derived analytics rather than trusting them. The checksum is not a cryptographic signature and legal replay does not prove the named agent chose a move. See the [provenance and integrity contract](../../../../docs/evaluation-provenance.md).

Separate **whole-file SHA-256 hashes**, equal in source and destination:

| Raw artifact | File SHA-256 |
| --- | --- |
| [evaluation.json](evaluation.json) | `89f30b45918bbb9ce45a2e1d9da82e93ee6d19b9efd73faed7d7381883dadd88` |
| [games.csv](games.csv) | `eaf8f0f733a3019293b73bf78eb67770f619add8104c0ebe82cd4cd0fb1c7541` |
| [summary.csv](summary.csv) | `47ee6292515b18c3c893c6714b0a430e737051d2455f3104b9a7f867f2a01827` |
| [run-metadata.json](run-metadata.json) | `3cb72329695620bfff166cd1d74a8061a99c73934289967de8b86511fbccf6e8` |

## Verify without rerunning games

From the repository root:

```sh
node ui/scripts/verify-evaluation-export.mjs \
  benchmarks/connect4/results/canonical-season-v1/evaluation.json
```

Expected: `Verification PASS`, field 8, `COMPLETE: 112/112 games`, and the exact evidence digest above. The verifier checks schema/methodology/provenance, reconstructs the seeded schedule and balance, replays through terminal outcomes, and compares recomputed standings, Elo/history, sides, pairwise results, and bootstrap intervals. Verification makes no API or provider calls and does not launch another benchmark.

For whole-file integrity:

```sh
shasum -a 256 benchmarks/connect4/results/canonical-season-v1/evaluation.json \
  benchmarks/connect4/results/canonical-season-v1/games.csv \
  benchmarks/connect4/results/canonical-season-v1/summary.csv \
  benchmarks/connect4/results/canonical-season-v1/run-metadata.json
```

No supplied result disagreed with the verified artifacts. Remaining scientific limits are fixed empty-board coverage, dependent deterministic repetitions, modest four-game pairwise samples, pool/order-relative Elo, descriptive observed-score bootstrap intervals, and incomplete backend runtime capture. The benchmark is evidence for this reproducible experiment, not a solved-game or definitive-ranking claim.
