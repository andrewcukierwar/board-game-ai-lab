# Canonical Connect 4 season v1

**Completed canonical run:** all 112 games verified against frozen source
`c7d0ce65e0a2a23dc6f398164429bd1f2162228f`. See the [preserved results,
intervals, runtime, provenance, and verification command](results/canonical-season-v1/README.md).
The planning and Phase 5E smoke notes below retain their original historical context;
the canonical run occurred after that source was frozen.

A practical portfolio benchmark, not proof of solved playing strength. The JSON
freezes slot order and exact settings:

| Slot | Competitor | Config |
| --- | --- | --- |
| 1 | Random | `{"type":"random"}` |
| 2 | Negamax 2 | `{"type":"negamax","depth":2}` |
| 3 | Negamax 4 | `{"type":"negamax","depth":4}` |
| 4 | Negamax 6 | `{"type":"negamax","depth":6}` |
| 5 | Negamax 8 | `{"type":"negamax","depth":8}` |
| 6 | MCTS 100 | `{"type":"mcts","simulations":100}` |
| 7 | MCTS 400 | `{"type":"mcts","simulations":400}` |
| 8 | MCTS 800 | `{"type":"mcts","simulations":800}` |

Season seed **20261008**, **4 games per pairing**, 28 pairs, **112 games**.
Each entrant plays **28 games**, 14 Red and 14 Yellow. Each pair plays twice in
each color. Red always starts. Draws count immediately with no rematch/tiebreak.

## Methods

Schedule v1 uses a seeded circle rotation and mirrored halves across two cycles.
Domain-separated FNV-1a/uint32-avalanche seeds determine shuffle, color and game
seed. Exact fixture order is exported. Stochastic seed v1 XORs the game seed with
`(revision+1)*0x9E3779B9` and `(player+1)*0x85EBCA6B`, then applies the uint32
avalanche. Random/MCTS receive fresh local Python `random.Random(ply_seed)`;
Negamax ties are deterministic. Pin Python/dependencies along with implementations.

Win/draw/loss scores are **1 / 0.5 / 0**. Standings sort points descending,
score rate descending, then slot seed ascending. Elo begins at **1500**, K=**24**:

```text
expected(A) = 1 / (1 + 10^((rating(B) - rating(A)) / 400))
delta = 24 * (score(A) - expected(A))
rating(A) += delta; rating(B) -= delta
```

Updates follow schedule completion order at full precision. History retains each
entrant's rating after every game. **Elo is pool-relative and order-dependent**,
not a universal Connect 4 rating or an uncertainty interval.

Score intervals are **descriptive bootstrap intervals**: minimum **8 games**,
**1000** IID with-replacement resamples of each entrant's observed score vector,
each of the original length. Mulberry32 uses a seed derived from Season seed and
`season:bootstrap:{entrant_id}`. Sort sample means and linearly interpolate at
`(1000-1)*p`, p=0.025/0.975. These intervals do not model repeated-opponent/fixed-
search dependence or opponent-pool uncertainty. Homogeneous samples collapse
without proving certainty. **Four games per pair remain a modest sample.**

## Planning and reproduction

From the repository root:

```sh
node ui/scripts/run-connect4-benchmark.mjs \
  --config benchmarks/connect4/canonical-season-v1.json
```

Default is dry-run: no HTTP, files or games. It prints configs, game count,
API/output targets and upper workload estimates. At 42 plies/game: 2352 stochastic
decisions including Random, 1764 MCTS searches and 764400 MCTS simulations before
early tactical shortcuts. These are work bounds, not runtime predictions.

For a **separately authorized future canonical run**, start a local backend:

```sh
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' \
  .venv/bin/gunicorn api.app:app --bind 127.0.0.1:8000 --workers 1 --threads 4
# In another terminal, only after separate canonical-run authorization:
node ui/scripts/run-connect4-benchmark.mjs \
  --config benchmarks/connect4/canonical-season-v1.json \
  --api http://127.0.0.1:8000 --execute
```

**Execution should target a local backend.** Default 127.0.0.1:8000; non-loopback
origins require `--allow-nonlocal`; credentials/paths/queries/fragments and redirects
are refused. Do not run automated production/Render campaigns. No provider or
explanation endpoint is called. Configure the actual reviewed backend commit with
`EVALUATION_SOURCE_COMMIT` if available; never invent one.

Execution manually steps the existing Season controller: sequential mutations,
one live game, known session replacement, confirmed history and no blind retry.
Backend manifest changes stop the run. A new exclusive timestamp/UUID directory
contains:

```text
benchmarks/connect4/runs/canonical-season-v1-<run-id>/
  evaluation.json
  games.csv
  summary.csv
  run-metadata.json
```

Validated JSON checkpoints atomically after each completed game. CSVs finalize
on completion or caught interruption. Hard termination preserves the last completed
checkpoint, potentially without final CSVs/metadata. Previous directories are never
reused. **No automatic resume** or browser artifact import is implemented; review
an interrupted campaign rather than guessing/retrying across processes.

Runtime metadata stays outside the digest: UTC start/finish, total/per-game wall
time, Node version, OS/release/architecture, CPU description/count, API origin and
optional backend commit. Record Python/backend/dependency versions and memory,
power/concurrency conditions alongside a real run; backend runtime is null until
supplied externally. Per-game time includes HTTP/history/manifest checks, excluding
later checkpoint export; total time includes checkpoint work. **Single-machine
timing is machine-specific** and does not affect strength scoring.

```sh
node ui/scripts/verify-evaluation-export.mjs benchmarks/connect4/runs/<run-id>/evaluation.json
```

Publish reviewed evidence, CSVs, metadata, source/environment details and a separately
trusted digest. See [provenance](../../docs/evaluation-provenance.md) for hash coverage
and source/version limitations.

## Historical Phase 5E verification and runtime planning

Only `smoke-season-v1.json` was executed: Random / Negamax 1 / Negamax 2 / Random,
seed 1234, **12 games / 210 plies**, about **357 ms** on the local host. This
verifies infrastructure, not expensive-agent runtime or strength.

Existing [local latency measurements](../../docs/public-agent-strength-and-turn-order.md)
on Apple M5 measured MCTS 100 at roughly 43–87 ms, 400 at 151–327 ms, 800 at
297–654 ms, and Negamax 8 at 0.5–218 ms across five positions. Conditional on
roughly 20–35 plies/game and those sampled latencies, budget **3–12 minutes** on
similar hardware and reserve **15–20 minutes**. This is an extrapolation, not a
worst-case bound or measured canonical run; other positions/hardware/contention
can exceed it. **The 112-game benchmark was not executed in Phase 5E.**
