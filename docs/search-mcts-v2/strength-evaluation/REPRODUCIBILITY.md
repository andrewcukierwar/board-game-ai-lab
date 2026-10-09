# Reproduce locally

Use the laptop development checkout and CPython 3.11.17. No dependencies beyond
the existing development venv are needed. Do not run these commands against
the API or a production checkout. Use one writer and a fresh output directory.
Search runs, profiling, capacity measurements, and tests should run sequentially
to avoid contaminating latency measurements with one another.

To reproduce the exact manifest, including the recorded agent source commit:

```sh
mkdir -p /tmp/mcts-strength-reproduction
cp docs/search-mcts-v2/strength-evaluation/experiment.json /tmp/mcts-strength-reproduction/
.venv/bin/python -m scripts.evaluate_mcts_strength preflight --directory /tmp/mcts-strength-reproduction
.venv/bin/python -m scripts.evaluate_mcts_strength run --directory /tmp/mcts-strength-reproduction --max-games 28
.venv/bin/python -m scripts.evaluate_mcts_strength run --directory /tmp/mcts-strength-reproduction
.venv/bin/python -m scripts.profile_mcts_strength --directory /tmp/mcts-strength-reproduction
.venv/bin/python -m scripts.measure_mcts_capacity --directory /tmp/mcts-strength-reproduction
.venv/bin/python -m scripts.audit_mcts_strength --directory /tmp/mcts-strength-reproduction
.venv/bin/python -m scripts.analyze_mcts_strength --directory /tmp/mcts-strength-reproduction
.venv/bin/python -m scripts.ablate_mcts_loop_locals --directory /tmp/mcts-strength-reproduction
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs
```

Alternatively declare a new configurable experiment:

```sh
.venv/bin/python -m scripts.evaluate_mcts_strength declare \
  --directory /tmp/mcts-strength-new --pairs 128 --secondary-pairs 32 \
  --preflight-pairs 4 --seed 2026100903 --max-seconds 1800
```

The declaration fixes the five simulation budgets, the fourteen comparisons,
all histories/seeds, hardware/runtime, source hashes, pairing, and intervals.
Sample counts, master seed, output directory, preflight size, main compute cap,
and per-invocation game count are CLI parameters; declared configs can specify
different agent settings before collecting any data. Exact manifest reproduction
requires matching source hashes and Python version; changed sources must use a
new experiment. Git HEAD need not match because the final evidence commit also
contains the new harness; the agent files still match their declared source
commit and hashes.

`--max-games` checkpoints only at game boundaries. A later `run` resumes by
validating/replaying completed results and skipping their IDs. Interrupted games
are saved as incomplete attempts, then stop the invocation; they restart with
the same seeds on a deliberate resume. A nonpositive game limit is rejected.
The cumulative compute cap carries across resumptions. Results are not duplicated.

`experiment.json` and `DESIGN.md` precede all game observations. `preflight.jsonl`
is excluded from inference. `results.jsonl` contains every opening, configuration,
source commit, role RNG seed, complete history, color/result, per-move wall/CPU
time and simulation accounting, final RNG hashes, and aggregate search calls/times.
Its experiment hash binds each row to the complete source-hash manifest.
Main-game simulation counts are inferred from the immediate-win guard and fixed
production loop, outside timing; profiling captures actual root visits separately.
No absent/error game is assigned a score. A missing attempt file means no attempt
failed, as verified in `audit.json`.

`analysis.json` includes each opening-pair advantage and joint cross-budget
differences. Bootstrap randomness is independent of gameplay RNG. Sorted opening
IDs make analysis deterministic across Python hash seeds. `profiling.json` retains
all fresh-seed samples, full caller profiles, tree/allocation diagnostics, and
the full position list. `capacity.json` adds isolated process high-water RSS
and five warm synthetic contention samples per condition. RSS includes the
interpreter/imports, is not reachable tree size, and excludes separate processes.
`loop-ablation.json` separately records the rejected local-binding prototype,
its full candidate source and predeclared acceptance criteria, matched timings,
and tree/RNG equivalence. It never modifies production agent modules.

Timings will vary with hardware, system load, GC, and thermal conditions. At
matching source/runtime, game histories, results, and RNG hashes should repeat;
timing bytes and system metadata will not. The audit repeats both colors of all
fourteen conditions on the first declared opening (28 games) solely as a
determinism check; these repeats are not additional strength observations.

The test process disables local dotenv loading and live explanations without
editing environment files. The repository test fixture forbids live HTTPS
provider calls. Frontend/browser tests are outside this backend-only change.
