# Local reproduction

Use the laptop checkout and CPython 3.11.17 with the existing `.venv`.
No provider, server, production checkout, deployment, or model is needed.
Do not overwrite these evidence files: use a new directory for a new run.
Only one evaluator may write to an experiment directory at a time.

```sh
.venv/bin/python -m scripts.evaluate_negamax_depths declare --directory /tmp/negamax-phase3a
.venv/bin/python -m scripts.evaluate_negamax_depths preflight --directory /tmp/negamax-phase3a
.venv/bin/python -m scripts.profile_negamax_depths preflight-memory --directory /tmp/negamax-phase3a
.venv/bin/python -m scripts.evaluate_negamax_depths freeze --directory /tmp/negamax-phase3a
.venv/bin/python -m scripts.evaluate_negamax_depths run --directory /tmp/negamax-phase3a --max-games 12
.venv/bin/python -m scripts.evaluate_negamax_depths run --directory /tmp/negamax-phase3a
PYTHONHASHSEED=0 .venv/bin/python -m scripts.analyze_negamax_depths --directory /tmp/negamax-phase3a
.venv/bin/python -m scripts.profile_negamax_depths profile --directory /tmp/negamax-phase3a
.venv/bin/python -m scripts.audit_negamax_depths --directory /tmp/negamax-phase3a
.venv/bin/python -m scripts.diagnose_negamax_tail --directory /tmp/negamax-phase3a
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs
```

`declare` writes candidate.json before any strength observations; `freeze`
requires all 96 preflight games and matching memory evidence, then writes
experiment.json exactly once using the declared timing-only formula. `run`
uses the frozen manifest. Its cumulative 1,800-second cap includes previous
invocations and incomplete attempts. `--max-games` is an administrative
checkpoint, without changing the sample or cap. The saved 12-game checkpoint
is checkpoint-status.json. The main run resumes from those games.

Source hashes identify exact production and measurement code; the source
commit is the application baseline at declaration, because the new scripts
were still uncommitted then. The final Phase 3A commit contains those exact
hashed scripts. Replay does not require Git HEAD to remain the baseline SHA;
the checked files and Python version must match. Copying an experiment to a
different source/runtime requires a fresh declaration, never edited hashes.

The analysis command sets PYTHONHASHSEED=0 because paired contrasts iterate
shared-opening sets; the bootstrap seed alone does not fix Python set order.
Use the same setting for byte-reproducible bootstrap results. Game histories
do not depend on hash randomization. Timing and peak-memory results naturally
vary by host and scheduling. Exclude all preflight, profiling and audit repeats
from inference. Analysis validates every saved game with the array engine.

The profiling supplement was declared before main-result inspection, from
the first quiet midgame in candidate order absent from the base profiling set.
Its small candidate.json includes the original source hashes and parent
candidate digest. Recreate it by selecting that same labeled position, writing
a separate candidate manifest (copy `source_hashes`, `python`, `source_commit`,
and a one-element `profile_positions` list), and run the identical profiler:

```sh
.venv/bin/python -m scripts.profile_negamax_depths profile --directory docs/search-negamax-v2/deeper-search/profiling-supplement
```

Use a copied supplement directory without its profiling.json for a fresh run;
the profiler deliberately refuses to overwrite evidence. Base and supplementary
profiling are serial and use one warm-up plus three timing samples per depth,
separate cProfile, separate retained-TT tracemalloc, and depth-6 line tracing.
Normal production TT lifetime ends after each decision. Diagnostic retention
is deliberate and must not be described as a production leak or RSS bound.

`diagnose_negamax_tail` is explicitly post hoc: replay and profile the main
depth-10 moves with maximum observed nodes and maximum elapsed time, deduplicated
if they select the same move. Selection never uses outcomes. These diagnostics
measure the observed cost tail and remain outside both strength inference and
predeclared fixture aggregates. Their own script hash is recorded separately.

Evidence meanings: candidate/experiment manifests contain seeds, configurations,
histories, labels and hashes; JSONL files contain complete games and per-move
timings/counters; status files contain budget accounting; preflight-memory and
profiling JSON contain raw samples and allocation/function/line data; analysis
contains all strata, cluster intervals and paired contrasts; audit contains
independent replays and 12 excluded deterministic repetitions. Errors go to
`*-attempts.jsonl`; partial active checkpoints are validated on resume. A
hard kill can leave a partial JSONL requiring explicit repair from a preserved
copy; corruption is never silently ignored. The last uncheckpointed search
can lose wall-time accounting under a hard kill; orderly deadline handling is
tested and produces explicit unscored incomplete records.
