# Phase C profiling declaration

2026-10-10, after audited Phase A/B rejection. Strongest independently validated
engine is original direct at `60b99b0d42145da149f433907585b809060c946d`.
Source snapshot copied from Phase A frozen direct-source.py, digest in manifest.
No production integration or public cap change.

Hypothesis: after incremental evaluation and packed TT, per-node state updates,
terminal detection and integer-entry access account for substantial CPU. Use
measured profile fractions/calls to select at most two bounded experiments;
prepared code alone is not a reason to benchmark it.

Four predeclared D10 histories: empty, near-opening, seeded-04, post-hoc-tail,
from the immutable32-position manifest. Two complete decisions each after64
constructor warmups, cProfile self/cumulative time and calls; every root score,
move and node/entry/hit/cutoff vector must match corresponding Phase A A/direct.
All decisions profiled under shared lock, including warmups; budget120s.
Game construction outside profiling, complete choose_move inside. Include
normal table construction, all roots, restoration and disposal. Profile times
are explanatory and cannot satisfy a complete-decision latency gate.

Freeze manifest script/source/design hashes, commit/push before measurement.
After measurement, report top bottlenecks and choose final experiment sources
and independent acceptance criteria before any candidate timing.

```sh
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.profile_negamax_v3 --source docs/search-negamax-v3/profile/direct-source.py
```
