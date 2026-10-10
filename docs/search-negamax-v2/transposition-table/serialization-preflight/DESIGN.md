# Phase 3B.2 declaration — before benchmarking

Immutable source baseline: f2f57b46c7817bb7324f4390da0bb345ae2fc7e4,
loaded by `git show SHA:games/connect4/agents/negamax_agent.py`.
Only TT representation changes in A. Incremental evaluation, terminal checks,
alpha/beta classification, root completeness and historical ties remain fixed.
No depth limits, public presets, MCTS settings or frontend changes.

## Experiment A: unbounded representations

Compare (1) original tuple key / tuple value, (2) packed full-identity integer
key / tuple value, (3) packed integer key / packed arbitrary-width integer entry.
Keys allocate 49 bits each to X and O, one to mover, and all higher bits to
nonnegative remaining depth: X | (O << 49) | (mover << 98) | (depth << 99).
SearchState never sets sentinel bits; each bitboard is below 2**49, so disjoint
fields give a mathematical injection for every valid internal board and depth.
No hash identity, mirror canonicalization, pruning or ordering changes.

Packed entries use zigzag for signed score, two flag bits, and four hint bits:
(zigzag(score) << 6) | (hint_code << 2) | flag_code. Flags EXACT/LOWER/UPPER
are 0/1/2, moves 0..6, None is 15. Codes 7..14 permit invalid-hint tests and
are rejected by legal-move ordering. Encoding rejects other unsupported hints
and flags. Python integers have arbitrary precision, including scores
+/- (1,000,000 + any positive supported depth); no Victor WDL int8 reuse.

Required: identical ordered full root vectors, selected moves, nodes, entries,
hits, loop cutoffs, and separately counted heuristic leaves on all A runs.
Independent array minimax at tractable depths; all flags and narrow windows,
transpositions, both movers, multiple depths, fast/slower wins, defenses,
draws, invalid/absent/full-column hints, and complete exception rollback.
Prove field separation; deterministic legal identity collision and all field
boundary round trips (including enormous integers) must pass.

Default acceptance: all exact checks; >=30% reduction in summed reachable TT
bytes on non-tail conditions with >=2,000 baseline entries; geometric mean
wall latency ratio <=1.08 AND summed median ratio <=1.08 across all non-tail
conditions; every condition ratio <=1.20 OR added <=0.25 ms; traced peak growth
<=5% OR <=32 KiB in every condition. Report CPU independently. Among eligible
variants choose greatest measured retained-memory reduction. A large-memory,
CPU-cost alternative failing thresholds remains experimental, with tradeoffs
reported. If none pass, retain original production TT.

## Fixtures and measurement

All 20 Phase 3B.1 manifest histories read from the immutable Git baseline:
eleven Phase 3A quiet/tactical/endgame fixtures, eight seed-20261009 fixtures,
and separately identified post hoc [1,4,6,0,6]. Add four legal nonterminal
boards at 10/14/18/22 plies, uniform legal moves with terminal rejection only,
seed 20261010. Selection never examines optimization outcomes. Tail excluded
from primary acceptance; all 24 fixtures tested at depths 4/6/8/10.

Seven timed complete ordinary choose_move executions plus one discarded warmup
per variant/condition; rotate variant order per repetition. Fresh TT every
run; game construction, retained sizing and instrumented diagnostics excluded
from timing. Measure wall and process CPU. Separate leaf-count run and two
tracemalloc runs per condition/variant. Warm each implementation's table class
with 64 constructions before measurements, including fresh-process diagnostics,
to calibrate CPython split instance dictionaries.

Reachable retained memory counts unique objects from the SearchTable itself,
including its attribute dictionary and storage, separating dict structure /
allocated spare capacity (sys.getsizeof includes both), key tuples, bitboard
ints, other key fields, value tuples, scores, hints/flags and table overhead.
Report shared-object attribution order; do not multiply shallow entry size.
Traced current and peak are distinct from reachable TT bytes and process RSS.
One fresh subprocess per variant/condition measures before/after RSS where
available and ru_maxrss high water (macOS bytes, Linux KiB), with the TT retained
for diagnosis. RSS includes interpreter/imports/allocator; no memory-ceiling
claim. Microbenchmark key construction and successful lookup on real TT keys,
20,000 operations, warmup + seven rotating samples, outside acceptance.
Freeze source/design/fixture hashes and host metadata before run; fail on drift,
score/counter mismatch or attempted overwrite; preserve failed evidence.

## Experiment B: independent bounded ablation AFTER A

Start from selected A representation, or original if none is eligible. Compare
unbounded reference and deterministic direct-mapped tables with 16,384 / 32,768 /
65,536 slots. Index is hash(full_key) modulo capacity; Python integer/tuple-of-int
hashing is deterministic on the recorded runtime. Slots store the complete key
and complete entry; equality is checked before reuse. A collision replaces the
old occupant and counts as an eviction; same-key writes count as replacements.
No retained foreign-key bound, unknown result, budget interruption or premature
root termination. Different-depth scores never substitute for each other.
Entry counts, hits, nodes, cutoffs and leaves may change; exact ordered root
scores, chosen moves, terminal scores, and complete decisions must not.

Report occupancy, evictions, replacements, hits/nodes, extra nodes, wall/CPU,
reachable retained bytes, traced current/peak, fresh RSS/high water, and worst
observed expensive row/sample. Use same fixtures/depths and seven rotating
samples + warmup, two traces, counted and fresh-process checks. Pay explicit
attention to empty/near-opening D10, quiet-wide and tail. The reference and
bounded lookup/store paths are documented: representation/dispatch cost as well
as changed cache availability contributes to total bounded latency.

Default eligibility additionally requires >=30% summed reachable-memory saving
on reference conditions exceeding 16,384 entries, geometric AND summed latency
ratios <=1.05, each condition <=1.20 OR added <=0.25 ms, every expensive search
(reference >=50 ms, including tail) <=1.20 ratio and <=1.50 node ratio, and peak
growth <=5% OR <=32 KiB per condition. If multiple pass choose largest budget
that passes. Otherwise preserve unbounded public behavior and retain bounded
storage as benchmark-only experiment. Fixed entry count does not cap process
memory, temporary allocations, dictionary capacity or interpreter RSS.

Full backend suite: PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false
OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs; provider calls forbidden.
Review relevant changes, commit and push search/negamax-v2; no merge/deployment.
