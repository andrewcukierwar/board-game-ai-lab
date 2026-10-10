# Phase 3B.2: Negamax transposition-table storage

October 9, 2026. Laptop repository `andrewcukierwar/board-game-ai-lab`, branch
`search/negamax-v2`. Immutable baseline:
[f2f57b46c7817bb7324f4390da0bb345ae2fc7e4](https://github.com/andrewcukierwar/board-game-ai-lab/commit/f2f57b46c7817bb7324f4390da0bb345ae2fc7e4).

**Accept packed full-identity keys and packed integer entries. Keep public
storage unbounded.** Compact storage meets the declared memory/latency criteria.
Both bounded replacement strategies preserve exact decisions but fail default
eligibility. A and B are independent comparisons; their latency samples and
references are never pooled. No public depth/preset, MCTS, API, frontend,
production environment, training worktree or previous evidence changes.

## Architecture and correctness

Before: fresh `SearchTable` per complete decision; dict keyed by the tuple
`(X_bits, O_bits, mover, remaining_depth)` and storing `(flag, score, hint)`.
One shared table serves every legal root move. Normal choose_move releases it.
After: the same dict, table lifecycle, search algorithm and incremental evaluator,
with a single integer key and integer value. SearchState, terminal logic,
ordering, pruning, root loop, public interfaces and last_stats fields are unchanged.
The production diff consists of the key expression, entry encode/decode and an
import. The selected production bodies match the Git-derived measured variant.

Key, with arbitrary-precision nonnegative remaining depth d:

```python
key = X | (O << 49) | (mover << 98) | (d << 99)
```

Seven columns × seven bits occupy addresses 0..48; playable cells use at most
bit 47 and sentinels remain empty. Both bitboards fit below 2**49. The fields
are disjoint: masks recover X and O, bit 98 recovers mover, and `key >> 99`
recovers the entire depth. This proves injectivity for every valid internal
board (and even arbitrary 49-bit field combinations), both movers and any
nonnegative depth. No truncated/Zobrist hash or symmetry canonicalization is
used as identity. Python hash collisions remain possible; dictionary/full-slot
key equality prevents false hits. Tests force a genuine Python hash collision
using distinct depths separated by `sys.hash_info.modulus`.

Entry:

```python
z = 2 * score if score >= 0 else -2 * score - 1
entry = (z << 6) | (hint_code << 2) | flag_code
```

EXACT/LOWER/UPPER map to 0/1/2; flag code 3 is rejected. Legal hints 0..6
retain their values; 15 represents None. Invalid codes 7..14 round-trip and
are ignored by existing legal-hint validation. Unsupported encoder hints
(including 99) are rejected; the ordering function still ignores arbitrary
invalid hints such as 99. Round trips cover all supported hints, flags and
signed scores, including ±(1,000,000 + 10**100). Zigzag is a bijection from
signed integers to nonnegative integers. Arbitrary precision avoids overflow
for every supported positive search depth; Victor's int8 terminal-only WDL
representation is never reused. Packed value zero is a valid entry, not absence.

Terminal precedence, fast-win/slower-win scores and draw semantics are exact.
Alpha/beta bounds are classified using the original pre-probe window exactly
as before. A bound remains a bound; its move is only an ordering hint. Depth
and mover stay in the full key. Full-window searches complete every root move,
including alternatives to an immediate win, and CENTER_ORDER insertion preserves
historical root ties. Exceptions restore every detached state field, including
incremental window counts, score and history, and leave the caller unchanged.

Independent array minimax verifies quiet/tactical boards, wins, defenses,
near-full boards and draws at tractable depths. Tests cover all bound types,
narrow-window reuse, transpositions, multiple depths/movers, absent/invalid/full
column hints, huge signed values, collisions and forced one-slot eviction.
The identity test visits every legal opening history through five plies plus
200 seeded legal play/undo trajectories, testing more than 100,000 distinct
board/mover/depth identities. This complements the algebraic field-separation
proof; finite collision tests alone are not a proof over all boards.

## Design, provenance and measurement

[DESIGN.md](DESIGN.md) was written before measurements. A uses all 20 immutable
Phase 3B.1 histories: eleven Phase 3A quiet/tactical/endgame boards, eight seeded
boards, and the separately identified post hoc tail `[1,4,6,0,6]`. Four additional
nonterminal boards at 10/14/18/22 plies use seed 20261010 with terminal rejection
only. No fixture selection examines optimization outcomes. All 24 boards run
at depths 4/6/8/10 (96 conditions); the tail is excluded from acceptance aggregates.

A acceptance: exact scores/moves/counters/leaves; ≥30% summed reachable-TT saving
on non-tail conditions with ≥2,000 baseline entries; wall geometric and summed
median ratios ≤1.08; per-condition ratio ≤1.20 or added ≤0.25 ms; traced peak
growth ≤5% or ≤32 KiB. Select the eligible variant saving most retained memory.
The tail also satisfies the per-condition latency/peak limits.

B acceptance: ≥30% summed reachable saving where the non-tail reference exceeds
16,384 entries; wall geometric and summed ratios ≤1.05; per-condition ratio
≤1.20 or added ≤0.25 ms; every expensive reference (≥50 ms, including tail)
latency ≤1.20× and nodes ≤1.50×; peak growth ≤5% or ≤32 KiB. If no budget passes,
public behavior remains unbounded. These are fixed engineering thresholds,
not statistical population claims or production endpoint percentiles.

The baseline is executed directly from Git's immutable object database.
Assertion-checked TT-only source substitutions generate A variants; all three
complete sources are saved beside the manifest. Source/design/runtime/fixture
hashes freeze before each run and drift/overwrite/result mismatches fail closed.
64 table constructions per implementation stabilize CPython split instance
dictionaries, including fresh workers. Seven complete unwrapped choose_move
wall/CPU samples follow one discarded warmup, with rotating variant order and
fresh tables. Ordinary timings include creation, all root work and table release.
Game creation, instrumentation, retained sizing and fresh-worker launch are outside.
A separately counted run verifies heuristic leaves; two separate tracemalloc
runs retain the table diagnostically. One fresh subprocess per condition/variant
reports before/retained RSS and RUSAGE_SELF high water. macOS ru_maxrss is bytes;
Linux is normalized from KiB. All saved decisions are checked, including their
own deterministic counters for bounded variants.

Reachable retained bytes count unique Python objects from the SearchTable itself,
its attribute dict, TT storage and referenced keys/entries. Shared cached integers
and strings count once: traverse each dictionary record's key before value,
then surrounding table attributes. Classes, module globals/code and unrelated
interpreter objects are outside this logical instance graph. `traced_current`
and `traced_peak` measure Python allocation during the diagnostic decision;
shared preexisting objects/import allocations need not appear there. RSS and
high water include the interpreter, imports and allocator; they are not TT bytes.
The harness imports common helper metadata for every variant, so fresh RSS does
not separately estimate a production helper-import cost. No RSS ceiling follows
from bounded entry count or traced peak. Normal search releases the table; retained
measurements are deliberate diagnostics, not evidence of a leak.

The process was not isolated from other laptop activity. An extended correctness
run briefly overlapped A; no samples were selectively discarded. Rotating matched
samples, CPU measurements and variability evidence describe that limitation.
B1 and B2 ran serially without backend tests overlapping them. See manifest/runtime
metadata for Python, platform, CPU count, host load and hardware; results can vary
with interpreter, allocator, frequency and scheduling.

The first preflight failed solely because JSON converted tuple score pairs to
lists. [serialization-preflight/](serialization-preflight/) preserves the original
harness, declaration, source snapshots and failure log (no completed result row).
The corrected harness uses JSON-native ordered pairs and has a round-trip regression
test. The same design/fixtures/thresholds were redeclared before the full run.


## Experiment A — unbounded storage

| Variant | Retained saving (28 large conditions) | Wall geometric overhead | Summed median overhead | CPU geometric overhead | Worst non-tail overhead | Eligible |
| --- | --- | --- | --- | --- | --- | --- |
| packed-key | 35.2% | +0.78% | +0.62% | +0.80% | +3.39% | yes |
| packed-entry | 56.0% | +4.83% | +4.63% | +4.87% | +7.03% | yes |

Packed keys alone are acceptable but save less. The combined entry encoding is selected by the predeclared memory rule. Its measurable CPU cost is an accepted tradeoff, not a speedup claim. The exact incremental evaluator from Phase 3B.1 remains unchanged; the earlier 1.83× evaluation speedup is preserved structurally rather than re-estimated here.

| Depth | Baseline summed wall ms | Packed key ms | Packed entry ms | Selected geometric ratio |
| --- | --- | --- | --- | --- |
| 4 | 15.599 | 15.711 | 16.279 | 1.0435× |
| 6 | 101.256 | 102.538 | 106.760 | 1.0521× |
| 8 | 503.156 | 506.393 | 526.831 | 1.0478× |
| 10 | 2025.441 | 2037.342 | 2118.189 | 1.0496× |

### packed-key: all per-position/depth latency comparisons

Every cell is baseline → variant milliseconds / slowdown ratio. Ratios below one are faster. All regressions, including tiny/endgame and tail rows, remain visible.

| Position | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| empty | 0.800 → 0.816 / 1.019× | 7.247 → 7.425 / 1.025× | 46.046 → 46.460 / 1.009× | 209.619 → 209.711 / 1.000× |
| near-opening | 0.887 → 0.905 / 1.020× | 6.882 → 6.949 / 1.010× | 39.549 → 39.749 / 1.005× | 168.581 → 169.586 / 1.006× |
| midgame | 0.531 → 0.531 / 0.999× | 1.909 → 1.926 / 1.009× | 4.686 → 4.769 / 1.018× | 9.475 → 9.671 / 1.021× |
| midgame-wide | 1.183 → 1.175 / 0.993× | 5.240 → 5.268 / 1.005× | 14.866 → 14.939 / 1.005× | 36.246 → 36.447 / 1.006× |
| late | 0.084 → 0.084 / 1.006× | 0.226 → 0.228 / 1.008× | 0.572 → 0.557 / 0.974× | 1.084 → 1.082 / 0.999× |
| immediate-win | 0.528 → 0.531 / 1.006× | 3.241 → 3.248 / 1.002× | 20.403 → 20.661 / 1.013× | 101.186 → 101.863 / 1.007× |
| forced-reply | 0.537 → 0.542 / 1.009× | 2.540 → 2.581 / 1.016× | 14.063 → 14.015 / 0.997× | 52.708 → 53.146 / 1.008× |
| double-threat-loss | 0.188 → 0.189 / 1.008× | 0.193 → 0.190 / 0.988× | 0.195 → 0.196 / 1.007× | 0.190 → 0.192 / 1.009× |
| dense-endgame | 0.021 → 0.020 / 0.992× | 0.022 → 0.022 / 1.013× | 0.022 → 0.022 / 0.989× | 0.022 → 0.022 / 1.011× |
| quiet-5 | 0.630 → 0.634 / 1.005× | 3.737 → 3.797 / 1.016× | 11.579 → 11.760 / 1.016× | 20.185 → 20.191 / 1.000× |
| supplement-main-003 | 1.147 → 1.144 / 0.998× | 9.133 → 9.361 / 1.025× | 42.170 → 42.627 / 1.011× | 157.318 → 160.216 / 1.018× |
| seeded-00 | 1.325 → 1.344 / 1.015× | 11.263 → 11.356 / 1.008× | 56.814 → 57.147 / 1.006× | 226.287 → 229.075 / 1.012× |
| seeded-01 | 0.264 → 0.266 / 1.007× | 0.835 → 0.854 / 1.024× | 3.087 → 3.091 / 1.001× | 9.369 → 9.472 / 1.011× |
| seeded-02 | 0.436 → 0.444 / 1.019× | 0.991 → 1.013 / 1.023× | 2.539 → 2.519 / 0.992× | 4.563 → 4.654 / 1.020× |
| seeded-03 | 0.293 → 0.296 / 1.010× | 0.886 → 0.882 / 0.995× | 3.213 → 3.277 / 1.020× | 10.904 → 11.046 / 1.013× |
| seeded-04 | 1.493 → 1.489 / 0.997× | 13.286 → 13.486 / 1.015× | 85.579 → 85.647 / 1.001× | 434.830 → 439.043 / 1.010× |
| seeded-05 | 0.759 → 0.769 / 1.014× | 4.328 → 4.412 / 1.019× | 14.639 → 14.782 / 1.010× | 50.939 → 51.263 / 1.006× |
| seeded-06 | 0.485 → 0.490 / 1.010× | 1.664 → 1.675 / 1.006× | 4.444 → 4.539 / 1.021× | 10.473 → 10.571 / 1.009× |
| seeded-07 | 0.273 → 0.276 / 1.009× | 0.556 → 0.558 / 1.003× | 0.977 → 0.979 / 1.002× | 1.420 → 1.469 / 1.034× |
| post-hoc-tail | 1.809 → 1.778 / 0.983× | 16.864 → 16.752 / 0.993× | 127.229 → 127.382 / 1.001× | 711.347 → 715.203 / 1.005× |
| additional-00 | 1.259 → 1.262 / 1.002× | 10.612 → 10.791 / 1.017× | 62.846 → 63.172 / 1.005× | 281.086 → 282.465 / 1.005× |
| additional-01 | 1.377 → 1.387 / 1.007× | 12.996 → 13.040 / 1.003× | 66.103 → 66.707 / 1.009× | 218.387 → 215.565 / 0.987× |
| additional-02 | 0.678 → 0.688 / 1.016× | 2.418 → 2.428 / 1.004× | 6.621 → 6.643 / 1.003× | 15.320 → 15.311 / 0.999× |
| additional-03 | 0.421 → 0.428 / 1.017× | 1.052 → 1.050 / 0.998× | 2.142 → 2.135 / 0.996× | 5.247 → 5.281 / 1.007× |

### packed-entry: all per-position/depth latency comparisons

Every cell is baseline → variant milliseconds / slowdown ratio. Ratios below one are faster. All regressions, including tiny/endgame and tail rows, remain visible.

| Position | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| empty | 0.800 → 0.838 / 1.046× | 7.247 → 7.708 / 1.064× | 46.046 → 48.148 / 1.046× | 209.619 → 218.663 / 1.043× |
| near-opening | 0.887 → 0.926 / 1.044× | 6.882 → 7.249 / 1.053× | 39.549 → 41.436 / 1.048× | 168.581 → 176.432 / 1.047× |
| midgame | 0.531 → 0.549 / 1.035× | 1.909 → 2.000 / 1.048× | 4.686 → 4.973 / 1.061× | 9.475 → 10.080 / 1.064× |
| midgame-wide | 1.183 → 1.220 / 1.031× | 5.240 → 5.522 / 1.054× | 14.866 → 15.566 / 1.047× | 36.246 → 38.378 / 1.059× |
| late | 0.084 → 0.087 / 1.043× | 0.226 → 0.239 / 1.056× | 0.572 → 0.598 / 1.046× | 1.084 → 1.145 / 1.057× |
| immediate-win | 0.528 → 0.561 / 1.062× | 3.241 → 3.393 / 1.047× | 20.403 → 21.518 / 1.055× | 101.186 → 106.218 / 1.050× |
| forced-reply | 0.537 → 0.559 / 1.041× | 2.540 → 2.663 / 1.048× | 14.063 → 14.658 / 1.042× | 52.708 → 55.504 / 1.053× |
| double-threat-loss | 0.188 → 0.196 / 1.044× | 0.193 → 0.201 / 1.045× | 0.195 → 0.200 / 1.025× | 0.190 → 0.195 / 1.029× |
| dense-endgame | 0.021 → 0.021 / 1.014× | 0.022 → 0.022 / 1.025× | 0.022 → 0.022 / 1.017× | 0.022 → 0.022 / 1.015× |
| quiet-5 | 0.630 → 0.662 / 1.050× | 3.737 → 3.953 / 1.058× | 11.579 → 12.341 / 1.066× | 20.185 → 21.306 / 1.056× |
| supplement-main-003 | 1.147 → 1.208 / 1.053× | 9.133 → 9.775 / 1.070× | 42.170 → 44.613 / 1.058× | 157.318 → 165.345 / 1.051× |
| seeded-00 | 1.325 → 1.384 / 1.045× | 11.263 → 11.828 / 1.050× | 56.814 → 59.264 / 1.043× | 226.287 → 234.514 / 1.036× |
| seeded-01 | 0.264 → 0.273 / 1.036× | 0.835 → 0.880 / 1.055× | 3.087 → 3.217 / 1.042× | 9.369 → 9.917 / 1.058× |
| seeded-02 | 0.436 → 0.459 / 1.052× | 0.991 → 1.046 / 1.056× | 2.539 → 2.686 / 1.058× | 4.563 → 4.806 / 1.053× |
| seeded-03 | 0.293 → 0.309 / 1.054× | 0.886 → 0.939 / 1.061× | 3.213 → 3.424 / 1.066× | 10.904 → 11.637 / 1.067× |
| seeded-04 | 1.493 → 1.563 / 1.047× | 13.286 → 14.042 / 1.057× | 85.579 → 89.265 / 1.043× | 434.830 → 454.878 / 1.046× |
| seeded-05 | 0.759 → 0.776 / 1.023× | 4.328 → 4.554 / 1.052× | 14.639 → 15.237 / 1.041× | 50.939 → 53.148 / 1.043× |
| seeded-06 | 0.485 → 0.506 / 1.043× | 1.664 → 1.733 / 1.041× | 4.444 → 4.694 / 1.056× | 10.473 → 11.085 / 1.058× |
| seeded-07 | 0.273 → 0.290 / 1.062× | 0.556 → 0.591 / 1.063× | 0.977 → 1.012 / 1.035× | 1.420 → 1.509 / 1.062× |
| post-hoc-tail | 1.809 → 1.832 / 1.012× | 16.864 → 17.549 / 1.041× | 127.229 → 132.478 / 1.041× | 711.347 → 742.543 / 1.044× |
| additional-00 | 1.259 → 1.314 / 1.044× | 10.612 → 11.206 / 1.056× | 62.846 → 65.351 / 1.040× | 281.086 → 294.481 / 1.048× |
| additional-01 | 1.377 → 1.423 / 1.034× | 12.996 → 13.567 / 1.044× | 66.103 → 69.311 / 1.049× | 218.387 → 227.279 / 1.041× |
| additional-02 | 0.678 → 0.712 / 1.050× | 2.418 → 2.551 / 1.055× | 6.621 → 7.036 / 1.063× | 15.320 → 16.115 / 1.052× |
| additional-03 | 0.421 → 0.442 / 1.049× | 1.052 → 1.096 / 1.042× | 2.142 → 2.261 / 1.055× | 5.247 → 5.533 / 1.054× |

### Real TT populations, memory and process measurements

| D10 position | Entries (all identical) | TT MiB baseline → key → entry | Traced peak MiB before → after | Retained process RSS MiB before → after | Process high water MiB before → after |
| --- | --- | --- | --- | --- | --- |
| empty | 38395 | 8.259 → 5.325 → 3.624 | 8.416 → 3.758 | 51.64 → 47.02 | 51.64 → 47.02 |
| near-opening | 28999 | 6.746 → 4.470 → 3.094 | 6.869 → 3.285 | 49.70 → 46.52 | 49.70 → 46.52 |
| midgame | 1881 | 0.438 → 0.305 → 0.193 | 0.449 → 0.206 | 42.61 → 42.28 | 42.61 → 42.28 |
| midgame-wide | 6730 | 1.632 → 1.090 → 0.714 | 1.656 → 0.798 | 44.05 → 42.84 | 44.05 → 42.84 |
| late | 328 | 0.070 → 0.046 → 0.028 | 0.076 → 0.034 | 42.27 → 42.34 | 42.27 → 43.17 |
| immediate-win | 18565 | 3.998 → 2.563 → 1.732 | 4.085 → 1.803 | 46.45 → 44.19 | 46.45 → 44.19 |
| forced-reply | 10297 | 2.200 → 1.419 → 0.923 | 2.261 → 0.962 | 44.23 → 42.95 | 44.23 → 42.95 |
| double-threat-loss | 38 | 0.010 → 0.007 → 0.004 | 0.014 → 0.009 | 42.27 → 42.53 | 42.27 → 42.53 |
| dense-endgame | 3 | 0.002 → 0.001 → 0.001 | 0.005 → 0.005 | 42.33 → 42.02 | 42.33 → 42.02 |
| quiet-5 | 4174 | 0.985 → 0.650 → 0.411 | 1.002 → 0.433 | 43.17 → 42.69 | 43.17 → 42.81 |
| supplement-main-003 | 28395 | 6.681 → 4.387 → 3.024 | 6.738 → 3.263 | 49.86 → 46.25 | 49.86 → 46.25 |
| seeded-00 | 34557 | 7.972 → 5.196 → 3.467 | 8.098 → 3.600 | 50.97 → 46.56 | 50.97 → 46.56 |
| seeded-01 | 1796 | 0.420 → 0.278 → 0.185 | 0.428 → 0.202 | 42.50 → 42.14 | 42.50 → 42.14 |
| seeded-02 | 835 | 0.206 → 0.141 → 0.090 | 0.213 → 0.106 | 42.28 → 42.06 | 42.28 → 42.06 |
| seeded-03 | 2112 | 0.489 → 0.319 → 0.205 | 0.498 → 0.219 | 42.77 → 42.19 | 42.77 → 42.19 |
| seeded-04 | 78484 | 17.338 → 10.947 → 7.415 | 17.436 → 7.692 | 62.05 → 52.31 | 62.05 → 52.31 |
| seeded-05 | 9961 | 2.201 → 1.416 → 0.915 | 2.226 → 0.955 | 44.33 → 43.36 | 44.33 → 43.36 |
| seeded-06 | 1960 | 0.455 → 0.316 → 0.198 | 0.467 → 0.212 | 42.75 → 42.28 | 42.75 → 42.28 |
| seeded-07 | 289 | 0.068 → 0.046 → 0.028 | 0.073 → 0.035 | 42.31 → 42.47 | 42.31 → 42.47 |
| post-hoc-tail | 119958 | 27.605 → 17.902 → 12.574 | 27.860 → 13.333 | 75.33 → 60.66 | 75.33 → 60.66 |
| additional-00 | 45821 | 11.211 → 7.505 → 5.336 | 12.123 → 6.606 | 55.55 → 49.83 | 55.55 → 49.83 |
| additional-01 | 39129 | 8.927 → 5.694 → 3.737 | 9.011 → 3.883 | 52.45 → 47.02 | 52.45 → 47.02 |
| additional-02 | 2979 | 0.731 → 0.490 → 0.334 | 0.762 → 0.403 | 42.84 → 42.39 | 42.84 → 42.39 |
| additional-03 | 1056 | 0.245 → 0.160 → 0.103 | 0.251 → 0.112 | 42.38 → 42.27 | 42.38 → 42.67 |

All depths, both trace repetitions, traced-current bytes, RSS-before measurements and raw timings are in [A-results.jsonl](A-results.jsonl) and [performance.csv](performance.csv). These real-population savings are not extrapolations from single shallow entry objects.

| Tail retained category | Tuple baseline bytes | Packed key bytes | Packed entry bytes |
| --- | --- | --- | --- |
| attribute_names | 333 | 333 | 333 |
| dictionary_structure_and_capacity | 5242960 | 5242960 | 5242960 |
| hints_flags_shared | 246 | 330 | 0 |
| key_bitboard_integers | 6335404 | 0 | 0 |
| key_mover_depth_integers | 84 | 0 | 0 |
| key_tuples | 8636976 | 0 | 0 |
| packed_entry_integers | 0 | 0 | 3142832 |
| packed_key_integers | 0 | 4798320 | 4798320 |
| search_table_attributes | 120 | 120 | 120 |
| search_table_object | 56 | 56 | 56 |
| stored_scores | 1052240 | 1052240 | 0 |
| surrounding_settings_counters | 165 | 165 | 165 |
| value_tuples | 7677312 | 7677312 | 0 |

The tail dictionary itself is 5,242,960 bytes in all three variants: 96 bytes of headers, 1,048,576 bytes of hash indexes, 2,878,992 bytes of occupied dense records and 1,315,296 bytes of unused dense-record reserve (174,762 available records / 262,144 sparse slots). Capacity is inferred from the exact CPython 3.11 64-bit general-key table size formula and checked against measured sys.getsizeof for every A table. Local pycore_dict.h establishes the header/record/index widths; a regression test checks several resize boundaries. There are no deletions in A, so dense-record count equals occupancy. The hash index is metadata for the entire allocated table; it is not counted again as extra unused bytes. [dictionary-layout.json](dictionary-layout.json) preserves every inference. Shared scalars explain why score/hint/category totals cannot be obtained by multiplying shallow sizes by entry count. SearchTable object and calibrated attribute dictionary are included explicitly.

Tail bytes per entry fall from 241.3 to 109.9. Retained TT 27.60 → 12.57 MiB (54.5% saving); traced peak 27.86 → 13.33 MiB. Median tail wall 711.3 → 742.5 ms (+4.4%). Do not compare this matched result causally with the previous phase’s 723.4 ms from another run. All tail variants retain 423,857 nodes, 119,958 entries, 47,944 hits, 93,664 loop cutoffs and 236,149 heuristic leaves, with move 4 and best heuristic score −4. This is not a solved forced loss.

### Microbenchmarks

| Variant | Loop | Geometric wall ratio to tuple baseline |
| --- | --- | --- |
| packed-key | creation | 1.436× |
| packed-key | lookup | 0.985× |
| packed-entry | creation | 1.435× |
| packed-entry | lookup | 0.981× |

[A-microbenchmark.json](A-microbenchmark.json) records seven rotating samples of 20,000 operations on real depth-six TT identities per fixture, after a warmup. The creation loop includes key construction plus hash/checksum; the successful lookup loop includes membership and retrieval, without entry decoding. Common loop/index overhead remains. These are explanatory microbenchmarks, not pure instruction timings or acceptance metrics; complete choose_move timings decide acceptance.

## Experiment B — independent deterministic replacement

B starts only after A completion and uses accepted packed keys/entries for its
unbounded reference. Parallel fixed-size key/value lists implement direct mapping.
A probe compares the complete arbitrary-width stored key before returning its
entry. Different-key writes evict; same-key writes count as replacements. Every
store rechecks its slot after recursive descendants may have replaced it; bounds
are overwritten with the current complete entry, never merged with a foreign key.
Misses resume normal search, without budget exceptions, unknown values or partial
root results. Capacities are slot counts and occupancy may be lower.

B1 was declared initially: `hash(full_key) % capacity`. B2 is a separate,
supplemental strategy declared in [mixed-index/DESIGN.md](mixed-index/DESIGN.md)
after interim B1 opening results exposed clustering. It changes only indexing to
the high log2(capacity) bits of `(hash(key) * 11400714819323198485) & (2**64-1)`.
Truncation chooses a slot; identity is never truncated. Full root equality and
all original thresholds/fixtures/samples remain mandatory. B2 does not retroactively
replace B1 evidence or enter A acceptance. The supplemental manifest freezes its
runner/design/input hashes before any B2 measurements. Python integer hashing
is deterministic on this recorded runtime.

Both strategies add Python get/store dispatch and, for B2, mixing arithmetic.
Measured total cost therefore includes dispatch/indexing/allocation/release as
well as work caused by losing cache entries. Occupancy, evictions and extra nodes
separately quantify cache effects; ratios do not isolate eviction CPU alone.


| Strategy | Capacity | Wall geometric ratio | Summed median ratio | CPU ratio | Retained saving (large reference) | Failed criteria | Default eligible |
| --- | --- | --- | --- | --- | --- | --- | --- |
| B1 modulo | 16384 | 1.580× | 2.255× | 1.580× | 92.5% | geometric_latency, summed_latency, per_condition, peak, expensive_latency, expensive_nodes | no |
| B1 modulo | 32768 | 1.678× | 2.269× | 1.677× | 85.9% | geometric_latency, summed_latency, per_condition, peak, expensive_latency, expensive_nodes | no |
| B1 modulo | 65536 | 1.830× | 2.264× | 1.829× | 72.9% | geometric_latency, summed_latency, per_condition, peak, expensive_latency, expensive_nodes | no |
| B2 mixed | 16384 | 1.207× | 1.202× | 1.207× | 70.9% | geometric_latency, summed_latency, per_condition, peak, expensive_latency | no |
| B2 mixed | 32768 | 1.278× | 1.152× | 1.278× | 53.0% | geometric_latency, summed_latency, per_condition, peak, expensive_latency | no |
| B2 mixed | 65536 | 1.404× | 1.128× | 1.403× | 30.1% | geometric_latency, summed_latency, per_condition, peak | no |

### B1 modulo: per-depth aggregate ratios

| Capacity | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| 16384 | 1.239× | 1.349× | 1.685× | 2.212× |
| 32768 | 1.394× | 1.421× | 1.749× | 2.287× |
| 65536 | 1.663× | 1.532× | 1.841× | 2.390× |

### B1 modulo: expensive D10 search behavior

| Position | Slots | Wall ms | Wall ratio | Occupancy | Evictions | Same-key replacements | Nodes | Node ratio | Extra nodes | Hits/probe | TT MiB | Peak MiB | RSS MiB | High water MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| empty | unbounded | 219.2 | 1.000× | 38395 | 0 | None | 128308 | 1.000× | 0 | 29.19% | 3.624 | 3.758 | 46.66 | 46.66 |
| empty | 16384 | 486.2 | 2.218× | 1059 | 93214 | 14 | 267222 | 2.083× | 138914 | 1.46% | 0.318 | 0.329 | 42.11 | 42.11 |
| empty | 32768 | 484.0 | 2.208× | 1310 | 91535 | 27 | 264914 | 2.065× | 136606 | 2.25% | 0.584 | 0.596 | 42.69 | 42.69 |
| empty | 65536 | 481.0 | 2.194× | 1626 | 89708 | 38 | 261900 | 2.041× | 133592 | 3.15% | 1.104 | 1.117 | 43.14 | 43.14 |
| near-opening | unbounded | 178.1 | 1.000× | 28999 | 0 | None | 99441 | 1.000× | 0 | 33.09% | 3.094 | 3.285 | 46.02 | 46.02 |
| near-opening | 16384 | 482.8 | 2.711× | 713 | 88813 | 38 | 270565 | 2.721× | 171124 | 1.75% | 0.297 | 0.307 | 41.95 | 41.95 |
| near-opening | 32768 | 493.5 | 2.771× | 945 | 87998 | 45 | 269306 | 2.708× | 169865 | 2.07% | 0.562 | 0.573 | 42.44 | 42.44 |
| near-opening | 65536 | 485.7 | 2.727× | 1063 | 86976 | 77 | 267742 | 2.692× | 168301 | 2.68% | 1.069 | 1.081 | 42.62 | 42.62 |
| midgame-wide | unbounded | 38.0 | 1.000× | 6730 | 0 | None | 19097 | 1.000× | 0 | 29.08% | 0.714 | 0.798 | 42.59 | 42.59 |
| midgame-wide | 16384 | 70.9 | 1.866× | 243 | 14174 | 1 | 36813 | 1.928× | 17716 | 1.60% | 0.267 | 0.275 | 42.22 | 42.22 |
| midgame-wide | 32768 | 71.3 | 1.877× | 330 | 13948 | 2 | 36556 | 1.914× | 17459 | 1.90% | 0.522 | 0.531 | 42.44 | 42.44 |
| midgame-wide | 65536 | 71.2 | 1.874× | 330 | 13948 | 2 | 36556 | 1.914× | 17459 | 1.90% | 1.022 | 1.031 | 42.80 | 42.80 |
| supplement-main-003 | unbounded | 163.8 | 1.000× | 28395 | 0 | None | 87810 | 1.000× | 0 | 36.53% | 3.024 | 3.263 | 46.41 | 46.41 |
| supplement-main-003 | 16384 | 700.7 | 4.278× | 234 | 150562 | 14 | 398885 | 4.543× | 311075 | 1.64% | 0.266 | 0.274 | 42.05 | 42.28 |
| supplement-main-003 | 32768 | 704.0 | 4.299× | 365 | 149741 | 24 | 397631 | 4.528× | 309821 | 1.97% | 0.524 | 0.533 | 42.33 | 42.33 |
| supplement-main-003 | 65536 | 699.3 | 4.269× | 472 | 148821 | 46 | 396171 | 4.512× | 308361 | 2.32% | 1.031 | 1.040 | 42.72 | 42.72 |
| seeded-04 | unbounded | 456.5 | 1.000× | 78484 | 0 | None | 253274 | 1.000× | 0 | 27.50% | 7.415 | 7.692 | 52.39 | 52.39 |
| seeded-04 | 16384 | 964.9 | 2.114× | 714 | 190897 | 80 | 512863 | 2.025× | 259589 | 1.67% | 0.296 | 0.305 | 42.06 | 42.06 |
| seeded-04 | 32768 | 977.4 | 2.141× | 1025 | 188231 | 115 | 509521 | 2.012× | 256247 | 2.50% | 0.566 | 0.576 | 42.33 | 42.33 |
| seeded-04 | 65536 | 974.9 | 2.136× | 1352 | 185574 | 158 | 505998 | 1.998× | 252724 | 3.40% | 1.087 | 1.098 | 42.95 | 42.95 |
| post-hoc-tail | unbounded | 740.7 | 1.000× | 119958 | 0 | None | 423857 | 1.000× | 0 | 28.55% | 12.574 | 13.333 | 60.38 | 60.38 |
| post-hoc-tail | 16384 | 1628.3 | 2.198× | 1110 | 292608 | 224 | 886773 | 2.092× | 462916 | 1.43% | 0.321 | 0.332 | 42.23 | 42.23 |
| post-hoc-tail | 32768 | 1638.8 | 2.212× | 1669 | 289705 | 294 | 878020 | 2.072× | 454163 | 1.89% | 0.606 | 0.619 | 42.36 | 42.36 |
| post-hoc-tail | 65536 | 1626.3 | 2.196× | 2083 | 287549 | 343 | 872152 | 2.058× | 448295 | 2.28% | 1.132 | 1.146 | 43.03 | 43.03 |

Largest observed B1 modulo bounded median: 1638.8 ms at post-hoc-tail D10 / 32768 slots. Largest individual timed sample: 1872.6 ms at near-opening D10 / 65536 slots. These are observed maxima, not worst-case guarantees.

### B1 modulo, 16384 slots: all position/depth latencies

| Position | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| empty | 0.840 → 0.951 / 1.132× | 7.623 → 9.878 / 1.296× | 47.590 → 83.087 / 1.746× | 219.196 → 486.215 / 2.218× |
| near-opening | 0.929 → 1.052 / 1.133× | 7.291 → 9.873 / 1.354× | 41.649 → 74.392 / 1.786× | 178.106 → 482.789 / 2.711× |
| midgame | 0.551 → 0.622 / 1.129× | 2.017 → 2.758 / 1.368× | 5.010 → 10.444 / 2.085× | 10.072 → 34.234 / 3.399× |
| midgame-wide | 1.235 → 1.337 / 1.083× | 5.523 → 6.803 / 1.232× | 15.572 → 22.506 / 1.445× | 37.979 → 70.881 / 1.866× |
| late | 0.089 → 0.161 / 1.800× | 0.245 → 0.386 / 1.578× | 0.588 → 0.918 / 1.561× | 1.166 → 1.738 / 1.490× |
| immediate-win | 0.556 → 0.667 / 1.199× | 3.405 → 4.707 / 1.382× | 21.637 → 39.290 / 1.816× | 106.659 → 256.017 / 2.400× |
| forced-reply | 0.561 → 0.636 / 1.134× | 2.682 → 3.306 / 1.233× | 14.479 → 21.427 / 1.480× | 54.665 → 108.900 / 1.992× |
| double-threat-loss | 0.194 → 0.258 / 1.327× | 0.197 → 0.258 / 1.311× | 0.193 → 0.260 / 1.347× | 0.192 → 0.257 / 1.339× |
| dense-endgame | 0.021 → 0.082 / 3.916× | 0.023 → 0.084 / 3.657× | 0.023 → 0.085 / 3.710× | 0.023 → 0.086 / 3.688× |
| quiet-5 | 0.658 → 0.721 / 1.096× | 3.927 → 4.895 / 1.246× | 12.254 → 21.260 / 1.735× | 20.873 → 43.375 / 2.078× |
| supplement-main-003 | 1.182 → 1.409 / 1.192× | 9.646 → 15.200 / 1.576× | 44.353 → 116.622 / 2.629× | 163.783 → 700.692 / 4.278× |
| seeded-00 | 1.410 → 1.591 / 1.129× | 11.873 → 15.301 / 1.289× | 59.604 → 104.556 / 1.754× | 235.800 → 605.793 / 2.569× |
| seeded-01 | 0.278 → 0.339 / 1.221× | 0.868 → 1.004 / 1.156× | 3.220 → 4.066 / 1.263× | 9.825 → 15.929 / 1.621× |
| seeded-02 | 0.462 → 0.542 / 1.175× | 1.031 → 1.154 / 1.120× | 2.596 → 3.528 / 1.359× | 4.806 → 8.464 / 1.761× |
| seeded-03 | 0.307 → 0.375 / 1.220× | 0.917 → 1.065 / 1.161× | 3.389 → 4.934 / 1.456× | 11.439 → 27.881 / 2.437× |
| seeded-04 | 1.544 → 1.658 / 1.074× | 13.923 → 17.499 / 1.257× | 89.143 → 138.540 / 1.554× | 456.461 → 964.922 / 2.114× |
| seeded-05 | 0.788 → 0.848 / 1.077× | 4.524 → 5.617 / 1.242× | 15.173 → 22.094 / 1.456× | 52.922 → 95.450 / 1.804× |
| seeded-06 | 0.505 → 0.574 / 1.137× | 1.731 → 2.356 / 1.361× | 4.617 → 9.539 / 2.066× | 11.010 → 36.875 / 3.349× |
| seeded-07 | 0.279 → 0.362 / 1.299× | 0.582 → 0.764 / 1.313× | 1.024 → 1.423 / 1.390× | 1.498 → 2.372 / 1.584× |
| post-hoc-tail | 1.852 → 2.071 / 1.118× | 17.437 → 21.633 / 1.241× | 132.074 → 211.801 / 1.604× | 740.701 → 1628.294 / 2.198× |
| additional-00 | 1.267 → 1.378 / 1.087× | 10.684 → 13.783 / 1.290× | 62.498 → 99.981 / 1.600× | 280.432 → 635.620 / 2.267× |
| additional-01 | 1.423 → 1.563 / 1.099× | 13.187 → 16.888 / 1.281× | 66.800 → 106.880 / 1.600× | 224.150 → 499.694 / 2.229× |
| additional-02 | 0.701 → 0.779 / 1.111× | 2.559 → 3.277 / 1.280× | 7.048 → 13.648 / 1.936× | 16.257 → 45.680 / 2.810× |
| additional-03 | 0.444 → 0.529 / 1.190× | 1.137 → 1.289 / 1.134× | 2.289 → 2.934 / 1.282× | 5.593 → 7.784 / 1.392× |

### B1 modulo, 32768 slots: all position/depth latencies

| Position | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| empty | 0.840 → 0.999 / 1.189× | 7.623 → 9.886 / 1.297× | 47.590 → 83.149 / 1.747× | 219.196 → 484.018 / 2.208× |
| near-opening | 0.929 → 1.115 / 1.201× | 7.291 → 9.869 / 1.354× | 41.649 → 74.732 / 1.794× | 178.106 → 493.507 / 2.771× |
| midgame | 0.551 → 0.679 / 1.232× | 2.017 → 2.767 / 1.372× | 5.010 → 10.527 / 2.101× | 10.072 → 34.094 / 3.385× |
| midgame-wide | 1.235 → 1.378 / 1.116× | 5.523 → 6.940 / 1.257× | 15.572 → 22.516 / 1.446× | 37.979 → 71.294 / 1.877× |
| late | 0.089 → 0.220 / 2.461× | 0.245 → 0.451 / 1.841× | 0.588 → 0.982 / 1.669× | 1.166 → 1.806 / 1.548× |
| immediate-win | 0.556 → 0.726 / 1.305× | 3.405 → 4.759 / 1.398× | 21.637 → 39.260 / 1.814× | 106.659 → 257.179 / 2.411× |
| forced-reply | 0.561 → 0.692 / 1.234× | 2.682 → 3.342 / 1.246× | 14.479 → 21.527 / 1.487× | 54.665 → 109.182 / 1.997× |
| double-threat-loss | 0.194 → 0.319 / 1.642× | 0.197 → 0.320 / 1.627× | 0.193 → 0.320 / 1.658× | 0.192 → 0.319 / 1.663× |
| dense-endgame | 0.021 → 0.143 / 6.837× | 0.023 → 0.145 / 6.281× | 0.023 → 0.145 / 6.336× | 0.023 → 0.146 / 6.271× |
| quiet-5 | 0.658 → 0.788 / 1.198× | 3.927 → 4.907 / 1.249× | 12.254 → 21.395 / 1.746× | 20.873 → 43.306 / 2.075× |
| supplement-main-003 | 1.182 → 1.460 / 1.235× | 9.646 → 15.188 / 1.575× | 44.353 → 116.227 / 2.621× | 163.783 → 704.033 / 4.299× |
| seeded-00 | 1.410 → 1.640 / 1.163× | 11.873 → 15.437 / 1.300× | 59.604 → 105.190 / 1.765× | 235.800 → 606.816 / 2.573× |
| seeded-01 | 0.278 → 0.400 / 1.437× | 0.868 → 1.025 / 1.180× | 3.220 → 4.146 / 1.288× | 9.825 → 15.640 / 1.592× |
| seeded-02 | 0.462 → 0.624 / 1.351× | 1.031 → 1.208 / 1.172× | 2.596 → 3.600 / 1.387× | 4.806 → 8.525 / 1.774× |
| seeded-03 | 0.307 → 0.435 / 1.415× | 0.917 → 1.123 / 1.225× | 3.389 → 4.932 / 1.456× | 11.439 → 27.699 / 2.421× |
| seeded-04 | 1.544 → 1.726 / 1.118× | 13.923 → 17.575 / 1.262× | 89.143 → 138.900 / 1.558× | 456.461 → 977.399 / 2.141× |
| seeded-05 | 0.788 → 0.916 / 1.163× | 4.524 → 5.680 / 1.255× | 15.173 → 22.142 / 1.459× | 52.922 → 95.472 / 1.804× |
| seeded-06 | 0.505 → 0.633 / 1.254× | 1.731 → 2.417 / 1.396× | 4.617 → 9.609 / 2.081× | 11.010 → 36.750 / 3.338× |
| seeded-07 | 0.279 → 0.411 / 1.475× | 0.582 → 0.802 / 1.378× | 1.024 → 1.419 / 1.386× | 1.498 → 2.277 / 1.520× |
| post-hoc-tail | 1.852 → 2.114 / 1.141× | 17.437 → 21.436 / 1.229× | 132.074 → 214.209 / 1.622× | 740.701 → 1638.763 / 2.212× |
| additional-00 | 1.267 → 1.432 / 1.130× | 10.684 → 13.770 / 1.289× | 62.498 → 99.787 / 1.597× | 280.432 → 637.694 / 2.274× |
| additional-01 | 1.423 → 1.641 / 1.154× | 13.187 → 16.630 / 1.261× | 66.800 → 106.176 / 1.589× | 224.150 → 508.617 / 2.269× |
| additional-02 | 0.701 → 0.836 / 1.193× | 2.559 → 3.323 / 1.299× | 7.048 → 13.311 / 1.889× | 16.257 → 44.552 / 2.740× |
| additional-03 | 0.444 → 0.597 / 1.344× | 1.137 → 1.331 / 1.171× | 2.289 → 2.964 / 1.295× | 5.593 → 7.795 / 1.394× |

### B1 modulo, 65536 slots: all position/depth latencies

| Position | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| empty | 0.840 → 1.133 / 1.348× | 7.623 → 9.970 / 1.308× | 47.590 → 82.941 / 1.743× | 219.196 → 480.984 / 2.194× |
| near-opening | 0.929 → 1.223 / 1.317× | 7.291 → 9.870 / 1.354× | 41.649 → 73.486 / 1.764× | 178.106 → 485.744 / 2.727× |
| midgame | 0.551 → 0.805 / 1.461× | 2.017 → 2.885 / 1.431× | 5.010 → 10.567 / 2.109× | 10.072 → 34.368 / 3.412× |
| midgame-wide | 1.235 → 1.507 / 1.221× | 5.523 → 7.017 / 1.271× | 15.572 → 22.650 / 1.455× | 37.979 → 71.190 / 1.874× |
| late | 0.089 → 0.341 / 3.814× | 0.245 → 0.454 / 1.853× | 0.588 → 1.104 / 1.877× | 1.166 → 1.905 / 1.634× |
| immediate-win | 0.556 → 0.843 / 1.516× | 3.405 → 4.831 / 1.419× | 21.637 → 38.982 / 1.802× | 106.659 → 255.958 / 2.400× |
| forced-reply | 0.561 → 0.813 / 1.449× | 2.682 → 3.511 / 1.309× | 14.479 → 21.609 / 1.492× | 54.665 → 108.139 / 1.978× |
| double-threat-loss | 0.194 → 0.440 / 2.263× | 0.197 → 0.440 / 2.235× | 0.193 → 0.431 / 2.235× | 0.192 → 0.437 / 2.277× |
| dense-endgame | 0.021 → 0.264 / 12.610× | 0.023 → 0.266 / 11.529× | 0.023 → 0.267 / 11.623× | 0.023 → 0.267 / 11.471× |
| quiet-5 | 0.658 → 0.844 / 1.284× | 3.927 → 5.008 / 1.275× | 12.254 → 21.527 / 1.757× | 20.873 → 43.554 / 2.087× |
| supplement-main-003 | 1.182 → 1.543 / 1.305× | 9.646 → 15.256 / 1.582× | 44.353 → 115.801 / 2.611× | 163.783 → 699.255 / 4.269× |
| seeded-00 | 1.410 → 1.756 / 1.245× | 11.873 → 15.480 / 1.304× | 59.604 → 104.670 / 1.756× | 235.800 → 608.652 / 2.581× |
| seeded-01 | 0.278 → 0.527 / 1.897× | 0.868 → 1.162 / 1.338× | 3.220 → 4.252 / 1.321× | 9.825 → 15.711 / 1.599× |
| seeded-02 | 0.462 → 0.690 / 1.494× | 1.031 → 1.332 / 1.293× | 2.596 → 3.730 / 1.437× | 4.806 → 8.617 / 1.793× |
| seeded-03 | 0.307 → 0.560 / 1.825× | 0.917 → 1.248 / 1.362× | 3.389 → 5.098 / 1.505× | 11.439 → 27.796 / 2.430× |
| seeded-04 | 1.544 → 1.844 / 1.194× | 13.923 → 17.678 / 1.270× | 89.143 → 138.769 / 1.557× | 456.461 → 974.898 / 2.136× |
| seeded-05 | 0.788 → 1.049 / 1.332× | 4.524 → 5.822 / 1.287× | 15.173 → 22.191 / 1.463× | 52.922 → 97.389 / 1.840× |
| seeded-06 | 0.505 → 0.761 / 1.507× | 1.731 → 2.520 / 1.456× | 4.617 → 9.688 / 2.099× | 11.010 → 36.936 / 3.355× |
| seeded-07 | 0.279 → 0.529 / 1.896× | 0.582 → 0.877 / 1.508× | 1.024 → 1.440 / 1.407× | 1.498 → 2.246 / 1.500× |
| post-hoc-tail | 1.852 → 2.249 / 1.214× | 17.437 → 21.464 / 1.231× | 132.074 → 215.882 / 1.635× | 740.701 → 1626.334 / 2.196× |
| additional-00 | 1.267 → 1.544 / 1.219× | 10.684 → 13.810 / 1.293× | 62.498 → 99.813 / 1.597× | 280.432 → 634.455 / 2.262× |
| additional-01 | 1.423 → 1.738 / 1.221× | 13.187 → 17.082 / 1.295× | 66.800 → 106.300 / 1.591× | 224.150 → 507.559 / 2.264× |
| additional-02 | 0.701 → 0.957 / 1.366× | 2.559 → 3.433 / 1.342× | 7.048 → 13.357 / 1.895× | 16.257 → 46.070 / 2.834× |
| additional-03 | 0.444 → 0.706 / 1.589× | 1.137 → 1.465 / 1.289× | 2.289 → 3.110 / 1.359× | 5.593 → 7.901 / 1.413× |

Complete occupancy, eviction/replacement counts, extra nodes, all counters, leaves, retained/current/peak memory, before/after RSS and high water appear in [B-results.jsonl](B-results.jsonl) and [performance.csv](performance.csv).

### B2 mixed: per-depth aggregate ratios

| Capacity | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| 16384 | 1.247× | 1.185× | 1.179× | 1.221× |
| 32768 | 1.401× | 1.255× | 1.223× | 1.241× |
| 65536 | 1.686× | 1.376× | 1.297× | 1.291× |

### B2 mixed: expensive D10 search behavior

| Position | Slots | Wall ms | Wall ratio | Occupancy | Evictions | Same-key replacements | Nodes | Node ratio | Extra nodes | Hits/probe | TT MiB | Peak MiB | RSS MiB | High water MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| empty | unbounded | 219.2 | 1.000× | 38395 | 0 | None | 128308 | 1.000× | 0 | 29.19% | 3.624 | 3.758 | 46.73 | 46.73 |
| empty | 16384 | 256.3 | 1.169× | 14774 | 26773 | 1001 | 133835 | 1.043× | 5527 | 26.43% | 1.166 | 1.221 | 43.14 | 43.14 |
| empty | 32768 | 247.9 | 1.131× | 22362 | 17851 | 1203 | 131291 | 1.023× | 2983 | 27.46% | 1.884 | 1.965 | 43.95 | 43.95 |
| empty | 65536 | 242.9 | 1.108× | 28711 | 10656 | 1353 | 129764 | 1.011× | 1456 | 28.12% | 2.776 | 2.878 | 44.83 | 44.83 |
| near-opening | unbounded | 177.4 | 1.000× | 28999 | 0 | None | 99441 | 1.000× | 0 | 33.09% | 3.094 | 3.285 | 46.16 | 46.16 |
| near-opening | 16384 | 211.8 | 1.193× | 13567 | 18368 | 2031 | 105717 | 1.063× | 6276 | 31.17% | 1.116 | 1.171 | 43.12 | 43.12 |
| near-opening | 32768 | 203.2 | 1.145× | 19099 | 11487 | 2179 | 102630 | 1.032× | 3189 | 31.85% | 1.717 | 1.792 | 43.66 | 43.66 |
| near-opening | 65536 | 196.5 | 1.108× | 23178 | 6680 | 2290 | 100994 | 1.016× | 1553 | 32.33% | 2.475 | 2.565 | 44.50 | 44.50 |
| midgame-wide | unbounded | 38.0 | 1.000× | 6730 | 0 | None | 19097 | 1.000× | 0 | 29.08% | 0.714 | 0.798 | 42.84 | 42.84 |
| midgame-wide | 16384 | 42.6 | 1.122× | 5253 | 1763 | 298 | 19370 | 1.014× | 273 | 26.91% | 0.589 | 0.613 | 42.45 | 42.45 |
| midgame-wide | 32768 | 42.0 | 1.107× | 5715 | 1266 | 306 | 19323 | 1.012× | 226 | 27.18% | 0.868 | 0.895 | 42.69 | 42.69 |
| midgame-wide | 65536 | 42.0 | 1.107× | 5975 | 984 | 317 | 19311 | 1.011× | 214 | 27.36% | 1.385 | 1.412 | 43.25 | 43.25 |
| supplement-main-003 | unbounded | 164.2 | 1.000× | 28395 | 0 | None | 87810 | 1.000× | 0 | 36.53% | 3.024 | 3.263 | 46.02 | 46.02 |
| supplement-main-003 | 16384 | 217.6 | 1.325× | 13093 | 22404 | 2796 | 103365 | 1.177× | 15555 | 32.50% | 1.069 | 1.119 | 43.02 | 43.02 |
| supplement-main-003 | 32768 | 199.9 | 1.217× | 18077 | 14590 | 2983 | 96748 | 1.102× | 8938 | 33.57% | 1.629 | 1.696 | 43.59 | 43.59 |
| supplement-main-003 | 65536 | 192.6 | 1.173× | 21663 | 9719 | 3041 | 93769 | 1.068× | 5959 | 34.03% | 2.353 | 2.432 | 44.39 | 44.39 |
| seeded-04 | unbounded | 453.3 | 1.000× | 78484 | 0 | None | 253274 | 1.000× | 0 | 27.50% | 7.415 | 7.692 | 52.28 | 52.28 |
| seeded-04 | 16384 | 603.8 | 1.332× | 16236 | 80869 | 2672 | 293308 | 1.158× | 40034 | 22.80% | 1.274 | 1.337 | 43.12 | 43.12 |
| seeded-04 | 32768 | 552.3 | 1.218× | 29489 | 59139 | 3178 | 272849 | 1.077× | 19575 | 24.31% | 2.352 | 2.461 | 44.11 | 44.11 |
| seeded-04 | 65536 | 524.9 | 1.158× | 44741 | 39275 | 3586 | 262589 | 1.037× | 9315 | 25.50% | 3.805 | 3.966 | 46.17 | 46.17 |
| post-hoc-tail | unbounded | 741.6 | 1.000× | 119958 | 0 | None | 423857 | 1.000× | 0 | 28.55% | 12.574 | 13.333 | 60.30 | 60.30 |
| post-hoc-tail | 16384 | 1002.1 | 1.351× | 16372 | 136680 | 4728 | 505512 | 1.193× | 81655 | 22.91% | 1.285 | 1.356 | 43.12 | 43.12 |
| post-hoc-tail | 32768 | 932.3 | 1.257× | 31904 | 108615 | 5512 | 472202 | 1.114× | 48345 | 24.68% | 2.516 | 2.635 | 44.38 | 44.38 |
| post-hoc-tail | 65536 | 876.0 | 1.181× | 54794 | 75921 | 6223 | 447609 | 1.056× | 23752 | 26.02% | 4.462 | 4.662 | 46.83 | 46.83 |

Largest observed B2 mixed bounded median: 1002.1 ms at post-hoc-tail D10 / 16384 slots. Largest individual timed sample: 1080.6 ms at post-hoc-tail D10 / 16384 slots. These are observed maxima, not worst-case guarantees.

### B2 mixed, 16384 slots: all position/depth latencies

| Position | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| empty | 0.817 → 0.927 / 1.135× | 7.595 → 8.226 / 1.083× | 47.651 → 51.856 / 1.088× | 219.175 → 256.268 / 1.169× |
| near-opening | 0.927 → 1.029 / 1.110× | 7.283 → 7.894 / 1.084× | 41.683 → 45.526 / 1.092× | 177.437 → 211.760 / 1.193× |
| midgame | 0.549 → 0.648 / 1.179× | 1.985 → 2.229 / 1.123× | 4.971 → 5.522 / 1.111× | 10.169 → 11.061 / 1.088× |
| midgame-wide | 1.238 → 1.391 / 1.124× | 5.560 → 6.164 / 1.109× | 15.736 → 17.266 / 1.097× | 37.952 → 42.591 / 1.122× |
| late | 0.090 → 0.156 / 1.735× | 0.249 → 0.331 / 1.331× | 0.591 → 0.711 / 1.203× | 1.151 → 1.320 / 1.148× |
| immediate-win | 0.562 → 0.656 / 1.167× | 3.401 → 3.668 / 1.078× | 21.571 → 23.377 / 1.084× | 106.933 → 122.219 / 1.143× |
| forced-reply | 0.575 → 0.655 / 1.139× | 2.672 → 2.873 / 1.075× | 14.475 → 15.579 / 1.076× | 54.214 → 60.155 / 1.110× |
| double-threat-loss | 0.206 → 0.278 / 1.353× | 0.198 → 0.279 / 1.405× | 0.198 → 0.270 / 1.362× | 0.205 → 0.276 / 1.349× |
| dense-endgame | 0.021 → 0.083 / 3.906× | 0.023 → 0.085 / 3.720× | 0.024 → 0.086 / 3.647× | 0.023 → 0.086 / 3.731× |
| quiet-5 | 0.659 → 0.758 / 1.152× | 3.931 → 4.349 / 1.106× | 12.166 → 13.307 / 1.094× | 20.692 → 23.384 / 1.130× |
| supplement-main-003 | 1.164 → 1.288 / 1.106× | 9.570 → 10.601 / 1.108× | 44.235 → 51.707 / 1.169× | 164.242 → 217.647 / 1.325× |
| seeded-00 | 1.408 → 1.517 / 1.077× | 11.861 → 12.690 / 1.070× | 59.647 → 66.124 / 1.109× | 235.643 → 274.013 / 1.163× |
| seeded-01 | 0.287 → 0.359 / 1.249× | 0.857 → 0.963 / 1.124× | 3.219 → 3.464 / 1.076× | 9.755 → 10.540 / 1.080× |
| seeded-02 | 0.466 → 0.546 / 1.172× | 1.033 → 1.144 / 1.108× | 2.600 → 2.826 / 1.087× | 4.742 → 5.246 / 1.106× |
| seeded-03 | 0.307 → 0.386 / 1.256× | 0.938 → 1.037 / 1.106× | 3.398 → 3.736 / 1.099× | 11.376 → 12.622 / 1.109× |
| seeded-04 | 1.520 → 1.682 / 1.107× | 13.849 → 15.124 / 1.092× | 88.708 → 102.362 / 1.154× | 453.319 → 603.835 / 1.332× |
| seeded-05 | 0.791 → 0.893 / 1.130× | 4.525 → 4.877 / 1.078× | 15.228 → 16.712 / 1.097× | 53.031 → 59.533 / 1.123× |
| seeded-06 | 0.510 → 0.600 / 1.177× | 1.723 → 1.921 / 1.115× | 4.649 → 4.944 / 1.063× | 11.041 → 12.077 / 1.094× |
| seeded-07 | 0.286 → 0.363 / 1.268× | 0.599 → 0.707 / 1.180× | 1.027 → 1.151 / 1.121× | 1.486 → 1.651 / 1.111× |
| post-hoc-tail | 1.844 → 2.009 / 1.090× | 17.434 → 18.864 / 1.082× | 131.639 → 150.038 / 1.140× | 741.605 → 1002.099 / 1.351× |
| additional-00 | 1.278 → 1.408 / 1.102× | 10.648 → 11.420 / 1.073× | 62.178 → 68.934 / 1.109× | 278.776 → 340.424 / 1.221× |
| additional-01 | 1.443 → 1.589 / 1.101× | 12.948 → 14.411 / 1.113× | 66.656 → 77.384 / 1.161× | 220.890 → 279.598 / 1.266× |
| additional-02 | 0.700 → 0.799 / 1.142× | 2.523 → 2.815 / 1.116× | 7.031 → 7.826 / 1.113× | 16.020 → 18.017 / 1.125× |
| additional-03 | 0.426 → 0.514 / 1.207× | 1.085 → 1.221 / 1.125× | 2.217 → 2.438 / 1.099× | 5.524 → 5.980 / 1.082× |

### B2 mixed, 32768 slots: all position/depth latencies

| Position | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| empty | 0.817 → 0.987 / 1.208× | 7.595 → 8.184 / 1.078× | 47.651 → 51.588 / 1.083× | 219.175 → 247.884 / 1.131× |
| near-opening | 0.927 → 1.103 / 1.191× | 7.283 → 7.886 / 1.083× | 41.683 → 45.257 / 1.086× | 177.437 → 203.191 / 1.145× |
| midgame | 0.549 → 0.704 / 1.282× | 1.985 → 2.243 / 1.130× | 4.971 → 5.573 / 1.121× | 10.169 → 11.101 / 1.092× |
| midgame-wide | 1.238 → 1.457 / 1.177× | 5.560 → 6.168 / 1.109× | 15.736 → 17.214 / 1.094× | 37.952 → 42.029 / 1.107× |
| late | 0.090 → 0.216 / 2.401× | 0.249 → 0.390 / 1.570× | 0.591 → 0.766 / 1.296× | 1.151 → 1.388 / 1.207× |
| immediate-win | 0.562 → 0.707 / 1.259× | 3.401 → 3.753 / 1.104× | 21.571 → 23.243 / 1.078× | 106.933 → 119.392 / 1.117× |
| forced-reply | 0.575 → 0.717 / 1.247× | 2.672 → 2.933 / 1.097× | 14.475 → 15.620 / 1.079× | 54.214 → 59.565 / 1.099× |
| double-threat-loss | 0.206 → 0.337 / 1.640× | 0.198 → 0.337 / 1.697× | 0.198 → 0.329 / 1.658× | 0.205 → 0.337 / 1.646× |
| dense-endgame | 0.021 → 0.143 / 6.762× | 0.023 → 0.145 / 6.381× | 0.024 → 0.146 / 6.176× | 0.023 → 0.146 / 6.364× |
| quiet-5 | 0.659 → 0.820 / 1.245× | 3.931 → 4.389 / 1.117× | 12.166 → 13.351 / 1.097× | 20.692 → 23.185 / 1.120× |
| supplement-main-003 | 1.164 → 1.339 / 1.150× | 9.570 → 10.575 / 1.105× | 44.235 → 50.852 / 1.150× | 164.242 → 199.857 / 1.217× |
| seeded-00 | 1.408 → 1.584 / 1.125× | 11.861 → 12.635 / 1.065× | 59.647 → 65.056 / 1.091× | 235.643 → 266.061 / 1.129× |
| seeded-01 | 0.287 → 0.415 / 1.446× | 0.857 → 1.042 / 1.216× | 3.219 → 3.516 / 1.092× | 9.755 → 10.581 / 1.085× |
| seeded-02 | 0.466 → 0.609 / 1.306× | 1.033 → 1.204 / 1.166× | 2.600 → 2.872 / 1.104× | 4.742 → 5.294 / 1.116× |
| seeded-03 | 0.307 → 0.451 / 1.469× | 0.938 → 1.114 / 1.189× | 3.398 → 3.827 / 1.126× | 11.376 → 12.621 / 1.109× |
| seeded-04 | 1.520 → 1.742 / 1.146× | 13.849 → 15.134 / 1.093× | 88.708 → 98.719 / 1.113× | 453.319 → 552.291 / 1.218× |
| seeded-05 | 0.791 → 0.953 / 1.205× | 4.525 → 4.950 / 1.094× | 15.228 → 16.717 / 1.098× | 53.031 → 58.393 / 1.101× |
| seeded-06 | 0.510 → 0.656 / 1.286× | 1.723 → 1.964 / 1.140× | 4.649 → 5.023 / 1.080× | 11.041 → 11.861 / 1.074× |
| seeded-07 | 0.286 → 0.426 / 1.488× | 0.599 → 0.760 / 1.269× | 1.027 → 1.223 / 1.191× | 1.486 → 1.718 / 1.157× |
| post-hoc-tail | 1.844 → 2.054 / 1.114× | 17.434 → 18.856 / 1.082× | 131.639 → 146.657 / 1.114× | 741.605 → 932.262 / 1.257× |
| additional-00 | 1.278 → 1.471 / 1.151× | 10.648 → 11.434 / 1.074× | 62.178 → 67.991 / 1.093× | 278.776 → 324.260 / 1.163× |
| additional-01 | 1.443 → 1.649 / 1.142× | 12.948 → 14.391 / 1.111× | 66.656 → 75.475 / 1.132× | 220.890 → 264.058 / 1.195× |
| additional-02 | 0.700 → 0.860 / 1.229× | 2.523 → 2.889 / 1.145× | 7.031 → 7.864 / 1.118× | 16.020 → 18.053 / 1.127× |
| additional-03 | 0.426 → 0.575 / 1.350× | 1.085 → 1.278 / 1.178× | 2.217 → 2.492 / 1.124× | 5.524 → 6.019 / 1.090× |

### B2 mixed, 65536 slots: all position/depth latencies

| Position | D4 | D6 | D8 | D10 |
| --- | --- | --- | --- | --- |
| empty | 0.817 → 1.107 / 1.355× | 7.595 → 8.320 / 1.096× | 47.651 → 51.704 / 1.085× | 219.175 → 242.927 / 1.108× |
| near-opening | 0.927 → 1.213 / 1.309× | 7.283 → 7.977 / 1.095× | 41.683 → 45.204 / 1.084× | 177.437 → 196.538 / 1.108× |
| midgame | 0.549 → 0.836 / 1.522× | 1.985 → 2.375 / 1.196× | 4.971 → 5.693 / 1.145× | 10.169 → 11.292 / 1.110× |
| midgame-wide | 1.238 → 1.578 / 1.275× | 5.560 → 6.302 / 1.133× | 15.736 → 17.353 / 1.103× | 37.952 → 42.022 / 1.107× |
| late | 0.090 → 0.337 / 3.747× | 0.249 → 0.511 / 2.056× | 0.591 → 0.887 / 1.500× | 1.151 → 1.520 / 1.321× |
| immediate-win | 0.562 → 0.832 / 1.480× | 3.401 → 3.852 / 1.133× | 21.571 → 23.386 / 1.084× | 106.933 → 117.774 / 1.101× |
| forced-reply | 0.575 → 0.845 / 1.469× | 2.672 → 3.062 / 1.146× | 14.475 → 15.714 / 1.086× | 54.214 → 59.464 / 1.097× |
| double-threat-loss | 0.206 → 0.463 / 2.252× | 0.198 → 0.450 / 2.269× | 0.198 → 0.451 / 2.273× | 0.205 → 0.455 / 2.218× |
| dense-endgame | 0.021 → 0.265 / 12.473× | 0.023 → 0.266 / 11.713× | 0.024 → 0.267 / 11.321× | 0.023 → 0.266 / 11.613× |
| quiet-5 | 0.659 → 0.937 / 1.423× | 3.931 → 4.502 / 1.145× | 12.166 → 13.401 / 1.101× | 20.692 → 23.268 / 1.125× |
| supplement-main-003 | 1.164 → 1.479 / 1.271× | 9.570 → 10.731 / 1.121× | 44.235 → 50.694 / 1.146× | 164.242 → 192.575 / 1.173× |
| seeded-00 | 1.408 → 1.706 / 1.211× | 11.861 → 12.753 / 1.075× | 59.647 → 64.937 / 1.089× | 235.643 → 260.562 / 1.106× |
| seeded-01 | 0.287 → 0.538 / 1.873× | 0.857 → 1.165 / 1.359× | 3.219 → 3.639 / 1.131× | 9.755 → 10.673 / 1.094× |
| seeded-02 | 0.466 → 0.750 / 1.609× | 1.033 → 1.333 / 1.291× | 2.600 → 2.999 / 1.153× | 4.742 → 5.439 / 1.147× |
| seeded-03 | 0.307 → 0.571 / 1.859× | 0.938 → 1.218 / 1.300× | 3.398 → 3.906 / 1.149× | 11.376 → 12.673 / 1.114× |
| seeded-04 | 1.520 → 1.852 / 1.219× | 13.849 → 15.279 / 1.103× | 88.708 → 97.805 / 1.103× | 453.319 → 524.927 / 1.158× |
| seeded-05 | 0.791 → 1.074 / 1.359× | 4.525 → 5.017 / 1.109× | 15.228 → 16.850 / 1.106× | 53.031 → 58.082 / 1.095× |
| seeded-06 | 0.510 → 0.775 / 1.521× | 1.723 → 2.104 / 1.221× | 4.649 → 5.158 / 1.109× | 11.041 → 12.137 / 1.099× |
| seeded-07 | 0.286 → 0.545 / 1.906× | 0.599 → 0.885 / 1.477× | 1.027 → 1.330 / 1.296× | 1.486 → 1.833 / 1.234× |
| post-hoc-tail | 1.844 → 2.174 / 1.179× | 17.434 → 19.089 / 1.095× | 131.639 → 143.848 / 1.093× | 741.605 → 875.994 / 1.181× |
| additional-00 | 1.278 → 1.599 / 1.252× | 10.648 → 11.613 / 1.091× | 62.178 → 67.541 / 1.086× | 278.776 → 313.089 / 1.123× |
| additional-01 | 1.443 → 1.755 / 1.216× | 12.948 → 14.612 / 1.128× | 66.656 → 75.424 / 1.132× | 220.890 → 256.449 / 1.161× |
| additional-02 | 0.700 → 0.982 / 1.403× | 2.523 → 3.027 / 1.200× | 7.031 → 7.931 / 1.128× | 16.020 → 18.144 / 1.133× |
| additional-03 | 0.426 → 0.710 / 1.667× | 1.085 → 1.406 / 1.296× | 2.217 → 2.614 / 1.179× | 5.524 → 6.124 / 1.109× |

Complete occupancy, eviction/replacement counts, extra nodes, all counters, leaves, retained/current/peak memory, before/after RSS and high water appear in [mixed-index/B-results.jsonl](mixed-index/B-results.jsonl) and [mixed-index/performance.csv](mixed-index/performance.csv).

Actual cache-hit rates use hits divided by TT probes, excluding terminal nodes and heuristic leaves. Separate instrumented runs in A/B-probe-rates.jsonl and mixed-index/B-probe-rates.jsonl reproduce the original vectors/counters/leaves; their timings never enter medians. CSV also reports the simpler hits/nodes ratio. [index-distribution.json](index-distribution.json) maps the complete unbounded D10 populations to each index scheme/capacity, retaining occupied-slot and maximum collision-bucket counts. This is a diagnostic of slot distribution, not a replacement for the measured bounded search.

**Keep bounded replacement experimental; do not enable it publicly.** B1's
low-bit clustering causes severe underoccupancy and repeated work; its result
must not be treated as the inherent cost of all bounded tables. B2 tests a better
distribution independently, but still fails the unchanged combined criteria.
Fixed slot lists also preallocate memory and impose construction/release costs
on tiny trees. A 65,536-slot table can allocate more than an unbounded tiny TT.
Smaller budgets trade memory for eviction/recomputation; a larger budget can
save too little or retain dispatch costs. Neither policy establishes a process
memory ceiling. Default storage remains an unbounded compact dictionary.

## Validation and limitations


Complete backend result: **2614 passed, 15 skipped in 107.54s (0:01:47)**. [backend-tests.txt](backend-tests.txt) preserves the full output. The suite includes Negamax/array oracle and incremental evaluation, MCTS, public API validation, concurrency gates, rate limits, history/replay, and evidence/provenance contracts. PyTorch-dependent skips retain the existing unavailable-dependency limitation.

[audit.json](audit.json) checks all 7,392 saved A/B1 decisions, the 80 reused prior-phase score/move/counter/leaf vectors, complete condition coverage, exact self-counter determinism, two-repeat retained sizes, capacities and source/hash/AST integrity. [mixed-index/audit.json](mixed-index/audit.json) independently checks B2 against the same copied immutable A input and selected production bodies. All 96 integrated-production conditions match selected A scores, moves, counters and leaves. The primary runners save 11,616 distinct decisions across A/B1/B2 (7,392 timed plus 4,224 diagnostic), excluding warmups/reference setup/microbenchmarks. The copied A input under mixed-index is not a second A experiment and is never double-counted in conclusions.

Backend runs use dotenv disabled, explanations disabled and an empty API key;
the global fixture forbids live HTTPS provider calls. No paid/provider calls,
production operations, merges or deployments occurred. Frontend/shared public
interfaces are unchanged, so frontend tests were not run. Python compilation
and Git whitespace checks pass. Earlier completed phase evidence is untouched.

These are fixed-depth laptop engineering results, with finite boards/samples,
one Python/runtime and one fresh RSS observation per condition. No strength,
endpoint throughput, concurrent RSS limit or worst-case latency claim follows.
Finite oracle depths are tractable; deeper comparisons use the pinned validated
algorithm plus exact vector/counter parity. B2 is explicitly a supplemental,
interim-evidence-motivated experiment, not an initially preregistered strategy.

## Reproduction

Use a new output directory; runners refuse evidence overwrite. Preserve the
immutable baseline commit in Git. Full source snapshots/hashes are recorded.

```sh
mkdir -p /tmp/negamax-tt-replay/mixed-index
cp docs/search-negamax-v2/transposition-table/DESIGN.md /tmp/negamax-tt-replay/
cp docs/search-negamax-v2/transposition-table/mixed-index/DESIGN.md /tmp/negamax-tt-replay/mixed-index/
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt declare --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt A --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt B --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt_mixed declare --directory /tmp/negamax-tt-replay/mixed-index --parent /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.benchmark_negamax_tt_mixed run --directory /tmp/negamax-tt-replay/mixed-index
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.diagnose_negamax_tt --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.audit_negamax_tt --directory /tmp/negamax-tt-replay
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m scripts.audit_negamax_tt --directory /tmp/negamax-tt-replay/mixed-index
PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs
```

RSS uses read-only process inspection (`ps`) and may need a local sandbox
exception. No service or network access is needed for benchmarking. Reports/CSV
are generated from complete saved evidence; no result-dependent fixture filters.
`python -m scripts.summarize_negamax_tt` regenerates this report after audit and
backend log creation. [timing-stability.json](timing-stability.json) retains
sample variability; [runtime-end.json](runtime-end.json) records host metadata.

## Recommended Phase 3B.3

Test move-only cross-depth hints and iterative deepening as separate ablations.
A hint cache may omit depth but must include both full boards and mover, validate
legality, and store only a move: it must never supply another depth's score or
bound. Keep score TT identity depth-sensitive. Compare direct fixed-depth search,
move-only hints, iterative deepening, then their combination on the same frozen
quiet/tactical/endgame distribution and separate tail. Continue all root moves
to exact scores at the requested final depth and preserve historical ties.
Predeclare complete-decision latency thresholds, ordering/node/leaf changes,
memory cost and exception rollback; oracle/vector correctness and the full backend
suite remain mandatory. Do not change depth limits or public agent settings.

Published phase commit: [search/negamax-v2 head](https://github.com/andrewcukierwar/board-game-ai-lab/commit/search/negamax-v2).
The final delivery records the immutable commit SHA and verified remote equality;
this branch link can move with subsequent phases.

