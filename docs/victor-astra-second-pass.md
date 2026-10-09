# Victor second pass: a narrow winning pairing cutoff

October 9, 2026. Started with a clean `victor-astra-breakthrough` checkout at
`ac02f237ee2c2518119e75255e359e1f08e7cbf5`; created `victor-astra-second-pass`
directly from it. All work was local to this MacBook. No prior benchmark, book,
oracle, AlphaZero artifact, feature flag, deployment or remote branch changed.

**Result: a modest, sound improvement, not a second breakthrough of the previous
magnitude.** Fresh difficult held-out accuracy rises from **73/75 to 74/75**;
including controls, **113/115 to 114/115**. Optimal-move proofs rise from 100 to
101, with **8.5% fewer nodes** across all 120 held-out inputs. Only the native
Claimeven cutoff changes in production. Scheduling, table replacement, budgets,
selection policy and public interfaces remain unchanged.

## Independent correctness assessment

I found **no counterexample or correctness defect** in the original engine under
its stated preconditions: a basic-valid, gravity-respecting standard position,
consistent counts/turn, nonterminal recursive search, and inputs through the
validated Python boundary. This is an audit and testing result, not formal
verification or a proof of historical reachability.

- Both null windows have the correct negamax interpretation. A completed child
  query `[-1,0]` proves/refutes a root win; `[0,1]` proves/refutes root non-loss.
  Fail-low/high values are bounds, not automatically exact values. Original
  window endpoints govern table updates. Narrowing a unit integer window either
  leaves it unchanged or immediately cuts off.
- The gravity encoding `p + mask` is injective for the mover-relative board;
  mirror minimisation preserves value. Entries compare full keys, including
  after recursive replacement. A collision loses information, never creates a
  proof. Cancelled ancestors store nothing; completed descendant bounds remain
  valid across slices. There is no mutable shared search table.
- Non-losing generation maintains the recursive no-immediate-win precondition.
  Two forced blocks, losing support moves and the two-empty-cell draw cutoff
  are valid under that invariant. Terminal inputs are handled before C search.
- Root intervals enclose exact move values even after interruption. The optimal
  criterion is `lower(move) >= max(upper)`. A safe draw with a possible win stays
  partial and has no exact root value. Fair geometric slices prevent one move
  monopolising initial search, but do not guarantee completion before the cap.
- ctypes declarations match the ABI; board/order/budget validation precedes the
  call. Allocation failure/zero entries return unavailable, and missing-library
  fallback, terminal handling, zero time and concurrent isolation pass. Memory
  is per invocation and freed; stack depth is bounded by remaining cells.
  Time remains cooperative, checked every 1,024 entries, not hard real time.

Evidence: the original 19 native tests passed before modification. Added tests
exhaustively traverse **4,946 distinct reachable nonterminal suffix states** from
32 fresh ten-empty-cell roots, comparing all returned intervals with the separate
exhaustive Python oracle. Budgets cycle through 0–66 nodes; table sizes 1, 2, 3
and 17 force collisions; sampled states also complete serial searches. Existing
mirror, concurrency, frozen-label and CL certificate tests remain intact.
Standalone ASan/UBSan checks cover 371 frozen plus 233 fresh labelled positions,
each under 20 mode/budget/table combinations, without findings. Strict-warning
`-O0` and `-O3` builds also pass all 371 frozen harness inputs. No claim is made
that sanitizers establish mathematical correctness or verify untested platforms.

## Implemented mathematical improvement

I re-read Allis's original thesis, especially §§4.1, 6.1 and 7.1–7.4, and inspected
the relevant diagrams/table. The original cutoff correctly uses Claimeven as
**White <= draw**, never as an automatic draw or Black win. The following narrower
condition justifies an actual Black win.

At a nonterminal **White-to-move** position, let `U` contain every empty even-row
square whose lower neighbour is empty, and `B` the existing Black stones. Require:

1. Every possible White four intersects `B ∪ U` (the existing full CL cover).
2. `B ∪ U` itself contains four connected squares.

**Constructive proof.** Pair each odd/even empty vertical pair, with upper in
`U`. The only remaining empty squares are the immediately playable even squares
at the bottom of the empty region in odd-height columns. White to move implies
an even number of empty cells, so these singleton squares have even cardinality;
pair them across columns in any fixed order. This partitions every empty cell.
Black always answers White in the same pair. A vertical response is directly
above White's move. A cross-column response was initially playable and remains
playable until its pair is consumed. Thus every response is legal and Black
owns every square in `U` by completion. White can never own a square in `U`, and
condition 1 therefore prevents White winning first. Condition 2 guarantees Black
wins by the time the board would fill. Hence `value(White) = -1`.

This also explains why the singleton pairs need no added cover claim: they supply
legal responses; only guaranteed CL uppers enter conditions 1 and 2. The new
predicate does **not** accept a winning Black pattern without full White coverage,
or apply on Black's turn. It extends neither the general CL/BI/VE theorem's scope
nor the uncertified nine-rule/composite system.

C now returns `-1` for this sufficient condition in either null window. When it
fails, the original upper-bound-zero cutoff remains restricted to `alpha >= 0`.
An exported test predicate exposes the same conjunction. No new certification
architecture, book entry or heuristic leaf value is introduced.

An independent cell-based implementation checks the conjunction on 800 fresh
endgames: **24 positives**, all exact White losses in the exhaustive Python
oracle. Exhaustive replay of the explicit pairing policy across all White choices
visits 75 policy states, with every Black response legal and every line ending in
a Black win. There are also **374 White wins** with condition 2 but without full
coverage, demonstrating that dropping condition 1 would be unsound.

## Evaluation protocol and limitations

Before changing algorithms, generated 1,200 seeded prefixes (`1029000..1030199`),
retaining 945 unique quiet states after exclusions. Prefixes have 6–24 stones:
four random nonterminal opening moves, followed by depth-4 or depth-6 play with
8% exploration, or depth-4 play with 35% exploration. Both immediate wins and
immediate opponent threats are excluded. Continuations avoid terminal moves to
reach the requested depth; this deliberately favours quiet analytical positions
and is **not** an unbiased gameplay distribution.

Canonical boards exclude every recorded history in the prior performance,
opening and Astra JSON artifacts, plus the explicitly decoded compact opening
book. For each colour, select 80 baseline probe failures and 40 successes by
canonical SHA-256 order; alternating hashes within each stratum assign development
and held-out splits. Each split has 80 targeted and 40 control positions, with
60 positions per colour. Whole-board canonical overlap is zero. Similar opening
families may still be correlated; canonical disjointness does not remove that.

The split was written before experimental variants or oracle labels. Its SHA-256:
`929e608a7b4bedba67c132f455c19e73a898c3b20c5946c05b30febadb5998a1`.
The unchanged independent C oracle uses 100M nodes per position, batches of eight,
180-second batch timeouts and at most four workers. It resolves 118/120 development
and 115/120 held-out states. All seven UNKNOWNs stay in resource measurements but
are excluded from accuracy and proof-success denominators. Runtime and C oracle
share bitboard ideas, which is why exhaustive Python and raw-cell policy checks
also matter.

The algorithm choice was recorded in `selection.json` before evaluating held-out
variants. The final configuration remains 10M nodes, a 16 MiB table and existing
root ordering/selection. No tuning followed held-out inspection. Raw histories,
labels, intervals, decisions, costs and strata are in [victor-second-pass/](victor-second-pass/).

| Set, labelled positions | Original optimal | New optimal | Original/new native optimal proofs |
|---|---:|---:|---:|
| Fresh development, 118 | 114 | 114 | 96 / 98 |
| Development targeted, 78 | 74 | 74 | 56 / 58 |
| Fresh held-out, 115 | 113 | **114** | 100 / **101** |
| Held-out targeted, 75 | 73 | **74** | 60 / **61** |
| Held-out controls, 40 | 40 | 40 | 40 / 40 |
| Held-out decisive moves, 68 | 66 | **67** | 58 / **59** |
| Prior Astra development, 300 | 299 | 299 | 297 / 297 |
| Prior Astra held-out, 298 | 297 | 297 | 296 / 296 |
| Frozen suite, 371 | 370 | 370 | 323 / 324 (+46 book hits each) |

Held-out phase accuracy: 6–8 stones **31/32 unchanged**; 9–12 stones **40/41 →
41/41**; 13–18 **21/21**, 19–24 **21/21**, both unchanged. White improves 54/56 →
55/56; Black stays 59/59. The held-out error change is one draw→loss removed;
win→loss stays zero and win→draw stays one. Four development win→draw errors
remain. One improved decision is insufficient evidence of a large strength gain.

Of 115 labelled held-out positions, all-move resolution rises **36 → 37**;
optimal-only resolution remains **64**, and unresolved optimality falls **15 →
14**. Better proofs need not resolve every move, or even change the selected move.
Among the 20 held-out positions exhausting the original full 10M cap, 15 have
labels: accuracy improves 13/15 → 14/15 and proofs 0 → 1.
In particular, the improved held-out decision is still heuristic: it now excludes
one proved losing move; its selected draw is not yet proved. The additional
optimal proof occurs on a different position.

## Experiments rejected and resource results

Development results at 10M nodes; node totals include the two UNKNOWNs:

| Variant | Optimal / 118 | Optimal proofs | Total nodes |
|---|---:|---:|---:|
| Original | 114 | 96 | 417.83M |
| Skip moves unable to beat an established draw | 114 | 96 | 411.82M |
| Two-slot table, retain expensive completed proofs | 114 | 96 | 401.10M |
| Two-slot table, prefer shallow positions | 114 | 96 | 400.39M |
| Focus + shallow table | 114 | 96 | 395.08M |
| Retain completed child refutations across cancellation | 114 | 96 | 417.76M |
| Safe-first root queries | 115 | 93 | 430.77M |
| Adaptive safe-first queries | 115 | 94 | 425.06M |
| **Winning CL cutoff only** | **114** | **98** | **373.46M** |

Table variants add overhead without more proofs; child-refutation retention adds
complexity for negligible savings. Safe-first variants gain one heuristic decision
but lose several proofs and consume more nodes. None is shipped. Cross-turn cache
reuse and broader strategy searches were not pursued after the narrower cutoff
showed a benefit; there is no claim that these alternatives are exhausted.

The retained cutoff reduces total nodes by **10.6% development, 8.5% held-out,
23.1% frozen, 16.0% prior development and 9.9% prior held-out**. These totals
include unknown positions, unlike accuracy denominators. Cache replacement and
capacity are unchanged; the gain is avoided search, not a larger table.

Serial local public-profile measurements on all 120 held-out positions, including
UNKNOWNs, with 0.4-second native and 1.0-second cooperative whole-move limits:

| Measure | Original | New |
|---|---:|---:|
| Native search median / p95 / maximum | 24.3 / 209.1 / 231.3 ms | 18.7 / 215.8 / 226.3 ms |
| Whole move median / p95 / maximum | 27.9 / 914.5 / 1,036.3 ms | 22.0 / 873.5 / 1,017.3 ms |
| Total process CPU time | 15.13 s | 14.31 s |
| Whole-move deadline hits | 5 | 4 |
| Peak worker RSS (Python, imports and table included) | 86.9 MiB | 86.3 MiB |
| Median / p95 nodes among completed optimal proofs | 409,161 / 5,779,959 | 375,914 / 4,988,246 |

The final instrumented trial is `*-public-timed.json`; earlier trials are retained.
The native p95 gets slightly worse: additional predicate work costs time when a
hard search still uses its entire node cap. End-to-end CPU falls 5.5%, and public
accuracy/proof counts match node-only results. The pure-Python strategic fallback
still dominates the slowest moves. These figures concern a deliberately harder
sample than the previous report's 42 ms p95, and must not be compared as a speed
regression against that easier distribution.

The table remains exactly 1,048,576 16-byte entries (16 MiB), with the same 64 MiB
configuration ceiling and per-call ownership; no retained cross-turn memory is
added. RSS is an observed process high-water mark, not a claimed memory-bound
improvement. Table hits fall 60.46M → 54.20M (17.73% → 17.36% of entered nodes),
and combined CL bound hits rise 6.35M → 9.07M. The table did not become more
selective: stronger cutoffs simply avoid work. Timing is local arm64 macOS with
Apple Clang 21 and CPython 3.11; it does not establish hosting performance.

**Budget sensitivity is a real limitation.** At 1M nodes, development accuracy
109/118 → 110/118 and proofs 54 → 59; held-out accuracy **103/115 → 102/115** despite
proofs improving 58 → 62. History `[4,1,5,5,3,6,6,2,0,3,2,2,2]` explains the latter:
the original heuristic happens to choose winning column 3 without a guarantee;
the new search proves draws at columns 5 and 1 and preserves a safe draw while
column 3 remains unresolved. Both prove the win at the retained 10M cap. This is
an information/selection tradeoff under truncation, not an invalid interval. I
kept the 10M profile and the guarantee-preserving selector; I did not tune on
held-out data to override proved safety with an uncertain win. Slower hosts can
still expose this limitation through the wall-clock cap.

## Remaining failures, integration and verification

Fresh held-out failure: `[4,6,5,2,2,5,2,2]` has winning column 1, while the engine
chooses drawing column 5 at 10M nodes. Fourteen labelled held-out positions lack
optimality proofs; five additional ground truths remain unknown. The previous
`[4,4,4,4,4,1,1,1,1]` missed win and frozen `e0-012` drawing-move failure also
remain. Thus the main challenge of reliably finding wins after a safe draw has
**not** been solved. The cutoff helps a sufficient subset, not arbitrary early
positions or composite-rule strategies.

No new match campaign was run: the evidence is oracle-judged decision quality
and proof efficiency, not demonstrated Elo or a significant gameplay-score gain.
The practical change is slightly stronger bounded play with less proof work.
Perfect play still requires coverage/proofs for hard off-book positions and
completion within available resources. General composite execution, White threat
strategies and hard real-time guarantees remain unproved/unavailable.

Build-time compilation, optional loading, Python fallback, per-call isolation,
research API/CLI, opening book, saved-game behaviour and disabled public feature
flags are preserved. The existing Linux Docker build recipe still compiles this
portable C11/POSIX code. Docker is installed but its daemon socket is absent:
**the Linux image build/smoke test and Render latency remain unverified**. No
service was started, deployment attempted or environment variable changed.

Validation: **1,473 passed, 15 skipped** in the exact requested full backend
command; 27 focused native/second-pass tests pass across their final runs.
`git diff --check` passes. Compiler settings/hashes and audit counts are recorded
in [validation.json](victor-second-pass/validation.json). Earlier artifacts and
the independent oracle are unchanged.

Reproduction (run from repository root; outputs go only to the new directory):

```sh
.venv/bin/python -m games.connect4.victor.native --build
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment build
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment label
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment eval --split dev --configs baseline cl_win
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment eval --split heldout --configs baseline current
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment eval --split heldout --configs current baseline --public --tag timed
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment eval --split frozen --configs baseline current
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment eval --split prior_dev --configs baseline current
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment eval --split prior_heldout --configs baseline current
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.second_pass_experiment summary
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q
git diff --check
```

The benchmark reconstructs the original C directly from the pinned base commit,
compiles temporary experiment libraries offline, and never changes runtime flags
or imports the reference oracle into production. Re-running generation should
be done only when intentionally reproducing this new evaluation, not while
preserving its committed artifacts as frozen future baselines.
