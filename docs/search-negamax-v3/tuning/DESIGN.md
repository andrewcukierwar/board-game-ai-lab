# Phase C: independent profile-guided per-node fast paths

2026-10-10, before candidate timing. Baseline optimized direct at
`60b99b0d42145da149f433907585b809060c946d`; Phase A/B both rejected. Profile
frozen at92a272f, actual8decisions7.48s, exact saved vectors/counters verified.
Has_four3,619,536calls for1,809,760nodes,15.9% profiled self time; terminal method
6.3%; play/undo35.8%; pack/unpack2.8%. Search body23.6% also includes lookup,
comparison and storage. Profile overhead cannot establish actual speedup.

## Independent hypotheses and scope

A/direct: unchanged immutable engine.
B/terminal-proof: every recursive/root move is made only after its parent was
verified nonterminal. Only the player who dropped changed its board. Therefore
the other player cannot have newly acquired four; descendant terminal checks
can inspect only the previous mover. An internal optional parent_checked flag
carries this proof; default arbitrary entries still inspect both players in
original precedence. No assumption about alternating piece counts. Every
terminal-before-leaf, draw, signed horizon value and root full window remains.
State representation/play/undo/heuristic/ties/cache keys/bounds are untouched.
This should remove approximately one has_four per searched child, including
leaves; actual complete-decision gain must exceed new parameter/branch costs.

C/trusted-tt: under existing internal guarantees that entries are nonnegative
packed integers and searched scores/moves are valid, use one dict.get, direct
integer field decoding, numeric flags, inline zigzag/column/flag packing.
Retain invalid flag-code3 rejection and all validated public helper codecs.
Zero-valued entries remain hits. Bounds are classified against original input
window exactly as before. No capacity/replacement/hash/depth change. This is a
lower-confidence hypothesis: helper time alone is small; combined lookup,
allocation/dispatch and string/codec validation avoidance may amortize enough.
Malformed externally injected noninteger entry behavior is outside this private
trusted cache invariant; public codecs keep their input validation unchanged.
No combined B+C unless independently justified later. Pick eligible fastest
summed wall; if none passes, retain A. No production integration before full suite.

## Workload and exactness

Identical32 immutable Phase A histories, including four held-out and labeled
post-hoc tail, depths4/6/8/10. Seven warmed paired interleaved samples, one
warmup per variant/condition,64 constructor warmups, two separate memory runs.
Wall/processCPU complete choose_move including all roots/state/TT disposal;
nodes/leaves/terminal/probes/hits/cutoffs and reachable/traced memory separately.
Each candidate must match EVERY ordered root vector/move AND all baseline
search counters, leaf/terminal counts and packed TT payloads. Independent array
oracles at tractable depths, arbitrary initial terminal perspectives, both movers,
all retained bound truth, unrestricted signed depths, injected exceptions and
caller restoration. Never turn a bound into exact without full verification.

## Fixed gates (all required)

Non-tail geometric wall speedup>=1.05; summed median wall<=0.95*A and CPU<=0.97*A.
Broad D8/10 with>=2000A nodes: geometric>=1.10 and>=70% faster. Held-out>=1.00.
Every condition incl.tail: wall<=1.20*A OR added<=0.25ms; expensive A>=50ms,
wall<=1.20*A AND nodes<=1.25*A. Every retained/peak<=1.25*A OR added<=64KiB;
broad summed retained<=1.15*A. Exact mandatory parity and independent tests.
No threshold changes after seeing results. Profile timings never acceptance.

## Budget, freeze and reproduction

Maximum measured/diagnostic1200s plus validation300s. Each shared-lock batch<=450s;
release between independent eight-position blocks. No competing CPU benchmark.
Freeze generator/test/source/design/manifest hashes and commit/push BEFORE timing.
Research sources separate from accepted production patch. Preserve rejections.

```sh
.venv/bin/python -m scripts.benchmark_negamax_v3 declare --phase tuning --variants direct terminal-proof trusted-tt
# Source inputs generated/frozen before declaration; commit and push everything.
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 run --phase tuning --start 0 --stop 8
# Repeat8:16,16:24,24:32, released/reacquired between blocks.
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 audit --phase tuning
.venv/bin/python -m scripts.benchmark_negamax_v3 summarize --phase tuning
.venv/bin/python -m scripts.negamax_v3_stability --phase tuning
# Full backend before any production-path validation, under shared lock:
# PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false OPENAI_API_KEY= .venv/bin/python -m pytest tests -q -rs
```

Use frozen checkpoint and fresh --directory with DESIGN.md AND input snapshots
for reproduction; exclusive outputs never overwrite evidence. TT-payload equality
requires separate capture audit before declaring eligible; root-score match alone
cannot hide changed trees. No frontend changes or public depth/contract changes.
