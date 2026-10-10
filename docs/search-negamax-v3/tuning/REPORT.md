# Phase C: accept terminal-parent proof; reject trusted TT fast path

2026-10-10. Immutable original direct baseline
`60b99b0d42145da149f433907585b809060c946d`. Profile declaration92a272f,
final candidate design/sources ee9ec57, payload audit f48401d, WIP evidence141169b.
**Terminal-proof meets every fixed research gate. Trusted-TT does not.**
Production-path integration and full backend validation are a separate checkpoint.

| Variant | Geometric | Broad | Wall sum/A | CPU sum/A | Held-out | Broad retained/A |
| --- | --- | --- | --- | --- | --- | --- |
| terminal-proof | 1.135x | 1.138x | 0.878 | 0.878 | 1.130x | 1.000 |
| trusted-tt | 1.019x | 1.024x | 0.979 | 0.979 | 1.022x | 1.000 |

A summed median3.625s; terminal3.182s (12.2% lower); trustedTT3.549s. All128
conditions pass individual latency and memory limits for both. Terminal passes
all aggregate/broad/held-out gates; trusted fails geometric, wall/CPU sums,
and broad geometric, so is rejected despite small improvements. No thresholds
changed. [RESULTS.md](RESULTS.md) contains all per-position/depth numbers and
[analysis.json](analysis.json) every gate.

## Proof, implementation and exact validation

The parent is checked for terminal outcomes before searching descendants.
Only the player who drops changes its bitboard; the unchanged opponent therefore
cannot newly acquire four. An optional internal parent_checked argument passed
only by verified recursive/root transitions permits terminal_value to check
only the previous mover. Default arbitrary entry points retain both-player
checks and original precedence. This does not assume balanced stone counts or
legal alternating history beyond the already-verified nonterminal parent and
one legal drop. Draw/terminal-before-leaf/horizon scores remain exact.

No heuristic, state representation, play/undo, move ordering, root windows,
TT identity/payload, bound classification or tree changes. Function parameter
and branch overhead are included in unprofiled complete-decision measurement.
This is a correctness proof for eliminating a redundant test, not tactical pruning.

[Audit](audit.json):4224 recorded decisions, all exact vectors/moves and
repeatable per-variant counters;111 independent array minimax vectors match all
three;112 inherited baseline vectors/counters match A. [Payload audit](payload-audit.json)
compares complete final TT dictionaries on EVERY128condition, not samples:
all keys/packed values and insertion-order checksums match exactly. All measured
nodes/entries/hits/cutoffs AND diagnostic leaves/terminals/probes match A.
Retained bytes repeat and equal baseline. Focused preparation22tests passed,
covering initial terminal perspectives, arbitrary depth magnitudes, PVS
compatibility, retained bound truth and nested/root exception restoration.
Original focused89tests and inherited evidence checks remain active.

Trusted TT remains research-only. Its internal nonnegative-int/valid-score/move
invariants justify direct field accesses and numeric flags, while validated
public helper codecs stay unchanged. Malformed externally injected noninteger
cache entries are outside its private invariant. This extra specialization does
not earn sufficient measured gain and is not integrated or combined.

## Measurement and stability

Apple M5 MacBook Pro Mac17,2,10cores,32GB,CPython3.11.17. Shared lock across
complete warmup/calibration/measured batches;59.7/87.1/128.9/65.9s,341.6s total,
plus separate payload audit, within1200s. Released between independent blocks;
MCTS pilots owned intervening slots, no bypass or competing heavy acceptance
work. Seven warmed paired samples, rotated order;64 constructor warmups;
normal choose_move/state/all roots/fresh TT disposal inside wall/processCPU.
Separate twice-traced retained-table captures measure reachable bytes and peak,
not allocator/RSS. Every condition records runtime/load and owner metadata.

[Stability](stability.json): leave-one-repetition-out terminal geometric1.1340–1.1357,
wall sum0.8767–0.8787; all7omissions eligible. Trusted geometric1.0172–1.0215,
none eligible. This descriptive sensitivity is not a population confidence
interval. Fixed survival-conditioned histories and OS/thermal activity limit
external validity. No strength improvement or measured Mac Mini claim.

## Reproduction and integration

DESIGN.md commands with frozen checkpoint, copied DESIGN/input sources and a
new --directory; exclusive evidence writes forbid replacement. Source/generator/
test hashes are in manifest. Run full payload audit under lock before accepting
per-node parity. Recovery script resumes only absent declared conditions without
changing sample counts or completed evidence.

Next: checkpoint this accepted research result; copy exact terminal-proof-source.py
to production negamax_agent.py; run mandated full backend under shared lock with
live providers disabled; verify production bytes equal measured source; commit/
push production-path integration separately. Keep trusted TT, PVS and mirror
research-only. Then freeze depth10/12 feasibility design using terminal-proof
as strongest validated candidate and original direct as reference. No public
limit, contract, frontend, deployment, other worktree or canonical evidence edits.
