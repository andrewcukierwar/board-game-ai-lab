# Independent prelaunch review — Phase 4D.4I-A

The reviewer did not author the prospective design, interventions, evaluator or
analysis. This initial review read the final reports and independent reviews for
4D.4G, 4D.4H-B and 4D.4H-C, and explicitly selected project source files.
No neural training, self-play collection, checkpoint modification or measured
fixture/model evaluation was performed by the reviewer before the prospective
push. The launch freeze and completed experiment still require final review.

## Source findings

The new `Intervention` subclasses `V2Inference`, preserving the unchanged
PUCTSearch adapter path. It invokes base inference once and transforms either
the single prediction or each ordered batch prediction. There is no double
application of the 0.5 factor. Uniform policy gives equal mass on legal actions
and exact zero elsewhere; zero logits retain the effective uniform-softmax
contract. Learned-policy variants retain learned logits and policy exactly.
Learned-value variants retain the raw scalar exactly. The baseline path retains
all three output fields exactly.

The unchanged search assigns Q to the node's player to move, selects using
negative child Q and alternates signs on backup. Root neural value is diagnostic
only. Terminal leaves bypass inference and use exact terminal_value; no wrapper
is involved. The evaluator uses the existing H-C search and arena namespaces,
with identities independent of checkpoint, architecture and intervention.
The arena pairs agent ownership of X/O while respecting the actual opening turn.
Its per-game role streams start identically across variants; divergent game paths
naturally consume their streams differently.

Strict loading delegates to the established H-C loader with expected boundary
SHA-256, expected configuration and restore_global_rng=False; model freezing
uses the established architecture factory. The final ledger, source/runtime
freeze, prospective push and full preservation inventories remain to be checked.

## Independent correctness checks

A separate plain Python board implementation rebuilt all 628 development
fixture histories. All histories are legal and nonterminal. All tactical200
legal sets, immediate wins, opponent threats, safe responses, categories and
expected actions were re-derived without project rules and agreed. All solved300
and deep128 legal action vectors are complete, values are W/D/L, the state value
is the maximum legal-action value, and the optimal-action set agrees. Immediate
terminal winning moves have value +1. Counts/cluster counts are tactical200
200/71, solved300 300/102 and deep128 128/28. Exact deeper labels were checked for
internal consistency; no independent solver was run. Solved300 has 100 wins,
100 draws and 100 losses; deep128 has no losses.

Independent synthetic adapters checked A–E single inference, a two-state batch,
empty batches, masking a full column and head isolation. A forced immediate-win
position searched for 64 simulations under each intervention retained winning
child Q=-1 for the opponent-to-move terminal node, no root noise and 64 total
visits. These are correctness checks using a synthetic fixed-value adapter,
not measured checkpoint evaluation. A preliminary 16-simulation check did not
visit the low-prior winning column under B; the 64-simulation correctness probe
did. No source change followed from either probe.

One initial fixture-check assertion compared ordered optimal-action lists and
failed on a mirrored deep row with descending action order. Comparing action
sets, as the scientific definition requires, passed all rows.

## Scientific interpretation requirements

Retain per-seed effects. Bootstrap uncertainty over opening/ancestry families is
conditional on these fixed models; two training seeds provide little inference
over future training runs. Pair rows by explicit IDs. Keep mirror/shared ancestry
clusters intact and recompute row-weighted means with cluster multiplicities.
Keep both colors of each match opening together.

Every achievable WDL decrease is an exact mistake, including win-to-draw;
tactical labels cannot supply unavailable deeper WDL. Report fixed theoretical
win/draw/loss strata because the value comparison changes with the label mixture.
Replay outcome MSE and exact game-theoretic MSE target different quantities.
Generation 6 has no generation 7 evidence, so generation-5 prediction on frozen
generation-6 games is the available final prequential measure. Correlations with
confidence, ply or repetition do not establish why learning failed. A tactical
benefit does not establish a playing-strength benefit, and this reused development
bank does not establish unseen generalization.

## Unresolved access incident

The reviewer issued this overly broad read-only source search during initial
orientation:

```text
rg -n 'def load|predict|terminal|value|rng|seed|optimal|cluster' research/alphazero_resnet_selfplay/adapters.py research/alphazero_resnet_selfplay/evaluate.py games/connect4/alphazero*
```

The expanded directory scope traversed `games/connect4/alphazero_v2/frozen/solved-sealed.json`
and printed partial action/value text in a heavily truncated tool result. This
violates the user's no-sealed-access requirement. The reviewer immediately
reported it to the lead agent, who disclosed it to the user. No sealed bytes were
modified; no solving or model evaluation was performed on them; no evaluation
membership, intervention, statistical design or scientific inference used them.
Every subsequent review read is restricted to explicitly selected nonsealed
fixtures or source paths. This incident cannot be undone and remains an unresolved
protocol limitation even if all frozen-byte preservation checks pass.

Initial disposition: no material intervention or fixture-label defect identified.
This is conditional source/correctness review, not final execution approval or an
experimental result.
