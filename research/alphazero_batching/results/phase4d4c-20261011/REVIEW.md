# Independent critical review — Phase 4D.4C

A separate reviewer agent inspected the implementation without editing it,
compared against the original search/collector, independently ran the initial
67 focused tests (67 passed in 4.97 s), and recalculated results from raw records.
It reviewed numerical behavior, RNG/tree ownership, benchmark fairness, memory,
provenance, preservation and the phase report. Parent implementation and reviewer
work were separate; this is an independent engineering review, not a third-party
certification or replication on separate hardware.

## Findings and disposition

1. **Complete-game numerical screens were not initially aggregated.** The raw
   matching records were present. `summarize.py` now reports trajectory/outcome
   differences per requested capacity and evidence source. All are zero.
2. **Pooling action disagreement could dilute a failing configuration.** Final
   postprocessing checks each requested capacity and temperature independently.
   All visit/action screens pass with zero differences.
3. **Primary repeated runs needed explicit record/RNG checks.** Final aggregation
   compares completed examples, sorted decisions and per-ID RNG hashes across
   repeated trials, including actual pool eight with sixteen games. All match.
4. **Requested sixteen fixed jobs actually has fourteen inputs; validation pool
   eight has only four games.** The report explicitly labels these limits. It
   does not claim frozen-network batch-sixteen validation or throughput. Actual
   pool eight is covered by the primary repeated sixteen-game collections.
5. **Observer duration includes suspension.** Attribution uses direct inference
   and generator-advance clocks plus driver residual, never summed observer times.
6. **Batch validation is structural, not full historical reachability.** Legal
   histories are a precondition for trusted-constructor bitboards. The report and
   README describe this explicitly; scheduler positions originate from legal
   play, are checked against the engine and replayed to verify complete labeling.
7. **Serial cancellation checks occur per move; batched also per simulation.**
   This remains included in timings and documented. No overhead is subtracted.
8. **Frozen batch-one games should retain deeper accounting.** The numerical run
   compares 517 full-tree decision hashes across sixteen matching completed games
   and all final RNG states, in addition to exact record equality.
9. **Complete-game root maxima are larger than fixed-input maxima.** Both are
   reported: logits 1.14440918e-5 / values 4.47034836e-7 / policies 9.34088538e-7
   for primary roots, all within the original limits. Fixed maxima are retained.
10. **Julia documentation has no retained source hash.** Final wording identifies
    hashes for the three Python repositories and the Julia documentation URL/method.

No numerical tolerance, behavior threshold, throughput target or memory gate was
weakened. Findings were resolved through stronger postprocessing/evidence and
accurate reporting, without changing the search mathematics or measured trials.

## Independent conclusions

The reviewer found no material scheduler/search defect. One suspended traversal
per game preserves tree sequentiality, root expansion, terminal backup, visit
budgets, root noise and action temperature. Waiting consumes no RNG draws.
Singleton inference delegates the original method; larger encoding preserves
signed zeros and legal-only float64 softmax.

The reviewer independently verified 20 primary trials, four per condition. Every
condition has 64 games / 2,272 plies / 581,632 simulations / 453,870 neural positions.
Aggregate speedups reproduce 0.975086 / 1.222033 / 1.394725 / 1.560888 at capacities
1/2/4/8. All four paired capacity-eight gains are 1.549624–1.574408, above 1.3.
Repeat records/decisions/RNG states match, as do all root visits and actions.

All 560 fixed prediction/search comparisons and all-eight-checkpoint matching
complete-game screens pass. The reviewer recalculated exact numerical maxima.
Every invocation preserves the same 150 official bytes/inventory entries.

The reviewer recalculated ten successful measurement invocations totaling
1,145.329056875 seconds / 19.088817615 minutes. Capacity-eight memory is
28.102775574 MiB traced / 354.21875 MiB lifetime RSS, max_active 8, within declared
64 MiB / 1 GiB gates. Observer-free confirmation is 1.547521424× with matching
record/RNG hashes. Fixed-work medians and profiling categories agree with the
report. No unresolved material issue remained in implementation or measurements.

Final full-regression result and acceptance are recorded in acceptance.json and
tests-final.txt; the reviewer was sent these final records before the commit.

## Final approval

After the full suite completed (776 passed, two optional skips), the reviewer
read the final ten-PASS acceptance table, decoded the compressed evidence,
independently rehashed all 150 official files, checked the twelve unchanged
baseline source files and original network AST sections, and matched delivered
implementation bytes to the primary executed-source archive. It approved commit
and push on the dedicated branch with no unresolved material or minor finding.
