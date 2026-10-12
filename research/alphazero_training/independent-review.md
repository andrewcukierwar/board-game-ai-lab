# Independent critical review — Phase 4D.4F

Review performed independently of implementation, against the phase requirements,
the Phase 4D.4C–4D.4E reports, the new experimental modules, original runner and
artifact contracts, and adversarial scratch tests. No official checkpoint was
trained, no ablation was started, and the reviewer changed only this report.

## Findings raised during review

1. **Historical reader gates must survive optimized Python.** The first reader
   used `assert` for generation-20 identity, eligible seeds, replay window,
   parameter count and state attestation. `PYTHONOPTIMIZE` can remove assertions.
   Requested explicit exceptions. The historical loader itself already enforced
   original file, source, runtime, replay and state identities without relaxed
   flags; this finding concerned the additional import eligibility gates.
2. **Optimizer compatibility belongs to the model adapter.** The first lifecycle
   invoked its AdamW validator directly. Requested a factory restore/validation
   method so a future architecture can own optimizer grouping/state compatibility
   without restructuring the collector or generation lifecycle.
3. **Persist measured resource accounting.** Returned wall-clock diagnostics
   alone are lost if a process exits after publication but before writing its
   sidecar. Requested generation resource records inside the boundary, clearly
   separated from deterministic computational-state comparisons. Persistence
   completion time necessarily remains an external measurement.
4. **Cross-check semantic history counters.** Requested retained replay position
   counts and cumulative trainer steps be checked against generation summaries,
   rather than relying only on an integrity hash and aggregate counter sums.

## Assessed safeguards

- Historical conversion runs the pinned original strict loader in a separate
  interpreter, with the historical checkout first on its import path and bytecode
  writes disabled. Source verification is never bypassed. Conversion preserves
  weights, optimizer moments/groups, training counters, replay, sampling and
  augmentation streams; it records a fork and its historical parent.
- New collection freezes independent weights and uses the existing pool-eight
  independent-game collector. Matched arms share seed/generation/game identities;
  action schedules are explicit. Neither schedule changes normalized visit targets.
- Replay reconstruction checks the schedule assigned to each producing generation,
  exact visit budgets, legal histories, terminal outcomes, greedy action maxima,
  normalized targets and actor-relative labels. Legacy metadata remains tied to
  verified parent configuration. Whole-generation expiration remains unchanged.
- Boundaries enforce experimental format, source content, runtime, configuration,
  model/optimizer compatibility, RNG states and integrity. Recovery after a failed
  generation requires reloading a valid boundary. Complete-generation publication
  reuses the original no-overwrite validated temporary-file/hard-link primitive.
- A directory-fsync error after publication can leave a complete valid checkpoint
  with durability uncertain. The runner is poisoned; inspection/reload resolves
  which complete boundary exists. This is distinct from partial publication.
- Evaluation freezes an independent model and restores global RNGs. The adapter
  rejects unsupported architectures. No ResNet or automatic arm-selection policy
  is implemented.

## Final re-review and validation

All four raised findings are resolved. The historical subprocess uses explicit
exceptions; the factory owns optimizer restoration; finite per-generation
resource records are persisted; and loader checks bind retained-game positions,
cumulative trainer steps and the model-hash chain to history. Additional guards
reject publication anywhere beneath a campaign declaration, including paths
through symlinks. CLI arm-equivalence gates also use explicit exceptions.

The reviewer independently ran the final focused suite:

```text
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/Users/andrewcukierwar/.venvs/board-game-alphazero-m4/bin/python
  -m pytest -q tests/test_alphazero_experimental.py
56 passed in 43.99s
```

This independent run preceded a test-only isolation fix. The first broad
AlphaZero regression invocation reported **663 passed, 2 skipped, 1 failed**:
the experimental module fixture permanently set Torch native inter-op threads
and deterministic algorithms in the shared pytest interpreter, so the existing
historical fresh-process continuation test correctly rejected its incompatible
parent/child runtime identities. No historical contract or check was relaxed.

The same 56 experimental cases now reside in
`tests/alphazero_experimental_cases.py`. The default-discovered
`tests/test_alphazero_experimental.py` wrapper executes that explicitly named
file in a separate interpreter with all five thread controls pinned to one and
propagates a nonzero child result. It imports no Torch or experimental module
in the parent. The cases filename is outside default pytest test discovery;
the child explicitly selects only that file, so there is no wrapper recursion.
The reviewer independently verified collection: explicitly selecting both files
collects **56 case nodes and one wrapper node**. The ordinary regression suite
therefore retains all 56 checks through its one discovered wrapper without
changing the historical suite's native runtime. Implementation source identity
and final-source historical readiness are unaffected by this test-file move.
The reviewer inspected the author's final regression log: **609 passed,
2 skipped in 337.36 seconds**, including the wrapper that executes all 56 cases.
The two skips concern existing retained-v1 checkpoint fixtures. Separately,
engine/search/neural dependency checks report **307 passed in 3.54 seconds**.
The reduced outer pytest count reflects 56 former nodes becoming one wrapper,
not removal of checks.

These bounded scratch tests cover same-process and fresh-process exact resume,
pool-eight collection, repeated deterministic trajectories, global and independent
sampling/augmentation RNG restoration, interruption recovery at collection,
finalization, replay, update and persistence stages, serialization/validation/fsync
failures, complete atomic publication, altered game/schedule/budget/outcome/state
rejection, mixed legacy/extended replay and expiration, model-factory dispatch,
evaluation RNG neutrality and official-output/source-contract isolation. Equality
checks compare weights, complete optimizer state, replay records and digest,
counters, deterministic generation summaries and persistent RNGs. Wall time is
measured separately and cannot be deterministic across reruns.

The reviewer also inspected the real-seed readiness evidence. Both seeds preserve
2,048 replay games; retained positions are 68,856 (42) and 64,514 (314159).
Both arms preserve the original model and respectively 4,957/4,890 trainer steps,
and have identical initialization-state hashes within each seed. Readiness does
zero collection and zero optimizer updates. The reviewer rechecked the final
readiness evidence's execution digest:
`b8aae0d6e14292dad2d2bcdb86f6048fad93a5eeaa5d342d7b5c340e9fc2bde5`.
The before/after official inventories contain exactly 150 entries and are equal;
the preservation record also reports equality to Phase 4D.4E and the unchanged
declaration hash. Subsequent execution-source edits intentionally invalidate
strict continuation and require re-import.

The final phase report and proposed Phase 4D.4G design were also reviewed. One
prospective decision rule originally referred to undefined “material regression”
in strength support. This is resolved by explicit per-seed paired opening-family
score noninferiority bounds, a -5 percentage-point margin, fixed bootstrap draws,
and separate PASS/FAIL/INCONCLUSIVE outcomes. The two-generation, 64-game,
64-update pilot matches games and updates while recording unequal positions/time;
learning, tactical and supporting-strength endpoints remain separate. Its reused
development fixtures and two seeds limit generalization, and the confidence gates
may remain inconclusive. No material report overclaim or unresolved prospective
decision-rule ambiguity was found in the final version. Launch still requires
separate user approval and the committed exact prospective declaration.

**Disposition: PASS for implementation and bounded exact-recovery verification;
no unresolved material finding.** This supports preparing the separately approved
Phase 4D.4G pilot. It provides no learning-strength result and no tactical-safety
guarantee for extended exploration. Official artifact preservation must remain a
condition of each subsequent launch, and the actual ablation is not authorized
by this engineering review.
