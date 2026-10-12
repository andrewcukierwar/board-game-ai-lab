# Independent critical review

A separate review agent inspected the implementation, new regressions, benchmark
boundaries, source snapshots, preservation records and completed measurements.
It did not implement the backend or alter artifacts.

Findings and resolutions:

- **High: self-play capture assumed `__dict__`.** Immutable slots states failed
  example capture. Resolved with full canonical-state ownership comparison;
  historical engine strictness is preserved. Independently reproduced valid
  capture and stale-decision refusal. Complete-game and wrong-position-at-same-
  ply regressions cover it.
- **Medium: training source partition omitted bitboard.py.** The new module is
  imported on the training path even with engine defaults. Added it to the
  reporting partition; the entire execution closure remains enforced together.
- **Low: optimized Python could disable assertion gates.** Benchmark now refuses
  `python -O`. Its refusal was exercised separately.
- **Provenance completeness.** Designated runs retain all 62 execution-source
  hashes/snapshots, including engine, board and reused baseline tooling.
- **Prediction exactness.** Designated archived-checkpoint validation captures
  actual forward input/output bytes and exact legal-policy/value values.
- **Historical declaration regression.** Updated the obsolete current-source
  equality check to pin historical bytes/source digest and require source
  refusal. Reviewer independently verified pins against 2fbbdee. Frozen bytes
  were not changed or re-authorized.

Core review found Node semantics, PUCT arithmetic, value signs, move order,
noise, actions, architecture and checkpoint format preserved. State geometry
matches the pinned validated main reference. No forbidden search feature was
introduced. All delivered core state/search/encoding/capture files match the
measured source bytes.

Independent evidence checks confirmed the 672 paired validation searches,
1,045,908 nodes, 179,238 predictions, 10,138 transitions, 200 tactical rows and
four identical complete games per backend. All 150 campaign file hashes match.
Traced peaks decrease in all 112 probes. Fresh process RSS is labeled as lifetime
peak and is distinguished from traced bytes.

The reviewer approved **criterion 4 after investigation**, conditional on keeping
its qualification visible. The primary ten-trial `p95_pass: false` must remain:
one isolated wall/CPU outlier is consistent with a transient disturbance but
scheduling causation is unproven. The independent 40-pair repeat has 1.921× median
speedup and p95 ratio .528, with its slowest bitboard trial faster than its fastest
engine trial. This supports no persistent meaningful regression, rather than a
claim that the original strict tail screen passed.

No remaining implementation correctness or evidence-integrity issue was found.
Final acceptance additionally requires the root agent's completed test suite,
final artifact verification and Git diff review.
