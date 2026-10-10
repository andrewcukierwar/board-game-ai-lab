# Negamax v3 research

2026-10-10. Existing worktree `research/negamax-v3`, starting main
`60b99b0d42145da149f433907585b809060c946d`. No merge/deployment/public-cap change.

| Experiment | Decision | Geometric speedup | Sum median wall/A | Broad retained/A |
| --- | --- | --- | --- | --- |
| Full mirror TT | reject | 0.884x | 1.124 | 0.956 |
| Depth>=3 mirror TT | reject | 0.966x | 1.010 | 0.959 |
| PVS | reject | 1.011x | 0.886 | 0.898 |
| Terminal parent proof | frozen, measurement pending | — | — | — |
| Trusted TT fast path | frozen, measurement pending | — | — | — |

The strongest validated engine remains the starting optimized direct
alpha-beta. Mirror saves substantial empty-opening work but adds overhead on
asymmetric boards; PVS improves several costly quiet trees but fails geometric
and regression gates. Thresholds were not changed. Negative sources/evidence
remain available in [mirror/REPORT.md](mirror/REPORT.md) and
[pvs/REPORT.md](pvs/REPORT.md).

Correctness:4,224 mirror and2,816 PVS decisions match all root vectors/moves;
each phase matches111 independent oracle vectors and112 inherited baseline
vectors. Focused89tests, provisional tuning22tests and isolated lock4tests pass.
Full backend integration gate awaits an eligible production-path candidate.

Complete-decision [profile](profile/REPORT.md): near-two has_four calls per node,
state play/undo dominant. Two independent bounded candidates are predeclared in
[tuning/DESIGN.md](tuning/DESIGN.md). Exact tree/TT payload parity is required.
No per-node microbenchmark or profile fraction serves as acceptance evidence.

All results: Apple M5 MacBook Pro, CPython3.11.17; paired warmed interleaved
complete decisions under shared lock; retained/traced memory separately. Other
OS/thermal activity and finite survival-conditioned histories limit claims.
No strength or measured Mac Mini latency claim. Depth10/12 feasibility follows
Phase C. [PROGRESS.md](PROGRESS.md) gives exact durable recovery commands.
