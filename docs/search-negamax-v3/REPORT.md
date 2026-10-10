# Negamax v3 research

2026-10-10. Existing worktree `research/negamax-v3`, starting main
`60b99b0d42145da149f433907585b809060c946d`. No merge or deployment.

Phase A completed: **reject full and remaining-depth>=3 selective mirror TT**.
Both preserve exact decisions; full geometric speedup 0.884x / summed wall
1.124x baseline, selective 0.966x / summed wall1.010x. Symmetric empty opening
benefits substantially, asymmetric boards pay overhead. All declared gates
and negative results are preserved in [mirror/REPORT.md](mirror/REPORT.md).

Strongest validated production engine remains the starting optimized direct
alpha-beta. 4,224 audited Phase A decisions, 111 independent oracle vectors,
112 inherited baseline vectors and 89 focused tests pass. No depth cap change
or strength claim. Runtime: Apple M5 MacBook Pro, CPython3.11.17, shared-lock
paired warmed complete decisions; memory measured separately.

Phase B PVS is frozen and validated, awaiting the shared laptop lock for
measurement. See [PROGRESS.md](PROGRESS.md) for exact durable recovery steps.
Profile-guided optimization and depth10/12 feasibility follow B's checkpoint.
