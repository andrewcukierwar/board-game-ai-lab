# Negamax v3 research handoff

2026-10-10, America/New_York. Existing branch/worktree `research/negamax-v3`,
starting main60b99b0d42145da149f433907585b809060c946d. No merge/deployment/public
limit/API/frontend change; other worktrees and frozen canonical evidence untouched.

**Accepted production-path change: reuse the verified nonterminal-parent proof.**
At a child position only the player who dropped changed its stones; checking the
unchanged player again for four is redundant. Default arbitrary search entries
still check both sides. Production integration51bd80ed65413fe0d14dc6e6b4d04d60a3381fb6
matches the measured source exactly. Against original optimized direct60b99b0:
1.135x geometric complete-decision speedup,12.2% lower summed median wall/CPU,
1.130x held-out speedup, identical TT retained memory, trees, root vectors and moves.

All measurements below are paired warmed rotated complete decisions on Apple M5
MacBook Pro Mac17,2,10cores,32GB, native CPython3.11.17, with the mandatory shared
laptop lock. Seven samples and separate twice-traced retained-table captures.
Finite histories, OS/thermal activity and survival conditioning limit claims.
Memory is reachable/traced Python heap, not RSS or server/container RAM.

| Experiment | Baseline | Outcome | Geometric speedup | Sum median wall/reference |
| --- | --- | --- | --- | --- |
| Full mirror score TT | original60b99b0 | reject default | 0.884x | 1.124 |
| Mirror only depth>=3 | original60b99b0 | reject default | 0.966x | 1.010 |
| PVS integer scouts/re-search | original60b99b0 | reject default | 1.011x | 0.886 |
| Terminal-parent proof | original60b99b0 | accepted production | 1.135x | 0.878 |
| Trusted integer TT fast path | original60b99b0 | reject | 1.019x | 0.979 |
| Initial-root-symmetry policy | acceptedterminal-proof51bd80e | eligible research-only | 1.012x originalcohort;1.864x broad symmetric | 0.971 originalcohort |

Mirror reuse benefits the symmetric empty opening but unconditional reflection
adds overhead elsewhere. PVS lowers total costly-tree work and broad retained
memory10.2%, yet fails fixed geometric/regression gates. Trusted TT gains are too
small for eligibility. Every rejection/source/result remains preserved. Thresholds
were never moved. Profile-guided terminal proof passes every original fixed gate;
complete final TT dictionaries and all tree counters match on128conditions.

The additional conditional policy selects mirror keys only when BOTH initial
player boards are symmetric. All12broad symmetric cases improve; asymmetric
geometric cost0.85% stays within its separately predeclared neutrality tolerance.
Original-cohort retained TT falls3.8%. It is **research-only**, not a public default;
its targeted criteria are not relaxed Phase A gates. Production remains terminal
proof. Historical counter-ablation scope and a separately validated integration
remain future work; all frozen results and exact root contracts must be preserved.

## Deeper-search feasibility

| Laptop condition | Depth10 accepted seconds | Depth12 accepted seconds | D12 retained/peak MiB |
| --- | --- | --- | --- |
| empty | 0.193 | 0.592 | 12.35/13.25 |
| near-opening | 0.156 | 0.548 | 11.61/13.39 |
| seeded quiet | 0.396 | 1.766 | 42.06/53.24 |
| expensive tail | 0.656 | 2.907 | 54.52/56.49 |

D12 grows latency roughly3.1–4.5timesD10 on these expensive classes; endgames
can finish in tens/hundreds of microseconds. Deeper feasibility is highly
position-dependent. No worst-case, stronger-play or measured Mac Mini claim.
Conservative planning multipliers2/4/8 on the worst observed D12 give5.8/11.6/23.3s;
those are explicit throughput/concurrency assumptions, not server guarantees.
Two largest TT snapshots imply about109MiB retained plus process/allocator/data
and transient overhead. Public cap unchanged; no production benchmark performed.

## Validation and durable evidence

[Final evidence audit](validation/final-evidence-audit.json) verifies16,544recorded
vectors/moves and all saved raw hashes across five studies. Independent array
minimax checks tractable horizons and draw prefixes; largest phase129vectors.
Original A scores/stats match112inherited vectors; allfour frozen historicalD12
vectors/counters match. Terminal proof TT/tree audit128conditions and deeper
TT/tree audit12conditions are exact; root-policy asymmetric TT audit124conditions
is exact. Bounds, signed scores, hints, sentinel reflection, center ties and
recursive/root exception restoration remain tested. No mismatched score hidden.

Production integration full backend:2867passed,15Torch-dependent skips126.21s,
under shared lock with `PYTHON_DOTENV_DISABLED=1 EXPLANATIONS_ENABLED=false
OPENAI_API_KEY=`. Final full-branch suite: **2916passed,15Torch-dependent skips in123.95s**,
recorded in [backend-final.txt](validation/backend-final.txt), under the shared lock.
No frontend work, so frontend checks were unnecessary. No paid provider call.

Shallow/squashed-checkout loader tries immutable Git then a literal-SHA-256-pinned
committed source copy. Original tooling bytes remain archived. Certified metadata
substitutions leave every measured/acceptance function unchanged; altered gates,
unknown revisions and bad archive/baseline bytes fail closed.147focused checks
plus missing-Git source parity pass. Existing canonical evidence and tests remain.

Reports/designs/raw files:
[mirror](mirror/REPORT.md), [PVS](pvs/REPORT.md), [profile](profile/REPORT.md),
[tuning](tuning/REPORT.md), [deeper](deeper/REPORT.md),
[conditional policy](root-mirror/REPORT.md), [tooling compatibility](tooling-compatibility/REPORT.md).
Every premeasurement design and completed result was committed/pushed; WIP
checkpoints preserve intermediate batches. Shared lock was held through complete
warmups/processes and released between bounded batches; MCTS ownership respected.

## Principal checkpoints

| Purpose | SHA prefix |
| --- | --- |
| Initial design | 244b0c5 |
| Mirror frozen/completed | e45cc91/b133403 |
| PVS frozen/completed | 59b996f/1e515a9 |
| Profile/tuning frozen | 92a272f/ee9ec57 |
| Tuning accepted/rejected evidence | 116c3e2 |
| Validated production integration | 51bd80e |
| Deeper frozen/completed | 5e79a7d/0f742c2 |
| Tooling exact recovery | f5651b7 |
| Conditional policy frozen/preflight/completed | c38792d/6fdb058/211ac78 |

Final documentation/validation tip: resolve `git rev-parse HEAD` and compare
`git ls-remote origin refs/heads/research/negamax-v3`; full verified tip is in final
handoff. Do not confuse documentation tip with production implementation51bd80e.

## Unfinished scope and exact next action

All declared research studies are completed; no incomplete measured condition.
Research-only conditional-policy production integration, broader symmetric
prevalence/strength studies, higher horizons and server validation were not part
of accepted integration. No deployment/merge authorized. Further state-update
rewrites have no sufficiently small measured hypothesis to justify starting now;
small TT specialization was already rejected. Native rewrite/pruning not justified.

Recommended next direction: scope an optional initial-root-symmetry mode with
explicit historical counter-ablation compatibility, preserving exact vectors,
freeze design before timing, then full backend validation before promotion.
Alternatively predeclare a bounded play/undo optimization based on the measured
35.8% profiled self time, measuring whole decisions rather than microbenchmarks.

Resume in THIS worktree/branch: read PROGRESS, verify origin/ancestor/status, read
the chosen DESIGN/REPORT, create a NEW phase directory, copy immutable source/
inputs/manifest histories, declare/push prospective criteria, then acquire shared
lock using `scripts/wait_benchmark_slot.sh COMMAND...`. Never overwrite existing
condition files or another owner's lock. Frozen studies can replay into a fresh
--directory; scripts.resume_negamax_v3 completes only missing conditions. Exact
reproduction commands and declared resources are in each DESIGN.md.
