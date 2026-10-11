# Phase D: depth10/12 complete-decision feasibility

2026-10-10, before measurement. Original optimized reference A:
`60b99b0d42145da149f433907585b809060c946d`. Strongest production-path validated B:
terminal-parent-proof at51bd80e (full SHA/source digests in manifest). Full backend
2867passed,15Torchskips; Phase C passed all speed/regression/memory/exact gates.
No public depth limit change, strength inference, merge or deployment.

## Hypothesis and fixed conditions

The accepted per-node gain should carry to depths10/12 with identical trees;
longer horizons remain materially more expensive despite the gain. Six frozen
histories from original32-position manifest, chosen before new timing: empty,
near-opening, seeded-04, post-hoc-tail, dense-endgame, heldout-03. D10 andD12.
The tail is an expensive diagnostic, not an acceptance aggregate override.
Original direct and exact accepted production source, seven warmed rotated
paired decisions,64constructor warmups, one discarded warmup, two separate
retained/traced runs; same complete choose_move wall/processCPU method as prior
phases. Nodes/leaves/terminal/probes/hits/cutoffs, TT entries, reachable retained
and traced peak/current, every complete root vector/move, runtime/lock metadata.

No new candidate/default acceptance gate: this is descriptive feasibility for
an already independently accepted implementation. Report every position/depth,
geometric and summed median ratios, D12/D10 growth, expensive classes, sample
stability limits. Root vectors/moves must equal A, trees/TT payloads identical.
Array minimax on tractable horizons/draw prefixes, all existing golden vectors,
and frozen v2 D12 direct vectors where available; no unpruned broad-D12 oracle
claim. Keep exact depth-sensitive scores, root ties, full roots and restoration.

## Resource plan and limit

Budget900s measured+diagnostic, validation/report300s; each whole shared-lock
batch<=450s. Two independent three-position blocks0:3 and3:6, release between.
Immutable previous D12 direct measurements (same source digest) forecast
roughly0.7s empty,0.6s near-opening,2s seeded-04,3.3s tail, <57MiB traced peak.
Those numbers are planning input only, not this phase's evidence. The2x safety
allowance fits900s for the declared paired timing/memory work. B's tree is known
identical from proof/Phase C. If actual budget exhausted, preserve incomplete
conditions WIP and do not claim complete feasibility. Do not terminate or remove
another lock owner. No heavy concurrent work. No native rewrite/tournament.

## Mac Mini interpretation

Only laptop measurements. Present sensitivity planning multipliers2/4/8 on the
largest observed candidate latency to cover unknown CPU throughput/concurrency.
These are assumptions, not measured or guaranteed server factors. Retained/
traced heap excludes interpreter, allocator/native buffers, RSS and concurrency;
RAM planning needs extra overhead and concurrent searches. No inspection/run of
Services checkout, containers, Tailscale, Render or deployment. Do not exposeD12
or infer stronger play from speed. Larger depths need separate bounded study.

## Reproduction

Freeze copied accepted input source, DESIGN/manifest/source hashes and commit/
push before measurement. Runner manifest explicitly fixes selected histories
and depths10/12 before that push (default declaration is4/6/8/10).

```sh
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 run --phase deeper --start 0 --stop 3
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 run --phase deeper --start 3 --stop 6
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 audit --phase deeper
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.audit_negamax_v3_payloads --phase deeper
.venv/bin/python -m scripts.benchmark_negamax_v3 summarize --phase deeper
.venv/bin/python -m scripts.negamax_v3_stability --phase deeper
```

Generic analysis gates against starting direct are diagnostic only here; no new
accept/reject default is decided by deeper results. Use frozen checkpoint and
fresh --directory with design/input snapshots; source hashes/exclusive writes
protect evidence. Missing conditions can be resumed without replacing completed
files through scripts.resume_negamax_v3 under shared lock.
