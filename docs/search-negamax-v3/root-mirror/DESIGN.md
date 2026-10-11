# Additional Phase E: initial-root-symmetry policy (research-only)

2026-10-10, before performance. Strongest baseline A/terminal-proof at full
SHA51bd80ed65413fe0d14dc6e6b4d04d60a3381fb6, source digest in manifest.
Original direct60b99b0 included only as a historical secondary reference.
This is a NEW conditional policy, not changed Phase A thresholds. Phase A full/
selective variants remain rejected. Production remains accepted terminal proof.

## Hypothesis and independent policy

Phase A actual opposite-orientation reuse:885full/319selective D10empty, zero
on near-opening,seeded04,tail. B/root-mirror selects once after initial terminal
check: only if BOTH full player bitboards equal their horizontal reflections,
enable remaining-depth>=3 canonical score keys/hint remapping for that decision.
Otherwise use direct keys, with no reflection call per node. A boolean branch,
root predicate and additional table attribute still cost time and must earn
eligibility. No depth gate tuning or fixture names. Every root full window,
center-first ties, bound/depth/mover identity and C terminal-parent proof remain.
Reflection/hints use independently validated Phase A lossless implementation.

## Prospective workload and exactness

Original32histories with tail separated, D4/6/8/10; add SIX legal root-symmetric
histories at4/8/9/16/17/25plies usingRandom20261013, repeated blocks[a,b,6-a,6-b]
for uniform a,b in0..3, plus center3for odd target. Reject full columns,
intermediate/final terminals and duplicate identities only; never timing/score
filtering. Freeze histories BEFORE preflight. Both movers represented. Original
cohort measures general cost; new cohort measures conditional-domain breadth.

Seven warmed rotated interleaved samples per engine/condition, one discarded
warmup,64table constructor warmups, two separate retained/traced runs. Include
complete choose_move/state/roots/predicate/TTdisposal. Original direct, accepted
terminal-proof and policy sources independently frozen. All ordered vectors and
moves must match; counters deterministic. Asymmetric decisions must keep exact
baseline tree counters and TT dictionaries (excluding new table option metadata).
Symmetric decisions may change tree/TT reuse, must match array oracles/root truth,
all bounds/hints/exception restoration. No public cap/contract changes. Policy
is research-only even if eligible; production integration would require a
separate scoped review of historical counter-ablation tests and full suite.

## Fixed conditional-policy gates against strongest terminal-proof

ALL required, declared before preflight:
- Original non-tail cohort geometric>=1.00; summed wall<=0.98*A,CPU<=0.99*A.
- Asymmetric cohort geometric>=0.985, every condition <=1.20*A OR added<=0.25ms.
- Root-symmetric broad domain (D8/10,>=2000A nodes): geometric>=1.25,
  >=75%faster; at least FOUR broad domain conditions required, otherwise
  insufficient evidence, never eligible.
- Every condition includingtail: wall<=1.20*A OR added<=0.25ms; ifA>=50ms,
  wall<=1.20*A AND nodes<=1.25*A.
- Retained/peak<=1.05*A OR added<=32KiB per condition; core retained sum<=1.02*A.
- All exact/oracle/bound/restoration checks and asymmetric TT/tree equality pass.
These criteria evaluate targeted savings plus general neutrality, not broad
unconditional1.05/1.10speed criteria from rejected Phase A. No threshold moves.
Raw generic starting-direct analysis is secondary; conditional-analysis.json
is the sole performance decision for THIS new policy.

## Budget and preflight

Total measured/diagnostic1200s, validation/report300s; each lockbatch<=450s.
Preflight all SIX NEW D10 baseline positions: one warmed untraced decision and
one traced decision each,<=180s. Estimate full target as1.5*(341.6s previous
original-cohort three-engine run + sum(30*baselinewall+6*baseline tracedwall)).
If >1200s, label incomplete/resource-rejected; do not cherry-pick survivors or
relax criteria. Primary runs only if forecast passes. Preflight not pooled.
Shared lock entire warmup/calibration/run; release between eight-position blocks
and final six-target block. All bytes/evidence/source hashes frozen and pushed
before preflight. Reproduction uses freshdirectory; exclusive writes never alter
frozen results. No lengthy tournament, native rewrite, deployment or strength claim.

## Exact command sequence

```sh
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.negamax_v3_root_mirror_study preflight
# Proceed ONLY if preflight.json passed=true; separate forecast data not pooled.
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 run --phase root-mirror --start 0 --stop 8
# Then8:16,16:24,24:32,32:38; release/reacquire each block.
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.benchmark_negamax_v3 audit --phase root-mirror
.venv/bin/python -m scripts.benchmark_negamax_v3 summarize --phase root-mirror
scripts/wait_benchmark_slot.sh .venv/bin/python -m scripts.negamax_v3_root_mirror_study analyze
```
