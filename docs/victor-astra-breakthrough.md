# Victor: bounded move proofs bridge the opening gap

October 8, 2026. Base: `5afadfcd8db9a3771714b2acde1e3c27cc35e69b` on
`victor-opening-strength-and-app`; clean starting tree verified. Work branch:
`victor-astra-breakthrough`. Nothing pushed, deployed, enabled, or changed in
AlphaZero, the opening book, the independent oracle, or previous benchmark artifacts.

**Result:** on fresh, quiet held-out positions with 9–18 stones, optimal decisions
increased from **212/233 to 232/233**. Restricting to positions where move choice
actually matters gives **108/129 to 128/129**. The main advance is practical
bounded proof search, not a general proof of the nine-rule system.

## Diagnosis and implemented algorithm

The old architecture made the difficult region largely unreachable:

- Python exact search refused positions with more than 24 empty cells. At 9–17
  stones, it did no search at all.
- Once admitted, it tried to solve every root move, in centre order, and threw
  away all completed root results if any later branch exhausted the budget.
- Strategic knowledge ran after exact search, not inside it. An exploratory
  cover could redirect a move, but could not preserve an exact winning value.

I initially considered a more extensive strategic leaf evaluator, White context
changes, and composite execution improvements. Development measurements instead
favoured a small native engine with a different proof objective:

1. Maintain a sound W/D/L interval `[lower, upper]` for every legal move.
2. In Negamax-4 order, ask whether a move **wins** using a null-window search.
   If it cannot win, separately ask whether it **avoids losing**.
3. Give unresolved moves 2,048-node slices, then multiply the allowance by four
   each sweep. Completed subtrees survive in a shared table between slices;
   cancelled ancestors never store a bound. One hard move cannot consume the
   entire budget before alternatives get a chance.
4. Stop when a move's lower bound reaches every move's upper bound. A single
   proved win is sufficient; all seven move values need not be solved.
5. If unfinished, retain intervals. Preserve a proved non-loss; otherwise remove
   proved losing alternatives when an unresolved alternative exists. The old
   strategic selector operates only within these permitted moves.

`native_search.c` is a runtime implementation derived from the existing Python
bitboard search, not a wrapper around `tests/victor_validation/c4_oracle.c`.
It keeps forced-block and losing-support pruning, threat-count ordering, and
mirror keys. The replacement table stores full position keys and both bounds;
collisions can discard knowledge but cannot create it. Every invocation owns its
memory, so concurrent searches do not share mutable search state.

**Semantics:** `move_proof` exposes all intervals and resource status. A completed
optimal-move proof permits `SolverResult.move_kind == 'exact'` and an exact root
value even when alternatives remain unresolved. `ExactResult.status == 'exact'`
continues to require *all* legal move values. A proved draw with unresolved
winning alternatives is `search_nonloss`, with a bound and **no exact root value**.
Incomplete branches never acquire heuristic values or a solved label.

## A sound strategic cutoff, and what remains unproved

The implemented cutoff is a narrow sufficient condition of the **existing**
CL/BI/VE theorem, without changing that theorem or its verifier:

- At a White-to-move node, collect every empty even square whose lower neighbour
  is empty. These are the uppers of mutually disjoint Claimevens.
- If those upper squares and existing Black stones intersect every White group,
  the Claimevens cover all White targets. The established theorem gives
  `value(White) <= 0`.
- Use this only to refute a search asking whether White wins (`alpha >= 0`).
  Returning zero here is an **upper bound**, not a draw or a Black-win claim.

The bit implementation asks whether any four remains in the board after removing
Black stones and these upper squares. This is equivalent to checking the 69
windows. Independent raw-cell certificate verification checks positive instances
in tests; independent exact values check its consequences in search. Neither
nine-rule covers nor White threat contexts enter the proof engine.

I reviewed Allis Chapters 6–9, including the §7.4 table visually, and the preserved
Baseclaim case. No general composition theorem was obtained:

- §7.1's even-release argument must apply to the **union of accessible regions**,
  not a sum of rule footprints. Shared Before/Aftereven components and inverse
  residuals make an additive parity argument insufficient.
- Aftereven/Before timing clauses can remain live after their component responses
  are discharged. Local response availability alone does not establish their
  global timing guarantee.
- Lowinverse/Highinverse change the obligations in another column. Initial
  pairwise compatibility does not by itself prove spare existence after all such
  transitions, especially with shared Before components.
- Baseclaim and Specialbefore require role-sensitive responses. The existing
  retiring-spare repair is justified locally by permanently blocking **every**
  original group of each affected rule. It does not establish spare existence
  for every reachable residual collection. The preserved Baseclaim example and
  its legacy failure still pass the original complete policy audit tests.
- §8.1's argument that a full cover defeats White's zugzwang threat presupposes
  successful execution of the rules; it cannot substitute for that missing
  constructive argument.

Thus no composite subset was promoted to a new general theorem. All conditional
execution, nine-rule witness verification, compatibility constraints and research
labels remain intact. Native mode re-proves each turn, like the stateless public
adapter; original retained certificate/policy execution remains available with
`native=None`, and is preserved when the native library is unavailable.

**White:** the detector's nonplayable odd hole, adjacent odd/even threat-combination
holes, reserved columns, and playable-bottom exception agree with §§8.2–8.4.
Recognition is not the missing execution proof. Broadening detection or treating
“unrefuted” as winning would be unsound. I did not add a reserved-column executor
or change the White heuristic. Search now resolves the known `e0-006` draw and
chooses its sole drawing column, 1. White held-out accuracy rose from 142/156 to
156/156. On that sample the final selector needed no exploratory White decision.
This supports prioritising proof search, not a claim that White contexts can
never be useful.

## Evaluation design and independent validation

Before tuning, I froze 600 quiet positions from random/epsilon-Negamax-2/4
prefixes, primarily 9–18 stones, with both colours and later-game coverage.
Canonical mirrored board states, not seeds or histories, determine uniqueness.
A hash-ordered alternating split assigns 300 development and 300 held-out states.
Prior recorded histories are excluded. No positions were added to the book.

A tooling correction deserves disclosure: the initial generic history walker
missed the book's compact row format. An explicit subsequent canonical audit
found **zero** book overlap and zero cross-split overlap; the generator now
correctly decodes those rows. The split and all decisions stayed unchanged.
See [split-audit.json](victor-astra/split-audit.json).

The unchanged C oracle generated labels before algorithm tuning, with a
20-million-node limit per position. All 300 development positions and 298/300
held-out positions completed. The two unknowns remain in the artifacts and the
latency sweep, but **not** in accuracy denominators. They are not assumed solved.

The runtime and C oracle use related bitboard techniques, so agreement alone
would not establish algorithmic independence. Validation includes:

- Frozen pre-implementation exact outputs, including the unchanged 371-position
  suite; every reported runtime interval contains its independently recorded
  move value.
- Forty fresh endgames checked move-for-move against the separate exhaustive
  Python oracle, with the CL cutoff both enabled and disabled.
- Interrupted searches at 0, 1, 128, 2,048 and 200,000 nodes, including deliberate
  one-entry/17-entry table collisions, against frozen exact values.
- Independent CL certificate checks, mirror validity, deterministic node-only
  replay, concurrent isolation, zero time, missing-library fallback, the known
  White failure, and explicit partial-nonloss labels.

No oracle code or frozen benchmark was modified. This is broader than
self-agreement, but it is not a formally verified C implementation.

## Decision results

The fixed final configuration is ten million native nodes, 1,048,576 table slots,
CL cutoff enabled, fair threshold scheduling, original depth-4 ranking and
strategic fallback. Research comparisons use **node-only** limits. The public
profile additionally caps native search at 0.4 seconds and the move at a
cooperative 1.0-second deadline. Baseline uses the prior node-only profile and
reproduces its frozen score, 312/371.

| Set | n | Previous optimal | New optimal |
|---|---:|---:|---:|
| Development, quiet | 300 | 280 (93.3%) | 299 (99.7%) |
| Held-out, quiet | 298 | 277 (93.0%) | 297 (99.7%) |
| Held-out, quiet, 9–18 stones | 233 | 212 (91.0%) | 232 (99.6%) |
| Held-out, decisive, 9–18 stones | 129 | 108 (83.7%) | 128 (99.2%) |
| Held-out, decisive, all phases | 161 | 140 (87.0%) | 160 (99.4%) |
| Held-out, White / Black | 156 / 142 | 142 / 135 | 156 / 141 |
| Held-out, 19–25 / 26–33 stones | 37 / 28 | 37 / 28 | 37 / 28 |
| Frozen suite | 371 | 312 (84.1%) | 370 (99.7%) |
| Frozen suite, 9–18 stones | 149 | 90 (60.4%) | 148 (99.3%) |

The all-position fresh scores include equal-valued positions; the decisive rows
avoid that inflation. These are sampled distributions, not uniform coverage of
Connect 4. Held-out values improved in 21 cases, worsened in zero; 20 previously
suboptimal decisions became optimal. Shared opening structure can still correlate
positions even after canonical deduplication.

| Held-out error | Previous | New |
|---|---:|---:|
| Win → draw | 6 | 1 |
| Win → loss | 8 | 0 |
| Draw → loss | 7 | 0 |
| Winning positions preserved | 144/158 | 157/158 |

The new engine proved an optimal move on **296/298** held-out positions:
128 completed all move values and 168 required only an optimal-move proof. Two
exhausted ten million nodes; one still chose correctly, while one deliberately
preserved a proved draw without finding the win. At 9–18 stones, 231/233 decisions
were proved optimal. This is distinct from a 296-position full-value solve rate.
Previous Python exact search completed on 91/298.

The two remaining regression/development draw-to-loss cases, and the held-out
win-to-draw case, are retained in the position/result artifacts. The held-out
failure is zero-based history `[4,4,4,4,4,1,1,1,1]`: column 3 wins, but it remains
unresolved; column 2 is proved drawing and selected. I did not tune against it.
The frozen failure is `e0-012`; its drawing column 1 is unresolved at the cap.

### What the development ablations say

| Method, per-position cap | Optimal / 300 | Optimal proofs | Total native nodes |
|---|---:|---:|---:|
| Native fair thresholds, 0.2M, no CL | 281 | 213 | 24.0M |
| Native fair thresholds, 2M, no CL | 296 | 282 | 94.3M |
| Native serial all-values, 2M, no CL | 298 | 282 | 136.1M |
| Native fair thresholds, 2M, CL | 298 | 289 | 81.0M |
| Native fair thresholds, 10M, no CL | 299 | 297 | 142.1M |
| Native serial all-values, 10M, no CL | 300 | 297 | 248.2M |
| Final hybrid, 10M, CL | 299 | 297 | 119.0M |

Most of the gain comes from making terminal proof search fast enough and allowing
it before 18 stones. The CL cutoff reduces nodes by about 16% at the final cap;
it is useful, not the dominant breakthrough. Full-value serial search scored one
additional development position correctly but used over twice the final node
count. Fair scheduling is an efficiency choice, not a demonstrated universal
accuracy advantage. Two million nodes left more unresolved cases; ten million
fit the measured interactive allowance. Raising Negamax depth, expanding the
book, expensive general cover evaluation at every native node, and speculative
composite/White proofs were not added.

Raw decisions, intervals, ablations and regenerated counts are in
[victor-astra/](victor-astra/), particularly [summary.json](victor-astra/summary.json).

## Playing strength and resources

Paired games use 16 seeded two-move openings, both colours, against Negamax-6 and
MCTS-400 (64 games/configuration), plus eight empty-board seeds per colour against
epsilon-Negamax-6 and MCTS-400 (32/configuration). Four worker processes; 96 games
per configuration. Node-only budgets make move selection reproducible.

| Measure | Previous | New |
|---|---:|---:|
| W / D / L | 83 / 6 / 7 | 88 / 5 / 3 |
| Score | 0.8958 | 0.9427 |
| Score vs Negamax-6 | 0.7656 | 0.8750 |
| Empty-board score | 0.9219 | 0.9688 |
| Oracle-judged errors from nine stones onward | 40/1,248 | 1/1,141 |
| Win → loss among those decisions | 16 | 0 |

Seven pairs improved and two worsened. Mean paired improvement is +0.0469;
a normal-approximation 95% interval is [+0.0008, +0.0929], but the more cautious
exact paired sign-flip test gives **p=0.0781**. Treat this as promising, not a
conclusive strength claim. This protocol differs from the previous report's
112-game score of 0.821; those scores must not be directly subtracted.

The unchanged oracle resolved 2,071/2,080 distinct game positions under its cap;
nine decisions per configuration remain unjudged. Adjudication starts at nine
stones. Therefore the table does **not** claim zero errors over complete games
or all winning book exits. All resolved move errors occurred at 9–18 stones.
Native searches can still exhaust their budgets on earlier off-book positions.

Serial public-profile latency on all 300 held-out states, including the two
oracle-unknown cases, measured on this local arm64 Mac with CPython 3.11:

| Profile | Median | p95 | Maximum | Whole-move deadline hits |
|---|---:|---:|---:|---:|
| Previous | 62.6 ms | 738.5 ms | 1,075.2 ms | 7 |
| New | 3.4 ms | 41.8 ms | 303.3 ms | 0 |

Public-profile accuracy on the 298 labelled states also remained 297/298. Search
uses a fixed 16 MiB table per call, a maximum 42-ply stack, and checks elapsed time
every 1,024 recursive entries. It releases the GIL. These are local measurements,
not Render measurements; slower hosts can complete fewer proofs. The remaining
Python cover work still has cooperative deadline granularity. Node-only games
had a 2.79-second worst move under concurrent workers, which is why the public
wall limit remains necessary.

## Integration, verification and reproduction

`SolverBudget(native=NativeBudget(...))` opts into the accelerator. The research
CLI exposes `--native` and `--native-nodes`. The existing public research profile
requests the optional accelerator; its external feature flags are still **off**.
`SolverBudget()` itself preserves the old pure-Python mode. If the matching library
is absent, Python remains available; there is no on-request compilation.

The API Dockerfile compiles in its builder stage, excludes host libraries from
the build context, and copies the target-platform library into the runtime image.
No compiler or test oracle is required at runtime. Linux compilation uses portable
C11/POSIX plus the GCC/Clang popcount builtin. Local Clang build and strict warning
checks passed. **The Docker daemon was unavailable**, so the Linux container build
and actual hosting latency remain unverified; no Docker daemon or service was
started and no deployment was attempted.

Validation: **1,465 passed, 15 skipped** in the full backend suite; 19 focused
native tests pass. Existing Baseclaim/Highinverse, White, witness, theorem,
strategy and adapter regressions are included. `git diff --check` passes. No
frontend behaviour changed, so frontend tests were not rerun.

```sh
.venv/bin/python -c 'from games.connect4.victor.native import build; print(build())'
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment build
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment eval --split dev --configs baseline native:200000 native:2000000 serial:2000000
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment eval --split dev --configs cl:2000000 cl:10000000
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment eval --split dev --configs hybrid_nocl:2000000 hybrid_nocl:10000000 native:10000000 serial:10000000
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment eval --split dev --configs hybrid:10000000 cl:10000000 native:10000000
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment eval --split heldout --configs baseline hybrid:10000000
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment eval --split frozen --configs baseline hybrid:10000000
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment games
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment latency
PYTHONPATH=.:tests .venv/bin/python -m victor_validation.astra_experiment summary
EXPLANATIONS_ENABLED=false OPENAI_API_KEY='' .venv/bin/python -m pytest tests -q
git diff --check
```

The generator's corrected book-exclusion count differs from the original frozen
split metadata, but the selected states and split remain the same. All oracle
jobs are node-bounded, chunks have a 180-second process timeout, and only four
workers are used. No unrestricted opening solve or multi-hour campaign was run.
[provenance.json](victor-astra/provenance.json) records budgets, compiler and hashes.

Changed implementation files: `games/connect4/victor/native.py`,
`native_search.c`, `solver.py`, `cli.py`,
`games/connect4/agents/victor_research_agent.py`, `docker/api.Dockerfile`, and
`.dockerignore`. Tests and experiments: `tests/test_victor_native_search.py`,
`tests/victor_validation/astra_experiment.py`, and `docs/victor-astra/`.

**Assessment and next step:** this is a much stronger bounded player, with most
sampled middle openings now carrying optimal-move proofs. It is not a perfect
player: rare budget failures, off-book earlier openings, incomplete composite
execution proofs and finite validation remain. Before public exposure, validate
the Linux build and the unchanged public budgets on the actual host. The next
research target should be reusable proof/TT information across game turns and
sound bound propagation on the small unresolved frontier, evaluated on a newly
sealed holdout. Simply enlarging the book or promoting nine-rule covers to proof
would not resolve those limitations.
