# Phase 4D.4J-B independent prelaunch review

Reviewer: independent prelaunch subagent (same host; did not author the runner).
Review time: 2026-10-14 ~04:05-04:12 UTC. Worktree `phase4d4j-b-worktree`, HEAD `1420cc8`, package untracked.
**Disposition: PASS** (no blockers; 2 minor, several notes).

## Scope and constraints honoured
I read DESIGN.md, common/openings/calibrate/evaluate/worker/manage/prepare/analyze/selftest_analysis
(and the head of validate.py), all results JSONs except `protected-before.json` (key names and counts only),
the frozen 4D.4J-A guard/evaluate/independent_verify, and arena.py. No sealed file was opened, no
neural checkpoint was loaded, no confirmation match was run, no recursive repo grep. Runs: `openings`
(torch-free) and `selftest_analysis` (synthetic, model-free; output in scratchpad).

## Checks
1. **Guard unchanged.** `git show ffb8009:research/alphazero_tactical_guard/guard.py | shasum -a 256` =
   working file = `GUARD_SHA256` in common.py = design.json = validation.json =
   `d2ee54ba…2d04`. The new package defines no guard (no `def guard` in it). It imports `guard` and
   4D.4J-A `evaluate.Agent/decision/positions`. `enforce()` checks `sha(GUARD)`, the full source map,
   runtime and checkpoint bytes against `freeze.json`. worker.py calls it at start and end, and manage.py calls it before launch and after analysis. PASS.
2. **Checkpoints.** design.json checkpoints (paths and SHA256 `ba2bde2e…f58b`, `2646d7cd…401d`, gen 6)
   are identical to `research/alphazero_tactical_guard/results/freeze.json`. PASS.
3. **Opening independence.** openings.py imports only `seeded_rng`, `engine_position` and the
   independent `ReferenceBoard`. It reads only `openings-development.json` and asserts `split=='development'`.
   Rejection uses only rules, duplicates/mirrors, threats and self-symmetry. Nothing from a model, the guard, an opponent or an outcome goes in.
   The re-run printed `416bfb10…59ff5`, attempts 26/27/37/51 and ancestry 22/54, and left the file byte-identical
   (`be340435…d3dc` before and after). Only the mtime changed. My own script used a separate rules implementation (bottom-first grid) and confirmed:
   all 96 confirmation rows and 16 calibration rows are legal, nonterminal, X to move, of stratum length, free of any immediate
   win for either side, and not self-symmetric. They are also unique under (actor, mirror-canonical) and disjoint from the 100 development rows
   (41 distinct canonical positions) and from each other. Strata are 24/24/24/24 and 4/4/4/4. The payload hash reproduces.
   No sequence exactly matches a development row. Ancestry disclosure is accurate: I count 22 families with a
   development-equal shorter prefix and 54 shared new prefixes. PASS.
4. **Opponent selection.** The rule file (mtime 00:00:36 local) predates calibration.json (00:01:46). The run took 68 s,
   so it started after the rule existed. Evidence is mtimes only, because the files are uncommitted. calibrate.py applies the rule faithfully:
   (i) 16 families × 2 colours = 32 h2h games ≥ 0.55; (ii) `openings[::2]` = 8 families × 2 colours × 2
   seeds = 32 control games in [0.10, 0.90]; (iii) a + c + regression ≤ 1800. The choice cascade is (i∧ii∧iii), then
   (i∧iii), then depth 4. Both 5 and 4 were eligible, so the declared preference order alone picked **depth 5**.
   Calibration uses `Agent(inference, False)` and only `openings['calibration']`. PASS.
5. **Colour balance and pairing.** `evaluate.matches` plays game indices 0 and 1 (agent X, then O) for every family.
   Both arms use the same namespace `phase4d4j-b-confirmation:negamax{d}`, seed 4420003, row id and game index.
   `arena.agent_rng` derives per-role streams from (namespace, id, game, role, seed), so the streams are identical until the arms diverge.
   The opening filter guarantees that the agent always makes at least one decision, so the first-decision pair check is well defined. PASS.
6. **Statistics and verdict.** analyze.py uses SEED 4420004, B = 10,000 and PCG64. One family index matrix is shared across
   all seed/opponent/colour/arm conditions, with within-stratum generators for strata. Intervals are percentile 2.5/97.5, and
   the B−A and seed contrasts use the shared draws. The thresholds match DESIGN.md and design.json exactly:
   - FAIL if correctness (reference mismatch, tactical avoidable/missed win, or first-pair difference), exact-WDL
     regression, reproduction difference, latency, or pooled ≤ −0.02 with upper < 0 or any opponent ≤ −0.02 with upper < 0.
   - CONFIRMED needs pooled > 0, lower > 0, each seed > −0.02 and no opponent regression.
   - Otherwise INCONCLUSIVE. Practical significance (pooled ≥ +0.02) and SMALL are reported separately.
   No ties at ±0.02 are possible, because differences are multiples of 1/1536. The reference check rebuilds each position from
   `moves[:prefix + k]`, where k indexes all decisions, which is correct. All 4D.4J-A saved suites have 0 guarded tactical
   failures and 0 WDL regressions, so exact reproduction is a fair criterion. In the synthetic dry run: 0 reference mismatches, 2512/2512
   suite rows identical, latency pass, and `correctness` FAIL only from the expected first-pair difference. PASS.
7. **Feasibility.** Projection 387 + 469 + 108 = 964 CPU-s ≤ 1800. Four parallel tasks, the heaviest about 250 CPU-s
   (Negamax3 task with suites). `compute_deadline_epoch` = 1791950274 + 4500 − 1200 = **1791953574** (04:52:54 UTC).
   The hard maximum is 1791954774 (05:12:54 UTC), and the review clock read 1791950946 (04:09 UTC). PASS. Wall-clock margin is
   ample for compute, but the whole run is tight for review and publication.
8. **Isolation and preservation.** `protected()` hashes trees and files and never parses them. Sealed fixtures enter only as
   hashes (4D.4J-A comment). The scopes are 26 non-empty inventories, including the 4D.4J-A worktree and 150 official files.
   Code writes only to `experiment-output/phase4d4j-b` (gitignored) and to this package's `results/`.
   manage.py requires a clean pushed branch and a review PASS, makes exclusive `open('x')`, and does no retries. The preservation audit is asserted before analysis. PASS.
9. **Non-regression A/B/C.** Checked against guard.py. A winning drop is terminal, so it can never be unsafe. A final-cell
   draw has no replies, so it is safe. AVOID_IMMEDIATE_LOSS fires only when there is no win and the original allows an immediate opponent four, so the original is
   an optimal-play loss and any replacement is ≥ it. IMMEDIATE_WIN always returns a win. B (imperfect opponents) and C
   (safe ≠ theoretically drawn or won) are correctly stated limitations. PASS.

## Findings
- **MINOR M1 (doc):** DESIGN.md says criterion (ii) is evaluated "over 64 calibration games". The rule file, code and
  calibration.json all use **32** games (8 families × 2 colours × 2 seeds). DESIGN.md should be corrected to 32 before freeze.
  This does not affect the selection.
- **MINOR M2 (verdict wording):** "integrity failure" is a FAIL condition in DESIGN and design.json, but analyze.py has no
  integrity branch. Integrity is enforced instead by `enforce()` and the preservation assert in worker/manage, which raise and
  abort before any verdict. The report must treat an aborted or integrity-failed run as FAIL or not-evaluable, never as INCONCLUSIVE.
- NOTE N1: `qualified_heterogeneity` uses `np.sign`, so an exact 0 estimate next to a positive one counts as a "sign
  disagreement". This is conservative (more qualification), which is acceptable.
- NOTE N2: There is no explicit check that control-arm decisions carry `guard is None` and `move == original_action`. This is
  structurally guaranteed by `Agent(inference, False)` and the frozen source map.
- NOTE N3: openings.py's overwrite guard compares only `confirmation_sha256`, not calibration rows or the whole file. The
  file was verified byte-identical after my re-run, and freeze.json will hash it.
- NOTE N4: Rule-before-calibration ordering is evidenced only by mtimes. With both candidates eligible, the outcome is
  determined by the declared order, so there is no selection-bias channel.
- NOTE N5: The h2h strength evidence is 32 games, and Negamax4 scored higher than Negamax5 versus Negamax3 (0.703 vs
  0.656). DESIGN already says deeper is not assumed stronger.
- NOTE N6: The exact-reproduction FAIL compares floats (root value/prior/logits) bit-for-bit. That is fail-safe, but a
  numerical-runtime drift would yield FAIL, not a strength conclusion.
- NOTE N7: The confirmation openings exclude roots with immediate threats, and 22/96 share development ancestry. Both are
  disclosed. Inference is conditional on this generator, as DESIGN states.

## Disposition
PASS. No guard modification, contamination, unfair pairing, rule mismatch or invalidating bug was found.
M1 is a documentation fix. M2 is a reporting obligation. Neither blocks launch.
