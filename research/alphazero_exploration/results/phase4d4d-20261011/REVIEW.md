# Independent critical review — Phase 4D.4D

A separate reviewer agent, which wrote none of the code, reviewed the
collector change, the tools, the raw evidence and the draft report. It was
not allowed to edit the repository or the campaign; it worked from its own
scratch scripts. It loaded the `99cec1c` collector beside the current one,
recomputed the headline numbers from the raw game files without importing the
analysis modules, ran the project's exact solver on every flagged tactical
event, re-hashed the official campaign twice and ran the focused tests (137
passed). This is an independent engineering review, not third-party
certification or replication on separate hardware.

**Verdict.** No defect in the collector, execution, preservation or
arithmetic: every recomputed number matched, and the predeclared rule was
applied exactly as written. Two material problems, both about interpretation
and robustness.

## Findings and disposition

1. **MATERIAL — overclaim. Most counted tactical errors do not change the
   game-theoretic result.** With the exact solver, 62 / 104 / 89 / 105 of the
   71 / 118 / 99 / 122 unsafe moves were played by a side already lost under
   perfect play, and 27 / 48 / 46 / 45 of the 29 / 49 / 49 / 50 missed wins
   still won. Value-changing errors per game are 0.043 / 0.059 / 0.051 / 0.086
   and no paired difference excludes zero. The draft's wording ("tactically
   worse", "blunders that end one game in nine", "the baseline is already
   weak") overstated what the one-reply proxy shows. The reviewer's own
   caveat: neutral under perfect play is not neutral against a 256-simulation
   opponent.
   *Disposition:* confirmed independently. `exact.py` reproduces every figure
   (370 positions, all solved at 6,000,000 nodes). The breakdown is in
   `analysis/results.json` (`post_hoc.exact_values`), `TABLES.md` and the
   report, labeled post hoc. The report's lead, tactical section,
   recommendation and next experiment were rewritten; the next experiment now
   uses an exact-value endpoint instead of targeting one-reply events. The
   predeclared decision output is unchanged.
2. **MATERIAL — robustness. `smoothed`'s MATERIAL REGRESSION rests on one
   game, and both of its labels are knife-edge.** It trips only T3 (+0.188
   [+0.027, +0.348] against 0.15); removing identity (314159, cohort 0, game
   23), which has 12 value-neutral errors against 0, gives +0.141 and
   INCONCLUSIVE. Its seed-42 diversity estimate is 25.5% against a 25%
   threshold, with 14 of 128 delete-one replicates below it.
   *Disposition:* confirmed. `analyze.robustness` computes single-identity
   sensitivity for every label (`post_hoc.robustness`); the report's headline
   table and a dedicated subsection mark `smoothed` as not robust, and the
   limitations state the multiplicity and point-estimate issues.
3. **MINOR — checkpoint heterogeneity stated only for `smoothed`.** At seed 42
   no tactical endpoint excludes zero for any schedule; the pooled regression
   of the extended schedules is carried by seed 314159.
   *Disposition:* per-checkpoint paired differences are now shown beside the
   pooled ones in the report and `TABLES.md`.
4. **MINOR — "before any collection" was not strictly true.** A smoke test
   with a randomly initialized network (10 games and a deadline-stop test, in
   the session scratch directory) predates the design commit.
   *Disposition:* disclosed in the report's design section, together with the
   pilot cost printout showing ply totals per cell and the analysis commit
   landing while 9 of 16 main cells were on disk.
5. **MINOR — the ablation plan conflicted with existing code.**
   `generation.rebuild_game` rejects stored temperatures that are not the
   config's two-level schedule, and `V2Config` cannot express τ = 0.3.
   *Disposition:* the plan now uses `extended`, which `V2Config` can express
   as `exploratory_plies = 16`. The second pass showed that this is still not
   enough for a warm start (see below); the plan now names a per-generation
   schedule in boundary storage, and the batched-collector wiring, as
   prerequisite runner changes.
6. **MINOR — test gaps.** No direct test of the diversity thresholds; the
   escalation branch unexercised and vacuously true on empty data; the
   compressed loader and `package.py` untested; three acceptance rows
   hard-coded; record hashes not checked by the loader; `from_dict` raising
   `TypeError`.
   *Disposition:* `diversity_checks` is now a pure function with boundary
   tests; `recommend` takes a gain function, never escalates on empty
   evidence and has four escalation tests; packaging and the compressed
   loader are tested end to end; the analysis rebuilds all 1,152 main and
   supplement games and matches the collector's record hashes; the three rows
   now state that the loader raises on violation; `from_dict` raises
   `ValueError`. Still untested by unit tests: the top-level `analyze` driver,
   `archived_reference`, `cross_check` and `performance`, which need the
   campaign and are exercised by the real analysis runs.
7. **NOTE — acceptance row 1 wording.** RNG hashes against retained 4C
   evidence exist only for seed 314159. *Disposition:* wording corrected.
8. **NOTE — results folder, placeholders, `package.py`.** *Disposition:*
   evidence packaged; placeholders filled; `package.py` docstring corrected,
   gzip round trip verified, registry paths made relative.
9. **NOTE — schedule names and unbounded τ.** *Disposition:* names restricted
   to `[a-z0-9_]+`. τ stays unbounded above; the existing visit mathematics
   handles any finite nonnegative value. Because this touched `batching.py`
   after collection, the regression against Phase 4D.4C evidence was re-run
   with the delivered source: all ten cells equal the first regression in
   records and RNG hashes (`regression-final/`).
10. **NOTE — wording.** *Disposition:* "intervals in all four cohorts"
    corrected; the window share now reads 5.8–7.4%.
11. **NOTE — forking paths.** The analysis code is absent from the main run's
    source snapshot and "no collected game had been inspected" cannot be
    verified. *Disposition:* stated in the report's design section and
    limitations.

No threshold, margin or decision rule was changed after results.

## What the reviewer verified and found correct

- Collector diff limited to `TemperatureSchedule` and one conditional;
  `config`, `search`, `selfplay`, `data`, `bitboard`, `network`, `oracle`
  byte-identical to `99cec1c`.
- Default-path games, RNG states and batch histograms identical to the
  `99cec1c` module across four configurations and pools 1, 3 and 8.
- Exactly one action draw and one noise draw per move across five schedules,
  including τ = 1e-300 and τ = 5.
- All raw data contracts over 1,024 games and 32,295 positions.
- Every per-game and per-cell record hash of the 1,280 games in the first
  four runs, rebuilt by hand. RNG hashes were checked for consistency between
  the game files and `cells.json`, and against retained Phase 4D.4C evidence
  for the seed-314159 regression; they were not rebuilt.
- Negative control: 0 mismatches over 1,024 matched prefix comparisons.
- Game counts, positions, mean length, D1, D2, tactical counts, pooled T1–T4
  with all twelve intervals, the supplement and the replay-window figures: 0
  discrepancies at 1e-9.
- `DESIGN.md` unchanged since `9f8db74`, pushed before the first invocation,
  and equal clause by clause to the coded thresholds and rules.
- Tactical definitions match the design; no double counting; no off-by-one in
  game-ending errors.
- Jackknife correctly implemented and paired.
- 150 official files byte-identical, declaration hash unchanged, latest file
  mtime 2026-10-09.
- Measured time under budget, all cells complete.

## Second pass on the fixes and the rewritten report

The same reviewer checked the changes read-only. **No material finding
remains unresolved.**

- The final `decision.json` differs from the pre-review one only by two added
  boolean keys per checkpoint; every `results.json` section other than
  `performance` and the new `post_hoc` is identical. Re-running the delivered
  analysis on the packaged compressed evidence reproduces `decision.json`,
  `acceptance.json` and `TABLES.md` exactly.
- `post_hoc.exact_values` and `post_hoc.robustness` equal the reviewer's own
  figures exactly. Under every one of the 256 single-identity deletions the
  two extended schedules stay MATERIAL REGRESSION; `smoothed` flips in exactly
  one.
- The `batching.py` change since `83927d5` is validation only, and the second
  regression equals the first in all ten cells.
- 147 focused tests passed; the test-gap finding is adequately addressed.
  `robustness` is only shape-checked by unit tests.
- The campaign was re-hashed a third time: 150 files unchanged.

Remaining items from the second pass and their disposition:

1. **MINOR — the ablation plan still claimed a warm start the runner cannot
   do.** `load_resume_boundary` rebuilds the whole replay window against one
   stored `exploratory_plies`: all archived generation-20 games are rejected
   under 16 and 248 of 256 `extended` games under 8. *Addressed:* the report
   states the per-generation schedule storage as a prerequisite change.
2. **NOTE — "the one-reply events explain little of it" was unsupported.**
   Under `extended`, 23 of the 52 identities that go from draw to decisive end
   on a one-reply loss. *Addressed:* the sentence now says only that the
   value-changing events are too few to account for the shift and that the
   rest is unmeasured.
3. **NOTE — this record misstated the window share and the RNG-hash check.**
   *Addressed* above. Two details of finding 2 omitted earlier: five
   identities carry +39 of `smoothed`'s +48 errors, and excluding
   result-neutral missed wins gives +0.125 [+0.030, +0.220], below the margin.
4. **NOTE — stale manifest.** *Addressed:* regenerated last.
5. **NOTE — "the delivered collector is the one that was measured" was
   slightly strong.** *Addressed:* reworded to identical behaviour on the 64
   regression games with a validation-only diff.
6. **NOTE — wording.** The sampled-loss count is partly definitional because
   the original schedule cannot sample after ply 8; a code comment said
   "before any collection". *Addressed:* both reworded.

## What the reviewer could not verify

- The full 844-test run (it ran the three focused files).
- Whether any collected game was inspected before the analysis commit, or
  whether other attempts were run and deleted.
- Bitwise reproducibility by re-collection, which it was not permitted to do.
- Exact per-seed replay sampling values (it used a different RNG; means agree
  within noise).
- Policy and prior statistics, which it spot-checked only.
- Exact values of positions that were not one-reply-flagged.
