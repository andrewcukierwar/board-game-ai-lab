# Phase 4D.4I-A prospective design

Research branch phase4d4i-a-alphazero-value-diagnostics from
3b563e1753ad93a1315b7d0f88a7cc45e1ee7359. No neural training, new self-play,
checkpoint modification, promotion, merge or deployment. All code resides in
this new research package. Existing games and previously frozen sources stay
byte-identical. Strict H-C boundary loaders enforce SHA-256, expected full
configuration, execution source identity and one-thread deterministic CPU runtime.

## Mechanisms and fixed interventions

A: learned policy and value; B: learned policy and zero nonterminal value;
C: uniform legal policy and learned value; D: uniform legal policy and zero
nonterminal value; E: learned policy and 0.5 times learned value. No additional
scale is evaluated. Same unchanged bitboard PUCT, c=1.41, 256 simulations,
legal masking, actor-relative alternating backup and exact terminal values.
Root noise, tactical guard and proof propagation are absent. Final action is
maximum visits with the existing seeded tie rule. Root neural value is only
a diagnostic, never backed up. Priors are modified at every neural expansion,
not only the root. Controls still execute inference before transformation;
compute savings from skipping neural evaluation are outside this study.

The factorial contrasts for accuracy/score are B-A, D-C, C-A, D-B and
A-B-C+D. E-A tests prospectively fixed attenuation, without tuning. These
identify the effect of changing the frozen model's outputs within this search,
not why the trained model learned them. No explanation is assumed in advance.

## Models, fixtures, identities and work

Four primary generation-6 checkpoints: CNN/ResNet3x64 x seeds42/314159.
Generation-zero A-only evaluations are negative controls. Generation5 is
loaded only for replay and prequential value diagnostics. All 28 recorded H-C
boundary SHA identities are copied into freeze.json; each accessed boundary
is strictly loaded against its recorded identity. Historical generation20
checkpoints are prospectively omitted to reserve the budget for the complete
primary factorial and all five variants in matches; no historical loader is
relaxed or checkpoint rewritten.

Use the byte-identical H-C/G nonsealed fixtures SHA256
80f6851838b18172d6f818ebd306ce0cd36e4796cac1e3a71df29694feb3ca2c:
tactical200 (200), solved300 (300), deep128 (128), unchanged membership.
Ancestry/reflection clusters:71/102/28. Use all40 base opening families of
openings-development.json, both colors, against existing reference negamax3.
80 games x5 variants x4 checkpoints =1600 matches. Six position configurations
(5 trained +1 untrained) x4 arms x628 =15072 roots. No g0 matches.

Search and match namespaces and seeds are exactly H-C's (4404901/4404902).
Intervention, architecture and training seed never enter evaluation RNG
identity. Each row/role starts from its matched stream; diverging search and
match trajectories may consume different draws. This pairs initial identities,
not common random numbers at semantically different later nodes.

## Endpoints and uncertainty

Tactical: immediate-win and unique-safe-response accuracy, missed wins and
avoidable one-ply errors. Exact: optimal-action accuracy and every selected
legal-action WDL decrease, including win-to-draw, win-to-loss, draw-to-loss.
No tactical or unresolved row receives invented deeper labels. Action changes,
fixed mistakes and introduced mistakes are paired on explicit row IDs.

Matches: W/D/L, score(win1/draw0.5/loss0), paired differences from A, colors,
per-training-seed and pooled-within-architecture effects. All five variants
are fixed before results; no result-dependent intervention selection.

10,000 seeded percentile bootstrap resamples, analysis seed4410903. Entire
frozen ancestry/opening clusters are sampled with replacement, preserving all
rows/reflections and both match colors. Recompute row-weighted means. Pooled
contrasts average the two fixed seed effects per shared row before jointly
resampling families. Intervals are conditional on these two networks and
fixtures; they do not measure training-seed uncertainty. Report both seeds.
No binary promotion rule or multiplicity-adjusted confirmation is claimed.
All contrasts are diagnostic; isolated interval exclusions need caution.

The theoretical win/draw/loss denominators and suite composition are retained.
Exact decision regression is zero by definition on an already theoretically
lost position; optimal accuracy there alone can conceal poor action quality.
Deep128 has no lost positions. Do not average suites as if exchangeable.

## Read-only value evidence

All retained generation6 replay positions and all retained generation5 replay
positions are scored in eval mode. Generation5 scores ONLY newly collected
generation6 games for the prequential measure; split by novel/repeated mirror
keys relative to g5 training replay. G6 has no g7 evidence, so its prequential
MSE is unavailable. No new games are collected to fill that gap.

Compare realized outcome MSE, constant-zero MSE, exact WDL MSE, excess error
over zero,10 equal-width predicted-value bins, side-to-move bias, W/D/L errors,
absolute magnitude, confidence >0.8 and saturation >0.95, actor, ply8-wide bins,
repetitions and observed terminal distance8-wide bins. Independently report
conflicting realized labels on identical repeated boards and the empirical
within-position outcome-variance MSE floor; these reflect imperfect-policy
outcome noise/nonstationarity, not a sign implementation error. Cluster replay by
canonical mirrored first-eight-ply prefix. Short-game ancestry and missing
original generator trajectory identities are limitations, not independent
sampling claims. Report per-position descriptive strata and cluster intervals.

Audit every replay label by legal history replay, recorded winner and actor;
verify frozen exact fixture legal vectors independently. Compare exact versus
realized targets ONLY on identical fixture/replay boards, disclose overlap
count. Sparse/no overlap cannot establish target mismatch. Optimal WDL and
imperfect-policy realized outcomes need not agree. Confidence/repetition/ply
correlations do not establish causes. No sign bug is inferred from high MSE.

## Resources, stopping and prelaunch checks

Cost-only pilot uses four model-blind stride-selected rows per suite per arm
and one opening pair per arm, retaining timings only. Pilot19.4s projected
three-worker main work886.5s; x1.8 contingency1595.7s. All variants retained.
Target30–45 minutes is an upper planning range, not a minimum workload.
Hard total measured maximum3600s includes validation, pilot, run, review model
reproduction and analysis; main run ceiling2700s reserves remaining budget.
No automatic retries, replacement jobs or expansion. Exactly48 tasks in three
owned subprocesses. Any failed task/cancel/deadline stops scheduling and cleans
up owned process groups. No incomplete task is analyzed as complete.
A detached manager persists status and partial evidence. No session is kept
open merely to wait; postexecution analysis/review may be continued separately.

The full prospective sources/design/checkpoint hashes/runtime/preservation map
are committed and pushed BEFORE measured tasks. Manager refuses a dirty or
unpushed HEAD and workers refuse altered sources/runtime. Historical byte
inventories, including150 official campaign files and G/H-B/H-C evidence,
are checked at freeze and completion. Source hashes of canonical evaluation
files protect their bytes without reading labels.

## Independent review and retained protocol incident

Independent reviewer checks strict loads, wrappers, terminal/sign handling,
legal priors, RNG identities, fixture labels, WDL definitions, match histories,
paired uncertainty, evidence and preservation, then reproduces headline
contrasts with its own aggregation and a bounded model rerun. Same-host review
is not external replication. Any unresolved finding remains in final report.

During reviewer orientation, an overly broad rg directory search printed
partial solved-sealed.json text before this design. This violated the no-sealed-
access requirement, was disclosed immediately and cannot be undone. No design,
fixture membership, intervention or analysis uses that content. Subsequent
access is restricted to explicit nonsealed paths. The incident remains a
protocol limitation even if byte-preservation passes. See independent-prelaunch-
review.md for the exact command. No sealed evaluation is run.

Final conclusions must distinguish tactics, exact decisions and actual match
strength. Reused development fixtures do not validate generalization. Recommend
at most one controlled value-training follow-up justified by results; do not
implement or launch it here.
