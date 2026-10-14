# Independent prelaunch review — Phase 4D.4J-A

Disposition: **PASS for prospective launch.** No unresolved launch blocker remains.

Reviewer: `/root/independent_review`. I did not write the intervention,
evaluation runner or primary analyzer. I independently authored the matrix
reference and saved-record verifier. This is review on the same machine,
not external replication. The review made no neural model queries, optimizer
updates, self-play collection or extra playing-strength experiments.

## Tactical rules and unchanged PUCT

I inspected `guard.py`, `evaluate.py`, the original bitboard/search/reference
Negamax sources, the standalone Tactical MCTS guard, and `validate.py`. The
intervention is ordinary immediate-win filtering and one-ply loss avoidance.
The existing Tactical MCTS applies these checks before UCT; this experiment
postprocesses an unchanged learned PUCT result. Reusing the immutable bitboard
rules avoids production rule duplication and engine deepcopy overhead.

The actor comparison uses the player who placed the candidate, while the
successor's flipped actor owns replies. Terminal successors have no replies;
this includes draws. Root terminals are rejected. Every physical legal column
is checked, including partially filled columns. Restricted candidates use their
original visits with lowest physical column for equal visits. An already-safe
original selection is returned exactly. All-unsafe roots retain that original
selection and do not acquire a theoretical-loss claim. Visits, priors, values,
PUCT formulas, expansion, backups, simulations and terminal evaluation are
unchanged. The guard consumes no RNG and calls no neural inference.

The independently authored `ReferenceBoard` uses top-first immutable matrices,
gravity drops and direct directional scans, importing no production rule code.
It independently confirmed all 200 tactical fixture label sets and the legal
sets, optimal sets, value maxima and immediate-win consistency of all 428
complete solved/deep oracle vectors. This checks label consistency; it does not
independently re-solve every position's full game-theoretic value.

The reviewed final validation records 4,872 independently classified decisions,
79 terminal rejections, both actors, all seven immediate-win columns,
horizontal/vertical/both diagonal wins, mirrored/color-swapped boards,
multiple wins/safe actions, all-unsafe states, full columns, zero-visit
overrides, immutability and original visit preservation. Six preservation
roots compare canonical/disabled/enabled searches, full RNG states,
network outputs, model weights and learner state. Directed final-cell draw
selection, full-draw rejection and both-actor exact winning-successor signs pass;
there are 14 additional directed terminal rejections and one full-draw rejection.

## Fairness, provenance and statistics

The primary endpoints use both original H-C generation-six CNNs, 256
simulations, historical evaluation namespaces/seeds, no root noise,
and the same Negamax-3 opponent. Fixture variants repeat identical raw search
and assert original action, visits, logits, value and prior equality. Match
pairs share opening/color/role RNG identities. Diverging games may subsequently
consume different numbers of draws; the prospective design discloses this.

Every 200/300/128 fixture and all 40 base development opening families with
both colors are fixed. Expected work is 2,512 fixture searches, 1,256 paired
fixture records and 320 matches, across exactly two seed tasks. Secondary
matches and ResNet runs are omitted before results. No sealed package is
parsed. I read only explicitly named development data, new-package sources,
historical reports and clean historical reviews; hash-only inventories do
not print or interpret protected fixture contents. The prior sealed-access
incident remains disclosed in the design and history.

The primary comparison is guarded minus control match score. The 10,000
paired bootstrap replicates retain both colors and both fixed networks in
whole shared opening-family clusters. Fixture clusters retain ancestry and
reflection dependence and row weighting. Equal seed weights, practical +5pp
PASS threshold, material -5pp FAIL threshold, interval conditions and both-seed
condition are explicit. Intervals condition on these two networks, rather
than estimating retraining variation. Tactical performance is a targeted
correctness check. Oracle regressions block positive publication. Lost-row
denominators and descriptive activation correlations are treated appropriately.

My saved-record verifier independently replays every game and classifies every
agent decision, checks all visits/original/final actions and reply witnesses,
recomputes fixture metrics and match W/D/L/colors/families, and reproduces the
frozen bootstrap stream separately from an additional reviewer stream. It
compares all headline counters, intervals, latency summaries and verdicts
without importing primary analysis arithmetic. It also rehashes every frozen
source and checkpoint and checks task identities and before/after state.

## Performance and lifecycle

The cost-only pilot retains timings, not outcomes: 12 roots and four matches,
12.54 seconds, projected primary 331.33 seconds, conservative contingency
962.66 seconds. This supports the primary budget. Serial measured workers
use deterministic one-thread CPU settings. Prospective latency limits are
p95 guard 1ms, worst 5ms and mean relative overhead 5%; raw search, guard and
total decision times remain separate. No extra inference is used for tactics.

The manager requires a clean dedicated branch, remote branch equal to HEAD,
prelaunch PASS and source/runtime/checkpoint matches. Attempt and task paths
are exclusive, without retries. A supervisor checks the absolute deadline
and cancellation marker; cooperative checks run between roots, moves and
games, with owned process-group TERM/KILL cleanup. Computation stops 180
seconds before the original 45-minute deadline, reserving review/publication.
Incomplete work cannot acquire a complete positive interpretation.

One prelaunch launcher defect was found and corrected: invoking `independent_verify` without
`--results --output ...` ran only the verifier's synthetic self-check. It must
was corrected before freeze so full saved-record verification is required.
This was found before measured outcomes and does not authorize a measured retry.

I confirmed the launcher correction, final successful validation,
prepared machine-readable design, both original checkpoint SHA-256s,
development dataset/opening hashes and all 22 nonempty actual historical
inventories, including the 150 official campaign files. The task origin is
02:53:39 UTC and hard deadline 03:38:39 UTC on October 14 (October 13 local).
The inventories contain the actual H/H-B/H-C/G/I-A/I-B artifact paths and
canonical/current source scopes. A final pre-freeze preservation strengthening
adds the actual G/H-B/H-C docs trees (43/44/45 files) and expands the canonical
base-commit scope to all 1,473 inherited game/model/test/API/UI/docs/research
files. This is hash-only protection, with no fixture content parsing or change
to intervention, evaluation, membership or statistics. I inspected and approved
the revised additive protection code and all nonempty inventories.
Frozen identities will be written after this
review and the manager will enforce the pushed prospective commit before launch.
No public agent,
shared source, checkpoint, deployment configuration, merge or promotion is
authorized or required by this review.
