# Phase 4D.4K-B — AlphaZero frontend and release preparation

## 1. Executive summary

Release candidate implemented on `phase4d4k-b-alphazero-frontend`, with independent
review and bounded local ARM64 staging. Final browser/remote CI results are being
recorded below before the branch is declared complete. **Not merged or deployed;
AlphaZero has not been enabled publicly.**

The experimental generation-6 CNN can be selected in Play, Match, Tournament and
Season Labs only when the backend reports it available. Safeguard Off preserves
the original baseline; On applies the same verified one-ply tactical selection
after unchanged PUCT. No model, search budget, guard algorithm or research
checkpoint changed. This is not an optimal solver or a state-of-the-art claim.

Across two CNNs against Negamax3/5, historical confirmation measured +1.302 pp
match score (95% [+0.521,+2.344]), below the prospectively declared +2pp practical
threshold. The UI states this context and does not promise stronger play in every
position. The safeguard is not a separately trained model.

## 2. Base, preflight and protected environment

Base: completed backend branch `phase4d4k-a-alphazero-backend`, commit
`5ac8387207fceb8d6852c79610a14b9246fefe4f`. Remote refs fetched; clean historical
checkout observed; ancestry against current main verified at
`8704142702da273e6571f0ca9df3fd4a793be285`. A new isolated worktree was created at
`experiment-output/phase4d4k-b-worktree`. No historical research branch was merged.

Reviewed K-A architecture/contract and the H-C, I-A, I-B, J-A and J-B reports
read-only from their committed refs. Inspected existing selectors, hooks,
controllers, export/verifier, persistence, deployment and CI before edits.

The live Mac mini container stayed at its original image, container ID, start
time and restart count. It remained on localhost port 8000 and was never used for
testing. No Services checkout, Render configuration, Funnel setting, credential,
production container or research job was changed. No active training was found
 before bounded cached Docker builds. No training or research experiment ran.

## 3. Frontend architecture and discovery

`connect4/alphazero.js` owns the optional-agent contract, provenance validation,
identity comparison and shared per-HTTP-client discovery cache.
`useAlphaZero.js` subscribes via `useSyncExternalStore`, with effect cleanup for
focus listeners. Mount/focus reuse a 60-second cache; explicit refresh and
AlphaZero game starts recheck status. There is no polling and no discovery request
before every move. Sequence identities prevent an older response overwriting a
newer refresh. Unknown availability fails closed.

`GET /v1/connect4/agents/alphazero` is authoritative; no VITE flag controls
AlphaZero. Available status must have real booleans, consistent model/search
fields, supported versioned provenance and a boolean guard option defaulting Off.
404 means unsupported backend; disabled, enabled-but-unavailable, malformed and
network-failed responses hide the new option and leave legacy agents usable.
Availability messages and a normal accessible refresh control do not replace the
game status/error UI. Agent errors invalidate stale discovery. Saved configurations
remain identifiable even when continuation is unavailable.

## 4. Selector and tactical safeguard UX

Play offers **AlphaZero (Neural MCTS)** with an Experimental designation. Match
and field presets identify **AlphaZero (Experimental) · safeguard Off/On**.
`AlphaZeroSettings.jsx` uses existing controls, helper text and a collapsible
research explanation. It exposes only Off/On; no seed, simulation, temperature,
noise or exploration tuning. Payloads contain a real JSON boolean. Selection
starts Off. Strict payload/config validation rejects unsupported settings.

Existing typography, spacing, mobile layouts, focus rings and reduced-motion
behavior are retained. Home research copy now explains that optional AlphaZero
appears only when supported by the connected backend; it does not announce a
public deployment. The refresh control uses the existing accessible action style.

## 5. Play

The existing game hook and start/move endpoints serve both colors, both guard
settings, restart/new configuration, terminal games and ordinary analysis UI.
Settings apply to a new game; active-game settings are shown separately and never
mutated by the toggle. AI search uses the existing thinking state.

AlphaZero starts compare authoritative history with discovered identity and
expected canonical players before allowing play. Integrity failure quarantines
the board and offers restoring the original model or a verified new game.
Uncertain request recovery uses full AlphaZero history and identity validation,
not just an ordinary snapshot that could unlock mismatched evidence. Explicit
retry rechecks availability/identity and current revision. Position/last-AI
explanation UI is exercised with mocks; no paid provider call was made.

## 6. Match Lab

Shared payloads, labels, snapshot/history validation and move transport support
Human in either color, all eligible existing agents, AlphaZero versus itself,
and guarded versus unguarded configurations. Step, autoplay, pause, timeline,
terminal winner, replay and reset retain the existing hook's serialization.
Discovery/history pinning and busy cancellation share its lifetime/abort guards.
AlphaZero ignores `rng_seed`; this is stated in lab/provenance documentation.

## 7. Tournament Lab

Field presets include both guard variants when available. Entrant IDs distinguish
duplicates; exact boolean guard settings survive bracket planning, color changes,
match records, local replay, winner descriptions and persistence. Existing sizes,
seeding, draw/rematch/tiebreak rules and Human workflow are unchanged.

AlphaZero fields use local storage **version 2**, with a single pinned model/search
identity; legacy fields remain version 1. Existing storage keys let the validated
loader read both. Status is checked before each fixture start; history is checked
before accepting results. Model change/unavailability stops advancement without
replacing an entrant. Restore the original model or create a new tournament;
completed results remain reviewable.

## 8. Season Lab

Both guard presets work with unchanged balanced circle schedules, swapped-color
pairs, multi-game pairings, standings, ratings, intervals, pairwise summaries,
replay and exports. AlphaZero Seasons use local persistence version 2 and pin one
identity. Every completed AlphaZero game carries its captured identity; mixed or
missing identity is rejected. Ratings and scoring were not changed. The canonical
112-game Season and all prior benchmark/export bytes were not regenerated.

## 9. Versioned evaluation provenance

Evaluation **schema 3 / methodology 3** is used only when entrants include
AlphaZero. Backend core `/provenance` is unchanged. Available status adds
`provenance.version=1`; an AlphaZero session stores a deep copy at creation and
returns it as `alphazero_provenance` in history even if the runtime later becomes
unavailable. Before AlphaZero search, a different runtime identity is refused
without committing a move. This is the smallest explicit backend addition.

The captured identity includes type, implementation version, model ID,
architecture/parameters, artifact and source-checkpoint SHA-256, PUCT algorithm,
256 simulations, exploration 1.41, no root noise, maximum-visit final selection,
exact-position tie-break contract, ignored game seed, guard contract and source
hash. Entrant/player configs separately carry guard enablement. v3 evidence and
provenance include the pin; each completed AlphaZero game's evidence includes it.

Canonical sorted JSON and SHA-256 coverage remain unchanged in structure. v3 CSVs
add explicitly named provenance/guard columns. The verifier reconstructs schedules,
columns, results and analytics and rejects downgrade/missing/mixed provenance,
even after the digest is recomputed. Hashes prove internal consistency, not server
authentication or identical future floating-point search across every runtime.

## 10. Legacy compatibility and history/replay

Ordinary v1 and Victor v2 schema/methodology/provenance/CSV identities remain
frozen, with no silent upgrade. Independent base/current generation compared
JSON and both CSVs byte-for-byte for v1 and v2; both verified. Canonical fixtures
and the immutable 112-game benchmark still verify with their original digest.

History validation accepts optional AlphaZero model/guard/original-action/visit
evidence without changing move ordering, actor, revision or legal reconstruction.
It checks guard identity, integer256 visits, model agreement and valid guard
reasons when that evidence is present. Compact competition storage keeps columns
and a small versioned provenance block, not duplicate boards and root vectors.
Full server history retains the move evidence; local replay remains column based.

## 11. Busy, unavailable and reconciliation

Only explicit503 `agent_busy`/`game_busy` responses enter AlphaZero bounded busy
retry: at most two retries,250/750 ms minimum delays, honoring Retry-After up to
five seconds. Invalid/long Retry-After requires explicit continuation. No uncertain
network failure, stale revision, rate limit or failed-agent response is blindly
resubmitted. Established controllers reconcile authoritative revision/history.

Pause epochs cancel pending retries; unmount abort/lifetime guards prevent old
hook retries. An in-flight accepted competition POST may settle/persist after
pause or route departure, under the existing sole mutation owner; it never queues
another ply. Timers and synchronous locks prevent overlapping autoplay, duplicate
advancement and replacing a busy session. Unavailable/invalid agents stop autoplay,
preserve completed results and never fall back. Expired/restarted sessions require
explicit new/restarted games. Saved evidence can be reviewed offline.

## 12. Docker deployment preparation

`deploy-mac-mini-api.sh [lite|alphazero]` defaults to lite. Both build and runtime
flags are explicit and override stale private env values. Full commit+mode image
tags, native ARM64 validation, existing private env handling and one-worker/four-
thread Gunicorn are retained. No shared search limit or Funnel topology changed.

Candidate creation records an immutable owned container ID; cleanup never removes
a foreign named candidate and handles start/port-binding failure. Localhost8001
candidate checks include health, exact core/source provenance, mode-specific
status, real legacy-agent moves, optional Victor and both AlphaZero guard moves.
An unavailable model fails before cutover. Checks repeat after cutover; failure
restores the original container/environment/image and preserves rollback behavior.
**The actual production deployment script was never run.** Only mocked Docker
commands tested cutover/rollback logic.

## 13. Dependency supply chain and CI

The ARM64/Python3.11 runtime now has a verified nine-wheel CPU hash lock, including
all applicable torch transitive dependencies at already-tested versions. Native
resolver metadata, all nine official index digests, and force-reinstalled hashed
Docker downloads were independently checked. Lite still contains no torch.
See [dependency procedure](alphazero-runtime-dependencies.md) for exact update
commands and limits: optional AMD64 remains version-only; base OS/apt/ordinary API
packages are outside this narrow lock.

The new `alphazero-inference.yml` uses native `ubuntu-24.04-arm` (supported in the
[GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)),
relevant path filters including the lock, bounded timeouts, read-only repository
permissions and cached builds. It covers frontend unit/build, ARM64 image/model
integrity, CPU/thread settings, inference/guard/API/deployment mocks and Chromium
against its isolated service. It never deploys, contacts Mac mini, or publishes an
image. The existing main-only lite publishing workflow and triggers are unchanged.

## 14. Isolated staging architecture

Cached K-A images were inspected but final K-B images were built separately.
Owned identities/ports only:

| Service | Port | Purpose |
| --- | --- | --- |
| `bgai-k-b-staging` | localhost port 18114 | final hash-locked AlphaZero+Victor API |
| Vite | localhost port 18115 | React proxy via `BGAI_DEV_API_TARGET=...18114` |
| `bgai-k-b-lite` | localhost port 18116 | default torch-free/disabled API |
| earlier `bgai-k-b-locked` | localhost port 18117 | locked-build smoke, later removed |
| `bgai-k-b-unavailable` | localhost port 18118 | lite image enabled without torch, controlled unavailable |

Explanations disabled; explanation browser tests mock responses. No live port 8000,
production Services checkout, research fixture or public backend request was used.
Staging source-commit env used the baseSHA as a declared local marker while the
working tree was under construction; image identities and final Git changes
identify actual candidate content. Production/CI use their exact committed SHA.
Only one real game per small Tournament/Season smoke was needed; large fields in
legacy browser tests use existing synthetic HTTP fixtures, not neural evaluation.

## 15. Regression results

| Check | Final result |
| --- | --- |
| Full torch-free backend |3077 passed,0 failed,17 skipped |
| Native targeted API/history/deployment after session pin |149 passed,0 failed |
| Final linux/arm64 inference/API/guard/provenance/deployment |122 passed,0 failed |
| v2 inference/contracts |90 passed,0 failed,14 deselected (training/generation/resume) |
| Frontend unit/contract suites (final) |463 passed,0 failed,0 skipped |
| Production Vite build |PASS |
| Deployment mock suite |13 passed,0 failed |
| AlphaZero Chromium matrix before added reverse-Human/explanation cases |22 passed,0 failed |
| Corrected14 legacy browser failures targeted rerun |14 passed,0 failed |
| Final full Chromium |183 passed,0 failed,1 skipped (disabled-Victor UI case run separately) |
| Firefox |57 passed,0 failed,0 skipped |
| WebKit |pending final recorded results |
| Disabled-Victor UI, separate localhost UI |3 passed,0 failed |
| Final isolated lite/enabled/unavailable builds and release smoke |PASS; no torch in lite; missing runtime refuses AlphaZero |
| Remote native ARM64 workflow |pending branch publication/check |

The17 torch-free skips cover optional torch-dependent tests and existing platform-specific tests. No training,
self-play campaign, optimizer experiment, paid provider traffic or sealed input
was run to eliminate skips. Bounded v2 tests deliberately deselect14 training,
generation and resume cases. Existing optional native-platform skips remain.

Test iteration failures are retained rather than omitted: initial frontend
integration16 failures were discovery-unaware mocks/lifetime timing/status-role
issues, fixed with440/440 legacy units preserved; first new focused suite3 failed
on boolean-null handling and test move-config fixtures, fixed. Docker11 failures
were inherited service flags contaminating pytest, corrected by sanitizing the
subprocess env (also inCI). Initial22 browser launches failed before execution
because Chromium was absent, then5/2 native-test interaction/replay assertions
were corrected, ending22/22. Initial broad legacy browser run77 failed/82 passed/
1 skipped due catch-all mocks treating discovery as history plus obsolete labels/
presets. Subsequent full run14 failed/169 passed/1 skipped isolated remaining
stale selectors and one legacyMatch error-message regression; all14 passed after
specific corrections. Backend mock tests initially5 failed because a Python path
with spaces broke generated shebangs;13 now pass with a portable temp interpreter
symlink. One concurrent staging release probe saw503 search contention; serial
ready-server probes pass. These are resolved iterations, not excluded failures.

The first combined Firefox/WebKit attempt reused sessions left by earlier runs
and exhausted the unchanged 128-session staging capacity: 54 passed, 6 failed,
1 interrupted and 53 not run. One Firefox busy/pause failure additionally exposed
pointer scheduling ambiguity in the test; the explicit busy response is now held
until Pause completes. No application concurrency/session limit was raised.
Resetting only the phase-owned staging container between browser projects gives
each suite a clean session store. Firefox then passed all 57 cases; the final
WebKit result is recorded above. Separate output directories prevent concurrent
test artifact cleanup. This initial failed attempt is retained as test evidence.

Existing gameplay, sequence/revision, schedule, draw, seed, compact-history and
winner assertions were retained. Browser mock backends now explicitly answer
optional discovery. Legacy browser selectors were updated to the backend branch's
existing ClassicMCTS names/custom budgets/preset counts, preserving exact request
budgets and duplicate-configuration/bracket assertions. Frozen export fixtures were
not edited. Full browser suites are rerun, not replaced by focused success alone.

## 16. Browser scenario coverage

The AlphaZero matrix covers available/disabled/unavailable/404/malformed/network
discovery; Human in both colors with each guard; restart/fixed active config;
AlphaZero versusHuman/Negamax/ClassicalMCTS/TacticalMCTS/Victor/AlphaZero; guarded
versus unguarded; step/autoplay/pause/busy cancellation; timeline/replay; uncertain
committed response reconciliation; active unavailability/discovery refresh; small
Tournament and Season with both variants; persisted reload/local replay; v3 JSON
export+self-verification; mobile320/375px and native keyboard typeahead. Added
reverse-Human Match and position/last-AI mocked explanation cases complete the
matrix. Legacy canonical, seeded comparisons, ratings and larger synthetic fields
remain covered by their existing unit/browser suites.

## 17. Independent review

[Independent report](phase4d4k-b-independent-review.md) was written by a reviewer
who wrote none of implementation/tests. Three initial findings were fixed and
re-reviewed: major Play provenance recovery unlock; major AlphaZero-to-Victor
Play switching; minor Match mismatch messaging. A later minor lock-onlyCI path
filter omission was fixed/re-reviewed. Candidate-ID cleanup, all nine upstream
hashes/dependency closure, CI lifecycle, and per-session backend pinning received
additional targeted review. The reviewer independently reproduced v1/v2 byte
compatibility, v3 tamper/rehash rejection, real both-color AlphaZero moves, busy
pause/unmount, unavailable/mismatched session pinning,86 final frontend tests and13
mocked deployment tests. No unresolved material finding was identified. The
reviewer's documented limits (including remoteCI/full browser evidence) are
retained. Final controller recovery, legacy Match error wording, homepage/guard
copy and busy/pause browser harness corrections received targeted re-review.
For both controllers, the reviewer independently preserved completed-game bytes
and two validated unfinished plies through a model mismatch, blocked restart
under the changed identity, then explicitly restarted only the unfinished game
after restoring the original model and seed. No unresolved major issue remains.

## 18. Historical preservation

No research/campaign/checkpoint/benchmark path was written. All17 other worktrees
(including the original historical checkout) were observed with0 tracked changes.
Selected artifactSHA remains `4c3cf26aa58495f43094436f231486969658d70b94b246a912a5bb54756dc528`;
manifestSHA `004e76c685ad011d0bcb4d67aeedee10372069d7621600c717ac27aa6f3449cc`;
guardSHA `d2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04`.
The canonical 112-game benchmark, old export fixtures, all K-A evidence and
production model paths are absent from the branch diff. Original H-C generation-6
checkpoint hashes were checked read-only: seed42
`ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b`;
seed314159 `2646d7cdee7d7b069412688bf95fd9794bea30402b4952120c904dbcb399401d`.
Live container read-only
before/after identity observations match. Original campaign/checkpoints remain
in their untouched historical worktrees. Sealed fixtures were never opened or
searched; the older broad preservation script was not run because it hashes
sealed payloads. No claim is made to a new per-file sealed audit.

## 19. Remaining limitations and launch recommendation

Final remoteCI/browser evidence must be green before declaring release readiness.
There is one AlphaZero search per process; UI busy handling bounds demand but does
not establish per-user fairness. Neural floating-point runtime differences may
change future decisions; game seeds ignoreAlphaZero and do not reproduce research
streams. Compact replay keeps columns/model identity rather than full diagnostics.
OptionalAMD64 and base OS/API supply-chain limitations are explicitly documented.
No public latency/availability claim can be made before separately approved
production verification. No unresolved review finding is bypassed to finish.

## 20. Future production release and rollback (approval required)

1. Review this branch, final test evidence and native ARM64CI. Separately approve
   merging and separately approve public deployment; neither occurred here.
2. In the **approved Services release checkout**, fetch and select the exact
   approved releaseSHA; confirm clean status and available resource headroom.
   Preserve the private env file, CORS/Funnel/explanation settings and search limits.
3. Run the reviewed `bash scripts/deploy-mac-mini-api.sh alphazero`. Omission or
   `lite` remains torch-free/disabled. Candidate localhost port 8001 must pass all
   probes before replacing the live container. Keep the stopped rollback container.
4. Release the frontend from the approvedSHA with the existing Render static-site
   procedure/API origin; optionally set its evaluation source commit. Do not use
   a VITE availability flag. Verify public discovery, both Play colors/guards,
   small Labs/export, mobile/CORS/health and resource use without paid explanations.
5. Keep rollback until public behavior is verified. On failure, the script restores
   the old container automatically; manual approved rollback is:

   ```sh
   docker rm -f bgai-api
   docker rename bgai-api-rollback bgai-api
   docker start bgai-api
   curl -fsS http://127.0.0.1:8000/v1/connect4/health
   ```

   Funnel still points at8000. Sessions reset on cutover/rollback. If frontend
   rollback is needed, release the previous staticUI revision; it tolerates an
   absentAlphaZero endpoint. Inspect public core/source provenance and existing
   agents before declaring recovery. Remove rollback deliberately only after
   stable verification. These commands are future operator instructions only.
