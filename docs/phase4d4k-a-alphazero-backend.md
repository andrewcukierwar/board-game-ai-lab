# Phase 4D.4K-A — AlphaZero inference packaging and backend integration

## 1. Executive summary

**Outcome: backend integration complete on an isolated branch; ready for Phase
4D.4K-B frontend work and a separately planned, controlled deployment. Not deployed.**

| Question | Result |
|---|---|
| Model staged | Phase 4D.4H-C generation-6 CNN, seed 42 (provisional engineering choice, not a strength claim) |
| Artifact | `models/alphazero/az2-hc-g6-cnn-s42-v1.pt`, 1,309,171 bytes, inference-only, sha256 `4c3cf26a…c528`, from source checkpoint `ba2bde2e…f58b` |
| Research equivalence (research runtime) | **Exact**: 12,485 production decisions = every seed-42 4D.4J-A suite record + every seed-42 4D.4J-A/4D.4J-B match decision, plus 664 mirror/both-actor/full-column decisions; 0 mismatches, max float difference 0.0 |
| Research equivalence (Docker linux/arm64) | **Within the declared tolerance**: same 12,485 decisions, 0 visit/action/guard mismatches; max float difference 9.06e-6 (≤ 1e-4 declared beforehand); encodings bitwise identical |
| Feature flag | `ALPHAZERO_ENABLED=false` by default; disabled path never imports torch; `/provenance` unchanged |
| Configurations | `{"type":"alphazero","tactical_guard":false}` (default, original research agent) and `true` (same checkpoint and search plus the verified root guard) |
| Move latency (Docker, warm) | mean 53.6 ms, median 17 ms, p95 131 ms, worst 153 ms (guarded = unguarded; guard 0.05 ms mean) |
| Cold start | AlphaZero available 1.25 s after container start (torch import 0.58 s, strict load + self-check 0.03 s) |
| Memory | +171 MiB container memory (51 → 222 MiB), stable over repeated games |
| Docker | Default image unchanged in kind (349 MB, torch-free). Opt-in `ALPHAZERO_RUNTIME=true` image 1.3 GB (CPU-only torch 2.10.0, no CUDA) |
| Tests | Torch-free 3064 passed / 17 skipped; torch 3487 passed / 1 skipped / 37 deselected (training/self-play); 107 AlphaZero tests also pass inside the linux/arm64 image; frontend 440/440 |
| Independent review | PASS WITH NOTES (review and re-review), no blocker/major; all three MINOR findings fixed (§17) |
| Preservation | Intact: J-A/J-B hash inventories (3,950 / 5,607 files), 150 official campaign files, both checkpoints, all research worktrees, live container and Services checkout |


Scope guardrails held throughout: nothing was merged into `main`, nothing was
deployed, the live Mac mini Services checkout and `bgai-api` container were not
touched, no Render configuration changed, AlphaZero is not enabled publicly, no
model was promoted, and no training or research experiment ran.

**What the guard is.** A conventional one-ply tactical safeguard applied after
the unchanged 256-simulation search. Phase 4D.4J-B measured a repeatable but
**small** match-score effect, +1.302 pp (95% [+0.521, +2.344]), below its own
+2 pp practical margin, with no observed tactical or exact-WDL regression.
It is offered as optional blunder protection, **off by default**; it is not a
stronger or separately trained agent.

## 2. Production architecture findings

| Component | Finding | Integration decision |
|---|---|---|
| Live backend | Mac mini Services checkout at `8704142` (= `origin/main`), container `bgai-api:8704142702da`, Docker bound to `127.0.0.1:8000`, Tailscale Funnel in front, Render static UI | Branch from `origin/main`; never touch Services/`bgai-api` |
| Flask app | `api/app.py` `create_app`; process-local `GameStore`; per-session lock + revision guard; global token-bucket limiter; `connect4_search_gate` (`PUBLIC_SEARCH_CONCURRENCY=2`) | Reuse all of it unchanged |
| Agent dispatch | `api/connect4/__init__.py` `player_config` (strict) and `make_move` (per-request agents, nonblocking reservations, commit only after a legal move) | Add `alphazero` following the opt-in Victor pattern |
| Legacy factory | `games/connect4/agents/agent_factory.py` is CLI-only, imports torch-based `mcts_nn`, not used by the API | Not modified |
| History | `MoveRecord` immutable, built before commit; `to_dict` serialized in `/games/<id>/history` | Additive optional `alphazero` evidence for AlphaZero moves only |
| Provenance | `/provenance` manifest is compared **exactly** by the frontend evaluation verifier (`ui/src/evaluation/provenance.js`); unknown keys stop Season runs | Left byte-identical; AlphaZero status is a separate endpoint |
| Labs | Match/Tournament/Season Labs are frontend orchestrators over `start_game`/`make_move` with one mutation at a time | Backend supports AlphaZero in labs with no backend change; UI changes are 4D.4K-B |
| Gunicorn | 1 worker, 4 threads, no `--preload` | Model loads once in the worker after fork; do not add `--preload` |
| Docker | `python:3.11-slim`, torch-free `requirements-api.txt`, `.dockerignore` excludes `**/*.pt` | Optional `ALPHAZERO_RUNTIME` build arg; one artifact allow-listed |

## 3. Git ancestry and integration strategy

- Research branches (`phase4d4a` … `phase4d4j-b`) and production `main` diverge at
  `a58abcc`. `main` has 151 newer production commits; research has 40 commits
  and ~1,450 changed files (12.4 M inserted lines, mostly evidence).
- On `main`, `games/connect4/alphazero_v2/` holds the 4D.3A core (artifacts,
  config, data, generation, network, search, selfplay, training), unchanged since
  `a58abcc`. Research changed `network.py`, `search.py`, `data.py` additively and
  added `bitboard.py` plus many research-only modules (campaign, experimental,
  oracle, packages, evaluation, frozen fixtures including **sealed** packages).
- The research search path's other dependencies (`agents/mcts_nn_agent.py`,
  `neural_mcts.py`, `connect4.py`) are **blob-identical** on base, `main` and research.
- Branch `phase4d4k-a-alphazero-backend` starts at `origin/main` `8704142`.
  Only five research files were brought in, **byte-identical** to research commit
  `99ceb370cf9c197627f27e86099976304597902e` (test-enforced SHA-256 locks):

| Production path | Research source | SHA-256 |
|---|---|---|
| `games/connect4/alphazero_v2/bitboard.py` (new) | same path | `61a262e4…5229` |
| `games/connect4/alphazero_v2/network.py` | same path | `097712ae…ee5c` |
| `games/connect4/alphazero_v2/search.py` | same path | `49d719df…993d` |
| `games/connect4/alphazero_v2/data.py` | same path | `dbe4fec9…e254` |
| `games/connect4/alphazero_v2/tactical_guard.py` (new) | `research/alphazero_tactical_guard/guard.py` | `d2ee54ba…2d04` |

  No experimental branch was merged or cherry-picked; no research-only module,
  frozen fixture or sealed package is on the branch. No architectural conflict was
  found: the research changes are additive (`backend='engine'` remains the default,
  and main's AlphaZero v2 tests pass on the ported files).
- Unrelated unmerged work (Victor branches, `research/mcts-v3`, `search/negamax-v2`,
  `hardening/mac-mini-public-api`) was not touched. The historical checkout
  (`phase4d3-m4-runtime`, clean) and all research worktrees were only read.

## 4. Selected checkpoint and justification

**Selected (provisional engineering choice): Phase 4D.4H-C generation-6 CNN, seed 42.**
This is not a claim that it is the strongest trained model.

| Evidence (development fixtures; 256 sims) | Seed 42 | Seed 314159 |
|---|---:|---:|
| 4D.4J-A tactical200 correct, control → guarded | 187 → 200 | 186 → 200 |
| 4D.4J-A avoidable errors / missed wins (control) | 10 / 3 | 13 / 1 |
| 4D.4J-A solved300 optimal, control → guarded | 76.33% → 77.67% | 77.67% → 79.00% |
| 4D.4J-A deep128 optimal | 76.56% | 75.00% |
| 4D.4J-A match score vs Negamax3 (40 families), control → guarded | 57.50% → 60.00% | 46.25% → 48.13% |
| 4D.4J-B score vs Negamax3 (96 new families), control → guarded | 54.43% → 55.47% | 46.62% → 49.22% |
| 4D.4J-B score vs Negamax5, control → guarded | 36.20% → 36.72% | 37.50% → 38.54% |
| 4D.4J-B guard effect | +0.78 pp [0.00, +1.82] | +1.82 pp [+0.52, +3.39] |

- **Playing strength:** seed 42 scored higher against Negamax3 in both studies
  (+11 pp in 4D.4J-A, +7.8 pp in 4D.4J-B); against Negamax5 the seeds are
  similar (314159 is 1.3 pp higher). Seed 42's measured guard gain is smaller.
- **Tactical and exact value:** mixed and close; neither seed dominates.
- **Compatibility:** both gen-6 checkpoints load through the strict 4D.4J-A loader,
  are the exact checkpoints the guard was validated on, and have the identical
  architecture (325,896 parameters), so size, load time and latency are equal.
- **Generation 20 (official campaign):** the only measurement is Phase 4D.4G's
  value MSE (0.59–0.61 vs the constant's 0.667). There is no match, tactical or
  guard evaluation under the 4D.4J protocol. It needs the official-campaign loader,
  which 4D.4J-B excluded as not verified strict-compatible. Choosing it would have
  needed new evaluation (not authorized) or a weaker loader, so it was not selected.
  Generation number is not evidence of strength here.

The existing evidence does not strongly favour another checkpoint, so the user's
provisional preference stands.

## 5. Source checkpoint

| Field | Value |
|---|---|
| Checkpoint | `phase4d4h-c-worktree/experiment-output/phase4d4h-c/run/tasks/42-cnn-g6-train/generation-0006.pt` |
| SHA-256 | `ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b` |
| Size | 4,903,328 bytes (full resume boundary: model, AdamW, replay, RNG) |
| Ledger | `research/alphazero_resnet_selfplay/results/run-evidence/tasks/42-cnn-boundaries.json` |
| Identity | CNN `connect4-conv64-128-128-policy7-scalar-tanh-v2`, original schedule, generation 6, tensor hash `aa008c3c…b646` |

## 6. Inference artifact contract

Two layers, reusing the existing contract instead of inventing a format:

1. **Checkpoint**: the existing v2 inference checkpoint (`network.save_inference_checkpoint`):
   exactly `{"contract": INFERENCE_CONTRACT, "model_state_dict": …}` where
   `INFERENCE_CONTRACT` is `connect4-alphazero-v2-inference` format v2. It holds no
   optimizer, replay, RNG, training or provenance state.
2. **Manifest** `connect4-alphazero-production-artifact` v1 (`models/alphazero/az2-hc-g6-cnn-s42-v1.json`):
   artifact file/SHA-256/size, the inference contract, architecture identity and
   parameter count, tensor-content hash, fixed search settings, guard contract and
   source SHA-256, source checkpoint identity and research commits, export tool
   commit/SHA-256/runtime/gates, and 12 reference predictions.

| Property | Value |
|---|---|
| Model id | `az2-hc-g6-cnn-s42-v1` |
| Artifact | `models/alphazero/az2-hc-g6-cnn-s42-v1.pt` |
| Size | **1,309,171 bytes** (325,896 float32 parameters) |
| SHA-256 | `4c3cf26aa58495f43094436f231486969658d70b94b246a912a5bb54756dc528` |
| Manifest SHA-256 | `004e76c685ad011d0bcb4d67aeedee10372069d7621600c717ac27aa6f3449cc` |
| Tensor hash | `aa008c3c3b166a36a754a45ccb00384826e9b76ca4b0280049b7600a0dadb646` (equal to the strict research load) |
| Distribution | committed to Git and copied into the image as an immutable layer; no download, no external hosting |
| Loading needs | Python 3.11, CPU PyTorch 2.10.0, ~141 MiB extra resident memory |

## 7. Artifact generation (reproducible)

`scripts/alphazero/export_inference_artifact.py` runs **only** in the research
environment, with every `games`/`research` import resolved to the clean research
commit:

```sh
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  ~/.venvs/board-game-alphazero-m4/bin/python -B scripts/alphazero/export_inference_artifact.py \
    --research-root experiment-output/phase4d4j-b-worktree --output-dir <new directory>
```

Gates, all enforced and all PASS in the recorded export (integration commit `26df82b`):

1. The export tool is committed (its commit and SHA-256 are recorded).
2. The research root is exactly `99ceb37` with no tracked modifications; thread variables are 1.
3. The 4D.4J-A `enforce()`: frozen execution sources, runtime identity and checkpoint bytes.
4. The 4D.4J-B freeze: the guard SHA-256, the runtime identity and every frozen source, except one
   documented post-freeze file. `independent_verify.py` was added by the results reviewer in
   `99ceb37`, after the `ae4715f` freeze; it is checked against the research commit blob.
5. Strict 4D.4J-A load of seed 42 generation 6: SHA-256, size, ledger and freeze checkpoint entry,
   original schedule, CNN architecture and parameter count.
6. Save through the existing atomic, no-overwrite `save_inference_checkpoint`.
7. Reload: identical keys/contract, `torch.equal` on every tensor, identical tensor hash, and
   bit-identical predictions on the reference positions.
8. Learner/optimizer/replay digests, the weights and the source checkpoint bytes are unchanged.
9. The manifest is written with exclusive create (no overwrite).

Re-exporting into the existing directory fails by design. A changed model requires a new
model id, a new approved hash in `production.py` and review.

## 8. Artifact integrity checks (production loader)

`games/connect4/alphazero_v2/production.py` `load_production_model`, run once per
process at application start:

| Check | Failure code |
|---|---|
| Manifest readable, ≤ 256 KiB, strict JSON (no NaN), exact key set, format v1 | `artifact_missing` / `artifact_contract` |
| Artifact entry is a plain sibling file name (no traversal), size ≤ 8 MiB | `artifact_contract` |
| Inference contract, architecture identity and 325,896 parameters | `unsupported_architecture` |
| Search settings equal the validated settings; guard contract | `artifact_contract` |
| Model id approved, and **both** the manifest SHA-256 (`004e76c6…49cc`) and the artifact SHA-256 equal the hashes pinned in code (any manifest edit needs a code change) | `artifact_integrity` |
| One nonblocking no-follow open; must be a regular file (no symlink, directory or FIFO); size checked on the descriptor; capped read; SHA-256 of the **same in-memory bytes** that are loaded | `artifact_missing` / `artifact_integrity` |
| `torch.load(weights_only=True)` + strict keys/shapes/dtypes/finite (existing loader) | `unsupported_architecture` |
| Tensor-content hash equals the manifest | `artifact_integrity` |
| Reference predictions within 1e-4 (declared before cross-runtime testing) | `self_check_failed` |

Arbitrary pickles are rejected by the weights-only loader (tested with a
`__reduce__` payload). Failures are not cached, and success is cached per (manifest, approved hash).

## 9. Backend agent implementation

- `ProductionModel.decide(game, tactical_guard, rng=None)` mirrors the 4D.4J-A
  `evaluate.decision` line for line: `BitboardState.from_game`, then
  `EvaluationAgent(inference, 256, rng, backend='bitboard').search(root)`, then assert
  256 visits, no root noise and temperature 0, then the guard if requested, then return.
  Before searching, `validate_batch_state` rejects inconsistent turn counts, floating
  stones, existing winners and terminal boards.
- No PUCT was rewritten; search, network, encoding and backup code is the research code.
- **Determinism:** each decision gets a fresh tie-break RNG derived only from the exact
  position (`connect4-alphazero-production-tie-break-v1`). Identical positions give
  identical moves, `rng_seed` has no effect on AlphaZero, and no RNG is shared between
  games or threads. The equivalence harness injects the research RNG streams instead.
- `games/connect4/agents/alphazero_agent.py` `AlphaZeroAgent(model, tactical_guard)`
  follows the existing `choose_move(game)` contract; `last_decision` keeps diagnostics.
- Works as first or second player, from the empty board or any legal position, with full
  columns and near-terminal boards. Both colours and mirrored boards are tested.

## 10. API configuration and feature flags

Backend-only flags (never `VITE_*`):

| Variable | Default | Meaning |
|---|---|---|
| `ALPHAZERO_ENABLED` | `false` | `true`/`1` exposes AlphaZero for new games; anything other than true/false/1/0 fails startup |
| `ALPHAZERO_MANIFEST` | packaged manifest | optional override; must be a regular file (symlinks rejected); the pinned hashes in code still apply |

Player configuration (canonical stored form always includes the flag):

```json
{"type": "alphazero", "tactical_guard": false}
{"type": "alphazero", "tactical_guard": true}
```

`tactical_guard` must be a JSON boolean and is never coerced (`"true"`, `1` and `null` are
rejected). Unknown keys (`simulations`, `simulation_limit`, `depth`, `temperature`, …) are
rejected; simulations and search settings are fixed. A session's configuration cannot change
during a game: there is no update endpoint, and `replace_game_id` creates a new session.

| State | `GET /v1/connect4/agents/alphazero` | `start_game` with AlphaZero |
|---|---|---|
| Disabled (default) | `200 {"type":"alphazero","enabled":false,"available":false,"code":"alphazero_disabled"}` | `400 invalid_agent` "The AlphaZero agent is not enabled on this server." |
| Enabled, load failed | `200 {…,"enabled":true,"available":false,"code":"alphazero_unavailable"}` | `503 agent_unavailable` |
| Enabled, ready | `200` with `agent_version`, model id/description/architecture/parameters/artifact and source hashes, fixed search settings, `options.tactical_guard` | `201` |

Errors never fall back to another agent and never include paths, exception text or stack
traces. The specific failure code is logged privately. Model payloads are never logged.

Examples against a local staging server (`ALPHAZERO_ENABLED=true`):

```sh
# Status (the frontend should gate the option on "available": true)
curl -s http://127.0.0.1:18103/v1/connect4/agents/alphazero

# Unguarded AlphaZero as Yellow (original research agent)
curl -s -X POST http://127.0.0.1:18103/v1/connect4/start_game -H 'Content-Type: application/json' \
  -d '{"player1":{"type":"human"},"player2":{"type":"alphazero","tactical_guard":false}}'

# Guarded AlphaZero as Red; request its move without a column
curl -s -X POST http://127.0.0.1:18103/v1/connect4/start_game -H 'Content-Type: application/json' \
  -d '{"player1":{"type":"alphazero","tactical_guard":true},"player2":{"type":"human"}}'
curl -s -X POST http://127.0.0.1:18103/v1/connect4/make_move -H 'Content-Type: application/json' \
  -d '{"game_id":"<id>","revision":0}'
```

Disabled flag (default):

```text
POST /start_game {"player2":{"type":"alphazero"}}
400 {"code":"invalid_agent","error":"The AlphaZero agent is not enabled on this server."}
```

Invalid or missing artifact (enabled):

```text
GET  /agents/alphazero  200 {"type":"alphazero","enabled":true,"available":false,"code":"alphazero_unavailable"}
POST /start_game        503 {"code":"agent_unavailable","error":"The AlphaZero agent is unavailable on this server. Choose another opponent."}
log: AlphaZero unavailable (artifact_integrity): AlphaZero artifact size or SHA-256 differs from the approved hash
```

Invalid settings: `{"type":"alphazero","tactical_guard":"true"}` or `{"type":"alphazero","simulations":512}` →
`400 invalid_agent`. Busy: `503 agent_busy` ("Another AlphaZero search is running…" or "Too many AI searches…");
failure: `503 agent_failed`; neither commits a move. A successful AlphaZero history entry:

```json
{"agent":{"type":"alphazero","tactical_guard":true},"column":3,
 "alphazero":{"model_id":"az2-hc-g6-cnn-s42-v1","simulations":256,"original_column":3,
              "root_visits":[4,3,14,222,8,2,3],"tactical_guard":true,"guard_reason":"UNCHANGED"}, "...": "..."}
```

The unchanged generic "Supported player types are …" message (used for unknown types) still lists only the
public non-opt-in types, as it does for Victor.

## 11. Tactical guard integration

- `tactical_guard.py` is the research `guard.py` verbatim (SHA-256 `d2ee54ba…2d04`). Its rules:
  take an immediate win, else keep a safe original, else take the most-visited safe move,
  else keep the original (NO_SAFE_MOVE). Restricted ties go to the most visits, then the
  lowest column.
- It runs after the unchanged search and cannot change the tree, priors, values, simulations,
  backups, visits, RNG or weights. Unguarded play reproduces the original research agent exactly.
- Default **off**. Both variants use the same checkpoint and search; only the root safeguard differs.
- History (`/games/<id>/history`) records, for AlphaZero moves only:
  `agent: {"type":"alphazero","tactical_guard":bool}` and
  `alphazero: {model_id, simulations, original_column, root_visits[7], tactical_guard, guard_reason}`.
  Records of existing agents are byte-identical to before.

## 12. Determinism and equivalence evidence

Prospective rule (`benchmarks/alphazero/phase4d4k-a/tolerance-declaration.json`, committed in
`26df82b` before any cross-runtime measurement): encodings, visits, actions, legal sets, guard
outcomes and terminal handling must match exactly. Floats must match exactly in the research
runtime and within 1e-4 in any other runtime. Any unexplained visit or action change is a blocker.

Production path under test: the packaged artifact through `load_production_model` and
`ProductionModel.decide` (the API agent's own call), with the research RNG stream injected.
Research inputs were read as data (`scripts/alphazero/verify_equivalence.py`).

| Set (seed 42) | Decisions | Guard changes | Research runtime | Docker linux/arm64 |
|---|---:|---:|---|---|
| Research reference: 200 tactical, 60 solved, 32 deep **mirrors**, 12 reference boards + mirrors, 8 full-column boards + mirrors; both actors | 664 | 15 | exact | 0 int mismatches |
| 4D.4J-A tactical200 records | 400 | 13 | exact | 0 |
| 4D.4J-A solved300 records | 600 | 7 | exact | 0 |
| 4D.4J-A deep128 records | 256 | 0 | exact | 0 |
| 4D.4J-A control / guarded matches (80 + 80 games) | 940 / 994 | 0 / 8 | exact | 0 / 0 |
| 4D.4J-B control / guarded vs Negamax3 (192 + 192) | 2,001 / 2,096 | 0 / 28 | exact | 0 / 0 |
| 4D.4J-B control / guarded vs Negamax5 (192 + 192) | 2,247 / 2,287 | 0 / 18 | exact | 0 / 0 |
| **Total** | **12,485** | **89** | **PASS_EXACT** | **PASS_WITHIN_TOLERANCE** |

Each decision compares the original PUCT action, final action, all seven root visits, raw
logits, legal priors, root value and the full guard diagnostic (legal/winning/safe/unsafe sets,
witnesses, reason, flags; timing excluded). Match replays also check that every played move and
every game winner equal the record. Guard-change counts reproduce the published figures
(tactical 13, solved 7, J-A matches 8, J-B seed-42 46).

- **Encoded inputs:** 332/332 nonterminal reference boards bitwise identical to the research encoding,
  including signed zeros, through both the engine and bitboard encoders, in both runtimes.
- **Legal sets and terminal handling:** all equal; the 4 terminal boards (X vertical, O horizontal,
  X diagonal, full-board draw) are rejected by research and production alike.
- **Cross-runtime floats:** max |Δ| logits 9.06e-6, priors 8.56e-7, value 9.83e-7. No visit count,
  action or guard outcome changed.
- **Independent reproduction:** the reviewer's own code matched 6,645 historical decisions and 154
  direct research-vs-production decisions exactly, plus a 1,410-decision Linux sample (§17).
- **Default serving RNG (important caveat):** equivalence is exact *given the same RNG stream*. The
  API uses a deterministic per-position tie-break stream instead of the research per-game streams,
  which no position-only rule can reproduce. Policy, values, search and settings are identical, but
  tie-breaks may resolve differently: the reviewer measured 8/1,256 suite decisions (0.64%) with a
  different original action and 5/1,256 with a different final action. Served games are therefore
  "the same agent", not replays of the evaluated games.

Evidence files: `benchmarks/alphazero/phase4d4k-a/` (`equivalence-macos-research-runtime.json`,
`equivalence-docker-linux-arm64.json`, `research-reference.json`, `tolerance-declaration.json`), and
`equivalence-docker-linux-arm64-before-openmp-bound.json`, an earlier Docker run with identical results
taken before the thread fix in §13.

To reproduce:

```sh
# research-runtime reference (research venv, research commit)
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 ~/.venvs/board-game-alphazero-m4/bin/python -B scripts/alphazero/research_reference.py \
    --research-root ../phase4d4j-b-worktree --manifest models/alphazero/az2-hc-g6-cnn-s42-v1.json --output <new.json>
# production verification (any runtime with torch)
python scripts/alphazero/verify_equivalence.py --research-root ../phase4d4j-b-worktree \
    --reference benchmarks/alphazero/phase4d4k-a/research-reference.json --workers 4 --output <result.json>
```

In Docker, copy the scripts, `benchmarks/` and the ten research JSON inputs in with `docker cp` rather
than bind-mounting `/Volumes` paths: Docker Desktop stalled waiting for removable-volume access.

## 13. Docker and dependency impact

| Image | Disk | Compressed | Contents |
|---|---:|---:|---|
| Live `bgai-api:8704142702da` | 346 MB | 74.8 MB | reference |
| `bgai-k-a-staging:lite` (default build) | 349 MB | 76.0 MB | + 1.3 MB artifact, manifest and code; **no torch** |
| `bgai-k-a-staging:alphazero` (`--build-arg ALPHAZERO_RUNTIME=true`) | 1.3 GB | 264 MB | + `torch==2.10.0+cpu` (611 MB) from the official CPU index; no CUDA, nvidia or triton packages |

- **ARM64:** the CPU wheel installs and runs on linux/aarch64 (`torch.cuda.is_available() == False`).
  An invalid `ALPHAZERO_RUNTIME` value fails the build.
- **Default build:** the image still contains no ML framework; `ALPHAZERO_ENABLED=true` on it reports
  `alphazero_unavailable` (logged `runtime_unavailable`), health and existing agents keep working,
  and the CI smoke test is unaffected.
- **Thread fix (found in staging):** on linux/arm64, `torch.set_num_threads(1)` alone left about one
  spinning GNU OpenMP worker per core, sized from `OMP_NUM_THREADS` at first torch import. Each search
  used 3.2 cores and ran about 2.6× slower (350 ms vs 136 ms per opening decision under load). The
  enabled service now sets `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to 1 with `setdefault` before importing
  torch (test-covered). Each search now uses exactly one core (CPU/wall 0.999). The disabled default
  image environment is unchanged.
- **Gunicorn:** keep the existing no-`--preload` command. Forking after torch initialisation deadlocked
  the equivalence harness on Linux until it switched to spawn workers.
- **Alternatives considered:** ONNX Runtime or a custom NumPy forward pass would be lighter, but would
  replace the verified inference implementation and lose exact equivalence; a separate inference server
  would add a second public surface for no operational gain. The opt-in build arg keeps the default
  deployment unchanged.
- **Distribution:** the 1.3 MB artifact is versioned in Git and baked into an immutable image layer
  (`.dockerignore` re-includes exactly that `.pt`). No download, no external hosting, no training state
  in the image.
- **Open:** torch is version-pinned, not hash-pinned (`--require-hashes`). The deploy script builds
  only the default image, so enabling AlphaZero in production needs a reviewed deploy change (§19).

## 14. Memory and latency measurements

Packaged production path, warm, one torch thread, 100 seeded legal positions (10 per phase × actor),
each measured twice per variant (`scripts/alphazero/benchmark_backend.py`; JSON in `benchmarks/alphazero/phase4d4k-a/`).
These are bounded local/staging measurements on the Mac mini M4 (Docker Desktop VM: 10 CPUs, 8 GB),
not public-backend measurements.

| Docker linux/arm64 (ms) | Mean | Median | p95 | Worst |
|---|---:|---:|---:|---:|
| Network inference (one position) | 0.49 | 0.49 | 0.50 | 0.91 |
| Unguarded move | 53.6 | 17.1 | 130.4 | 152.2 |
| Guarded move | 53.6 | 16.8 | 130.9 | 152.7 |
| Guard alone | 0.05 | 0.05 | 0.07 | 0.21 |

| Phase (Docker, guarded) | Opening 0–6 | Early middle 7–13 | Midgame 14–24 | Late 25–34 | Near-terminal 35–41 |
|---|---:|---:|---:|---:|---:|
| Mean ms | 129.9 | 82.4 | 38.2 | 13.9 | 3.8 |
| p95 ms | 149.3 | 128.9 | 123.4 | 61.8 | 7.5 |

By actor (Docker, guarded): X to move 54.4 ms, O to move 52.8 ms. Late positions are cheaper because
simulations end at terminal leaves without network calls.

| Other measurements | Value |
|---|---|
| macOS native (same code) | moves mean 23.3 / median 7.5 / p95 55.4 / worst 77.7 ms; inference 0.20 ms |
| HTTP end to end (staging, 6 full AlphaZero-vs-AlphaZero games, 252 moves) | mean 105.6 ms, p95 ≤ 149 ms, worst 171 ms |
| Cold start (Docker) | torch import 576 ms + strict load and self-check 31 ms; container start to `available` 1.25 s |
| Cold start (macOS) | torch import 741 ms + load 51 ms |
| CPU | CPU/wall 0.999 during searches: one core per active search |
| Memory, container | idle lite 51 MiB; AlphaZero idle with model 192 MiB; after games 214–222 MiB, flat across 24 further full games (+171 MiB) |
| Memory, worker process | RSS 52 MB (lite) → 293 MB (AlphaZero, includes shared torch pages); benchmark peak RSS 264 MiB |
| Model loads | exactly one per process (cached; test-covered) |

Research latency (4D.4J-B: 49 ms mean PUCT, 0.065 ms guard) was macOS native with four parallel tasks.
Production Docker is about 2.3× slower per move than native macOS on the same machine; the guard's cost
is unchanged and negligible.

## 15. Concurrency and failure testing

- **Limits (unchanged public settings):** an AlphaZero move holds a dedicated one-per-process
  reservation **and** a shared `connect4_search_gate` slot (`PUBLIC_SEARCH_CONCURRENCY=2`). Both are
  nonblocking, so requests never queue. The existing session lock, revision guard, rate limits and
  session capacity apply unchanged. No limit was raised.
- **Simultaneous requests (staging HTTP):** 5 bursts of 4 simultaneous AlphaZero moves on different
  games. 5 ran, 15 returned `503 agent_busy`, 0 partial commits (every refused game stayed at
  revision 0).
- **Tests (flag on, real model):**
  - A move blocked mid-search makes a second game's move get `agent_busy` and the same game's move get
    `game_busy`; neither commits. Both succeed after release.
  - Holding every search-gate slot gives "Too many AI searches" with no commit; it succeeds after release.
  - A stale revision returns 409 without searching.
  - `RuntimeError`/`ValueError`/`MemoryError` inside the search return `503 agent_failed` with no exception
    text; revision and history are unchanged, and the gate and reservation are released.
  - An illegal agent output is rejected without commit.
  - A model initialisation failure leaves the agent unavailable, with health and other agents unaffected.
- **Thread safety:** one immutable eval-mode model, autograd disabled, `inference_mode`, one torch thread.
  Every decision builds its own `EvaluationAgent`, tree and RNG; nothing mutable is shared between games.
  Four-thread concurrent decisions equal sequential ones exactly (test-covered).
- **Repeated games:** 30+ full staging games and repeated in-process games show no growth in memory or
  latency, and the model is never reloaded.
- **Note:** one client can keep the single AlphaZero slot busy, so others see `agent_busy`. This is the
  same as MCTS/Victor; revisit before public enablement (§19).

## 16. Regression test results

| Suite | Environment | Result |
|---|---|---|
| Full backend suite, baseline (`origin/main`) | torch-free production deps | 3047 passed, 15 skipped |
| Full backend suite, final | torch-free production deps | **3064 passed, 17 skipped** (skips: torch modules) |
| Full backend suite, final | research torch 2.10 venv | **3487 passed, 1 skipped, 37 deselected** |
| AlphaZero tests (107) | inside the final linux/arm64 AlphaZero image | **107 passed** (exercises the cross-runtime tolerance branch) |
| Existing `test_alphazero_v2.py` on the ported modules | torch venv | 93 passed, 11 deselected |
| Frontend unit/contract tests (`ui`, `npm test`, Node 22) | local | **440 passed, 0 failed** (no UI files changed) |

New tests:
- `tests/test_alphazero_provenance.py` (torch-free): ported-file and guard hashes, artifact/manifest
  binding, and the disabled backend never importing torch.
- `tests/test_alphazero_api_disabled.py` (torch-free): flag parsing, disabled errors, unchanged
  provenance and health, existing agents with the flag on or off, unavailable status without details.
- `tests/test_alphazero_production.py`: strict loader failure codes (missing, flipped byte, truncated,
  unapproved, any manifest edit, architecture, contract, parameters, traversal, symlinks, directories, FIFOs, search
  settings, extra fields, weights hash, reference drift, missing source, pickle payloads); load-once;
  decisions in every phase and both colours; mirrors; full columns; terminal/malformed rejection; all
  four guard reasons; research-reference equivalence; thread safety; determinism.
- `tests/test_alphazero_api.py`: coverage items C, H–Y. These are: status and settings; canonical and
  invalid configs; first/second player; both sides; full games and history replay; guard rules
  re-checked by an independent matrix scan against seeded Random; concurrency, capacity, stale
  revision, exception cleanup, illegal output; lab-style seeded and replaced starts; one model reused;
  the OpenMP bound.

Not run, by design:
- DQN and neural self-play modules, and the alphazero_v2 training, self-play, generation and resume
  tests: these train networks or generate self-play. This follows the research phases' precedent, and
  no training was authorized.
- No paid LLM provider calls; explanation tests use their existing mocks.

## 17. Independent reviewer findings

An independent reviewer agent wrote none of the implementation and did not use the author's
verification scripts as evidence. It reviewed `26df82b`+`5e4ce91`, then re-reviewed `5e4ce91..72562db`.
Its full report and scripts were kept outside the repository, in the session scratch area.

**First review: PASS WITH NOTES (no BLOCKER or MAJOR).** Independent verifications, all PASS:

| Check | Result |
|---|---|
| A. Ported sources | 4 modules byte-identical to `99ceb37`; guard `d2ee54ba…2d04`; transitive search dependencies identical |
| B. Artifact contents | Original checkpoint loaded weights-only: 16/16 tensors `torch.equal`/bitwise identical to the artifact; 325,896 parameters; own tensor hash `aa008c3c…` = stored `model_hash` = manifest. Pickle globals only `_rebuild_tensor_v2`, `FloatStorage`, `OrderedDict`: no optimizer, replay or RNG state |
| C. Historical decisions | Own harness: **6,645** stored seed-42 decisions (all J-A suites; all 384 J-B guarded games + 80 control) with 0 mismatches. Ran research `evaluate.decision` on the original checkpoint: **154/154** decisions on 77 own positions (mirrors, 41 O-to-move) identical to production. API path 12/12. Linux sample 1,410 decisions: actions/visits/guard identical, max float diff 9.06e-6 |
| D. Values and terminals | Terminal/malformed roots rejected; winning child has Q = −1 for the loser, so it scores +1 for the mover, for X and O; encoding equals engine on 200/200; bitboard/engine terminal agreement on 6,420/6,420 plies |

| Finding | Disposition |
|---|---|
| MINOR-1: manifest not pinned; missing `source` fields could make `/agents/alphazero` return 500 | **Fixed** in `72562db`: manifest SHA-256 pinned with the artifact hash; `source` validated. Re-review confirmed (including a one-space manifest edit → rejected) |
| MINOR-2: stat-then-read TOCTOU, unbounded reads | **Fixed** in `72562db` (single no-follow open, fstat, capped read). Re-review confirmed |
| NOTE-1: default serving RNG differs from research streams (0.64% different original action on suites) | By design; documented in §12 |
| NOTE-2: cross-runtime evidence uncommitted; mid-review edits "by another session" | Those edits were the author's (thread fix and spawn). Evidence committed and re-reviewed |
| NOTE-3: one AlphaZero search per process; one client can keep it busy | Same as MCTS/Victor; listed for public enablement (§19) |
| NOTE-4: stale text | Package docstring fixed. The generic unsupported-type message is unchanged on purpose (§10). The export tool's docstring mentions a non-existent `verify_inference_artifact.py`; the tool is left byte-identical because its SHA-256 and commit are recorded in the manifest. Independent artifact verification is `load_production_model` plus `verify_equivalence.py` and the tests |
| NOTE-5: torch not hash-pinned; deploy script builds only the default image | Open; §19 |
| NOTE-6: selection rationale not written down | Done (§4) |

**Re-review (`5e4ce91..72562db`): PASS WITH NOTES.**
- **Fixes:** both MINORs fixed. The OpenMP bound was measured in the reviewer's own container: service
  path CPU/wall 1.00 at 78–80 ms, vs torch-first (unbounded) 3.43 at 146 ms. The real gunicorn
  container served 12 moves at 131–165 ms.
- **No regressions:** macOS 154/154 bit-identical again; the Docker equivalence JSON is consistent with
  the reviewer's own per-suite maxima (7.15e-6 / 9.06e-6 / 6.68e-6). The performance and preservation
  JSONs agree with its measurements.
- **NEW-MINOR-1:** a directory artifact or manifest raised a bare `IsADirectoryError` and leaked a file
  descriptor; a FIFO blocked startup. **Fixed** after the re-review: nonblocking `O_CLOEXEC` open,
  `fstat`/`S_ISREG` on the raw descriptor, close on every path. New tests cover directory and FIFO at
  both paths, check for no descriptor growth, and confirm the FIFO no longer blocks. This last fix is
  small and test-covered but was not re-reviewed by the reviewer.
- **NEW-NOTE-1:** the bound depends on torch not being imported earlier. A warning is now logged if it
  was. Future equivalence results record `thread_environment`; the committed Docker files predate this
  field. Their thread settings are stated in §12/§13: the "before" file had no bound, the final file had
  `OMP_NUM_THREADS=1`.
- **NEW-NOTE-2:** symlinked manifests and artifacts are rejected (intentional). Operators must point
  `ALPHAZERO_MANIFEST` at a regular file, and the artifact must be a regular sibling file.

No unresolved BLOCKER, MAJOR or correctness/security issue remains.

## 18. Historical preservation audit

`scripts/alphazero/preservation_audit.py` (hash-only; result in `benchmarks/alphazero/phase4d4k-a/preservation-audit.json`): **intact**.

- 4D.4J-A `preservation()`: 22 scopes, 3,950 files, 0 changed. This covers G/H/I outputs and results,
  H-C checkpoints and evidence, I-A/I-B, docs, canonical tracked files and the historical games source.
- 4D.4J-B `preservation()`: 26 scopes, 5,607 files, 0 changed. This adds the J-A package and
  outputs, the base tracked files and the official campaign.
- All 150 official campaign files are byte-identical, with declaration hash `c8ab2227…b716`.
- Both generation-6 checkpoints have unchanged SHA-256.
- All 16 research worktrees and the historical checkout have no tracked modifications. HEADs are
  recorded as observations only.
- The branch changes nothing under `benchmarks/connect4` (including the canonical 112-game Season
  result), `models/connect4`, `ui`, the deployment doc or script, `.github` or `docker-compose.yml`.
- The live `bgai-api` container (id, image, start time, restart count) and the Services checkout
  (`8704142`) equal the pre-work baseline. Staging used only `bgai-k-a-*` containers on ports
  18101–18103.
- Sealed fixtures were never opened, searched or copied. The research inputs read were the
  sha-checked nonsealed `fixtures.json` and committed result records. The research reference is
  generated from development fixtures only.
- No historical file, checkpoint, evidence, trainer or frozen code was modified. No training or
  experiment ran.

## 19. Deployment-readiness assessment

**Backend: ready for Phase 4D.4K-B integration and a controlled staging deployment. Not yet ready
for public enablement.** No unresolved correctness or security issue; both MINOR review findings fixed.

Before AlphaZero is enabled publicly:
1. **Deploy path:** add an explicit, reviewed option to `scripts/deploy-mac-mini-api.sh` to build
   `ALPHAZERO_RUNTIME=true` and set `ALPHAZERO_ENABLED=true` in the private env file. Keep the
   rollback container procedure, and check health, provenance, `/agents/alphazero` and one AlphaZero
   move on port 8001 before cutover.
2. **Capacity:** about 50–150 ms of one core per move in Docker, one AlphaZero search at a time,
   +171 MiB memory and a 1.3 GB image. This fits the Mac mini, but confirm headroom while research
   jobs run, and decide whether one-at-a-time with `agent_busy` is the right public behaviour.
3. **Supply chain:** consider `--require-hashes` for the torch wheel set, and add a CI job that builds
   the AlphaZero image and runs the 107 AlphaZero tests. Current CI has no torch, so those tests skip.
4. **Product copy:** present the guard as optional one-ply blunder protection with a small measured
   effect (+1.3 pp, below the +2 pp practical margin), off by default. Present the model as a research
   agent that scores about 36–37% against Negamax5, not a strong or perfect player.

Remaining limitations: equivalence is exact for injected research RNG streams, not for served
tie-breaks (§12); Docker latency is VM-based staging, not the live service; two checkpoints and two
opponents underlie all strength evidence.

## 20. Next frontend integration steps (Phase 4D.4K-B)

Backend contract for the frontend (no backend change needed):

1. **Discovery:** `GET /v1/connect4/agents/alphazero`. Show the option only when `available === true`.
   Never use a `VITE_*` flag for availability.
2. **Payloads:** `{type: "alphazero", tactical_guard: boolean}`. Send `tactical_guard` explicitly, as
   a boolean, never a string. No simulation, depth or temperature controls.
3. **Labels:** e.g. "AlphaZero (CNN, 256 simulations)" and "+ tactical guard". Describe the guard as
   one-ply protection with a small measured effect.
4. **Errors:** handle `invalid_agent` (disabled), `agent_unavailable` (503, model not loaded) and
   `agent_busy` / `agent_failed` (503, nothing committed) like the existing Victor recovery flows.
5. **History and replay:** accept `agent.tactical_guard` and the optional `alphazero` evidence block in
   move records. `players[i]` carries the canonical config.
6. **Labs and evaluation exports:** `competitorConfig.js`, `matchRecord.js`, Season/Tournament models
   and `ui/src/evaluation/provenance.js` must learn the new type. AlphaZero is not in the backend
   `/provenance` manifest (kept byte-identical for the exact-match verifier). If AlphaZero games enter
   evaluation exports, introduce an explicit, versioned provenance change in both backend and frontend
   together, including the model id and artifact hash from `/agents/alphazero`.
7. **Determinism:** AlphaZero ignores `rng_seed`; the same position always yields the same move.

## Reproducible local/staging setup

```sh
git switch phase4d4k-a-alphazero-backend
# torch-free regression (production deps)
python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/python -m pytest tests -q
# AlphaZero image and isolated staging server (never port 8000/8001)
docker build --platform linux/arm64 --build-arg ALPHAZERO_RUNTIME=true -f docker/api.Dockerfile -t bgai-az-staging .
docker run -d --rm --name bgai-az-staging -p 127.0.0.1:18103:8000 -e ALPHAZERO_ENABLED=true bgai-az-staging
curl -s http://127.0.0.1:18103/v1/connect4/agents/alphazero
python scripts/alphazero/benchmark_backend.py http --base http://127.0.0.1:18103 --output /tmp/http.json
```

Production deployment is deliberately not part of this phase.
