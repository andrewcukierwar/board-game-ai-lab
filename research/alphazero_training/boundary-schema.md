# Experimental boundary version 1

`contract.format = connect4-alphazero-experimental-resume`, `contract.version = 1`.
A weights-only CPU Torch archive; historical and inference-only loaders reject
this format. Experimental loading accepts no relaxed source/runtime flags.

| Field | Contract |
|---|---|
| `contract` | Exact version, complete-generation barrier, replay and continuation contracts |
| `config` | Full frozen V2 training configuration, explicit optimizer updates, architecture, pool eight, independent-game RNG protocol |
| `schedule` | Current `TemperatureSchedule`: canonical name and complete `(first_ply, temperature)` segments |
| `lineage`, `arm` | New experimental identity; historical generation numbering supplies matched game identities only |
| `parent` | Null for scratch, otherwise parent file/state/weight/replay hashes, historical source/runtime/config/counters/history/RNGs, verified legacy schedule, conversion method, changed configuration and fork guarantee |
| `counters` | Completed generation ID, all-lineage games/positions and persistent trainer steps |
| `model`, `model_hash` | Architecture-validated model state and parameter/buffer byte digest |
| `optimizer` | Ordered parameter groups and complete AdamW moments; options, shape, dtype, finiteness and step validation |
| `replay` | Ordered complete generations and example-content digest; each generation has schedule, simulations, protocol, games_count and complete game records |
| `sampling` | Independent Python replay-sampling state |
| `augmentation` | Independent reflection RNG state, reflected and unreflected counters |
| `global_rng` | Owned process Python, NumPy and Torch CPU stochastic state, installed before each attempted generation |
| `history` | Deterministic per-generation collection, model, update, replay, loss, gradient and sampling diagnostics |
| `resources` | Separate finite generation-level collection/training/wall accounting before persistence; excluded from computational equality |
| `runtime` | Complete runtime identity, including CPU, Torch build, algorithms and all thread controls |
| `source` | Complete package/dependency file map and combined content digest; Git provenance recorded separately |
| `integrity` | Canonical recursive SHA-256 over all other fields; tensors use dtype, shape and contiguous bytes |

Each compact game record contains index, moves, winner, visits and executed
per-ply temperatures. Examples are reconstructed by legal engine replay and
finalization: observations, actor, normalized raw visits, action and final
actor-relative outcome must match. Before replay insertion, the reconstructed
completed games must equal every newly collected example. Thus no independently
stored policy/value labels can silently disagree with histories.

Schedules and budgets belong to producing generations. Legacy records use
`connect4-az-historical-sequential-v2` and the frozen parent's derived schedule;
new records use `connect4-az-independent-games-v1`. Window identities must be
contiguous and end at the completed generation, with whole oldest generations
expiring. Mixed historical/new game counts (256/64) are intentional. Once old
generations expire, no replay validation depends on interpreting their games
under a new schedule; immutable parent provenance remains.

Per-game search/action/noise seeds are SHA-derived from protocol, training seed,
generation ID, index and domain. Completed-game streams are discarded because
they cannot influence later games/generations. Scheduling draws no RNG values.
The new protocol does not reproduce historical sequential-RNG trajectories.

Integrity hashes detect corruption and unintended changes; they are not signed
attestations against an attacker who can rewrite both artifact and manifest.
Use the expected file SHA from a trusted experiment manifest for strict recovery.
Source content and runtime checks remain mandatory even with a supplied hash.
Changing code/runtime/configuration requires a separately versioned conversion;
this phase offers no relaxed experimental continuation.

Publication is one sibling temporary archive, file fsync, full strict reload,
atomic no-replace hard link, then directory fsync. The boundary is the single
completion marker: collection, replay insertion and all updates must precede it.
Failure before publication leaves no completed boundary. Failure after publication
can leave a valid complete boundary with uncertain directory crash durability.
Reload before using it. The live attempted runner is always marked failed.

Only the existing CNN adapter is registered. The small factory owns construction,
architecture/output contract, state validation, parameter hashing, freezing,
inference, trainer construction and optimizer restoration. A future ResNet
adapter must declare its own architecture and checkpoint/optimizer/output
validation; collector/replay/lifecycle/evaluation interfaces remain reusable.
No ResNet or generic ML framework is implemented.
