# Prospective Phase 4D.4C design (before results)

CPU float32, deterministic Torch, intra/inter-op 1/1, five vector thread
variables pinned to 1. Frozen eight official inference checkpoints; no training.
Reference is unchanged Phase 4D.4B bitboard PUCT with a NEW experimental game-owned
RNG protocol, never a reproduction of the official sequential RNG campaign.

Seed = big-endian SHA256 of ASCII
`connect4-az-independent-games-v1:{base_seed}:{generation}:{game_index}:{domain}`.
Domains search/action/noise are distinct; random.Random receives search/action
seeds; existing V2RootNoise receives noise seed (and retains its own domain hash).
Nonnegative integer generation/index, integer base seed. IDs sort output only;
input ID order governs scheduling and never draws randomness on waiting.

Numerical gates: batch 1 input/output bytes, predictions, every node, decisions,
completed examples and all three final RNG states must equal serial exactly.
For N>1 fixed-input max absolute error <= 2e-5 logits, <= 2e-6 values,
<= 2e-6 legal policy probabilities across all eight checkpoints. No tolerance
changes after results. Illegal probabilities exactly zero; finite tensors,
normalized targets, exact simulation counts, labels and legal replay mandatory.
Identical runtime/configuration/IDs/order/batch size/checkpoint must reproduce
all game records exactly. Fake deterministic predictions must yield identical
full trees and RNG states irrespective of scheduling order and pool size.

Fixed-state noisy search screening: root visit L1 / budget <= .10 and selected
action disagreement <= 5% over matched checkpoint/position/seed cases, comparing
both temperatures. Exceedance requires investigation and is FAIL until evidence
establishes harmless numerical sensitivity rather than a semantic defect.
Complete trajectories and outcomes may diverge through discrete PUCT/action
thresholds; report percentages and first divergence, never assert equality.
Screen >10% divergent trajectories or >5% changed outcomes as material requiring
investigation; if unresolved, numerical acceptance FAIL. Repeated-batch exact
reproducibility remains mandatory even when different batch sizes diverge.

Primary collection: 16 same IDs per checkpoint condition (42/g10 and
314159/g20), 256 simulations, base seed 4404003, generation 21. Two interleaved
trials reverse condition order. Both colors use the same frozen model in every
game, retaining empty-board X/O balance. Reference, scheduler 1/2/4/8. At most
30 minutes aggregate measurement; bounded checks each simulation and inference
batch. Add 16 only if size8 is best and memory supports another bounded sample.
Fixed-state search and inference use equal inputs/work, independently of game
length. Collection wall includes scheduler, search, capture, finalization and
callbacks; excludes checkpoint loading, hashing, warmup, tracing and GC before
trials. Automatic GC remains enabled. Retain all trials; no timing outlier removal.
Report aggregate games/s plus paired ratios; no statistical generalization from
this small development sample. Trace allocations separately; lifetime RSS fresh
process per condition; never call tracemalloc native tensor memory.

Memory gate: <1 GiB lifetime RSS for pool<=8, <64 MiB traced peak and bounded
active games/one pending leaf per game. No queues/threads/process workers.
Throughput gate >=1.3x matched complete-game throughput; retain FAIL if missed.
Independent review before finalization. Source snapshots/hashes, official 150
file inventory before/after, raw results and failed attempts retained.
