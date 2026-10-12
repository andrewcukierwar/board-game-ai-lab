# Independent review of the Phase 4D.4H-A ResNet implementation

Reviewer: the Phase 4D.4H-B session, which did not write
`games/connect4/alphazero_v2/resnet.py` or its tests. Phase 4D.4H-A reported
that no independent review had been commissioned; this is that review. It was
completed before any benchmark was measured. It is a review on the same host
by a different session, not an external replication.

Scope: `resnet.py` (285 lines), `experimental_models.py`, and the unchanged
`network.py`, `training.py` and `experimental.py` paths the ResNets inherit.
Method: reading every line, then `validate.py`, which checks behaviour
numerically without reusing the implementation. Evidence is in
`results/validation.json` (79 checks, all passed, 56 s).

## Verdict

**No material correctness finding.** Nothing had to be fixed, so no file under
`games/` was changed and the execution-source identity is the one Phase 4D.4H-A
published (`3523ee04…da934`). Six observations are recorded below; none blocks
the benchmark, and three matter for interpreting it.

## What was checked independently

| Question | Check | Result |
|---|---|---|
| Does the network compute what the specification says? | A float64 reference built from the state dict alone: convolution by explicit patch extraction and one matrix product, batch norm from its definition, the residual sum, both heads. It shares no code with `resnet.py` and does not call `conv2d` or `BatchNorm2d`. Compared on 128 random boards. | Largest difference 3.5e-7 at initialization and 6.8e-7 after six training updates (both variants). In train mode against batch statistics: 1.3e-5. |
| Are running statistics updated correctly? | Forward hooks capture each batch-norm input during six real `trainer.step` calls; the exponential average (momentum 0.1, unbiased variance) is recomputed in float64 and compared with the module buffers. | Largest difference 2.5e-7. Every update counter equals the trainer step count. |
| Does inference use running statistics and leave them alone? | Eval output matches the running-statistics reference; train-mode output differs from it; the weight-and-statistics hash is unchanged by inference. | Pass. |
| Is each row independent of its batch in eval mode? | Eight rows scored alone and inside a batch of 128. | Agree to 7.3e-7 (see observation 4). |
| Does depth give whole-board context? | A change at one corner cell reaches the opposite corner of the trunk output. | Pass for both variants. |
| Parameter counts, layer counts, defaults | 322,440 and 470,408; 9 and 13 batch norms with eps 1e-5, momentum 0.1; every convolution bias-free and followed by a batch norm. | Pass. |
| Checkpoint contracts | Inference artifacts round-trip bit-identically and refuse to overwrite; all six ordered cross-architecture artifact loads and all six state restores are rejected. | Pass. |
| Fresh-process resume | Generation 2 computed in-process and in a brand-new interpreter from the saved generation-1 boundary (8 games, 16 simulations, 8 updates of 32), for both ResNets and the CNN. Model, optimizer, replay, counters, three RNG streams and all batch-norm buffers compared. | Bit-identical for all three. This closes item 4 of Phase 4D.4H-A's remaining requirements. |
| Optimizer compatibility | AdamW with two groups; every setting except parameter membership identical across the three architectures; a restore after three of five updates continues bit-identically; each architecture rejects the other two optimizer states. | Pass. |
| CNN backward compatibility | The official generation-20 inference artifacts of both seeds load through the current CNN loader and match the boundary weights; the strict historical reader's output forks through the current CNN factory with matching weight hash, replay digest and step count. | Pass, read-only. |
| Historical isolation | Both ResNets refuse the historical CNN weights and refuse to fork the historical boundary; experimental output is refused inside the campaign directory; no file under `games/`, `api/`, `ui/` or `models/` differs from the Phase 4D.4H-A commit; 150 official files and the Phase 4D.4G output inventory are unchanged. | Pass. |

Existing suites on this worktree: 73 focused ResNet cases and 56 experimental
contract cases passed before any new code was added.

## Observations

1. **Weight decay means something different under batch norm.** The ResNet
   trainer decays convolution weights and never batch-norm scales or biases,
   which is the conventional choice and what Phase 4D.4H-A documented. But every
   ResNet convolution feeds a batch norm, so its output is invariant to the
   scale of those weights: decay on them changes the effective learning rate
   over time and does not shrink the function. In the CNN the same decay acts on
   the function directly. The optimizer settings are identical as required, yet
   their effect is not. This is a property of the comparison, not a defect.

2. **Train and eval mode are different networks early in training.** Running
   statistics start at mean 0 and variance 1 and move by 10% per update. A model
   trained for only a few updates is evaluated with statistics that lag its
   weights. In the 20-update smoke run the 5x64's eval-mode policy loss on
   training positions was 0.25 nats worse than its train-mode loss and 78% of
   its choices differed between modes. Self-play always uses eval mode, so the
   first generations of a from-scratch ResNet run would search with a lagging
   network. The benchmark measures how fast this gap closes.

3. **An untrained ResNet is not normalized at inference.** With initial running
   statistics the eval-mode batch norm is the identity, so the generation-1
   collection network is an un-normalized residual stack. It is finite and
   near-uniform, as Phase 4D.4H-A said, but the collection benchmark's game
   lengths for untrained networks say little about trained ones.

4. **Batched and single predictions agree to about 1e-6, not bit for bit.**
   The CNN shows the same thing (1.8e-7 at initialization, against 3.3e-7 and
   2.4e-7 for the untrained ResNets on the same boards), so it is batch-size-dependent rounding in the
   convolution arithmetic and not an effect of batch norm. Existing tests
   assert 1e-5. A fixed pool composition is deterministic, which is what exact resume
   relies on, and the fresh-process check confirms it.

5. **Any exception inside a ResNet step poisons the trainer.**
   `ResNetTrainer.step` catches `BaseException`, including a keyboard
   interrupt, because the forward pass has already moved the running
   statistics. This is deliberate and correct; an interrupted fixed-data run
   therefore cannot be continued in place, only restarted.

6. **Zero padding and the single signed input plane.** An off-board cell is
   indistinguishable from an empty one at the first layer. The CNN shares this,
   so it does not bias the comparison, and it remains a separate question.

None of these required a change before proceeding.
