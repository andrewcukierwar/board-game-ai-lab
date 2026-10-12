# Phase 4D.4H-C — controlled CNN vs ResNet 3x64 self-play experiment

**Status: prospective design frozen; the experiment has not been run.** This
document is replaced by the results report when the run and its analysis are
complete.

## Question

Can ResNet 3x64 produce a stronger AlphaZero agent than the existing CNN when
both get approximately equal computation for self-play and optimization?

## Design in brief

The full protocol is
[research/alphazero_resnet_selfplay/DESIGN.md](../research/alphazero_resnet_selfplay/DESIGN.md).

- Worktree `experiment-output/phase4d4h-c-worktree`, branch
  `phase4d4h-c-resnet-selfplay`, from `6ef9148503a6e268cdcc2a51d9621c00e8e3a92e`.
- Four arms from scratch: CNN and ResNet 3x64, seeds 42 and 314159, empty
  replay and zero optimizer steps. No historical weights are imported.
- Six generations per arm. CNN: 144 games and 144 updates per generation.
  ResNet: 96 and 96. Original eight-ply schedule, 256 simulations, pool-eight
  self-play, unchanged replay, losses and AdamW settings.
- Expected computation per seed: 853 s (CNN) and 864 s (ResNet), ratio 1.01,
  from a calibration with trained weights.
- Evaluation at generations 0, 2, 4 and 6 on nonsealed development fixtures:
  raw policy, raw PUCT at 256 simulations, and 80 match games per checkpoint
  against reference negamax at depth 3.
- Primary decision: match-score difference at generation 6, with PASS, FAIL
  and INCONCLUSIVE rules fixed in advance. Tactical reliability is an
  independent gate. Exact-value learning is supporting.

## Changes to the proposed design, made before the freeze

| Proposed | Frozen | Reason |
|---|---|---|
| 192 / 128 games and updates per generation | 144 / 96 | The proposed size projects to about 75 minutes of training alone and does not fit the 120-minute limit with evaluation and calibration |
| Evaluation budget not specified | 256 simulations | Time limit; equals the self-play budget |
| Sequential execution implied | Training sequential; evaluation in three parallel workers | Evaluation timing is not compared and its results do not depend on scheduling |

## Evidence available before the run

| File | Content |
|---|---|
| `results/calibration.json` | Measured self-play, update and evaluation cost with trained weights; opponent saturation probe |
| `results/smoke.json` | Tiny disposable pipeline test (not results) |
| `results/validation.json` | Model-factory and boundary compatibility; fresh-process resume for both architectures |
| `results/design.json` | Frozen budgets, seeds, thresholds and feasibility projection |
| `results/freeze.json` | Frozen source hashes, runtime identity and generation-zero boundary identities |

Nothing is merged, promoted or deployed by this phase.
