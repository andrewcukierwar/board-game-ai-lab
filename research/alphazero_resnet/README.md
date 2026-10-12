# Phase 4D.4H-A — AlphaZero ResNet architecture records

The report is `docs/phase4d4h-alphazero-resnet.md`. This directory holds its
supporting records. Nothing here is a benchmark or a training result.

| File | Content |
|---|---|
| `describe.py` | Regenerates the static specification; constructs models only |
| `results/architectures.json` | Identities, specifications, parameter and multiply-accumulate counts for the CNN and both ResNets |
| `external_sources.json` | Pinned commits and SHA-256s of the external network definitions that were read |
| `results/focused-tests.txt` | The 73 focused ResNet cases |
| `results/regression-tests.txt` | Existing experimental contract cases, source-closure checks and the wrapper |
| `results/full-regression.txt` | Complete AlphaZero sweep and dependency suite, run after Phase 4D.4G completed |
| `results/mutation-checks.txt` | Eleven injected defects, all detected |

Regenerate the specification with one thread:

```
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  python -B -m research.alphazero_resnet.describe > research/alphazero_resnet/results/architectures.json
```

Run the focused cases the same way with `python -B -m pytest -q tests/alphazero_resnet_cases.py`,
or through the isolating wrapper `tests/test_alphazero_resnet.py`.
