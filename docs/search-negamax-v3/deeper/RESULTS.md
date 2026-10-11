# Complete-decision results

Source-frozen baseline: `60b99b0d42145da149f433907585b809060c946d`.
Research-only variants; acceptance uses every fixed gate in DESIGN.md.

| Variant | Decision | Geometric | Broad | Wall sum/A | CPU sum/A | Retained broad/A |
| --- | --- | --- | --- | --- | --- | --- |
| terminal-proof | eligible | 1.110× | 1.138× | 0.878 | 0.878 | 1.000 |

Seven rotated paired warmed samples; two separate traced memory decisions.
Geometric ratios use non-tail conditions. Broad: D8/D10 and ≥2,000 baseline nodes.
Single laptop; finite workload and host activity limit generalization. No strength inference.

| Position | Depth | Variant | ms | Speedup | Nodes | Leaves | Hits | Entries | Retained MiB | Peak MiB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| empty | 10 | direct | 218.995 | 1.000× | 128308 | 68511 | 15828 | 38395 | 3.624 | 3.759 |
| empty | 10 | terminal-proof | 193.262 | 1.133× | 128308 | 68511 | 15828 | 38395 | 3.624 | 3.759 |
| empty | 12 | direct | 672.348 | 1.000× | 380468 | 182826 | 56562 | 118065 | 12.354 | 13.252 |
| empty | 12 | terminal-proof | 591.682 | 1.136× | 380468 | 182826 | 56562 | 118065 | 12.354 | 13.252 |
| near-opening | 10 | direct | 176.315 | 1.000× | 99441 | 49039 | 14340 | 28999 | 3.094 | 3.285 |
| near-opening | 10 | terminal-proof | 155.633 | 1.133× | 99441 | 49039 | 14340 | 28999 | 3.094 | 3.285 |
| near-opening | 12 | direct | 630.324 | 1.000× | 340974 | 153874 | 53844 | 103764 | 11.610 | 13.393 |
| near-opening | 12 | terminal-proof | 548.005 | 1.150× | 340974 | 153874 | 53844 | 103764 | 11.610 | 13.393 |
| seeded-04 | 10 | direct | 454.313 | 1.000× | 253274 | 130691 | 29767 | 78484 | 7.415 | 7.692 |
| seeded-04 | 10 | terminal-proof | 395.805 | 1.148× | 253274 | 130691 | 29767 | 78484 | 7.415 | 7.692 |
| seeded-04 | 12 | direct | 2004.718 | 1.000× | 1089090 | 513050 | 140539 | 350545 | 42.061 | 53.239 |
| seeded-04 | 12 | terminal-proof | 1765.821 | 1.135× | 1089090 | 513050 | 140539 | 350545 | 42.061 | 53.239 |
| post-hoc-tail | 10 | direct | 746.167 | 1.000× | 423857 | 236149 | 47944 | 119958 | 12.574 | 13.334 |
| post-hoc-tail | 10 | terminal-proof | 655.979 | 1.137× | 423857 | 236149 | 47944 | 119958 | 12.574 | 13.334 |
| post-hoc-tail | 12 | direct | 3319.943 | 1.000× | 1820265 | 922517 | 237803 | 545595 | 54.523 | 56.487 |
| post-hoc-tail | 12 | terminal-proof | 2907.495 | 1.142× | 1820265 | 922517 | 237803 | 545595 | 54.523 | 56.486 |
| dense-endgame | 10 | direct | 0.024 | 1.000× | 4 | 0 | 0 | 3 | 0.001 | 0.005 |
| dense-endgame | 10 | terminal-proof | 0.023 | 1.032× | 4 | 0 | 0 | 3 | 0.001 | 0.005 |
| dense-endgame | 12 | direct | 0.022 | 1.000× | 4 | 0 | 0 | 3 | 0.001 | 0.005 |
| dense-endgame | 12 | terminal-proof | 0.021 | 1.048× | 4 | 0 | 0 | 3 | 0.001 | 0.005 |
| heldout-03 | 10 | direct | 0.155 | 1.000× | 65 | 1 | 1 | 32 | 0.004 | 0.009 |
| heldout-03 | 10 | terminal-proof | 0.142 | 1.087× | 65 | 1 | 1 | 32 | 0.004 | 0.009 |
| heldout-03 | 12 | direct | 0.162 | 1.000× | 67 | 1 | 1 | 34 | 0.004 | 0.009 |
| heldout-03 | 12 | terminal-proof | 0.147 | 1.103× | 67 | 1 | 1 | 34 | 0.004 | 0.009 |

## Fixed-gate failures and individual regressions

- terminal-proof: gates `[]`; latency regressions `[]`; memory `[]`.
