# Phase 4D.4I-A — AlphaZero value-head and PUCT causal diagnostics

**Learned values actively cause some one-ply search errors, while also providing useful playing-strength guidance.** Zeroing values improves tactical accuracy in all four frozen checkpoints, principally mandatory safe responses. It lowers match-score point estimates in all four; the CNN decline is substantial and its paired interval excludes zero. Neither zeroing nor halving values consistently improves deeper exact decisions.

**Keep the existing CNN baseline for now.** Half values improve tactics in both architectures. They improve ResNet match-score estimates on both seeds, but the declared pooled interval touches zero and an independent bootstrap stream changes that boundary. CNN match-score estimates fall on seed42 and remain flat on seed314159. These development results do not justify a universal scale change, architecture replacement or promotion.

Recommend one controlled **CNN value-head regularization ablation**, described below. No training was implemented or launched.

## Execution, prospective design and scope

The study completed all48 tasks,15072 fixture searches and1600 matches without retries, failures or abandoned games. The independent clean rerun reproduced2400 additional A/B/E tactical searches exactly. No neural training or new self-play was collected. All checkpoints were accessed read-only.

The scientific design was committed as `c993d87c92215e39f1144cdcc0a0f84d1c3480da`; clean incident isolation and supplemental resource controls were committed as `83c365777f79a4883c70ac8cfcaca1a76755d160`. Both were pushed before launch. The manager recorded the latter as its clean pushed prospective HEAD. Worktree: `experiment-output/phase4d4i-a-worktree`; branch: `phase4d4i-a-alphazero-value-diagnostics`; starting commit: `3b563e1753ad93a1315b7d0f88a7cc45e1ee7359`.

Protocol: [DESIGN.md](../research/alphazero_value_diagnostics/DESIGN.md). Original55 execution/design hashes and runtime: [freeze.json](../research/alphazero_value_diagnostics/results/freeze.json). Supplemental3 source hashes and clean launch gate: [integrity-resolution.json](../research/alphazero_value_diagnostics/results/integrity-resolution.json). The original scientific sources/design were not amended after freeze. The final reporting supplement only aggregates already-measured fixed endpoints; it makes no new model query or membership choice.

The nonsealed H-C/G fixture identity is `80f6851838b18172d6f818ebd306ce0cd36e4796cac1e3a71df29694feb3ca2c`. The development-opening identity is `3990bc230084bd5c9293c15a44eae05f35749d2a3a18e66566461dae98f82f72`. All40 base families were played with both colors against unchanged reference negamax depth3,80 games per checkpoint/variant. Evaluation seeds4404901/4404902 and H-C namespaces are independent of variant, architecture and training seed. Initial random streams are paired; divergent trajectories may consume later draws differently.

| Variant | Neural policy | Nonterminal neural value |
| --- | --- | --- |
| A | Learned | Learned, unchanged baseline |
| B | Learned | Zero |
| C | Uniform on legal actions | Learned |
| D | Uniform on legal actions | Zero |
| E | Learned | Learned multiplied by0.5 |

All use unchanged bitboard PUCT256, c=1.41, identical legal masking, no root noise, tactical guard or proof propagation, and greedy maximum-visit selection with seeded ties. Wrappers run base inference then transform outputs; they do not change search. Terminal outcomes bypass neural inference and remain exact. Node values belong to the player to move and alternate signs on backup. Root neural value is diagnostic only, never backed up. Head-isolation assertions and single/batch behavior passed24 focused check groups; the clean reviewer separately checked both actors' terminal wins, draws and synthetic search behavior.

Four generation0 A-only position evaluations serve as negative controls. All score100% on tactical200; their solved300 accuracies range70.33–76.00%, and deep12861.72–70.31%. No generation0 matches were added. Historical generation20 checkpoints were prospectively omitted to preserve the primary full-factorial/match budget, rather than relaxing a loader.

## Uncertainty and denominators

Primary estimates use10000 percentile bootstrap resamples with analysis seed4410903. Entire fixed ancestry/reflection clusters are resampled together (tactical71, solved102, deep28); means are weighted by actual row multiplicities. Match pairs remain together within40 opening families. Within each architecture, the two fixed seeds are averaged on shared row IDs before common-family resampling. Intervals condition on these frozen networks and reused development families: they exclude training-run/seed uncertainty and do not validate unseen generalization. There is no multiplicity correction or promotion test.

Tactical200 contains100 unique immediate wins and100 unique safe responses. Solved300 has100 theoretical wins,100 draws and100 losses. Deep128 has70 wins,58 draws and no losses. Exact mistakes are any achievable WDL decrease, including win-to-draw. On a theoretically lost solved row every legal action is WDL-optimal, so its automatic100% WDL-preserving accuracy is not evidence of good action quality. No unresolved/tactical action was assigned a deeper oracle value.

All following differences are intervention minus A; bracketed intervals are95% paired intervals. Percentage-point differences are labeledpp. Complete unrounded results, calibration bins, strata and per-position transitions are in [analysis.json](../research/alphazero_value_diagnostics/results/analysis.json) and [report-supplement.json](../research/alphazero_value_diagnostics/results/report-supplement.json).

## One-ply tactics and attribution

| Checkpoint | Variant | Immediate win | Safe response | Avoidable one-ply errors /200 | Missed wins /200 |
| --- | --- | --- | --- | --- | --- |
| 42-cnn | A | 97.00% | 90.00% | 10 | 3 |
| 42-cnn | B | 99.00% | 96.00% | 4 | 1 |
| 42-cnn | C | 100.00% | 91.00% | 9 | 0 |
| 42-cnn | D | 100.00% | 100.00% | 0 | 0 |
| 42-cnn | E | 100.00% | 96.00% | 4 | 0 |
| 42-resnet3x64 | A | 97.00% | 90.00% | 10 | 3 |
| 42-resnet3x64 | B | 98.00% | 97.00% | 3 | 2 |
| 42-resnet3x64 | C | 100.00% | 92.00% | 8 | 0 |
| 42-resnet3x64 | D | 100.00% | 100.00% | 0 | 0 |
| 42-resnet3x64 | E | 98.00% | 96.00% | 4 | 2 |
| 314159-cnn | A | 99.00% | 87.00% | 13 | 1 |
| 314159-cnn | B | 99.00% | 98.00% | 2 | 1 |
| 314159-cnn | C | 100.00% | 88.00% | 12 | 0 |
| 314159-cnn | D | 100.00% | 100.00% | 0 | 0 |
| 314159-cnn | E | 99.00% | 93.00% | 7 | 1 |
| 314159-resnet3x64 | A | 96.00% | 93.00% | 7 | 4 |
| 314159-resnet3x64 | B | 98.00% | 98.00% | 2 | 2 |
| 314159-resnet3x64 | C | 100.00% | 97.00% | 3 | 0 |
| 314159-resnet3x64 | D | 100.00% | 100.00% | 0 | 0 |
| 314159-resnet3x64 | E | 98.00% | 95.00% | 5 | 2 |

Both zero and half values improve overall tactical accuracy in all four checkpoints. Value removal has its clearest benefit on mandatory safe responses. Uniform priors with learned values recover every immediate win across both architectures/seeds, while B retains some missed wins. Thus learned priors contribute to missed terminal alternatives; values contribute more strongly to the safe-response deficits. Uniform-prior safe-response gains alone have broad paired intervals and can introduce new errors. Neither head is the sole cause.

Pooled category contrasts, two fixed seeds per architecture:

| Architecture | Category | Baseline | B−A, pp | C−A, pp | E−A, pp |
| --- | --- | --- | --- | --- | --- |
| cnn | safe response | 88.50% | +8.50 [+3.45, +14.80] | +1.00 [-5.73, +7.33] | +6.00 [+2.40, +10.42] |
| cnn | immediate win | 98.00% | +1.00 [-1.11, +3.47] | +2.00 [+0.00, +4.50] | +1.50 [+0.00, +3.80] |
| resnet3x64 | safe response | 91.50% | +6.00 [+2.45, +10.33] | +3.00 [-3.92, +10.19] | +4.00 [+1.42, +7.39] |
| resnet3x64 | immediate win | 96.50% | +1.50 [+0.00, +4.08] | +3.50 [+0.00, +7.74] | +1.50 [+0.00, +4.08] |

D solves every one-ply fixture. D searches/actions and all80 match histories are identical across checkpoints because both neural outputs are neutralized; these are repeated controls, not four independent neutral-network replications. The untrained A controls also solve every one-ply fixture. This shows the fixed budget is sufficient for these fixtures under neutral guidance, not that exploration/search-budget effects are ruled out elsewhere: only c=1.41 and256 simulations were tested.

The factorial interaction is A−B−C+D. On tactical accuracy its interval includes zero in both architectures. The errors require distinguishing head effects and individual position interactions, rather than attributing all mistakes to values.

## Exact decisions and action changes

| Checkpoint | Variant | Solved300 optimal | Deep128 optimal | Solved Δ, pp | Deep Δ, pp |
| --- | --- | --- | --- | --- | --- |
| 42-cnn | A | 76.33% | 76.56% | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 42-cnn | B | 74.33% | 66.41% | -2.00 [-7.35, +2.45] | -10.16 [-20.18, +0.00] |
| 42-cnn | C | 75.33% | 75.00% | -1.00 [-3.40, +1.82] | -1.56 [-9.42, +5.97] |
| 42-cnn | D | 74.00% | 66.41% | -2.33 [-7.84, +3.37] | -10.16 [-24.07, +3.57] |
| 42-cnn | E | 76.00% | 72.66% | -0.33 [-2.72, +2.40] | -3.91 [-8.75, +0.85] |
| 42-resnet3x64 | A | 76.00% | 67.97% | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 42-resnet3x64 | B | 76.33% | 68.75% | +0.33 [-4.90, +5.18] | +0.78 [-6.35, +9.80] |
| 42-resnet3x64 | C | 75.00% | 67.19% | -1.00 [-4.91, +2.80] | -0.78 [-6.43, +5.77] |
| 42-resnet3x64 | D | 74.00% | 66.41% | -2.00 [-6.99, +3.25] | -1.56 [-13.39, +12.07] |
| 42-resnet3x64 | E | 78.00% | 68.75% | +2.00 [-0.84, +5.52] | +0.78 [-2.99, +5.39] |
| 314159-cnn | A | 77.67% | 75.00% | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 314159-cnn | B | 77.67% | 76.56% | +0.00 [-5.21, +4.41] | +1.56 [-5.08, +10.00] |
| 314159-cnn | C | 78.00% | 72.66% | +0.33 [-3.26, +3.75] | -2.34 [-7.63, +1.85] |
| 314159-cnn | D | 74.00% | 66.41% | -3.67 [-9.06, +1.75] | -8.59 [-23.24, +6.86] |
| 314159-cnn | E | 78.67% | 78.91% | +1.00 [-1.32, +3.48] | +3.91 [+0.76, +8.49] |
| 314159-resnet3x64 | A | 76.67% | 67.19% | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| 314159-resnet3x64 | B | 76.67% | 74.22% | +0.00 [-4.62, +4.42] | +7.03 [-1.39, +16.94] |
| 314159-resnet3x64 | C | 75.67% | 64.84% | -1.00 [-4.40, +2.43] | -2.34 [-8.70, +3.64] |
| 314159-resnet3x64 | D | 74.00% | 66.41% | -2.67 [-8.38, +3.17] | -0.78 [-12.50, +10.66] |
| 314159-resnet3x64 | E | 76.33% | 67.97% | -0.33 [-3.13, +2.81] | +0.78 [-3.95, +6.48] |

No pooled exact-decision improvement excludes zero. B on deep128 decreases CNN accuracy by4.30pp [−10.42,+2.83] and increases ResNet accuracy by3.91pp [−2.70,+12.28]. CNN seed42 loses13 correct decisions while seed314159 gains2; ResNet gains1 and9. E's deep changes are−5,+5 decisions for CNN and+1,+1 for ResNet. Their pooled effects are0.00pp and+0.78pp, both uncertain. Tactical gains do not establish improved game-theoretic choices.

Selected exact WDL downgrades for the prioritized comparisons; columns retain the theoretical-class denominators:

| Checkpoint | Suite | Variant | Theoretical wins | W→D | W→L | Theoretical draws | D→L |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42-cnn | solved | A | 100 | 7 | 22 | 100 | 42 |
| 42-cnn | solved | B | 100 | 9 | 22 | 100 | 46 |
| 42-cnn | solved | E | 100 | 9 | 21 | 100 | 42 |
| 42-cnn | deep | A | 70 | 7 | 10 | 58 | 13 |
| 42-cnn | deep | B | 70 | 9 | 17 | 58 | 17 |
| 42-cnn | deep | E | 70 | 8 | 12 | 58 | 15 |
| 42-resnet3x64 | solved | A | 100 | 7 | 19 | 100 | 46 |
| 42-resnet3x64 | solved | B | 100 | 10 | 24 | 100 | 37 |
| 42-resnet3x64 | solved | E | 100 | 9 | 18 | 100 | 39 |
| 42-resnet3x64 | deep | A | 70 | 7 | 19 | 58 | 15 |
| 42-resnet3x64 | deep | B | 70 | 8 | 16 | 58 | 16 |
| 42-resnet3x64 | deep | E | 70 | 8 | 18 | 58 | 14 |
| 314159-cnn | solved | A | 100 | 4 | 20 | 100 | 43 |
| 314159-cnn | solved | B | 100 | 9 | 23 | 100 | 35 |
| 314159-cnn | solved | E | 100 | 6 | 17 | 100 | 41 |
| 314159-cnn | deep | A | 70 | 9 | 11 | 58 | 12 |
| 314159-cnn | deep | B | 70 | 7 | 8 | 58 | 15 |
| 314159-cnn | deep | E | 70 | 6 | 10 | 58 | 11 |
| 314159-resnet3x64 | solved | A | 100 | 9 | 19 | 100 | 42 |
| 314159-resnet3x64 | solved | B | 100 | 7 | 18 | 100 | 45 |
| 314159-resnet3x64 | solved | E | 100 | 8 | 20 | 100 | 43 |
| 314159-resnet3x64 | deep | A | 70 | 2 | 17 | 58 | 23 |
| 314159-resnet3x64 | deep | B | 70 | 6 | 12 | 58 | 15 |
| 314159-resnet3x64 | deep | E | 70 | 2 | 16 | 58 | 23 |

All A–E WDL transition vectors, including uniform controls, are retained in report-supplement.json. Accuracy and mistake rate are complements on complete exact vectors. Deep128's absence of theoretical losses changes both value-MSE interpretation and available downgrade classes.

Fixed baseline mistakes / newly introduced mistakes / changed actions, each suite's complete rows:

| Checkpoint | Intervention | Tactical fixed/new/changed | Solved fixed/new/changed | Deep fixed/new/changed |
| --- | --- | --- | --- | --- |
| 42-cnn | B | 9/1/11 | 16/22/125 | 9/22/56 |
| 42-cnn | C | 10/6/18 | 5/8/67 | 8/10/31 |
| 42-cnn | D | 13/0/13 | 21/28/188 | 12/25/77 |
| 42-cnn | E | 9/0/9 | 8/9/53 | 2/7/19 |
| 42-resnet3x64 | B | 8/0/10 | 24/23/142 | 14/13/53 |
| 42-resnet3x64 | C | 10/5/15 | 12/15/99 | 6/7/27 |
| 42-resnet3x64 | D | 13/0/13 | 26/32/184 | 16/18/78 |
| 42-resnet3x64 | E | 7/0/10 | 14/8/64 | 6/5/25 |
| 314159-cnn | B | 11/0/13 | 25/25/139 | 10/8/41 |
| 314159-cnn | C | 7/5/17 | 13/12/73 | 2/5/18 |
| 314159-cnn | D | 14/0/14 | 21/32/193 | 14/25/81 |
| 314159-cnn | E | 6/0/11 | 10/7/58 | 5/0/12 |
| 314159-resnet3x64 | B | 7/0/8 | 21/21/145 | 24/15/75 |
| 314159-resnet3x64 | C | 11/3/14 | 8/11/87 | 9/12/40 |
| 314159-resnet3x64 | D | 11/0/11 | 24/32/181 | 17/18/85 |
| 314159-resnet3x64 | E | 4/0/6 | 5/6/63 | 7/6/30 |

B fixes35 tactical mistakes and introduces1 across800 checkpoint-position decisions. E fixes26 and introduces0; C fixes38 but introduces19. These totals count checkpoint-position decisions, not independent situations. On solved300 and deep128, fixes coexist with many new mistakes; no overall deeper improvement follows from the tactical totals.

Examples from nonsealed development fixtures (zero-based physical columns; ✓ means the chosen action meets that fixture's label):

| Checkpoint | Development row | Sensitivity at A | A | B | C | D | E |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 314159-cnn | `development-unique_safe-o-004-mirror` | value_only_at_A | 5 × | 3 ✓ | 5 × | 3 ✓ | 3 ✓ |
| 314159-cnn | `development-unique_win-o-011-mirror` | prior_only_at_A | 6 × | 6 × | 2 ✓ | 2 ✓ | 6 × |
| 314159-cnn | `development-unique_safe-o-011-base` | both | 5 × | 0 ✓ | 0 ✓ | 0 ✓ | 0 ✓ |

The full ranked position list includes every action-sensitive row, flags separate B/C changes at the unchanged baseline, and retains all A–E actions/correctness. Those flags are local intervention sensitivities, not immutable causal categories. High root magnitude is descriptive: the root value is not backed up, and oracle values for unlabelled internal leaves are unavailable.

## Playing strength

| Checkpoint | Variant | W/D/L | Score | Δ from A, pp [95%] |
| --- | --- | --- | --- | --- |
| 42-cnn | A | 44/4/32 | 57.50% | +0.00 [+0.00, +0.00] |
| 42-cnn | B | 18/10/52 | 28.75% | -28.75 [-43.12, -13.75] |
| 42-cnn | C | 38/8/34 | 52.50% | -5.00 [-17.50, +7.50] |
| 42-cnn | D | 9/1/70 | 11.88% | -45.62 [-58.75, -31.87] |
| 42-cnn | E | 35/9/36 | 49.38% | -8.12 [-17.50, +1.25] |
| 42-resnet3x64 | A | 22/5/53 | 30.63% | +0.00 [+0.00, +0.00] |
| 42-resnet3x64 | B | 18/8/54 | 27.50% | -3.12 [-13.75, +7.50] |
| 42-resnet3x64 | C | 26/2/52 | 33.75% | +3.12 [-8.12, +14.37] |
| 42-resnet3x64 | D | 9/1/70 | 11.88% | -18.75 [-27.50, -9.38] |
| 42-resnet3x64 | E | 28/5/47 | 38.12% | +7.50 [-2.50, +16.88] |
| 314159-cnn | A | 35/4/41 | 46.25% | +0.00 [+0.00, +0.00] |
| 314159-cnn | B | 22/12/46 | 35.00% | -11.25 [-21.25, -1.25] |
| 314159-cnn | C | 28/2/50 | 36.25% | -10.00 [-20.62, +0.62] |
| 314159-cnn | D | 9/1/70 | 11.88% | -34.38 [-46.25, -21.88] |
| 314159-cnn | E | 34/6/40 | 46.25% | +0.00 [-10.00, +10.00] |
| 314159-resnet3x64 | A | 32/2/46 | 41.25% | +0.00 [+0.00, +0.00] |
| 314159-resnet3x64 | B | 22/6/52 | 31.25% | -10.00 [-23.75, +3.75] |
| 314159-resnet3x64 | C | 29/2/49 | 37.50% | -3.75 [-18.12, +10.62] |
| 314159-resnet3x64 | D | 9/1/70 | 11.88% | -29.38 [-41.25, -17.50] |
| 314159-resnet3x64 | E | 37/5/38 | 49.38% | +8.12 [-3.75, +20.00] |

CNN B−A is−20.00pp [−29.38,−10.31], with both seeds negative (−28.75,−11.25pp). ResNet B−A is−6.56pp [−16.56,+3.44], also negative on both seeds (−3.13,−10.00pp), with uncertainty spanning zero. The conclusion is strong for these CNNs and descriptive/inconclusive for ResNet, not proof of universal harm from zero values.

D scores11.875% (9 wins,1 draw,70 losses) despite100% tactical accuracy. With uniform priors, retained learned values improve match score over zero by32.50pp for CNN and23.75pp for ResNet; both intervals exclude zero. With zero values, retained learned priors improve score over uniform by20.00pp and17.50pp; both intervals exclude zero. Both heads contain useful match guidance even though they can misdirect short tactics.

For match score the interaction A−B−C+D is−12.50pp [−23.44,−2.19] for CNN and−17.19pp [−28.12,−5.94] for ResNet. On this score scale the joint benefit is subadditive; this is an output-intervention effect in fixed search, not an explanation of how learning produced it or evidence the combined baseline is inferior to either isolated head.

E−A is−4.06pp [−10.94,+2.50] for CNN and+7.81pp [0.00,+15.63] for ResNet. ResNet's positive point effects occur on both seeds, but each seed's interval spans zero. The primary pooled interval touches zero; the independent bootstrap stream yields [+0.31,+15.94]pp. That boundary is sensitive to Monte Carlo resampling and cannot support a confirmed strength gain. No scale sweep or result-dependent match selection was performed.

Color results, preserving the agent's actual X/O ownership rather than assuming the first post-opening move:

| Checkpoint | Variant | Color | Score | Δ from A, pp [95%] |
| --- | --- | --- | --- | --- |
| 42-cnn | A | X | 56.25% | +0.00 [+0.00, +0.00] |
| 42-cnn | A | O | 58.75% | +0.00 [+0.00, +0.00] |
| 42-cnn | B | X | 32.50% | -23.75 [-43.75, -2.50] |
| 42-cnn | B | O | 25.00% | -33.75 [-48.75, -18.75] |
| 42-cnn | C | X | 55.00% | -1.25 [-21.25, +18.75] |
| 42-cnn | C | O | 50.00% | -8.75 [-23.75, +5.00] |
| 42-cnn | D | X | 18.75% | -37.50 [-57.50, -16.25] |
| 42-cnn | D | O | 5.00% | -53.75 [-70.00, -37.50] |
| 42-cnn | E | X | 48.75% | -7.50 [-22.50, +6.25] |
| 42-cnn | E | O | 50.00% | -8.75 [-22.50, +5.00] |
| 42-resnet3x64 | A | X | 36.25% | +0.00 [+0.00, +0.00] |
| 42-resnet3x64 | A | O | 25.00% | +0.00 [+0.00, +0.00] |
| 42-resnet3x64 | B | X | 35.00% | -1.25 [-17.53, +15.00] |
| 42-resnet3x64 | B | O | 20.00% | -5.00 [-20.00, +10.00] |
| 42-resnet3x64 | C | X | 33.75% | -2.50 [-18.75, +13.75] |
| 42-resnet3x64 | C | O | 33.75% | +8.75 [-10.00, +27.50] |
| 42-resnet3x64 | D | X | 18.75% | -17.50 [-33.75, -1.25] |
| 42-resnet3x64 | D | O | 5.00% | -20.00 [-33.75, -7.47] |
| 42-resnet3x64 | E | X | 47.50% | +11.25 [-5.00, +27.50] |
| 42-resnet3x64 | E | O | 28.75% | +3.75 [-11.25, +18.75] |
| 314159-cnn | A | X | 47.50% | +0.00 [+0.00, +0.00] |
| 314159-cnn | A | O | 45.00% | +0.00 [+0.00, +0.00] |
| 314159-cnn | B | X | 36.25% | -11.25 [-27.50, +5.00] |
| 314159-cnn | B | O | 33.75% | -11.25 [-25.00, +2.50] |
| 314159-cnn | C | X | 42.50% | -5.00 [-18.75, +8.75] |
| 314159-cnn | C | O | 30.00% | -15.00 [-30.00, +0.00] |
| 314159-cnn | D | X | 18.75% | -28.75 [-48.75, -8.75] |
| 314159-cnn | D | O | 5.00% | -40.00 [-57.50, -22.50] |
| 314159-cnn | E | X | 52.50% | +5.00 [-11.25, +21.25] |
| 314159-cnn | E | O | 40.00% | -5.00 [-18.75, +8.75] |
| 314159-resnet3x64 | A | X | 52.50% | +0.00 [+0.00, +0.00] |
| 314159-resnet3x64 | A | O | 30.00% | +0.00 [+0.00, +0.00] |
| 314159-resnet3x64 | B | X | 31.25% | -21.25 [-41.25, -1.25] |
| 314159-resnet3x64 | B | O | 31.25% | +1.25 [-17.50, +18.75] |
| 314159-resnet3x64 | C | X | 46.25% | -6.25 [-23.75, +11.25] |
| 314159-resnet3x64 | C | O | 28.75% | -1.25 [-18.75, +17.50] |
| 314159-resnet3x64 | D | X | 18.75% | -33.75 [-52.50, -16.22] |
| 314159-resnet3x64 | D | O | 5.00% | -25.00 [-40.00, -10.00] |
| 314159-resnet3x64 | E | X | 58.75% | +6.25 [-10.00, +22.50] |
| 314159-resnet3x64 | E | O | 40.00% | +10.00 [-6.25, +26.25] |

## Factorial contrasts across endpoints

| Architecture | Endpoint | B−A, pp | C−A, pp | E−A, pp | A−B−C+D, pp |
| --- | --- | --- | --- | --- | --- |
| cnn | tactical | +4.75 [+1.99, +8.26] | +1.50 [-2.22, +4.82] | +3.75 [+1.73, +6.25] | +0.50 [-2.94, +4.44] |
| cnn | solved | -1.00 [-5.44, +2.57] | -0.33 [-2.44, +1.95] | +0.33 [-1.33, +2.18] | -1.67 [-6.53, +3.92] |
| cnn | deep | -4.30 [-10.42, +2.83] | -1.95 [-7.69, +3.31] | +0.00 [-3.08, +3.30] | -3.12 [-14.58, +8.33] |
| cnn | matches | -20.00 [-29.38, -10.31] | -7.50 [-15.31, +0.00] | -4.06 [-10.94, +2.50] | -12.50 [-23.44, -2.19] |
| resnet3x64 | tactical | +3.75 [+1.64, +6.40] | +3.25 [-0.59, +7.30] | +2.75 [+1.10, +4.85] | -1.00 [-3.96, +2.16] |
| resnet3x64 | solved | +0.17 [-3.52, +3.51] | -1.00 [-3.57, +1.67] | +0.83 [-1.27, +3.15] | -1.50 [-5.94, +3.42] |
| resnet3x64 | deep | +3.91 [-2.70, +12.28] | -1.56 [-6.67, +3.93] | +0.78 [-2.41, +4.76] | -3.52 [-15.73, +8.19] |
| resnet3x64 | matches | -6.56 [-16.56, +3.44] | -0.31 [-8.44, +8.12] | +7.81 [+0.00, +15.62] | -17.19 [-28.12, -5.94] |

Full D−C and D−B contrasts remain in analysis.json. These identify effects of intervening on the frozen outputs under one search/opponent/budget; they do not measure training changes, causal reasons for high value MSE, or the optimal exploration constant.

## Value learning: separate targets and calibration

Training-replay predictions use full retained g6/g5 replay, not the earlier phase's2048-position probe. Prequential predictions use g5 on only new g6 games that it generated and had not trained on. No g7 exists, so g6 prequential MSE is unavailable. Realized self-play outcome and optimal game-theoretic WDL are different targets.

MSE [95% opening/ancestry interval]; zero columns give constant-zero MSE on the same data. Exact zero baselines are0.667 on solved300 and0.547 on deep128:

| Checkpoint | G6 training MSE | G6 zero | G5 training MSE | G5→G6 prequential MSE | Prequential zero | Solved exact MSE | Deep exact MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42-cnn | 0.550 [0.519, 0.582] | 0.795 | 0.559 [0.526, 0.593] | 0.701 [0.556, 0.821] | 0.818 | 0.647 [0.534, 0.765] | 0.580 [0.409, 0.801] |
| 42-resnet3x64 | 0.461 [0.431, 0.491] | 0.870 | 0.480 [0.453, 0.508] | 0.972 [0.855, 1.101] | 0.903 | 0.803 [0.672, 0.938] | 0.480 [0.334, 0.671] |
| 314159-cnn | 0.547 [0.507, 0.591] | 0.822 | 0.580 [0.540, 0.622] | 0.668 [0.487, 0.832] | 0.665 | 0.689 [0.546, 0.834] | 0.592 [0.407, 0.831] |
| 314159-resnet3x64 | 0.524 [0.498, 0.551] | 0.855 | 0.425 [0.399, 0.452] | 0.951 [0.804, 1.106] | 0.846 | 0.862 [0.717, 1.023] | 0.484 [0.253, 0.763] |

All four fit retained outcomes better than zero. CNN42 g5 prequential MSE beats zero: excess−0.117 [−0.228,−0.024]. CNN314159 is tied: excess+0.003 [−0.083,+0.067]. Both ResNet prequential point estimates exceed zero (+0.070,+0.105), but their intervals span zero. ResNet's larger training-to-next-generation gaps are consistent with poor out-of-replay generalization; they also include changes in game distribution/outcomes and are not a controlled demonstration of memorization.

On balanced solved300, CNN is approximately at zero's error, while ResNet has higher errors and a positive side-to-move bias. Pooled ResNet−CNN MSE is+0.165 [+0.050,+0.285]. On deep128, which has no lost positions, it is−0.104 [−0.314,+0.066]. The suite composition reverses the point contrast; it is not evidence of a uniformly superior value function.

Solved300 prediction strata (calibration is the ten-bin count-weighted absolute mean prediction/target gap in [−1,+1] units):

| Checkpoint | Loss MSE | Draw MSE | Win MSE | Mean value | Mean \|value\| | Share \|value\|>0.95 | Calibration gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42-cnn | 0.702 | 0.255 | 0.985 | -0.079 | 0.463 | 1.67% | 0.226 |
| 42-resnet3x64 | 1.303 | 0.413 | 0.694 | +0.181 | 0.561 | 4.33% | 0.376 |
| 314159-cnn | 0.919 | 0.202 | 0.945 | +0.017 | 0.454 | 1.00% | 0.289 |
| 314159-resnet3x64 | 1.715 | 0.372 | 0.499 | +0.334 | 0.556 | 8.33% | 0.436 |

ResNet's errors are concentrated in theoretical losses:1.303/1.715 versus zero's1.000, while its win errors0.694/0.499 beat zero's1.000. On exact solved rows, high-confidence (|v|>0.8) errors exceed lower-confidence errors in all four models. On deep128 that relationship reverses in three models, so confidence alone is not a universal error detector. Replay high-confidence predictions are substantially more accurate on training data; their held-out accuracy is model/distribution dependent. All fixed value bins and confidence/actor/ply/observed-terminal-distance strata are retained, including empty-bin denominators.

Training error is highest before ply8 (0.78–0.85 MSE) and generally lower later, but prequential curves are not uniformly monotonic. CNN prequential error on repeated mirror positions is higher than on nonrepeated positions; ResNet's direction differs. Novel-to-g5 prequential positions have CNN MSE0.743/0.860 versus seen0.684/0.544, while ResNet is1.037/0.926 versus seen0.952/0.957. These associations do not establish why these models show out-of-replay error. Terminal distance here means remaining plies in the observed imperfect game, not an oracle distance to victory. Per-opening descriptive errors are retained; unequal opening sizes and correlated histories limit inference.

## Outcome noise, mismatch and independent signs

| Checkpoint | G6 examples | Unique exact boards | Conflicting repeated boards | Examples on conflicting boards | Empirical same-board variance floor | Seen prequential MSE | Novel prequential MSE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42-cnn | 24198 | 18477 | 586 | 5224 | 0.156 | 0.684 | 0.743 |
| 42-resnet3x64 | 15539 | 12120 | 385 | 3228 | 0.169 | 0.952 | 1.037 |
| 314159-cnn | 25019 | 16774 | 593 | 5877 | 0.180 | 0.544 | 0.860 |
| 314159-resnet3x64 | 15620 | 12513 | 385 | 3038 | 0.161 | 0.957 | 0.926 |

The empirical floor is the minimum squared error on these recorded outcomes for a deterministic predictor assigning one value to each exact actor/board. It measures observed within-position outcome disagreement over changing self-play policies; it is not an irreducible Bayes-error estimate on future data. Repetition-error strata use mirror-canonical keys, whereas this floor uses exact board keys. Outcome averaging would not automatically change the expected MSE gradient for duplicated deterministic inputs, so this is not a recommendation to relabel duplicates without a controlled objective change.

Only CNN42 has exact fixture/replay overlap: four occurrences, one realized-versus-optimal disagreement. All other arms have zero overlap. This establishes one actual target disagreement, not widespread mismatch. Different imperfect continuations can produce different realized outcomes from the same board even when optimal WDL is fixed. Noise and target mismatch are plausible limitations; neither proves targets were implemented incorrectly.

Independent legal-history/actor/encoding/winner checks found zero label errors in160752 stored training/prequential occurrences. Every terminal sign and alternating backup remained correct. There is no evidence of an actor-relative sign/label implementation bug. Replay data here is not an independently sampled optimal-play training set.

## Answers and controlled follow-up

| Question | Answer |
| --- | --- |
| 1. Does zero value improve PUCT? | For one-ply tactics, yes on these four checkpoints; not as a general search-quality or strength claim. |
| 2. Does it improve one-ply tactics? | Yes, especially safe response (+8.5pp CNN,+6.0pp ResNet pooled). |
| 3. Does it improve deeper decisions? | Not consistently; paired pooled intervals span zero and seeds/suites differ. |
| 4. Does it improve actual matches? | No observed score improvement in any checkpoint; CNN decline is well supported conditionally, ResNet decline uncertain. |
| 5. Priors or values primarily responsible? | Values contribute most to safe-response deficits; priors account for missed-win alternatives. Both can create/fix individual errors. |
| 6. Does shrinkage help? | Tactics improve throughout. ResNet match estimates improve, CNN does not; no general deployment recommendation. |
| 7. Consistent across seeds? | Tactical benefits and zero-value match-score decreases share signs. Deeper and attenuation-strength effects do not uniformly agree. |
| 8. Same CNN/ResNet mechanism? | Shared short-tactic interference and useful strength guidance; ResNet has more side-to-move bias, confidence and out-of-replay weakness. |
| 9. Is the target noisy or mismatched? | Realized labels conflict on repeated boards; exact overlap is too sparse to assess widespread mismatch. No label/sign bug found. |
| 10. Most justified value-training change? | A single value-head regularization probe to improve out-of-replay usefulness while retaining strength, not zero-value training or universal inference shrinkage. |

**Recommend one controlled CNN ablation: value-head-only AdamW weight decay1e-3 versus the original1e-4.** Use a fixed-data comparison first: identical frozen replay examples and a prospectively frozen held-out game/family/novel-position split, with identical minibatch/reflection order and fixed update budget. Keep shared-trunk/policy decay1e-4, raw realized outcome targets, architecture, learning rate, clipping, initial boundaries and search/evaluation budgets matched. Do not collect new self-play in that offline comparison. Define value-head parameters explicitly (value_conv,value_fc1,value_fc2) and preserve strict new-phase checkpoint contracts. The factor10 is a prospectively chosen engineering probe, not an inferred optimum; do not add a decay/scale sweep.

The hypothesis is that targeted regularization improves prequential/novel outcome prediction and value-guided decisions while retaining the baseline's match guidance. These data motivate testing that hypothesis; they do not prove overfitting as the cause, nor that1e-3 will succeed. Full-network training can change shared-trunk gradients and hence learned priors even when only the head's decay hyperparameter changes. Retain head-intervention diagnostics to detect that interaction. Use prequential outcome MSE relative to zero and fixed-opponent match score as principal checks; report calibration, both seeds, exact class strata and tactics as safeguards. Freeze fresh nonsealed confirmation identities before the follow-up, rather than interpreting optimization on these reused fixtures as unbiased generalization. Do not open sealed data to design that study.

No part of this training ablation is implemented or launched here. Keep CNN A as the current research baseline; do not promote any checkpoint or adopt a universal zero/half/uniform-prior intervention from this diagnostic.

## Independent review, incident and preservation

The clean reviewer with no inherited exposed task history performed [prelaunch review](../research/alphazero_value_diagnostics/independent-clean-prelaunch-review.md), authored new independent arithmetic and bounded rerun code, and reviewed completed evidence in [independent-clean-results-review.md](../research/alphazero_value_diagnostics/independent-clean-results-review.md). Arithmetic used its own board/rules and multinomial family bootstrap, comparing640 means with maximum discrepancy0. The2400-search model rerun used its own scaler and matched all actions, visits, root priors and values; its independent replay audit matched160752 labels. All1600 stored match histories are legal and agree with recorded colors/results. This is same-host replication; C/D search endpoints, match decision generation and all exact oracle proofs were not independently rerun/re-solved.

The original reviewer had accidentally received partial sealed text during a broad search. The [metadata scope audit/isolation](../research/alphazero_value_diagnostics/integrity-resolution.md) treats three sealed files as within access scope and records149 visible references in a truncated response. The exposed reviewer was retired; its old review/script remain history and were not used for launch or results. Fixed inputs were independently reconstructed from the pre-incident request/starting commit/nonsealed contracts, and the clean launch passed. The breach and automatic transcript retention are not erased. Mental non-use cannot be proven absolutely; provenance and clean reconstruction support isolation. No sealed contents or raw exposure transcript were copied into the new research publication. This phase is not a sealed generalization evaluation.

The primary manager's historical inventories passed:150 official campaign files unchanged; G output151 and results132; H-B output78 and results37; H-C output450, results290, source84 and research307; historical games77. Original sources/checkpoints were not changed. The extra [final-integrity-audit.json](../research/alphazero_value_diagnostics/results/final-integrity-audit.json) verifies all28 checkpoint SHA256 identities, all55 frozen source/design hashes, supplemental3 source hashes and170 starting-commit tracked game/model/test/application paths, including11 actual frozen-package files.

One preserved reporting defect: the manager's `canonical-benchmarks` label pointed at nonexistent root benchmarks/ and inventoried0 files. That empty check is not evidence. Actual frozen packages were already present in the protected historical/H-C games trees; the supplemental actual-path Git-blob/current-byte audit confirms all11 against starting commit. Original inventories/status were not rewritten. No game, benchmark, model, API, UI or deployed source changed. No merge, deployment or promotion occurred.

## Resources and evidence

Run/controller: October12,2026,18:34:27–18:51:04 Eastern. Full primary phase864.38s; frozen analysis19.68s; fresh arithmetic22.72s; bounded independent model reproduction90.35s. Postlaunch lifecycle997.16s plus conservative prelaunch charge180s =1177.16s (19.62min), below3600s. The final reporting supplement is0.97s; final integrity audit was under1s. Report generation and final-review accounting are recorded separately in resource-final.json. Drafting/publication wall time is not neural measured work.

The prelaunch charge covers two focused validation passes, the19.38s cost-only pilot, synthetic checks and hashing/freezes. The full-comparison pilot projected886.52s, or1595.74s with contingency. The fixed target was an upper30–45min planning range; no work was added to consume spare budget. Main worker CPU sums to2422.01s, with three workers; summed worker wall2425.23s and largest worker peak413.39MiB. No retries, new campaign or optimizer updates.

Evidence: [primary status](../research/alphazero_value_diagnostics/results/status.json), [lifecycle](../research/alphazero_value_diagnostics/results/lifecycle.json), [analysis](../research/alphazero_value_diagnostics/results/analysis.json), [report supplement](../research/alphazero_value_diagnostics/results/report-supplement.json), [independent arithmetic](../research/alphazero_value_diagnostics/results/independent-clean-recomputed.json), [independent rerun](../research/alphazero_value_diagnostics/results/independent-clean-model-rerun.json). Per-task identities, complete markers, exact visits, all matches and full value diagnostics remain in results/tasks/. The final SHA256 [evidence manifest](../research/alphazero_value_diagnostics/results/evidence-manifest.json) covers published evidence; checkpoints remain only in their original local output trees.

### Accessed checkpoint identities

| Arm | Generation | Boundary SHA256 |
| --- | --- | --- |
| 314159-cnn | 0 | `de89e765309e30376ac3e3e3f16861533751de2ab0264a79a8d6285525bdbe41` |
| 314159-cnn | 5 | `15d3afe28c28a48987a55d8dfcedd9de34bb10d6708a79f26b1d8cccb99b6b64` |
| 314159-cnn | 6 | `2646d7cdee7d7b069412688bf95fd9794bea30402b4952120c904dbcb399401d` |
| 314159-resnet3x64 | 0 | `60c864e8a0303d80844b129cedd694a31cd3a0b34e063c2516fdf1958cd87c4f` |
| 314159-resnet3x64 | 5 | `b7801756d133c26907e5b9ce0173031938bcc55d4dfa0c48c0d96ad5d43f8dbe` |
| 314159-resnet3x64 | 6 | `b688095312730bc34886a9819864b772498544c83b6d0ae84ac3eb656879ec9d` |
| 42-cnn | 0 | `5f9800bc86ff1efa21f4d2be5af04fa882c0a7015a454651c351479579e7ac21` |
| 42-cnn | 5 | `4418c2b4fa1ec26bccaae098fc86e7ce42b5847b51967de1011a651798d371be` |
| 42-cnn | 6 | `ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b` |
| 42-resnet3x64 | 0 | `a514f4df9c5b432f17f6ab06f177441272f4d2a0adb28b52f0294a71a145fc86` |
| 42-resnet3x64 | 5 | `102249ac6ff6d854a3c7d365c28380760e4a2975e2bf330d1c92106a76df804f` |
| 42-resnet3x64 | 6 | `429289d22e8b37354f9bd77dae97975081d3cdbae8c65357be4ad53b748d5503` |

All28 identities are retained in freeze.json and final-integrity-audit.json; only g0/g5/g6 were used. Remaining limitations: two seeds, one decision checkpoint, one reference opponent, one budget/c, reused development fixtures, unequal correlated replay families, missing g7, sparse exact/replay overlap, no full independent oracle re-solving, no multiplicity correction, and the retained access breach. None is resolved by a tactical improvement alone.
