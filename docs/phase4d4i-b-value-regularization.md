# Phase 4D.4I-B — Value prediction regularization

Prospective design, before measured execution. Full protocol: [DESIGN.md](../research/alphazero_value_regularization/DESIGN.md).

Compare the original CNN objective `policy_CE + value_MSE` with the same objective plus `0.2 * mean(predicted_value ** 2)`. The penalty backpropagates through the value pathway and shared trunk; this is not a value-head-only change. Architecture, AdamW, targets, clipping and matched sample/reflection streams remain unchanged.

At LR3e-4, raising AdamW weight decay1e-4→1e-3 gives only about2.7e-7 extra direct shrink per update, or0.0552807% over2048 updates. The explicit output penalty is the single tested intervention.

Strict frozen H-C generation5 CNN boundaries, seeds42/314159, initialize identical arm states. Train retained replay only, without collecting self-play. Evaluate frozen next-generation6 games, excluding exact/mirror overlap for the primary novel-outcome MSE endpoint. Per-seed final improvement must reach0.02 absolute MSE with95% CI upper below0. Complete-game/opening-family paired bootstrap uncertainty, tactical/deep/strength safeguards, constant-zero and exact-label endpoints are fixed prospectively.

Cost-only pilot selected2048 updates/arm. Evaluate0/1024/2048, always final2048 primary. All nonsealed tactical200/solved300/deep128 development fixtures and40 paired development opening families against frozen Negamax3, raw PUCT256. Target20–30min, hard45min total, no retries/extensions/sweeps/additional generations. Source/runtime/checkpoint and all150 official campaign bytes are preserved. Original sealed-access incident remains disclosed historical provenance; no sealed fixture contents are opened.

Focused tests and independent prelaunch review must pass, then prospective sources/design are committed and pushed before launch. Final results, independent verification/review, resource accounting and one next-phase recommendation will replace this new phase’s prospective summary after the bounded execution. No merge, deployment or promotion.
