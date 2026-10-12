"""Render final Markdown tables from audited retained evidence; no model calls."""
from pathlib import Path
import re
from research.alphazero_training_ablation.common import ROOT, REPO, read

LINK='../research/alphazero_training_ablation/'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,r))+' |' for r in rows])
def mean(result): return result['mean']
def estimate(result,scale=1,digits=4):
    lo,hi=result['ci95']
    return f'{result["mean"]*scale:+.{digits}f} [{lo*scale:+.{digits}f}, {hi*scale:+.{digits}f}]'
def pct(v):return f'{100*v:.2f}%'

def main():
    a=read(ROOT/'results/analysis.json'); m=read(ROOT/'results/mechanism-resources.json')
    b=read(ROOT/'results/checkpoint-behavior.json'); audit=read(ROOT/'results/integrity-audit.json')
    independent=read(ROOT/'results/independent-recomputed.json'); overlap=read(ROOT/'results/observed-replay-overlap.json')
    sections=['# Phase 4D.4G — controlled AlphaZero exploration training ablation']
    sections.append('''## Executive summary

**Extending exploration from eight to sixteen plies did not demonstrate a better
trained AlphaZero agent in this pilot.** It reduced self-play repetition in all
four seed/generation comparisons, but failed the prospective neural-learning and
tactical gates. Playing-strength noninferiority is inconclusive overall. This is
not proof that longer exploration generally harms learning or strength.

The controlled experiment completed correctly: both seeds, both arms, generations
21/22,512 complete self-play games,512 AdamW updates, all6,280 position searches
and256 strength matches. No retries, abandoned tasks, additional generations,
model promotion or deployment occurred. All150 official campaign files remain
byte-identical. Primary wall time was2,016.159 seconds (33.60 minutes).

| Independent conclusion | Decision | Interpretation |
|---|---|---|
| Execution | PASS | Full budgets, strict lineages/evidence and preservation verified |
| Neural learning | FAIL | Practical improvement criteria not met; seed314159 value-MSE regression versus control exceeds margin |
| Tactical reliability | FAIL | Four prospective2pp point gates fail; two additional deep gates remain inconclusive |
| Playing strength | INCONCLUSIVE | Seed314159 noninferiority passes; seed42 cannot exclude the declared5pp regression |

Keep eight exploratory plies as the working schedule. Prioritize Phase4D.4H-B:
a controlled CNN/ResNet cost, throughput and fixed-data learning comparison with
a leakage-aware held-out split. Do not adopt sixteen plies or automatically extend
this training pilot. These are four separate conclusions, without a composite
score or selected champion.''')
    sections.append(f'''## Design, source freeze and prior research

Dedicated branch `phase4d4g-alphazero-training-ablation`, isolated worktree
`experiment-output/phase4d4g-worktree`, started at
`9325f0a19e6b0106ea1dcd265316b57323f8eb4f`. The pushed prospective design/execution
commit was **b4b1d9f5a42db0889fb2eed2acea6252da8526c9**, recorded before collection.
The complete [DESIGN.md]({LINK}DESIGN.md) remains unchanged. Preparation, detached
execution and this final-analysis session were separate.

Four strict generation20 forks: seeds42/314159, control and treatment. Both arms
within seed had identical model, ordered AdamW state, replay, counters and persistent
sampling/reflection/process RNGs. Control usesτ1 at zero-based plies0–7 then0;
treatmentτ1 at0–15 then0. Each new generation collects64 complete games at256
simulations/move,pool8, then64 fixed updates,batch128. The325,896-parameter
three-layer CNN, noise.25/alpha1, LR3e-4, AdamW configuration, clipping5,
reflection.5, losses, PUCT and eight-generation replay window remain matched.
Positions, length and time are measured, not artificially equalized.

[4A](phase4d4a-alphazero-baseline.md) identified weak raw tactics and training
bottlenecks; [4B](phase4d4b-alphazero-bitboard-puct.md) and
[4C](phase4d4c-alphazero-batched-inference.md) validated bitboard/pool-eight
optimizations. [4D](phase4d4d-alphazero-exploration-diversity.md) and
[4E](phase4d4e-alphazero-exploration-confirmation.md) replicated reduced repetition
while leaving safety and learning unresolved. [4F](phase4d4f-alphazero-training-infrastructure.md)
provided strict experimental forks and complete-boundary recovery. This phase
tests actual learning rather than treating diversity as strength.

Frozen source/runtime are in [freeze.json]({LINK}results/freeze.json), and parent
conversion/state equality in [readiness.json]({LINK}results/readiness.json).
All44 enforced files still match, including the original source digest
`b8aae0d6e14292dad2d2bcdb86f6048fad93a5eeaa5d342d7b5c340e9fc2bde5`.
Fixture SHA-256 is `{audit['fixture_sha256']}`. Runtime: AppleM4 CPU,
CPython3.11.17,Torch2.10.0,NumPy1.26.4, deterministic algorithms and all intra/inter-op
and five library thread controls fixed to one. Final analysis adds only read-only
tools in the sibling `research/alphazero_training_ablation_postexecution/`, outside
the immutable execution closure; no frozen source/configuration/fixture/rule changes.''')
    sections.append(f'''## Completion, integrity and provenance

The manager finished `COMPLETE_PENDING_FINAL_ANALYSIS` normally; all**22** scheduled
tasks completed (8 training,10 position-evaluation,4 match tasks). A read-only
[process check]({LINK}results/process-verification.json) found no remaining supervisor
or worker. Exact task order/directories/success markers and empty failure logs
show no unreported retry. No temporary boundary remains. The final
[integrity audit]({LINK}results/integrity-audit.json) independently reloads all12
experimental generation20/21/22 files with trusted SHA and strict source/runtime
validation, verifies full deterministic state chains, normalized visits/actor
outcomes, legal complete histories, optimizer persistence, counters and replay
expiration. Exported games and recorded per-game RNG seeds agree.

Execution order:42C21,42T21,314159T21,314159C21,42T22,42C22,314159C22,314159T22,
with complete development evaluation after every new boundary. Frozen collection
checks verify immutable learner/snapshot weights before replay/updates; no update
occurs during collection. Generations21/22 evict historical13/14 respectively,
retaining14–21/15–22 with mixed schedule metadata. Atomic publication follows the
reviewed fsync/strict-validation/no-replace-hardlink/directory-fsync path and all
successful reloads; no independent syscall trace is claimed.

Evaluation checkpoint assignment is reconstructed from the frozen manager's
current-boundary transitions and strict evaluation loader. Raw rows lack their
own model-hash attestation; they were not re-evaluated to add one. File/evidence
hashes detect changes, not malicious jointly rewritten artifacts.

Historical parent boundary SHA-256:42=`2866722b5e55726b75ae27d4bdb70e98d9b01b9ca503b0925fe9aa8b17d210b1`;
314159=`328022e5f4c45c11650b7fd2b76fbafbbe4734f1161a75d91b5d0b4cb274f9d6`.
Matched initial-state SHA:42=`2dd15035fd4e10654a80a78cf5dec16425d4ad6e23f21323adcc219ca8bee5e4`;
314159=`5aa9916bc689710aa8ed4e6a5d268cb94d913cb8375e5d14542280587be29c77`.
These are new experimental forks. They do not reproduce the historical official
campaign's sequential stochastic trajectories.''')
    checkpoint_rows=[]
    for seed in (42,314159):
        for arm in ('control','treatment'):
            for gen in (20,21,22):
                r=audit['boundaries']['records'][f'{seed}-{arm}-g{gen}']
                checkpoint_rows.append([seed,arm,gen,f'`{Path(r["path"]).relative_to(REPO/"experiment-output/phase4d4g-primary")}`',f'`{r["file_sha256"]}`'])
    sections.append('### Local checkpoint recovery identities\n\nAll paths below are relative to this worktree’s ignored `experiment-output/phase4d4g-primary/`.\n\n'+table(['Seed','Arm','Gen','Relative checkpoint path','File SHA-256'],checkpoint_rows)+f'\n\nEvery model, optimizer, replay, RNG, counter, archive-integrity and deterministic computational-state hash is recorded in [integrity-audit.json]({LINK}results/integrity-audit.json). Git reporting is excluded from computational equality; enforced content/runtime are not. Binary checkpoints remain local and unchanged.')
    sections.append(f'''## Frozen evaluation and uncertainty

The [fixtures]({LINK}fixtures.json) contain nonsealed solved300, tactical200,
deep128 (64 complete-vector4E families plus mirrors), and32 opening base families.
solved/deep rows have100% complete legal-action WDL coverage. Tactical rows have
independent immediate-win/safe labels, not complete deeper values; no arbitrary
tactical action receives an invented exact label. No new solving was performed.

Raw policy and raw PUCT512 (noise/guards/proof propagation off) use fixed independent
evaluation identities, matched across checkpoints/arms. Each seed's identical
parent model was evaluated once and reused for its two arms. Generation22 matches
use both colors on each opening,64 games/arm, against the own seed's frozen parent.
These are paired parent comparisons, not direct treatment-control games.

The unchanged10,000-replicate seed4404703 bootstrap resamples entire mirror/shared
opening-ancestry clusters and recomputes row-weighted means:102 solved,71 tactical,
28 deep clusters. Mirrors are not independent. Learning pools equal seed contrasts
before shared-family resampling; strength retains color pairs and independently
resamples32 opening families within each seed. Two-sided95% intervals use.025/.975;
one-sided95% bounds use.05/.95. All intervals are conditional on these two fixed
networks and development families, not uncertainty over arbitrary training seeds.
No multiplicity correction was declared. Full unrounded estimates and per-seed
intervals are in [analysis.json]({LINK}results/analysis.json).''')
    learning_rows=[]; change_rows=[]; calibration_rows=[]
    for seed in ('42','314159'):
        for label in ('parent','control-g21','treatment-g21','control-g22','treatment-g22'):
            r=a['learning'][seed][label]; metric=r['metrics']
            learning_rows.append([seed,label,f'{mean(metric["ce"]):.6f}',pct(mean(metric['accuracy'])),f'{mean(metric["mse"]):.6f}',f'{mean(metric["optimal_mass"]):.6f}'])
            cal=r['calibration']; delta=r.get('calibration_change',{})
            calibration_rows.append([seed,label,f'{cal["policy_ece"]:.4f}',f'{delta.get("policy_ece",0):+.4f}',f'{cal["value_absolute_calibration_error"]:.4f}',f'{delta.get("value_absolute_calibration_error",0):+.4f}'])
            if label!='parent':
                ch=r['change_from_parent']; change_rows.append([seed,label,estimate(ch['ce']),estimate(ch['accuracy'],100,2),estimate(ch['mse'])])
    sections.append('## Neural learning: parents and both new generations\n\nCE is `−log(sum policy on exact-optimal actions)`; accuracy uses raw legal-policy argmax; MSE uses actor-relative exactWDL.\n\n'+table(['Seed','Checkpoint','Optimal-set CE (nats)','Raw optimal accuracy','Value MSE','Optimal probability'],learning_rows)+'\n\n### Change from own generation20 parent\n\nPoint change [two-sided95% family interval]; accuracy changes are percentage points.\n\n'+table(['Seed','Checkpoint','ΔCE','Δaccuracy(pp)','ΔMSE'],change_rows)+'''\n\nBoth arms improve generation22 value MSE relative to their own parent. Seed314159
treatment improves less than control: the treatment-control value regression is
a relative comparison, not an assertion that treatment worsened its own parent.
Small policy changes and their uncertainty differ by checkpoint/seed.''')
    contrast_rows=[]
    for seed in ('42','314159'):
        for gen in (21,22):
            r=a['learning'][seed][f'contrast-g{gen}']; contrast_rows.append([seed,gen,estimate(r['ce'],digits=6),estimate(r['accuracy'],100,2),estimate(r['mse'],digits=6),estimate(r['optimal_mass'],digits=6)])
    r=a['learning']['pooled_g22']; contrast_rows.append(['Equal-seed pooled',22,estimate(r['ce'],digits=6),estimate(r['accuracy'],100,2),estimate(r['mse'],digits=6),estimate(r['optimal_mass'],digits=6)])
    sections.append('### Primary treatment−control contrasts\n\n'+table(['Seed','Gen','ΔCE [95%]','Δaccuracy(pp) [95%]','ΔMSE [95%]','Δoptimal probability [95%]'],contrast_rows)+'''\n\n**Neural learning FAIL under the frozen rules.** Pooled CE+0.001420 fails≤−0.02;
its lower95% bound−0.006656 is above−0.02. Pooled accuracy−0.833pp fails≥+2pp;
its upper95% bound+1.025pp is below+2pp. Both seed CE point differences are
positive; seed42 accuracy is−1.667pp. Seed314159 MSE+0.041322 exceeds+0.02
(95% interval[+0.006618,+0.080432]). Each stated frozen FAIL condition is retained.
The pooled CE/accuracy improvement intervals overlap zero and their required
confidence conditions also fail. Policy harm is not conclusively established;
the practical learning improvement sought here was not demonstrated.''')
    sections.append('### Calibration and generalization\n\n'+table(['Seed','Checkpoint','Policy ECE proxy','Δfrom parent','Value bin error','Δfrom parent'],calibration_rows)+f'''\n\nCalibration uses the frozen ten bins: max-action probability versus optimal-set
correctness, and mean raw value versus exact meanWDL. PolicyECE is a diagnostic
proxy when several actions are optimal, not a validated probability-of-correctness
model. Detailed bins are in [analysis.json]({LINK}results/analysis.json). Lower
ECE does not override worse accuracy/value endpoints; calibration effects are mixed.

There is **no prospective unseen test split or valid generalization-gap estimate**.
Training policy CE targets visit distributions over seven logits; evaluated CE
targets exact optimal sets with legal masking. Training value loss uses sampled
self-play outcomes, whereas solvedMSE uses exactWDL on different positions.
Subtracting these losses would misstate a generalization gap. The descriptive
[observed replay overlap]({LINK}results/observed-replay-overlap.json) finds solved
rows already present in observed parent replay:2/300 for42,4/300 for314159;
tactical0/200 for both; deep28/128 for42 and6/128 for314159. Generation22 overlap
is2/300 solved and0/200 tactical in all arms, and deep28/34 for42C/T,8/8 for314159C/T.
These reflection/actor matches are material for interpreting the reused4E table.
Historical generations1–12 are outside available retained replay, so absence
from observed replay does not establish unseen generalization. No fixture was
removed or regrouped after this check.''')
    training_rows=[]
    for r in m['rows']:
        t=r['training']; training_rows.append([r['seed'],r['arm'],r['generation'],f'{t["policy_loss"]["mean"]:.4f}',f'{t["value_mse"]["mean"]:.4f}',f'{t["combined_loss"]["first16"]:.4f}→{t["combined_loss"]["last16"]:.4f}',f'{t["gradient_norm"]["mean"]:.3f}/{t["gradient_norm"]["max"]:.3f}'])
    sections.append('### Training losses and gradient stability\n\nEach row has64 updates; loss trajectory shows means over first/last16 sampled minibatches, not a fixed validation set.\n\n'+table(['Seed','Arm','Gen','Mean policy loss','Mean value loss','Combined first16→last16','Grad mean/max'],training_rows)+f'\n\nAll512 updates are finite, with zero clipped updates and maximum gradient norm2.591 below clip5. Generation22 mean losses often decline modestly; sampling noise and heterogeneous historical targets prevent monotonic-loss claims. Complete64-step trajectories, gradient quantiles and diagnostics are in each [generation sidecar]({LINK}results/execution-evidence/) and [mechanism-resources.json]({LINK}results/mechanism-resources.json). Training stability is not evidence of improved playing strength.')
    tactical_rows=[]; errors=[]
    for r in b['rows']:
        t=r['tactical']; tactical_rows.append([r['seed'],f'{r["arm"]}-g{r["generation"]}',*[pct(t[f'{method}-{cat}']['metrics']['mean']) for method in ('policy','puct') for cat in ('unique_win','unique_safe')],f'{t["policy-avoidable"]["count"]}/{t["policy-missed_win"]["count"]}',f'{t["puct-avoidable"]["count"]}/{t["puct-missed_win"]["count"]}'])
    sections.append('## Tactical reliability\n\nWin and safe categories each contain100 rows. Unique-safe is the established mandatory-block/safe-response endpoint; no separate multiple-safe stratum is available. Last columns are avoidable one-ply/missed-win counts over200 rows.\n\n'+table(['Seed','Checkpoint','Policy win','Policy safe/block','PUCT win','PUCT safe/block','Policy errors/missed','PUCT errors/missed'],tactical_rows)+f'\n\nSearch greatly raises absolute win/safe accuracy relative to raw policy, while retaining mistakes. Exact per-checkpoint/parent intervals and paired search−policy contrasts are in [checkpoint-behavior.json]({LINK}results/checkpoint-behavior.json); raw measurements remain separate from gates.')
    gates=[]
    for seed in ('42','314159'):
        for gen in (21,22):
            for method in ('policy','puct'):
                for cat in ('unique_win','unique_safe'):
                    r=a['tactical'][f'{seed}-g{gen}-{method}-{cat}']
                    gates.append([seed,gen,method,'win' if cat=='unique_win' else 'safe/block',estimate(r['vs_control'],100,2),estimate(r['vs_parent'],100,2),r['label']])
    sections.append('### Frozen2pp point-regression gates\n\nTreatment differences in percentage points [two-sided95% interval]. A point drop greater than2pp versus either parent or control fails, regardless of interval width.\n\n'+table(['Seed','Gen','Method','Category','T−C(pp) [95%]','T−parent(pp) [95%]','Point gate'],gates)+'''\n\nFour FAIL cells:42/g22 policy-safe−3pp versus control;314159/g21 policy-safe−6pp
versus parent;314159/g21 PUCT-win−3pp versus parent;314159/g22 policy-safe−3pp
versus control. These are prospective point-margin failures, not confidence-bound
proofs of broad tactical harm. The remaining point gates pass; no margin changed.''')
    deep=[]
    for seed in ('42','314159'):
        for gen in (21,22):
            for method in ('policy','puct'):
                for suite in ('solved','deep'):
                    r=a['tactical'][f'{seed}-g{gen}-{method}-{suite}-deep']; effect=r['vs_control']
                    deep.append([seed,gen,method,suite+'300' if suite=='solved' else '4E128',pct(next(iter(r['control_rate'].values()))['mean']),pct(next(iter(r['treatment_rate'].values()))['mean']),f'{100*effect["mean"]:+.2f}',f'{100*effect["one_sided95_lower"]:+.2f}, {100*effect["one_sided95_upper"]:+.2f}',f'{100*r["treatment_vs_parent"][method+"_deep_mistake"]["mean"]:+.2f}',r['label']])
    sections.append('## Deeper game-theoretic mistakes\n\nA mistake decreases exact achievable actorWDL (win→draw/win→loss/draw→loss), not merely an eventual loss. Complete legal vectors cover every included action. Bounds shown are distinct one-sided95% lower/upper bounds, compared with+5pp.\n\n'+table(['Seed','Gen','Method','Suite','Control error','Treatment error','Δ(pp)','L05,U95(pp)','T−parent(pp)','NI'],deep)+f'''\n\nAll deep gates PASS except42 raw-policy4E128 atg21/g22: upper bounds+5.83/+5.56pp
exceed+5pp, yielding INCONCLUSIVE. Neither lower bound exceeds the margin, so
there is no deep-margin FAIL. These bounds do not certify arbitrary positions;
4E has only28 clusters and substantial observed-replay overlap. All parent/arm
rates, changes and intervals are in [checkpoint-behavior.json]({LINK}results/checkpoint-behavior.json).
The separate overall tactical **FAIL** is driven by the four immediate point gates.''')
    strength_rows=[]; strength_effect=[]
    for seed in ('42','314159'):
        for arm in ('control','treatment'):
            r=a['strength'][seed][arm]; strength_rows.append([seed,arm,f'{r["wins"]}/{r["draws"]}/{r["losses"]}',pct(r['metrics']['score']['mean']),f'{100*r["metrics"]["score"]["ci95"][0]:.2f}–{100*r["metrics"]["score"]["ci95"][1]:.2f}%',pct(r['by_color']['0']['score']['mean']),pct(r['by_color']['1']['score']['mean'])])
        r=a['strength'][seed]; c=r['contrast']; strength_effect.append([seed,estimate(c,100,2),f'{100*c["one_sided95_lower"]:+.2f}',f'{100*c["one_sided95_upper"]:+.2f}',r['noninferiority']])
    sections.append('## Playing strength against frozen parents\n\nEvery row has64 matches,32 per color. Scores=(wins+0.5draws)/64.\n\n'+table(['Seed','Arm','W/D/L','Score','Two-sided95%','As X score','As O score'],strength_rows)+'\n\n'+table(['Seed','T−C(pp) [two-sided95%]','One-sided lower(pp)','One-sided upper(pp)','−5pp NI'],strength_effect)+'''\n\nBoth treatment point scores exceed50%, but equal-seed pooled score53.52% has
95% interval[47.27%,59.77%]; lower bound<50%, so the supporting positive signal
does not pass. Seed314159 demonstrates the declared noninferiority margin on
these openings. Seed42's lower bound−7.03pp crosses−5pp; its upper bound+9.38pp
does not establish a material regression. Overall **INCONCLUSIVE**. Small positive
matched point differences do not establish a superior agent. No additional
tournament or checkpoint promotion was performed.''')
    diversity=[]; labels=[]; replay=[]; sampling=[]; cost=[]
    for r in m['rows']:
        tag=f'{r["seed"]}/{r["arm"]}/g{r["generation"]}'
        diversity.append([tag,r['positions'],f'{r["mean_length"]:.2f}',pct(r['duplicates_rate']),pct(r['mirror_duplicates_rate']),pct(r['post8_duplicate_rate']),r['opening_families'],f'{r["policy_target_entropy"]:.3f}'])
        labels.append([tag,f'{r["outcomes"]["first_win"]}/{r["outcomes"]["second_win"]}/{r["outcomes"]["draw"]}',f'{r["outcome_labels"].get("-1",0)}/{r["outcome_labels"].get("0",0)}/{r["outcome_labels"].get("1",0)}',r['avoidable_oneply'],r['missed_win'],pct(r['avoidable_rate'])])
        replay.append([tag,r['replay_positions'],r['experimental_replay_positions'],pct(r['experimental_replay_fraction']),f'{r["new_sample_appearances"]}/8192',pct(r['new_sample_fraction']),pct(r['current_generation_sample_fraction'])])
        sampling.append([tag,', '.join(f'{g}:{n}' for g,n in sorted(r['sampled_by_generation'].items(),key=lambda x:int(x[0])))])
        cost.append([tag,f'{r["collection_seconds"]:.2f}',f'{r["optimization_stage_seconds"]:.2f}',f'{r["generation_task"]["wall_seconds"]:.2f}',r['collection_forward_calls'],f'{r["mean_batch_occupancy"]:.2f}',f'{r["neural_positions_per_collection_second"]:.0f}',f'{r["generation_task"]["peak_rss_mib"]:.1f}'])
    sections.append('## Self-play diversity and outcome diagnostics\n\nEach row is64 complete games. Duplicate fractions count repeat occurrences after the first unique state; mirror fractions canonicalize reflection/actor. Post8 uses only plies≥8.\n\n'+table(['Seed/arm/gen','Positions','Mean length','Duplicates','Mirror duplicates','Post8 duplicates','Opening families','Target entropy(nats)'],diversity)+'\n\n'+table(['Seed/arm/gen','X wins/O wins/draws','Actor z counts −1/0/+1','Avoidable errors','Missed wins','Avoidable/position'],labels)+'''\n\nTreatment reduces duplicate and mirror-duplicate rates in every matched cell
(duplicate reductions9.48,14.67,17.72,26.54pp). It also produces fewer positions
and shorter games in every cell. Opening-family counts and target entropy do not
uniformly increase. Self-play outcome-label distributions shift; they are not
head-to-head strength scores. One-ply errors are mixed:314159 treatment has more
avoidable errors in both generations, whereas42/g22 has fewer. No complete deep
oracle scan of all self-play positions was prospectively budgeted; existing
immediate-win/all-unsafe exact-proof diagnostics remain in the boundaries.
Unresolved deeper self-play states are not labeled safe or erroneous.''')
    sections.append('## Replay dilution and actual optimizer exposure\n\nParents initialize with68,856/64,514 examples and4,957/4,890 AdamW steps for42/314159,2,048 games across13–20. Generation21 retains14–21; generation22 retains15–22, replacing entire old generations despite smaller64-game additions.\n\n'+table(['Seed/arm/gen','Replay positions','Experimental positions','Experimental share','New sample appearances','New sampling share','Current-gen sampling share'],replay)+'\n\nSampling appearances by producing generation (generation:count), each row totals8,192=64×128:\n\n'+table(['Seed/arm/gen','Sampling by source generation'],sampling)+'''\n\nOnly2.42–3.94% of generation21 samples and6.21–7.80% of generation22 samples
come from experimental data. Across all65,536 sample appearances,3,397 (5.18%)
are experimental:62,139 (94.82%) remain historical. Treatment's shorter collections
lower new replay/sampling shares versus its control in every cell. Each lineage
gets only128 new updates, approximately2.6% of its parent's accumulated steps.

**Observed:** data repetition falls, historical replay dominates actual updates,
and practical policy gains do not emerge. **Hypotheses:** replay dilution/small
update dose may limit sensitivity; changed outcomes, harder sampled actions or
checkpoint-specific value effects may offset diversity gains. This experiment
does not identify these mechanisms causally or isolate architecture versus dose.
It does not prove exploration diversity unimportant, nor justify automatically
adding data or updates after the failed gates.''')
    resources=m['resources']; st=audit['structure']
    sections.append('## Computational cost and budget reconciliation\n\n'+f'''Execution began**{resources['start_eastern']}**, finishing approximately
**{resources['finish_eastern_approx']}**. Finish is reconstructed from saved epoch
start plus monotonic elapsed, not a separately saved process-exit timestamp.
Primary wall**{resources['wall_seconds']:.3f}s /{resources['wall_seconds']/60:.2f}min**
is below180min; every generation attempt is below20min (maximum
{resources['max_generation_attempt_seconds']:.3f}s). Child walls sum
{resources['child_wall_seconds']:.3f}s, with{resources['supervisor_and_process_wall_overhead']:.3f}s
supervisor/process/I/O overhead. Child CPU is{resources['child_cpu_seconds']:.3f}s;
supervisor/startup CPU before worker timers is not fully recorded. Highest child
peakRSS is{resources['max_child_peak_rss_mib']:.1f}MiB, not total machine memory.

Training-task walls total721.044s; position evaluations608.190s; strength657.040s.
Direct collection totals489.552s. Optimization stage totals28.210s: first
before_update→before_checkpoint includes64 updates plus summary/RNG bookkeeping,
excluding persistence; it is **not pure optimizer time**. The rest of training
task time includes strict loads, reconstruction, hashing, persistence and reloads.
Actual primary counts are512 games/updates,16,397 positions,256 strength games,
7,274 strength decisions. PUCT simulations:4,197,632 self-play,3,215,360 development
positions,3,724,288 strength; **11,137,280 total**. Collection executed426,755 batch
forwards and3,211,665 neural positions (mean occupancy7.53). Full evaluation
neural-forward throughput counters were not recorded.

Generation resource detail (all eight have64 updates):\n\n'''+table(['Seed/arm/gen','Collection(s)','Optimizer stage(s)','Full task(s)','Batch forwards','Mean batch','Neural pos/s','PeakRSS MiB'],cost)+f'''\n\nAll10 per-checkpoint evaluation walls, all4 match task walls/CPU/RSS and exact
resource reconciliation are in [mechanism-resources.json]({LINK}results/mechanism-resources.json)
and the [retained status ledger]({LINK}results/execution-evidence/status.json).
No aborted/failed/retried primary work is present. Readiness132.656s and8 disposable
updates/72 bounded search probes were separate prelaunch work, not primary data.
Final evidence-only audits/analysis/tests are separate from the33.60min primary
clock; no unused training budget was filled.''')
    sections.append(f'''## Independent critical review and retained corrections

A new reviewer agent, which authored neither the frozen implementation nor main
results analysis, independently reconstructed6,280 evaluation metrics,512 legal
self-play histories/targets/diversity/tactics,256 strength outcomes/colors,
cluster components and10,000-replicate multiplicity-weighted bootstraps.
It compared1,316 main-analysis numerical scalars; maximum discrepancy5.55e-16.
Labels, confidence directions, thresholds, replay exposure/resources and support
signal agree. See [independent-results-review.md]({LINK}results/independent-results-review.md)
and [independent-recomputed.json]({LINK}results/independent-recomputed.json).
This is independent arithmetic/source review on the same host, not laboratory replication.

Frozen analysis sources remain untouched. Limitations: its execution label alone
does not audit the full lineage; this report requires the separate strict audit.
Pooled learning uses positional zip without a fresh ID check; all actual row
orders match frozen membership, and independent ID-based pairing reproduces it.
There is no material arithmetic defect requiring a changed scientific analysis.

The new auxiliary audit retained and corrected three reporting issues: JSON
integer-key/tuple normalization when comparing binary histories with sidecars;
an unrelated H-B worktree's untracked review/report status drift while tracked
bytes/HEAD remained equal during the protection audit; and reconstructed payload
integrity reflecting current Git reporting rather than the stored archive's
integrity field. Original failure logs, before/current/after inventories and
[integrity metadata correction]({LINK}results/archive-integrity-reporting-correction.json)
are preserved. The final audit reads the original verified archive integrity.
No runtime/source validation or decision rule was weakened, no original result
was discarded, and no training/evaluation was repeated. The prelaunch ImportError,
wrong test-filename attempt and prior Git-reporting comparison failure remain
documented in the prospective report/review evidence and validation summary.''')
    sections.append(f'''## Historical preservation, publication and recovery

All**150** official campaign files match the4F byte inventory before/after final
audits. Declaration SHA remains
`c8ab2227d4b2c707b3185da8695a991f0914faede88ecd5f1c7fc84fb017b716`.
Historical checkout remains`1894bc4d8f035cf738df8e3fc0ce234f1ce2c8be`;
the designated ResNet branch/worktree remains`7f5182daa10156e04a8b0f7727d9bec9cad39cf1`.
Tracked prior source/research/benchmark files and these protected worktrees are
unchanged. The separate H-B worktree was observed active outside this task;
its untracked status drift is reported without touching it. This task issued no
write to either ResNet worktree, production/API/deployment, main or canonical benchmark.

The unchanged `publish_evidence.py` packages completed JSON/log evidence under
[execution-evidence]({LINK}results/execution-evidence/), with
[evidence-manifest.json]({LINK}results/evidence-manifest.json). Large checkpoint
binaries stay ignored/local. Only reviewed research logs are force-added where
the repository's generic*.log ignore rule applies. No checkpoints are committed.
[integrity-audit.json]({LINK}results/integrity-audit.json) records every absolute
recovery path and checkpoint/model/optimizer/deterministic-state hash.

From this worktree, set all five thread variables to1 and use the existing M4
Python with`-B`; follow [README.md]({LINK}README.md): run evidence-only
`python -m research.alphazero_training_ablation.analyze`. Publication uses the
existing `publish_evidence` command into a fresh destination; never overwrite the
retained first package. Supplementary checks use the sibling postexecution
`audit`, `summarize` and `overlap` modules. The packaged independent scripts
reproduce metrics, diagnostics and complete-game checks without original analysis
helpers. See [postexecution instructions](../research/alphazero_training_ablation_postexecution/README.md).
Strict recovery uses original `experimental_cli inspect` on a visible boundary
and its trusted SHA; neither launcher nor training command is needed here.''')
    sections.append('''## Limitations and recommended next phase

The pilot estimates these fixed CNN checkpoints, two seeds and reused development
families after two64-game/64-update generations. It has low new-data exposure,
few optimizer steps, small strength samples, only28 deep clusters, some directly
overlapping replay positions and unknown early historical replay provenance.
Bootstrap intervals do not include between-training-seed uncertainty; one fixed
evaluation seed per position and no multiplicity correction were declared.
Tactical point FAIL does not equal statistical proof of broad harm. Strength
does not demonstrate superiority and does not rule out the5pp regression in42.
Checkpoint21/22 effects differ, including314159 treatment's smaller value gain
at22. There is no evidence here to adopt sixteen exploratory plies.

Recommend **Phase4D.4H-B controlled CNN versus ResNet inference-cost, training-
throughput and fixed-data learning comparison**, using read-only historical
generations13–20 replay and a prospectively frozen split that groups reflected
boards, transpositions and related trajectories/openings across partitions.
Prioritize ResNet3×64 (322,440 parameters) against CNN (325,896), with equal
optimizer/data budgets and measured wall/throughput. ResNet5×64 (470,408) can
be a separately declared capacity/cost comparator, not presumed equally sized.
The existing ResNet implementation at7f5182d is unchanged and no comparison is
launched here. This separates learning/architecture sensitivity from the
exploration trajectory and dilution confound before allocating more self-play.

Retain the eight-ply schedule. Longer exploration may remain a future hypothesis
if later learning studies justify it, but reduced repetition alone, this pilot's
positive match point scores or unused compute budget do not justify further
training, treatment adoption, a champion or deployment.''')
    report='\n\n'.join(sections)+'\n'
    report=report.replace('these two fixed\nnetworks and development families','these fixed checkpoints from two training seeds\nand development families')
    report=report.replace('Both arms improve generation22 value MSE','Both arms have lower generation22 value-MSE point estimates')
    report=report.replace('all**22**','all **22**').replace('treatmentτ1','treatment uses τ1')
    report=report.replace('noise.25/alpha1','root noise epsilon0.25/alpha1').replace('reflection.5','reflection0.5')
    report=report.replace('Execution order:42C21,42T21,314159T21,314159C21,42T22,42C22,314159C22,314159T22,',
        'Execution order (C=control,T=treatment):42C21,42T21,314159T21,314159C21,42T22,42C22,314159C22,314159T22,')
    report=report.replace('Small policy changes and their uncertainty differ by checkpoint/seed.',
        'Most parent-change value intervals overlap zero; only314159 control at22 excludes zero. Small policy changes and their uncertainty differ by checkpoint/seed.')
    report=report.replace('Frozen analysis sources remain untouched. Limitations:',
        'Final evidence-only analysis/integrity tests:11 passed in2.21s, without model queries or optimizer work. The unchanged frozen analysis was reproduced to a fresh file with identical parsed JSON and byte SHA; see [frozen-analysis-reproduction.json]('+LINK+'results/frozen-analysis-reproduction.json) and [final-analysis-tests.txt]('+LINK+'results/final-analysis-tests.txt). Prelaunch6 ablation/56 boundary cases and341 relevant regressions remain documented in the frozen [validation summary]('+LINK+'results/validation-summary.json).\n\nFrozen analysis sources remain untouched. Limitations:')
    # Improve prose spacing while preserving code, hashes, link destinations and tables.
    protected=[]
    def hide(match):
        protected.append(match.group()); return f'\x00{len(protected)-1}\x00'
    report=re.sub(r'`[^`]+`|\]\([^\n)]+\)|\b[0-9a-f]{40,64}\b',hide,report)
    lines=[]
    for line in report.splitlines():
        if line.startswith('|'):
            line=re.sub(r'\b(seed|prospective|declared)(?=\d)',r'\1 ',line)
            line=re.sub(r'(\d)(?=pp\b)',r'\1 ',line)
        else:
            line=line.replace('AppleM4','Apple M4').replace('atg21','at generation21').replace('atg22','at generation22')
            line=re.sub(r'([A-Za-z]{2,})(?=\d)',r'\1 ',line)
            line=re.sub(r'(\d)(?=[A-Za-z]{2,})',r'\1 ',line)
            line=re.sub(r'(\d+/\d+),(?=\d)',r'\1, ',line)
            line=re.sub(r'([,;:])(?=[A-Za-z])',r'\1 ',line)
            line=re.sub(r'(?<=[A-Za-z]),(?=\d)',', ',line)
            line=line.replace('τ1','τ=1').replace('τ0','τ=0').replace('usesτ','uses τ').replace('use.05','use .05')
            for dataset in ('solved','tactical','deep'):
                line=re.sub(r'\b'+dataset+r' (\d+)',dataset+r'\1',line)
        lines.append(line)
    report='\n'.join(lines)+'\n'
    report=re.sub(r'\x00(\d+)\x00',lambda match:protected[int(match.group(1))],report)
    (REPO/'docs/phase4d4g-alphazero-training-ablation.md').write_text(report)

if __name__=='__main__':main()
