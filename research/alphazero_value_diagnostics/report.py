"""Final report/figure from completed frozen evidence only; no model queries."""
from collections import Counter
from pathlib import Path
import time
import json
from .common import REPO, RESULTS, ARMS, read, write


def table(header,rows):
    return '\n'.join(['| '+' | '.join(x.replace('|',r'\|') for x in header)+' |','| '+' | '.join(['---']*len(header))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def percent(x):return f'{100*x:.2f}%'
def difference(e):return f'{100*e["mean"]:+.2f} [{100*e["interval95"][0]:+.2f}, {100*e["interval95"][1]:+.2f}]'
def number(e):return f'{e["mean"]:.3f} [{e["interval95"][0]:.3f}, {e["interval95"][1]:.3f}]'


def main():
    started=time.monotonic()
    a=read(RESULTS/'analysis.json');s=read(RESULTS/'report-supplement.json')
    lifecycle=read(RESULTS/'lifecycle.json');status=read(RESULTS/'status.json')
    assert lifecycle['status']=='COMPLETE_PENDING_FINAL_REPORT' and lifecycle['hard_cap_satisfied']
    assert status['status']=='COMPLETE' and len(status['completed'])==48 and not status['failed']
    assert read(RESULTS/'independent-clean-recomputed.json')['comparison']['max_mean_discrepancy']==0
    assert read(RESULTS/'independent-clean-model-rerun.json')['complete']
    assert read(RESULTS/'final-integrity-audit.json')['all_identical']
    arms=list(ARMS)
    tactical_rows=[];match_rows=[];exact_rows=[];transition_rows=[];value_rows=[];color_rows=[]
    for arm in arms:
        for v in 'ABCDE':
            p=a['positions'][arm][v];m=a['matches'][arm][v]
            tactical_rows.append([arm,v,percent(p['tactical']['immediate_win']['mean']),percent(p['tactical']['safe_response']['mean']),
                round(p['tactical']['avoidable']['mean']*200),round(p['tactical']['missed_win']['mean']*200)])
            match_rows.append([arm,v,'/'.join(str(m['wdl'].get(k,0)) for k in ('win','draw','loss')),
                               percent(m['score']['mean']),difference(m['delta_from_A'])])
            exact_rows.append([arm,v,percent(p['solved']['correct']['mean']),percent(p['deep']['correct']['mean']),
                               difference(p['solved']['delta_from_A']['correct']),difference(p['deep']['delta_from_A']['correct'])])
            if v!='A':
                transition_rows.append([arm,v]+['/'.join(str(p[suite]['transitions']['counts'].get(k,0)) for k in ('fixed','introduced','action_changed')) for suite in ('tactical','solved','deep')])
            for color in ('0','1'):
                entry=m['by_color'][color]
                color_rows.append([arm,v,'X' if color=='0' else 'O',percent(entry['score']['mean']),difference(entry['delta_from_A'])])
        value=a['values'][arm]
        value_rows.append([arm,number(value['training_g6']['mse']),f"{value['training_g6']['zero_mse']['mean']:.3f}",
                           number(value['training_g5']['mse']),number(value['prequential_g5_on_g6']['mse']),
                           f"{value['prequential_g5_on_g6']['zero_mse']['mean']:.3f}",number(value['exact']['solved']['mse']),number(value['exact']['deep']['mse'])])
    contrasts=[]
    for arch in ('cnn','resnet3x64'):
        for suite in ('tactical','solved','deep','matches'):
            c=a['pooled'][arch][suite]['contrasts']
            contrasts.append([arch,suite]+[difference(c[k]) for k in ('zero_value_learned_prior','uniform_prior_learned_value','half_value','interaction')])
    category=[]
    for arch,d in s['tactical_categories'].items():
        for name,vs in d.items():
            category.append([arch,'immediate win' if name=='unique_win' else 'safe response',
                             percent(vs['A']['accuracy']['mean'])]+[difference(vs[v]['delta_from_A']) for v in 'BCE'])
    checkpoints=[]
    for arm,entries in read(RESULTS/'freeze.json')['checkpoints'].items():
        for e in entries:
            if e['generation'] in (0,5,6):checkpoints.append([arm,e['generation'],f"`{e['sha256']}`"])
    values_strata=[];noise=[]
    for arm in arms:
        d=a['values'][arm];e=d['exact']['solved'];tr=d['training_g6'];held=d['prequential_g5_on_g6']
        values_strata.append([arm]+[f"{e['by_target'][str(z)]['mean']:.3f}" for z in (-1,0,1)]+[
            f"{e['mean_value']:+.3f}",f"{e['mean_absolute_value']:.3f}",percent(e['saturated_share']),f"{e['calibration_error']:.3f}"])
        n=tr['target_noise']
        noise.append([arm,tr['positions'],n['unique_positions'],n['conflicting_repeated_positions'],
                      n['conflicting_position_examples'],f"{n['empirical_identical_position_mse_floor']:.3f}",
                      f"{held['by_seen_in_training']['False']['mean']:.3f}",f"{held['by_seen_in_training']['True']['mean']:.3f}"])
    sensitive=[]
    for mode in ('value_only_at_A','prior_only_at_A','both'):
        row=next(r for r in s['sensitive_positions'] if r['suite']=='tactical' and r['baseline_wrong'] and r['head_sensitivity']==mode)
        sensitive.append([row['arm'],f"`{row['id']}`",mode]+[str(row['actions'][v])+(' ✓' if row['correct'][v] else ' ×') for v in 'ABCDE'])
    losses=[]
    for arm in arms:
        for suite in ('solved','deep'):
            for v in 'ABE':
                d=s['exact_decisions'][arm][suite][v]
                losses.append([arm,suite,v,d['1']['rows'],d['1']['selected_values']['0'],d['1']['selected_values']['-1'],
                              d['0']['rows'],d['0']['selected_values']['-1']])
    text=f'''# Phase 4D.4I-A — AlphaZero value-head and PUCT causal diagnostics

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

{table(['Checkpoint','Variant','Immediate win','Safe response','Avoidable one-ply errors /200','Missed wins /200'],tactical_rows)}

Both zero and half values improve overall tactical accuracy in all four checkpoints. Value removal has its clearest benefit on mandatory safe responses. Uniform priors with learned values recover every immediate win across both architectures/seeds, while B retains some missed wins. Thus learned priors contribute to missed terminal alternatives; values contribute more strongly to the safe-response deficits. Uniform-prior safe-response gains alone have broad paired intervals and can introduce new errors. Neither head is the sole cause.

Pooled category contrasts, two fixed seeds per architecture:

{table(['Architecture','Category','Baseline','B−A, pp','C−A, pp','E−A, pp'],category)}

D solves every one-ply fixture. D searches/actions and all80 match histories are identical across checkpoints because both neural outputs are neutralized; these are repeated controls, not four independent neutral-network replications. The untrained A controls also solve every one-ply fixture. This shows the fixed budget is sufficient for these fixtures under neutral guidance, not that exploration/search-budget effects are ruled out elsewhere: only c=1.41 and256 simulations were tested.

The factorial interaction is A−B−C+D. On tactical accuracy its interval includes zero in both architectures. The errors require distinguishing head effects and individual position interactions, rather than attributing all mistakes to values.

## Exact decisions and action changes

{table(['Checkpoint','Variant','Solved300 optimal','Deep128 optimal','Solved Δ, pp','Deep Δ, pp'],exact_rows)}

No pooled exact-decision improvement excludes zero. B on deep128 decreases CNN accuracy by4.30pp [−10.42,+2.83] and increases ResNet accuracy by3.91pp [−2.70,+12.28]. CNN seed42 loses13 correct decisions while seed314159 gains2; ResNet gains1 and9. E's deep changes are−5,+5 decisions for CNN and+1,+1 for ResNet. Their pooled effects are0.00pp and+0.78pp, both uncertain. Tactical gains do not establish improved game-theoretic choices.

Selected exact WDL downgrades for the prioritized comparisons; columns retain the theoretical-class denominators:

{table(['Checkpoint','Suite','Variant','Theoretical wins','W→D','W→L','Theoretical draws','D→L'],losses)}

All A–E WDL transition vectors, including uniform controls, are retained in report-supplement.json. Accuracy and mistake rate are complements on complete exact vectors. Deep128's absence of theoretical losses changes both value-MSE interpretation and available downgrade classes.

Fixed baseline mistakes / newly introduced mistakes / changed actions, each suite's complete rows:

{table(['Checkpoint','Intervention','Tactical fixed/new/changed','Solved fixed/new/changed','Deep fixed/new/changed'],transition_rows)}

B fixes35 tactical mistakes and introduces1 across800 checkpoint-position decisions. E fixes26 and introduces0; C fixes38 but introduces19. These totals count checkpoint-position decisions, not independent situations. On solved300 and deep128, fixes coexist with many new mistakes; no overall deeper improvement follows from the tactical totals.

Examples from nonsealed development fixtures (zero-based physical columns; ✓ means the chosen action meets that fixture's label):

{table(['Checkpoint','Development row','Sensitivity at A','A','B','C','D','E'],sensitive)}

The full ranked position list includes every action-sensitive row, flags separate B/C changes at the unchanged baseline, and retains all A–E actions/correctness. Those flags are local intervention sensitivities, not immutable causal categories. High root magnitude is descriptive: the root value is not backed up, and oracle values for unlabelled internal leaves are unavailable.

## Playing strength

{table(['Checkpoint','Variant','W/D/L','Score','Δ from A, pp [95%]'],match_rows)}

CNN B−A is−20.00pp [−29.38,−10.31], with both seeds negative (−28.75,−11.25pp). ResNet B−A is−6.56pp [−16.56,+3.44], also negative on both seeds (−3.13,−10.00pp), with uncertainty spanning zero. The conclusion is strong for these CNNs and descriptive/inconclusive for ResNet, not proof of universal harm from zero values.

D scores11.875% (9 wins,1 draw,70 losses) despite100% tactical accuracy. With uniform priors, retained learned values improve match score over zero by32.50pp for CNN and23.75pp for ResNet; both intervals exclude zero. With zero values, retained learned priors improve score over uniform by20.00pp and17.50pp; both intervals exclude zero. Both heads contain useful match guidance even though they can misdirect short tactics.

For match score the interaction A−B−C+D is−12.50pp [−23.44,−2.19] for CNN and−17.19pp [−28.12,−5.94] for ResNet. On this score scale the joint benefit is subadditive; this is an output-intervention effect in fixed search, not an explanation of how learning produced it or evidence the combined baseline is inferior to either isolated head.

E−A is−4.06pp [−10.94,+2.50] for CNN and+7.81pp [0.00,+15.63] for ResNet. ResNet's positive point effects occur on both seeds, but each seed's interval spans zero. The primary pooled interval touches zero; the independent bootstrap stream yields [+0.31,+15.94]pp. That boundary is sensitive to Monte Carlo resampling and cannot support a confirmed strength gain. No scale sweep or result-dependent match selection was performed.

Color results, preserving the agent's actual X/O ownership rather than assuming the first post-opening move:

{table(['Checkpoint','Variant','Color','Score','Δ from A, pp [95%]'],color_rows)}

## Factorial contrasts across endpoints

{table(['Architecture','Endpoint','B−A, pp','C−A, pp','E−A, pp','A−B−C+D, pp'],contrasts)}

Full D−C and D−B contrasts remain in analysis.json. These identify effects of intervening on the frozen outputs under one search/opponent/budget; they do not measure training changes, causal reasons for high value MSE, or the optimal exploration constant.

## Value learning: separate targets and calibration

Training-replay predictions use full retained g6/g5 replay, not the earlier phase's2048-position probe. Prequential predictions use g5 on only new g6 games that it generated and had not trained on. No g7 exists, so g6 prequential MSE is unavailable. Realized self-play outcome and optimal game-theoretic WDL are different targets.

MSE [95% opening/ancestry interval]; zero columns give constant-zero MSE on the same data. Exact zero baselines are0.667 on solved300 and0.547 on deep128:

{table(['Checkpoint','G6 training MSE','G6 zero','G5 training MSE','G5→G6 prequential MSE','Prequential zero','Solved exact MSE','Deep exact MSE'],value_rows)}

All four fit retained outcomes better than zero. CNN42 g5 prequential MSE beats zero: excess−0.117 [−0.228,−0.024]. CNN314159 is tied: excess+0.003 [−0.083,+0.067]. Both ResNet prequential point estimates exceed zero (+0.070,+0.105), but their intervals span zero. ResNet's larger training-to-next-generation gaps are consistent with poor out-of-replay generalization; they also include changes in game distribution/outcomes and are not a controlled demonstration of memorization.

On balanced solved300, CNN is approximately at zero's error, while ResNet has higher errors and a positive side-to-move bias. Pooled ResNet−CNN MSE is+0.165 [+0.050,+0.285]. On deep128, which has no lost positions, it is−0.104 [−0.314,+0.066]. The suite composition reverses the point contrast; it is not evidence of a uniformly superior value function.

Solved300 prediction strata (calibration is the ten-bin count-weighted absolute mean prediction/target gap in [−1,+1] units):

{table(['Checkpoint','Loss MSE','Draw MSE','Win MSE','Mean value','Mean |value|','Share |value|>0.95','Calibration gap'],values_strata)}

ResNet's errors are concentrated in theoretical losses:1.303/1.715 versus zero's1.000, while its win errors0.694/0.499 beat zero's1.000. On exact solved rows, high-confidence (|v|>0.8) errors exceed lower-confidence errors in all four models. On deep128 that relationship reverses in three models, so confidence alone is not a universal error detector. Replay high-confidence predictions are substantially more accurate on training data; their held-out accuracy is model/distribution dependent. All fixed value bins and confidence/actor/ply/observed-terminal-distance strata are retained, including empty-bin denominators.

Training error is highest before ply8 (0.78–0.85 MSE) and generally lower later, but prequential curves are not uniformly monotonic. CNN prequential error on repeated mirror positions is higher than on nonrepeated positions; ResNet's direction differs. Novel-to-g5 prequential positions have CNN MSE0.743/0.860 versus seen0.684/0.544, while ResNet is1.037/0.926 versus seen0.952/0.957. These associations do not establish why these models show out-of-replay error. Terminal distance here means remaining plies in the observed imperfect game, not an oracle distance to victory. Per-opening descriptive errors are retained; unequal opening sizes and correlated histories limit inference.

## Outcome noise, mismatch and independent signs

{table(['Checkpoint','G6 examples','Unique exact boards','Conflicting repeated boards','Examples on conflicting boards','Empirical same-board variance floor','Seen prequential MSE','Novel prequential MSE'],noise)}

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

Run/controller: October12,2026,18:34:27–18:51:04 Eastern. Full primary phase864.38s; frozen analysis19.68s; fresh arithmetic22.72s; bounded independent model reproduction90.35s. Postlaunch lifecycle997.16s plus conservative prelaunch charge180s =1177.16s (19.62min), below3600s. The final reporting supplement is{s['reporting_seconds']:.2f}s; final integrity audit was under1s. Report generation and final-review accounting are recorded separately in resource-final.json. Drafting/publication wall time is not neural measured work.

The prelaunch charge covers two focused validation passes, the19.38s cost-only pilot, synthetic checks and hashing/freezes. The full-comparison pilot projected886.52s, or1595.74s with contingency. The fixed target was an upper30–45min planning range; no work was added to consume spare budget. Main worker CPU sums to{s['resource']['total_worker_cpu_seconds']:.2f}s, with three workers; summed worker wall{s['resource']['summed_worker_wall_seconds']:.2f}s and largest worker peak{s['resource']['largest_worker_peak_rss_mib']:.2f}MiB. No retries, new campaign or optimizer updates.

Evidence: [primary status](../research/alphazero_value_diagnostics/results/status.json), [lifecycle](../research/alphazero_value_diagnostics/results/lifecycle.json), [analysis](../research/alphazero_value_diagnostics/results/analysis.json), [report supplement](../research/alphazero_value_diagnostics/results/report-supplement.json), [independent arithmetic](../research/alphazero_value_diagnostics/results/independent-clean-recomputed.json), [independent rerun](../research/alphazero_value_diagnostics/results/independent-clean-model-rerun.json). Per-task identities, complete markers, exact visits, all matches and full value diagnostics remain in results/tasks/. The final SHA256 [evidence manifest](../research/alphazero_value_diagnostics/results/evidence-manifest.json) covers published evidence; checkpoints remain only in their original local output trees.

### Accessed checkpoint identities

{table(['Arm','Generation','Boundary SHA256'],checkpoints)}

All28 identities are retained in freeze.json and final-integrity-audit.json; only g0/g5/g6 were used. Remaining limitations: two seeds, one decision checkpoint, one reference opponent, one budget/c, reused development fixtures, unequal correlated replay families, missing g7, sparse exact/replay overlap, no full independent oracle re-solving, no multiplicity correction, and the retained access breach. None is resolved by a tactical improvement alone.
'''
    (REPO/'docs/phase4d4i-a-alphazero-value-diagnostics.md').write_text(text)
    write(RESULTS/'report-generation.json',dict(seconds=time.monotonic()-started,model_queries=0,training_updates=0))
    print('Final report generated from complete evidence')

if __name__=='__main__':main()
