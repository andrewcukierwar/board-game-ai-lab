"""Render the fixed endpoint report from complete saved evidence; no model queries."""
from pathlib import Path
from .common import RESULTS, ROOT, REPO, read, write, sha


def fmt(x):return f'{x:.4f}'
def effect(x,scale=1):return f'{x["mean"]*scale:+.4f} [{x["ci95"][0]*scale:+.4f}, {x["ci95"][1]*scale:+.4f}]'


def main():
    a=read(RESULTS/'analysis.json');d=read(RESULTS/'design.json');r=read(RESULTS/'resources.json')
    independent=read(RESULTS/'independent-verification.json')
    m=read(RESULTS/'membership.json');n=d['updates_per_arm']
    content=['# Phase 4D.4I-B — Value prediction regularization','',
        '**'+('Moderate explicit value regularization improved the primary endpoint with qualified supporting safeguards.' if a['decision']=='PROMISING_FOR_NEXT_VALIDATION' else 'This experiment does not demonstrate improved held-out outcome prediction with preserved search guidance.')+'**','',
        'The final milestone was selected prospectively, not from the measured curves. Both seeds are reported separately. The treatment adds `0.2 * mean(predicted_value ** 2)` to unchanged policy CE and realized-outcome value MSE. The added gradient reaches the shared trunk; this is full-network objective regularization, not a value-head-only change. AdamW settings, architecture, target definitions and training data are unchanged.','',
        'Prospective protocol: [DESIGN.md](../research/alphazero_value_regularization/DESIGN.md). Fixed machine-readable design/membership/source-runtime contracts: [design.json](../research/alphazero_value_regularization/results/design.json), [membership.json](../research/alphazero_value_regularization/results/membership.json), [freeze.json](../research/alphazero_value_regularization/results/freeze.json). Independent prelaunch review: [review](../research/alphazero_value_regularization/independent-prelaunch-review.md). Final independent results review is recorded beside this report after review of complete evidence.','',
        f'Prospective pushed commit: `{r["prospective_commit"]}`. Base: `{d["base_commit"]}`. Milestones: {d["milestones"]} additional optimizer updates per arm/seed. No self-play games, additional arms, sweeps, generations, merge, deployment or promotion.','',
        '## Why increasing AdamW decay is a weak direct intervention','',
        'At LR3e-4, decay-only multiplication is `(1 − lr*wd)^U`. Changing decay1e-4→1e-3 yields only about2.7e-7 additional shrink per step. At2048 steps, the two cumulative shrinkages are0.0061438% and0.0614211%, an additional0.0552807% relative shrink. These figures isolate direct decoupled decay, not all possible optimizer/training feedback. Biases would not decay. The explicit penalty instead adds derivative `0.4*v/N` at each predicted output; useful values may also be suppressed, so magnitude reduction alone cannot establish success.','',
        '## Frozen data, novelty and uncertainty','',
        '| Seed | G5 train games/positions | G6 heldout games/positions | Exact overlap | Mirror overlap | Novel positions/games/families |','|---|---|---|---|---|---|']
    for seed in (42,314159):
        x=m[str(seed)]['audit'];content.append(f'| {seed} | {x["train_games"]}/{x["train_positions"]} | {x["heldout_games"]}/{x["heldout_positions"]} | {x["exact_overlap"]} | {x["mirror_overlap"]} | {x["novel_positions"]}/{x["novel_games"]}/{x["novel_families"]} |')
    content+=['','Train only replay retained at the original generation5 checkpoint (generations1–5). Heldout is precisely each seed’s144 frozen generation6 new games, which never enter training. Independent history reconstruction found zero label errors. Exact and mirror-equivalent overlaps are audited against every retained training position. Primary membership excludes mirror overlap. Novel repeated heldout boards are retained with position weighting; they are not independent observations.','',
        'Intervals are paired10000-resample95% percentile bootstrap, RNG4411403. Entire opening-family clusters are resampled, preserving complete-game rows and paired colors; fixed development ancestry/reflection clusters are resampled for position suites. Means retain row weighting. Each seed’s own game distribution is evaluated separately. Intervals condition on two fixed networks and reused development families, exclude training-seed population uncertainty and have no multiplicity correction. The final novel-outcome MSE is the only primary endpoint.','',
        '## Value prediction across all declared milestones','',
        '| Seed | Updates | Arm | Train MSE | All heldout MSE | Novel MSE | Novel zero MSE | Novel−train gap | Novel bias | Mean abs value | Saturated >.95 | Calibration |','|---|---|---|---|---|---|---|---|---|---|---|---|']
    for seed,s in a['seeds'].items():
        entries=[(0,'shared initial',s['initial'])]+[(int(u),arm,data) for u,ms in s['milestones'].items() for arm,data in ms.items()]
        for u,arm,x in sorted(entries,key=lambda x:(x[0],x[1])):
            v=x['values'];nov=v['novel']
            content.append(f'| {seed} | {u} | {arm} | {fmt(v["training"]["mse"])} | {fmt(v["heldout"]["mse"])} | {fmt(nov["mse"])} | {fmt(nov["zero_mse"])} | {fmt(v["generalization_gap"])} | {fmt(nov["bias"])} | {fmt(nov["mean_abs_value"])} | {100*nov["saturation"]:.2f}% | {fmt(nov["calibration"])} |')
    content+=['','Zero predicts0 everywhere; its MSE is the empirical mean squared target, not a favorable default due to class balancing. All datasets’ zero excess, means, fixed calibration bins, magnitude, saturation and per-target strata are in analysis.json. Realized imperfect-play outcomes and exact game-theoretic labels remain distinct. The gaps include policy/data distribution differences and cannot identify memorization alone.','',
        '| Seed | Updates | Novel treatment−control MSE [95%] | All-heldout Δ [95%] | Training Δ [95%] |','|---|---|---|---|---|']
    for seed,s in a['seeds'].items():
        for u,c in s['contrasts'].items():content.append(f'| {seed} | {u} | {effect(c["novel_mse"])} | {effect(c["heldout_mse"])} | {effect(c["training_mse"])} |')
    content+=['','The prespecified practical margin is at least0.02 absolute novel MSE reduction in **each seed**, with each95% interval upper bound below0.','']
    for seed,s in a['seeds'].items():content.append(f'Seed{seed}: primary rule **{"PASS" if s["primary_pass"] else "NOT MET"}**; final novel MSE difference {effect(s["contrasts"][str(n)]["novel_mse"])}.')
    content+=['','## Search guidance, exact labels and playing strength','',
        '| Seed | Updates | Arm | Raw tactical accuracy | PUCT tactical accuracy | Solved300 exact value MSE | Solved PUCT accuracy | Deep128 mistakes | Negamax3 W/D/L | Score |','|---|---|---|---|---|---|---|---|---|---|']
    for seed,s in a['seeds'].items():
        entries=[(0,'shared initial',s['initial'])]+[(int(u),arm,data) for u,ms in s['milestones'].items() for arm,data in ms.items()]
        for u,arm,x in sorted(entries,key=lambda x:(x[0],x[1])):
            p=x['positions'];st=x['strength'];wdl=st['wdl']
            content.append(f'| {seed} | {u} | {arm} | {100*p["tactical"]["raw_accuracy"]:.2f}% | {100*p["tactical"]["puct_accuracy"]:.2f}% | {fmt(p["solved"]["value"]["mse"])} | {100*p["solved"]["puct_accuracy"]:.2f}% | {p["deep"]["mistakes"]}/128 | {wdl.get("win",0)}/{wdl.get("draw",0)}/{wdl.get("loss",0)} | {100*st["score"]:.2f}% |')
    content+=['','Solved300 has100 each exact win/draw/loss, with zero MSE2/3; deep128 has70 wins,58 draws and no losses, with zero MSE70/128. Exact value strata and W→D/W→L/D→L downgrade counts are retained for each milestone in analysis.json. All actions on an exact lost board preserve WDL automatically, so solved accuracy on those rows is not evidence of useful tactics. Raw policy and PUCT are separate quantities. All milestones use unattenuated learned values, PUCT256 and frozen Negamax3, on all40 base families with both colors (80 games/task).','',
        '| Seed | Final comparator | Tactical category differences pp [95%] | Deep mistake difference pp [95%] | Match-score difference pp [95%] | Safeguards |','|---|---|---|---|---|---|']
    for seed,s in a['seeds'].items():
        for comparator,c in [('control',s['contrasts'][str(n)]),('initial',s['treatment_minus_initial'])]:
            cats='; '.join(k.replace('tactical_puct_','')+': '+effect(v,100) for k,v in c.items() if k.startswith('tactical_puct_') and k!='tactical_puct_accuracy')
            gates='; '.join(k+': '+v for k,v in s['safeguards'][comparator].items())
            content.append(f'| {seed} | {comparator} | {cats} | {effect(c["deep_mistake"],100)} | {effect(c["strength"],100)} | {gates} |')
    content+=['','Tactical pass margins are point>=−3pp and lower>=−10pp; regression requires point<=−5pp and upper<0. Deep regression requires mistake increase>=5pp and lower>0. Supporting strength pass requires point>=−5pp and lower>=−15pp; regression requires point<=−5pp and upper<0. These compare treatment with both final control and initial state in both seeds. Strength evidence from40 pairs is underpowered for small differences; supportive intervals cannot establish noninferiority or prove general playing-strength preservation. Favorable MSE cannot override search regression.','',
        '## Correctness, independent review and preservation','',
        'Focused correctness tests: [validation.json](../research/alphazero_value_regularization/results/validation.json). Both seeds’ control loss and entire optimizer update matched the historical trainer exactly. Analytic penalty/gradient, unchanged policy CE, finite gradients and nonzero shared-trunk penalty gradients passed. Both arms started with identical model/optimizer/full state, saw identical sampled addresses and reflections, and matched fresh-process continuation exactly. Strict rejection checks cover altered identities and overwrite paths. Every evaluation checked complete learner/RNG preservation. New checkpoints use an explicitly versioned fixed-replay contract; original H-C checkpoints were strictly loaded read-only.','',
        'Independent arithmetic, cluster intervals, membership/action/result identities and all task completion markers: [independent-verification.json](../research/alphazero_value_regularization/results/independent-verification.json). This same-host saved-evidence audit makes no new model query and is not an external replication or fresh exact-game solver proof. Final independent review separately examines conclusions and resource limits.','',
        'All150 official campaign files and required historical evidence inventories passed [preservation.json](../research/alphazero_value_regularization/results/preservation.json). The Phase4D.4H architecture-only phase has no experiment-output tree; its actual source/evidence/report paths are inventoried. No historical files or frozen training sources were changed.','',
        'The [prior sealed-access incident](../research/alphazero_value_diagnostics/integrity-resolution.md) remains disclosed historical research provenance. No sealed fixture content or raw exposed response was opened/copied for this phase. Hash-only preservation checks do not inspect labels. This study measures novel positions within known development/self-play distributions, not sealed independent generalization.','',
        '## Resource accounting and recommendation','',
        f'Cost-only pilot: {read(RESULTS/"pilot.json")["wall_seconds"]:.2f}s. Focused successful validation: {read(RESULTS/"validation.json")["wall_seconds"]:.2f}s. Total lifecycle through analysis and independent verification: {r["wall_seconds"]:.2f}s, plus conservative240s prelaunch charge = {r["charged_seconds"]:.2f}s ({r["charged_seconds"]/60:.2f}min), below2700s. Main work was capped at2360s;100s was reserved for postprocessing/cleanup. Work completed8192 optimizer updates,6280 fixture roots and800 matched games. There were no measured retries, additional arms or self-play collection. Detailed per-task CPU/RSS/timings and phase times are retained. Publication/editorial time is not additional neural work.','',
        f'Prespecified combined decision: **{a["decision"]}**. Primary pass: {a["primary_pass"]}; demonstrated safeguard regression: {a["regression"]}; unresolved safeguards: {a["unresolved_safeguards"]}.','']
    if a['decision']=='PROMISING_FOR_NEXT_VALIDATION':
        content+=['Recommend retaining the CNN baseline and testing the same single regularization hypothesis on a new prospectively grouped heldout dataset in a separately authorized phase. These two seeds and development games do not justify model promotion or a stronger strength claim.']
    else:
        content+=['Retain the CNN baseline. Recommend one next value-learning hypothesis: weight only the realized-outcome value loss inversely by mirror-equivalent training-position frequency, giving each board equal total value-loss weight while leaving policy CE unchanged. Repeated positions with noisy outcomes may otherwise dominate fixed replay. This is a distinct prospective hypothesis, not a conclusion about its effectiveness; no follow-up is launched, and this experiment is not expanded.']
    content+=['','Full raw rows/visits/games, per-seed calibration/target strata and independent intervals remain in the new research package. No best milestone is selected retrospectively.']
    target=REPO/'docs/phase4d4i-b-value-regularization.md'
    target.write_text('\n'.join(content)+'\n')
    write(RESULTS/'report-generation.json',dict(report_sha256=sha(target),analysis_sha256=sha(RESULTS/'analysis.json'),measured_queries=0))
    print('Final report rendered from complete evidence',flush=True)

if __name__=='__main__':main()
