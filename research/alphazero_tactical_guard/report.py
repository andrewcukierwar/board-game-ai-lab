"""Editorial generation solely from complete frozen evidence; no model queries."""
from .common import read,RESULTS,REPO


def pct(x):return f'{100*x:.3f}%'
def interval(c):return f"{100*c['difference']:+.3f} [{100*c['ci95'][0]:+.3f}, {100*c['ci95'][1]:+.3f}]"


def main():
    a=read(RESULTS/'analysis.json');execution=read(RESULTS/'execution.json');freeze=read(RESULTS/'freeze.json');validation=read(RESULTS/'validation.json')
    lines=['# Phase 4D.4J-A — AlphaZero tactical root guard','',
        f"**Playing-strength verdict: {a['strength_verdict']}.** The root guard fixes directly targeted one-ply decisions; actual strength is judged separately using the prospectively frozen paired match rule.",'',
        '| Question | Verdict |','|---|---|',
        '| Mechanical correctness | PASS: independent rules and original PUCT preservation |',
        f"| One-ply tactical reliability | {a['tactical_verdict']} |",
        f"| Deeper exact-value safety | {a['deep_verdict']} |",
        f"| Actual playing strength | {a['strength_verdict']} |",
        f"| Latency budget | {a['latency_verdict']} |",'',
        '## Problem, motivation and established methods','',
        'CNN remains preferred: H-C found no equal-compute ResNet advantage. I-A showed useful strategic information in learned policy and value: zero values raised CNN tactics but lost20pp match score; uniform-policy/zero-value PUCT solved tactical200 yet scored11.875%. I-B reduced held-out MSE without establishing preserved strength. This phase uses ORIGINAL H-C generation6 CNNs, not I-B treatment.', '',
        'Immediate-win detection and one-ply loss avoidance are conventional tactical safeguards. Existing Tactical MCTS (`MCTSAgent`) takes wins/filters unsafe moves before UCT; F1 neural MCTS filters root expansion and renormalizes allowed priors. Existing `tactical_value.py` also contains exact engine successor/reply checks. This experiment reuses immutable bitboard drop/winner rules and applies selection AFTER unchanged PUCT. It does not introduce a novel AI algorithm or replace learned search guidance.', '',
        '## Algorithm and implementation','',
        'Run256 original PUCT simulations, then inspect the ACTUAL immutable root and unmodified seven-column visits. Immediate winning actions take priority, selecting most visits then lowest physical column. With no win, preserve an already-safe original exactly; otherwise choose the most-visited safe action with the same deterministic restricted tie rule. If all legal moves allow a winning opponent reply, return original and record NO_SAFE_MOVE, without a theoretical-loss claim. A terminal draw successor is safe. Terminal roots are rejected. No recursion, model call, RNG draw, policy/value modification, tree change, visit rewrite, proof propagation or inference interface change.', '',
        'Versioned contracts: `connect4-tactical-root-guard-v1` and `connect4-guarded-decision-v1`. Every guarded decision stores original/final actions, legal/winning/safe/unsafe sets, winning reply witnesses, complete original visits, selected/zero visits, change/reason/no-safe flags, occupancy, candidate/reply counts and timing. Zero visits are not replaced with synthetic evidence.', '',
        '## Correctness and preservation of original search','',
        f"Prelaunch [validation](../research/alphazero_tactical_guard/results/validation.json): {validation['checks']}. Coverage: {validation['coverage']}. Random attempt cap160; {validation['random_nonterminal']} nonterminal legal samples, plus synthetic physical-column wins, horizontal/vertical/both diagonals, both colors, mirrors, multiple wins/safe moves, all-unsafe, full columns, terminal wins/draws, last-cell draw, zero visits and deterministic ties.", '',
        'A separately authored matrix rules engine scans directional cells rather than bit operations. Both checkpoint preservation tests compare disabled/enabled wrapper searches against canonical PUCT: exact actions before guard, visits, neural logits/priors/values, evaluation RNG states, board bytes, predictions, full learner/RNG/optimizer/replay state and weights. Model checks use6 positions across two seeds. All measured fixture pairs additionally assert the same raw search identities. No original source under games/ or live app path changed.', '',
        '## Checkpoint and execution provenance','',
        '| Seed | Original generation6 boundary SHA256 |','|---|---|']
    for e in freeze['checkpoints']:
        seed=42 if '42-cnn' in e['path'] else 314159
        lines.append(f"| {seed} | `{e['sha256']}` |")
    lines += ['', 'CNN identity `connect4-conv64-128-128-policy7-scalar-tanh-v2`,325896 parameters. Strict original loader validates full architecture/configuration/model state, original schedule, generation, original source/runtime/inference contract and checkpoint hash. Checkpoints are read-only; no optimizer/replay modification. Runtime and complete execution-source identities: [freeze.json](../research/alphazero_tactical_guard/results/freeze.json). Per-seed before/after states and weight hashes: [tasks](../research/alphazero_tactical_guard/results/tasks/).', '',
        '## Prospective design and execution','',
        f"Prospective pushed commit `{execution['prospective_commit']}`; base `37ec500ce70bd2bbe63ac6578cd7bf576004def6`; dedicated `phase4d4j-a-tactical-root-guard` branch/worktree. Protocol [DESIGN.md](../research/alphazero_tactical_guard/DESIGN.md), [design.json](../research/alphazero_tactical_guard/results/design.json), independent prelaunch [review](../research/alphazero_tactical_guard/independent-prelaunch-review.md). All freezes preceded measured work.", '',
        'Both original fixed CNN seeds; PUCT256/c1.41/no root noise; unchanged values/priors/terminal handling. Fixtures have identical board and search RNG identities; matches have same opening/color/role streams and reference Negamax3 configuration. Diverging trajectories may consume later random numbers differently. All40 development base opening families, both colors. No secondary/ResNet evaluation, chosen before freeze. No sealed input was accessed; the prior I-A accidental sealed-text incident remains disclosed history.', '',
        f"Execution: {execution['fixture_searches']} fixture searches, {execution['fixture_paired_records']} paired records, {execution['matches']} matches, two completed seed tasks, zero retries/training/self-play/abandoned games. Durable logs and exclusive attempt identity remain in ignored experiment-output/phase4d4j-a/; raw decisions/results are tracked under results/tasks/.", '',
        '## One-ply tactical results','',
        'Perfect targeted tactical performance is principally a correctness check, not independent evidence of stronger gameplay.', '',
        '| Seed | Variant | Immediate wins /100 | Unique safe /100 | Avoidable errors /200 | Missed wins /200 |','|---|---|---:|---:|---:|---:|']
    tactical_notes=[]
    for seed,endpoint in a['seeds'].items():
        t=endpoint['tactical']
        for v in ('control','guarded'):
            lines.append(f"| {seed} | {v} | {t['categories']['unique_win']['summary'][v]['correct']['count']} | {t['categories']['unique_safe']['summary'][v]['correct']['count']} | {t['summary'][v]['avoidable']['count']} | {t['summary'][v]['missed_win']['count']} |")
        tactical_notes += ['',f"Seed{seed}: {t['changed']} changes, {t['fixes']} baseline mistakes fixed, {t['introduced']} new mistakes, {t['baseline_correct']} already-correct decisions, {t['baseline_incorrect_with_safe']} incorrect despite safe alternative, {t['no_safe']} no-safe rows. Reasons: {t['reasons']}. Category contrasts pp [95%]: win {interval(t['categories']['unique_win']['contrast'])}; safe {interval(t['categories']['unique_safe']['contrast'])}.",'']
    lines += tactical_notes
    lines += ['## Exact solved300 and deep128 decisions','',
        'Existing COMPLETE legal-action oracle vectors are used without invented labels. Solved300 has100 each win/draw/loss; deep128 has70 wins,58 draws,0 losses. All actions in theoretically lost rows preserve WDL automatically; that100% stratum says nothing about good gameplay. Optimal-action and WDL-preserving accuracy coincide for these WDL vectors.', '',
        '| Seed | Suite | Variant | Optimal/WDL-preserving | Avoidable mistakes | W→D | W→L | D→L |','|---|---|---|---:|---:|---:|---:|---:|']
    exact_notes=[]
    for seed,e in a['seeds'].items():
        for suite in ('solved','deep'):
            q=e[suite]
            for v in ('control','guarded'):
                m=q['summary'][v]
                lines.append(f"| {seed} | {suite} | {v} | {pct(m['correct']['mean'])} | {m['mistake']['count']} | {m['win_draw']['count']} | {m['win_loss']['count']} | {m['draw_loss']['count']} |")
            exact_notes += ['',f"Seed{seed} {suite}: accuracy difference {interval(q['contrasts']['correct'])}pp; {q['changed']} changed decisions, reasons {q['reasons']}; selected exact-value differences {q['value_differences']}; regressions {q['regressions']}.",'']
    lines += exact_notes
    lines += ['All class denominators, change IDs/reasons and original/guarded exact labels: [analysis.json](../research/alphazero_tactical_guard/results/analysis.json) and raw suite rows. No observed downgrade is stronger than a claimed absence of unexplained regressions within these development boards; it does not establish a universal strategic guarantee.', '',
        '## Actual playing strength — principal outcome','',
        '| Seed | Variant | W/D/L | Score | X score | O score |','|---|---|---|---:|---:|---:|']
    match_notes=[]
    for seed,e in a['seeds'].items():
        for v,m in e['matches']['summary'].items():
            d=m['wdl'];lines.append(f"| {seed} | {v} | {d.get('win',0)}/{d.get('draw',0)}/{d.get('loss',0)} | {pct(m['score'])} | {pct(m['colors']['0'])} | {pct(m['colors']['1'])} |")
        match_notes += ['',f"Seed{seed} guarded−control: {interval(e['matches']['contrasts']['score'])} percentage points.",'']
    lines += match_notes
    lines += [f"Equal-weight pooled guarded−control: **{interval(a['pooled']['matches']['score'])}pp**. Frozen rule: PASS requires>=+5pp, both seeds nonnegative, pooled95% lower>0; FAIL requires<=−5pp and upper<0; otherwise INCONCLUSIVE. Result: **{a['strength_verdict']}**.", '',
        '10000 prospective paired cluster percentile resamples, PCG64 seed4410903: whole40 opening families retain both colors and both fixed seeds together. Fixtures retain ancestry/reflection clusters and row weighting. Intervals condition on these networks/development families and omit retraining-seed variability; no multiplicity correction. No favorable seed/checkpoint chosen. All per-family/color summaries and per-seed/pooled intervals are machine readable.', '',
        '## Guard activation and action changes','']
    for seed,e in a['seeds'].items():lines += [f"Seed{seed} matches: `{e['matches']['guard']}`",'']
    lines += ['IMMEDIATE_WIN can activate without changing an already-winning choice. NO_SAFE_MOVE is recorded, not an escape or theoretical proof. All changed selections were independently reclassified; changes fix local omissions/unsafe choices, not necessarily all deeper mistakes. Descriptive game-activation/outcome counts do not identify whether an activation caused a win; paired intervention matches provide the causal contrast.', '',
        '## Performance and latency','',
        '| Measure | Mean | Median | p95 | Worst |','|---|---:|---:|---:|---:|']
    for metric in ('search_seconds','classification_seconds','guard_seconds','total_seconds'):
        m=a['latency'][metric];lines.append(f"| {metric}, ms | {m['mean']*1000:.5f} | {m['median']*1000:.5f} | {m['p95']*1000:.5f} | {m['worst']*1000:.5f} |")
    m=a['latency']['relative_overhead'];lines += ['',f"Mean guard/search overhead {pct(m['mean'])}, median {pct(m['median'])}, p95 {pct(m['p95'])}, worst {pct(m['worst'])}. Practical budget p95<=1ms,worst<=5ms,mean relative<=5%: {a['latency_verdict']}. Timing pools guarded fixture/game decisions (not independent samples). Whole guard clock ends before final diagnostic object assembly; total-call clock includes assembly. Root bitboard adaptation occurs identically before timed search in both variants.", '',
        f"Candidate counts: {a['latency']['candidates']}; opponent replies: {a['latency']['opponent_replies']}. Occupancy timing bins0–6,7–13,...: {a['latency']['occupancy']}.", '',
        'One-thread deterministic CPU on Apple M4, sequential workers; exact runtime in freeze. No competing neural/training workload was found at preflight; ordinary GUI/OS load remains, so wall-clock outliers are observed system-dependent measurements. Per-task CPU/wall/RSS are in identity.json. No guard neural inference.', '',
        '## Independent review and historical preservation','',
        'Independent reviewer authored the matrix rules/record checker and prelaunch review without writing the guard or analyzer. Saved-record verification reproduces headlines and frozen bootstrap stream, with separate-stream sensitivity, checks every fixture pair/all own match decisions, complete visits/labels/original selections, colors/results/openings and latency. This is same-host verification, not external replication or a fresh full-game oracle solve. See [independent-verification.json](../research/alphazero_tactical_guard/results/independent-verification.json) and [results review](../research/alphazero_tactical_guard/independent-results-review.md).', '',
        'All150 official campaign files remain byte-identical; required G/H/I checkpoints/reports/data/results and actual canonical source/fixture/app paths are protected by nonempty hash-only inventories. [preservation.json](../research/alphazero_tactical_guard/results/preservation.json) records every scope. No historical file, original checkpoint, deployment configuration, API, public agent or live app was changed. Only the new package/report are committed; no oversized model is committed.', '',
        '## Resources and recommendation','',
        f"Cost-only pilot12.54s:12 roots/4 games, only times retained; primary projection331.33s. Prelaunch validation {validation['seconds']:.2f}s final successful pass, independent review and all preparation charged from actual task origin. Measured phases: {execution['phases']}. Execution finish charge {execution['charged_seconds']:.2f}s; final review/publication accounting is recorded separately in results/final-audit.json. Absolute total limit2700s includes ALL work, no artificial padding to20–30min. No measured retry/additional experiment.", '']
    if a['strength_verdict']=='PASS' and a['deep_verdict'].startswith('PASS') and a['latency_verdict']=='PASS':
        lines += ['**Recommend a separate production-integration phase.** These fixed CNNs are stronger against the declared development Negamax3 endpoint, with reliable targeted tactics, no observed deeper WDL regression and acceptable measured overhead. This does not establish broader opponent/unseen-position strength. Integration remains separate; nothing is deployed here.']
    elif a['strength_verdict']=='FAIL':
        lines += ['**Do not adopt the guard on this strength evidence.** Investigate which changed match continuations lose useful strategic options, using retained evidence and a separately frozen stronger-search comparison in a future phase. No follow-up is launched.']
    else:
        lines += ['**Tactics improved; stronger gameplay remains inconclusive.** Keep the public baseline unchanged. The next highest-value research step is a separately preregistered, larger paired opening/opponent evaluation of this simple guard, sized for the observed effect and preserving both seeds, rather than another learning intervention. No follow-up or integration is implemented.']
    (REPO/'docs/phase4d4j-a-alphazero-tactical-guard.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
