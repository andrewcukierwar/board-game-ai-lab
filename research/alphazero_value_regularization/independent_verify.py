"""Independent saved-evidence arithmetic; never trains or queries a model.

Authored by the isolated reviewer. Uses explicit new-package task paths and the
permitted development fixtures only. Bootstrap independently constructs sums
and denominators from raw paired records; no analyzer function is imported.
"""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import time

import numpy as np

from .common import RESULTS, read, write


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'Independent arithmetic mismatch: {name}: {actual} != {expected}')


def mean(rows, function):
    if not rows:
        raise ValueError('Empty endpoint')
    return sum(function(r) for r in rows) / len(rows)


def paired(a, b, function, *, key='id', cluster='cluster'):
    """a-b, resampling every row of a selected family together."""
    if len(a) != len(b) or [r[key] for r in a] != [r[key] for r in b]:
        raise ValueError('Paired endpoint memberships differ')
    groups = defaultdict(list)
    for x, y in zip(a, b):
        if x[cluster] != y[cluster]:
            raise ValueError('Paired cluster identity differs')
        groups[x[cluster]].append(function(x) - function(y))
    order = sorted(groups)
    sums = np.array([sum(groups[k]) for k in order], dtype=np.float64)
    counts = np.array([len(groups[k]) for k in order], dtype=np.int64)
    rng = np.random.default_rng(4411403)
    # Independent bounded chunks avoid a huge 10000 x training-family array.
    boot = []
    for begin in range(0, 10000, 250):
        selected = rng.integers(0, len(order), size=(min(250, 10000-begin), len(order)))
        boot.extend((sums[selected].sum(1) / counts[selected].sum(1)).tolist())
    lo, hi = np.quantile(boot, [.025, .975])
    return dict(difference=sum(sums)/sum(counts), lower=float(lo), upper=float(hi),
                rows=len(a), clusters=len(order), replicates=10000, rng_seed=4411403)


def value_summary(rows):
    values = [r['value'] for r in rows]
    for r in rows:
        if r['target'] not in (-1, 0, 1) or not math.isfinite(r['value']) or abs(r['value']) > 1:
            raise ValueError('Invalid target or raw value')
        close(r['se'], (r['value']-r['target'])**2, 'per-position squared error')
    bins = defaultdict(list)
    for r in rows:
        bins[min(9, int((r['value']+1)/2*10))].append(r)
    reliability = sum(len(rs)*abs(mean(rs, lambda r:r['value'])-mean(rs, lambda r:r['target']))
                      for rs in bins.values()) / len(rows)
    return dict(positions=len(rows), value_mse=mean(rows, lambda r:(r['value']-r['target'])**2),
                constant_zero_value_mse=mean(rows, lambda r:r['target']**2),
                mean_value=sum(values)/len(values), mean_target=mean(rows, lambda r:r['target']),
                bias=mean(rows, lambda r:r['value']-r['target']),
                mean_abs_value=sum(abs(v) for v in values)/len(values),
                saturated_share=sum(abs(v)>.95 for v in values)/len(values),
                calibration_error=reliability)


def inspect_task(path, seed, membership, fixtures, opening_ids):
    complete = read(path/'complete.json')
    if not complete['complete'] or not complete['evaluation_rng_neutral']:
        raise ValueError('Incomplete or mutating evaluation task')
    values = read(path/'values.json')
    out = dict(values={})
    for name, expected in (('training', membership['training']), ('heldout', membership['heldout']),
                           ('novel', [r for r in membership['heldout'] if not r['mirror_overlap']])):
        rows = values[name]['rows']
        if [{k:v for k,v in r.items() if k not in ('value','se')} for r in rows] != expected:
            raise ValueError('Evaluation membership/target/novelty differs from prospective membership')
        stats = value_summary(rows)
        for k,v in stats.items():
            observed = values[name]['summary']['value_calibration']['error'] if k=='calibration_error' else values[name]['summary'][k]
            close(observed, v, f'{path.name}/{name}/{k}')
        out['values'][name] = stats
    positions = read(path/'positions.json')
    out['positions'] = {}
    for suite in ('tactical','solved','deep'):
        rows, expected = positions[suite], fixtures[suite]
        if [r['id'] for r in rows] != [r['id'] for r in expected]:
            raise ValueError('Development position membership differs')
        for r,e in zip(rows, expected):
            s = r['search']
            if s['id'] != e['id'] or s['cluster'] != e['cluster'] or sum(s['visits']) != 256:
                raise ValueError('Search identities or simulation count differ')
            correct = lambda a: int(a == e['expected_action']) if suite=='tactical' else int(a in e['optimal_actions'])
            close(s['correct'], correct(s['action']), 'search correctness')
            close(r['policy_correct'] if suite=='tactical' else r['accuracy'], correct(r['policy_action']), 'raw policy correctness')
            if suite != 'tactical':
                close(r['se'], (r['value']-e['value'])**2, 'exact WDL squared error')
                close(s['mistake'], int(e['action_values'][str(s['action'])] < e['value']), 'exact WDL downgrade')
        out['positions'][suite] = dict(rows=len(rows), raw_accuracy=mean(rows, lambda r:r['policy_correct'] if suite=='tactical' else r['accuracy']),
                                      search_accuracy=mean(rows, lambda r:r['search']['correct']))
        if suite!='tactical':
            out['positions'][suite].update(exact_value_mse=mean(rows,lambda r:r['se']),
                                           zero_mse=mean(expected,lambda r:r['value']**2),
                                           search_mistakes=mean(rows,lambda r:r['search']['mistake']))
    matches = read(path/'matches.json')['records']
    if len(matches)!=80 or len({(r['opening_id'],r['game_index']) for r in matches})!=80:
        raise ValueError('Match memberships/counts differ')
    if [(r['opening_id'],r['game_index']) for r in matches] != [(i,c) for i in opening_ids for c in (0,1)]:
        raise ValueError('Match opening order differs from prospective design')
    for r in matches:
        if r['abandoned'] or r['game_index']!=r['agent_color']:
            raise ValueError('Invalid match completeness/color')
        expected_result = 'draw' if r['winner']==-1 else ('win' if r['winner']==r['agent_color'] else 'loss')
        if r['result']!=expected_result:
            raise ValueError('Result/winner/color differs')
        close(r['score'], {'win':1.,'draw':.5,'loss':0.}[r['result']], 'match score')
        r['id'] = f"{r['opening_id']}:{r['game_index']}"
        r['cluster'] = r['family']
    out['matches'] = dict(score=mean(matches,lambda r:r['score']),
                          wins=sum(r['result']=='win' for r in matches),
                          draws=sum(r['result']=='draw' for r in matches),
                          losses=sum(r['result']=='loss' for r in matches))
    return out, values, positions, matches


def verify(root):
    started = time.perf_counter()
    design, membership = read(RESULTS/'design.json'), read(RESULTS/'membership.json')
    fixtures = read(RESULTS.parent.parent/'alphazero_resnet_selfplay/fixtures.json')
    tasks_root = root/'tasks' if (root/'tasks').is_dir() else root
    tasks, raw, contrasts = {}, {}, {}
    for seed in design['seeds']:
        m=membership[str(seed)]
        exact={r['exact_key'] for r in m['training']};mirrors={r['mirror_key'] for r in m['training']}
        if any(int(r['game'].split(':')[0])>5 for r in m['training']) or len({r['game'] for r in m['heldout']})!=144:
            raise ValueError('Training/heldout generation or game count differs')
        for r in m['heldout']:
            if (r['exact_overlap'],r['mirror_overlap'])!=(r['exact_key'] in exact,r['mirror_key'] in mirrors) or not r['game'].startswith('6:'):
                raise ValueError('Independent exact/mirror overlap audit failed')
        records=read(root/'records'/f'{seed}-initial.json')
        if records['control']['state_sha256']!=records['treatment']['state_sha256'] or records['control']['optimizer_sha256']!=records['treatment']['optimizer_sha256']:
            raise ValueError('Initial full computational states differ')
        prior={arm:records[arm]['state_sha256'] for arm in ('control','treatment')}
        for previous,u in zip(design['milestones'],design['milestones'][1:]):
            pair=[]
            for arm in ('control','treatment'):
                r=read(root/'records'/f'{seed}-{arm}-{u}-train.json')
                if not r['complete'] or (r['seed'],r['arm'],r['updates'],r['previous'],r['steps'])!=(seed,arm,u,previous,u-previous):
                    raise ValueError('Declared training segment differs')
                if r['before_state']!=prior[arm] or len(r['trace'])!=u or len(r['losses'])!=u-previous or r['counters']['trainer_steps']!=720+u or r['counters']['completed_generations']!=5:
                    raise ValueError('Training state continuity/trace/counters differ')
                for loss in r['losses']:
                    if not math.isclose(loss['loss'],loss['ce']+loss['mse']+loss['penalty'],rel_tol=0,abs_tol=1e-5):raise ValueError('Stored loss components disagree')
                    if not 0<=loss['penalty']<=.2000001 or (arm=='control' and loss['penalty']!=0):raise ValueError('Stored regularization differs')
                prior[arm]=r['after_state'];pair.append(r)
            if pair[0]['trace']!=pair[1]['trace'] or pair[0]['replay_sha256']!=pair[1]['replay_sha256'] or pair[0]['counters']!=pair[1]['counters']:
                raise ValueError('Paired batches/reflections/replay/counters differ')
        for label in [f'{seed}-initial-0']+[f'{seed}-{arm}-{u}' for u in design['milestones'][1:] for arm in ('control','treatment')]:
            task, *evidence = inspect_task(tasks_root/label,seed,membership[str(seed)],fixtures,design['evaluation']['openings'])
            if label.endswith('initial-0'):
                expected_state=records['control']['state_sha256']
            else:
                _,arm,u=label.split('-');expected_state=read(root/'records'/f'{seed}-{arm}-{u}-train.json')['after_state']
            if read(tasks_root/label/'complete.json')['state_sha256']!=expected_state:raise ValueError('Evaluation checkpoint state differs from recorded training state')
            tasks[label], raw[label] = task,evidence
        baseline = raw[f'{seed}-initial-0']
        for u in design['milestones'][1:]:
            control, treatment = raw[f'{seed}-control-{u}'], raw[f'{seed}-treatment-{u}']
            for comparator, other in (('control',control),('initial',baseline)):
                tag = f'{seed}-{u}-treatment-minus-{comparator}'
                c = {}
                for dataset in ('training','heldout','novel'):
                    c[dataset+'_mse'] = paired(treatment[0][dataset]['rows'], other[0][dataset]['rows'],lambda r:(r['value']-r['target'])**2)
                c['match_score'] = paired(treatment[2],other[2],lambda r:r['score'])
                for suite in ('tactical','solved','deep'):
                    a,b=treatment[1][suite],other[1][suite]
                    c[suite+'_search_accuracy']=paired(a,b,lambda r:r['search']['correct'])
                    c[suite+'_raw_accuracy']=paired(a,b,lambda r:r['policy_correct'] if suite=='tactical' else r['accuracy'])
                    if suite=='tactical':
                        for category in sorted({r['category'] for r in a}):
                            aa,bb=[r for r in a if r['category']==category],[r for r in b if r['category']==category]
                            c['tactical_'+category+'_search']=paired(aa,bb,lambda r:r['search']['correct'])
                            c['tactical_'+category+'_raw']=paired(aa,bb,lambda r:r['policy_correct'])
                    else:
                        c[suite+'_exact_mse']=paired(a,b,lambda r:r['se'])
                        c[suite+'_mistakes']=paired(a,b,lambda r:r['search']['mistake'])
                contrasts[tag]=c
    analysis=read(RESULTS/'analysis.json')
    checked=0
    primary_passes=[]
    independent_gates=[]
    for seed in design['seeds']:
        reported=analysis['seeds'][str(seed)]
        for label in [f'{seed}-initial-0']+[f'{seed}-{arm}-{u}' for u in design['milestones'][1:] for arm in ('control','treatment')]:
            task=tasks[label]
            if label.endswith('initial-0'):
                summary=reported['initial']
            else:
                _,arm,u=label.split('-');summary=reported['milestones'][u][arm]
            for dataset,stats in task['values'].items():
                aliases={'value_mse':'mse','constant_zero_value_mse':'zero_mse','saturated_share':'saturation','calibration_error':'calibration'}
                for k,v in stats.items():
                    close(summary['values'][dataset][aliases.get(k,k)],v,f'reported {label}/{dataset}/{k}');checked+=1
            for suite,stats in task['positions'].items():
                for k in ('raw_accuracy','search_accuracy'):
                    close(summary['positions'][suite]['puct_accuracy' if k=='search_accuracy' else k],stats[k],f'reported {label}/{suite}/{k}');checked+=1
                if suite!='tactical':
                    close(summary['positions'][suite]['value']['mse'],stats['exact_value_mse'],'reported exact mse');checked+=1
                    close(summary['positions'][suite]['value']['zero_mse'],stats['zero_mse'],'reported exact zero mse');checked+=1
                    close(summary['positions'][suite]['mistakes']/stats['rows'],stats['search_mistakes'],'reported exact mistake');checked+=1
            close(summary['strength']['score'],task['matches']['score'],'reported match score');checked+=1
        for u in design['milestones'][1:]:
            for comparator in ('control','initial'):
                if comparator=='initial' and u!=design['updates_per_arm']:continue
                comparison=reported['contrasts'][str(u)] if comparator=='control' else reported['treatment_minus_initial']
                ours=contrasts[f'{seed}-{u}-treatment-minus-{comparator}']
                for k,stats in ours.items():
                    name=k.replace('_search_accuracy','_puct_accuracy').replace('_exact_mse','_value_mse').replace('_mistakes','_mistake')
                    if k=='match_score':name='strength'
                    elif k.startswith('tactical_') and (k.endswith('_search') or k.endswith('_raw')):
                        category=k[len('tactical_'):].rsplit('_',1)[0]
                        name='tactical_'+('puct' if k.endswith('_search') else 'raw')+'_'+category
                    r=comparison[name]
                    for a,b in ((r['mean'],stats['difference']),(r['ci95'][0],stats['lower']),(r['ci95'][1],stats['upper'])):
                        close(a,b,f'reported {seed}/{u}/{comparator}/{name}');checked+=1
                    if r['rows']!=stats['rows'] or r['clusters']!=stats['clusters']:raise ValueError('Reported bootstrap denominator differs')
        final=contrasts[f"{seed}-{design['updates_per_arm']}-treatment-minus-control"]['novel_mse']
        primary=final['difference']<=-.02 and final['upper']<0
        primary_passes.append(primary)
        if reported['primary_pass']!=primary:raise ValueError('Reported primary gate differs')
        for comparator in ('control','initial'):
            ours=contrasts[f"{seed}-{design['updates_per_arm']}-treatment-minus-{comparator}"]
            for name,e in ours.items():
                if name.startswith('tactical_') and name.endswith('_search'):
                    category=name[len('tactical_'):].rsplit('_',1)[0]
                    expected='REGRESSION' if e['difference']<=-.05 and e['upper']<0 else ('SUPPORTIVE' if e['difference']>=-.03 and e['lower']>=-.10 else 'UNRESOLVED')
                    key='tactical_puct_'+category
                elif name=='match_score':
                    expected='REGRESSION' if e['difference']<=-.05 and e['upper']<0 else ('SUPPORTIVE_UNDERPOWERED' if e['difference']>=-.05 and e['lower']>=-.15 else 'UNRESOLVED_UNDERPOWERED')
                    key='strength'
                elif name=='deep_mistakes':
                    expected='REGRESSION' if e['difference']>=.05 and e['lower']>0 else 'NO_DEMONSTRATED_REGRESSION'
                    key='deep'
                else:continue
                independent_gates.append(expected)
                if reported['safeguards'][comparator][key]!=expected:raise ValueError('Reported safeguard differs')
    regression=any(g=='REGRESSION' for g in independent_gates)
    unresolved=any(g.startswith('UNRESOLVED') for g in independent_gates)
    expected_decision='PROMISING_FOR_NEXT_VALIDATION' if all(primary_passes) and not regression and not unresolved else 'NO_CONFIRMED_PRESERVED_GUIDANCE_IMPROVEMENT'
    if (analysis['primary_pass'],analysis['regression'],analysis['unresolved_safeguards'],analysis['decision'])!=(all(primary_passes),regression,unresolved,expected_decision):
        raise ValueError('Reported overall decision differs')
    result = dict(passed=True, tasks=tasks, contrasts=contrasts, model_queries=0,training_updates=0,
                  method='independent saved raw-row arithmetic and family bootstrap; fresh RNG4411403 per contrast',
                  reported_arithmetic_comparisons=checked,
                  wall_seconds=time.perf_counter()-started)
    write(RESULTS/'independent-verification.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();result=verify(args.root)
    print('Independent saved-evidence verification PASS:',len(result['tasks']),'tasks',len(result['contrasts']),'paired contrasts')
