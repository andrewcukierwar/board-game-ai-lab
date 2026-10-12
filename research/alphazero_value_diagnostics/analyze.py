"""Prospectively fixed paired cluster bootstrap and factorial/transition diagnostics."""
from collections import defaultdict, Counter
import numpy as np
from .common import RESULTS, read, write, ARMS, VARIANTS

RESAMPLES = 10000
SEED = 4410903


def estimate(rows, values):
    if not rows:
        return dict(n=0,mean=None,interval95=None,clusters=0)
    sums = defaultdict(lambda: [0.,0])
    for row, value in zip(rows,values):
        sums[row['cluster']][0] += value
        sums[row['cluster']][1] += 1
    totals = np.array([sums[k] for k in sorted(sums)],dtype=np.float64)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0,len(totals),size=(RESAMPLES,len(totals)))
    summed = totals[samples].sum(axis=1)
    boot = summed[:,0]/summed[:,1]
    return dict(n=len(rows),mean=float(sum(values)/len(values)),
                interval95=np.quantile(boot,[.025,.975]).tolist(),clusters=len(totals))


def pair(a,b,metric):
    bm = {r['id']:r for r in b}
    assert len(bm) == len(b) and {r['id'] for r in a} == set(bm)
    assert all(r['cluster'] == bm[r['id']]['cluster'] for r in a)
    return estimate(a,[r[metric]-bm[r['id']][metric] for r in a])


def pool(arms, metric):
    # Shared opening/position clusters resampled jointly across the two fixed networks.
    keys = [{r['id']:r for r in rows} for rows in arms]
    assert set(keys[0]) == set(keys[1])
    rows = list(keys[0].values())
    return estimate(rows,[(keys[0][r['id']][metric]+keys[1][r['id']][metric])/2 for r in rows])


def transitions(rows,base):
    bm = {r['id']:r for r in base}
    counts = Counter()
    changed = []
    for r in rows:
        b = bm[r['id']]
        counts['action_changed' if r['action'] != b['action'] else 'action_unchanged'] += 1
        category = ('fixed' if r['correct'] and not b['correct'] else
                    'introduced' if b['correct'] and not r['correct'] else 'same_correctness')
        counts[category] += 1
        if r['action'] != b['action']:
            changed.append(dict(id=r['id'],cluster=r['cluster'],baseline=b['action'],action=r['action'],status=category))
    return dict(counts=dict(counts),positions=changed)


def value_summary(rows):
    if not rows:
        return dict(positions=0)
    values = [r['value'] for r in rows]; targets = [r['target'] for r in rows]
    bins=[]
    for i in range(10):
        subset=[r for r in rows if min(9,int((r['value']+1)*5)) == i]
        bins.append(dict(bin=i,n=len(subset),mean_value=np.mean([r['value'] for r in subset]).item() if subset else None,
                         mean_target=np.mean([r['target'] for r in subset]).item() if subset else None))
    mse=lambda subset: estimate(subset,[(r['value']-r['target'])**2 for r in subset])
    zero=[r['target']**2 for r in rows]
    out=dict(positions=len(rows),mse=mse(rows),zero_mse=estimate(rows,zero),
             excess_mse_over_zero=estimate(rows,[(r['value']-r['target'])**2-r['target']**2 for r in rows]),
             mean_value=float(np.mean(values)),mean_target=float(np.mean(targets)),
             mean_absolute_value=float(np.mean(np.abs(values))),saturated_share=float(np.mean(np.abs(values)>.95)),
             high_confidence_share=float(np.mean(np.abs(values)>.8)),bins=bins,
             calibration_error=sum(b['n']/len(rows)*abs(b['mean_value']-b['mean_target']) for b in bins if b['n']),
             by_target={str(z):mse([r for r in rows if r['target']==z]) for z in (-1,0,1)},
             by_ply={str(p):mse([r for r in rows if r['ply']//8==p]) for p in range(6)},
             by_actor={str(a):mse([r for r in rows if r['actor']==a]) for a in (0,1)},
             by_confidence={name:mse([r for r in rows if (abs(r['value'])>.8)==high])
                            for name,high in [('above_0.8',True),('at_most_0.8',False)]})
    if 'repetitions' in rows[0]:
        groups=defaultdict(list)
        for r in rows:
            groups[r['position_key']].append(r['target'])
        repeated=[zs for zs in groups.values() if len(zs)>1]
        out['target_noise']=dict(unique_positions=len(groups),repeated_positions=len(repeated),
             conflicting_repeated_positions=sum(len(set(zs))>1 for zs in repeated),
             conflicting_position_examples=sum(len(zs) for zs in repeated if len(set(zs))>1),
             empirical_identical_position_mse_floor=sum(sum((z-np.mean(zs))**2 for z in zs) for zs in groups.values())/len(rows))
        out['by_repetition']={str(repeated):mse([r for r in rows if (r['repetitions']>1)==repeated]) for repeated in (False,True)}
        out['by_terminal_distance']={str(p):mse([r for r in rows if r['terminal_distance']//8==p]) for p in range(6)}
        out['by_seen_in_training']={str(seen):mse([r for r in rows if r['seen_in_training']==seen]) for seen in (False,True)}
    return out


def main():
    status=read(RESULTS/'status.json')
    if status['status'] != 'COMPLETE':
        raise ValueError('Full analyses require complete prospective work')
    plan=read(RESULTS/'design.json')
    result=dict(positions={},matches={},values={},pooled={},negative_controls={})
    data={}
    for arm in ARMS:
        result['positions'][arm]={};result['matches'][arm]={}
        for variant in VARIANTS:
            directory=RESULTS/'tasks'/f'{arm}-g6-{variant}-positions'
            data[arm,variant]={s:read(directory/f'{s}.json')['rows'] for s in ('tactical','solved','deep')}
            suites={}
            for suite, rows in data[arm,variant].items():
                subsets=({'immediate_win':[r for r in rows if r['category']=='unique_win'],
                          'safe_response':[r for r in rows if r['category']=='unique_safe']} if suite=='tactical' else {})
                metrics=('correct','avoidable','missed_win') if suite=='tactical' else ('correct','mistake','win_draw','win_loss','draw_loss')
                entry={m:estimate(rows,[r[m] for r in rows]) for m in metrics}
                entry.update({name:estimate(rs,[r['correct'] for r in rs]) for name,rs in subsets.items()})
                base=data[arm,'A'][suite]
                entry['delta_from_A']={m:pair(rows,base,m) for m in metrics}
                if suite=='tactical':
                    entry['delta_from_A'].update({name:pair(rs,[b for b in base if b['category']==rs[0]['category']],'correct') for name,rs in subsets.items()})
                entry['transitions']=transitions(rows,base)
                suites[suite]=entry
            result['positions'][arm][variant]=suites
            matches=read(RESULTS/'tasks'/f'{arm}-g6-{variant}-matches'/'matches.json')['records']
            rows=[dict(r,id=f'{r["opening_id"]}:{r["game_index"]}',cluster=r['family']) for r in matches]
            data[arm,variant]['matches']=rows
            base=data[arm,'A']['matches']
            result['matches'][arm][variant]=dict(wdl=dict(Counter(r['result'] for r in rows)),
                score=estimate(rows,[r['score'] for r in rows]),delta_from_A=pair(rows,base,'score'),
                by_color={str(c):dict(score=estimate([r for r in rows if r['agent_color']==c],
                                                   [r['score'] for r in rows if r['agent_color']==c]),
                                     delta_from_A=pair([r for r in rows if r['agent_color']==c],
                                                       [r for r in base if r['agent_color']==c],'score')) for c in (0,1)})
        result['negative_controls'][arm]={}
        for suite in ('tactical','solved','deep'):
            rows=read(RESULTS/'tasks'/f'{arm}-g0-A-positions'/f'{suite}.json')['rows']
            result['negative_controls'][arm][suite]=estimate(rows,[r['correct'] for r in rows])
        values=read(RESULTS/'tasks'/f'{arm}-values'/'values.json')
        result['values'][arm]={k:value_summary(values[k]) for k in ('training_g6','training_g5','prequential_g5_on_g6')}
        result['values'][arm]['exact']={s:value_summary(rows) for s,rows in values['exact'].items()}
        result['values'][arm]['exact_replay_overlap']=dict(positions=len(values['exact_replay_overlap']),
             disagreements=sum(r['exact']!=r['realized'] for r in values['exact_replay_overlap']))
        result['values'][arm]['label_audit']=values['label_audit']
    for architecture in ('cnn','resnet3x64'):
        arms=[f'{seed}-{architecture}' for seed in (42,314159)]
        pooled={}
        for suite in ('tactical','solved','deep','matches'):
            metric='score' if suite=='matches' else 'correct'
            tables={v:pool([data[a,v][suite] for a in arms],metric) for v in VARIANTS}
            contrasts={}
            # Factorial A=learned P/V, B=learned P/zero V, C=uniform P/learned V, D=uniform P/zero V.
            formulas={'zero_value_learned_prior':{'B':1,'A':-1},
                      'zero_value_uniform_prior':{'D':1,'C':-1},
                      'uniform_prior_learned_value':{'C':1,'A':-1},
                      'uniform_prior_zero_value':{'D':1,'B':-1},
                      'interaction':{'A':1,'B':-1,'C':-1,'D':1},
                      'half_value':{'E':1,'A':-1}}
            for name,formula in formulas.items():
                rows=data[arms[0],'A'][suite]
                maps={(a,v):{r['id']:r for r in data[a,v][suite]} for a in arms for v in formula}
                contrast=[sum(sum(c*maps[a,v][r['id']][metric] for v,c in formula.items()) for a in arms)/2 for r in rows]
                contrasts[name]=estimate(rows,contrast)
            pooled[suite]=dict(absolute=tables,contrasts=contrasts)
        result['pooled'][architecture]=pooled
    result['architecture_exact_value_difference']={}
    for suite in ('solved','deep'):
        differences=[]
        for seed in (42,314159):
            c=read(RESULTS/'tasks'/f'{seed}-cnn-values'/'values.json')['exact'][suite]
            r=read(RESULTS/'tasks'/f'{seed}-resnet3x64-values'/'values.json')['exact'][suite]
            rm={row['id']:row for row in r}
            differences.append([(rm[row['id']]['value']-row['target'])**2-(row['value']-row['target'])**2 for row in c])
        result['architecture_exact_value_difference'][suite]=estimate(c,[(a+b)/2 for a,b in zip(*differences)])
    write(RESULTS/'analysis.json',result)
    print('Prospective analysis complete')

if __name__ == '__main__':
    main()
