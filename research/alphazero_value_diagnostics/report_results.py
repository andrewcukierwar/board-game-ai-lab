"""Evidence-only reporting supplement; fixed endpoints/membership, no model query or source mutation."""
from collections import defaultdict
import json
from pathlib import Path
import time
from .common import RESULTS, ARMS, VARIANTS, read, write
from .analyze import estimate


def main():
    started=time.monotonic()
    analysis=read(RESULTS/'analysis.json')
    output=dict(tactical_categories={},exact_decisions={},sensitive_positions=[],value_strata={},resource={})
    data={(arm,v,s):read(RESULTS/'tasks'/f'{arm}-g6-{v}-positions'/f'{s}.json')['rows']
          for arm in ARMS for v in VARIANTS for s in ('tactical','solved','deep')}
    for arch in ('cnn','resnet3x64'):
        arms=[f'{seed}-{arch}' for seed in (42,314159)]
        output['tactical_categories'][arch]={}
        for category in ('unique_win','unique_safe'):
            rows=[r for r in data[arms[0],'A','tactical'] if r['category']==category]
            maps={(a,v):{r['id']:r for r in data[a,v,'tactical']} for a in arms for v in VARIANTS}
            output['tactical_categories'][arch][category]={v:dict(
                accuracy=estimate(rows,[sum(maps[a,v][r['id']]['correct'] for a in arms)/2 for r in rows]),
                delta_from_A=estimate(rows,[sum(maps[a,v][r['id']]['correct']-maps[a,'A'][r['id']]['correct'] for a in arms)/2 for r in rows])) for v in VARIANTS}
    fixtures=read(RESULTS.parents[1]/'alphazero_resnet_selfplay/fixtures.json')
    for arm in ARMS:
        output['exact_decisions'][arm]={}
        output['value_strata'][arm]={}
        for suite in ('solved','deep'):
            labels={r['id']:r for r in fixtures[suite]}
            output['exact_decisions'][arm][suite]={}
            for v in VARIANTS:
                rows=data[arm,v,suite]
                output['exact_decisions'][arm][suite][v]={str(z):dict(
                    rows=sum(r['exact']==z for r in rows),
                    accuracy=estimate([r for r in rows if r['exact']==z],[r['correct'] for r in rows if r['exact']==z]),
                    selected_values={str(w):sum(r['exact']==z and r['chosen_value']==w for r in rows) for w in (-1,0,1)}) for z in (-1,0,1)}
        for suite in ('tactical','solved','deep'):
            maps={v:{r['id']:r for r in data[arm,v,suite]} for v in VARIANTS}
            for base in data[arm,'A',suite]:
                flags={v:maps[v][base['id']]['action']!=base['action'] for v in 'BCDE'}
                if not any(flags.values()):continue
                row=dict(arm=arm,suite=suite,id=base['id'],cluster=base['cluster'],
                         actions={v:maps[v][base['id']]['action'] for v in VARIANTS},
                         correct={v:maps[v][base['id']]['correct'] for v in VARIANTS},
                         changed=flags,head_sensitivity='both' if flags['B'] and flags['C'] else
                         'value_only_at_A' if flags['B'] else 'prior_only_at_A' if flags['C'] else 'interaction_or_scale_only',
                         baseline_wrong=not base['correct'],
                         root_absolute_value=abs(base['root_value']))
                row['priority']=sum(flags.values())+2*row['baseline_wrong']
                output['sensitive_positions'].append(row)
        values=read(RESULTS/'tasks'/f'{arm}-values'/'values.json')
        for name,rows in list(values['exact'].items())+[(k,values[k]) for k in ('training_g6','training_g5','prequential_g5_on_g6')]:
            groups=defaultdict(list)
            for row in rows:groups[row['cluster']].append(row)
            output['value_strata'][arm][name]=dict(
                by_opening={f:dict(positions=len(rs),mse=sum((r['value']-r['target'])**2 for r in rs)/len(rs),zero_mse=sum(r['target']**2 for r in rs)/len(rs)) for f,rs in groups.items()},
                by_ply={str(k):dict(positions=len(rs),mse=sum((r['value']-r['target'])**2 for r in rs)/len(rs)) for k,rs in
                        ((k,[r for r in rows if r['ply']//8==k]) for k in range(6)) if rs})
    output['sensitive_positions'].sort(key=lambda r:(-r['priority'],r['arm'],r['suite'],r['id']))
    neutral_matches=[]
    for arm in ARMS:
        records=read(RESULTS/'tasks'/f'{arm}-g6-D-matches'/'matches.json')['records']
        neutral_matches.append([(r['opening_id'],r['game_index'],r['moves'],r['result']) for r in records])
    assert all(x==neutral_matches[0] for x in neutral_matches)
    assert all(data[a,'D',s]==data[ARMS[0],'D',s] for a in ARMS for s in ('tactical','solved','deep'))
    output['neutral_control_identical_across_checkpoints']=True
    timings=[read(p) for p in (RESULTS/'tasks').glob('*/complete.json')]
    output['resource']=dict(completed_tasks=len(timings),total_worker_cpu_seconds=sum(t['cpu_seconds'] for t in timings),
                            summed_worker_wall_seconds=sum(t['wall_seconds'] for t in timings),
                            largest_worker_peak_rss_mib=max(t['peak_rss_mib'] for t in timings))
    output['reporting_seconds']=time.monotonic()-started
    write(RESULTS/'report-supplement.json',output)
    print(json.dumps(dict(tactical_categories=output['tactical_categories'],resource=output['resource'],reporting_seconds=output['reporting_seconds']),indent=1))

if __name__=='__main__':
    main()
