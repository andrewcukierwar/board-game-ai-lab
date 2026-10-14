"""Prespecified paired family/ancestry cluster bootstrap and all endpoint summaries."""
from collections import defaultdict,Counter
import numpy as np
from .common import RESULTS,read,write

ANALYSIS_SEED=4410903


def contrast(rows,field):
    groups=defaultdict(list)
    for r in rows:groups[r['cluster']].append(r['guarded'][field]-r['control'][field])
    keys=sorted(groups); sums=np.array([sum(groups[k]) for k in keys]);counts=np.array([len(groups[k]) for k in keys])
    rng=np.random.default_rng(ANALYSIS_SEED)
    indices=rng.integers(0,len(keys),(10000,len(keys)))
    draws=sums[indices].sum(1)/counts[indices].sum(1)
    return dict(difference=float(sums.sum()/counts.sum()),ci95=np.quantile(draws,[.025,.975]).tolist(),clusters=len(keys),rows=len(rows))


def summary(rows,fields):
    return {v:{f:dict(count=sum(r[v][f] for r in rows),mean=float(np.mean([r[v][f] for r in rows]))) for f in fields}
            for v in ('control','guarded')}


def timings(values):
    x=np.array(values,dtype=float)
    return dict(n=len(x),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.quantile(x,.95)),worst=float(x.max()))


def main():
    result=dict(seeds={},pooled={},bootstrap_replicates=10000,analysis_seed=ANALYSIS_SEED)
    all_rows=defaultdict(list);all_timing=[]
    for seed in (42,314159):
        out={};base=RESULTS/'tasks'/str(seed)
        for suite in ('tactical','solved','deep'):
            rows=read(base/f'{suite}.json')['rows'];all_rows[suite].append(rows)
            fields=('correct','avoidable','missed_win') if suite=='tactical' else ('correct','mistake','win_draw','win_loss','draw_loss')
            item=dict(n=len(rows),summary=summary(rows,fields),contrasts={f:contrast(rows,f) for f in fields})
            if suite=='tactical':
                item['categories']={cat:dict(summary=summary([r for r in rows if r['category']==cat],fields),
                    contrast=contrast([r for r in rows if r['category']==cat],'correct')) for cat in ('unique_win','unique_safe')}
                item['fixes']=sum(not r['control']['correct'] and r['guarded']['correct'] for r in rows)
                item['introduced']=sum(r['control']['correct'] and not r['guarded']['correct'] for r in rows)
                item['baseline_correct']=sum(r['control']['correct'] for r in rows)
                item['baseline_incorrect_with_safe']=sum(not r['control']['correct'] and bool(r['guarded']['guard']['safe_actions']) for r in rows)
            else:
                item['classes']={str(c):dict(n=sum(r['control']['exact']==c for r in rows),
                    **summary([r for r in rows if r['control']['exact']==c],fields)) for c in (-1,0,1) if any(r['control']['exact']==c for r in rows)}
                item['value_differences']=dict(Counter(str(r['guarded']['chosen_value']-r['control']['chosen_value']) for r in rows))
                item['regressions']=[r['id'] for r in rows if r['guarded']['chosen_value']<r['control']['chosen_value']]
            item['reasons']=dict(Counter(r['guarded']['guard']['reason'] for r in rows))
            item['changed']=sum(r['guarded']['guard']['changed'] for r in rows)
            item['no_safe']=sum(r['guarded']['guard']['no_safe_action'] for r in rows)
            item['changes']=[dict(id=r['id'],reason=r['guarded']['guard']['reason'],original=r['control']['final_action'],
                final=r['guarded']['final_action'],exact_difference=None if suite=='tactical' else r['guarded']['chosen_value']-r['control']['chosen_value']) for r in rows if r['guarded']['guard']['changed']]
            out[suite]=item
            all_timing.extend(r['guarded'] for r in rows)
        games={v:read(base/f'{v}-matches.json')['records'] for v in ('control','guarded')}
        paired=[]; own=[]
        for a,b in zip(games['control'],games['guarded']):
            assert (a['family'],a['game_index'])==(b['family'],b['game_index'])
            paired.append(dict(cluster=a['family'],control=dict(score=a['score']),guarded=dict(score=b['score'])))
        all_rows['matches'].append(paired)
        m=dict(contrasts={'score':contrast(paired,'score')},summary={})
        for v,rs in games.items():
            decisions=[d for r in rs for d in r['decisions'] if d['color']==r['agent_color']]
            m['summary'][v]=dict(wdl=dict(Counter(r['result'] for r in rs)),score=float(np.mean([r['score'] for r in rs])),
                colors={str(c):float(np.mean([r['score'] for r in rs if r['agent_color']==c])) for c in (0,1)},
                families={f:float(np.mean([r['score'] for r in rs if r['family']==f])) for f in sorted({r['family'] for r in rs})},
                decisions=len(decisions))
            if v=='guarded':
                own=decisions;all_timing.extend(decisions)
                changed=[d for d in decisions if d['guard']['changed']]
                m['guard']=dict(reasons=dict(Counter(d['guard']['reason'] for d in decisions)),changed=len(changed),
                    activation_rate=sum(d['guard']['reason'] in ('IMMEDIATE_WIN','AVOID_IMMEDIATE_LOSS') for d in decisions)/len(decisions),
                    changed_rate=len(changed)/len(decisions),zero_visit_selected=sum(d['guard']['selected_zero_visits'] for d in decisions),
                    changed_unsafe_original=sum(d['original_action'] in d['guard']['unsafe_actions'] for d in changed),
                    changed_missed_win=sum(bool(d['guard']['immediate_wins']) and d['original_action'] not in d['guard']['immediate_wins'] for d in changed),
                    game_activation_by_result={res:dict(games=sum(r['result']==res for r in rs),
                        activated_games=sum(r['result']==res and any(d.get('guard') and d['guard']['changed'] for d in r['decisions']) for r in rs)) for res in ('win','draw','loss')})
        out['matches']=m;result['seeds'][str(seed)]=out
    for suite,pairs in all_rows.items():
        pooled=[]
        for a,b in zip(*pairs):
            assert a['cluster']==b['cluster']
            fields=('score',) if suite=='matches' else (('correct','avoidable','missed_win') if suite=='tactical' else ('correct','mistake','win_draw','win_loss','draw_loss'))
            pooled.append(dict(cluster=a['cluster'],**{v:{f:(a[v][f]+b[v][f])/2 for f in fields} for v in ('control','guarded')}))
        result['pooled'][suite]={f:contrast(pooled,f) for f in fields}
    match=result['pooled']['matches']['score'];seeds=[result['seeds'][str(s)]['matches']['contrasts']['score']['difference'] for s in (42,314159)]
    result['strength_verdict']='PASS' if match['difference']>=.05 and min(seeds)>=0 and match['ci95'][0]>0 else ('FAIL' if match['difference']<=-.05 and match['ci95'][1]<0 else 'INCONCLUSIVE')
    result['deep_verdict']='FAIL' if any(result['seeds'][str(s)][suite]['regressions'] for s in (42,314159) for suite in ('solved','deep')) else 'PASS_NO_OBSERVED_WDL_REGRESSION'
    result['tactical_verdict']='PASS' if all(result['seeds'][str(s)]['tactical']['summary']['guarded']['correct']['count']==200 for s in (42,314159)) else 'FAIL'
    result['latency']=dict(search_seconds=timings([r['search_seconds'] for r in all_timing]),
        guard_seconds=timings([r['guard']['guard_seconds'] for r in all_timing]),
        classification_seconds=timings([r['guard']['classification_seconds'] for r in all_timing]),total_seconds=timings([r['total_seconds'] for r in all_timing]),
        relative_overhead=timings([r['guard']['guard_seconds']/r['search_seconds'] for r in all_timing]),
        candidates=timings([r['guard']['candidate_count'] for r in all_timing]),opponent_replies=timings([r['guard']['opponent_replies_inspected'] for r in all_timing]),
        occupancy={str(i):timings([r['guard']['guard_seconds'] for r in all_timing if i<=r['occupancy']<i+7]) for i in range(0,42,7) if any(i<=r['occupancy']<i+7 for r in all_timing)})
    result['latency_verdict']='PASS' if result['latency']['guard_seconds']['p95']<=.001 and result['latency']['guard_seconds']['worst']<=.005 and result['latency']['relative_overhead']['mean']<=.05 else 'FAIL_BUDGET_EXCEEDED'
    write(RESULTS/'analysis.json',result)
    print(result['strength_verdict'],match)
if __name__=='__main__':main()
