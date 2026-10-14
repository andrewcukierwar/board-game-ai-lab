"""Prospectively fixed final contrast; separate realized targets, exact labels and strength."""
from collections import defaultdict, Counter
import numpy as np
from .common import RESULTS, read, write, sha


def interval(rows, field='delta', group='cluster'):
    """Whole-family paired percentile bootstrap of position-weighted means."""
    grouped=defaultdict(list)
    for row in rows: grouped[row[group]].append(float(row[field]))
    keys=sorted(grouped)
    sums=np.array([sum(grouped[k]) for k in keys]); counts=np.array([len(grouped[k]) for k in keys])
    rng=np.random.default_rng(4411403)
    chosen=rng.integers(0,len(keys),size=(10000,len(keys)))
    means=sums[chosen].sum(1)/counts[chosen].sum(1)
    return dict(mean=float(sums.sum()/counts.sum()),ci95=[float(x) for x in np.quantile(means,[.025,.975])],
                clusters=len(keys),rows=len(rows))


def delta(a,b,field,group='cluster',id_field='id'):
    """a minus b, membership and order verified before subtracting."""
    assert len(a)==len(b) and [x[id_field] for x in a]==[x[id_field] for x in b]
    assert all(x[group]==y[group] for x,y in zip(a,b))
    return interval([dict(cluster=x[group],delta=x[field]-y[field]) for x,y in zip(a,b)])


def value_summary(rows):
    v=np.array([x['value'] for x in rows]); z=np.array([x['target'] for x in rows])
    bins=[]
    indices=np.minimum(9,((v+1)*5).astype(int))
    for b in range(10):
        mask=indices==b
        bins.append(dict(bin=b,count=int(mask.sum()),predicted=None if not mask.any() else float(v[mask].mean()),
            observed=None if not mask.any() else float(z[mask].mean())))
    return dict(positions=len(rows),mse=float(np.mean((v-z)**2)),zero_mse=float(np.mean(z*z)),
        mean_value=float(v.mean()),mean_target=float(z.mean()),bias=float((v-z).mean()),
        mean_abs_value=float(np.abs(v).mean()),saturation=float(np.mean(np.abs(v)>.95)),
        high_confidence=float(np.mean(np.abs(v)>.8)),calibration=sum(x['count']*abs(x['predicted']-x['observed']) for x in bins if x['count'])/len(rows),bins=bins,
        mse_by_target={str(k):dict(n=int((z==k).sum()),mse=float(np.mean((v[z==k]-k)**2)) if (z==k).any() else None,
            zero_mse=float(k*k)) for k in (-1,0,1)})


def load_task(name):
    root=RESULTS/'tasks'/name
    assert read(root/'complete.json')['complete']
    return dict(values=read(root/'values.json'),positions=read(root/'positions.json'),matches=read(root/'matches.json')['records'])


def summaries(task):
    values={name:value_summary(task['values'][name]['rows']) for name in ('training','heldout','novel')}
    values['generalization_gap']=values['novel']['mse']-values['training']['mse']
    values['all_generalization_gap']=values['heldout']['mse']-values['training']['mse']
    positions={}
    for suite,rows in task['positions'].items():
        raw_field='policy_correct' if suite=='tactical' else 'accuracy'
        entry=dict(n=len(rows),raw_accuracy=sum(r[raw_field] for r in rows)/len(rows),
            puct_accuracy=sum(r['search']['correct'] for r in rows)/len(rows))
        if suite=='tactical':
            entry['categories']={c:dict(n=sum(r['category']==c for r in rows),
                raw_accuracy=np.mean([r[raw_field] for r in rows if r['category']==c]).item(),
                puct_accuracy=np.mean([r['search']['correct'] for r in rows if r['category']==c]).item()) for c in sorted({r['category'] for r in rows})}
        else:
            entry['value']=value_summary([dict(value=r['value'],target=r['exact']) for r in rows])
            entry['mistakes']=sum(r['search']['mistake'] for r in rows)
            entry['downgrades']={k:sum(r['search'][k] for r in rows) for k in ('win_draw','win_loss','draw_loss')}
        positions[suite]=entry
    matches=task['matches']
    return dict(values=values,positions=positions,strength=dict(n=len(matches),score=sum(r['score'] for r in matches)/len(matches),
        wdl=dict(Counter(r['result'] for r in matches)),opponent='frozen reference Negamax-3'))


def contrasts(t,c):
    result={}
    for dataset in ('training','heldout','novel'):
        result[dataset+'_mse']=delta(t['values'][dataset]['rows'],c['values'][dataset]['rows'],'se')
        for arm,task in (('treatment',t),('control',c)):
            rows=task['values'][dataset]['rows']
            result[dataset+'_'+arm+'_excess_zero']=interval([dict(cluster=r['cluster'],delta=r['se']-r['target']**2) for r in rows])
    for suite in ('tactical','solved','deep'):
        a,b=t['positions'][suite],c['positions'][suite]
        for metric,extract in (('raw_accuracy',lambda r:r['policy_correct'] if suite=='tactical' else r['accuracy']),
                               ('puct_accuracy',lambda r:r['search']['correct'])):
            result[suite+'_'+metric]=delta([dict(r,x=extract(r)) for r in a],[dict(r,x=extract(r)) for r in b],'x')
        if suite=='tactical':
            for category in sorted({r['category'] for r in a}):
                for prefix,extract in (('puct',lambda r:r['search']['correct']),('raw',lambda r:r['policy_correct'])):
                    aa=[dict(r,x=extract(r)) for r in a if r['category']==category]
                    bb=[dict(r,x=extract(r)) for r in b if r['category']==category]
                    result[f'tactical_{prefix}_{category}']=delta(aa,bb,'x')
        else:
            result[suite+'_value_mse']=delta(a,b,'se')
            result[suite+'_mistake']=delta([dict(r,x=r['search']['mistake']) for r in a],[dict(r,x=r['search']['mistake']) for r in b],'x')
    # paired_game records have opening_id, family and game_index; explicit frozen order is identity.
    aa=[dict(cluster=r['family'],id=f'{r["opening_id"]}:{r["game_index"]}',score=r['score']) for r in t['matches']]
    bb=[dict(cluster=r['family'],id=f'{r["opening_id"]}:{r["game_index"]}',score=r['score']) for r in c['matches']]
    result['strength']=delta(aa,bb,'score')
    return result


def gate(effect,kind):
    point=effect['mean']; lo,hi=effect['ci95']
    if kind=='tactical':
        if point<=-.05 and hi<0:return 'REGRESSION'
        return 'SUPPORTIVE' if point>=-.03 and lo>=-.10 else 'UNRESOLVED'
    if kind=='strength':
        if point<=-.05 and hi<0:return 'REGRESSION'
        return 'SUPPORTIVE_UNDERPOWERED' if point>=-.05 and lo>=-.15 else 'UNRESOLVED_UNDERPOWERED'
    return 'REGRESSION' if point>=.05 and lo>0 else 'NO_DEMONSTRATED_REGRESSION'


def main():
    design=read(RESULTS/'design.json');n=design['updates_per_arm']; result=dict(seeds={},design_sha256=sha(RESULTS/'design.json'),membership_sha256=design['membership_sha256'])
    for seed in design['seeds']:
        initial=load_task(f'{seed}-initial-0'); out=dict(initial=summaries(initial),milestones={},contrasts={},safeguards={})
        for milestone in design['milestones'][1:]:
            t=load_task(f'{seed}-treatment-{milestone}');c=load_task(f'{seed}-control-{milestone}')
            out['milestones'][str(milestone)]=dict(control=summaries(c),treatment=summaries(t))
            out['contrasts'][str(milestone)]=contrasts(t,c)
            if milestone==n:
                out['treatment_minus_initial']=contrasts(t,initial)
                for comparator,comparison in (('control',out['contrasts'][str(n)]),('initial',out['treatment_minus_initial'])):
                    gates={k:gate(v,'tactical') for k,v in comparison.items() if k.startswith('tactical_puct_') and k!='tactical_puct_accuracy'}
                    gates['strength']=gate(comparison['strength'],'strength')
                    gates['deep']=gate(comparison['deep_mistake'],'deep')
                    out['safeguards'][comparator]=gates
        primary=out['contrasts'][str(n)]['novel_mse']
        out['primary_pass']=primary['mean']<=-.02 and primary['ci95'][1]<0
        result['seeds'][str(seed)]=out
    result['primary_pass']=all(s['primary_pass'] for s in result['seeds'].values())
    safeguard_values=[v for s in result['seeds'].values() for g in s['safeguards'].values() for v in g.values()]
    result['regression']=any(v=='REGRESSION' for v in safeguard_values)
    result['unresolved_safeguards']=any(v.startswith('UNRESOLVED') for v in safeguard_values)
    result['decision']='PROMISING_FOR_NEXT_VALIDATION' if result['primary_pass'] and not result['regression'] and not result['unresolved_safeguards'] else 'NO_CONFIRMED_PRESERVED_GUIDANCE_IMPROVEMENT'
    write(RESULTS/'analysis.json',result)
    print('Frozen analysis complete',result['decision'],flush=True)

if __name__=='__main__':main()
