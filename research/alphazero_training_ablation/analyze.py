"""Evidence-only family bootstrap; no model queries, solving or training."""
import argparse
from collections import defaultdict
import numpy as np
from .common import ROOT, OUTPUT, read, write

def bootstrap(rows, fields, seed=4404703, replicates=10000):
    grouped=defaultdict(list)
    for row in rows: grouped[row['cluster']].append(row)
    families=sorted(grouped)
    totals=np.array([[sum(r[f] for r in grouped[k]) for f in fields] for k in families],dtype=float)
    counts=np.array([len(grouped[k]) for k in families],dtype=float)
    draws=np.random.default_rng(seed).integers(0,len(families),(replicates,len(families)))
    samples=totals[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
    estimates=totals.sum(axis=0)/counts.sum()
    return {f:dict(mean=float(estimates[i]),ci95=[float(v) for v in np.quantile(samples[:,i],[.025,.975])],
        one_sided95_lower=float(np.quantile(samples[:,i],.05)),one_sided95_upper=float(np.quantile(samples[:,i],.95)),
        families=len(families),positions=len(rows)) for i,f in enumerate(fields)}

def paired(a,b,fields):
    lookup={r['id']:r for r in b}
    if set(lookup)!={r['id'] for r in a}: raise ValueError('Unmatched evaluation rows')
    rows=[]
    for r in a:
        other=lookup[r['id']]
        if r['cluster']!=other['cluster']: raise ValueError('Unmatched clusters')
        rows.append(dict(id=r['id'],cluster=r['cluster'],**{f:r[f]-other[f] for f in fields}))
    return rows

def noninferiority(result, margin, lower_better=True):
    lower=result['one_sided95_lower']; upper=result['one_sided95_upper']
    if lower_better:
        return 'PASS' if upper<=margin else 'FAIL' if lower>margin else 'INCONCLUSIVE'
    return 'PASS' if lower>=margin else 'FAIL' if upper<margin else 'INCONCLUSIVE'

def calibration(rows):
    policy=[]; value=[]
    fixtures={r['id']:r for r in read(ROOT/'fixtures.json')['solved']}
    for lo in range(10):
        chosen=[r for r in rows if min(9,int(max(r['policy'])*10))==lo]
        if chosen: policy.append(dict(bin=lo,n=len(chosen),confidence=float(np.mean([max(r['policy']) for r in chosen])),
                                     accuracy=float(np.mean([r['accuracy'] for r in chosen]))))
        chosen=[r for r in rows if min(9,int((r['value']+1)*5))==lo]
        if chosen: value.append(dict(bin=lo,n=len(chosen),prediction=float(np.mean([r['value'] for r in chosen])),
                    exact_mean=float(np.mean([fixtures[r['id']]['value'] for r in chosen]))))
    return dict(policy_bins=policy,value_bins=value,policy_ece=sum(r['n']*abs(r['confidence']-r['accuracy']) for r in policy)/len(rows),
                value_absolute_calibration_error=sum(r['n']*abs(r['prediction']-r['exact_mean']) for r in value)/len(rows))

def score_rows(records):
    return [dict(id=f'{r["opening_id"]}-{r["game_index"]}',cluster=r['family'],
                 score={'win':1.,'draw':.5,'loss':0.}[r['result']],color=r['agent_color']) for r in records]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'results/analysis.json'));args=parser.parse_args()
    status=read(OUTPUT/'execution/status.json')
    if status['status']!='COMPLETE_PENDING_FINAL_ANALYSIS': raise ValueError('Incomplete run: cannot analyze as full experiment')
    root=OUTPUT/'execution'; fields=('ce','accuracy','mse','optimal_mass'); report=dict(execution=status,learning={},tactical={},strength={})
    g22=[]; learning_fail=False
    tactical_labels=[]; strength_labels=[]
    for seed in (42,314159):
        parent=read(root/f'parent-{seed}/solved.json')['rows']; report['learning'][str(seed)]={}
        parent_calibration=calibration(parent)
        report['learning'][str(seed)]['parent']=dict(metrics=bootstrap(parent,fields),calibration=parent_calibration)
        for generation in (21,22):
            suite={a:{s:read(root/f'{seed}-{a}-g{generation}-evaluation/{s}.json')['rows'] for s in ('solved','tactical','deep')} for a in ('control','treatment')}
            for a in ('control','treatment'):
                rows=suite[a]['solved']
                report['learning'][str(seed)][f'{a}-g{generation}']=dict(metrics=bootstrap(rows,fields),
                    change_from_parent=bootstrap(paired(rows,parent,fields),fields),calibration=calibration(rows),
                    calibration_change={f:calibration(rows)[f]-parent_calibration[f] for f in ('policy_ece','value_absolute_calibration_error')})
            contrast=bootstrap(paired(suite['treatment']['solved'],suite['control']['solved'],fields),fields)
            report['learning'][str(seed)][f'contrast-g{generation}']=contrast
            if generation==22:
                g22.append(paired(suite['treatment']['solved'],suite['control']['solved'],fields))
                learning_fail |= contrast['ce']['ci95'][0]>0 or contrast['accuracy']['ci95'][1]<0 or contrast['mse']['mean']>.02
            for method in ('policy','puct'):
                error_fields=(f'{method}_avoidable',f'{method}_missed_win')
                baseline=read(root/f'parent-{seed}/tactical.json')['rows']
                report['tactical'][f'{seed}-g{generation}-{method}-one-ply']=dict(
                    parent=bootstrap(baseline,error_fields),
                    arms={a:dict(rates=bootstrap(suite[a]['tactical'],error_fields),
                        counts={f:sum(r[f] for r in suite[a]['tactical']) for f in error_fields},
                        change_from_parent=bootstrap(paired(suite[a]['tactical'],baseline,error_fields),error_fields))
                        for a in ('control','treatment')},
                    contrast=bootstrap(paired(suite['treatment']['tactical'],suite['control']['tactical'],error_fields),error_fields))
                for category in ('unique_win','unique_safe'):
                    filtered=lambda rows:[r for r in rows if r['category']==category]
                    treatment=filtered(suite['treatment']['tactical']); control=filtered(suite['control']['tactical'])
                    baseline=filtered(read(root/f'parent-{seed}/tactical.json')['rows'])
                    field=f'{method}_correct'; absolute=bootstrap(treatment,(field,))
                    vs_control=bootstrap(paired(treatment,control,(field,)),(field,))[field]
                    vs_parent=bootstrap(paired(treatment,baseline,(field,)),(field,))[field]
                    label='PASS' if min(vs_control['mean'],vs_parent['mean'])>=-.02-1e-12 else 'FAIL'
                    tactical_labels.append(label)
                    report['tactical'][f'{seed}-g{generation}-{method}-{category}']=dict(label=label,accuracy=absolute,
                         vs_control=vs_control,vs_parent=vs_parent,control_accuracy=bootstrap(control,(field,)),
                         control_vs_parent=bootstrap(paired(control,baseline,(field,)),(field,)))
                for name in ('solved','deep'):
                    field=f'{method}_deep_mistake'
                    result=bootstrap(paired(suite['treatment'][name],suite['control'][name],(field,)),(field,))[field]
                    label=noninferiority(result,.05); tactical_labels.append(label)
                    baseline=read(root/f'parent-{seed}/{name}.json')['rows']
                    report['tactical'][f'{seed}-g{generation}-{method}-{name}-deep']=dict(label=label,vs_control=result,
                        treatment_rate=bootstrap(suite['treatment'][name],(field,)),control_rate=bootstrap(suite['control'][name],(field,)),
                        treatment_vs_parent=bootstrap(paired(suite['treatment'][name],baseline,(field,)),(field,)))
        matches={a:read(root/f'{seed}-{a}-strength/matches.json')['records'] for a in ('control','treatment')}
        scores={a:score_rows(r) for a,r in matches.items()}
        contrast=bootstrap(paired(scores['treatment'],scores['control'],('score',)),('score',))['score']
        label=noninferiority(contrast,-.05,False); strength_labels.append(label)
        report['strength'][str(seed)]=dict(noninferiority=label,contrast=contrast)
        for a in ('control','treatment'):
            report['strength'][str(seed)][a]=dict(metrics=bootstrap(scores[a],('score',)),
                wins=sum(r['result']=='win' for r in matches[a]),draws=sum(r['result']=='draw' for r in matches[a]),
                losses=sum(r['result']=='loss' for r in matches[a]),
                by_color={str(c):bootstrap([r for r in scores[a] if r['color']==c],('score',)) for c in (0,1)})
    # Same frozen positions in both seeds: average seed-specific contrasts BEFORE shared family resampling.
    pooled=[dict(id=a['id'],cluster=a['cluster'],**{f:(a[f]+b[f])/2 for f in fields}) for a,b in zip(*g22)]
    p=bootstrap(pooled,fields); report['learning']['pooled_g22']=p
    seed_contrasts=[report['learning'][str(s)]['contrast-g22'] for s in (42,314159)]
    practical=p['ce']['mean']<=-.02 and p['accuracy']['mean']>=.02 and all(c['ce']['mean']<=0 and c['accuracy']['mean']>=0 and c['mse']['mean']<=.02 for c in seed_contrasts)
    confirmed=p['ce']['ci95'][1]<0 and p['accuracy']['ci95'][0]>0
    learning_fail |= p['ce']['ci95'][0]>-.02 or p['accuracy']['ci95'][1]<.02
    combine=lambda labels:'FAIL' if 'FAIL' in labels else 'INCONCLUSIVE' if 'INCONCLUSIVE' in labels else 'PASS'
    # Independent opening-family resampling within each seed for pooled parent-match support.
    distributions=[]
    for seed in (42,314159):
        rows=score_rows(read(root/f'{seed}-treatment-strength/matches.json')['records']); groups=defaultdict(list)
        for r in rows: groups[r['cluster']].append(r['score'])
        means=np.array([np.mean(v) for _,v in sorted(groups.items())]); rng=np.random.default_rng(4404703+seed)
        distributions.append(means[rng.integers(0,len(means),(10000,len(means)))].mean(axis=1))
    lower=float(np.quantile((distributions[0]+distributions[1])/2,.025))
    report['strength']['support']=dict(pooled_lower95=lower,positive=lower>=.5 and all(report['strength'][str(s)]['treatment']['metrics']['score']['mean']>.5 for s in (42,314159)))
    report['decisions']=dict(execution='PASS' if status['official_preservation']=='PASS' else 'FAIL',
        neural_learning='PASS' if practical and confirmed else 'FAIL' if learning_fail else 'INCONCLUSIVE',
        tactical_reliability=combine(tactical_labels),playing_strength=combine(strength_labels),
        learning_practical_thresholds_met=practical,learning_improvement_confirmed=confirmed)
    write(args.output,report)

if __name__=='__main__': main()
