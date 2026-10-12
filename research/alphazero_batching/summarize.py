"""Derive prospective acceptance screens without modifying raw records."""
import argparse
import json
from pathlib import Path
from .benchmark import CRITERIA, require
from research.alphazero_baseline import audit


def comparisons(reference,candidate):
    refs={g['index']:g for g in reference}
    paired=[(refs[g['index']],g) for g in candidate if g['index'] in refs]
    return dict(games=len(paired),trajectories=sum(a['moves']!=b['moves'] for a,b in paired),
                outcomes=sum(a['winner']!=b['winner'] for a,b in paired),
                examples=sum(a['examples']!=b['examples'] for a,b in paired))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--numerical',type=Path,required=True)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    require(not args.output.exists(),'Summary output already exists')
    args.output.mkdir(parents=True)
    validation=audit.read(args.numerical/'complete_validation.json')
    trials=[audit.read(p) for p in sorted(args.collection.glob('trial-*.json'))]
    divergences=[]
    for r in validation:
        ref=next(t for t in validation if t['seed']==r['seed'] and t['generation']==r['generation'] and t['size']==1)
        divergences.append(dict(source='all8-checkpoint-validation',seed=r['seed'],generation=r['generation'],
                                size=r['size'],**comparisons(ref['games'],r['games'])))
    reproducibility=[]
    for r in trials:
        if r['repeat']==0: continue
        ref=next(t for t in trials if t['repeat']==0 and t['seed']==r['seed'] and t['condition']==r['condition'])
        key=lambda d:(d['index'],d['ply'])
        exact=(r['games']==ref['games'] and r['rng_sha256']==ref['rng_sha256']
               and sorted(r['decisions'],key=key)==sorted(ref['decisions'],key=key))
        reproducibility.append(dict(seed=r['seed'],condition=r['condition'],repeat=r['repeat'],exact=exact))
    for r in trials:
        ref=next(t for t in trials if t['repeat']==r['repeat'] and t['seed']==r['seed'] and t['condition']=='serial')
        divergences.append(dict(source='primary-collection',seed=r['seed'],generation=r['generation'],
            repeat=r['repeat'],size=r['condition'],**comparisons(ref['games'],r['games'])))
    screens=[]
    for source in ('all8-checkpoint-validation','primary-collection'):
        rows=[r for r in divergences if r['source']==source and str(r['size']) not in ('1','serial')]
        total=sum(r['games'] for r in rows)
        trajectory=sum(r['trajectories'] for r in rows)/total
        outcome=sum(r['outcomes'] for r in rows)/total
        screens.append(dict(source=source,paired_game_comparisons=total,trajectory_rate=trajectory,outcome_rate=outcome,
            screen_pass=trajectory<=CRITERIA['trajectory_screen'] and outcome<=CRITERIA['outcome_screen']))
    search_rows=audit.read(args.numerical/'fixed_search.json')
    fixed_screens=[]
    for size in (1,2,4,8,16):
        rows=[r for r in search_rows if r['size']==size]
        for temperature in (0.,1.):
            decisions=[a for r in rows for a in r['actions'] if a['temperature']==temperature]
            disagreement=sum(a['serial']!=a['batched'] for a in decisions)/len(decisions)
            max_l1=max(r['visit_l1'] for r in rows)
            fixed_screens.append(dict(requested_size=size,maximum_actual_size=min(size,14),temperature=temperature,
                searches=len(rows),action_disagreement=disagreement,max_visit_l1=max_l1,
                screen_pass=disagreement<=CRITERIA['action_disagreement'] and max_l1<=CRITERIA['visit_l1']))
    # Per-size complete-game screens avoid dilution by singleton/reference cells.
    for source in ('all8-checkpoint-validation','primary-collection'):
        for size in ('2','4','8'):
            rows=[r for r in divergences if r['source']==source and str(r['size'])==size]
            total=sum(r['games'] for r in rows)
            trajectory=sum(r['trajectories'] for r in rows)/total
            outcome=sum(r['outcomes'] for r in rows)/total
            screens.append(dict(source=source,size=size,paired_game_comparisons=total,
                trajectory_rate=trajectory,outcome_rate=outcome,
                screen_pass=trajectory<=CRITERIA['trajectory_screen'] and outcome<=CRITERIA['outcome_screen']))
    primary_root_errors={}
    for size in ('1','2','4','8'):
        errors=dict(logits=0.,value=0.,policy=0.)
        changed_visits=changed_actions=positions=0
        for r in trials:
            if r['condition']!=size: continue
            ref=next(t for t in trials if t['repeat']==r['repeat'] and t['seed']==r['seed'] and t['condition']=='serial')
            ds={(d['index'],d['ply']):d for d in ref['decisions']}
            moves={g['index']:g['moves'] for g in r['games']}
            reference_moves={g['index']:g['moves'] for g in ref['games']}
            for d in r['decisions']:
                key=(d['index'],d['ply'])
                if key not in ds or moves[d['index']][:d['ply']]!=reference_moves[d['index']][:d['ply']]: continue
                e=ds[key]
                positions+=1
                errors['logits']=max(errors['logits'],max(abs(x-y) for x,y in zip(d['root_logits'],e['root_logits'])))
                errors['value']=max(errors['value'],abs(d['root_value']-e['root_value']))
                errors['policy']=max(errors['policy'],max(abs(x-y) for x,y in zip(d['root_prior'],e['root_prior'])))
                changed_visits+=d['visits']!=e['visits']
                changed_actions+=d['action']!=e['action']
        primary_root_errors[size]=dict(positions=positions,max_errors=errors,changed_visits=changed_visits,
            changed_actions=changed_actions,numerical_pass=all(errors[k]<=CRITERIA[k] for k in errors))
    numerical=audit.read(args.numerical/'summary.json')
    rates=audit.read(args.collection/'summary.json')
    best=max((k for k in rates if k!='serial'),key=lambda k:rates[k]['speedup'])
    results=dict(numerical=numerical,primary_root_errors=primary_root_errors,throughput=rates,best=best,best_speedup=rates[best]['speedup'],
        throughput_pass=rates[best]['speedup']>=1.3,fixed_screens=fixed_screens,divergences=divergences,divergence_screens=screens,
        reproducibility=reproducibility,reproducible=bool(reproducibility) and all(r['exact'] for r in reproducibility))
    audit.write(args.output/'summary.json',results)
    print(json.dumps({k:results[k] for k in ('best','best_speedup','throughput_pass','divergence_screens','reproducible')},indent=2))


if __name__=='__main__': main()
