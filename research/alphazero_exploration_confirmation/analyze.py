"""All-decision exact proof collection and reproducible missingness-aware analysis."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import time

from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.oracle import engine_position
from research.alphazero_baseline import audit
from research.alphazero_exploration import analysis as a, replay
from research.alphazero_exploration.analyze import clean, ENDPOINTS, TACTICAL
from research.alphazero_batching.benchmark import digest, game_records, require
from .solver import NativeSession, classify, decision_interval
from . import schedules as s

ROOT=Path('research/alphazero_exploration_confirmation')

def key(state):
    return ':'.join(map(str,a.canonical(state.pieces)))

def set_value(values, k, v):
    if k in values and values[k]!=v: raise ValueError('Contradictory exact cached values')
    values[k]=v

def roots_for(games):
    roots={}
    values={}
    for g in games:
        state=BitboardState.from_game(engine_position([]))
        for p in g['plies']:
            if p['ply']>=8:
                k=key(state)
                roots.setdefault(k,dict(history=g['moves'][:p['ply']],ply=p['ply'],state=state))
                if p['wins']: set_value(values,k,1)
            state=state.drop(p['action'])
    return roots,values

def action_value(state, action, values):
    child=state.drop(action)
    if child.is_game_over(): return 0 if child.winner==-1 else 1
    v=values.get(key(child))
    return None if v is None else -v

def collect_proofs(games, output, ceiling=600):
    started=time.perf_counter()
    roots,values=roots_for(games)
    ordered=sorted(roots, key=lambda k:(-roots[k]['ply'], tuple(map(int,k.split(':')))))
    queries=[]
    for pass_id,(nodes,seconds) in enumerate(((200_000,.3),(2_000_000,.5)),1):
        session=NativeSession(nodes,seconds)
        try:
            for k in ordered:
                if time.perf_counter()-started>=ceiling: break
                r=roots[k]; state=r['state']
                # Cached successors can prove a full vector without another query.
                vector={c:action_value(state,c,values) for c in state.get_valid_moves()}
                if all(v is not None for v in vector.values()):
                    set_value(values,k,max(vector.values()))
                    continue
                q=session.query(r['history'],min(seconds,ceiling-(time.perf_counter()-started)))
                q.update(key=k,history=r['history'],ply=r['ply'],pass_id=pass_id,node_limit=nodes)
                queries.append(q)
                if q['status']=='exact':
                    require(set(q['actions'])==set(state.get_valid_moves()),'Oracle legal columns mismatch')
                    require(q['value']==max(q['actions'].values()),'Oracle max/action mismatch')
                    set_value(values,k,q['value'])
                    for c,v in q['actions'].items():
                        child=state.drop(c)
                        if child.is_game_over(): require(v==(0 if child.winner==-1 else 1),'Terminal perspective mismatch')
                        else: set_value(values,key(child),-v)
            print(f'Exact pass {pass_id}: {len(queries)} queries, {len(values)} scalar proofs, {time.perf_counter()-started:.1f}s',flush=True)
        finally: session.close()
        if time.perf_counter()-started>=ceiling: break
    proofs=dict(values=values,queries=queries,unique_eligible_states=len(roots),elapsed_seconds=time.perf_counter()-started,
                launch_ceiling_seconds=ceiling,exhausted_budget=time.perf_counter()-started>=ceiling,
                native_source_sha256=audit.sha(ROOT/'c4_oracle.c'),design_sha256=audit.sha(ROOT/'DESIGN.md'))
    audit.write(output/'proofs.json',proofs)
    return proofs

def sampling_comparison(executed, alternatives):
    if executed is None or any(v is None for v in alternatives): return dict(kind='unresolved')
    worse=sum(executed<v for v in alternatives)/len(alternatives)
    better=sum(executed>v for v in alternatives)/len(alternatives)
    equal=sum(executed==v for v in alternatives)/len(alternatives)
    kind='worse_all' if worse==1 else 'better_all' if better==1 else 'equal_all' if equal==1 else 'mixed_tie'
    return dict(kind=kind,worse_fraction=worse,better_fraction=better,equal_fraction=equal,
                regret=statistics.mean(alternatives)-executed)

def adjudicate(games, proofs):
    values=proofs['values']
    reasons={q['key']:q['reason'] for q in proofs['queries'] if q['status']!='exact'}
    decisions=[]
    for g in games:
        state=BitboardState.from_game(engine_position([])); counts=Counter()
        for p in g['plies']:
            if p['ply']>=8:
                k=key(state); before=values.get(k); after=action_value(state,p['action'],values)
                if before is not None and after is not None and after>before:
                    raise ValueError('Cached state/action exact proofs contradict minimax')
                if before==-1: after=-1   # Every action from an exactly lost state loses.
                if after==1: before=1     # Achievable win is the maximum possible value.
                kind=classify(before,after)
                top=sorted(p['top']); topvals=[action_value(state,c,values) for c in top]
                if before==-1: topvals=[-1]*len(top)
                attribution=None
                if kind not in ('neutral','unresolved'):
                    if any(v is None for v in topvals): attribution='unresolved'
                    elif all(v==before for v in topvals): attribution='sampling'
                    elif all(v<before for v in topvals): attribution='search'
                    else: attribution='mixed_tie'
                categories=['max_visit' if p['action'] in top else 'nonmax_visit',
                            'immediate_win_opportunity' if p['wins'] else 'mandatory_block_opportunity' if p['threats'] else 'other']
                comparison=sampling_comparison(after,topvals) if g['schedule']=='extended' and p['ply']<16 else None
                row=dict(id=g['id'],schedule=g['schedule'],ply=p['ply'],stage=a.stage_of(p['ply']),key=k,
                         history=g['moves'][:p['ply']],action=p['action'],categories=categories,before=before,
                         action_value=after,classification=kind,top=top,top_values=topvals,attribution=attribution,
                         sampling_comparison=comparison,
                         unresolved_reason=None if kind!='unresolved' else
                         'pre_or_successor_'+str(reasons.get(k,'cache_value_unavailable_or_total_budget')))
                decisions.append(row)
                counts['eligible']+=1; counts[kind]+=1
                if kind not in ('neutral','unresolved'):
                    counts['errors']+=1; counts['attribution:'+attribution]+=1
                if comparison:
                    counts['sampling_comparisons']+=1
                    counts['nonmax_sampled']+=p['action'] not in top
                    counts['sampling:'+comparison['kind']]+=1
            state=state.drop(p['action'])
        g['exact']=dict(counts)
    return decisions

def mean_se(values):
    return statistics.mean(values),math.sqrt(statistics.variance(values)/len(values)) if len(values)>1 else float('inf')

def effect_bounds(reference,candidate):
    pairs=a.pair(reference,candidate)
    confirmed=[c['exact'].get('errors',0)-r['exact'].get('errors',0) for r,c in pairs]
    lows=[d-r['exact'].get('unresolved',0) for d,(r,c) in zip(confirmed,pairs)]
    highs=[d+c['exact'].get('unresolved',0) for d,(r,c) in zip(confirmed,pairs)]
    cm,cs=mean_se(confirmed); lm,ls=mean_se(lows); hm,hs=mean_se(highs)
    return dict(n=len(pairs),confirmed_difference=cm,confirmed_se=cs,confirmed_interval=[cm-1.96*cs,cm+1.96*cs],
                identification_interval=[lm,hm],low_se=ls,high_se=hs,
                uncertainty_interval=[lm-1.96*ls,hm+1.96*hs])

def pooled_effect(parts):
    if len(parts)!=2: raise ValueError('Both checkpoints required')
    cm=statistics.mean(p['confirmed_difference'] for p in parts)
    cs=math.sqrt(sum(p['confirmed_se']**2 for p in parts))/2
    lm=statistics.mean(p['identification_interval'][0] for p in parts)
    hm=statistics.mean(p['identification_interval'][1] for p in parts)
    ls=math.sqrt(sum(p['low_se']**2 for p in parts))/2
    hs=math.sqrt(sum(p['high_se']**2 for p in parts))/2
    return dict(n=sum(p['n'] for p in parts),confirmed_difference=cm,confirmed_se=cs,
                confirmed_interval=[cm-1.96*cs,cm+1.96*cs],identification_interval=[lm,hm],
                uncertainty_interval=[lm-1.96*ls,hm+1.96*hs],
                achieved_confirmed_half_width=1.96*cs)

def coverage(decisions):
    buckets=defaultdict(Counter)
    for r in decisions:
        seed=r['id'][0]; schedule=r['schedule']
        names=['all',f'checkpoint:{seed}',f'schedule:{schedule}',f'cell:{seed}:{schedule}',
               f'ply:{r["ply"]}',f'stage:{r["stage"]}',f'cell_stage:{seed}:{schedule}:{r["stage"]}']
        names.extend('category:'+c for c in r['categories'])
        names.extend(f'cell_category:{seed}:{schedule}:{c}' for c in r['categories'])
        for name in names:
            c=buckets[name]; c['eligible']+=1
            c['unresolved' if r['classification']=='unresolved' else 'resolved']+=1
            if r['classification']!='unresolved': c[r['classification']]+=1
            c['pre_value_exact']+=r.get('before') is not None
            c['executed_value_exact']+=r.get('action_value') is not None
        if r['classification']=='unresolved': buckets['reasons'][r['unresolved_reason']]+=1
    return {name:dict(c,coverage=c.get('resolved',0)/c['eligible']) if 'eligible' in c else dict(c) for name,c in sorted(buckets.items())}

def verify_records(games,cells):
    grouped=defaultdict(list)
    for g in games: grouped[g['id'][:3],g['schedule']].append(g)
    checked=0
    for cell in cells:
        if not cell['complete']: continue
        members=sorted(grouped[(cell['seed'],cell['generation'],cell['cohort']),cell['schedule']['name']],key=lambda g:g['id'])
        if not members: continue # Entire unmatched block excluded, never only its adverse games.
        completed=replay.completed_games(members,s.GENERATION)
        require(digest(game_records(completed))==cell['records_sha256'],'Cell record hash mismatch')
        for g,c in zip(members,completed): require(digest(game_records([c]))==g['record_sha256'],'Game record hash mismatch'); checked+=1
    return checked

def matched_games(games,cells):
    by_block=defaultdict(dict)
    for c in cells: by_block[(c['seed'],c['generation'],c['cohort'])][c['schedule']['name']]=c
    accepted=set(); excluded=[]
    for block,arms in sorted(by_block.items()):
        if set(arms)==set(s.BY_NAME) and all(c['complete'] for c in arms.values()) and arms['original']['ids']==arms['extended']['ids']:
            accepted.add(block)
        else: excluded.append(dict(block=block,reason='incomplete_or_unmatched_checkpoint_cohort'))
    return [g for g in games if g['id'][:3] in accepted],excluded

def safety_conclusion(pooled,parts,adequate,complete):
    if not complete: return 'INCONCLUSIVE'
    conclusion=decision_interval(*pooled['uncertainty_interval'])
    if conclusion.startswith('NO MATERIAL') and (not adequate or any(p['uncertainty_interval'][1]>=.05 for p in parts)):
        return 'INCONCLUSIVE'
    return conclusion

def summarize(games,decisions,cells,excluded=None):
    data=defaultdict(lambda:defaultdict(list))
    for g in games: data[g['id'][0]][g['schedule']].append(g)
    primary={str(seed):effect_bounds(by['original'],by['extended']) for seed,by in sorted(data.items())}
    pooled=pooled_effect(list(primary.values()))
    cov=coverage(decisions)
    adequate=all(v['coverage']>=.99 for k,v in cov.items() if k.startswith('cell_stage:'))
    complete=all(c['complete'] for c in cells) and len(cells)==8 and len(games)==512
    conclusion=safety_conclusion(pooled,list(primary.values()),adequate,complete and not excluded)
    profiles={}; secondary={}; cohorts={}
    for seed,by in sorted(data.items()):
        profiles[str(seed)]={}
        for name,group in by.items():
            counts=Counter()
            for g in group: counts.update(g['exact'])
            profiles[str(seed)][name]=dict(games=len(group),exact_counts=dict(counts),
                confirmed_errors_per_game=counts['errors']/len(group),
                error_rate_bounds=[counts['errors']/len(group),(counts['errors']+counts['unresolved'])/len(group)],
                diversity=a.diversity(group),tactics=a.tactical_profile(group))
        endpoints={n:fn for n,fn in ENDPOINTS.items() if n in ('d1_duplicate_fraction_8_plus','d2_mirror_duplicate_fraction_8_plus','d3_collision_probability_8_plus','mean_length','target_entropy_8_plus')}
        endpoints.update(TACTICAL)
        endpoints.update(x_win_share=lambda gs:sum(g['winner']==0 for g in gs)/len(gs),
                         o_win_share=lambda gs:sum(g['winner']==1 for g in gs)/len(gs),
                         draw_share=lambda gs:sum(g['winner']==-1 for g in gs)/len(gs))
        secondary[str(seed)]={n:a.paired_difference(by['original'],by['extended'],fn) for n,fn in endpoints.items()}
        cohorts[str(seed)]={}
        for cohort in (0,1):
            groups={name:[g for g in group if g['id'][2]==cohort] for name,group in by.items()}
            if not all(groups.values()):
                cohorts[str(seed)][str(cohort)]=dict(excluded=True)
                continue
            cohorts[str(seed)][str(cohort)]=dict(primary=effect_bounds(groups['original'],groups['extended']),
                duplicate_fraction={name:a.duplicate_fraction(group,8,42) for name,group in groups.items()})
    prefix_mismatches=[]
    for seed,by in data.items():
        for r,c in a.pair(by['original'],by['extended']):
            if r['moves'][:8]!=c['moves'][:8] or any(x['visits']!=y['visits'] or x['key']!=y['key'] or x['prior']!=y['prior'] or x['value']!=y['value'] or x['distribution']!=y['distribution'] for x,y in zip(r['plies'][:8],c['plies'][:8])):
                prefix_mismatches.append(r['id'])
    require(not prefix_mismatches,'Matched pre-ply8 decision mismatch')
    sampling=[r for r in decisions if r['sampling_comparison'] is not None]
    sc=Counter(r['sampling_comparison']['kind'] for r in sampling)
    resolved=[r for r in sampling if r['sampling_comparison']['kind']!='unresolved']
    diversity_confirmed=all(secondary[str(seed)]['d1_duplicate_fraction_8_plus']['high']<0 and
        all(secondary[str(seed)][n]['estimate']<0 for n in ('d2_mirror_duplicate_fraction_8_plus','d3_collision_probability_8_plus')) and
        all(v.get('excluded') or v['duplicate_fraction']['extended']<v['duplicate_fraction']['original'] for v in cohorts[str(seed)].values()) for seed in data)
    unresolved=[r for r in decisions if r['classification']=='unresolved']
    return clean(dict(primary=dict(by_checkpoint=primary,pooled=pooled,margin=.05,conclusion=conclusion,adequate_coverage=adequate,complete_design=complete),
        profiles=profiles,secondary_paired=secondary,cohorts=cohorts,coverage=cov,
        sampling=dict(eligible=len(sampling),nonmax_selected=sum(r['action'] not in r['top'] for r in sampling),comparisons=dict(sc),
            resolved=len(resolved),uniform_top_mean_regret=statistics.mean(r['sampling_comparison']['regret'] for r in resolved) if resolved else None,
            tie_states=sum(len(r['top'])>1 for r in sampling),
            expected_worse_count=sum(r['sampling_comparison']['worse_fraction'] for r in resolved),
            expected_better_count=sum(r['sampling_comparison']['better_fraction'] for r in resolved),
            expected_equal_count=sum(r['sampling_comparison']['equal_fraction'] for r in resolved)),
        correctness=dict(checked_records=verify_records(games,cells),paired_prefix_mismatches=prefix_mismatches,
                         paired_games=sum(p['n'] for p in primary.values())),
        diversity_confirmed=diversity_confirmed,unresolved_decisions=len(unresolved),excluded_blocks=excluded or []))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--proofs',type=Path)
    parser.add_argument('--ceiling',type=float,default=600)
    args=parser.parse_args(); started=time.perf_counter()
    require(0 < args.ceiling <= 600, 'Frozen solver launch ceiling is at most 600 seconds')
    audit.torch.set_num_threads(1)
    audit.torch.set_num_interop_threads(1)
    audit.torch.use_deterministic_algorithms(True)
    output=audit.fresh_output(args.campaign,args.output)
    before=audit.campaign_fingerprint(args.campaign)
    require(len(before)==150 and before==audit.read('research/alphazero_baseline/results/phase4d4a-20261011/profile/campaign_before.json'),'Campaign inventory mismatch')
    require(audit.sha(args.campaign/'declaration.json')==audit.DECLARATION,'Declaration mismatch')
    try:
        games,cells,skipped=a.load_collection(args.collection)
        games,excluded=matched_games(games,cells)
        if {g['id'][0] for g in games}!={42,314159}:
            audit.write(output/'results.json',dict(primary=dict(conclusion='INCONCLUSIVE',reason='insufficient_matched_checkpoints'),excluded_blocks=excluded))
            return
        proofs=json.loads(args.proofs.read_text()) if args.proofs else collect_proofs(games,output,args.ceiling)
        require(proofs['design_sha256']==audit.sha(ROOT/'DESIGN.md'),'Proof/design hash mismatch')
        decisions=adjudicate(games,proofs)
        audit.write(output/'results.json',summarize(games,decisions,cells,excluded))
        for name,rows in (('decisions',decisions),('unresolved',[r for r in decisions if r['classification']=='unresolved'])):
            with gzip.open(output/f'{name}.jsonl.gz','xt') as stream:
                for r in rows: stream.write(json.dumps(r,sort_keys=True)+'\n')
        audit.write(output/'provenance.json',dict(commit=audit.command('git','rev-parse','HEAD'),
             design_sha256=audit.sha(ROOT/'DESIGN.md'),sources={str(p):audit.sha(p) for p in ROOT.glob('*.py')},
             collection_provenance_sha256=audit.sha(args.collection/'provenance.json'),runtime=audit.runtime(),elapsed_seconds=time.perf_counter()-started,
             reused_proofs=args.proofs is not None,solver_seconds=proofs['elapsed_seconds'] if not args.proofs else 0))
    finally:
        after=audit.campaign_fingerprint(args.campaign)
        audit.write(output/'preservation.json',dict(files=len(after),unchanged=before==after,changed=[k for k in before if before[k]!=after.get(k)],declaration_sha256=audit.sha(args.campaign/'declaration.json')))
        require(before==after,'Campaign modified')
    print(json.dumps(dict(elapsed_seconds=time.perf_counter()-started,output=str(output))),flush=True)
if __name__=='__main__': main()
