import json,math,hashlib
from pathlib import Path
from collections import Counter
P=Path(__file__).resolve().parents[2];D=P/'research/alphazero_training_ablation';E=P/'experiment-output/phase4d4g-primary/execution'
read=lambda p:json.loads(p.read_text());out=read(D/'results/independent-recomputed.json');status=read(E/'status.json');F=read(D/'fixtures.json')
for s in ('solved','deep'):
 groups={}
 for r in F[s]:groups.setdefault(r['family'],[]).append(r)
 for rows in groups.values():
  assert len(rows)==2
  a,b=rows;assert b['moves']==[6-c for c in a['moves']]
  assert b['action_values']=={str(6-int(k)):v for k,v in a['action_values'].items()};assert b['value']==a['value']
  assert set(b['optimal_actions'])=={6-c for c in a['optimal_actions']}
info={};positions=neural=forwards=0
for seed in (42,314159):
 for arm in ('control','treatment'):
  for gen in (21,22):
   label=f'{seed}-{arm}-g{gen}';c=read(E/label/'complete.json');data=read(E/label/'games.json');ex=c['extra'];r=c['result'];g=data['games']
   assert len(g)==64 and len(r['metrics'])==64 and r['updates']==64
   n=sum(len(q['moves']) for q in g);assert n==ex['positions']==r['new_positions']==r['collection']['search']['decisions']
   assert all(len(q['visits'])==len(q['moves'])==len(q['action_temperatures']) and all(sum(v)==256 for v in q['visits']) for q in g)
   threshold=8 if arm=='control' else 16
   assert all(all(t==float(p<threshold) for p,t in enumerate(q['action_temperatures'])) for q in g)
   sample=r['diagnostics']['sampling']['samples_by_source_generation'];total=sum(sample.values());assert total==8192==ex['total_sample_exposure'];new=sum(v for k,v in sample.items() if int(k)>20);assert new==ex['experimental_sample_exposure']
   replay=sum(x['positions'] for x in ex['replay_counts'].values());rn=sum(x['positions'] for k,x in ex['replay_counts'].items() if int(k)>20);assert rn==ex['experimental_replay_positions']
   assert set(map(int,ex['replay_counts']))==set(range(gen-7,gen+1));assert r['evicted_generations']==[gen-8]
   opens={min(tuple(q['moves'][:8]),tuple(6-a for a in q['moves'][:8])) for q in g};assert len(opens)==ex['opening_families']
   entropy=sum(-sum((x/256)*math.log(x/256) for x in v if x) for q in g for v in q['visits'])/n;assert abs(entropy-ex['target_entropy'])<1e-12
   assert sum(q['winner']==0 for q in g)==ex['first_player_wins'];assert sum(q['winner']==1 for q in g)==ex['second_player_wins'];assert sum(q['winner']==-1 for q in g)==ex['draws']
   outcomes=Counter(0 if q['winner']==-1 else 1 if p%2==q['winner'] else -1 for q in g for p in range(len(q['moves'])))
   assert all(all(math.isfinite(float(v)) for k,v in m.items() if k!='clipped') for m in r['metrics'])
   assert all(abs(m['combined_loss']-m['policy_loss']-m['value_mse'])<2e-7 for m in r['metrics'])
   stages=c['stage_resources'];optimization=stages['before_checkpoint']['wall_seconds']-stages['before_update']['wall_seconds']
   info[label]={'positions':n,'mean_game_length':n/64,'opening_families':len(opens),'target_entropy':entropy,'actor_relative_outcome_counts':dict(outcomes),'replay_positions':replay,'new_replay_positions':rn,'new_replay_fraction':rn/replay,'new_sample_appearances':new,'sample_appearances':total,'new_sample_fraction':new/total,'sample_by_generation':sample,'optimizer_updates':len(r['metrics']),'max_gradient':max(m['gradient_norm'] for m in r['metrics']),'collection_seconds':r['resources']['collection_seconds'],'optimization_to_checkpoint_stage_seconds':optimization,'optimization_stage_caveat':'before_checkpoint minus first before_update includes training and summary/RNG bookkeeping, excludes persistence; no separately recorded pure optimization timer'}
   positions+=n;neural+=r['collection']['neural_positions'];forwards+=r['collection']['forward_calls']
assert all(t['state']=='COMPLETE' and t['resources']['complete'] for t in status['tasks']);assert len(status['tasks'])==22
child=sum(t['resources']['wall_seconds'] for t in status['tasks']);match_decisions=sum(len(r['decisions']) for seed in (42,314159) for a in ('control','treatment') for r in read(E/f'{seed}-{a}-strength/matches.json')['records'])
out['independent_training_diagnostics']=info;out['resource_crosscheck']={'manager_elapsed_seconds':status['elapsed_seconds'],'sum_child_task_wall_seconds':child,'supervisor_overhead_seconds':status['elapsed_seconds']-child,'task_counts':dict(Counter(t['task'] for t in status['tasks'])),'training_positions':positions,'training_puct_simulations':positions*256,'position_evaluation_puct_simulations':6280*512,'strength_decisions':match_decisions,'strength_puct_simulations':match_decisions*512,'total_puct_simulations':positions*256+(6280+match_decisions)*512,'collection_neural_positions':neural,'collection_forward_calls':forwards,'wall_ceiling_respected':status['elapsed_seconds']<10800,'generation_ceiling_respected':all(t['resources']['wall_seconds']<1200 for t in status['tasks'] if t['task']=='train')};out['checks']['mirrored_exact_vectors_validated']=428;out['checks']['training_game_budgets_schedules_checked']=512;out['checks']['training_update_metrics_checked']=512
(D/'results/independent-recomputed.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'resources':out['resource_crosscheck'],'diagnostics':info},indent=2))
