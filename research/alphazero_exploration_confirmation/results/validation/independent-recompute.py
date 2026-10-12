"""Independent review arithmetic from raw games and committed proof outputs.

Does no new solving or collection and imports no experiment analysis code.
"""
from collections import Counter, defaultdict
import argparse, gzip, hashlib, json, math, statistics, time
from pathlib import Path
from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2.bitboard import BitboardState

ROOT=Path('research/alphazero_exploration_confirmation/results')

def read(p): return json.loads(p.read_text())
def mirror(b): return sum(((b>>(7*c))&127)<<(7*(6-c)) for c in range(7))
def canon(st): return min(st.pieces,tuple(mirror(b) for b in st.pieces))
def key(st): return ':'.join(map(str,canon(st)))
def mean_se(xs): return statistics.mean(xs),math.sqrt(statistics.variance(xs)/len(xs))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--analysis',type=Path,default=ROOT/'exact');parser.add_argument('--output',type=Path,default=ROOT/'validation/independent-recompute.json');args=parser.parse_args()
 started=time.perf_counter(); cells=read(ROOT/'main/cells.json'); proofs=read(ROOT/'exact/proofs.json'); results=read(args.analysis/'results.json')
 games=[]; state_by_decision={}; roots={}; decoded=0
 for cell in cells:
  assert cell['complete']; file=ROOT/'main'/cell['file']; text=file.read_text() if file.exists() else gzip.open(str(file)+'.gz','rt').read()
  records=[json.loads(l) for l in text.splitlines()]; assert sorted(r['index'] for r in records)==cell['ids']
  for record in records:
   g=dict(record,seed=cell['seed'],cohort=cell['cohort'],schedule=cell['schedule']['name'],id=(cell['seed'],cell['generation'],cell['cohort'],record['index']))
   game=Connect4(); states=[]
   for ply,(move,row,z) in enumerate(zip(g['moves'],g['plies'],g['outcomes'])):
    assert not game.is_game_over(); st=BitboardState.from_game(game); states.append(st)
    assert row['ply']==ply and row['action']==move and move in game.get_valid_moves()
    visits=row['visits']; assert len(visits)==7 and sum(visits)==256 and all(type(n) is int and n>=0 for n in visits)
    assert all(visits[c]==0 for c in range(7) if c not in game.get_valid_moves()) and visits[move]>0
    cutoff=8 if g['schedule']=='original' else 16; tau=1.0 if ply<cutoff else 0.0
    assert row['temperature']==tau
    top=[c for c,n in enumerate(visits) if n==max(visits)]
    expected=[n/256 if tau else float(c in top)/len(top) for c,n in enumerate(visits)]
    assert max(abs(x-y) for x,y in zip(row['distribution'],expected))<=1e-12
    assert z==(0 if g['winner']==-1 else 1 if g['winner']==ply%2 else-1)
    if ply>=8: state_by_decision[g['id'],g['schedule'],ply]=st; roots[key(st)]=st
    assert game.make_move(move)
   assert game.is_game_over() and game.check_winner()==g['winner']
   assert len(g['moves'])==len(g['plies'])==len(g['outcomes']); g['states']=states; games.append(g); decoded+=1
 assert decoded==512
 byid={(g['id'],g['schedule']):g for g in games};diagnostic_count=0
 for g in games:
  if g['schedule']!='original': continue
  c=byid[g['id'],'extended']; assert g['moves'][:8]==c['moves'][:8]
  for x,y in zip(g['plies'][:8],c['plies'][:8]):
   assert all(x[k]==y[k] for k in x if k!='root_value')
   assert abs(x['root_value']-y['root_value'])<=2e-6
   diagnostic_count+=x['root_value']!=y['root_value']
 # Independently reconstruct scalar proofs from exact action vectors plus one-ply wins.
 known={}
 def put(k,v):
  assert v in(-1,0,1)
  assert k not in known or known[k]==v
  known[k]=v
 for k,st in roots.items():
  if any(st.drop(c).winner==st.current_player for c in st.get_valid_moves()): put(k,1)
 for q in proofs['queries']:
  if q['status']!='exact': assert q['value'] is None and not q['actions']; continue
  game=Connect4()
  for c in q['history']: assert not game.is_game_over() and game.make_move(c)
  st=BitboardState.from_game(game); assert key(st)==q['key']; acts={int(c):v for c,v in q['actions'].items()}
  assert set(acts)==set(st.get_valid_moves()) and q['value']==max(acts.values()); put(key(st),q['value'])
  for c,v in acts.items():
   child=st.drop(c)
   if child.is_game_over(): assert v==(0 if child.winner==-1 else 1)
   else: put(key(child),-v)
 changed=True
 while changed:
  changed=False
  for k,st in roots.items():
   av=[]
   for c in st.get_valid_moves():
    child=st.drop(c); v=(0 if child.winner==-1 else 1) if child.is_game_over() else(-known[key(child)] if key(child) in known else None)
    av.append(v)
   if all(v is not None for v in av) and k not in known: put(k,max(av));changed=True
 assert all(known.get(k)==v for k,v in proofs['values'].items())
 decisions=[json.loads(l) for l in gzip.open(args.analysis/'decisions.jsonl.gz','rt')]; counts=defaultdict(Counter); sample=Counter(); stages=defaultdict(Counter)
 def action(st,c):
  child=st.drop(c)
  return(0 if child.winner==-1 else 1) if child.is_game_over() else(-proofs['values'][key(child)] if key(child) in proofs['values'] else None)
 for row in decisions:
  ident=tuple(row['id']); st=state_by_decision[ident,row['schedule'],row['ply']]; before=proofs['values'].get(key(st)); after=action(st,row['action'])
  if before==-1:
   assert after in (None,-1)
   after=-1
  if after==1:
   assert before in (None,1)
   before=1
  kind='unresolved' if before is None or after is None else'neutral' if before==after else{(1,0):'win_draw',(1,-1):'win_loss',(0,-1):'draw_loss'}[before,after]
  assert(before,after,kind)==(row['before'],row['action_value'],row['classification'])
  c=counts[ident,row['schedule']]; c['eligible']+=1;c[kind]+=1;c['errors']+=kind not in('neutral','unresolved')
  b=stages[ident[0],row['schedule'],row['stage']];b['eligible']+=1;b['unresolved']+=kind=='unresolved'
  vals=[action(st,c) for c in row['top']]
  if before==-1: vals=[-1]*len(vals)
  assert vals==row['top_values']
  if kind not in('neutral','unresolved'):
   attrib='unresolved' if any(v is None for v in vals) else'sampling' if all(v==before for v in vals) else'search' if all(v<before for v in vals) else'mixed_tie'
   assert attrib==row['attribution']
  if row['sampling_comparison'] is not None:
   comp='unresolved' if after is None or any(v is None for v in vals) else'worse_all' if all(after<v for v in vals) else'better_all' if all(after>v for v in vals) else'equal_all' if all(after==v for v in vals) else'mixed_tie'
   assert comp==row['sampling_comparison']['kind'];sample[comp]+=1
   if comp!='unresolved': assert abs(statistics.mean(vals)-after-row['sampling_comparison']['regret'])<1e-12
 assert len(decisions)==sum(max(len(g['moves'])-8,0) for g in games)
 checkeffects=[]; profiles={}; discrepancies=[]
 for seed in(42,314159):
  ids=sorted(g['id'] for g in games if g['seed']==seed and g['schedule']=='original'); ds=[];lo=[];hi=[]
  for ident in ids:
   r=counts[ident,'original'];c=counts[ident,'extended'];d=c['errors']-r['errors'];ds.append(d);lo.append(d-r['unresolved']);hi.append(d+c['unresolved'])
  dm,dsd=mean_se(ds);lm,ls=mean_se(lo);hm,hs=mean_se(hi); effect={'confirmed_difference':dm,'confirmed_se':dsd,'identification_interval':[lm,hm],'uncertainty_interval':[lm-1.96*ls,hm+1.96*hs]}
  official=results['primary']['by_checkpoint'][str(seed)]
  for n,v in effect.items():
   assert official[n]==v,(seed,n,v,official[n])
  checkeffects.append((dm,dsd,lm,ls,hm,hs));profiles[str(seed)]={}
  for schedule in('original','extended'):
   group=[g for g in games if g['seed']==seed and g['schedule']==schedule]; rows=[st for g in group for st in g['states'][8:]]; positions=Counter(st.pieces for st in rows);canons=Counter(canon(st) for st in rows);reached=Counter(ply for g in group for ply in range(8,len(g['moves'])))
   duplicate=1-len(positions)/len(rows);mirror_dup=1-len(canons)/len(rows);collision=sum(n*(n-1) for n in positions.values())/sum(n*(n-1) for n in reached.values())
   winner=Counter(g['winner'] for g in group); totals=Counter()
   for g in group: totals.update(counts[g['id'],schedule])
   p=results['profiles'][str(seed)][schedule];assert duplicate==p['diversity']['stages']['8+']['duplicate_fraction'];assert mirror_dup==p['diversity']['stages']['8+']['mirror_canonical_duplicate_fraction'];assert collision==p['diversity']['stages']['8+']['collision_probability'];assert p['exact_counts']['errors']==totals['errors'] and p['exact_counts'].get('unresolved',0)==totals['unresolved']
   assert p['diversity']['results']==dict(x_wins=winner[0],o_wins=winner[1],draws=winner[-1])
   profiles[str(seed)][schedule]=dict(games=len(group),confirmed_errors=totals['errors'],unresolved=totals['unresolved'],duplicate_fraction=duplicate,mirror_duplicate_fraction=mirror_dup,collision_probability=collision,actual_outcomes=dict(winner))
 dm=statistics.mean(x[0] for x in checkeffects);ds=math.sqrt(sum(x[1]**2 for x in checkeffects))/2;lm=statistics.mean(x[2] for x in checkeffects);ls=math.sqrt(sum(x[3]**2 for x in checkeffects))/2;hm=statistics.mean(x[4] for x in checkeffects);hs=math.sqrt(sum(x[5]**2 for x in checkeffects))/2
 pooled=dict(confirmed_difference=dm,confirmed_interval=[dm-1.96*ds,dm+1.96*ds],identification_interval=[lm,hm],uncertainty_interval=[lm-1.96*ls,hm+1.96*hs]);assert all(results['primary']['pooled'][k]==v for k,v in pooled.items())
 sufficient=all(1-c['unresolved']/c['eligible']>=.99 for c in stages.values()); interval=pooled['uncertainty_interval'];conclusion='MATERIAL HARM' if interval[0]>.05 else'NO MATERIAL HARM WITHIN THE DECLARED MARGIN' if interval[1]<.05 and sufficient and all(results['primary']['by_checkpoint'][str(seed)]['uncertainty_interval'][1]<.05 for seed in(42,314159)) else'INCONCLUSIVE'
 if conclusion.startswith('NO MATERIAL') and diagnostic_count:conclusion='INCONCLUSIVE'
 assert conclusion==results['primary']['conclusion'];assert dict(sample)==results['sampling']['comparisons']
 assert results['correctness']['literal_full_record_prefix_gate_pass']==(diagnostic_count==0)
 out=dict(games_checked=decoded,decisions_checked=len(decisions),all_values_independently_reconstructed=True,paired_prefix_mismatches=0,paired_prefix_decisions=sum(min(8,len(g['moves'])) for g in games if g['schedule']=='original'),root_value_diagnostic_mismatches=diagnostic_count,profiles=profiles,pooled=pooled,safety_conclusion=conclusion,sampling_comparisons=dict(sample),numeric_discrepancies=0,elapsed_seconds=time.perf_counter()-started,new_solves=0,new_games=0)
 with args.output.open('x') as f: json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(out,sort_keys=True))
if __name__=='__main__':main()
