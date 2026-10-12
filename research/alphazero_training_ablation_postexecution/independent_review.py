import json, math, hashlib
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
P=Path(__file__).resolve().parents[2]
D=P/'research/alphazero_training_ablation'; E=P/'experiment-output/phase4d4g-primary/execution'
read=lambda p:json.loads(Path(p).read_text())
F=read(D/'fixtures.json'); checks=Counter(); evidence={}
def board(moves):
 b=[[None]*6 for _ in range(7)]
 for n,c in enumerate(moves):
  h=sum(v is not None for v in b[c]); assert h<6; b[c][h]=n%2
 return b
def winner(b,a):
 for c in range(7):
  for h in range(6):
   for dc,dh in ((1,0),(0,1),(1,1),(1,-1)):
    if all(0<=c+i*dc<7 and 0<=h+i*dh<6 and b[c+i*dc][h+i*dh]==a for i in range(4)): return True
 return False
def legal(b): return [c for c in range(7) if b[c][5] is None]
def play(b,c,a):
 q=[v[:] for v in b]; q[c][sum(v is not None for v in q[c])]=a; return q
def wins(b,a):return [c for c in legal(b) if winner(play(b,c,a),a)]
def safe(b,a):return [c for c in legal(b) if winner(play(b,c,a),a) or not wins(play(b,c,a),1-a)]
# Independent graph-connected components: mirror families or equivalent prefix ancestry.
for s in ('solved','tactical','deep'):
 rows=F[s]; n=len(rows); adj=[set() for _ in rows]
 for i in range(n):
  a=rows[i]; p=a['moves'][:8]
  for j in range(i):
   q=rows[j]['moves'][:8]; k=min(len(p),len(q))
   if a['family']==rows[j]['family'] or p[:k]==q[:k] or p[:k]==[6-c for c in q[:k]]:adj[i].add(j);adj[j].add(i)
 seen=set(); parts=[]
 for i in range(n):
  if i in seen:continue
  stack=[i]; component=set()
  while stack:
   v=stack.pop()
   if v in component:continue
   component.add(v); stack.extend(adj[v]-component)
  seen|=component;parts.append(component)
 actual={frozenset(i for i,r in enumerate(rows) if r['cluster']==c) for c in {r['cluster'] for r in rows}}
 assert actual=={frozenset(c) for c in parts}
 evidence[s+'_clusters']={'rows':n,'clusters':len(parts),'sizes':dict(sorted(Counter(map(len,parts)).items()))}
 for r in rows:
  b=board(r['moves']);assert not winner(b,0) and not winner(b,1)
  if s=='tactical':
   assert safe(b,len(r['moves'])%2)==r['safe_actions'];assert wins(b,len(r['moves'])%2)==r['immediate_wins']
   assert (r['category']=='unique_win' and len(r['immediate_wins'])==1 and r['expected_action']==r['immediate_wins'][0]) or (r['category']=='unique_safe' and len(r['safe_actions'])==1 and r['expected_action']==r['safe_actions'][0])
  else:
   v={int(c):x for c,x in r['action_values'].items()};assert set(v)==set(legal(b));assert set(v.values())<={-1,0,1}
   assert max(v.values())==r['value'];assert set(r['optimal_actions'])=={c for c,x in v.items() if x==r['value']}
  checks['fixture_rows_validated']+=1
# Independently reconstruct metric fields; no imports from phase analysis, model or solver.
R={}
for task in ['parent-42','parent-314159']+[f'{s}-{a}-g{g}-evaluation' for s in (42,314159) for a in ('control','treatment') for g in (21,22)]:
 R[task]={}
 for suite in ('solved','tactical','deep'):
  raw=read(E/task/(suite+'.json'));assert raw['complete'] is True
  rows=raw['rows'];assert len({r['id'] for r in rows})==len(rows)==len(F[suite])
  fx={r['id']:r for r in F[suite]};assert set(fx)=={r['id'] for r in rows}
  out=[]
  for r in rows:
   f=fx[r['id']];b=board(f['moves']);assert r['cluster']==f['cluster'] and r['family']==f['family']
   assert len(r['policy'])==7 and all(x>=0 and math.isfinite(x) for x in r['policy']) and abs(sum(r['policy'])-1)<1e-12
   assert all(r['policy'][c]==0 for c in set(range(7))-set(legal(b)))
   assert r['policy_action'] in legal(b) and r['puct_action'] in legal(b)
   assert r['policy'][r['policy_action']]==max(r['policy']);assert sum(r['visits'])==512
   t={'id':r['id'],'cluster':f['cluster'],'category':f.get('category')}
   if suite!='tactical':
    mass=sum(r['policy'][a] for a in f['optimal_actions']);t.update(ce=-math.log(max(mass,1e-300)),optimal_mass=mass,accuracy=int(r['policy_action'] in f['optimal_actions']),mse=(r['value']-f['value'])**2)
   for method in ('policy','puct'):
    a=r[method+'_action']
    if suite=='tactical':t.update({method+'_correct':int(a==f['expected_action']),method+'_avoidable':int(a not in safe(b,len(f['moves'])%2)),method+'_missed_win':int(bool(wins(b,len(f['moves'])%2)) and a not in wins(b,len(f['moves'])%2))})
    else:t[method+'_deep_mistake']=int(f['action_values'][str(a)]<f['value'])
   for k,v in t.items():
    if k in ('id','cluster','category'):continue
    assert abs(v-r[k])<1e-12,(task,suite,r['id'],k)
   checks['evaluation_rows_recomputed']+=1;out.append(t)
  R[task][suite]=out
# Multinomial multiplicity matrix independent from frozen totals[draws] implementation.
BOOT={}
def boot(rows,fields,seed=4404703):
 labels=sorted({r['cluster'] for r in rows});idx={c:i for i,c in enumerate(labels)}; sums=np.zeros((len(labels),len(fields))); counts=np.zeros(len(labels))
 for r in rows:
  i=idx[r['cluster']];counts[i]+=1
  sums[i]+=[r[f] for f in fields]
 key=(tuple(labels),seed)
 if key not in BOOT:
  draws=np.random.default_rng(seed).integers(len(labels),size=(10000,len(labels)))
  multiplicity=np.zeros((10000,len(labels)))
  for i in range(10000): multiplicity[i]=np.bincount(draws[i],minlength=len(labels))
  BOOT[key]=multiplicity
 m=BOOT[key];dist=(m@sums)/(m@counts)[:,None]
 return {f:{'mean':float(sums[:,j].sum()/counts.sum()),'ci95':np.quantile(dist[:,j],[.025,.975]).tolist(),'one_sided95_lower':float(np.quantile(dist[:,j],.05)),'one_sided95_upper':float(np.quantile(dist[:,j],.95)),'families':len(labels),'positions':len(rows)} for j,f in enumerate(fields)}
def diff(a,b,fields):
 bm={r['id']:r for r in b};assert set(bm)=={r['id'] for r in a}
 return [{'id':r['id'],'cluster':r['cluster'],**{f:r[f]-bm[r['id']][f] for f in fields}} for r in a]
def ni(r,margin,lowgood=True):
 if lowgood:return 'PASS' if r['one_sided95_upper']<=margin else 'FAIL' if r['one_sided95_lower']>margin else 'INCONCLUSIVE'
 return 'PASS' if r['one_sided95_lower']>=margin else 'FAIL' if r['one_sided95_upper']<margin else 'INCONCLUSIVE'
fields=('ce','accuracy','mse','optimal_mass');learning={};tactical={};strength={};g22=[];tf=[];sf=[];learning_fail=False
for seed in (42,314159):
 learning[str(seed)]={};parent=R[f'parent-{seed}']['solved'];learning[str(seed)]['parent']=boot(parent,fields)
 for g in (21,22):
  arms={a:R[f'{seed}-{a}-g{g}-evaluation'] for a in ('control','treatment')}
  for a in arms:learning[str(seed)][f'{a}-g{g}']={'metrics':boot(arms[a]['solved'],fields),'change_from_parent':boot(diff(arms[a]['solved'],parent,fields),fields)}
  d=diff(arms['treatment']['solved'],arms['control']['solved'],fields);c=boot(d,fields);learning[str(seed)][f'contrast-g{g}']=c
  if g==22:
   g22.append({r['id']:r for r in d});learning_fail|=c['ce']['ci95'][0]>0 or c['accuracy']['ci95'][1]<0 or c['mse']['mean']>.02
  for method in ('policy','puct'):
   for cat in ('unique_win','unique_safe'):
    field=method+'_correct';t=[r for r in arms['treatment']['tactical'] if r['category']==cat];c=[r for r in arms['control']['tactical'] if r['category']==cat];p=[r for r in R[f'parent-{seed}']['tactical'] if r['category']==cat]
    dc=boot(diff(t,c,(field,)),(field,))[field];dp=boot(diff(t,p,(field,)),(field,))[field];label='PASS' if min(dc['mean'],dp['mean'])>=-.02-1e-12 else 'FAIL';tf.append(label)
    tactical[f'{seed}-g{g}-{method}-{cat}']={'label':label,'vs_control':dc,'vs_parent':dp,'treatment':boot(t,(field,))[field],'control':boot(c,(field,))[field],'parent':boot(p,(field,))[field]}
   for suite in ('solved','deep'):
    f=method+'_deep_mistake';c=boot(diff(arms['treatment'][suite],arms['control'][suite],(f,)),(f,))[f];label=ni(c,.05);tf.append(label)
    tactical[f'{seed}-g{g}-{method}-{suite}-deep']={'label':label,'vs_control':c}
 scores={}
 for a in ('control','treatment'):
  d=read(E/f'{seed}-{a}-strength/matches.json');assert d['complete'];m=d['records'];assert len(m)==64
  fm={r['id']:r for r in F['openings']};group=defaultdict(list);srows=[]
  for r in m:
   f=fm[r['opening_id']];assert r['family']==f['family'] and r['agent_color'] in (0,1) and not r['abandoned']
   group[r['opening_id']].append(r['agent_color']);assert all(sum(d['visits'])==512 for d in r['decisions'])
   assert len(r['decisions'])==len(r['moves'])-len(f['moves'])
   b=board(r['moves']);iswin=winner(b,r['agent_color']);isloss=winner(b,1-r['agent_color'])
   expected='win' if iswin else 'loss' if isloss else 'draw';assert r['result']==expected
   assert iswin or isloss or len(r['moves'])==42
   score={'win':1.,'draw':.5,'loss':0.}[expected];srows.append({'id':r['opening_id']+'-'+str(r['game_index']),'cluster':r['family'],'score':score,'color':r['agent_color']})
  assert set(group)==set(fm) and all(sorted(v)==[0,1] for v in group.values());checks['strength_games_recomputed']+=64
  scores[a]=srows
  strength.setdefault(str(seed),{})[a]={'wins':sum(r['result']=='win' for r in m),'draws':sum(r['result']=='draw' for r in m),'losses':sum(r['result']=='loss' for r in m),'metrics':boot(srows,('score',)),'by_color':{str(c):boot([r for r in srows if r['color']==c],('score',)) for c in (0,1)}}
 c=boot(diff(scores['treatment'],scores['control'],('score',)),('score',))['score'];label=ni(c,-.05,False);strength[str(seed)].update(contrast=c,noninferiority=label);sf.append(label)
pooled=[{'id':i,'cluster':g22[0][i]['cluster'],**{f:(g22[0][i][f]+g22[1][i][f])/2 for f in fields}} for i in g22[0]]
p=boot(pooled,fields);learning['pooled_g22']=p;cc=[learning[str(s)]['contrast-g22'] for s in (42,314159)]
practical=p['ce']['mean']<=-.02 and p['accuracy']['mean']>=.02 and all(c['ce']['mean']<=0 and c['accuracy']['mean']>=0 and c['mse']['mean']<=.02 for c in cc)
confirmed=p['ce']['ci95'][1]<0 and p['accuracy']['ci95'][0]>0
learning_fail|=p['ce']['ci95'][0]>-.02 or p['accuracy']['ci95'][1]<.02
combine=lambda v:'FAIL' if 'FAIL' in v else 'INCONCLUSIVE' if 'INCONCLUSIVE' in v else 'PASS'
result={'reviewer':'Independent separate reviewer agent; did not author frozen sources or final main analysis','method':'Independent board/tactical scans and metric reconstruction; cluster graph components; multinomial bootstrap multiplicity weights; no model or solver calls','checks':dict(checks),'fixtures':evidence,'learning':learning,'tactical':tactical,'strength':strength,'decisions':{'neural_learning':'PASS' if practical and confirmed else 'FAIL' if learning_fail else 'INCONCLUSIVE','tactical_reliability':combine(tf),'playing_strength':combine(sf)},'limitations':['No independent exact solving; frozen complete WDL vectors checked for coverage, consistency and mirrored identities','Execution and binary strict lineages audited by main agent, beyond this independent arithmetic review']}
(D/'results/independent-recomputed.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'checks':dict(checks),'fixtures':evidence,'decisions':result['decisions'],'pooled':p,'seed22':{str(s):learning[str(s)]['contrast-g22'] for s in (42,314159)},'strength':strength,'tactical_failures':{k:v for k,v in tactical.items() if v['label']=='FAIL'}},indent=2))
