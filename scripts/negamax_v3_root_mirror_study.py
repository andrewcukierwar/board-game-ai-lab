"""Prospective root-symmetry resource gate and strongest-reference analysis."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
from scripts import benchmark_negamax_v3 as bench
from scripts.negamax_v3_variants import load_source

DIR=bench.ROOT/'root-mirror'


def symmetric(history,module):
    state=module.SearchState(bench.position(history))
    return all(module.reflect(bits)==bits for bits in state.pieces)


def preflight():
    owner=bench.require_lock();config=bench.check(DIR)
    m=load_source((DIR/'terminal-proof-source.py').read_text())
    for _ in range(64):m.SearchTable()
    started=time.monotonic();results=[]
    for f in config['positions'][32:]:
        assert time.monotonic()-started<180
        game=bench.position(f['history'])
        warm=bench.decision(m,game,10)
        timed=bench.decision(m,game,10)
        memory=bench.memory_run(m,game,10)
        bench.same(warm,timed,True);bench.same(warm,memory,True)
        results.append(dict(position=f['id'],warmup=warm,timed=timed,memory=memory))
    forecast=1.5*(341.6+sum(30*r['timed']['wall_seconds']+6*r['memory']['wall_seconds'] for r in results))
    bench.write_new(DIR/'preflight.json',dict(passed=forecast<=1200,forecast_seconds=forecast,
                    elapsed_seconds=time.monotonic()-started,runtime=bench.runtime(),lock_owner=owner,results=results))
    print('resource preflight',forecast,'passed',forecast<=1200)


def analyze():
    config=bench.check(DIR);data=bench.rows(DIR)
    assert len(data)==len(config['positions'])*len(config['depths'])
    assert json.loads((DIR/'preflight.json').read_text())['passed']
    module=load_source((DIR/'root-mirror-source.py').read_text())
    histories={p['id']:p['history'] for p in config['positions']}
    core=[r for r in data if r['group']!='symmetric-target' and r['position']!='post-hoc-tail']
    asym=[r for r in data if not symmetric(histories[r['position']],module)]
    target=[r for r in data if symmetric(histories[r['position']],module) and r['depth'] in (8,10)
            and r['variants']['terminal-proof']['samples'][0]['stats']['nodes']>=2000]
    def ratio(r,metric='median_wall_seconds'):
        return r['variants']['root-mirror'][metric]/r['variants']['terminal-proof'][metric]
    def geo(rs):return math.exp(statistics.mean(math.log(1/ratio(r)) for r in rs))
    def summed(rs,metric):return sum(r['variants']['root-mirror'][metric] for r in rs)/sum(r['variants']['terminal-proof'][metric] for r in rs)
    metrics=dict(core_geometric_speedup=geo(core),core_wall_sum_ratio=summed(core,'median_wall_seconds'),
        core_cpu_sum_ratio=summed(core,'median_cpu_seconds'),asymmetric_geometric_speedup=geo(asym),
        target_broad_conditions=len(target),target_broad_geometric_speedup=geo(target),
        target_fraction_faster=sum(ratio(r)<1 for r in target)/len(target))
    failures=[]
    for gate,passed in [('core_geo',metrics['core_geometric_speedup']>=1),('core_wall',metrics['core_wall_sum_ratio']<=.98),
       ('core_cpu',metrics['core_cpu_sum_ratio']<=.99),('asymmetric_geo',metrics['asymmetric_geometric_speedup']>=.985),
       ('target_count',len(target)>=4),('target_geo',metrics['target_broad_geometric_speedup']>=1.25),('target_fraction',metrics['target_fraction_faster']>=.75)]:
        if not passed:failures.append(gate)
    regressions=[];memory_failures=[]
    for r in data:
        a,b=r['variants']['terminal-proof'],r['variants']['root-mirror']
        label=f"{r['position']}/D{r['depth']}"
        bench.same(a['samples'][0],b['samples'][0])
        if r in asym:assert a['samples'][0]['stats']==b['samples'][0]['stats']
        if ratio(r)>1.2 and b['median_wall_seconds']-a['median_wall_seconds']>.00025:regressions.append(label)
        if a['median_wall_seconds']>=.05 and (ratio(r)>1.2 or b['samples'][0]['stats']['nodes']>1.25*a['samples'][0]['stats']['nodes']):failures.append('expensive:'+label)
        for k in ('retained_bytes','peak_bytes'):
            am=max(m[k] for m in a['memory']);bm=max(m[k] for m in b['memory'])
            if bm>1.05*am and bm-am>32768:memory_failures.append(label+'/'+k)
    retained=sum(r['variants']['root-mirror']['memory'][0]['retained_bytes'] for r in core)/sum(r['variants']['terminal-proof']['memory'][0]['retained_bytes'] for r in core)
    metrics['core_retained_ratio']=retained
    if regressions:failures.append('per_condition')
    if memory_failures or retained>1.02:failures.append('memory')
    # Recompute/capture every asymmetric TT, not merely matching counters.
    owner=bench.require_lock();m=load_source((DIR/'root-mirror-source.py').read_text());ref=load_source((DIR/'terminal-proof-source.py').read_text())
    verified=[]
    for r in asym:
        with bench.capture(ref) as held: a=bench.decision(ref,bench.position(histories[r['position']]),r['depth'])
        entries=held[0].entries
        with bench.capture(m) as held: b=bench.decision(m,bench.position(histories[r['position']]),r['depth'])
        bench.same(a,b,True);assert entries==held[0].entries
        verified.append([r['position'],r['depth']])
    bench.write_new(DIR/'conditional-analysis.json',dict(eligible=not failures,research_only=True,metrics=metrics,
        failed_gates=failures,regressions=regressions,memory_failures=memory_failures,asymmetric_tt_verified=verified,lock_owner=owner))
    print(metrics,failures)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['preflight','analyze']);args=parser.parse_args()
    preflight() if args.command=='preflight' else analyze()
