"""Bounded frozen-weight Phase 4D.4C validation/benchmark; never trains.

Every invocation refuses existing output and verifies all 150 official bytes.
Serial baseline uses unchanged Phase 4D.4B search, with experimental game RNGs.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import ctypes
from contextlib import ExitStack
from unittest.mock import patch
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tarfile
import time
import tracemalloc

import numpy as np
import torch

from games.connect4.alphazero_v2 import batching as b, network, search
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.diagnostics import peak_rss_mib
from games.connect4.alphazero_v2.oracle import engine_position
from research.alphazero_baseline import audit
from research.alphazero_bitboard.validation import assert_result

BASE = Path('research/alphazero_baseline/results/phase4d4a-20261011')
SOURCE_ROOT = Path(__file__).resolve().parents[2]
CRITERIA = dict(logits=2e-5, value=2e-6, policy=2e-6, visit_l1=.10,
                action_disagreement=.05, trajectory_screen=.10, outcome_screen=.05)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def hardware():
    result = {}
    libc = ctypes.CDLL(None)
    for name in ('machdep.cpu.brand_string','hw.memsize','hw.ncpu'):
        size = ctypes.c_size_t()
        key = name.encode()
        if libc.sysctlbyname(key,None,ctypes.byref(size),None,0) == 0:
            buf = ctypes.create_string_buffer(size.value)
            if libc.sysctlbyname(key,buf,ctypes.byref(size),None,0) == 0:
                result[name] = (buf.value.decode() if name.endswith('string') else
                                int.from_bytes(buf.raw[:size.value],sys.byteorder))
    return result


class CountedInference(network.V2Inference):
    def __init__(self, model):
        super().__init__(model)
        self.reset()
        self.in_batch = False

    def reset(self):
        self.calls = Counter()
        self.seconds = 0.

    def predict(self, state):
        if self.in_batch:
            return super().predict(state)
        started = time.perf_counter()
        result = super().predict(state)
        self.seconds += time.perf_counter()-started
        self.calls[1] += 1
        return result

    def predict_batch(self, states):
        started = time.perf_counter()
        self.in_batch = True
        try:
            result = super().predict_batch(states)
        finally:
            self.in_batch = False
        self.seconds += time.perf_counter()-started
        if states: self.calls[len(states)] += 1
        return result


def tree_hash(root):
    rows, stack = [], [root]
    while stack:
        node = stack.pop()
        rows.append((asdict(node.game_state),node.move,node.visits,node.value_sum.hex(),
                     float(node.prior_p).hex(),tuple(node.children)))
        stack.extend(reversed(tuple(node.children.values())))
    return digest(rows)


def decision_record(index, decision, *, tree=False):
    r = decision.search
    row = dict(index=index,ply=decision.ply,visits=r.visits,action=decision.action.move,
               temperature=decision.action.temperature,simulation_count=r.simulations,
               root_logits=r.root_logits,root_value=r.root_value,root_prior=r.root_prior,
               searched_prior=r.search_prior,root_noise=r.root_noise,
               depth=r.max_depth,depth_sum=r.leaf_depth_sum,terminal_leaves=r.terminal_leaves)
    if tree: row['tree_sha256'] = tree_hash(r.root)
    return row


def game_records(games):
    return [dict(generation=g.generation,index=g.index,moves=g.moves,winner=g.winner,
                 examples=[e.record() for e in g.examples]) for g in games]


def run_collection(inference,config,ids,condition,deadline,*,observer=None):
    stats = b.CollectionStats()
    if condition == 'serial':
        games = b.collect_serial_reference(inference,config,21,ids,stats=stats,check=deadline,observer=observer)
    else:
        games = b.collect_batched(inference,config,21,ids,batch_size=int(condition),stats=stats,check=deadline,observer=observer)
    require(stats.simulations == stats.plies*config.self_play_simulations,'Simulation budget mismatch')
    return games,stats


def run_search_batch(inference,states,config,*,size,deadline):
    """Fixed independent search jobs; exactly one suspended leaf per tree."""
    results = []
    for offset in range(0,len(states),size):
        group = states[offset:offset+size]
        streams = [b.game_streams(config,21,offset+i) for i in range(len(group))]
        generators = [b.search_requests(state,config.self_play_simulations,s,check=deadline)
                      for state,s in zip(group,streams)]
        active = [(i,g,next(g)) for i,g in enumerate(generators)]
        ordered = {}
        try:
            while active:
                deadline()
                predictions = inference.predict_batch([state for _,_,state in active])
                following = []
                for (i,g,_),p in zip(active,predictions):
                    try: following.append((i,g,g.send(p)))
                    except StopIteration as done: ordered[i] = done.value
                active = following
            results.extend((ordered[i],streams[i].state()) for i in range(len(group)))
        finally:
            for g in generators: g.close()
    return results


def fixed_states():
    rows = audit.read(BASE/'profile/positions.json')
    # Exact Phase 4D.4A representative development positions and their mirrors.
    records = [dict(name=r['label']+suffix,moves=([6-m for m in r['moves']] if suffix else r['moves']))
               for r in rows for suffix in ('','_mirror')]
    states = [BitboardState.from_game(engine_position(r['moves'])) for r in records]
    return records,states


def numerical(campaign,output,deadline):
    records,states = fixed_states()
    audit.write(output/'positions.json',records)
    config = V2Config(seed=4404003)
    raw, searches, games_validation = [], [], []
    for seed in audit.SEEDS:
        for generation in audit.GENERATIONS:
            deadline()
            inference = CountedInference(network.load_inference_checkpoint(audit.checkpoint(campaign,seed,generation)).model)
            before = network.weights_sha256(inference.model)
            singles = [inference.predict(s) for s in states]
            baseline = []
            for i,state in enumerate(states):
                streams = b.game_streams(config,21,i)
                result = search.PUCTSearch(inference,256,backend='bitboard').run(state,rng=streams.search,root_noise=streams.noise)
                baseline.append((result,streams.state()))
            for size in (1,2,4,8,16):
                actual = []
                for off in range(0,len(states),size): actual.extend(inference.predict_batch(states[off:off+size]))
                repeated = []
                for off in range(0,len(states),size): repeated.extend(inference.predict_batch(states[off:off+size]))
                require(actual == repeated,'Fixed inference is not reproducible')
                for i,(a,e) in enumerate(zip(actual,singles)):
                    errors = dict(logits=max(abs(x-y) for x,y in zip(a.logits,e.logits)),
                                  value=abs(a.value-e.value),policy=max(abs(x-y) for x,y in zip(a.policy,e.policy)))
                    require(all(errors[k]<=CRITERIA[k] for k in errors),'Prospective numerical gate failed')
                    require(size!=1 or a==e,'Singleton inference differs')
                    raw.append(dict(seed=seed,generation=generation,size=size,position=records[i]['name'],
                                    errors=errors,serial=asdict(e),batched=asdict(a),exact=a==e))
                batched = run_search_batch(inference,states,config,size=size,deadline=deadline)
                for i,((a,ast),(e,est)) in enumerate(zip(batched,baseline)):
                    actions = []
                    for temp in (0.,1.):
                        x=search.select_action(e.visits,temp,random.Random(909+i))
                        y=search.select_action(a.visits,temp,random.Random(909+i))
                        actions.append(dict(temperature=temp,serial=x.move,batched=y.move))
                    l1 = sum(abs(x-y) for x,y in zip(a.visits,e.visits))/256
                    if size == 1:
                        assert_result(e,a)
                        require(ast==est,'Singleton search RNG differs')
                    searches.append(dict(seed=seed,generation=generation,size=size,position=records[i]['name'],
                        serial_visits=e.visits,batched_visits=a.visits,visit_l1=l1,actions=actions,
                        search_rng_equal=ast['search']==est['search'],noise_rng_equal=ast['noise']==est['noise']))
            # Complete games at the actual official search budget for every checkpoint.
            decisions = {}
            expected,serial_stats=run_collection(inference,config,range(2),'serial',deadline,
                observer=lambda i,d,t: decisions.__setitem__((i,d.ply),decision_record(i,d,tree=True)))
            actual_records=[]
            def compare(i,d,t):
                row=decision_record(i,d,tree=True)
                require(row==decisions[i,d.ply],'Singleton full decision/tree differs')
                actual_records.append(row)
            actual,stats=run_collection(inference,config,range(2),'1',deadline,observer=compare)
            require(expected==actual and stats.rng_states==serial_stats.rng_states,'Singleton complete game differs')
            games_validation.append(dict(seed=seed,generation=generation,size=1,exact=True,
                full_trees_equal=True,rng_equal=True,games=game_records(actual),decisions=actual_records))
            # 4 IDs at sizes 2/4/8, exact repeated-run records + RNG states.
            for size in (2,4,8):
                traces=[]
                def capture(i,d,t): traces.append(decision_record(i,d))
                x,xs=run_collection(inference,config,range(4),str(size),deadline,observer=capture)
                ys_trace=[]
                y,ys=run_collection(inference,config,range(4),str(size),deadline,
                    observer=lambda i,d,t: ys_trace.append(decision_record(i,d)))
                require(x==y and xs.rng_states==ys.rng_states and traces==ys_trace,'Batch collection not reproducible')
                games_validation.append(dict(seed=seed,generation=generation,size=size,repeat_exact=True,
                    rng_exact=True,games=game_records(x),decisions=traces))
            require(network.weights_sha256(inference.model)==before,'Weights changed')
            print(f'validated {seed}/g{generation}',flush=True)
    audit.write(output/'raw_predictions.json',raw)
    audit.write(output/'fixed_search.json',searches)
    audit.write(output/'complete_validation.json',games_validation)
    errors={k:max(r['errors'][k] for r in raw) for k in ('logits','value','policy')}
    disagreement=sum(a['serial']!=a['batched'] for r in searches for a in r['actions'])/sum(len(r['actions']) for r in searches)
    max_l1=max(r['visit_l1'] for r in searches)
    summary=dict(max_errors=errors,max_visit_l1=max_l1,action_disagreement=disagreement,
                 numerical_pass=all(errors[k]<=CRITERIA[k] for k in errors),
                 fixed_search_screen_pass=max_l1<=.10 and disagreement<=.05,
                 complete_repeat_exact=True,batch1_exact=True,raw_comparisons=len(raw),search_comparisons=len(searches))
    audit.write(output/'summary.json',summary)
    print(summary,flush=True)


def collection(campaign,output,deadline,args):
    trials=[]
    config=V2Config(seed=4404003)
    conditions=['serial','1','2','4','8']
    for repeat in range(args.repeats):
        for seed,generation in ((42,10),(314159,20)):
            inference=CountedInference(network.load_inference_checkpoint(audit.checkpoint(campaign,seed,generation)).model)
            frozen=network.weights_sha256(inference.model)
            order=conditions if (repeat+(seed==314159))%2==0 else list(reversed(conditions))
            for condition in order:
                # Same warmup state and budget, then condition-specific batch forward.
                state=BitboardState.from_game(engine_position([]))
                search.PUCTSearch(inference,32,backend='bitboard').run(state,rng=random.Random(0))
                inference.predict_batch([state]*(1 if condition=='serial' else int(condition)))
                gc.collect(); inference.reset()
                rows=[]
                def observe(i,d,t): rows.append(decision_record(i,d))
                cpu_start=time.process_time(); started=time.perf_counter()
                games,stats=run_collection(inference,config,range(args.games),condition,deadline,observer=observe)
                wall=time.perf_counter()-started; cpu=time.process_time()-cpu_start
                require(network.weights_sha256(inference.model)==frozen,'Weights changed')
                n=sum(inference.calls.values()); positions=sum(size*count for size,count in inference.calls.items())
                row=dict(repeat=repeat,seed=seed,generation=generation,condition=condition,wall_seconds=wall,cpu_seconds=cpu,
                    total_games=len(games),total_plies=stats.plies,simulations=stats.simulations,
                    forward_calls=n,neural_positions=positions,mean_batch=positions/n,
                    batch_distribution=dict(inference.calls),inference_seconds=inference.seconds,
                    inference_fraction=inference.seconds/wall,non_inference_seconds=wall-inference.seconds,
                    scheduler_overhead_seconds=(None if condition=='serial' else
                        stats.collection_seconds-stats.inference_seconds-stats.game_work_seconds),
                    game_work_seconds=(None if condition=='serial' else stats.game_work_seconds),
                    games_per_second=len(games)/wall,games_per_hour=len(games)*3600/wall,
                    positions_per_second=stats.plies/wall,peak_rss_mib=peak_rss_mib(),
                    completion_seconds=stats.completion_seconds,active_seconds=stats.active_seconds,
                    rng_sha256={i:digest(v) for i,v in stats.rng_states.items()},
                    games=game_records(games),decisions=rows)
                trials.append(row)
                audit.write(output/f'trial-{repeat}-{seed}-{condition}.json',row)
                print(f'{repeat} {seed}/g{generation} {condition}: {wall:.3f}s {row["games_per_hour"]:.1f} games/h',flush=True)
    summary={}
    for condition in conditions:
        cells=[r for r in trials if r['condition']==condition]
        wall=sum(r['wall_seconds'] for r in cells)
        summary[condition]=dict(wall_seconds=wall,games=sum(r['total_games'] for r in cells),
            plies=sum(r['total_plies'] for r in cells),games_per_hour=sum(r['total_games'] for r in cells)*3600/wall,
            positions_per_second=sum(r['total_plies'] for r in cells)/wall,
            neural_positions=sum(r['neural_positions'] for r in cells),
            forward_calls=sum(r['forward_calls'] for r in cells),
            inference_fraction=sum(r['inference_seconds'] for r in cells)/wall)
    differences=[]
    for r in trials:
        ref=next(t for t in trials if t['repeat']==r['repeat'] and t['seed']==r['seed'] and t['condition']=='serial')
        by_id={g['index']:g for g in ref['games']}
        moves=sum(g['moves']!=by_id[g['index']]['moves'] for g in r['games'])
        outcomes=sum(g['winner']!=by_id[g['index']]['winner'] for g in r['games'])
        decisions={ (d['index'],d['ply']):d for d in ref['decisions']}
        first=[]
        for g in r['games']:
            before=by_id[g['index']]['moves']
            diverge=next((i for i,(x,y) in enumerate(zip(g['moves'],before)) if x!=y),None)
            first.append(dict(index=g['index'],first_different_move_ply=diverge))
        exact_games=r['games']==ref['games']
        if r['condition']=='1':
            require(exact_games and r['rng_sha256']==ref['rng_sha256'],'Timed singleton differs')
            require(sorted(r['decisions'],key=lambda d:(d['index'],d['ply']))==sorted(ref['decisions'],key=lambda d:(d['index'],d['ply'])),'Timed singleton decisions differ')
        differences.append(dict(repeat=r['repeat'],seed=r['seed'],condition=r['condition'],
            trajectory_differences=moves,outcome_differences=outcomes,total_games=r['total_games'],
            examples_exact=exact_games,first_divergence=first,paired_speedup=ref['wall_seconds']/r['wall_seconds']))
    for condition,row in summary.items():
        row['speedup']=summary['serial']['wall_seconds']/row['wall_seconds']
    audit.write(output/'summary.json',summary)
    audit.write(output/'differences.json',differences)
    print(summary,flush=True)


def memory(campaign,output,deadline,args):
    inference=CountedInference(network.load_inference_checkpoint(audit.checkpoint(campaign,42,10)).model)
    config=V2Config(seed=4404003)
    state=BitboardState.from_game(engine_position([]))
    inference.predict_batch([state]*(1 if args.condition=='serial' else int(args.condition)))
    gc.collect(); inference.reset()
    tracemalloc.start()
    started=time.perf_counter()
    games,stats=run_collection(inference,config,range(args.games),args.condition,deadline)
    wall=time.perf_counter()-started
    current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    audit.write(output/'memory.json',dict(condition=args.condition,games=len(games),plies=stats.plies,
        wall_seconds=wall,traced_current_bytes=current,traced_peak_bytes=peak,peak_rss_mib=peak_rss_mib(),
        max_active=stats.max_active,rng_sha256={i:digest(s) for i,s in stats.rng_states.items()}))
    print(f'memory {args.condition}: peak {peak} bytes, RSS {peak_rss_mib():.1f} MiB',flush=True)


def fixed(campaign,output,deadline,args):
    _,states=fixed_states()
    config=V2Config(seed=4404003)
    rows=[]
    for repeat in range(3):
        for condition in (['serial','1','2','4','8'] if repeat%2==0 else ['8','4','2','1','serial']):
            inference=CountedInference(network.load_inference_checkpoint(audit.checkpoint(campaign,42,10)).model)
            inference.predict_batch(states[:(1 if condition=='serial' else int(condition))])
            gc.collect(); inference.reset()
            cpu=time.process_time(); started=time.perf_counter()
            if condition=='serial':
                result=[]
                for i,state in enumerate(states):
                    deadline(); streams=b.game_streams(config,21,i)
                    r=search.PUCTSearch(inference,256,backend='bitboard').run(state,rng=streams.search,root_noise=streams.noise)
                    result.append((r,streams.state()))
            else:
                result=run_search_batch(inference,states,config,size=int(condition),deadline=deadline)
            wall=time.perf_counter()-started; cpu=time.process_time()-cpu
            rows.append(dict(repeat=repeat,condition=condition,wall_seconds=wall,cpu_seconds=cpu,
                searches=len(states),simulations=256*len(states),neural_positions=sum(n*c for n,c in inference.calls.items()),
                forward_calls=sum(inference.calls.values()),visits=[r.visits for r,_ in result],
                inference_seconds=inference.seconds,simulations_per_second=256*len(states)/wall))
            # Equal-position forward workload, separate from search.
            inference.reset(); started=time.perf_counter()
            size=1 if condition=='serial' else int(condition)
            for _ in range(32):
                for off in range(0,len(states),size):
                    deadline()
                    if condition=='serial': inference.predict(states[off])
                    else: inference.predict_batch(states[off:off+size])
            elapsed=time.perf_counter()-started
            rows[-1].update(fixed_inference_positions=len(states)*32,fixed_inference_wall_seconds=elapsed,
                            fixed_inference_positions_per_second=len(states)*32/elapsed)
    audit.write(output/'fixed.json',rows)


def profile(campaign,output,deadline,args):
    """Separate instrumented attribution; never pooled with primary latencies."""
    rows=[]
    for condition in ('serial','1','2','4','8'):
        inference=network.load_inference_checkpoint(audit.checkpoint(campaign,42,10))
        config=V2Config(seed=4404003)
        timings=audit.Timings()
        targets=[(inference.model,'forward','neural_forward'),
                 (network.V2Inference,'predict','inference_pipeline'),
                 (network.V2Inference,'predict_batch','inference_pipeline'),
                 (network,'validate_batch_state','batch_state_validation'),
                 (network,'encode_bitboard_current_player','encoding_tensor'),
                 (network,'legal_policy_from_logits','legal_softmax'),
                 (b,'_expand','expansion'),(search.PUCTSearch,'_expand','expansion'),
                 (b,'backup','backup'),(search,'backup','backup'),
                 (BitboardState,'drop','state_transition')]
        gc.collect()
        with ExitStack() as patches:
            for owner,name,category in targets:
                patches.enter_context(patch.object(owner,name,timings.wrap(category,getattr(owner,name))))
            patches.enter_context(patch.object(search.PUCTSearch,'_select_child',
                staticmethod(timings.wrap('selection',search.PUCTSearch._select_child))))
            started=time.perf_counter()
            games,stats=run_collection(inference,config,range(args.games),condition,deadline)
            wall=time.perf_counter()-started
        row=timings.result(wall)
        row.update(condition=condition,games=len(games),plies=stats.plies,
                   scheduler_overhead_seconds=(None if condition=='serial' else
                        stats.collection_seconds-stats.inference_seconds-stats.game_work_seconds))
        rows.append(row)
    audit.write(output/'profile.json',rows)


def confirmation(campaign,output,deadline,args):
    """Original V2Inference, no recording observer or inference-counting wrapper."""
    rows=[]
    config=V2Config(seed=4404003)
    for seed,generation in ((42,10),(314159,20)):
        inference=network.load_inference_checkpoint(audit.checkpoint(campaign,seed,generation))
        conditions=('serial','8') if seed==42 else ('8','serial')
        for condition in conditions:
            state=BitboardState.from_game(engine_position([]))
            inference.predict_batch([state]*(1 if condition=='serial' else 8))
            gc.collect()
            cpu=time.process_time(); started=time.perf_counter()
            games,stats=run_collection(inference,config,range(args.games),condition,deadline)
            wall=time.perf_counter()-started; cpu=time.process_time()-cpu
            rows.append(dict(seed=seed,generation=generation,condition=condition,wall_seconds=wall,cpu_seconds=cpu,
                games=len(games),plies=stats.plies,simulations=stats.simulations,records_sha256=digest(game_records(games)),
                rng_sha256={i:digest(s) for i,s in stats.rng_states.items()},games_per_hour=len(games)*3600/wall))
            print(f'confirmation {seed} {condition}: {wall:.3f}s',flush=True)
    audit.write(output/'confirmation.json',rows)


def main():
    if not __debug__: raise RuntimeError('Validation refuses python -O')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=['numerical','collection','memory','fixed','profile','confirmation'],required=True)
    parser.add_argument('--seconds',type=float,default=900)
    parser.add_argument('--games',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--condition',choices=['serial','1','2','4','8'],default='8')
    args=parser.parse_args()
    require(args.seconds>0 and args.games>0 and args.repeats>0,'Invalid budget')
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')),
            'Set all five vector thread environment variables to 1 before Python starts')
    campaign=args.campaign.resolve()
    require(audit.sha(campaign/'declaration.json')==audit.DECLARATION,'Declaration changed')
    output=audit.fresh_output(campaign,args.output)
    before=audit.campaign_fingerprint(campaign)
    expected=audit.read(BASE/'profile/campaign_before.json')
    require(before==expected and len(before)==150,'Official 150-file inventory changed')
    audit.write(output/'campaign_before.json',before)
    source_files=[*Path('games/connect4/alphazero_v2').glob('*.py'),Path('games/connect4/agents/mcts_nn_agent.py'),
                  Path('games/connect4/neural_mcts.py'),Path('games/connect4/connect4.py'),
                  *Path('research/alphazero_batching').glob('*.py'),Path('research/alphazero_batching/DESIGN.md')]
    source={str(p):audit.sha(p) for p in source_files}
    with tarfile.open(output/'source.tar.gz','w:gz') as tar:
        for p in source_files: tar.add(p,arcname=str(p))
    checkpoints=[]
    for seed in audit.SEEDS:
        for gen in audit.GENERATIONS:
            path=audit.checkpoint(campaign,seed,gen)
            expected_hash=(audit.read(path.parent/'summary.json')['files']['inference']['sha256'] if gen else
                           audit.read(campaign/f'runs/seed-{seed}/selection.json')['inference']['sha256'])
            require(audit.sha(path)==expected_hash,'Checkpoint changed')
            checkpoints.append(dict(seed=seed,generation=gen,path=str(path),sha256=expected_hash))
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    audit.write(output/'provenance.json',dict(runtime=audit.runtime(),hardware=hardware(),source=source,
        source_base_commit='df8a586db40d8b2a2fea9052758d6917ae412dcd',checkpoints=checkpoints,
        criteria=CRITERIA,design_sha256=audit.sha('research/alphazero_batching/DESIGN.md'),
        rng_protocol=b.PROTOCOL,parameters={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}))
    deadline=audit.Deadline(args.seconds)
    started=time.perf_counter(); status=dict(complete=False)
    try:
        if args.mode=='numerical': numerical(campaign,output,deadline)
        elif args.mode=='collection': collection(campaign,output,deadline,args)
        elif args.mode=='fixed': fixed(campaign,output,deadline,args)
        elif args.mode=='profile': profile(campaign,output,deadline,args)
        elif args.mode=='confirmation': confirmation(campaign,output,deadline,args)
        else: memory(campaign,output,deadline,args)
        status['complete']=True
    except BaseException as error:
        status['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        status['elapsed_seconds']=time.perf_counter()-started
        audit.write(output/'status.json',status)
        after=audit.campaign_fingerprint(campaign)
        audit.write(output/'preservation.json',dict(files=len(after),unchanged=before==after,
            after=after,declaration_sha256=audit.sha(campaign/'declaration.json')))
        require(before==after,'Official campaign modified')


if __name__=='__main__': main()
