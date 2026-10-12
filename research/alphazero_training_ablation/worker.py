"""One bounded child task. Every training task loads a fresh strict boundary."""
import argparse
from collections import Counter
import resource
import signal
import time
from .common import ROOT, read, write, sha, enforce
from .evaluate import positions, matches
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
from games.connect4.alphazero_v2.experimental import load_experimental, evaluate_boundary, digest
from games.connect4.alphazero_v2.diagnostics import family, entropy, peak_rss_mib
from games.connect4.alphazero_v2.oracle import safe_moves_scan, winning_moves_scan, engine_position
from games.connect4.alphazero_v2.batching import game_seed
from games.connect4.alphazero_v2.experimental import game_record

def state(r):
    p=r._content()
    return dict(model=r.factory.hash(r.model), optimizer=digest(p['optimizer']), replay=r.replay.digest(),
        sampling=digest(p['sampling']),augmentation=digest(p['augmentation']),global_rng=digest(p['global_rng']),
        counters=r.counters(),source=p['source'],runtime=p['runtime'])

def extra(r):
    generation=r.completed_generations
    games=[g for g in r.replay.iter_games() if g.generation == generation]
    examples=[e for g in games for e in g.examples]
    post=[e for e in examples if e.ply >= 8]
    openings=Counter(min(tuple(g.moves[:8]), tuple(6-a for a in g.moves[:8])) for g in games)
    mistakes=missed=blocked=0
    for g in games:
        for e in g.examples:
            prefix=g.moves[:e.ply]; wins=winning_moves_scan(prefix); safe=safe_moves_scan(prefix)
            missed += bool(wins) and e.action not in wins
            blocked += not wins and bool(safe) and len(safe)<len(engine_position(prefix).get_valid_moves()) and e.action not in safe
            mistakes += bool(safe) and e.action not in safe
    counts=r.replay.counts()
    sample=r.history[-1]['diagnostics']['sampling']['samples_by_source_generation']
    return dict(first_player_wins=sum(g.winner==0 for g in games),second_player_wins=sum(g.winner==1 for g in games),
        draws=sum(g.winner==-1 for g in games),positions=len(examples),
        duplicates=len(examples)-len({(e.actor,e.observation) for e in examples}),
        mirror_duplicates=len(examples)-len({family(e) for e in examples}),
        post_ply8_positions=len(post),post_ply8_duplicates=len(post)-len({(e.actor,e.observation) for e in post}),
        post_ply8_mirror_duplicates=len(post)-len({family(e) for e in post}),
        opening_families=len(openings),opening_concentration=sum((n/len(games))**2 for n in openings.values()),
        target_entropy=sum(entropy(e.policy_target) for e in examples)/len(examples),
        avoidable_one_reply_errors=mistakes,missed_immediate_wins=missed,missed_safe_responses=blocked,
        replay_counts=counts,experimental_replay_positions=sum(c['positions'] for g,c in counts.items() if g>20),
        experimental_sample_exposure=sum(n for g,n in sample.items() if int(g)>20),total_sample_exposure=sum(sample.values()),
        exact_diagnostics='Existing immediate-win/all-unsafe proofs in boundary; no new unbounded solving')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('task',choices=('train','positions','matches'))
    parser.add_argument('--boundary',required=True); parser.add_argument('--sha256',required=True)
    parser.add_argument('--output',required=True); parser.add_argument('--deadline',type=float,required=True)
    parser.add_argument('--parent'); parser.add_argument('--parent-sha256'); parser.add_argument('--seed',type=int)
    args=parser.parse_args(); configure_deterministic_runtime(1); enforce()
    output=__import__('pathlib').Path(args.output); output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); cpu=time.process_time()
    def stop(*_): raise TimeoutError('Cooperative cancellation')
    signal.signal(signal.SIGTERM,stop)
    def check():
        if time.monotonic() >= args.deadline: raise TimeoutError('Declared deadline reached')
    check()
    if args.task == 'train':
        r=load_experimental(args.boundary,expected_sha256=args.sha256)
        before=state(r); events=Counter(); stages={}
        def progress(event):
            events[event]+=1
            if events[event]==1:
                stages[event]=dict(wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu)
            # fsync once per complete game/update, not per simulation.
            if event in ('game_finalized','after_update'):
                write(output/'progress.json',dict(events=events,elapsed=time.monotonic()-started))
        path=output/f'generation-{r.completed_generations+1:04d}.pt'
        result=r.run_generation(path,check=check,progress=progress)
        after=state(r)
        loaded=load_experimental(path,expected_sha256=result['boundary_sha256'])
        if state(loaded) != after: raise ValueError('Strict roundtrip state differs')
        write(output/'games.json',dict(complete=True,generation=loaded.completed_generations,
            metadata=loaded.metadata[loaded.completed_generations],
            games=[game_record(g) for g in loaded.replay.iter_games() if g.generation==loaded.completed_generations]))
        write(output/'complete.json',dict(complete=True,boundary=str(path),sha256=sha(path),before=before,after=after,
            result=result,extra=extra(loaded),events=events,stage_resources=stages,
            game_rng_identities=[dict(index=i, seeds={domain:game_seed(r.config.training.seed,r.completed_generations,i,domain)
                for domain in ('search','action','noise')}) for i in range(64)]))
    elif args.task == 'positions':
        evaluate_boundary(args.boundary,lambda m,c: positions(m,check,output),expected_sha256=args.sha256)
    else:
        parent=load_experimental(args.parent,expected_sha256=args.parent_sha256,restore_global_rng=False)
        evaluate_boundary(args.boundary,lambda m,c: matches(m,parent.factory.freeze(parent.model),args.seed,check,output),expected_sha256=args.sha256)
    check()
    write(output/'task-complete.json',dict(complete=True,task=args.task,wall_seconds=time.monotonic()-started,
         cpu_seconds=time.process_time()-cpu,peak_rss_mib=peak_rss_mib()))

if __name__ == '__main__': main()
