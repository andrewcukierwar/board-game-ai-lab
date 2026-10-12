"""Zero-primary-work readiness: strict forks and bounded throughput probes."""
from dataclasses import replace
import time
from .common import ROOT, REPO, HISTORICAL, OFFICIAL, OUTPUT, read, write, sha, preserve, source_map
from .worker import state
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime, execution_source_identity
from games.connect4.alphazero_v2.experimental import ExperimentalConfig, ORIGINAL, EXTENDED, load_experimental
from games.connect4.alphazero_v2.experimental_import import read_historical, fork_verified
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.batching import search_requests, game_streams
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.search import EvaluationAgent
from games.connect4.alphazero_v2.data import seeded_rng

def main():
    runtime=configure_deterministic_runtime(1)
    before=preserve(); started=time.monotonic(); deadline=started+300
    def check():
        if time.monotonic() >= deadline: raise TimeoutError('300-second readiness ceiling')
    OUTPUT.mkdir(parents=True,exist_ok=False)
    fixtures=read(ROOT/'fixtures.json'); seeds={}; probes=[]
    for seed in (42,314159):
        check(); base=OFFICIAL/f'runs/seed-{seed}/generations/generation-0020'
        parent=base/'generation-0020.resume.pt'; expected=read(base/'summary.json')['files']['resume']['sha256']
        verified=read_historical(parent,expected_sha256=expected,historical_checkout=HISTORICAL)
        old=V2Config.from_dict(verified['original']['config'])
        config=ExperimentalConfig(replace(old,games_per_generation=64,max_generations=22),64)
        arms={}; equality=[]
        for name,schedule in (('control',ORIGINAL),('treatment',EXTENDED)):
            check(); r=fork_verified(verified,config,parent_sha256=expected,schedule=schedule,
                lineage=f'phase4d4g-seed-{seed}',arm=name)
            directory=OUTPUT/f'seed-{seed}'/name; directory.mkdir(parents=True)
            path=directory/'fork-0020.pt'; hashed=r.save(path)
            loaded=load_experimental(path,expected_sha256=hashed)
            if state(r) != state(loaded): raise ValueError('Initial strict reload differs')
            equality.append(r.initial_state_sha256())
            arms[name]=dict(path=str(path),sha256=hashed,initial_state_sha256=equality[-1],state=state(r),config=config.to_dict())
        if len(set(equality)) != 1: raise ValueError('Arms initial state mismatch')
        seeds[str(seed)]=dict(parent=str(parent),parent_sha256=expected,arms=arms,matched=True)
        inference=r.factory.inference(r.factory.freeze(r.model))
        # Three groups, including empty/early/middle boards; eight simultaneous 256-budget roots.
        batched=[]
        histories=[[],[3,2],[3,2,4,3]]+[row['moves'] for row in fixtures['solved'][::60]]
        for repetition in range(3):
            check(); tick=time.monotonic()
            generators=[search_requests(BitboardState.from_game(engine_position(m)),256,
                game_streams(old,990+repetition,i),check=check) for i,m in enumerate(histories)]
            active=[(g,next(g)) for g in generators]
            while active:
                check(); predictions=inference.predict_batch([s for _,s in active]); continuing=[]
                for (g,_),prediction in zip(active,predictions):
                    try: continuing.append((g,g.send(prediction)))
                    except StopIteration as done:
                        if sum(done.value.visits)!=256: raise ValueError('Readiness visits mismatch')
                active=continuing
            batched.append((time.monotonic()-tick)/8)
        raw=[]
        for i,m in enumerate(histories+histories[:4]):
            check(); tick=time.monotonic()
            EvaluationAgent(inference,512,rng=seeded_rng('4g-readiness',i),backend='bitboard').search(engine_position(m))
            raw.append(time.monotonic()-tick)
        # Independent disposable learner only; parent and experimental arms get zero updates.
        scratch=r.factory.construct(4404799); trainer=r.factory.trainer(scratch,old)
        examples=next(r.replay.iter_games()).examples
        batch=tuple(examples[i%len(examples)] for i in range(128)); tick=time.monotonic()
        for _ in range(4): check(); trainer.step(batch)
        update_seconds=(time.monotonic()-tick)/4
        probes.append(dict(seed=seed,batched_seconds_per_move=batched,raw512_seconds_per_move=raw,
                           scratch_updates=4,scratch_update_seconds=update_seconds,primary_games=0,primary_updates=0))
    # Worst observed throughput, max game length 42, 2x runtime contingency.
    collection=512*42*max(max(p['batched_seconds_per_move']) for p in probes)
    updates=512*max(p['scratch_update_seconds'] for p in probes)
    position_roots=10*(300+200+128)  # two parents + eight trained checkpoints
    strength_roots=256*42
    evaluations=(position_roots+strength_roots)*max(max(p['raw512_seconds_per_move']) for p in probes)
    io_allowance=600
    projected=2*(collection+updates+evaluations)+io_allowance
    readiness=dict(runtime=runtime,training_source=execution_source_identity(),seeds=seeds,probes=probes,
        estimate=dict(collection_seconds=collection,optimization_seconds=updates,evaluation_seconds=evaluations,
            io_seconds=io_allowance,contingency_factor=2,total_seconds=projected,ceiling=10800,feasible=projected<10800),
        primary_games=0,primary_updates=0,wall_seconds=time.monotonic()-started,
        preservation_before=before,preservation_after=preserve())
    write(ROOT/'results/readiness.json',readiness)
    write(ROOT/'results/freeze.json',dict(files=source_map(),runtime=runtime,training_source=execution_source_identity(),
        fixtures_sha256=sha(ROOT/'fixtures.json'),readiness_sha256=sha(ROOT/'results/readiness.json')))
    print(__import__('json').dumps(dict(estimate=readiness['estimate'],readiness_seconds=readiness['wall_seconds'])))
    if projected>=10800: raise SystemExit('Infeasible; do not launch or reduce design')

if __name__=='__main__': main()
