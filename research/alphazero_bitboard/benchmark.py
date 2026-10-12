"""Paired end-to-end search benchmark; never trains or writes campaign artifacts.

Reuses Phase 4D.4A provenance, positions, timing instrumentation and statistics.
All setup, warm-up, hooks, comparison, GC, hashing and allocation tracing live
outside uninstrumented search latency. Outputs refuse existing directories.
"""
import argparse
from collections import defaultdict
from contextlib import ExitStack
import cProfile
import gc
import json
import os
from pathlib import Path
import pstats
import random
import sys
import time
import tracemalloc
from unittest.mock import patch

import numpy as np
import torch

from games.connect4.alphazero_v2 import network, search
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.diagnostics import peak_rss_mib
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.selfplay import play_game
from research.alphazero_baseline import audit
from .validation import assert_result, assert_state

BASE = Path('research/alphazero_baseline/results/phase4d4a-20261011')
DOMAIN = 'phase4d4b-paired-v1'
BACKENDS = ('engine', 'bitboard')


class RecordingInference(network.V2Inference):
    """Validation only: retains every encoded input and prediction, never timed."""
    def __init__(self, model):
        super().__init__(model)
        self.trace = []
        self.raw_trace = []

    def predict(self, game):
        states = (network.encode_bitboard_current_player(game) if isinstance(game, BitboardState)
                  else network.encode_current_player(game))
        prediction = super().predict(game)
        self.trace.append((states.numpy().tobytes(), tuple(x.hex() for x in prediction.logits),
                           tuple(x.hex() for x in prediction.policy), prediction.value.hex()))
        return prediction


def paired_validation(inference, game, budget, domain, *, noisy=False):
    results, traces, rngs, noises = {}, {}, {}, {}
    recorder = RecordingInference(inference.model)
    for backend in BACKENDS:
        recorder.trace = []
        recorder.raw_trace = []
        rng = seeded_rng(domain, 4404001)
        noise = search.V2RootNoise(seed=4404002) if noisy else None
        def capture(module, inputs, outputs):
            recorder.raw_trace.append((inputs[0].numpy().tobytes(),
                                       *(t.numpy().tobytes() for t in outputs)))
        hook = recorder.model.register_forward_hook(capture)
        try:
            results[backend] = search.PUCTSearch(recorder, budget, backend=backend).run(
                game, rng=rng, root_noise=noise)
        finally:
            hook.remove()
        traces[backend] = (tuple(recorder.trace), tuple(recorder.raw_trace))
        rngs[backend] = rng.getstate()
        noises[backend] = noise.get_state() if noise else None
    assert traces['engine'] == traces['bitboard'], 'Leaf tensors/predictions differ'
    assert rngs['engine'] == rngs['bitboard'], 'Search RNG differs'
    assert noises['engine'] == noises['bitboard'], 'Noise RNG differs'
    nodes = assert_result(results['engine'], results['bitboard'])
    actions = {}
    for temperature in (0., 1.):
        states, selected = [], []
        for backend in BACKENDS:
            rng = seeded_rng(domain + ':action', 4404003)
            selected.append(search.select_action(results[backend].visits, temperature, rng))
            states.append(rng.getstate())
        assert selected[0] == selected[1] and states[0] == states[1]
        actions[str(temperature)] = selected[0].move
    return dict(nodes=nodes, predictions=len(traces['engine'][0]), exact=True,
                visits=results['engine'].visits, actions=actions,
                max_depth=results['engine'].max_depth,
                leaf_depth_sum=results['engine'].leaf_depth_sum,
                terminal_leaves=results['engine'].terminal_leaves,
                noisy=noisy, rng_equal=True, noise_rng_equal=True, action_rng_equal=True)


def transitions(rows, minimum=10000):
    """All alternative successors from random histories and their reflections."""
    rng = random.Random(2026101144)
    count = states = terminals = 0
    actors, winners = set(), set()
    full_columns = 0
    # Explicit fixture draw plus every fixed position and both players' wins.
    draw = [2,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,3,6,4,4,4,4,4,4,5,5,5,5,5,5,6,6,6,6,6]
    fixtures = [r['moves'] for r in rows] + [[], [3], [3]*6, draw[:40], draw[:41], draw,
                    [0,1,0,1,0,1,0], [0,1,0,1,2,1,2,1]]
    def check(moves):
        nonlocal count, states, terminals, full_columns
        for history in (moves, [6-a for a in moves]):
            game = engine_position(history)
            bits = BitboardState.from_game(game)
            assert_state(bits, game)
            states += 1
            actors.add(game.current_player); winners.add(game.check_winner())
            full_columns += len(game.get_valid_moves()) < 7
            if game.is_game_over():
                terminals += 1
                try:
                    bits.drop(3)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Accepted terminal drop')
                continue
            for move in game.get_valid_moves():
                from copy import deepcopy
                after = deepcopy(game)
                assert after.make_move(move)
                assert_state(bits.drop(move), after)
                count += 1
    for moves in fixtures:
        check(moves)
    while count < minimum:
        moves, game = [], engine_position([])
        while not game.is_game_over():
            check(moves)
            move = rng.choice(game.get_valid_moves())
            game.make_move(move); moves.append(move)
        check(moves)
    assert actors == {0, 1} and winners == {-1, 0, 1}
    return dict(transitions=count, states=states, terminal_states=terminals,
                states_with_full_columns=full_columns, actors=sorted(actors),
                winners=sorted(winners), tensors_byte_equal=True, exact=True,
                includes_mirrors=True, includes_draw=True)


def memory_probe(inference, game, budget, domain, backend):
    gc.collect()
    rng = seeded_rng(domain, 4404001)
    engine = search.PUCTSearch(inference, budget, backend=backend)
    tracemalloc.start()
    try:
        result = engine.run(game, rng=rng)
        current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return dict(backend=backend, current_bytes=current, peak_bytes=peak,
                process_lifetime_peak_rss_mib=peak_rss_mib()), result


def profile_probe(inference, game, budget, domain, backend):
    meter = audit.Timings()
    engine = search.PUCTSearch(inference, budget, backend=backend)
    rng = seeded_rng(domain, 4404001)
    gc.collect()
    with meter.instrument(inference), ExitStack() as extra:
        if backend == 'bitboard':
            for owner, name, category in (
                (BitboardState, 'drop', 'state_transition'),
                (BitboardState, 'is_game_over', 'terminal'),
                (BitboardState, 'check_winner', 'terminal'),
                (BitboardState, 'get_valid_moves', 'legal_moves'),
                (network, 'encode_bitboard_current_player', 'encoding_tensor')):
                extra.enter_context(patch.object(owner, name, meter.wrap(category, getattr(owner, name))))
        started = time.perf_counter()
        result = engine.run(game, rng=rng)
        elapsed = time.perf_counter() - started
    # Compare against uninstrumented run, never using timers as semantic proof.
    expected_rng = seeded_rng(domain, 4404001)
    expected = engine.run(game, rng=expected_rng)
    assert_result(expected, result)
    assert rng.getstate() == expected_rng.getstate()
    return dict(backend=backend, semantics_verified=True, **meter.result(elapsed))


def selfplay_probe(inference, output, games, deadline):
    before = network.weights_sha256(inference.model)
    records = []
    for trial in range(games):
        paired = {}
        for backend in (BACKENDS if trial % 2 == 0 else BACKENDS[::-1]):
            deadline()
            player = search.SelfPlayer(inference, V2Config(), backend=backend,
                        search_rng=seeded_rng(f'{DOMAIN}:selfplay:{trial}:search', 4404011),
                        action_rng=seeded_rng(f'{DOMAIN}:selfplay:{trial}:action', 4404012),
                        root_noise=search.V2RootNoise(seed=4404013 + trial))
            gc.collect()
            cpu, wall = time.process_time(), time.perf_counter()
            game = play_game(player, 1, trial, check=deadline)
            wall, cpu = time.perf_counter()-wall, time.process_time()-cpu
            paired[backend] = (game, player.search_rng.getstate(), player.action_rng.getstate(),
                               player.root_noise.get_state())
            records.append(dict(trial=trial, backend=backend, wall_seconds=wall, cpu_seconds=cpu,
                plies=len(game.moves), moves=game.moves, winner=game.winner,
                games_per_hour=3600/wall, positions_per_second=len(game.moves)/wall,
                examples=[e.record() for e in game.examples], rng_equal=True))
        assert paired['engine'] == paired['bitboard'], 'Complete selfplay or RNG differs'
    assert before == network.weights_sha256(inference.model)
    audit.write(output/'selfplay.json', dict(records=records, simulations=256,
                  weights_before=before, weights_after=before, weights_unchanged=True,
                  exact_games_targets_rng=True))


def summary(records, memory, profiles):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record['position'], record['simulations'])].append(record)
    cells = []
    for (position, budget), group in grouped.items():
        by_backend = {}
        for backend in BACKENDS:
            trials = [t for r in group for t in r['trials'][backend]]
            medians = [audit.descriptive([t['wall_seconds'] for t in r['trials'][backend]])['median'] for r in group]
            by_backend[backend] = dict(wall=audit.descriptive([t['wall_seconds'] for t in trials]),
                cpu=audit.descriptive([t['cpu_seconds'] for t in trials]),
                median_of_checkpoint_medians=float(np.median(medians)),
                inference_calls=audit.descriptive([t['inference_calls'] for t in trials]),
                terminal_leaves=audit.descriptive([t['terminal_leaves'] for t in trials]),
                simulations_per_second=audit.descriptive([t['simulations_per_second'] for t in trials]))
        cells.append(dict(position=position, simulations=budget, backends=by_backend,
            speedup=by_backend['engine']['median_of_checkpoint_medians']/by_backend['bitboard']['median_of_checkpoint_medians'],
            p95_ratio=by_backend['bitboard']['wall']['p95']/by_backend['engine']['wall']['p95']))
    exclusive = {}
    for backend in BACKENDS:
        totals = defaultdict(float)
        elapsed = 0
        for p in profiles:
            if p['backend'] == backend:
                elapsed += p['instrumented_wall_seconds']
                for k, v in p['exclusive_seconds'].items(): totals[k] += v
                totals['other'] += p['other_seconds']
        exclusive[backend] = {k:100*v/elapsed for k,v in totals.items()}
    regressions = []
    for r in records:
        p95 = {b:audit.descriptive([t['wall_seconds'] for t in r['trials'][b]])['p95'] for b in BACKENDS}
        if p95['bitboard'] > 1.05*p95['engine']:
            regressions.append(dict(seed=r['seed'], generation=r['generation'], position=r['position'],
                                    simulations=r['simulations'], p95=p95))
    mem_ratios = [m['bitboard']['peak_bytes']/m['engine']['peak_bytes'] for m in memory]
    return dict(cells=cells, exclusive_runtime_percent=exclusive, per_checkpoint_p95_regressions=regressions,
        memory_peak_ratio=audit.descriptive(mem_ratios),
        performance_target_pass=all(c['speedup']>=1.5 for c in cells if c['position'] in ('empty','early_opening','quiet_midgame')),
        p95_pass=not regressions and all(c['p95_ratio']<=1.05 for c in cells),
        allocations_pass=all(r<=1 for r in mem_ratios),
        note='Aggregate latency is median of eight checkpoint medians. P95 pooled across eight checkpoints '
             'is descriptive of this fixed mixture; per-checkpoint tails with ten trials are unstable.')


def run(args, campaign, output, deadline):
    rows = audit.read(BASE/'profile/positions.json')
    audit.write(output/'positions.json', rows)
    if args.mode == 'focus':
        row = next(r for r in rows if r['label'] == args.position)
        game = engine_position(row['moves'])
        inference = network.load_inference_checkpoint(audit.checkpoint(campaign, args.seed, args.generation))
        engines = {b:search.PUCTSearch(inference,args.simulations,backend=b) for b in BACKENDS}
        for b in BACKENDS:
            engines[b].run(game,rng=random.Random(1))
        trials = {b:[] for b in BACKENDS}
        for trial in range(args.repeats):
            domain = f'{DOMAIN}:{row["id"]}:{args.simulations}:{trial}'
            pair = {}
            for b in (BACKENDS if trial % 2 == 0 else BACKENDS[::-1]):
                deadline(); gc.collect()
                rng = seeded_rng(domain,4404001)
                cpu, wall = time.process_time(),time.perf_counter()
                result = engines[b].run(game,rng=rng)
                wall,cpu = time.perf_counter()-wall,time.process_time()-cpu
                trials[b].append(dict(trial=trial,wall_seconds=wall,cpu_seconds=cpu,
                    inference_calls=1+args.simulations-result.terminal_leaves,
                    terminal_leaves=result.terminal_leaves,visits=result.visits))
                pair[b] = (result,rng.getstate())
            assert_result(pair['engine'][0],pair['bitboard'][0])
            assert pair['engine'][1] == pair['bitboard'][1]
            del pair,result
        stats = {b:{clock:audit.descriptive([t[clock+'_seconds'] for t in trials[b]])
                    for clock in ('wall','cpu')} for b in BACKENDS}
        audit.write(output/'focus.json',dict(seed=args.seed,generation=args.generation,
            position=args.position,simulations=args.simulations,trials=trials,statistics=stats,
            speedup=stats['engine']['wall']['median']/stats['bitboard']['wall']['median'],
            p95_ratio=stats['bitboard']['wall']['p95']/stats['engine']['wall']['p95'],
            exact=True,note='Preserved follow-up; original outlier is retained, never excluded or replaced.'))
        return
    if args.mode == 'rss':
        inference = network.load_inference_checkpoint(audit.checkpoint(campaign,42,10))
        game = engine_position(rows[2]['moves'])
        search.PUCTSearch(inference,512,backend=args.backend).run(game,rng=random.Random(1))
        probe, result = memory_probe(inference, game,512,DOMAIN+':rss',args.backend)
        probe['rss_before_probe_note'] = 'Fresh process; same checkpoint and warmed quiet 512 search. Lifetime peak includes imports/loading/warmup.'
        # getrusage is available inside the sandbox; this is lifetime peak RSS,
        # not current residency or an incremental search-only measurement.
        probe['rss_measurement'] = 'resource.getrusage(RUSAGE_SELF).ru_maxrss, macOS bytes / 2**20'
        audit.write(output/'rss.json',probe)
        return
    audit.write(output/'transitions.json', transitions(rows))
    records, validations, memory, profiles = [], [], [], []
    for seed in audit.SEEDS:
        for generation in audit.GENERATIONS:
            inference = network.load_inference_checkpoint(audit.checkpoint(campaign,seed,generation))
            weights = network.weights_sha256(inference.model)
            for row in rows:
                game = engine_position(row['moves'])
                for budget in (256,512):
                    engines = {b:search.PUCTSearch(inference,budget,backend=b) for b in BACKENDS}
                    # Full warm-up of both implementations outside timed trials.
                    for b in BACKENDS:
                        engines[b].run(game,rng=random.Random(1))
                    trials = {b:[] for b in BACKENDS}
                    for trial in range(args.repeats):
                        domain = f'{DOMAIN}:{row["id"]}:{budget}:{trial}'
                        pair = {}
                        for b in (BACKENDS if trial%2==0 else BACKENDS[::-1]):
                            deadline(); gc.collect()
                            rng = seeded_rng(domain,4404001)
                            cpu, wall = time.process_time(), time.perf_counter()
                            result = engines[b].run(game,rng=rng)
                            wall, cpu = time.perf_counter()-wall,time.process_time()-cpu
                            trials[b].append(dict(trial=trial, wall_seconds=wall,cpu_seconds=cpu,
                                inference_calls=1+budget-result.terminal_leaves,
                                simulations_per_second=budget/wall, terminal_leaves=result.terminal_leaves,
                                visits=result.visits,max_depth=result.max_depth,leaf_depth_sum=result.leaf_depth_sum))
                            pair[b] = (result,rng.getstate())
                        assert_result(pair['engine'][0],pair['bitboard'][0])
                        assert pair['engine'][1] == pair['bitboard'][1]
                        del pair, result
                    identity=dict(seed=seed,generation=generation,position=row['label'],simulations=budget)
                    records.append(dict(**identity,trials=trials))
                    with (output/'latency.jsonl').open('a') as f: f.write(json.dumps(records[-1])+'\n')
                    # Three independent trials, with and without selfplay noise;
                    # inspect every leaf input/raw output/legal policy, all nodes,
                    # both action temperatures and all private RNG states.
                    for trial in range(args.validation_trials):
                        for noisy in (False,True):
                            deadline()
                            validation=paired_validation(inference,game,budget,
                                f'{DOMAIN}:validation:{row["id"]}:{trial}',noisy=noisy)
                            validations.append(dict(**identity,trial=trial,**validation))
                    domain=f'{DOMAIN}:memory:{row["id"]}'
                    probes={}
                    for b in BACKENDS:
                        deadline()
                        probes[b], result=memory_probe(inference,game,budget,domain,b)
                        del result
                    memory.append(dict(**identity,**probes))
                    for b in BACKENDS:
                        deadline()
                        profiles.append(dict(**identity,**profile_probe(inference,game,budget,domain,b)))
            assert weights == network.weights_sha256(inference.model)
            print(f'complete seed={seed} generation={generation}',flush=True)
    audit.write(output/'latency.json',records)
    audit.write(output/'equivalence.json',dict(records=validations, exact=True,
        paired_searches=len(validations),nodes=sum(v['nodes'] for v in validations),
        predictions=sum(v['predictions'] for v in validations)))
    audit.write(output/'memory.json',memory); audit.write(output/'profile.json',profiles)
    audit.write(output/'summary.json',summary(records,memory,profiles))
    # Existing development tactical fixtures: exact differential checks, no new strength tournament.
    inference=network.load_inference_checkpoint(audit.checkpoint(campaign,42,10))
    tactical=[]
    for row in audit.package(campaign,'tactical'):
        deadline()
        tactical.append(dict(id=row['id'],**paired_validation(inference,engine_position(row['moves']),512,
                                                             DOMAIN+':tactical:'+row['id'])))
    audit.write(output/'tactical.json',dict(checkpoint='42/g10',budget=512,records=tactical,exact=True))
    selfplay_probe(inference,output,args.games,deadline)
    for b in BACKENDS:
        profiler=cProfile.Profile()
        profiler.runcall(search.PUCTSearch(inference,512,backend=b).run,
                        engine_position(rows[2]['moves']),rng=random.Random(4404001))
        profiler.dump_stats(str(output/f'{b}.pstats'))
        stats=pstats.Stats(profiler)
        functions=[dict(file=f,line=line,function=name,primitive_calls=v[0],calls=v[1],
                        self_seconds=v[2],cumulative_seconds=v[3])
                   for (f,line,name),v in stats.stats.items()]
        audit.write(output/f'{b}-cprofile.json',sorted(functions,key=lambda r:r['self_seconds'],reverse=True))


def main():
    if not __debug__:
        raise RuntimeError('Normal Python required: -O disables scientific equivalence gates')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=10)
    parser.add_argument('--validation-trials',type=int,default=3)
    parser.add_argument('--games',type=int,default=4)
    parser.add_argument('--seconds',type=float,default=1500)
    parser.add_argument('--mode',choices=('all','rss','focus'),default='all')
    parser.add_argument('--backend',choices=BACKENDS,default='engine')
    parser.add_argument('--seed',type=int,choices=audit.SEEDS,default=42)
    parser.add_argument('--generation',type=int,choices=audit.GENERATIONS,default=10)
    parser.add_argument('--position',default='tactical_midgame',choices=(
        'empty','early_opening','quiet_midgame','tactical_midgame','late_game','immediate_win','mandatory_block'))
    parser.add_argument('--simulations',type=int,choices=(256,512),default=256)
    args=parser.parse_args()
    if args.repeats<3 or args.validation_trials<3 or args.games<1 or not 0<args.seconds<=1800:
        parser.error('Require >=3 repeats/validation trials, >=1 game and 0<seconds<=1800')
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        if os.environ.get(key)!='1': parser.error(f'{key} must be pinned to 1 before importing torch')
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    campaign=args.campaign.resolve()
    output=audit.fresh_output(campaign,args.output)
    before=audit.campaign_fingerprint(campaign)
    audit.write(output/'campaign_before.json',before)
    status={'complete':False}
    started=time.perf_counter()
    try:
        assert len(before)==150 and before==audit.read(BASE/'profile/campaign_before.json')
        assert audit.sha(campaign/'declaration.json')==audit.DECLARATION
        checkpoints=audit.read(BASE/'profile/provenance.json')['checkpoints']
        for c in checkpoints: assert audit.sha(Path(c['path']))==c['sha256']
        source_paths=[*Path('games/connect4').rglob('*.py'),
                      *Path('research/alphazero_bitboard').glob('*.py'),
                      Path('research/alphazero_baseline/audit.py')]
        sources={str(p):audit.sha(p) for p in source_paths}
        import shutil
        for path in source_paths:
            target=output/'source'/path
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(path,target)
        params={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        from datetime import datetime, timezone
        audit.write(output/'provenance.json',dict(runtime=audit.runtime(),parameters=params,
            started_utc=datetime.now(timezone.utc).isoformat(),
            checkpoints=checkpoints,source_sha256=sources,
            base_commit='2fbbdeed452e9d73461f36601296b01e5fb34af6',
            bitboard_reference_commit='8704142702da273e6571f0ca9df3fd4a793be285'))
        run(args,campaign,output,audit.Deadline(args.seconds))
        status['complete']=True
    except BaseException as e:
        status['error']=repr(e)
        raise
    finally:
        after=audit.campaign_fingerprint(campaign)
        audit.write(output/'preservation.json',dict(files=len(after),unchanged=before==after,
                    changed=[p for p in set(before)|set(after) if before.get(p)!=after.get(p)]))
        status['execution_seconds']=time.perf_counter()-started
        audit.write(output/'status.json',status)
        if before!=after: raise RuntimeError('Official artifacts changed')


if __name__=='__main__': main()
