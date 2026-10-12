"""Bounded frozen-checkpoint Phase 4D.4D collection; never trains.

Every invocation refuses an existing output directory, verifies all 150
official campaign bytes before and after, and writes each COMPLETE game as one
JSON line the moment it finishes. A wall-clock stop leaves complete games on
disk and marks the interrupted cell incomplete; analysis never scores it.
"""
import argparse
from collections import defaultdict
import gc
import json
import os
from pathlib import Path
import random
import tarfile
import time

import torch

from games.connect4.alphazero_v2 import batching as b, network, search
from games.connect4.alphazero_v2.arena import StopEvaluation
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.diagnostics import peak_rss_mib
from games.connect4.alphazero_v2.oracle import engine_position
from research.alphazero_baseline import audit
from research.alphazero_batching.benchmark import digest, game_records, hardware, require
from . import schedules as s

BASE = Path('research/alphazero_baseline/results/phase4d4a-20261011')
PHASE_4C = Path('research/alphazero_batching/results/phase4d4c-20261011')
BASE_COMMIT = '99cec1c5f45584145d546b28f19b3d2097b37421'
BUDGET_SECONDS = 1800


def normalized(value):
    """JSON round trip, so tuples compare equal to retained list records."""
    return json.loads(json.dumps(value, allow_nan=False))


def load(campaign, seed, generation):
    inference = network.load_inference_checkpoint(audit.checkpoint(campaign, seed, generation))
    # Same warmup as Phase 4D.4C; a private RNG, never a game-owned stream.
    state = BitboardState.from_game(engine_position([]))
    search.PUCTSearch(inference, 32, backend='bitboard').run(state, rng=random.Random(0))
    inference.predict_batch([state] * s.POOL)
    return inference


def run_cell(inference, schedule, base_seed, ids, deadline, path, **identity):
    """Collect one checkpoint/schedule/identity cell at pool eight.

    ``schedule`` None exercises the unchanged default path (regression only).
    Returns (meta, games or None); a deadline stop keeps the finished lines.
    """
    config = V2Config(seed=base_seed)
    require(config.self_play_simulations == s.SIMULATIONS, 'Unexpected simulation budget')
    expected = s.ORIGINAL if schedule is None else schedule
    stats, rows = b.CollectionStats(), defaultdict(list)
    frozen = network.weights_sha256(inference.model)

    def observe(index, decision, seconds):
        result, action = decision.search, decision.action
        require(result.simulations == s.SIMULATIONS and sum(result.visits) == s.SIMULATIONS
                and result.root.visits == s.SIMULATIONS, 'Search did not execute exactly 256 simulations')
        require(action.temperature == expected.temperature(decision.ply), 'Schedule misapplied')
        rows[index].append(dict(ply=decision.ply, visits=result.visits, action=action.move,
                                temperature=action.temperature, distribution=action.distribution,
                                root_prior=result.root_prior, root_value=result.root_value,
                                terminal_leaves=result.terminal_leaves, max_depth=result.max_depth))

    gc.collect()
    games, stop = None, None
    with path.open('x') as stream:
        def on_game(game):
            stream.write(json.dumps(dict(
                index=game.index, moves=game.moves, winner=game.winner,
                outcomes=[e.outcome for e in game.examples], plies=rows.pop(game.index),
                record_sha256=digest(game_records([game])), rng_sha256=digest(stats.rng_states[game.index]),
                completion_seconds=stats.completion_seconds[game.index]), allow_nan=False) + '\n')
            stream.flush()
        cpu, started = time.process_time(), time.perf_counter()
        try:
            games = b.collect_batched(inference, config, s.GENERATION, ids, batch_size=s.POOL, check=deadline,
                                      observer=observe, on_game=on_game, stats=stats,
                                      temperature_schedule=schedule)
        except StopEvaluation as stopped:
            stop = str(stopped)
        wall, cpu = time.perf_counter() - started, time.process_time() - cpu
    require(network.weights_sha256(inference.model) == frozen, 'Weights changed')
    finished = len(stats.completion_seconds)
    require(games is None or (finished == len(ids) and stats.simulations == stats.plies * s.SIMULATIONS),
            'Complete cell accounting failed')
    meta = dict(identity, schedule=expected.to_dict(), explicit_schedule=schedule is not None,
                base_seed=base_seed, generation_id=s.GENERATION, ids=list(ids), pool=s.POOL,
                complete=games is not None, stop_reason=stop, file=path.name, completed_games=finished,
                wall_seconds=wall, cpu_seconds=cpu, plies=stats.plies, simulations=stats.simulations,
                terminal_leaves=stats.terminal_leaves, forward_calls=stats.forward_calls,
                neural_positions=stats.neural_positions,
                batch_sizes={str(k): v for k, v in sorted(stats.batch_sizes.items())},
                mean_batch=stats.neural_positions / max(1, stats.forward_calls),
                inference_seconds=stats.inference_seconds, max_active=stats.max_active,
                games_per_second=finished / wall, peak_rss_mib=peak_rss_mib(), weights_sha256=frozen)
    if games is not None:
        # Plies of unfinished games are counted by the collector but never stored or scored.
        meta.update(positions_per_second=stats.plies / wall, records_sha256=digest(game_records(games)),
                    rng_sha256={str(g.index): digest(stats.rng_states[g.index]) for g in games})
    return meta, games


def searched(game):
    """Outcome-free example content: later divergence may change final labels."""
    return [(e.observation, e.visits, e.action, e.action_temperature) for e in game.examples]


def regression(campaign, output, deadline, args):
    """Default path and explicit ORIGINAL against retained Phase 4D.4C records; repeats."""
    import gzip
    checks, cells = [], []
    config_ids = range(16)
    inference = load(campaign, 314159, 20)
    retained = json.load(gzip.open(PHASE_4C / 'collection/trial-0-314159-8.json.gz', 'rt'))
    for label, schedule in (('default', None), ('original', s.ORIGINAL)):
        meta, games = run_cell(inference, schedule, s.REGRESSION_SEED, config_ids, deadline,
                               output / f'regression-314159-g20-{label}.jsonl', seed=314159, generation=20,
                               check=label)
        require(games is not None, 'Regression cell stopped')
        cells.append(meta)
        checks.append(dict(check=f'314159/g20 16 games pool 8, {label} schedule vs retained 4C trial',
                           records_exact=normalized(game_records(games)) == retained['games'],
                           rng_exact=meta['rng_sha256'] == retained['rng_sha256'], games=len(games)))
    inference = load(campaign, 42, 20)
    validation = json.load(gzip.open(PHASE_4C / 'numerical/complete_validation.json.gz', 'rt'))
    retained = next(r for r in validation if (r['seed'], r['generation'], r['size']) == (42, 20, 8))
    originals = {}
    for label, schedule in (('default', None), ('original', s.ORIGINAL)):
        meta, games = run_cell(inference, schedule, s.REGRESSION_SEED, range(4), deadline,
                               output / f'regression-42-g20-{label}.jsonl', seed=42, generation=20, check=label)
        require(games is not None, 'Regression cell stopped')
        cells.append(meta)
        originals[label] = (games, meta['rng_sha256'])
        checks.append(dict(check=f'42/g20 4 games pool 8, {label} schedule vs retained 4C validation',
                           records_exact=normalized(game_records(games)) == retained['games'], games=len(games)))
    checks.append(dict(check='42/g20 explicit ORIGINAL equals default records and final RNG states',
                       records_exact=originals['default'][0] == originals['original'][0],
                       rng_exact=originals['default'][1] == originals['original'][1], games=4))
    for schedule in s.SCHEDULES[1:]:
        runs = []
        for repeat in range(2):
            meta, games = run_cell(inference, schedule, s.REGRESSION_SEED, range(4), deadline,
                                   output / f'regression-42-g20-{schedule.name}-{repeat}.jsonl',
                                   seed=42, generation=20, check=f'repeat-{repeat}')
            require(games is not None, 'Regression cell stopped')
            cells.append(meta)
            runs.append((games, meta['rng_sha256']))
        # Negative control: searched examples before the first changed ply equal ORIGINAL's.
        prefix = next(p for p in range(42) if schedule.temperature(p) != s.ORIGINAL.temperature(p))
        checks.append(dict(check=f'42/g20 {schedule.name} repeated identities reproduce exactly',
                           records_exact=runs[0][0] == runs[1][0], rng_exact=runs[0][1] == runs[1][1], games=4,
                           shared_prefix_plies=prefix,
                           prefix_exact=all(searched(x)[:prefix] == searched(y)[:prefix]
                                            for x, y in zip(runs[0][0], originals['original'][0]))))
    audit.write(output / 'cells.json', cells)
    audit.write(output / 'regression.json', checks)
    failed = [c for c in checks if not all(v for k, v in c.items() if k.endswith('_exact'))]
    print(json.dumps(dict(checks=len(checks), failed=[c['check'] for c in failed])), flush=True)
    require(not failed, 'Original-schedule regression or reproducibility failed')


def matrix(campaign, output, deadline, checkpoints, seeds_for, games, mode):
    """Cohort-major so a wall-clock stop still leaves complete matched cohorts."""
    models, cells = {}, []
    try:
        for cohort in range(len(seeds_for(checkpoints[0]))):
            for position, checkpoint in enumerate(checkpoints):
                if checkpoint not in models:
                    models[checkpoint] = load(campaign, *checkpoint)
                shift = (cohort + position) % len(s.SCHEDULES)
                for schedule in s.SCHEDULES[shift:] + s.SCHEDULES[:shift]:
                    seed, generation = checkpoint
                    name = f'{mode}-{seed}-g{generation}-c{cohort}-{schedule.name}.jsonl'
                    meta, collected = run_cell(models[checkpoint], schedule, seeds_for(checkpoint)[cohort],
                                               range(games), deadline, output / name, seed=seed,
                                               generation=generation, cohort=cohort, mode=mode)
                    cells.append(meta)
                    print(f'{mode} {seed}/g{generation} cohort {cohort} {schedule.name}: '
                          f'{meta["completed_games"]} games {meta["wall_seconds"]:.1f}s', flush=True)
                    if collected is None:
                        raise StopEvaluation(meta['stop_reason'])
    finally:
        audit.write(output / 'cells.json', cells)
    return cells


def main():
    if not __debug__:
        raise RuntimeError('Collection refuses python -O')
    invoked = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['regression', 'pilot', 'main', 'supplement'], required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--games', type=int, help='games per checkpoint/schedule/cohort cell')
    args = parser.parse_args()
    games = args.games or dict(regression=16, pilot=8, main=64, supplement=16)[args.mode]
    require(0 < args.seconds <= BUDGET_SECONDS and games > 0, 'Invalid budget')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                                   'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')),
            'Set all five vector thread environment variables to 1 before Python starts')
    campaign = args.campaign.resolve()
    require(audit.sha(campaign / 'declaration.json') == audit.DECLARATION, 'Declaration changed')
    output = audit.fresh_output(campaign, args.output)
    before = audit.campaign_fingerprint(campaign)
    require(before == audit.read(BASE / 'profile/campaign_before.json') and len(before) == 150,
            'Official 150-file inventory changed')
    audit.write(output / 'campaign_before.json', before)
    source_files = [*Path('games/connect4/alphazero_v2').glob('*.py'), Path('games/connect4/agents/mcts_nn_agent.py'),
                    Path('games/connect4/neural_mcts.py'), Path('games/connect4/connect4.py'),
                    *Path('research/alphazero_exploration').glob('*.py'),
                    Path('research/alphazero_exploration/DESIGN.md')]
    with tarfile.open(output / 'source.tar.gz', 'w:gz') as tar:
        for path in source_files:
            tar.add(path, arcname=str(path))
    checkpoints = []
    for seed in audit.SEEDS:
        for generation in audit.GENERATIONS:
            path = audit.checkpoint(campaign, seed, generation)
            expected = (audit.read(path.parent / 'summary.json')['files']['inference']['sha256'] if generation else
                        audit.read(campaign / f'runs/seed-{seed}/selection.json')['inference']['sha256'])
            require(audit.sha(path) == expected, 'Checkpoint changed')
            checkpoints.append(dict(seed=seed, generation=generation, path=str(path), sha256=expected))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    audit.write(output / 'provenance.json', dict(
        runtime=audit.runtime(), hardware=hardware(), source={str(p): audit.sha(p) for p in source_files},
        source_base_commit=BASE_COMMIT, checkpoints=checkpoints, rng_protocol=b.PROTOCOL,
        design_sha256=audit.sha('research/alphazero_exploration/DESIGN.md'),
        schedules=[x.to_dict() for x in s.SCHEDULES], generation_id=s.GENERATION, pool=s.POOL,
        simulations=s.SIMULATIONS, cohort_seeds={f'{k[0]}/g{k[1]}': v for k, v in s.COHORT_SEEDS.items()},
        pilot_seed=s.PILOT_SEED, regression_seed=s.REGRESSION_SEED, games_per_cell=games,
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    deadline = audit.Deadline(args.seconds)
    status = dict(complete=False, mode=args.mode)
    try:
        if args.mode == 'regression':
            regression(campaign, output, deadline, args)
        elif args.mode == 'pilot':
            matrix(campaign, output, deadline, s.PRIMARY, lambda c: (s.PILOT_SEED,), games, 'pilot')
        elif args.mode == 'main':
            matrix(campaign, output, deadline, s.PRIMARY, s.COHORT_SEEDS.get, games, 'main')
        else:
            matrix(campaign, output, deadline, s.SUPPLEMENT, s.COHORT_SEEDS.get, games, 'supplement')
        status['complete'] = True
    except StopEvaluation as stopped:
        status['stop_reason'] = str(stopped)
    except BaseException as error:
        status['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        # elapsed_seconds is the whole invocation, including both 150-file hash passes.
        status.update(elapsed_seconds=time.perf_counter() - invoked,
                      collection_seconds=time.perf_counter() - deadline.started, peak_rss_mib=peak_rss_mib())
        after = audit.campaign_fingerprint(campaign)
        audit.write(output / 'preservation.json', dict(
            files=len(after), unchanged=before == after,
            changed=sorted(p for p in set(before) | set(after) if before.get(p) != after.get(p)),
            declaration_sha256=audit.sha(campaign / 'declaration.json')))
        audit.write(output / 'status.json', status)
        require(before == after, 'Official campaign modified')
    print(json.dumps(status), flush=True)
    return 0 if status['complete'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
