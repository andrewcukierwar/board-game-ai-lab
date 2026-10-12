"""Bounded frozen-checkpoint Phase 4D.4E collection; never trains.

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
BASE_COMMIT = '08656faa1efb9d76fb9ae546f894164011e0b453'
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
    parser.add_argument('--mode', choices=['main'], required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--games', type=int, help='games per checkpoint/schedule/cohort cell')
    args = parser.parse_args()
    games = args.games or 64
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
                    *Path('research/alphazero_exploration_confirmation').glob('*.py'),
                    Path('research/alphazero_exploration_confirmation/DESIGN.md')]
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
        design_sha256=audit.sha('research/alphazero_exploration_confirmation/DESIGN.md'),
        schedules=[x.to_dict() for x in s.SCHEDULES], generation_id=s.GENERATION, pool=s.POOL,
        simulations=s.SIMULATIONS, cohort_seeds={f'{k[0]}/g{k[1]}': v for k, v in s.COHORT_SEEDS.items()},
        games_per_cell=games,
        parameters={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    deadline = audit.Deadline(args.seconds)
    status = dict(complete=False, mode=args.mode)
    try:
        matrix(campaign, output, deadline, s.PRIMARY, s.COHORT_SEEDS.get, games, 'main')
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
