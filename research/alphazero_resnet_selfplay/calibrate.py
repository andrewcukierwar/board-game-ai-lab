"""Bounded cost calibration before the budget freeze. Timing and saturation only.

What it measures, with TRAINED weights and one deterministic thread:

1. Pool-eight, 256-simulation self-play cost for the CNN and ResNet 3x64, in
   an ABBA order of half-generation blocks that add up to one proposed
   generation per architecture (192 CNN games, 128 ResNet games).
2. Batch-128 update cost on positions from those games, on disposable copies.
3. Raw-PUCT evaluation cost per fixture position and per match game.
4. Whether reference negamax at depth 1, 2 or 3 is saturated as an opponent for
   early-generation networks, using official CNN generation 2-4 inference
   artifacts.

The trained weights are Phase 4D.4H-B's saved 3,000-update states (both
architectures trained on identical batches) and official inference artifacts.
Every one is opened read-only, hash-checked against a tracked record, and the
protected trees are inventoried before and after. No experimental arm exists
yet; nothing here trains, saves or selects a model, and no outcome measured
here compares the two architectures' strength.
"""
from copy import deepcopy
import random
import time

import torch

from .common import (HISTORICAL, IDENTITY, NAMES, OFFICIAL, PHASE4D4HB, REPO, RESULTS, SEEDS, compare_protected,
                     cpu_seconds, peak_rss_mib, preserve, protected_inventory, read, runtime, sha, write)
from . import evaluate

CALIBRATION_SEED = 20261012
CEILING_SECONDS = 15 * 60
PROPOSED_GAMES = dict(cnn=192, resnet3x64=128)
SIMULATIONS = 256
UPDATES = 24
OPPONENT_DEPTHS = (1, 2, 3)
PROBE_CHECKPOINTS = ((42, 2), (42, 4), (314159, 3))
PROBE_OPENINGS = 20
OPPONENT_RULE = ('choose the depth whose mean score over the probed official checkpoints is closest to 0.50, '
                 'among depths where every probed checkpoint scores within [0.15, 0.85]')


def trained(name, seed):
    """Phase 4D.4H-B's 3,000-update state, verified against the tracked evidence record."""
    from games.connect4.alphazero_v2.experimental_models import model_factory
    path = PHASE4D4HB / f'experiment-output/phase4d4h-b/run/learning-seed-{seed}/{name}-model-03000.pt'
    record = read(REPO / f'research/alphazero_resnet_benchmark/results/run-evidence/learning-seed-{seed}/{name}.json')
    digest = sha(path)
    if digest != record['models']['3000']:
        raise ValueError(f'{path} differs from the tracked Phase 4D.4H-B record')
    factory = model_factory(IDENTITY[name])
    model = factory.restore(torch.load(path, map_location='cpu', weights_only=True))
    return factory, model, dict(path=str(path), file_sha256=digest, weights_sha256=factory.hash(model), updates=3000)


def collect(name, seed, first, count, check):
    from games.connect4.alphazero_v2.batching import CollectionStats, collect_batched
    from games.connect4.alphazero_v2.config import V2Config
    from games.connect4.alphazero_v2.experimental import ORIGINAL
    factory, model, identity = trained(name, seed)
    stats = CollectionStats()
    cpu, tick = cpu_seconds(), time.perf_counter()
    games = collect_batched(factory.inference(factory.freeze(model)), V2Config(seed=CALIBRATION_SEED), 1,
                            range(first, first + count), batch_size=8, stats=stats, check=check,
                            temperature_schedule=ORIGINAL)
    wall = time.perf_counter() - tick
    plies = sum(len(g.moves) for g in games)
    row = dict(architecture=name, weights=identity, games=len(games), plies=plies, wall_seconds=wall,
               cpu_seconds=cpu_seconds() - cpu, neural_positions=stats.neural_positions,
               forward_calls=stats.forward_calls, mean_batch=stats.neural_positions / stats.forward_calls,
               inference_seconds=stats.inference_seconds, simulations=stats.simulations,
               seconds_per_game=wall / len(games), ms_per_ply=1e3 * wall / plies,
               ms_per_neural_position=1e3 * wall / stats.neural_positions, mean_game_length=plies / len(games),
               draws=sum(g.winner == -1 for g in games))
    if factory.hash(model) != identity['weights_sha256']:
        raise ValueError('Calibration changed trained weights')
    return row, [e for g in games for e in g.examples]


def updates(name, seed, examples, check):
    """Batch-128 trainer steps on a disposable copy; the first two are warm-up and excluded."""
    from games.connect4.alphazero_v2.config import V2Config
    factory, model, identity = trained(name, seed)
    disposable = factory.restore(deepcopy(factory.state(model)))
    trainer = factory.trainer(disposable, V2Config())
    rng, seconds = random.Random(CALIBRATION_SEED), []
    for _ in range(UPDATES + 2):
        check()
        batch = tuple(rng.sample(examples, 128))
        tick = time.perf_counter()
        trainer.step(batch)
        seconds.append(time.perf_counter() - tick)
    if factory.hash(model) != identity['weights_sha256']:
        raise ValueError('Calibration changed trained weights')
    measured = seconds[2:]
    return dict(architecture=name, updates=len(measured), mean_ms=1e3 * sum(measured) / len(measured),
                min_ms=1e3 * min(measured), max_ms=1e3 * max(measured))


def search_cost(name, seed, check):
    """Seconds per raw-PUCT fixture search (every 15th row of each suite)."""
    from games.connect4.alphazero_v2.data import seeded_rng
    from games.connect4.alphazero_v2.oracle import engine_position
    from games.connect4.alphazero_v2.search import EvaluationAgent
    factory, model, _ = trained(name, seed)
    inference = factory.inference(factory.freeze(model))
    data = evaluate.fixtures()
    rows = [r for suite in evaluate.SUITES for r in data[suite][::15]]
    tick = time.perf_counter()
    for row in rows:
        check()
        EvaluationAgent(inference, SIMULATIONS, rng=seeded_rng(f'calibration:{row["id"]}', 0),
                        backend='bitboard').search(engine_position(row['moves']))
    return dict(architecture=name, searches=len(rows), seconds_per_search=(time.perf_counter() - tick) / len(rows))


def official_agent(seed, generation):
    from games.connect4.alphazero_v2.network import load_inference_checkpoint, weights_sha256
    base = OFFICIAL / f'runs/seed-{seed}/generations/generation-{generation:04d}'
    path = base / f'generation-{generation:04d}.inference.pt'
    digest = sha(path)
    if digest != read(base / 'summary.json')['files']['inference']['sha256']:
        raise ValueError('Official inference artifact differs from its summary')
    inference = load_inference_checkpoint(path)
    weights = weights_sha256(inference.model)
    return evaluate.RawAgent(inference, SIMULATIONS, name=f'official-{seed}-g{generation}', weights_sha256=weights), \
        dict(seed=seed, generation=generation, file_sha256=digest, weights_sha256=weights)


def probe(agent, depth, rows, check):
    tick = time.perf_counter()
    records = evaluate.matches(agent, depth, CALIBRATION_SEED, check, None, rows=rows)
    wall = time.perf_counter() - tick
    return dict(depth=depth, games=len(records), score=sum(r['score'] for r in records) / len(records),
                wins=sum(r['result'] == 'win' for r in records), draws=sum(r['result'] == 'draw' for r in records),
                losses=sum(r['result'] == 'loss' for r in records), seconds_per_game=wall / len(records),
                mean_plies=sum(len(r['moves']) - r['prefix_length'] for r in records) / len(records))


def main():
    identity = runtime()
    official, protected = preserve(), protected_inventory()
    started, cpu = time.time(), cpu_seconds()

    def check():
        if time.time() - started >= CEILING_SECONDS:
            raise TimeoutError('Calibration ceiling reached')
    result = dict(runtime=identity, simulations=SIMULATIONS, pool=8, schedule='original eight-ply',
                  proposed_games=PROPOSED_GAMES, calibration_seed=CALIBRATION_SEED)
    # 1. Collection, ABBA: CNN 96, ResNet 64, ResNet 64, CNN 96 -> one proposed generation each.
    blocks, examples = [], {}
    half = {name: PROPOSED_GAMES[name] // 2 for name in NAMES}
    for name, seed, first in (('cnn', SEEDS[0], 0), ('resnet3x64', SEEDS[0], 0),
                              ('resnet3x64', SEEDS[1], half['resnet3x64']), ('cnn', SEEDS[1], half['cnn'])):
        row, collected = collect(name, seed, first, half[name], check)
        blocks.append(dict(row, weights_seed=seed, first_game=first))
        examples.setdefault(name, []).extend(collected)
        print(f'collection {name} seed {seed}: {row["seconds_per_game"]:.3f} s/game, {row["ms_per_ply"]:.2f} ms/ply, '
              f'{row["mean_game_length"]:.1f} plies', flush=True)
    result['collection_blocks'] = blocks
    totals = {}
    for name in NAMES:
        rows = [b for b in blocks if b['architecture'] == name]
        total = {k: sum(r[k] for r in rows) for k in ('games', 'plies', 'wall_seconds', 'cpu_seconds', 'neural_positions')}
        totals[name] = dict(total, seconds_per_game=total['wall_seconds'] / total['games'],
                            ms_per_ply=1e3 * total['wall_seconds'] / total['plies'],
                            ms_per_neural_position=1e3 * total['wall_seconds'] / total['neural_positions'],
                            mean_game_length=total['plies'] / total['games'],
                            cpu_over_wall=total['cpu_seconds'] / total['wall_seconds'])
    result['collection'] = totals
    # 2. Updates, ABBA.
    rows = [updates(name, seed, examples[name], check)
            for name, seed in (('cnn', SEEDS[0]), ('resnet3x64', SEEDS[0]), ('resnet3x64', SEEDS[1]), ('cnn', SEEDS[1]))]
    result['update_blocks'] = rows
    result['update_ms'] = {name: sum(r['mean_ms'] for r in rows if r['architecture'] == name) / 2 for name in NAMES}
    print('update ms', result['update_ms'], flush=True)
    # 3. Evaluation cost.
    result['search'] = {name: search_cost(name, SEEDS[0], check) for name in NAMES}
    print('search', result['search'], flush=True)
    # 4. Opponent saturation probe with official early-generation CNN artifacts.
    openings = evaluate.openings()[:PROBE_OPENINGS]
    probes = []
    for seed, generation in PROBE_CHECKPOINTS:
        agent, provenance = official_agent(seed, generation)
        for depth in OPPONENT_DEPTHS:
            row = dict(probe(agent, depth, openings, check), checkpoint=provenance)
            probes.append(row)
            print(f'probe seed {seed} g{generation} negamax{depth}: score {row["score"]:.3f}, '
                  f'{row["seconds_per_game"]:.2f} s/game', flush=True)
    by_depth = {}
    for depth in OPPONENT_DEPTHS:
        scores = [p['score'] for p in probes if p['depth'] == depth]
        by_depth[str(depth)] = dict(scores=scores, mean=sum(scores) / len(scores),
                                    eligible=all(0.15 <= s <= 0.85 for s in scores),
                                    seconds_per_game=sum(p['seconds_per_game'] for p in probes if p['depth'] == depth) / len(scores))
    eligible = [d for d in OPPONENT_DEPTHS if by_depth[str(d)]['eligible']]
    chosen = min(eligible, key=lambda d: (abs(by_depth[str(d)]['mean'] - 0.5), d)) if eligible else None
    result['opponent'] = dict(rule=OPPONENT_RULE, probes=probes, by_depth=by_depth, chosen_depth=chosen,
                              openings=[r['id'] for r in openings])
    # Match cost of each architecture against the chosen opponent (cost only; the result is not recorded as strength).
    if chosen is not None:
        costs = {}
        for name in NAMES:
            factory, model, _ = trained(name, SEEDS[0])
            agent = evaluate.RawAgent(factory.inference(factory.freeze(model)), SIMULATIONS, name=name,
                                      weights_sha256=factory.hash(model))
            row = probe(agent, chosen, openings[:4], check)
            costs[name] = dict(games=row['games'], seconds_per_game=row['seconds_per_game'], mean_plies=row['mean_plies'])
        result['match_cost'] = costs
    # Ratios used by the budget freeze.
    c, r = totals['cnn'], totals['resnet3x64']
    result['ratios'] = dict(
        self_play_per_neural_position=r['ms_per_neural_position'] / c['ms_per_neural_position'],
        self_play_per_ply=r['ms_per_ply'] / c['ms_per_ply'], self_play_per_game=r['seconds_per_game'] / c['seconds_per_game'],
        update=result['update_ms']['resnet3x64'] / result['update_ms']['cnn'],
        search_per_fixture=result['search']['resnet3x64']['seconds_per_search'] / result['search']['cnn']['seconds_per_search'])
    after = protected_inventory()
    comparison = compare_protected(protected, after)
    if preserve() != official or not comparison['intact']:
        raise ValueError('A protected file changed during calibration')
    result.update(wall_seconds=time.time() - started, cpu_seconds=cpu_seconds() - cpu, peak_rss_mib=peak_rss_mib(),
                  ceiling_seconds=CEILING_SECONDS, official_files=len(official),
                  protected={k: dict(files=v['files_at_launch'], intact=v['intact']) for k, v in comparison['trees'].items()},
                  arms_trained=0, checkpoints_written=0, historical_checkout=str(HISTORICAL))
    write(RESULTS / 'calibration.json', result)
    print('ratios', result['ratios'], 'chosen depth', chosen, f'wall {result["wall_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
