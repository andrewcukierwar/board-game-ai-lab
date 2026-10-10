"""Separate uninstrumented timings, traced allocations, and cProfile runs."""
import argparse
import gc
import json
import random
import statistics
import time
import tracemalloc
from pathlib import Path

from games.connect4.agents.mcts_agent import MCTSAgent
from scripts.benchmark_mcts_bitboards import profile_run
from scripts.benchmark_public_agents import POSITIONS, position
from scripts.evaluate_mcts_strength import atomic_json, digest, seed_for, source_hashes


class Capture(MCTSAgent):
    """Untimed diagnostics only; never substitute this in the latency run."""
    root = None

    def _backpropagate(self, node, winner):
        if self.root is None:
            root = node
            while root.parent:
                root = root.parent
            self.root = root
        super()._backpropagate(node, winner)


def memory_run(game, budget, seed):
    gc.collect()
    tracemalloc.start()
    agent = Capture(budget, rng=random.Random(seed))
    move = agent.choose_move(game)
    retained, peak = tracemalloc.get_traced_memory()
    snapshot = tracemalloc.take_snapshot()
    allocations = [dict(file=Path(stat.traceback[0].filename).name,
                        line=stat.traceback[0].lineno, bytes=stat.size, count=stat.count)
                   for stat in snapshot.statistics('lineno')[:12]]
    tracemalloc.stop()
    root = agent.root
    pending = [(root, 0)] if root else []
    nodes = max_depth = 0
    while pending:
        node, depth = pending.pop()
        nodes += 1
        max_depth = max(max_depth, depth)
        pending.extend((child, depth + 1) for child in node.children.values())
    return dict(move=move, retained_bytes=retained, peak_bytes=peak, nodes=nodes,
                max_depth=max_depth, executed_simulations=root.visits if root else 0,
                top_retained_allocations=allocations, rng_sha256=digest(agent.rng.getstate()))


def categories(profile):
    functions = profile['functions']
    def cumulative(name):
        return sum(f['cumulative_seconds'] for f in functions if f['function'] == name)
    # Disjoint method subtrees; expansion components count only choose_move
    # callers so root tactical drop/Node work is not counted twice.
    expansion = sum(c['cumulative_seconds'] for f in functions
        if (f['function'] == 'drop' or (f['function'] == '__init__' and f['file'] == 'mcts_agent.py'))
        for c in f['callers'] if c['function'] == 'choose_move')
    parts = dict(selection=cumulative('_select_child'), rollout=cumulative('_simulate'),
                 expansion_components=expansion, backpropagation=cumulative('_backpropagate'),
                 root_tactics=cumulative('_winning_moves') + cumulative('_safe_moves'))
    # _safe_moves includes nested _winning_moves: report guard estimate separately.
    parts['root_tactics'] = sum(c['cumulative_seconds'] for f in functions
        if f['function'] in ('_winning_moves', '_safe_moves')
        for c in f['callers'] if c['function'] == 'choose_move')
    parts['other'] = profile['total_seconds'] - sum(parts.values())
    return parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=7)
    args = parser.parse_args()
    if args.samples < 1:
        parser.error('samples must be positive')
    config = json.loads((args.directory / 'experiment.json').read_text())
    if config['source_hashes'] != source_hashes():
        raise ValueError('Source drift')
    positions = [dict(id=name, history=POSITIONS[name]) for name in ('empty', 'near-opening', 'late')]
    positions += config['profile_openings']
    output = args.directory / 'profiling.json'
    if output.exists():
        raise ValueError('Refusing to overwrite profiling evidence')
    result = dict(experiment_sha256=digest(config), samples=args.samples,
                  warmups_per_condition=1, budgets=[400, 2000, 5000, 10000],
                  positions=positions, rows=[])
    atomic_json(output, result)
    for index, opening in enumerate(positions):
        game = position(opening['history'])
        budgets = result['budgets']
        samples = {b: [] for b in budgets}
        for repetition in range(args.samples + 1):
            offset = repetition % len(budgets)
            for budget in budgets[offset:] + budgets[:offset]:
                seed = seed_for(config['master_seed'], 'timing', index, repetition)
                agent = MCTSAgent(budget, rng=random.Random(seed))
                wall, cpu = time.perf_counter(), time.process_time()
                move = agent.choose_move(game)
                cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
                if move not in game.get_valid_moves():
                    raise RuntimeError('Illegal profile move')
                if repetition:
                    samples[budget].append(dict(seed=seed, move=move, wall_seconds=wall, cpu_seconds=cpu))
        seed = seed_for(config['master_seed'], 'diagnostic', index)
        for budget in budgets:
            memory = memory_run(game, budget, seed)
            if memory['executed_simulations'] not in (0, budget):
                raise RuntimeError('Simulation count mismatch')
            profile = profile_run(MCTSAgent, game, budget, seed)
            wall = statistics.median(s['wall_seconds'] for s in samples[budget])
            result['rows'].append(dict(position=opening['id'], budget=budget,
                diagnostic_seed=seed, samples=samples[budget], median_wall_seconds=wall,
                median_cpu_seconds=statistics.median(s['cpu_seconds'] for s in samples[budget]),
                simulations_per_second=memory['executed_simulations'] / wall,
                memory=memory, profile=profile, categories_seconds=categories(profile)))
        atomic_json(output, result)
        print(f'profile {opening["id"]}: complete', flush=True)


if __name__ == '__main__':
    main()
