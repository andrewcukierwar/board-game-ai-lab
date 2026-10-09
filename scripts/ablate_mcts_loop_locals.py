"""Measure a local-binding prototype without modifying production agents.

Acceptance is predeclared: >=5% aggregate median latency reduction, improvement
in >=75% of conditions, no >5% regression, and identical trees/RNG at equal
budgets. Smaller/noisy changes do not justify editing optimized agent code.
"""
import argparse
import hashlib
import inspect
import json
import random
import statistics
import textwrap
import time
from pathlib import Path

from games.connect4.agents import mcts_agent, mcts_bitboard
from scripts.benchmark_public_agents import position
from scripts.evaluate_mcts_strength import atomic_json, digest, source_hashes
from scripts.profile_mcts_strength import Capture


def prototype():
    rollout_source = inspect.getsource(mcts_bitboard.random_rollout)
    rollout_source = rollout_source.replace('    while legal:',
        '    bottom, column, top, four = BOTTOM, COLUMN, TOP, has_four\n    while legal:')
    rollout_source = rollout_source.replace('BOTTOM[col]', 'bottom[col]').replace(
        'COLUMN[col]', 'column[col]').replace('TOP[col]', 'top[col]').replace('has_four(own)', 'four(own)')
    rollout_namespace = dict(vars(mcts_bitboard))
    exec(compile(rollout_source, 'prototype_rollout.py', 'exec'), rollout_namespace)
    select_source = textwrap.dedent(inspect.getsource(mcts_agent.MCTSAgent._select_child))
    select_source = select_source.replace('    log_visits =',
        '    sqrt, choice = math.sqrt, self.rng.choice\n    log_visits =')
    select_source = select_source.replace('math.sqrt(log_visits', 'sqrt(log_visits').replace(
        'self.rng.choice(ties)', 'choice(ties)')
    namespace = dict(vars(mcts_agent))
    exec(compile(select_source, 'prototype_selection.py', 'exec'), namespace)
    simulate_source = textwrap.dedent(inspect.getsource(mcts_agent.MCTSAgent._simulate))
    namespace['random_rollout'] = rollout_namespace['random_rollout']
    exec(compile(simulate_source, 'prototype_simulation.py', 'exec'), namespace)
    class Prototype(mcts_agent.MCTSAgent):
        _select_child = namespace['_select_child']
        _simulate = namespace['_simulate']
    return Prototype, dict(selection=select_source, rollout=rollout_source, simulation=simulate_source)


def fingerprint(agent):
    pending = [((), agent.root)] if agent.root else []
    tree = []
    while pending:
        path, node = pending.pop()
        state = node.game_state
        tree.append((path, state.pieces, state.occupied, state.current_player, state.winner,
                     node.visits, node.wins, node.player_just_moved, node.untried_moves))
        pending.extend((path + (move,), child) for move, child in node.children.items())
    return digest(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.directory / 'experiment.json').read_text())
    if config['source_hashes'] != source_hashes():
        raise ValueError('Source drift')
    candidate, code = prototype()
    classes = {'baseline': mcts_agent.MCTSAgent, 'local_bindings': candidate}
    positions = json.loads((args.directory / 'profiling.json').read_text())['positions'][:11]
    output = args.directory / 'loop-ablation.json'
    if output.exists():
        raise ValueError('Refusing to overwrite ablation evidence')
    result = dict(experiment_sha256=digest(config), candidate_source=code,
        diagnostic_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        acceptance=dict(minimum_aggregate_gain=.05, minimum_improved_fraction=.75,
                        maximum_condition_regression=.05),
        positions=positions, budgets=[400, 2000, 5000, 10000],
        seeds=list(range(17001, 17008)), diagnostic_seed=17001, rows=[])
    atomic_json(output, result)  # Declare candidate and criteria before observations.
    for opening in positions:
        game = position(opening['history'])
        for budget in result['budgets']:
            samples = {name: [] for name in classes}
            for repetition in range(8):
                names = list(classes) if repetition % 2 else list(reversed(classes))
                for name in names:
                    agent = classes[name](budget, rng=random.Random(17000 + repetition))
                    start = time.perf_counter()
                    move = agent.choose_move(game)
                    elapsed = time.perf_counter() - start
                    if repetition:
                        samples[name].append(dict(seed=17000 + repetition, move=move,
                            wall_seconds=elapsed, rng_sha256=digest(agent.rng.getstate())))
            for a, b in zip(samples['baseline'], samples['local_bindings']):
                if (a['move'], a['rng_sha256']) != (b['move'], b['rng_sha256']):
                    raise ValueError('Candidate changed moves/RNG')
            fingerprints = []
            for cls in classes.values():
                class Diagnostic(Capture, cls):
                    _select_child = cls._select_child
                    _simulate = cls._simulate
                agent = Diagnostic(budget, rng=random.Random(17001))
                agent.choose_move(game)
                fingerprints.append(fingerprint(agent))
            if len(set(fingerprints)) != 1:
                raise ValueError('Candidate changed tree')
            medians = {name: statistics.median(s['wall_seconds'] for s in group)
                       for name, group in samples.items()}
            result['rows'].append(dict(position=opening['id'], budget=budget,
                samples=samples, median_seconds=medians, tree_sha256=fingerprints[0],
                speedup=medians['baseline'] / medians['local_bindings']))
        atomic_json(output, result)
    speedups = [r['speedup'] for r in result['rows']]
    result['summary'] = dict(median_speedup=statistics.median(speedups),
        improved_fraction=sum(x > 1 for x in speedups) / len(speedups),
        worst_speedup=min(speedups), best_speedup=max(speedups),
        accepted=(statistics.median(speedups) >= 1 / .95
                  and sum(x > 1 for x in speedups) / len(speedups) >= .75
                  and min(speedups) >= 1 / 1.05))
    atomic_json(output, result)
    print(json.dumps(result['summary']))


if __name__ == '__main__':
    main()
