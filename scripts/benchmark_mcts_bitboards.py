"""Provider-free, interleaved MCTS ablations; original source is read from Git.

Run with ``python -m scripts.benchmark_mcts_bitboards --output /tmp/mcts.json``.
Timing excludes fixtures, instrumentation, profiling, and memory tracing.
"""
import argparse
import cProfile
import gc
import hashlib
import json
import platform
import pstats
import random
import statistics
import subprocess
import time
import tracemalloc
from pathlib import Path

from scripts.benchmark_public_agents import POSITIONS, position

BASELINE_REF = '81aeb764eb3ab29d6b26ecd02a0378f7c5c9c0d9'
SUPPLEMENTAL_SEED = 20261010


def baseline_source(ref=BASELINE_REF):
    return subprocess.check_output(
        ['git', 'show', f'{ref}:games/connect4/agents/mcts_agent.py'], text=True)


def variants(ref=BASELINE_REF):
    namespace = {'__name__': 'original_mcts'}
    exec(compile(baseline_source(ref), 'original_mcts.py', 'exec'), namespace)
    original = namespace['MCTSAgent']
    from games.connect4.agents.mcts_agent import MCTSAgent

    class RolloutsOnly(original):
        _simulate = MCTSAgent._simulate

    class TreeOnly(MCTSAgent):
        def _simulate(self, state):
            from games.connect4.connect4 import Connect4
            # Conversion is part of this ablation's cost, not hidden setup.
            game = Connect4(state.board, state.current_player)
            return original._simulate(self, game)

    return {'original': original, 'rollouts': RolloutsOnly,
            'tree': TreeOnly, 'combined': MCTSAgent}


def supplemental_positions():
    """Two histories at each depth, rejection only for terminal positions.

    Predeclared before optimization; no filtering by latency or search result.
    """
    rng = random.Random(SUPPLEMENTAL_SEED)
    result = {}
    for depth in (8, 12, 16, 20):
        for index in range(2):
            while True:
                game = position([])
                moves = []
                for _ in range(depth):
                    if game.is_game_over():
                        break
                    move = rng.choice(game.get_valid_moves())
                    game.make_move(move)
                    moves.append(move)
                if len(moves) == depth and not game.is_game_over():
                    result[f'seeded-{depth:02d}-{index}'] = moves
                    break
    return result


def memory_run(cls, game, budget, seed):
    class Capture(cls):
        root = None

        def _backpropagate(self, node, winner):
            if self.root is None:
                root = node
                while root.parent is not None:
                    root = root.parent
                self.root = root
            super()._backpropagate(node, winner)

    gc.collect()
    tracemalloc.start()
    agent = Capture(budget, rng=random.Random(seed))
    move = agent.choose_move(game)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    root = agent.root
    nodes, fingerprint = 0, []
    pending = [((), root)] if root else []
    while pending:
        path, node = pending.pop()
        nodes += 1
        fingerprint.append((path, node.visits, node.wins, node.player_just_moved,
                            tuple(node.untried_moves), tuple(tuple(row) for row in node.game_state.board)))
        pending.extend(((*path, move), child) for move, child in node.children.items())
    return dict(retained_bytes=retained, peak_bytes=peak, tree_nodes=nodes,
                simulations=root.visits if root else 0, move=move,
                tree_sha256=hashlib.sha256(repr(fingerprint).encode()).hexdigest(),
                rng_sha256=hashlib.sha256(repr(agent.rng.getstate()).encode()).hexdigest())


def profile_run(cls, game, budget, seed):
    profiler = cProfile.Profile()
    profiler.runcall(cls(budget, rng=random.Random(seed)).choose_move, game)
    stats = pstats.Stats(profiler)
    functions = []
    for (file, line, function), (primitive, calls, own, cumulative, callers) in stats.stats.items():
        functions.append(dict(file=Path(file).name, line=line, function=function,
                              calls=calls, primitive_calls=primitive,
                              self_seconds=own, cumulative_seconds=cumulative,
                              callers=[dict(file=Path(caller[0]).name, line=caller[1], function=caller[2],
                                            primitive_calls=values[0], calls=values[1],
                                            self_seconds=values[2], cumulative_seconds=values[3])
                                       for caller, values in callers.items()]))
    functions.sort(key=lambda row: row['cumulative_seconds'], reverse=True)
    return dict(total_seconds=stats.total_tt, functions=functions)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--baseline-ref', default=BASELINE_REF)
    parser.add_argument('--variants', nargs='+', default=['original', 'rollouts', 'tree', 'combined'])
    parser.add_argument('--budgets', nargs='+', type=int, default=[100, 400, 800, 1000])
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--positions', nargs='+')
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    if args.samples < 1 or any(b < 1 for b in args.budgets):
        parser.error('samples and budgets must be positive')
    classes = variants(args.baseline_ref)
    histories = {**POSITIONS, **supplemental_positions()}
    if args.positions:
        histories = {name: histories[name] for name in args.positions}
    hardware = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], text=True) if platform.system() == 'Darwin' else platform.processor()
    hardware = [line.strip() for line in hardware.splitlines() if any(label in line for label in
                ('Model Name:', 'Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))]
    files = ['games/connect4/agents/mcts_agent.py', 'games/connect4/agents/mcts_bitboard.py',
             'scripts/benchmark_mcts_bitboards.py']
    result = dict(baseline_ref=args.baseline_ref,
                  baseline_sha256=hashlib.sha256(baseline_source(args.baseline_ref).encode()).hexdigest(),
                  runtime=platform.python_version(), platform=platform.platform(), hardware=hardware,
                  supplemental_seed=SUPPLEMENTAL_SEED, positions=histories,
                  samples=args.samples, warmups_per_row=1, seeds=[701 + i for i in range(args.samples)],
                  source_hashes={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files if Path(f).exists()},
                  rows=[], profiles={})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    save()  # Declare the full set before collecting any measurements.
    for name, history in histories.items():
        game = position(history)
        before = [row[:] for row in game.board], game.current_player, game.piece
        for budget in args.budgets:
            rows = {v: dict(variant=v, position=name, budget=budget, wall_seconds=[],
                            cpu_seconds=[], moves=[]) for v in args.variants}
            for repetition in range(args.samples + 1):
                order = args.variants[repetition % len(args.variants):] + args.variants[:repetition % len(args.variants)]
                for variant in order:
                    agent = classes[variant](budget, rng=random.Random(700 + repetition))
                    wall, cpu = time.perf_counter(), time.process_time()
                    move = agent.choose_move(game)
                    cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
                    assert move in game.get_valid_moves()
                    assert (game.board, game.current_player, game.piece) == before
                    if repetition:
                        rows[variant]['wall_seconds'].append(wall)
                        rows[variant]['cpu_seconds'].append(cpu)
                        rows[variant]['moves'].append(move)
            for variant, row in rows.items():
                row['memory'] = memory_run(classes[variant], game, budget, 701)
                count = row['memory']['simulations']
                assert count in (0, budget)
                row['median_wall_seconds'] = statistics.median(row['wall_seconds'])
                row['median_cpu_seconds'] = statistics.median(row['cpu_seconds'])
                row['simulations_per_second'] = count / row['median_wall_seconds']
                row['tactical_bypass'] = count == 0
                result['rows'].append(row)
                print(f'{variant} {name} {budget}: {row["median_wall_seconds"]*1000:.3f} ms, moves={row["moves"]}, sims={count}', flush=True)
            # Policy/order equivalence is a measured assertion, not assumed.
            assert len({tuple(row['moves']) for row in rows.values()}) == 1
            assert len({(row['memory']['tree_nodes'], row['memory']['simulations'], row['memory']['move']) for row in rows.values()}) == 1
            assert len({row['memory']['tree_sha256'] for row in rows.values()}) == 1
            assert len({row['memory']['rng_sha256'] for row in rows.values()}) == 1
            save()
    if args.profile:
        for variant in args.variants:
            result['profiles'][variant] = profile_run(classes[variant], position(POSITIONS['near-opening']), 1000, 9102)
        save()


if __name__ == '__main__':
    main()
