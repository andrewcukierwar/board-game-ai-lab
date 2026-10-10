"""Local Negamax ablations; never writes canonical evidence or calls services.

Run on an otherwise quiet host after checking training/service CPU contention:
python -m scripts.benchmark_negamax_ordering --baseline-ref <commit> \
    --output docs/search-negamax-v2/ordering.json
"""
import argparse
import hashlib
import json
import platform
from random import Random
import statistics
import subprocess
import sys
import time
import types
from pathlib import Path

from games.connect4.agents import negamax_agent as candidate
from scripts.benchmark_public_agents import POSITIONS, position

VARIANTS = {
    'baseline': None,
    'center': (False, 'none'),
    'tt': (True, 'none'),
    'wins': (False, 'wins'),
    'tt-wins': (True, 'wins'),
    'threats': (False, 'threats'),
    'tactical': (False, 'tactical'),
    'combined': (True, 'tactical'),
}


def table_bytes(table):
    """Unique Python object sizes reachable from the TT, not process RSS/peak."""
    seen = set()

    def size(obj):
        if id(obj) in seen:
            return 0
        seen.add(id(obj))
        total = sys.getsizeof(obj)
        if isinstance(obj, dict):
            total += sum(size(k) + size(v) for k, v in obj.items())
        elif isinstance(obj, (list, tuple)):
            total += sum(size(item) for item in obj)
        return total

    return size(vars(table))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--depths', nargs='+', type=int, default=[4, 6, 8, 10])
    parser.add_argument('--variants', nargs='+', choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument('--positions', nargs='+', choices=list(POSITIONS), default=list(POSITIONS))
    parser.add_argument('--seeded-positions', type=int, default=0,
                        help='Additional deterministic legal nonterminal positions (seed 20261009)')
    args = parser.parse_args()
    if args.samples < 1 or args.seeded_positions < 0 or any(depth < 1 for depth in args.depths):
        parser.error('samples and depths must be positive')
    path = 'games/connect4/agents/negamax_agent.py'
    commit = subprocess.check_output(['git', 'rev-parse', args.baseline_ref], text=True).strip()
    source = subprocess.check_output(['git', 'show', f'{commit}:{path}'])
    baseline = types.ModuleType('negamax_baseline')
    exec(compile(source, f'{commit}:{path}', 'exec'), baseline.__dict__)
    factories = {module: module.SearchTable for module in (baseline, candidate)}
    selected = {name: POSITIONS[name] for name in args.positions}
    rng = Random(20261009)
    for sample in range(args.seeded_positions):
        plies = (8, 12, 16, 20)[sample % 4]
        while True:
            game = position([])
            moves = []
            for _ in range(plies):
                move = rng.choice(game.get_valid_moves())
                assert game.make_move(move)
                moves.append(move)
                if game.is_game_over():
                    break
            if not game.is_game_over():
                break
        selected[f'seeded-{sample:02d}'] = moves
    hardware = []
    if platform.system() == 'Darwin':
        info = subprocess.run(['system_profiler', 'SPHardwareDataType'],
                              capture_output=True, text=True).stdout
        hardware = [line.strip() for line in info.splitlines() if any(label in line for label in (
            'Model Name:', 'Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))]
    result = dict(baseline_commit=commit, runtime=platform.python_version(),
                  platform=platform.platform(), hardware=hardware,
                  cpu_count=__import__('os').cpu_count(), seeded_position_seed=20261009,
                  source_hashes={'baseline': hashlib.sha256(source).hexdigest(),
                                 'candidate': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                                 'harness': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                  positions=selected, samples=args.samples, warmups_per_variant_row=1,
                  method='Rotating variant order per repetition; fresh shared TT per decision; '
                         'game construction and table sizing excluded from timings. '
                         'table_bytes is retained unique Python object size, not RSS/peak.', rows=[])
    try:
        for name, moves in selected.items():
            game = position(moves)
            for depth in args.depths:
                rows = {v: dict(variant=v, position=name, depth=depth, samples=[]) for v in args.variants}
                expected = None
                for repetition in range(args.samples + 1):
                    offset = repetition % len(args.variants)
                    order = args.variants[offset:] + args.variants[:offset]
                    for variant in order:
                        module = baseline if variant == 'baseline' else candidate
                        held = []

                        def factory():
                            options = VARIANTS[variant]
                            table = factories[module]() if options is None else factories[module](
                                tt_moves=options[0], tactical=options[1])
                            held.append(table)
                            return table

                        module.SearchTable = factory
                        agent = module.NegamaxAgent(depth)
                        start = time.perf_counter()
                        move = agent.choose_move(game)
                        elapsed = time.perf_counter() - start
                        assert move in game.get_valid_moves()
                        values = (move, agent.last_scores)
                        if expected is None:
                            expected = values
                        assert values == expected, (name, depth, variant, values, expected)
                        if repetition:
                            rows[variant]['samples'].append(dict(seconds=elapsed, move=move,
                                                                 scores=agent.last_scores,
                                                                 stats=agent.last_stats))
                        if repetition == args.samples:
                            rows[variant]['table_bytes'] = table_bytes(held[0])
                        held.clear()
                for row in rows.values():
                    times = [sample['seconds'] for sample in row['samples']]
                    assert all(sample['stats'] == row['samples'][0]['stats'] for sample in row['samples'])
                    row.update(median_seconds=statistics.median(times), maximum_seconds=max(times))
                    result['rows'].append(row)
                    print(f"{name} d{depth} {row['variant']}: {row['median_seconds'] * 1000:.3f} ms "
                          f"{row['samples'][0]['stats']['nodes']} nodes", flush=True)
                Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
    finally:
        for module, factory in factories.items():
            module.SearchTable = factory


if __name__ == '__main__':
    main()
