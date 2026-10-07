"""Local, provider-free public search benchmark; run from repository root.

python -m scripts.benchmark_public_agents --output /tmp/public-search.json
No API calls, training artifacts, or retained search tables are involved.
"""
import argparse
import hashlib
import json
import platform
import random
import statistics
import subprocess
import time
from pathlib import Path

from games.connect4.connect4 import Connect4
from games.connect4.agents.negamax_agent import NegamaxAgent
from games.connect4.agents.mcts_agent import MCTSAgent

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]
POSITIONS = {
    'empty': [],
    'near-opening': [3, 2, 4, 3],
    'midgame': [3, 2, 4, 3, 2, 4, 1, 5, 5, 1, 3, 2, 4, 3, 2, 4],
    'midgame-wide': [5, 6, 6, 5, 4, 3, 0, 1, 0, 2, 4, 6, 6, 2, 3, 0],
    'late': DRAW[:30],
}


def position(moves):
    game = Connect4()
    for col in moves:
        assert game.make_move(col)
    assert not game.is_game_over()
    return game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=3)
    parser.add_argument('--positions', nargs='+', choices=list(POSITIONS))
    args = parser.parse_args()
    selected = {name: moves for name, moves in POSITIONS.items() if not args.positions or name in args.positions}
    hardware = subprocess.run(['system_profiler', 'SPHardwareDataType'], capture_output=True, text=True).stdout if platform.system() == 'Darwin' else platform.processor()
    # Serial/UUID identifiers are intentionally excluded from the report.
    hardware = [line.strip() for line in hardware.splitlines()
                if any(label in line for label in ('Model Name:', 'Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))]
    result = dict(runtime=platform.python_version(), platform=platform.platform(), hardware=hardware,
                  positions=selected, samples=args.samples, warmups_per_row=1,
                  source_hashes={name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                                 ['games/connect4/agents/negamax_agent.py', 'games/connect4/agents/mcts_agent.py']}, rows=[])
    for kind, settings in [('negamax', [4, 6, 8, 10]), ('mcts', [50, 100, 250, 400, 800, 1000])]:
        for name, moves in selected.items():
            game = position(moves)
            for setting in settings:
                samples, chosen, stats = [], [], []
                for repetition in range(args.samples + 1):
                    agent = NegamaxAgent(setting) if kind == 'negamax' else MCTSAgent(setting, rng=random.Random(700 + repetition))
                    start = time.perf_counter()
                    move = agent.choose_move(game)
                    elapsed = time.perf_counter() - start
                    assert move in game.get_valid_moves()
                    if repetition:
                        samples.append(elapsed)
                        chosen.append(move)
                        if kind == 'negamax':
                            stats.append(agent.last_stats)
                    print(f'{kind} {name} {setting} sample {repetition}: {elapsed:.4f}s column {move}', flush=True)
                row = dict(agent=kind, position=name, budget=setting, seconds=samples,
                           median=statistics.median(samples), maximum=max(samples), moves=chosen, stats=stats)
                result['rows'].append(row)
                Path(args.output).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
