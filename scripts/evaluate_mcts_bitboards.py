"""Small paired/color-balanced MCTS evaluation; no production or provider IO.

Equal-budget comparison is a behavioral control. Increased-budget play explores
throughput headroom, not a statistically powered strength estimate.
"""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

from scripts.benchmark_mcts_bitboards import (
    BASELINE_REF, baseline_source, supplemental_positions, variants,
)
from scripts.benchmark_public_agents import POSITIONS, position


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--original-budget', type=int, default=400)
    parser.add_argument('--larger-budget', type=int, default=10000)
    args = parser.parse_args()
    if min(args.original_budget, args.larger_budget) < 1:
        parser.error('budgets must be positive')
    classes = variants()
    histories = supplemental_positions()
    result = dict(baseline_ref=BASELINE_REF,
                  baseline_sha256=hashlib.sha256(baseline_source().encode()).hexdigest(),
                  source_hashes={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                                ['games/connect4/agents/mcts_agent.py', 'games/connect4/agents/mcts_bitboard.py',
                                 'scripts/evaluate_mcts_bitboards.py']},
                  openings=histories, paired_seed_base=950000,
                  original_budget=args.original_budget, larger_budget=args.larger_budget,
                  latency=[], games=[])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save():
        output.write_text(json.dumps(result, indent=2) + '\n')
    save()
    # Matched, interleaved fresh-seed timings at the actual larger budget.
    for name in ('empty', 'near-opening', 'midgame-wide', 'late'):
        game = position(POSITIONS[name])
        rows = {v: dict(position=name, variant=v, budget=b, wall_seconds=[], cpu_seconds=[], moves=[])
                for v, b in [('original', args.original_budget), ('combined', args.larger_budget)]}
        for repetition in range(4):
            order = ['original', 'combined'] if repetition % 2 == 0 else ['combined', 'original']
            for variant in order:
                row = rows[variant]
                agent = classes[variant](row['budget'], rng=random.Random(960000 + repetition))
                wall, cpu = time.perf_counter(), time.process_time()
                move = agent.choose_move(game)
                cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
                if repetition:
                    row['wall_seconds'].append(wall)
                    row['cpu_seconds'].append(cpu)
                    row['moves'].append(move)
        for row in rows.values():
            row['median_wall_seconds'] = statistics.median(row['wall_seconds'])
            row['median_cpu_seconds'] = statistics.median(row['cpu_seconds'])
            result['latency'].append(row)
        print(f'latency {name}: {[(v, round(r["median_wall_seconds"]*1000, 2)) for v, r in rows.items()]}', flush=True)
        save()
    for optimized_budget in (args.original_budget, args.larger_budget):
        for index, (name, history) in enumerate(histories.items()):
            original_seed, optimized_seed = 950000 + 2 * index, 950001 + 2 * index
            for optimized_player in (0, 1):
                game = position(history)
                agents = {
                    optimized_player: classes['combined'](optimized_budget, rng=random.Random(optimized_seed)),
                    1 - optimized_player: classes['original'](args.original_budget, rng=random.Random(original_seed)),
                }
                moves, elapsed = [], []
                while not game.is_game_over():
                    mover = game.current_player
                    wall = time.perf_counter()
                    move = agents[mover].choose_move(game)
                    elapsed.append(time.perf_counter() - wall)
                    assert move in game.get_valid_moves()
                    assert game.make_move(move)
                    moves.append(move)
                winner = game.check_winner()
                score = 0.5 if winner == -1 else float(winner == optimized_player)
                result['games'].append(dict(opening=name, optimized_budget=optimized_budget,
                    optimized_player=optimized_player, original_seed=original_seed, optimized_seed=optimized_seed,
                    moves=moves, wall_seconds=elapsed, winner=winner, optimized_score=score))
                print(f'game {name} optimized={optimized_budget} color={optimized_player}: score={score}', flush=True)
                save()


if __name__ == '__main__':
    main()
