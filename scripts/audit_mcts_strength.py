"""Replay and audit saved evidence; repeat predetermined games for determinism."""
import argparse
import json
import math
from pathlib import Path

from games.connect4.agents.mcts_agent import MCTSAgent
from scripts.benchmark_public_agents import position
from scripts.evaluate_mcts_strength import (
    atomic_json, digest, load_rows, play_game, schedule, source_hashes, validate_rows,
)


def stable(row):
    result = {k: v for k, v in row.items() if k not in ('elapsed_seconds', 'search_totals', 'moves')}
    result['moves'] = [{k: v for k, v in move.items() if k not in ('wall_seconds', 'cpu_seconds')}
                       for move in row['moves']]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True, type=Path)
    args = parser.parse_args()
    config = json.loads((args.directory / 'experiment.json').read_text())
    if source_hashes() != config['source_hashes']:
        raise ValueError('Source drift')
    rows = load_rows(args.directory / 'results.jsonl')
    preflight = load_rows(args.directory / 'preflight.jsonl')
    validate_rows(config, rows)
    validate_rows(config, preflight, True)
    if len(rows) != len(list(schedule(config))) or len(preflight) != len(list(schedule(config, True))):
        raise ValueError('Incomplete experiment')
    calls = 0
    for row in rows + preflight:
        game = position(row['opening_history'])
        for move in row['moves']:
            agent_config = row[f'{move["role"]}_config']
            expected = None
            if agent_config['type'] == 'mcts':
                agent = MCTSAgent(agent_config['simulations'])
                expected = 0 if agent._winning_moves(game) else agent.simulation_limit
            if move['simulations'] != expected:
                raise ValueError('Incorrect guard/simulation accounting')
            if any(not math.isfinite(move[k]) or move[k] < 0 for k in ('wall_seconds', 'cpu_seconds')):
                raise ValueError('Invalid timing')
            game.make_move(move['column'])
            calls += 1
        for role, totals in row['search_totals'].items():
            moves = [m for m in row['moves'] if m['role'] == role]
            if totals != dict(calls=len(moves), wall_seconds=sum(m['wall_seconds'] for m in moves),
                              cpu_seconds=sum(m['cpu_seconds'] for m in moves),
                              simulations=sum(m['simulations'] or 0 for m in moves)):
                raise ValueError('Incorrect search totals')
    # Fixed first opening, both colors of every condition; never select repeats
    # based on winner, timing, or unusually interesting trajectories.
    saved = {r['id']: r for r in rows}
    repeated = []
    for matchup, opening, color in schedule(config):
        if opening['id'] != config['openings'][0]['id']:
            continue
        fresh = play_game(config, matchup, opening, color)
        if stable(fresh) != stable(saved[fresh['id']]):
            raise ValueError('Deterministic repeat mismatch')
        repeated.append(fresh['id'])
    attempts = {name: len(load_rows(args.directory / name))
                for name in ('preflight-attempts.jsonl', 'results-attempts.jsonl')}
    atomic_json(args.directory / 'audit.json', dict(experiment_sha256=digest(config),
        replayed_complete_games=len(rows) + len(preflight), audited_search_calls=calls,
        deterministic_repeat_ids=repeated, incomplete_attempts=attempts,
        simulation_accounting='inferred from production immediate-win guard; independently captured in profiling',
        status='passed'))


if __name__ == '__main__':
    main()
