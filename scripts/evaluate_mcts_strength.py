"""Deterministic, game-boundary resumable local Connect Four experiments.

No API, production configuration, or agent source is modified. See the report
for the sampling estimand, fixed budget, and opening-cluster inference.
"""
import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

from games.connect4.agents.mcts_agent import MCTSAgent
from games.connect4.agents.negamax_agent import NegamaxAgent
from scripts.benchmark_public_agents import position

BUDGETS = (400, 800, 2000, 5000, 10000)
LENGTHS = (2, 5, 8, 11, 14, 17, 20, 23)
SOURCE_FILES = (
    'games/connect4/agents/mcts_agent.py',
    'games/connect4/agents/mcts_bitboard.py',
    'games/connect4/agents/negamax_agent.py',
    'games/connect4/connect4.py', 'games/connect4/board.py',
    'scripts/evaluate_mcts_strength.py',
    'scripts/analyze_mcts_strength.py', 'scripts/profile_mcts_strength.py',
)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def seed_for(master, *parts):
    """Domain-separated 128-bit seeds; no global or shared mutable RNG."""
    return int(digest([master, *parts])[:32], 16)


def source_hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in SOURCE_FILES}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_json(path, value):
    with path.open('a') as stream:
        stream.write(json.dumps(value, separators=(',', ':')) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def load_rows(path):
    """Fail closed on damaged evidence rather than silently dropping a result."""
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def generate_openings(count, master, domain, lengths=LENGTHS):
    rows, seen = [], set()
    for index in range(count):
        length = lengths[index % len(lengths)]
        seed = seed_for(master, domain, index)
        rng = random.Random(seed)
        for attempt in range(100000):
            game = position([])
            history = []
            for _ in range(length):
                if game.is_game_over():
                    break
                col = rng.choice(game.get_valid_moves())
                if not game.make_move(col):
                    raise RuntimeError('Opening generator returned an illegal move')
                history.append(col)
            if len(history) == length and not game.is_game_over() and tuple(history) not in seen:
                break
        else:
            raise ValueError('Cannot generate enough unique nonterminal histories')
        seen.add(tuple(history))
        heights = [sum(row[col] != ' ' for row in game.board) for col in range(7)]
        rows.append(dict(id=f'{domain}-{index:03d}', history=history,
                         generator_seed=seed, rejected_attempts=attempt,
                         length=length, heights=heights,
                         challenger_seed=seed_for(master, domain, index, 'challenger'),
                         opponent_seed=seed_for(master, domain, index, 'opponent')))
    return rows


def matchups(primary_pairs, secondary_pairs):
    rows = [dict(id=f'mcts-{b}-vs-mcts-400', challenger=dict(type='mcts', simulations=b),
                 opponent=dict(type='mcts', simulations=400), pairs=primary_pairs,
                 primary=True) for b in BUDGETS if b != 400]
    rows += [dict(id=f'mcts-{b}-vs-negamax-{d}', challenger=dict(type='mcts', simulations=b),
                  opponent=dict(type='negamax', depth=d), pairs=secondary_pairs,
                  primary=False) for d in (4, 6) for b in BUDGETS]
    return rows


def declare(directory, *, pairs=128, secondary_pairs=32, preflight_pairs=4,
            master=2026100903, max_seconds=1800):
    if min(pairs, secondary_pairs, preflight_pairs) < 1 or secondary_pairs > pairs:
        raise ValueError('Pair counts must be positive and secondary <= primary')
    if max_seconds <= 0:
        raise ValueError('Compute cap must be positive')
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'experiment.json').exists():
        raise ValueError('Experiment already declared; use run to resume')
    hardware = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], text=True) if platform.system() == 'Darwin' else platform.processor()
    hardware = [line.strip() for line in hardware.splitlines() if any(label in line for label in
                ('Model Name:', 'Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))]
    config = dict(schema=1, master_seed=master, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), source_hashes=source_hashes(),
        python=platform.python_version(), platform=platform.platform(), hardware=hardware,
        pairing='role seeds reused across colors and budgets; independent role RNGs persist within each game',
        sampling='uniform legal-column histories; reject terminal or duplicate histories only',
        max_run_seconds=max_seconds, max_preflight_seconds=300,
        matchups=matchups(pairs, secondary_pairs),
        openings=generate_openings(pairs, master, 'main'),
        preflight_openings=generate_openings(preflight_pairs, master, 'preflight', (2, 8, 17, 23)),
        profile_openings=generate_openings(16, master, 'profile'),
        analysis=dict(bootstrap_seed=seed_for(master, 'bootstrap'), replicates=10000,
                      primary_family=4, interval='stratified opening-pair percentile bootstrap'))
    atomic_json(directory / 'experiment.json', config)
    return config


def make_agent(config, seed):
    if config['type'] == 'mcts':
        return MCTSAgent(config['simulations'], rng=random.Random(seed))
    return NegamaxAgent(config['depth'])


def play_game(config, matchup, opening, color):
    game = position(opening['history'])
    agents = {color: make_agent(matchup['challenger'], opening['challenger_seed']),
              1 - color: make_agent(matchup['opponent'], opening['opponent_seed'])}
    row = dict(id=f'{matchup["id"]}/{opening["id"]}/{color}',
               experiment_sha256=digest(config), source_commit=config['source_commit'],
               matchup=matchup['id'], opening=opening['id'], opening_history=opening['history'],
               challenger_color=color, challenger_config=matchup['challenger'],
               opponent_config=matchup['opponent'], challenger_seed=opening['challenger_seed'],
               opponent_seed=opening['opponent_seed'], moves=[], status='incomplete')
    start = time.perf_counter()
    try:
        while not game.is_game_over():
            mover = game.current_player
            role = 'challenger' if mover == color else 'opponent'
            agent = agents[mover]
            # Root guard accounting is outside the timed production method.
            simulations = (0 if agent._winning_moves(game) else agent.simulation_limit) if isinstance(agent, MCTSAgent) else None
            before = ([r[:] for r in game.board], game.current_player, game.piece)
            wall, cpu = time.perf_counter(), time.process_time()
            move = agent.choose_move(game)
            cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
            if before != (game.board, game.current_player, game.piece):
                raise RuntimeError('Agent mutated caller')
            if type(move) is not int or move not in game.get_valid_moves() or not game.make_move(move):
                raise RuntimeError('Agent returned an illegal move')
            row['moves'].append(dict(ply=len(opening['history']) + len(row['moves']),
                                     color=mover, role=role, column=move,
                                     wall_seconds=wall, cpu_seconds=cpu, simulations=simulations))
        winner = game.check_winner()
        row.update(status='complete', winner=winner,
                   challenger_score=0.5 if winner == -1 else float(winner == color))
        row['final_rng_hashes'] = {role: digest(agents[c].rng.getstate())
                                  for role, c in [('challenger', color), ('opponent', 1 - color)]
                                  if isinstance(agents[c], MCTSAgent)}
    except (Exception, KeyboardInterrupt) as error:
        row['error'] = f'{type(error).__name__}: {error}'
    row['elapsed_seconds'] = time.perf_counter() - start
    row['complete_history'] = opening['history'] + [m['column'] for m in row['moves']]
    row['search_totals'] = {role: dict(calls=sum(m['role'] == role for m in row['moves']),
        wall_seconds=sum(m['wall_seconds'] for m in row['moves'] if m['role'] == role),
        cpu_seconds=sum(m['cpu_seconds'] for m in row['moves'] if m['role'] == role),
        simulations=sum(m['simulations'] or 0 for m in row['moves'] if m['role'] == role))
        for role in ('challenger', 'opponent')}
    return row


def schedule(config, preflight=False):
    openings = config['preflight_openings' if preflight else 'openings']
    # Interleave conditions by opening; alternate first color and rotate matchup
    # order to spread host/thermal drift without changing game RNG streams.
    for index, opening in enumerate(openings):
        matches = config['matchups']
        offset = index % len(matches)
        for matchup in matches[offset:] + matches[:offset]:
            if not preflight and index >= matchup['pairs']:
                continue
            for color in (index % 2, 1 - index % 2):
                yield matchup, opening, color


def validate_rows(config, rows, preflight=False):
    plans = {f'{m["id"]}/{o["id"]}/{c}': (m, o, c) for m, o, c in schedule(config, preflight)}
    seen = set()
    for row in rows:
        if row['id'] in seen or row['id'] not in plans or row['experiment_sha256'] != digest(config):
            raise ValueError('Duplicate, unknown, or foreign result')
        seen.add(row['id'])
        matchup, opening, color = plans[row['id']]
        for key, expected in dict(matchup=matchup['id'], opening=opening['id'],
                opening_history=opening['history'], challenger_color=color,
                challenger_config=matchup['challenger'], opponent_config=matchup['opponent'],
                challenger_seed=opening['challenger_seed'], opponent_seed=opening['opponent_seed'],
                source_commit=config['source_commit']).items():
            if row[key] != expected:
                raise ValueError(f'Result plan mismatch: {key}')
        game = position(opening['history'])
        for index, move in enumerate(row['moves']):
            if (game.is_game_over() or move['color'] != game.current_player
                    or move['role'] != ('challenger' if game.current_player == color else 'opponent')
                    or move['ply'] != len(opening['history']) + index
                    or not game.make_move(move['column'])):
                raise ValueError('Invalid recorded move history')
        if row['complete_history'] != opening['history'] + [m['column'] for m in row['moves']]:
            raise ValueError('Inconsistent complete history')
        if row['status'] != 'complete' or not game.is_game_over() or game.check_winner() != row['winner']:
            raise ValueError('Nonterminal or incorrect recorded result')
        score = .5 if row['winner'] == -1 else float(row['winner'] == color)
        if row['challenger_score'] != score:
            raise ValueError('Incorrect score')
    return seen


def run(directory, preflight=False, max_games=None):
    config = json.loads((directory / 'experiment.json').read_text())
    if config['source_hashes'] != source_hashes() or config['python'] != platform.python_version():
        raise ValueError('Source/runtime drift: start a new experiment')
    prefix = 'preflight' if preflight else 'results'
    path = directory / f'{prefix}.jsonl'
    rows = load_rows(path)
    done = validate_rows(config, rows, preflight)
    attempts = load_rows(directory / f'{prefix}-attempts.jsonl')
    elapsed = sum(r['elapsed_seconds'] for r in rows + attempts)
    status_path = directory / f'{prefix}-status.json'
    if status_path.exists():
        elapsed = max(elapsed, json.loads(status_path.read_text())['cumulative_budget_seconds'])
    cap = config['max_preflight_seconds' if preflight else 'max_run_seconds']
    if not preflight:
        preflight_rows = load_rows(directory / 'preflight.jsonl')
        validate_rows(config, preflight_rows, True)
        if len(preflight_rows) != len(list(schedule(config, True))):
            raise ValueError('Complete the preflight before the main run')
    completed_now, start = 0, time.perf_counter()
    reason = 'complete'
    for matchup, opening, color in schedule(config, preflight):
        game_id = f'{matchup["id"]}/{opening["id"]}/{color}'
        if game_id in done:
            continue
        if elapsed + time.perf_counter() - start >= cap:
            reason = 'fixed compute cap'
            break
        if max_games is not None and completed_now >= max_games:
            reason = 'checkpoint requested'
            break
        row = play_game(config, matchup, opening, color)
        if row['status'] != 'complete':
            append_json(directory / f'{prefix}-attempts.jsonl', row)
            reason = row['error']
            break  # Never turn agent failures into losses or retry forever.
        append_json(path, row)
        completed_now += 1
        if completed_now % 32 == 0:
            print(f'{prefix}: {len(done) + completed_now} games; {time.perf_counter() - start:.1f}s this invocation', flush=True)
    summary = dict(status=reason, completed=len(done) + completed_now,
                   planned=len(list(schedule(config, preflight))),
                   cumulative_budget_seconds=elapsed + time.perf_counter() - start,
                   cap_seconds=cap)
    if preflight and reason == 'complete':
        measured = load_rows(path)
        summary['estimated_full_game_seconds'] = sum(
            sum(r['elapsed_seconds'] for r in measured if r['matchup'] == m['id'])
            / (2 * len(config['preflight_openings'])) * 2 * m['pairs'] for m in config['matchups'])
    atomic_json(directory / f'{prefix}-status.json', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['declare', 'preflight', 'run'])
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--pairs', type=int, default=128)
    parser.add_argument('--secondary-pairs', type=int, default=32)
    parser.add_argument('--preflight-pairs', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026100903)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--max-games', type=int)
    args = parser.parse_args()
    if args.max_games is not None and args.max_games < 1:
        parser.error('max-games must be positive')
    if args.command == 'declare':
        declare(args.directory, pairs=args.pairs, secondary_pairs=args.secondary_pairs,
                preflight_pairs=args.preflight_pairs, master=args.seed, max_seconds=args.max_seconds)
    else:
        run(args.directory, args.command == 'preflight', args.max_games)


if __name__ == '__main__':
    main()
