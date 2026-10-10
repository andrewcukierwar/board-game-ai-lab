"""Resumable paired-opening studies for the MCTS v3 strength research.

    python -m scripts.mcts_v3.harness openings
    python -m scripts.mcts_v3.harness declare --study pilot1
    python -m scripts.mcts_v3.harness run --study pilot1 --max-seconds 600
    python -m scripts.mcts_v3.analysis --study pilot1

Local and offline: no API, no production configuration, no agent source is
modified. Games are serial; run timed studies under the shared benchmark lock
(scripts/mcts_v3/with_benchmark_lock.sh). See docs/search-mcts-v3/DESIGN.md.
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
from games.connect4.agents.mcts_research_agent import ResearchConfig, ResearchMCTSAgent
from games.connect4.agents.negamax_agent import NegamaxAgent
from games.connect4.connect4 import Connect4
from scripts.evaluate_mcts_strength import (
    append_json, atomic_json, digest, load_rows, seed_for,
)

ROOT = Path('docs/search-mcts-v3')
MASTER_SEED = 2026101003
LENGTHS = (2, 5, 8, 11, 14, 17, 20, 23)
SETS = (('dev', 96), ('holdout', 256), ('preflight', 8))
EMPTY_PAIRS = 64
FAMILY = 8
SOURCE_FILES = (
    'games/connect4/agents/mcts_agent.py',
    'games/connect4/agents/mcts_bitboard.py',
    'games/connect4/agents/mcts_research_agent.py',
    'games/connect4/agents/negamax_agent.py',
    'games/connect4/agents/negamax_tt.py',
    'games/connect4/connect4.py', 'games/connect4/board.py',
    'scripts/mcts_v3/harness.py', 'scripts/evaluate_mcts_strength.py',
)


def position(history):
    game = Connect4()
    for col in history:
        if game.is_game_over() or not game.make_move(col):
            raise ValueError('Illegal opening history')
    return game


def immediate_wins(game, player):
    """Columns winning at once for ``player``, by the array engine only."""
    columns = []
    for col in game.get_valid_moves():
        trial = Connect4([row[:] for row in game.board], player)
        trial.make_move(col)
        if trial.check_winner() == player:
            columns.append(col)
    return columns


def decided_by_root_guards(game):
    """True when every agent's shared root guards fix the result in two plies."""
    mover = game.current_player
    return bool(immediate_wins(game, mover)) or len(immediate_wins(game, 1 - mover)) > 1


def generate_openings():
    """Deterministic, mutually disjoint development / held-out / preflight sets."""
    seen, sets = set(), {}
    for domain, count in SETS:
        rows = []
        for index in range(count):
            length = LENGTHS[index % len(LENGTHS)]
            seed = seed_for(MASTER_SEED, domain, index)
            rng = random.Random(seed)
            for attempt in range(1_000_000):
                game, history = Connect4(), []
                while len(history) < length and not game.is_game_over():
                    col = rng.choice(game.get_valid_moves())
                    game.make_move(col)
                    history.append(col)
                board = tuple(map(tuple, game.board))
                if (len(history) == length and not game.is_game_over()
                        and board not in seen and not decided_by_root_guards(game)):
                    break
            else:
                raise ValueError('Cannot generate enough openings')
            seen.add(board)
            rows.append(dict(id=f'{domain}-{index:03d}', history=history, length=length,
                             generator_seed=seed, rejected_attempts=attempt,
                             challenger_seed=seed_for(MASTER_SEED, domain, index, 'challenger'),
                             opponent_seed=seed_for(MASTER_SEED, domain, index, 'opponent')))
        sets[domain] = rows
    sets['empty'] = [dict(id=f'empty-{index:03d}', history=[], length=0,
                          challenger_seed=seed_for(MASTER_SEED, 'empty', index, 'challenger'),
                          opponent_seed=seed_for(MASTER_SEED, 'empty', index, 'opponent'))
                     for index in range(EMPTY_PAIRS)]
    return dict(schema=1, master_seed=MASTER_SEED, lengths=LENGTHS,
                rejection='terminal, duplicate board in any set, or decided by shared root guards',
                sets=sets)


def source_hashes():
    return {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in SOURCE_FILES}


def hardware():
    if platform.system() != 'Darwin':
        return [platform.processor()]
    text = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], text=True)
    return [line.strip() for line in text.splitlines() if any(label in line for label in (
        'Model Name:', 'Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))]


def mcts(simulations, **config):
    """Agent spec: production baseline, or a research variant when configured."""
    if not config:
        return dict(type='mcts', simulations=simulations)
    return dict(type='research', simulations=simulations, config=config)


def negamax(depth):
    return dict(type='negamax', depth=depth)


def matchup(name, challenger, opponent, pairs=None, primary=False):
    return dict(id=name, challenger=challenger, opponent=opponent, pairs=pairs, primary=primary)


def make_agent(spec, seed):
    if spec['type'] == 'mcts':
        return MCTSAgent(spec['simulations'], rng=random.Random(seed))
    if spec['type'] == 'research':
        return ResearchMCTSAgent(spec['simulations'], rng=random.Random(seed),
                                 config=ResearchConfig(**spec['config']))
    return NegamaxAgent(spec['depth'])


def declare(name, opening_set, matchups, cap_seconds, note=''):
    directory = ROOT / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'study.json').exists():
        raise ValueError('Study already declared; use run to resume')
    openings = json.loads((ROOT / 'openings.json').read_text())
    if openings != json.loads(json.dumps(generate_openings())):
        raise ValueError('openings.json does not match its generator')
    if len({m['id'] for m in matchups}) != len(matchups):
        raise ValueError('Duplicate matchup id')
    rows = openings['sets'][opening_set]
    for m in matchups:
        m['pairs'] = len(rows) if m['pairs'] is None else m['pairs']
        if not 0 < m['pairs'] <= len(rows):
            raise ValueError('Invalid pair count')
    config = dict(schema=1, name=name, note=note, opening_set=opening_set,
                  master_seed=MASTER_SEED, cap_seconds=cap_seconds,
                  source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  source_hashes=source_hashes(), python=platform.python_version(),
                  platform=platform.platform(), hardware=hardware(),
                  pairing='role seeds follow the agent across colours; fresh persistent RNG per game',
                  analysis=dict(bootstrap_seed=seed_for(MASTER_SEED, name, 'bootstrap'),
                                replicates=10000, family=FAMILY),
                  matchups=matchups, openings=rows)
    atomic_json(directory / 'study.json', config)
    return config


def schedule(config):
    """Interleave conditions within each opening; alternate the first colour."""
    matchups = config['matchups']
    for index, opening in enumerate(config['openings']):
        offset = index % len(matchups)
        for m in matchups[offset:] + matchups[:offset]:
            if index < m['pairs']:
                for color in (index % 2, 1 - index % 2):
                    yield m, opening, color


def play_game(config, m, opening, color):
    game = position(opening['history'])
    seeds = dict(challenger=opening['challenger_seed'], opponent=opening['opponent_seed'])
    roles = {color: 'challenger', 1 - color: 'opponent'}
    agents = {c: make_agent(m[role], seeds[role]) for c, role in roles.items()}
    row = dict(id=f'{m["id"]}/{opening["id"]}/{color}', study_sha256=digest(config),
               matchup=m['id'], opening=opening['id'], challenger_color=color,
               moves=[], wall_us=[], cpu_us=[], simulations=[], status='incomplete')
    start = time.perf_counter()
    try:
        while not game.is_game_over():
            mover = game.current_player
            agent = agents[mover]
            # Baseline guard accounting stays outside the timed production call.
            simulations = None
            if type(agent) is MCTSAgent:
                simulations = 0 if agent._winning_moves(game) else agent.simulation_limit
            before = ([r[:] for r in game.board], game.current_player)
            wall, cpu = time.perf_counter(), time.process_time()
            move = agent.choose_move(game)
            cpu, wall = time.process_time() - cpu, time.perf_counter() - wall
            if before != (game.board, game.current_player):
                raise RuntimeError('Agent mutated caller')
            if type(move) is not int or move not in game.get_valid_moves() or not game.make_move(move):
                raise RuntimeError('Agent returned an illegal move')
            if isinstance(agent, ResearchMCTSAgent):
                simulations = agent.last_stats['simulations']
            row['moves'].append(move)
            row['wall_us'].append(round(wall * 1e6))
            row['cpu_us'].append(round(cpu * 1e6))
            row['simulations'].append(simulations)
        winner = game.check_winner()
        row.update(status='complete', winner=winner,
                   challenger_score=0.5 if winner == -1 else float(winner == color))
        row['final_rng'] = {roles[c]: digest(agent.rng.getstate())[:16]
                            for c, agent in agents.items() if hasattr(agent, 'rng')}
    except (Exception, KeyboardInterrupt) as error:
        row['error'] = f'{type(error).__name__}: {error}'
    row['elapsed_seconds'] = time.perf_counter() - start
    return row


def validate_rows(config, rows):
    plans = {f'{m["id"]}/{o["id"]}/{c}': (m, o, c) for m, o, c in schedule(config)}
    seen = set()
    for row in rows:
        if row['id'] in seen or row['id'] not in plans or row['study_sha256'] != digest(config):
            raise ValueError('Duplicate, unknown, or foreign result')
        seen.add(row['id'])
        m, opening, color = plans[row['id']]
        if (row['matchup'], row['opening'], row['challenger_color']) != (m['id'], opening['id'], color):
            raise ValueError('Result plan mismatch')
        game = position(opening['history'])
        for move in row['moves']:
            if game.is_game_over() or not game.make_move(move):
                raise ValueError('Invalid recorded move history')
        if not (len(row['moves']) == len(row['wall_us']) == len(row['simulations'])):
            raise ValueError('Inconsistent per-move records')
        if row['status'] != 'complete' or not game.is_game_over() or game.check_winner() != row['winner']:
            raise ValueError('Nonterminal or incorrect recorded result')
        if row['challenger_score'] != (0.5 if row['winner'] == -1 else float(row['winner'] == color)):
            raise ValueError('Incorrect score')
    return seen


def run(name, max_seconds=None, max_games=None):
    directory = ROOT / name
    config = json.loads((directory / 'study.json').read_text())
    if config['source_hashes'] != source_hashes() or config['python'] != platform.python_version():
        raise ValueError('Source/runtime drift: declare a new study')
    path = directory / 'results.jsonl'
    rows = load_rows(path)
    done = validate_rows(config, rows)
    attempts = load_rows(directory / 'attempts.jsonl')
    elapsed = sum(r['elapsed_seconds'] for r in rows + attempts)
    planned = len(list(schedule(config)))
    completed, start, reason = 0, time.perf_counter(), 'complete'
    load_before = os.getloadavg()
    for m, opening, color in schedule(config):
        if f'{m["id"]}/{opening["id"]}/{color}' in done:
            continue
        spent = time.perf_counter() - start
        if elapsed + spent >= config['cap_seconds']:
            reason = 'fixed compute cap'
            break
        if (max_seconds is not None and spent >= max_seconds) or (
                max_games is not None and completed >= max_games):
            reason = 'batch limit'
            break
        row = play_game(config, m, opening, color)
        if row['status'] != 'complete':
            append_json(directory / 'attempts.jsonl', row)
            reason = row['error']
            break  # Never turn an agent failure into a loss or retry forever.
        append_json(path, row)
        completed += 1
    summary = dict(status=reason, completed=len(done) + completed, planned=planned,
                   games_this_invocation=completed,
                   seconds_this_invocation=time.perf_counter() - start,
                   cumulative_game_seconds=elapsed + time.perf_counter() - start,
                   cap_seconds=config['cap_seconds'],
                   load_average_before=load_before, load_average_after=os.getloadavg(),
                   finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    append_json(directory / 'run-log.jsonl', summary)
    print(json.dumps(summary), flush=True)
    return summary


def main():
    from scripts.mcts_v3 import studies
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['openings', 'declare', 'run'])
    parser.add_argument('--study')
    parser.add_argument('--max-seconds', type=float)
    parser.add_argument('--max-games', type=int)
    args = parser.parse_args()
    if args.command == 'openings':
        path = ROOT / 'openings.json'
        if path.exists():
            raise ValueError('openings.json already exists')
        atomic_json(path, generate_openings())
    elif args.command == 'declare':
        declare(args.study, **studies.STUDIES[args.study]())
    else:
        run(args.study, args.max_seconds, args.max_games)


if __name__ == '__main__':
    main()
