"""Local Phase 3A: frozen, paired, resumable deeper-search evaluation."""
import argparse
import copy
import hashlib
import json
import math
import platform
import random
import signal
import subprocess
import time
from pathlib import Path

from games.connect4.agents.mcts_agent import MCTSAgent
from games.connect4.agents.negamax_agent import NegamaxAgent, SearchState
from scripts.benchmark_public_agents import POSITIONS, DRAW, position
from scripts.evaluate_mcts_strength import (atomic_json, append_json, digest,
    load_rows, seed_for, make_agent, LENGTHS)

ROOT = Path(__file__).resolve().parents[1]
FILES = ('games/connect4/agents/negamax_agent.py', 'games/connect4/agents/mcts_agent.py',
         'games/connect4/agents/mcts_bitboard.py', 'games/connect4/connect4.py',
         'games/connect4/board.py', 'scripts/evaluate_mcts_strength.py',
         'scripts/benchmark_public_agents.py', 'scripts/benchmark_negamax_ordering.py',
         'scripts/evaluate_negamax_depths.py', 'scripts/analyze_negamax_depths.py',
         'scripts/profile_negamax_depths.py')


def hashes():
    return {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in FILES}


def board_key(game):
    return digest([list(map(list, game.board)), game.current_player])


def labels(history):
    game = position(history)
    guard = MCTSAgent(1, rng=random.Random(0))
    wins, safe = guard._winning_moves(game), guard._safe_moves(game)
    return dict(stage='early' if len(history) <= 8 else 'midgame' if len(history) <= 23 else 'late',
                tactics='tactical' if wins or len(safe) < len(game.get_valid_moves()) else 'quiet',
                immediate_wins=wins, safe_moves=safe, board_sha256=board_key(game))


def openings(master, domain, families, seen):
    """Four independent random boards + four related agent snapshots per block.

    Agent trajectory: two random initial moves, then 80% agent policy / 20%
    uniformly legal exploration; alternating Negamax 4 and seeded MCTS 400.
    Reject entire trajectories only on early termination or duplicate boards.
    """
    rows = []
    for family in range(families):
        random_rows = []
        for j in range(4):
            length = LENGTHS[(family * 4 + j) % len(LENGTHS)]
            seed = seed_for(master, domain, 'random', family, j)
            rng = random.Random(seed)
            for attempt in range(100000):
                game, history = position([]), []
                while len(history) < length and not game.is_game_over():
                    col = rng.choice(game.get_valid_moves())
                    assert game.make_move(col)
                    history.append(col)
                if len(history) == length and not game.is_game_over() and board_key(game) not in seen:
                    break
            else:
                raise RuntimeError('Random generator exhausted')
            seen.add(board_key(game))
            random_rows.append(dict(history=history, cohort='random', generator_seed=seed,
                                    rejected_attempts=attempt, family=f'{domain}-random-{family}-{j}'))
        seed = seed_for(master, domain, 'agent', family)
        rng = random.Random(seed)
        for attempt in range(100000):
            game, history, snapshots = position([]), [], []
            agents = [NegamaxAgent(4), MCTSAgent(400, rng=random.Random(seed_for(seed, attempt, 'mcts')))]
            while len(history) < 32 and not game.is_game_over():
                col = (rng.choice(game.get_valid_moves()) if len(history) < 2 or rng.random() < .2
                       else agents[game.current_player].choose_move(game))
                assert game.make_move(col)
                history.append(col)
                if len(history) in (5, 14, 23, 32) and not game.is_game_over():
                    snapshots.append(history[:])
            keys = [board_key(position(h)) for h in snapshots]
            if len(snapshots) == 4 and len(set(keys)) == 4 and not seen.intersection(keys):
                break
        else:
            raise RuntimeError('Agent generator exhausted')
        seen.update(keys)
        for j, history in enumerate(snapshots):
            for row in (random_rows[j], dict(history=history, cohort='agent', generator_seed=seed,
                    rejected_attempts=attempt, family=f'{domain}-agent-{family}')):
                index = len(rows)
                row.update(id=f'{domain}-{index:03d}', length=len(row['history']),
                           challenger_seed=seed_for(master, domain, index, 'challenger'),
                           opponent_seed=seed_for(master, domain, index, 'opponent'), **labels(row['history']))
                rows.append(row)
    return rows


def matchups(primary=64, secondary=32):
    conditions = [(8, 'negamax', 6), (10, 'negamax', 8), (10, 'negamax', 6),
                  (8, 'mcts', 2000), (10, 'mcts', 2000), (10, 'mcts', 5000)]
    return [dict(id=f'n{d}-vs-{kind[0]}{b}', challenger=dict(type='negamax', depth=d),
                 opponent={'type': kind, 'depth' if kind == 'negamax' else 'simulations': b},
                 primary=i < 2, pairs=primary if i < 2 else secondary)
            for i, (d, kind, b) in enumerate(conditions)]


def declare(directory, master=2026100904):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'candidate.json').exists():
        raise ValueError('Already declared')
    seen = set()
    # Preflight boards cannot overlap main or diagnostic boards.
    preflight = openings(master, 'preflight', 1, seen)
    main = openings(master, 'main', 8, seen)
    profile = [dict(id=k, history=v, **labels(v)) for k, v in POSITIONS.items()]
    profile += [dict(id=k, history=h, **labels(h)) for k, h in (
        ('immediate-win', [0, 1, 0, 1, 0, 2]),
        ('forced-reply', [1, 0, 1, 0, 2, 0]),
        ('double-threat-loss', [1, 0, 1, 0, 2, 0, 2, 6, 4, 6, 4, 6]),
        ('dense-endgame', DRAW[:38]))]
    profile += [dict(id='quiet-'+str(i), history=o['history'], **labels(o['history']))
                for i, o in enumerate(preflight) if o['tactics'] == 'quiet' and o['stage'] == 'midgame']
    hardware = subprocess.check_output(['system_profiler', 'SPHardwareDataType'], text=True) if platform.system() == 'Darwin' else platform.processor()
    config = dict(schema=1, master_seed=master, source_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), source_hashes=hashes(),
        python=platform.python_version(), platform=platform.platform(), hardware=[s.strip() for s in hardware.splitlines()
        if any(x in s for x in ('Model Name:', 'Model Identifier:', 'Chip:', 'Total Number of Cores:', 'Memory:'))],
        max_seconds=1800, preflight_seconds=600, matchups=matchups(), openings=main,
        preflight_openings=preflight, profile_positions=profile,
        analysis=dict(seed=seed_for(master, 'bootstrap'), replicates=10000, primary_family=2,
                      method='cohort-stratified trajectory-cluster percentile bootstrap'))
    atomic_json(directory / 'candidate.json', config)
    return config


def check_source(config):
    if hashes() != config['source_hashes'] or platform.python_version() != config['python']:
        raise ValueError('Source/runtime drift')


def schedule(config, preflight=False):
    for i, opening in enumerate(config['preflight_openings' if preflight else 'openings']):
        matches = config['matchups']
        offset = i % len(matches)
        for matchup in matches[offset:] + matches[:offset]:
            if preflight or i < matchup['pairs']:
                for color in (i % 2, 1 - i % 2):
                    yield matchup, opening, color


def game_id(matchup, opening, color):
    return f'{matchup["id"]}/{opening["id"]}/{color}'


def play(config, matchup, opening, color, checkpoint=None):
    game = position(opening['history'])
    agents = {color: make_agent(matchup['challenger'], opening['challenger_seed']),
              1-color: make_agent(matchup['opponent'], opening['opponent_seed'])}
    row = dict(id=game_id(matchup, opening, color), experiment_sha256=digest(config),
               source_commit=config['source_commit'], matchup=matchup['id'], opening=opening['id'],
               opening_history=opening['history'], challenger_color=color,
               challenger_config=matchup['challenger'], opponent_config=matchup['opponent'],
               challenger_seed=opening['challenger_seed'], opponent_seed=opening['opponent_seed'],
               moves=[], status='incomplete', complete_history=opening['history'][:])
    start = time.perf_counter()
    try:
        row['elapsed_seconds'] = 0
        if checkpoint:
            checkpoint(row)
        while not game.is_game_over():
            mover, before = game.current_player, copy.deepcopy((game.board, game.current_player, game.piece))
            agent = agents[mover]
            wall, cpu = time.perf_counter(), time.process_time()
            col = agent.choose_move(game)
            wall, cpu = time.perf_counter()-wall, time.process_time()-cpu
            if before != (game.board, game.current_player, game.piece):
                raise RuntimeError('Caller mutation')
            if type(col) is not int or not game.make_move(col):
                raise RuntimeError('Illegal move')
            move = dict(ply=len(row['complete_history']), color=mover,
                        role='challenger' if mover == color else 'opponent', column=col,
                        wall_seconds=wall, cpu_seconds=cpu,
                        negamax_stats=agent.last_stats.copy() if isinstance(agent, NegamaxAgent) else None)
            row['moves'].append(move)
            row['complete_history'].append(col)
            row['elapsed_seconds'] = time.perf_counter()-start
            if checkpoint:
                checkpoint(row)
        winner = game.check_winner()
        row.update(status='complete', winner=winner, challenger_score=.5 if winner == -1 else float(winner == color),
                   final_rng_hashes={str(c): digest(a.rng.getstate()) for c,a in agents.items() if isinstance(a, MCTSAgent)})
    except (Exception, KeyboardInterrupt) as error:
        row['error'] = f'{type(error).__name__}: {error}'
    row['elapsed_seconds'] = time.perf_counter()-start
    return row


def validate(config, rows, preflight=False, allow_incomplete=False):
    plans = {game_id(m,o,c): (m,o,c) for m,o,c in schedule(config, preflight)}
    seen = set()
    for row in rows:
        if row['id'] in seen or row['id'] not in plans or row['experiment_sha256'] != digest(config):
            raise ValueError('Duplicate/foreign/unknown result')
        seen.add(row['id'])
        m,o,c = plans[row['id']]
        for key, expected in dict(source_commit=config['source_commit'], matchup=m['id'], opening=o['id'],
            opening_history=o['history'], challenger_color=c, challenger_config=m['challenger'],
            opponent_config=m['opponent'], challenger_seed=o['challenger_seed'], opponent_seed=o['opponent_seed']).items():
            if row[key] != expected:
                raise ValueError(f'Plan mismatch: {key}')
        game = position(o['history'])
        for i, move in enumerate(row['moves']):
            if (game.is_game_over() or type(move['column']) is not int or move['color'] != game.current_player
                or move['role'] != ('challenger' if move['color'] == c else 'opponent')
                or move['ply'] != len(o['history'])+i or not game.make_move(move['column'])):
                raise ValueError('Invalid history')
            if any(not math.isfinite(move[k]) or move[k] < 0 for k in ('wall_seconds','cpu_seconds')):
                raise ValueError('Invalid timing')
            stats = move['negamax_stats']
            role_config = m[move['role']]
            if role_config['type'] == 'negamax':
                if set(stats or {}) != {'nodes','entries','hits','cutoffs'} or any(type(v) is not int or v < 0 for v in stats.values()):
                    raise ValueError('Invalid search counters')
            elif stats is not None:
                raise ValueError('MCTS has Negamax counters')
        if row['complete_history'] != o['history']+[v['column'] for v in row['moves']]:
            raise ValueError('History mismatch')
        if row['status'] == 'complete':
            winner = game.check_winner()
            if not game.is_game_over() or winner != row['winner'] or row['challenger_score'] != (.5 if winner == -1 else float(winner == c)):
                raise ValueError('Outcome mismatch')
        elif not allow_incomplete or row['status'] != 'incomplete' or 'challenger_score' in row:
            raise ValueError('Incomplete result cannot be scored')
    return seen


class BudgetExpired(Exception):
    pass


def run(directory, preflight=False, max_games=None):
    config = json.loads((directory / ('candidate.json' if preflight else 'experiment.json')).read_text())
    check_source(config)
    prefix = 'preflight' if preflight else 'results'
    path, status_path = directory / f'{prefix}.jsonl', directory / f'{prefix}-status.json'
    rows = load_rows(path)
    done = validate(config, rows, preflight)
    attempts_path = directory / f'{prefix}-attempts.jsonl'
    attempts = load_rows(attempts_path)
    for attempt in attempts:
        validate(config, [attempt], preflight, True)
    active = directory / f'{prefix}-active.json'
    if active.exists():
        orphan = json.loads(active.read_text())
        validate(config, [orphan], preflight, True)
        if orphan['id'] not in done:
            orphan['error'] = 'Unclean interruption; original seeds restart at game boundary'
            append_json(attempts_path, orphan)
            attempts.append(orphan)
        active.unlink()
    spent = max(sum(r['elapsed_seconds'] for r in rows+attempts),
                json.loads(status_path.read_text())['cumulative_budget_seconds'] if status_path.exists() else 0)
    cap = config['preflight_seconds' if preflight else 'max_seconds']
    start, count, reason = time.perf_counter(), 0, 'complete'
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise BudgetExpired('Fixed computational cap')
    signal.signal(signal.SIGALRM, expired)
    try:
        for m,o,c in schedule(config, preflight):
            if game_id(m,o,c) in done:
                continue
            remaining = cap-spent-(time.perf_counter()-start)
            if remaining <= 0:
                reason = 'fixed compute cap'
                break
            if max_games is not None and count >= max_games:
                reason = 'checkpoint requested'
                break
            signal.setitimer(signal.ITIMER_REAL, remaining)
            # Persist an initial record too, so interrupted first moves are explicit.
            def checkpoint(row):
                atomic_json(active, row)
                atomic_json(status_path, dict(status='running', completed=len(done)+count,
                    cumulative_budget_seconds=spent+time.perf_counter()-start, cap_seconds=cap))
            row = play(config, m, o, c, checkpoint)
            signal.setitimer(signal.ITIMER_REAL, 0)
            validate(config, [row], preflight, True)
            if row['status'] != 'complete':
                append_json(attempts_path, row)
                reason = row['error']
                active.unlink(missing_ok=True)
                break
            append_json(path, row)
            active.unlink(missing_ok=True)
            count += 1
            if count % 8 == 0:
                print(f'{prefix}: {len(done)+count} games, {time.perf_counter()-start:.1f}s', flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
    summary = dict(status=reason, completed=len(done)+count, planned=len(list(schedule(config,preflight))),
                   cumulative_budget_seconds=spent+time.perf_counter()-start, cap_seconds=cap)
    atomic_json(status_path, summary)
    print(json.dumps(summary), flush=True)
    return summary


def freeze(directory):
    if (directory / 'experiment.json').exists():
        raise ValueError('Design already frozen')
    candidate = json.loads((directory / 'candidate.json').read_text())
    check_source(candidate)
    rows = load_rows(directory / 'preflight.jsonl')
    validate(candidate, rows, True)
    if len(rows) != len(list(schedule(candidate, True))):
        raise ValueError('Preflight must finish before freezing')
    memory = json.loads((directory / 'preflight-memory.json').read_text())
    if memory['candidate_sha256'] != digest(candidate):
        raise ValueError('Foreign preflight memory')
    # Runtime-only rule, with 1.75x safety margin and 300 s spare budget.
    # Matchups all have identical 50/50 cohort mixtures.
    projection = sum(sum(r['elapsed_seconds'] for r in rows if r['matchup']==m['id']) / 8 * m['pairs']
                     for m in candidate['matchups'])
    units = min(4, int(1500 / (1.75 * projection) * 4))
    if units < 1:
        raise ValueError('Even 16/8 pairs infeasible: preserve preflight and redesign before main')
    primary, secondary = 16*units, 8*units
    config = copy.deepcopy(candidate)
    config.update(matchups=matchups(primary,secondary), openings=candidate['openings'][:primary],
                  candidate_sha256=digest(candidate), freeze=dict(target_projection_seconds=projection,
                    safety_multiplier=1.75, reserved_seconds=300, primary_pairs=primary, secondary_pairs=secondary,
                    scaled_projection_seconds=projection*units/4,
                    preflight_sha256=hashlib.sha256((directory/'preflight.jsonl').read_bytes()).hexdigest(),
                    preflight_memory_sha256=digest(memory)))
    atomic_json(directory / 'experiment.json', config)
    print(json.dumps(config['freeze']), flush=True)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['declare','preflight','freeze','run'])
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--max-games', type=int)
    args = parser.parse_args()
    if args.max_games is not None and args.max_games < 1:
        parser.error('max-games must be positive')
    if args.command == 'declare':
        declare(args.directory)
    elif args.command == 'freeze':
        freeze(args.directory)
    else:
        run(args.directory, args.command=='preflight', args.max_games)


if __name__ == '__main__':
    main()
