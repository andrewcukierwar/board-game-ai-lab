"""Model-blind symmetry holdout construction, independent of neural training.

Labels use a standalone four-cell window scanner, cross-checked against actual
engine successors and the existing tactical helpers. Freeze before any training.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import json
import random

from .connect4 import Connect4
from .agents.mcts_agent import MCTSAgent


def board_key(game):
    return (game.current_player, tuple(tuple(row) for row in game.board))


def reflected_key(key):
    return key[0], tuple(row[::-1] for row in key[1])


def scanner_wins(game):
    """Independently drop pieces and scan all horizontal/vertical/diagonal windows."""
    wins = []
    for col in game.get_valid_moves():
        board = [row[:] for row in game.board]
        row = max(r for r in range(6) if board[r][col] == ' ')
        board[row][col] = game.piece
        found = any(all(board[r+k*dr][c+k*dc] == game.piece for k in range(4))
                    for dr,dc in ((0,1),(1,0),(1,1),(1,-1))
                    for r in range(6) for c in range(7)
                    if 0 <= r+3*dr < 6 and 0 <= c+3*dc < 7)
        if found:
            wins.append(col)
    return wins


def independent_label(game):
    wins = scanner_wins(game)
    if len(wins) == 1:
        return 'immediate_win', wins[0]
    if wins:
        return None
    safe = []
    for col in game.get_valid_moves():
        successor = deepcopy(game)
        assert successor.make_move(col)
        if successor.is_game_over() or not scanner_wins(successor):
            safe.append(col)
    if len(safe) == 1 and len(game.get_valid_moves()) > 1:
        return 'forced_block', safe[0]
    return None


def validate_rows(rows):
    from .neural_self_play import position, require
    helper = MCTSAgent(1)
    seen = set()
    for row in rows:
        game = position(row['moves'])
        require(not game.is_game_over() and game.current_player == row['player'], 'Invalid blind fixture')
        require(independent_label(game) == (row['category'], row['expected_action']), 'Independent label mismatch')
        wins = helper._winning_moves(game)
        expected = wins if wins else helper._safe_moves(game)
        require(expected == [row['expected_action']], 'Engine/helper label mismatch')
        key = board_key(game)
        require(key not in seen, 'Duplicate blind board')
        seen.add(key)
    return seen


def freeze(path, *, seed=420402, excluded_rows=()):
    from .neural_self_play import position
    from .neural_evaluation import frozen_fixtures
    excluded = set()
    for rows in frozen_fixtures().values():
        for row in rows:
            key = board_key(position(row['moves']))
            excluded.update((key, reflected_key(key)))
    for row in excluded_rows:
        key = board_key(position(row['moves']))
        excluded.update((key, reflected_key(key)))
    rng = random.Random(seed)
    rows, quotas = [], {(kind,p):0 for kind in ('immediate_win','forced_block') for p in (0,1)}
    examined = 0
    while len(rows) < 96 and examined < 200000:
        game, moves = Connect4(), []
        for _ in range(rng.randrange(5, 31)):
            if game.is_game_over():
                break
            move = rng.choice(game.get_valid_moves())
            game.make_move(move)
            moves.append(move)
        examined += 1
        if game.is_game_over():
            continue
        key = board_key(game)
        mirror_key = reflected_key(key)
        if key in excluded or mirror_key in excluded or key == mirror_key:
            continue
        label = independent_label(game)
        if label is None or quotas[(label[0],game.current_player)] >= 12:
            continue
        name = f'blind_{len(rows)//2:02d}_{label[0]}'
        rows.append(dict(name=name, moves=moves, player=game.current_player,
                         category=label[0], expected_action=label[1], mirror_of=None))
        rows.append(dict(name=name+'_mirror', moves=[6-m for m in moves], player=game.current_player,
                         category=label[0], expected_action=6-label[1], mirror_of=name))
        quotas[(label[0],game.current_player)] += 1
        excluded.update((key,mirror_key))
    assert len(rows) == 96, 'Unable to fill predeclared quotas'
    validate_rows(rows)
    payload = dict(format_version=1, seed=seed, positions=rows, examined_prefixes=examined,
                   construction='random legal prefixes; 12 unique bases per actor/category plus mirrors',
                   exclusion=('board plus actor; all three old suites and their reflections' if not excluded_rows else
                              'board plus actor; all three old suites and supplied earlier rows, plus reflections'),
                   labels='independent four-cell window scan; actual-engine/helper cross-check',
                   training_feedback=False, model_blind=True)
    with Path(path).open('x') as stream:
        stream.write(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
