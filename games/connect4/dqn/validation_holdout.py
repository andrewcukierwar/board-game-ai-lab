"""Model-blind construction and independent engine validation of Phase 4C.2d data."""
import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ..connect4 import Connect4

PATH = Path(__file__).with_name('validation_positions.json')
OLD_PATH = Path(__file__).with_name('diagnostic_positions.json')
SEED = 420204


def replay(moves, *, terminal=False):
    game = Connect4()
    for col in moves:
        if type(col) is not int or game.is_game_over() or not game.is_valid_move(col):
            raise ValueError('Illegal prefix')
        game.make_move(col)
    if not terminal and game.is_game_over():
        raise ValueError('Terminal prefix')
    return game


def key(game):
    board = tuple(''.join(row) for row in game.board)
    return game.current_player, min(board, tuple(row[::-1] for row in board))


def wins(game):
    result = []
    for col in range(7):
        if game.is_valid_move(col):
            child = deepcopy(game)
            child.make_move(col)
            if child.check_winner() == game.current_player:
                result.append(col)
    return result


def label(game):
    own = wins(game)
    threat = game
    if len(own) == 1:
        category, action = 'win', own[0]
    elif not own:
        threat = deepcopy(game)
        threat.current_player = 1 - game.current_player
        threat.piece = 'XO'[threat.current_player]
        opposing = wins(threat)
        if len(opposing) != 1:
            return None
        safe = []
        for col in game.get_valid_moves():
            child = deepcopy(game)
            child.make_move(col)
            if child.is_game_over() or not wins(child):
                safe.append(col)
        if safe != opposing:
            return None
        category, action = 'block', safe[0]
    else:
        return None
    child = deepcopy(threat)
    piece = child.piece
    child.make_move(action)
    directions = set()
    for row in range(6):
        for col in range(7):
            for dr, dc, direction in ((0, 1, 'horizontal'), (1, 0, 'vertical'),
                                      (1, 1, 'diagonal'), (1, -1, 'diagonal')):
                if (0 <= row+3*dr < 6 and 0 <= col+3*dc < 7
                        and all(child.board[row+i*dr][col+i*dc] == piece for i in range(4))):
                    directions.add(direction)
    if len(directions) != 1:
        return None
    return category, directions.pop(), action


def reject_duplicate(game, seen):
    identity = key(game)
    if identity in seen:
        raise ValueError('Duplicate or mirrored duplicate')
    seen.add(identity)


def validate(rows):
    seen = {key(replay(f['moves'])) for f in json.loads(OLD_PATH.read_text())['positions']}
    originals = {}
    names = set()
    for row in rows:
        if row['name'] in names:
            raise ValueError('Duplicate name')
        names.add(row['name'])
        game = replay(row['moves'])
        if (label(game) != (row['category'], row['direction'], row['expected_action'])
                or game.current_player != row['player']):
            raise ValueError('Invalid tactical label')
        if row['mirror_of'] is None:
            reject_duplicate(game, seen)
            originals[row['name']] = row
        else:
            original = originals.get(row['mirror_of'])
            if (original is None or row['moves'] != [6-c for c in original['moves']]
                    or row['expected_action'] != 6-original['expected_action']):
                raise ValueError('Invalid mirror')
    if Counter(r['mirror_of'] for r in rows if r['mirror_of']) != Counter({n: 1 for n in originals}):
        raise ValueError('Each original requires exactly one mirror')
    return len(originals)


def construct():
    rng = random.Random(SEED)
    strata = [(k, d, p) for k in ('win', 'block')
              for d in ('horizontal', 'vertical', 'diagonal') for p in (0, 1)]
    # 48 families: 14 each outer column pair, 6 center-column families.
    quotas = Counter()
    for i, s in enumerate(strata):
        for bucket in (0, 1, 2, (0, 1, 2, 3, 3, 3)[i % 6]):
            quotas[s + (bucket,)] += 1
    seen = {key(replay(f['moves'])) for f in json.loads(OLD_PATH.read_text())['positions']}
    rows = []
    for attempt in range(200000):
        game, moves = Connect4(), []
        for _ in range(rng.randrange(6, 35)):
            if game.is_game_over():
                break
            move = rng.choice(game.get_valid_moves())
            moves.append(move)
            game.make_move(move)
        if game.is_game_over() or key(game) in seen:
            continue
        found = label(game)
        if found is None:
            continue
        category, direction, action = found
        slot = category, direction, game.current_player, min(action, 6-action)
        if quotas[slot] == 0:
            continue
        reject_duplicate(game, seen)
        quotas[slot] -= 1
        name = f'holdout_{len(rows)//2:02d}_{category}_{direction}_{game.current_player}'
        original = dict(name=name, moves=moves, expected_action=action, category=category,
                        direction=direction, player=game.current_player, mirror_of=None)
        rows.extend([original, dict(original, name=name+'_mirror', moves=[6-c for c in moves],
                                    expected_action=6-action, mirror_of=name)])
        if not sum(quotas.values()):
            validate(rows)
            return dict(version=1, construction_seed=SEED, attempted_rollouts=attempt+1,
                        original_families=48, mirrored_rows=48, positions=rows)
    raise RuntimeError(f'Construction exhausted: {quotas}')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


if __name__ == '__main__':
    data = construct()
    with PATH.open('x') as stream:
        stream.write(json.dumps(data, indent=2)+'\n')
    print(PATH, digest(PATH))
