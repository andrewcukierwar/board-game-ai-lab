"""Model-blind exact-value holdout; engine proof plus independent drop scanner.

Inert on import. This data never supplies optimizer targets or stopping decisions.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from .connect4 import Connect4
from .tactical_value import tactical_proof
from .neural_symmetry_diagnostics import board_key, reflected_key, scanner_wins


def position(moves):
    game = Connect4()
    for move in moves:
        if game.is_game_over() or move not in game.get_valid_moves() or not game.make_move(move):
            raise ValueError('Invalid holdout history')
    return game


def independent_proof(game):
    """Standalone drop/window wins; draw escape checked by board occupancy.

    Never use check_winner for independent labels. Successors cannot be wins for
    the mover once scanner_wins(game) is empty; remaining terminal escape is draw.
    """
    wins = sorted(scanner_wins(game))
    if wins:
        return dict(value=1, winning_actions=wins, reply_witnesses={})
    replies = {}
    for action in game.get_valid_moves():
        after = deepcopy(game)
        assert after.make_move(action)
        if all(c != ' ' for row in after.board for c in row):
            return dict(value=None, winning_actions=[], reply_witnesses={})
        winning = sorted(scanner_wins(after))
        if not winning:
            return dict(value=None, winning_actions=[], reply_witnesses={})
        replies[str(action)] = winning
    return dict(value=-1, winning_actions=[], reply_witnesses=replies)


def freeze(path, *, excluded_rows, bases_per_cell=16, seed=420404, max_prefixes=200000):
    """Predeclared 128 rows; fallback largest balanced completed cells at cap."""
    excluded = set()
    for row in excluded_rows:
        key = board_key(position(row['moves']))
        excluded.update((key, reflected_key(key)))
    quotas = {(v,a): [] for v in (1,-1) for a in (0,1)}
    rng = random.Random(seed)
    examined = 0
    for examined in range(1,max_prefixes+1):
        game, moves = Connect4(), []
        for _ in range(rng.randrange(5,35)):
            if game.is_game_over():
                break
            action = rng.choice(game.get_valid_moves())
            assert game.make_move(action)
            moves.append(action)
        if game.is_game_over():
            continue
        key, mirror = board_key(game), reflected_key(board_key(game))
        if key in excluded or mirror in excluded or key == mirror:
            continue
        proof = tactical_proof(game)
        if proof['value'] is None or len(quotas[(proof['value'],game.current_player)]) >= bases_per_cell:
            continue
        assert independent_proof(game) == proof
        quotas[(proof['value'],game.current_player)].append(moves)
        excluded.update((key,mirror))
        if all(len(rows)==bases_per_cell for rows in quotas.values()):
            break
    achieved = min(map(len,quotas.values()))
    assert achieved > 0
    rows, seen = [], set()
    for (value,actor),histories in quotas.items():
        for moves in histories[:achieved]:
            name = f'exact_{len(rows)//2:02d}'
            for reflected in (False,True):
                history = [6-m for m in moves] if reflected else moves
                game = position(history)
                proof = tactical_proof(game)
                assert proof == independent_proof(game) and proof['value']==value
                key = board_key(game)
                assert key not in seen
                seen.add(key)
                rows.append(dict(name=name+'_mirror' if reflected else name, moves=history,
                                 actor=actor, proven_value=value, proof=proof,
                                 mirror_of=name if reflected else None))
    payload = dict(format_version=1, seed=seed, predeclared_bases_per_actor_value=bases_per_cell,
                   achieved_bases_per_actor_value=achieved, max_prefixes=max_prefixes,
                   examined_prefixes=examined, model_blind=True, training_feedback=False,
                   exclusion='all existing frozen/blind suites and original probes; board+actor and reflections',
                   positions=rows)
    with Path(path).open('x') as stream:
        stream.write(json.dumps(payload,indent=2,allow_nan=False)+'\n')
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
