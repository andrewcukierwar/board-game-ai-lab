"""Frozen complete-game membership, actor labels and exact/mirror overlap audit."""
import hashlib
from collections import Counter
from games.connect4.alphazero_v2.data import encode_board
from games.connect4.alphazero_v2.oracle import engine_position
from .common import parent, read, RESULTS


def exact(e):
    return e.actor, e.observation


def mirrored(e):
    return e.actor, min(e.observation, tuple(row[::-1] for row in e.observation))


def identity(key):
    return hashlib.sha256(repr(key).encode()).hexdigest()


def records(games, exact_train=None, mirror_train=None):
    rows, examples = [], []
    for g in games:
        terminal = engine_position(g.moves)
        assert terminal.is_game_over() and terminal.check_winner() == g.winner
        prefix = tuple(g.moves[:8])
        family = repr(min(prefix, tuple(6-a for a in prefix)))
        for e in g.examples:
            board = engine_position(g.moves[:e.ply])
            outcome = 0 if g.winner == -1 else (1 if g.winner == board.current_player else -1)
            assert e.actor == board.current_player and e.observation == encode_board(board) and e.outcome == outcome
            rows.append(dict(id=f'{g.generation}:{g.index}:{e.ply}', game=f'{g.generation}:{g.index}',
                cluster=family, ply=e.ply, actor=e.actor, target=e.outcome,
                exact_key=identity(exact(e)), mirror_key=identity(mirrored(e)),
                exact_overlap=False if exact_train is None else exact(e) in exact_train,
                mirror_overlap=False if mirror_train is None else mirrored(e) in mirror_train))
            examples.append(e)
    return rows, examples


def datasets(seed):
    g5, _ = parent(seed,5)
    g6, _ = parent(seed,6)
    training = list(g5.replay.iter_games())
    held = [g for g in g6.replay.iter_games() if g.generation == 6]
    assert len(held) == 144 and all(g.generation <= 5 for g in training)
    old = {(g.generation,g.index):g for g in g6.replay.iter_games() if g.generation <= 5}
    assert all(old[(g.generation,g.index)] == g for g in training)
    keys = {exact(e) for g in training for e in g.examples}
    mirrors = {mirrored(e) for g in training for e in g.examples}
    return records(training), records(held,keys,mirrors)


def membership():
    result = {}
    for seed in (42,314159):
        (train,_),(held,_) = datasets(seed)
        novel = [r for r in held if not r['mirror_overlap']]
        result[str(seed)] = dict(training=train, heldout=held,
            audit=dict(train_positions=len(train), train_games=len({r['game'] for r in train}),
                heldout_positions=len(held), heldout_games=len({r['game'] for r in held}),
                exact_overlap=sum(r['exact_overlap'] for r in held),
                mirror_overlap=sum(r['mirror_overlap'] for r in held), novel_positions=len(novel),
                novel_games=len({r['game'] for r in novel}),novel_families=len({r['cluster'] for r in novel}),
                game_family_counts=dict(Counter(r['cluster'] for r in novel)), label_errors=0))
    return result


def enforce_membership(seed, train_rows, held_rows):
    expected = read(RESULTS/'membership.json')[str(seed)]
    if train_rows != expected['training'] or held_rows != expected['heldout']:
        raise ValueError('Frozen position/game/family/novelty membership differs')
