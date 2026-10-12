"""Read-only value diagnostics: realized outcomes and optimal WDL remain separate targets."""
from collections import Counter, defaultdict
import hashlib
from research.alphazero_resnet_selfplay.measures import score_examples, family
from games.connect4.alphazero_v2.data import encode_board
from games.connect4.alphazero_v2.oracle import engine_position
from .common import load, write
from .evaluate import fixtures


def exact_records(base):
    result = {}
    for suite in ('solved','deep'):
        rows = []
        for r in fixtures()[suite]:
            game = engine_position(r['moves'])
            p = base.predict(game)
            rows.append(dict(id=r['id'], cluster=r['cluster'], value=p.value, target=r['value'],
                             ply=len(r['moves']), actor=game.current_player, policy=list(p.policy)))
        result[suite] = rows
    return result


def key(e):
    return e.actor, e.observation


def replay_records(model, games, training_keys=None, check=lambda:None):
    examples = [e for g in games for e in g.examples]
    repetitions = Counter(family(e) for e in examples)
    scored, _, _, values = score_examples(model,examples)
    rows, label_errors = [], 0
    for g in games:
        check()
        end = engine_position(g.moves)
        assert end.is_game_over() and end.check_winner() == g.winner
        prefix = tuple(g.moves[:8]); reflected = tuple(6-a for a in prefix)
        opening = repr(min(prefix,reflected))
        for e in g.examples:
            game = engine_position(g.moves[:e.ply])
            outcome = 0 if g.winner == -1 else (1 if g.winner == game.current_player else -1)
            # Independent of outcome_for_actor and strict loader's audit.
            assert e.actor == game.current_player and e.observation == encode_board(game)
            label_errors += e.outcome != outcome
            rows.append(dict(id=f'{g.generation}:{g.index}:{e.ply}', cluster=opening,
                             game=f'{g.generation}:{g.index}', ply=e.ply, actor=e.actor,
                             position_key=hashlib.sha256(repr(key(e)).encode()).hexdigest(),
                             target=e.outcome, value=float(values[len(rows)]), repetitions=repetitions[family(e)],
                             seen_in_training=None if training_keys is None else family(e) in training_keys,
                             terminal_distance=len(g.moves)-e.ply))
    assert label_errors == 0
    return rows


def diagnostics(arm, freeze, output, check):
    runner6, model6, base6, entry6 = load(arm,6,freeze)
    hash6 = runner6.factory.hash(model6)
    games6 = list(runner6.replay.iter_games())
    raw = exact_records(base6)
    train6 = replay_records(model6,games6,check=check)
    runner5, model5, base5, entry5 = load(arm,5,freeze)
    hash5 = runner5.factory.hash(model5)
    games5 = list(runner5.replay.iter_games())
    keys5 = {family(e) for g in games5 for e in g.examples}
    train5 = replay_records(model5,games5,check=check)
    held = replay_records(model5,[g for g in games6 if g.generation == 6],keys5,check)
    # Exact-vs-realized target comparison ONLY for identical frozen nonsealed fixture boards.
    lookup = {}
    for suite in ('solved','deep'):
        for r in fixtures()[suite]:
            game = engine_position(r['moves'])
            lookup[(game.current_player,encode_board(game))] = r['value']
    overlap = []
    for g in games6:
        for e in g.examples:
            if key(e) in lookup:
                overlap.append(dict(game=f'{g.generation}:{g.index}',ply=e.ply,
                                    realized=e.outcome,exact=lookup[key(e)]))
    assert runner6.factory.hash(model6) == hash6 and runner5.factory.hash(model5) == hash5
    write(output / 'values.json', dict(complete=True, arm=arm,
          identities=dict(g6=entry6,g5=entry5),exact=raw,training_g6=train6,training_g5=train5,
          prequential_g5_on_g6=held,prequential_g6_on_g7=None,
          exact_replay_overlap=overlap,label_audit=dict(errors=0,positions=len(train6)+len(train5)+len(held))))
