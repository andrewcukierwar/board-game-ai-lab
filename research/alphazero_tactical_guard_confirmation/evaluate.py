"""Paired confirmation matches; the 4D.4J-A decision/agent wrapper is reused unchanged."""
from games.connect4.alphazero_v2.arena import paired_game, NegamaxOpponent
from research.alphazero_tactical_guard.evaluate import Agent, positions
from .common import RESULTS, read, write

MATCH_NAMESPACE = 'phase4d4j-b-confirmation'
MATCH_SEED = 4420003


def openings():
    package = read(RESULTS / 'openings.json')
    rows = package['confirmation']
    assert len(rows) == read(RESULTS / 'design.json')['families'] and len({r['family'] for r in rows}) == len(rows)
    return rows


def matches(inference, seed, depth, variant, check, output):
    """Both arms share namespace/seed/opening/game index/role, so every pair starts from identical streams."""
    agent, records, rows = Agent(inference, variant == 'guarded'), [], openings()
    for row in rows:
        for index in (0, 1):
            check()
            record = paired_game(agent, NegamaxOpponent(depth), row, index, namespace=f'{MATCH_NAMESPACE}:negamax{depth}',
                                 seed=MATCH_SEED, check=check, keep_decisions=True)
            assert not record['abandoned']
            record['score'] = {'win': 1., 'draw': .5, 'loss': 0.}[record['result']]
            records.append(record)
            if len(records) % 16 == 0:
                write(output / 'progress.json', dict(variant=variant, games=len(records), total=2 * len(rows)))
    write(output / f'{variant}-matches.json', dict(complete=True, seed=seed, opponent=f'negamax{depth}', variant=variant, records=records))
    return records


__all__ = ['matches', 'positions', 'openings']
