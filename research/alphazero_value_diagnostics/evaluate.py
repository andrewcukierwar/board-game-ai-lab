"""Same H-C deterministic identities, unchanged 256-simulation search and negamax3."""
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.search import EvaluationAgent
from games.connect4.alphazero_v2.arena import NegamaxOpponent, paired_game
from research.alphazero_resnet_selfplay.evaluate import fixtures, openings, RawAgent, SEARCH_NAMESPACE, MATCH_NAMESPACE
from .common import write


def search(inference, suite, row, seed):
    result, selection = EvaluationAgent(inference, 256, rng=seeded_rng(
        f'{SEARCH_NAMESPACE}:{suite}:{row["id"]}', seed), backend='bitboard').search(engine_position(row['moves']))
    assert sum(result.visits) == 256 and result.root_noise is None and selection.temperature == 0
    action = selection.move
    record = dict(id=row['id'], cluster=row['cluster'], action=action, visits=list(result.visits),
                  root_value=result.root_value, root_prior=list(result.root_prior),
                  terminal_leaves=result.terminal_leaves, max_depth=result.max_depth,
                  mean_depth=result.leaf_depth_sum / 256)
    if suite == 'tactical':
        record.update(category=row['category'], correct=int(action == row['expected_action']),
                      avoidable=int(bool(row['safe_actions']) and action not in row['safe_actions']),
                      missed_win=int(bool(row['immediate_wins']) and action not in row['immediate_wins']))
    else:
        value, chosen = row['value'], row['action_values'][str(action)]
        record.update(correct=int(action in row['optimal_actions']), mistake=int(chosen < value),
                      exact=value, chosen_value=chosen, win_draw=int(value == 1 and chosen == 0),
                      win_loss=int(value == 1 and chosen == -1), draw_loss=int(value == 0 and chosen == -1))
    return record


def positions(inference, plan, output, check):
    for suite, rows in fixtures().items():
        if suite not in ('tactical', 'solved', 'deep'):
            continue
        records = []
        for row in rows:
            check()
            records.append(search(inference, suite, row, plan['search_seed']))
        write(output / f'{suite}.json', dict(complete=True, rows=records))


def matches(inference, plan, output, check, *, rows=None):
    agent = RawAgent(inference, 256, name='frozen-diagnostic', weights_sha256='see-identity.json')
    records = []
    for row in openings() if rows is None else rows:
        for game_index in (0, 1):
            check()
            record = paired_game(agent, NegamaxOpponent(3), row, game_index,
                                 namespace=MATCH_NAMESPACE, seed=plan['match_seed'], check=check, keep_decisions=False)
            assert not record['abandoned']
            record['score'] = {'win': 1., 'draw': .5, 'loss': 0.}[record['result']]
            records.append(record)
            if output is not None:
                write(output / 'matches-progress.json', dict(complete=False, records=records))
    if output is not None:
        write(output / 'matches.json', dict(complete=True, records=records))
    return records
