"""Frozen development evaluations: raw policy, raw PUCT and fixed-opponent matches.

Deployment-mode search everywhere: raw PUCT with the bitboard backend, root
noise OFF, tactical guard OFF, temperature 0 with seeded maximum-visit ties.
Every random stream is derived from (namespace, row or opening, role, frozen
seed) and never from a training stream, so evaluation cannot change a training
trajectory and every arm, seed and checkpoint sees identical identities.
Only the nonsealed development packages are ever read.
"""
import math

from games.connect4.alphazero_v2.arena import NegamaxOpponent, paired_game
from games.connect4.alphazero_v2.data import seeded_rng
from games.connect4.alphazero_v2.evaluation import V2SearchArenaAgent, nn_only_choice
from games.connect4.alphazero_v2.oracle import engine_position
from games.connect4.alphazero_v2.search import EvaluationAgent

from .common import FIXTURES_SHA256, REPO, ROOT, read, sha, write

POLICY_NAMESPACE = 'phase4d4h-c-policy'
SEARCH_NAMESPACE = 'phase4d4h-c-evaluation'
MATCH_NAMESPACE = 'phase4d4h-c-strength'
SUITES = ('solved', 'tactical', 'deep')


def fixtures():
    """Phase 4D.4G's frozen development fixtures, byte-identical (solved300, tactical200, deep128)."""
    path = ROOT / 'fixtures.json'
    if sha(path) != FIXTURES_SHA256:
        raise ValueError('Development fixtures differ from the Phase 4D.4G freeze')
    return read(path)


def openings():
    """All 40 base prefix opening families of the development opening package (never the sealed one)."""
    package = read(REPO / 'games/connect4/alphazero_v2/frozen/openings-development.json')
    if package['split'] != 'development':
        raise ValueError('Only the development opening package may be used')
    rows = [r for r in package['rows'] if r['id'].endswith('-base')]
    if len(rows) != 40 or len({r['family'] for r in rows}) != 40:
        raise ValueError('Expected 40 distinct base opening families')
    return rows


class RawAgent(V2SearchArenaAgent):
    """Raw PUCT at a fixed budget on the bitboard backend; accepts CNN and ResNet inference alike."""

    def __init__(self, inference, simulations, *, name, weights_sha256):
        super().__init__(inference, simulations, name=name, identity=dict(weights_sha256=weights_sha256))

    def choose(self, game, rng):
        result, selection = EvaluationAgent(self.inference, self.simulations, rng=rng, backend='bitboard').search(game)
        if sum(result.visits) != self.simulations:
            raise ValueError('Evaluation simulation budget differs')
        return selection.move, dict(visits=list(result.visits), raw_value=result.root_value)


def raw_record(inference, suite, row):
    """Raw network only: legal-masked policy, seeded-tie argmax and the raw value for the player to move."""
    prediction = inference.predict(engine_position(row['moves']))
    action = nn_only_choice(inference, row, namespace=POLICY_NAMESPACE)
    record = dict(id=row['id'], cluster=row['cluster'], policy=list(prediction.policy), value=prediction.value,
                  policy_action=action, confidence=max(prediction.policy))
    if suite in ('solved', 'deep'):
        mass = sum(prediction.policy[a] for a in row['optimal_actions'])
        record.update(ce=-math.log(max(mass, 1e-300)), optimal_mass=mass, exact=row['value'],
                      accuracy=int(action in row['optimal_actions']), se=(prediction.value - row['value']) ** 2,
                      policy_deep_mistake=int(row['action_values'][str(action)] < row['value']))
    else:
        record.update(category=row['category'], policy_correct=int(action == row['expected_action']),
                      policy_avoidable=int(action not in row['safe_actions']),
                      policy_missed_win=int(bool(row['immediate_wins']) and action not in row['immediate_wins']))
    return record


def raw_suites(inference, check=None):
    data, result = fixtures(), {}
    for suite in SUITES:
        rows = []
        for row in data[suite]:
            if check is not None:
                check()
            rows.append(raw_record(inference, suite, row))
        result[suite] = rows
    return result


def positions(inference, simulations, search_seed, check, output):
    """Raw policy plus raw PUCT on every fixture row; one fixed search seed per row, shared by every model."""
    data = fixtures()
    for suite in SUITES:
        rows = []
        for row in data[suite]:
            check()
            record = raw_record(inference, suite, row)
            rng = seeded_rng(f'{SEARCH_NAMESPACE}:{suite}:{row["id"]}', search_seed)
            result, selection = EvaluationAgent(inference, simulations, rng=rng, backend='bitboard').search(
                engine_position(row['moves']))
            if sum(result.visits) != simulations:
                raise ValueError('Evaluation simulation budget differs')
            record.update(puct_action=selection.move, visits=list(result.visits))
            if suite == 'tactical':
                record.update(puct_correct=int(selection.move == row['expected_action']),
                              puct_avoidable=int(selection.move not in row['safe_actions']),
                              puct_missed_win=int(bool(row['immediate_wins']) and selection.move not in row['immediate_wins']))
            else:
                record.update(puct_accuracy=int(selection.move in row['optimal_actions']),
                              puct_deep_mistake=int(row['action_values'][str(selection.move)] < row['value']))
            rows.append(record)
        write(output / f'{suite}.json', dict(complete=True, simulations=simulations, rows=rows))


def matches(agent, depth, match_seed, check, output, rows=None):
    """Both colors on every opening family against the fixed reference-negamax opponent."""
    opponent = NegamaxOpponent(depth)
    records = []
    for row in (openings() if rows is None else rows):
        for game_index in (0, 1):
            check()
            record = paired_game(agent, opponent, row, game_index, namespace=MATCH_NAMESPACE, seed=match_seed,
                                 check=None, keep_decisions=False)
            if record['abandoned']:
                raise RuntimeError('Abandoned strength game')
            record['score'] = {'win': 1.0, 'draw': 0.5, 'loss': 0.0}[record['result']]
            records.append(record)
    if output is not None:
        write(output / 'matches.json', dict(complete=True, agent=agent.describe(), opponent=opponent.describe(),
                                            records=records))
    return records
