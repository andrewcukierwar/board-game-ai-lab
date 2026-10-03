from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from threading import Event
import json

import pytest

from api.app import create_app
from api.connect4.evidence import get_explanation_context
from api.connect4.state import GameError
from games.connect4.agents.random_agent import RandomAgent
from games.connect4.grounding import build_explanation_context

BASE = '/v1/connect4'


@pytest.fixture
def app():
    return create_app({'TESTING': True})


def start(app, opponent='human', **extra):
    response = app.test_client().post(BASE + '/start_game', json={
        'player1': {'type': 'human'}, 'player2': {'type': opponent}, **extra})
    assert response.status_code == 201
    return response.json


def move(app, state, column=None):
    data = {'game_id': state['game_id'], 'revision': state['revision']}
    if column is not None:
        data['column'] = column
    return app.test_client().post(BASE + '/make_move', json=data)


def evidence(app, state, **kwargs):
    return get_explanation_context(app.extensions['connect4_games'], state['game_id'],
                                   state['revision'], **kwargs)


def history(app, state):
    with app.extensions['connect4_games'].access(state['game_id']) as session:
        return session.history


@pytest.mark.parametrize('opponent', ['human', 'random', 'negamax', 'mcts'])
def test_records_real_agent_type_boards_revisions_and_outcome(app, opponent):
    initial = start(app, opponent)
    first = move(app, initial, 3).json
    second = move(app, first, 2 if opponent == 'human' else None).json
    records = history(app, second)
    assert len(records) == 2
    assert records[0].agent_type == 'human' and records[0].player == 0
    assert records[1].agent_type == opponent and records[1].player == 1
    assert records[1].agent_depth == (2 if opponent == 'negamax' else None)
    data = [r.to_dict() for r in records]
    assert data[0]['board_before'] == initial['board']
    assert data[0]['board_after'] == data[1]['board_before'] == first['board']
    assert data[1]['board_after'] == second['board']
    assert [r['move_number'] for r in data] == [1, 2]
    assert [r['revision_before'] for r in data] == [0, 1]
    assert [r['revision'] for r in data] == [1, 2]
    assert data[1]['outcome'] == {'status': 'ongoing', 'winner': None}
    assert evidence(app, second)['move_history'] == data
    with pytest.raises(FrozenInstanceError):
        records[0].column = 0
    data[0]['board_after'][5][3] = 'O'
    assert records[0].board_after[5][3] == 'X'
    # Existing gameplay JSON remains unchanged; history is accessed internally.
    assert 'history' not in second and 'explanation' not in second


def test_rejections_never_append_history(app, monkeypatch):
    state = start(app, 'random')
    old = state
    state = move(app, state, 3).json
    recorded = history(app, state)
    assert move(app, old, 3).json['code'] == 'stale_revision'
    assert move(app, state, 2).status_code == 400
    assert move(app, state, True).status_code == 400
    def broken(self, game):
        game.make_move(0)
        raise RuntimeError('broken')
    monkeypatch.setattr(RandomAgent, 'choose_move', broken)
    assert move(app, state).status_code == 503
    monkeypatch.setattr(RandomAgent, 'choose_move', lambda self, game: -1)
    assert move(app, state).status_code == 503
    assert history(app, state) == recorded
    assert evidence(app, state)['position']['board'] == state['board']
    monkeypatch.setattr(RandomAgent, 'choose_move', lambda self, game: 2)
    state = move(app, state).json
    assert len(history(app, state)) == 2
    with pytest.raises(GameError) as exc:
        evidence(app, old)
    assert exc.value.code == 'stale_revision'


def test_full_column_rejection_does_not_record(app):
    state = start(app)
    for _ in range(6):
        state = move(app, state, 3).json
    assert move(app, state, 3).status_code == 400
    assert len(history(app, state)) == 6
    assert evidence(app, state)['confirmed_tactical_facts']['legal_columns'] == [0, 1, 2, 4, 5, 6]


def test_terminal_history_is_complete_and_replayable(app):
    state = start(app)
    for c in [0, 1, 0, 1, 0, 1, 0]:
        state = move(app, state, c).json
    context = evidence(app, state)
    assert context['move_history'][-1]['outcome'] == {'status': 'win', 'winner': 0}
    assert context['last_move_facts']['was_immediate_win']
    assert context['confirmed_tactical_facts']['legal_columns'] == []
    assert move(app, state, 3).status_code == 409
    assert len(history(app, state)) == 7
    assert json.loads(json.dumps(context, allow_nan=False)) == context


def test_draw_recorded_through_complete_legal_game(app):
    # Complete alternating play reaching rows XXOOXXO / OOXXOOX, repeated.
    sequence = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
                6, 6, 6, 6, 6]
    state = start(app)
    for c in sequence:
        response = move(app, state, c)
        assert response.status_code == 200
        state = response.json
    context = evidence(app, state)
    assert len(context['move_history']) == 42
    assert context['move_history'][-1]['outcome'] == {'status': 'draw', 'winner': None}


def test_context_is_detached_and_rule_retrieval_never_means_application(app):
    state = start(app)
    context = evidence(app, state, concept_ids=['claimeven', 'baseinverse', 'zugzwang'])
    # Empty board has plentiful plausible empty pairs, but no coverage proof.
    assert context['supported_allis_rule_applications'] == []
    assert context['analysis_limits']['formal_allis_rules_implemented'] == []
    assert context['provenance']['agent_reasoning_available'] is False
    assert context['last_move_facts'] is None
    assert context['general_strategic_observations'] == []
    assert {x['id'] for x in context['knowledge']['entries']} >= {'claimeven', 'baseinverse', 'zugzwang'}
    assert all(e['application_status'] == 'reference_only'
               for e in context['knowledge']['entries'] if e['kind'] == 'rule')
    context['position']['board'][5][0] = 'O'
    context['position']['players'][0]['type'] = 'changed'
    context['coordinates']['players'][0]['piece'] = 'changed'
    assert evidence(app, state)['position']['board'] == state['board']
    assert evidence(app, state)['coordinates']['players'][0]['piece'] == 'X'
    for c in [3, 3, 2, 3, 4]:
        state = move(app, state, c).json
    context = evidence(app, state)
    assert context['general_strategic_observations'][0]['status'] == 'context_only'
    assert {'winning_square', 'parity'} <= {e['id'] for e in context['knowledge']['entries']}


@pytest.mark.parametrize('corruption', ['before', 'after', 'column', 'revision', 'outcome', 'agent', 'truncate', 'position'])
def test_context_refuses_mismatched_history(app, corruption):
    state = move(app, start(app), 3).json
    records = history(app, state)
    board = state['board']
    if corruption in ('before', 'after'):
        records = (replace(records[0], **{'board_' + corruption: tuple(tuple('O' for _ in range(7)) for _ in range(6))}),)
    if corruption == 'column': records = (replace(records[0], column=2),)
    if corruption == 'revision': records = (replace(records[0], revision=4),)
    if corruption == 'outcome': records = (replace(records[0], outcome_status='win', winner=0),)
    if corruption == 'agent': records = (replace(records[0], agent_type='random'),)
    if corruption == 'truncate': records = ()
    if corruption == 'position': board[5][3] = 'O'
    with pytest.raises(ValueError):
        build_explanation_context(board=board, player_to_move=1, revision=1,
                                  move_history=records, players=state['players'])


def test_history_isolation_replacement_expiry_and_process_restart(app):
    state, other = start(app), start(app)
    state = move(app, state, 3).json
    assert evidence(app, other)['move_history'] == []
    fresh = start(app, replace_game_id=state['game_id'])
    assert evidence(app, fresh)['move_history'] == []
    with pytest.raises(GameError) as exc:
        evidence(app, state)
    assert exc.value.code == 'session_not_found'
    with pytest.raises(GameError):
        evidence(create_app({'TESTING': True}), fresh)
    store = app.extensions['connect4_games']
    store.clock = lambda: 0
    expiring = start(app)
    expiring = move(app, expiring, 2).json
    store.clock = lambda: store.ttl
    with pytest.raises(GameError) as exc:
        evidence(app, expiring)
    assert exc.value.code == 'session_not_found'
    assert expiring['game_id'] not in store._games


def test_evidence_and_history_respect_move_lock(app, monkeypatch):
    state = move(app, start(app, 'random'), 3).json
    other = start(app)
    entered, release = Event(), Event()
    def slow(self, game):
        entered.set()
        assert release.wait(5)
        return 2
    monkeypatch.setattr(RandomAgent, 'choose_move', slow)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(move, app, state)
        try:
            assert entered.wait(5)
            with pytest.raises(GameError) as exc:
                evidence(app, state)
            assert exc.value.code == 'game_busy'
            assert move(app, state).json['code'] == 'game_busy'
            assert evidence(app, other)['move_history'] == []
        finally:
            release.set()
        committed = future.result().json
    assert len(evidence(app, committed)['move_history']) == 2
    assert move(app, state).json['code'] == 'stale_revision'
    assert len(history(app, committed)) == 2


def test_analysis_after_capture_stays_at_its_revision(app, monkeypatch):
    import api.connect4.evidence as service
    state = start(app)
    real_builder = service.build_explanation_context
    def builder(**kwargs):
        assert move(app, state, 3).status_code == 200  # Lock was released.
        return real_builder(**kwargs)
    monkeypatch.setattr(service, 'build_explanation_context', builder)
    context = evidence(app, state)
    assert context['provenance']['revision'] == 0
    assert context['move_history'] == []
    assert context['position']['board'] == state['board']
    assert len(history(app, state)) == 1


def test_last_move_retrieves_threat_context_even_when_it_removed_every_threat(app):
    state = start(app)
    for column in [0, 1, 0, 1, 2, 1, 1]:
        state = move(app, state, column).json
    context = evidence(app, state)
    assert not any(p['squares'] for p in context['confirmed_tactical_facts']['winning_squares'])
    assert context['last_move_facts']['was_mandatory_block']
    assert context['last_move_facts']['winning_square_changes'][1]['removed']
    assert {'winning_square', 'parity'} <= {e['id'] for e in context['knowledge']['entries']}


@pytest.mark.parametrize('revision', [True, -1, '1', None])
def test_evidence_revision_validation(app, revision):
    state = start(app)
    with pytest.raises(GameError) as exc:
        get_explanation_context(app.extensions['connect4_games'], state['game_id'], revision)
    assert exc.value.code == 'invalid_revision'
