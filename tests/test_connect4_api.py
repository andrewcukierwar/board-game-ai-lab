from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from api.app import create_app
from games.connect4.connect4 import Connect4
from games.connect4.agents.random_agent import RandomAgent

BASE = '/v1/connect4'


@pytest.fixture
def app():
    return create_app({'TESTING': True})


@pytest.fixture
def client(app):
    return app.test_client()


def start(client, opponent='human', **extra):
    response = client.post(BASE + '/start_game', json={
        'player1': {'type': 'human'}, 'player2': {'type': opponent}, **extra})
    assert response.status_code == 201, response.json
    return response.json


def move(client, state, column=None):
    data = {'game_id': state['game_id'], 'revision': state['revision']}
    if column is not None:
        data['column'] = column
    return client.post(BASE + '/make_move', json=data)


def read(client, state):
    return client.get(BASE + '/games/' + state['game_id'])


def test_independent_sessions_and_app_instances(client, app):
    first, second = start(client), start(client)
    response = move(client, first, 3)
    assert response.status_code == 200
    assert response.json['board'][5][3] == 'X'
    assert read(client, second).json == second
    other_app = create_app({'TESTING': True})
    assert read(other_app.test_client(), first).status_code == 404


@pytest.mark.parametrize('config', [None, [], 'random', {}, {'type': []},
    {'type': 'mcts_nn'}, {'type': 'negamax', 'depth': True}, {'type': 'negamax', 'depth': False},
    {'type': 'negamax', 'depth': 0}, {'type': 'negamax', 'depth': 9}, {'type': 'negamax', 'depth': 10}, {'type': 'negamax', 'depth': None}, {'type': 'negamax', 'depth': 1000000},
    {'type': 'negamax', 'depth': 2.0}, {'type': 'negamax', 'depth': '2'},
    {'type': 'random', 'depth': 2}, {'type': 'human', 'extra': 1}])
def test_invalid_configuration_does_not_replace_game(client, config):
    old = start(client)
    response = client.post(BASE + '/start_game', json={
        'player2': config, 'replace_game_id': old['game_id']})
    assert response.status_code == 400
    assert response.json['code'] == 'invalid_agent'
    assert read(client, old).json == old


@pytest.mark.parametrize('data,content_type', [('null', 'application/json'),
    ('[]', 'application/json'), ('{', 'application/json'), ('{}', 'text/plain')])
def test_non_object_or_malformed_json(client, data, content_type):
    response = client.post(BASE + '/start_game', data=data, content_type=content_type)
    assert response.status_code == 400
    assert response.json['code'] == 'invalid_request'


def test_large_body(client):
    response = client.post(BASE + '/start_game', json={'x': 'x' * 5000})
    assert response.status_code == 413
    assert response.json['code'] == 'request_too_large'


@pytest.mark.parametrize('column', [-1, 7, True, False, 1.5, '3', [], {}, None])
def test_bad_move_types_do_not_mutate(client, column):
    state = start(client)
    response = client.post(BASE + '/make_move', json={
        'game_id': state['game_id'], 'revision': 0, 'column': column})
    assert response.status_code == 400
    assert response.json['code'] == 'invalid_move'
    assert read(client, state).json == state


@pytest.mark.parametrize('revision', [None, True, -1, '0', 0.5])
def test_bad_revisions(client, revision):
    state = start(client)
    response = client.post(BASE + '/make_move', json={
        'game_id': state['game_id'], 'revision': revision, 'column': 3})
    assert response.status_code == 400
    assert read(client, state).json == state


def test_missing_game_and_unknown_fields(client):
    assert client.post(BASE + '/make_move', json={'revision': 0, 'column': 3}).json['code'] == 'invalid_game_id'
    assert client.get(BASE + '/games/not-a-game').status_code == 404
    assert client.post(BASE + '/start_game', json={'unexpected': 1}).status_code == 400
    state = start(client)
    assert client.post(BASE + '/make_move', json={
        'game_id': state['game_id'], 'revision': 0, 'column': 3, 'extra': 1}).status_code == 400
    assert read(client, state).json == state


def test_full_column_and_duplicate_request(client):
    state = start(client)
    old = state
    state = move(client, state, 3).json
    assert move(client, old, 3).json['code'] == 'stale_revision'
    for _ in range(5):
        state = move(client, state, 3).json
    response = move(client, state, 3)
    assert response.status_code == 400
    assert read(client, state).json == state


def test_terminal_win_and_restart_change_opponent(client):
    state = start(client)
    for column in [0, 1, 0, 1, 0, 1, 0]:
        state = move(client, state, column).json
    assert state['gameOver'] and state['winner'] == 'Player 1'
    assert state['legalMoves'] == []
    assert move(client, state, 2).json['code'] == 'game_over'
    assert read(client, state).json == state
    fresh = start(client, 'random', replace_game_id=state['game_id'])
    assert fresh['revision'] == 0 and fresh['players'][1]['type'] == 'random'
    assert read(client, state).status_code == 404
    assert all(cell == ' ' for row in fresh['board'] for cell in row)


def test_draw(client, app):
    state = start(client)
    board = [list(row) for row in ['XXOOXXO', 'OOXXOOX'] * 3]
    board[0][0] = ' '
    with app.extensions['connect4_games'].access(state['game_id']) as session:
        session.game = Connect4(board, 0)
        assert session.game.check_winner() == -1
    result = move(client, state, 0).json
    assert result['gameOver'] and result['winner'] == 'Draw'
    assert move(client, result, 1).json['code'] == 'game_over'


@pytest.mark.parametrize('output', [-1, 7, True, 1.5, None])
def test_bad_agent_output_is_atomic(client, monkeypatch, output):
    state = move(client, start(client, 'random'), 3).json
    monkeypatch.setattr(RandomAgent, 'choose_move', lambda self, game: output)
    response = move(client, state)
    assert response.status_code == 503
    assert response.json['code'] == 'agent_failed'
    assert read(client, state).json == state


def test_agent_exception_and_input_mutation_are_isolated(client, monkeypatch):
    state = move(client, start(client, 'random'), 3).json
    def broken(self, game):
        game.make_move(0)
        raise RuntimeError('private implementation detail')
    monkeypatch.setattr(RandomAgent, 'choose_move', broken)
    response = move(client, state)
    assert response.status_code == 503
    assert 'private implementation detail' not in response.json['error']
    assert read(client, state).json == state
    monkeypatch.setattr(RandomAgent, 'choose_move', lambda self, game: 2)
    assert move(client, state).json['revision'] == 2


def test_wrong_turn_request(client):
    state = start(client, 'random')
    assert move(client, state).status_code == 400
    state = move(client, state, 3).json
    assert move(client, state, 3).status_code == 400
    assert read(client, state).json == state


@pytest.mark.parametrize('opponent', ['random', 'negamax'])
def test_complete_game_through_api(client, opponent):
    state = start(client, opponent)
    for _ in range(42):
        response = move(client, state, state['legalMoves'][0]) if state['currentPlayer'] == 0 else move(client, state)
        assert response.status_code == 200, response.json
        next_state = response.json
        assert next_state['revision'] == state['revision'] + 1
        state = next_state
        if state['gameOver']:
            break
    assert state['gameOver'] and state['winner'] in ['Player 1', 'Player 2', 'Draw']


def test_capacity_expiration_and_replacement():
    app = create_app({'TESTING': True, 'GAME_SESSION_CAPACITY': 1, 'GAME_SESSION_TTL': 10})
    store = app.extensions['connect4_games']
    now = [0]
    store.clock = lambda: now[0]
    client = app.test_client()
    old = start(client)
    assert client.post(BASE + '/start_game', json={}).status_code == 503
    replacement = start(client, replace_game_id=old['game_id'])
    assert read(client, old).status_code == 404
    now[0] = 10
    assert read(client, replacement).json['code'] == 'session_not_found'
    fresh = start(client, replace_game_id=replacement['game_id'])
    assert fresh['game_id'] != replacement['game_id']
    assert len(store._games) == 1


def test_concurrent_moves_and_restart_are_locked(app, client, monkeypatch):
    state = move(client, start(client, 'random'), 3).json
    other = start(client)
    entered, release = Event(), Event()
    def slow(self, game):
        entered.set()
        assert release.wait(5)
        return 2
    monkeypatch.setattr(RandomAgent, 'choose_move', slow)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda: move(app.test_client(), state))
        try:
            assert entered.wait(5)
            assert move(client, state).json['code'] == 'game_busy'
            assert client.post(BASE + '/start_game', json={'replace_game_id': state['game_id']}).json['code'] == 'game_busy'
            assert move(client, other, 4).status_code == 200
        finally:
            release.set()
        assert future.result().status_code == 200
    assert move(client, state).json['code'] == 'stale_revision'
    assert read(client, state).json['revision'] == 2


@pytest.mark.parametrize('depth', range(1, 9))
def test_supported_negamax_depths(client, depth):
    state = start(client, player2={'type': 'negamax', 'depth': depth})
    state = move(client, state, 3).json
    result = move(client, state)
    assert result.status_code == 200
    assert result.json['revision'] == 2


def test_ai_cannot_choose_full_column(client, app, monkeypatch):
    state = start(client, 'random')
    with app.extensions['connect4_games'].access(state['game_id']) as session:
        for _ in range(6):
            session.game.make_move(0)
        session.game.make_move(3)
    before = read(client, state).json
    monkeypatch.setattr(RandomAgent, 'choose_move', lambda self, game: 0)
    assert move(client, before).json['code'] == 'agent_failed'
    assert read(client, state).json == before


def test_active_session_is_not_expired_or_evicted():
    app = create_app({'TESTING': True, 'GAME_SESSION_CAPACITY': 1, 'GAME_SESSION_TTL': 10})
    store = app.extensions['connect4_games']
    now = [0]
    store.clock = lambda: now[0]
    client = app.test_client()
    state = start(client)
    with store.access(state['game_id']):
        now[0] = 20
        assert client.post(BASE + '/start_game', json={}).status_code == 503
        assert read(client, state).json['code'] == 'game_busy'
    assert read(client, state).status_code == 200
    now[0] = 29
    assert read(client, state).status_code == 200
    now[0] = 38
    assert read(client, state).status_code == 200
    now[0] = 48
    assert read(client, state).status_code == 404


def test_game_snapshots_are_not_cacheable(client):
    state = start(client)
    assert read(client, state).headers['Cache-Control'] == 'no-store'


def test_default_start_keeps_human_first(client):
    result = client.post(BASE + '/start_game', json={})
    assert result.status_code == 201
    assert result.json['players'] == [{'type': 'human'}, {'type': 'negamax', 'depth': 2}]
    assert result.json['revision'] == 0 and result.json['currentPlayer'] == 0


@pytest.mark.parametrize('config', [{'type': 'random'}, {'type': 'negamax', 'depth': 8},
                                    {'type': 'mcts', 'simulation_limit': 800}])
@pytest.mark.parametrize('human_first', [True, False])
def test_both_orders_separate_atomic_opener_and_human_move(client, config, human_first):
    players = [{'type': 'human'}, config] if human_first else [config, {'type': 'human'}]
    result = client.post(BASE + '/start_game', json=dict(zip(('player1', 'player2'), players)))
    assert result.status_code == 201
    state = result.json
    assert state['players'] == players and state['revision'] == 0 and state['currentPlayer'] == 0
    assert all(piece == ' ' for row in state['board'] for piece in row)
    first = move(client, state, 3 if human_first else None)
    assert first.status_code == 200
    assert first.json['revision'] == 1 and first.json['currentPlayer'] == 1
    assert sum(piece == 'X' for row in first.json['board'] for piece in row) == 1
    assert move(client, state, 3 if human_first else None).json['code'] == 'stale_revision'
    second = move(client, first.json, None if human_first else 3)
    assert second.status_code == 200 and second.json['revision'] == 2
    assert sum(piece == 'O' for row in second.json['board'] for piece in row) == 1


def test_failed_ai_opener_keeps_revision_zero_and_recovers(client, monkeypatch):
    state = client.post(BASE + '/start_game', json={
        'player1': {'type': 'random'}, 'player2': {'type': 'human'}}).json
    with monkeypatch.context() as patch:
        patch.setattr(RandomAgent, 'choose_move', lambda self, game: -1)
        assert move(client, state).json['code'] == 'agent_failed'
        assert read(client, state).json == state
    assert move(client, state).json['revision'] == 1
