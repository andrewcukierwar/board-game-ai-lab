"""Read-only replay contract, using real immutable session history."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from api.app import create_app
from games.connect4.agents.random_agent import RandomAgent
from games.connect4.connect4 import Connect4

BASE = '/v1/connect4'
DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]

@pytest.fixture
def app():
    return create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False,
                       'CORS_ALLOWED_ORIGINS': 'https://board-game-ai-lab-ui.onrender.com'})


def start(app, players=None, **kwargs):
    players = players or [{'type': 'human'}] * 2
    return app.test_client().post(BASE + '/start_game', json={
        'player1': players[0], 'player2': players[1], **kwargs}).json


def history(app, state):
    return app.test_client().get(BASE + '/games/' + state['game_id'] + '/history')


@pytest.mark.parametrize('sequence', [[], [3, 2], [0, 1, 0, 1, 0, 1, 0],
                                     [0, 1, 0, 1, 2, 1, 2, 1], DRAW])
def test_atomic_ordered_detached_history_and_terminal_replay(app, sequence):
    state = start(app)
    initial = state
    for column in sequence:
        state = app.test_client().post(BASE + '/make_move', json={
            'game_id': state['game_id'], 'revision': state['revision'], 'column': column}).json
    response = history(app, state)
    assert response.status_code == 200
    assert response.headers['Cache-Control'] == 'no-store'
    data = response.json
    assert set(data) == {'game_id', 'revision', 'players', 'state', 'moves', 'provenance'}
    assert data['provenance'] == app.test_client().get(BASE + '/provenance').json
    assert data['state'] == state
    assert data['revision'] == len(data['moves']) == len(sequence)
    replay = Connect4()
    for i, record in enumerate(data['moves']):
        assert record['board_before'] == replay.board
        assert record['column'] == sequence[i] and record['player'] == i % 2
        assert record['revision_before'] == i and record['revision'] == record['move_number'] == i + 1
        assert replay.make_move(record['column'])
        assert record['board_after'] == replay.board
    assert replay.board == state['board']
    with app.extensions['connect4_games'].access(state['game_id']) as session:
        assert data['moves'] == [r.to_dict() for r in session.history]
        original = session.history
    data['state']['board'][5][0] = '?'
    data['players'][0]['type'] = 'changed'
    if data['moves']:
        data['moves'][0]['board_after'][5][sequence[0]] = '?'
    assert history(app, state).json['state'] == state
    with app.extensions['connect4_games'].access(state['game_id']) as session:
        assert session.history is original
    if not sequence:
        assert history(app, state).json['moves'] == [] and state == initial
    assert app.test_client().post(BASE + '/games/' + state['game_id'] + '/history', json={}).status_code == 405


@pytest.mark.parametrize('players', [
    [{'type': 'human'}, {'type': 'human'}],
    [{'type': 'human'}, {'type': 'random'}],
    [{'type': 'random'}, {'type': 'human'}],
    [{'type': 'negamax', 'depth': 6}, {'type': 'mcts', 'simulation_limit': 400}],
    [{'type': 'negamax', 'depth': 1}, {'type': 'negamax', 'depth': 8}],
    [{'type': 'mcts', 'simulation_limit': 100}, {'type': 'mcts', 'simulation_limit': 800}],
])
def test_configs_identity_and_no_opening_search(app, players):
    state = start(app, players)
    data = history(app, state).json
    assert data['revision'] == 0 and data['moves'] == []
    assert data['players'] == data['state']['players'] == players


def test_history_expiry_replacement_isolation_and_cors(app):
    state, other = start(app), start(app)
    response = app.test_client().get(BASE + '/games/' + state['game_id'] + '/history',
        headers={'Origin': 'https://board-game-ai-lab-ui.onrender.com'})
    assert response.headers['Access-Control-Allow-Origin'] == 'https://board-game-ai-lab-ui.onrender.com'
    start(app, replace_game_id=state['game_id'])
    assert history(app, state).json['code'] == 'session_not_found'
    assert history(app, other).json['moves'] == []
    store = app.extensions['connect4_games']
    store.clock = lambda: 0
    expiring = start(app)
    store.clock = lambda: store.ttl
    assert history(app, expiring).status_code == 404


def test_history_uses_existing_nonblocking_session_lock(app, monkeypatch):
    state = start(app, [{'type': 'random'}, {'type': 'random'}])
    entered, release = Event(), Event()
    def slow(self, game):
        entered.set()
        assert release.wait(5)
        return 3
    monkeypatch.setattr(RandomAgent, 'choose_move', slow)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: app.test_client().post(BASE + '/make_move', json={
            'game_id': state['game_id'], 'revision': 0}))
        try:
            assert entered.wait(5)
            assert history(app, state).json['code'] == 'game_busy'
        finally:
            release.set()
        assert future.result().status_code == 200
    data = history(app, state).json
    assert data['revision'] == len(data['moves']) == 1
    assert data['state']['board'] == data['moves'][0]['board_after']
