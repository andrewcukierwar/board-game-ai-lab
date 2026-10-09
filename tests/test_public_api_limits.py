"""Public Funnel protections: deterministic budgets and search contention."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from api.app import create_app
from api.public_limits import PublicApiLimiter, environment_config, validate_config
from games.connect4.agents.negamax_agent import NegamaxAgent

BASE = '/v1/connect4'


def test_two_atomic_token_buckets_and_refill():
    clock = [0.0]
    cfg = {**environment_config(), 'PUBLIC_REQUEST_RATE_PER_SECOND': 2.0,
           'PUBLIC_REQUEST_BURST': 3, 'PUBLIC_START_RATE_PER_SECOND': 0.5,
           'PUBLIC_START_BURST': 1}
    gate = PublicApiLimiter(cfg, clock=lambda: clock[0])
    assert gate.check('POST', BASE + '/start_game') is None
    assert gate.check('POST', BASE + '/start_game') == 2
    # Denied starts must not consume the general-request token.
    assert gate.check('GET', BASE + '/provenance') is None
    assert gate.check('GET', BASE + '/provenance') is None
    assert gate.check('GET', BASE + '/provenance') == 1
    # Monitoring and CORS preflight remain available under saturation.
    assert gate.check('GET', BASE + '/health') is None
    assert gate.check('OPTIONS', BASE + '/start_game') is None
    clock[0] = 2.0
    assert gate.check('POST', BASE + '/start_game') is None


def test_disabled_still_validates_but_does_not_meter():
    cfg = {**environment_config(), 'PUBLIC_RATE_LIMIT_ENABLED': False,
           'PUBLIC_REQUEST_BURST': 1, 'PUBLIC_START_BURST': 1}
    gate = PublicApiLimiter(cfg)
    for _ in range(10):
        assert gate.check('POST', BASE + '/start_game') is None


@pytest.mark.parametrize('setting,value', [
    ('PUBLIC_RATE_LIMIT_ENABLED', 'yes'),
    ('PUBLIC_REQUEST_RATE_PER_SECOND', 0),
    ('PUBLIC_REQUEST_RATE_PER_SECOND', float('nan')),
    ('PUBLIC_REQUEST_RATE_PER_SECOND', float('inf')),
    ('PUBLIC_START_RATE_PER_SECOND', -1),
    ('PUBLIC_REQUEST_BURST', 0),
    ('PUBLIC_START_BURST', True),
    ('PUBLIC_SEARCH_CONCURRENCY', 0),
    ('PUBLIC_SEARCH_CONCURRENCY', 4),
    ('PUBLIC_SEARCH_CONCURRENCY', 1.5),
])
def test_config_rejects_unsafe_values(setting, value):
    cfg = environment_config()
    cfg[setting] = value
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_flask_returns_structured_429_and_preserves_cors():
    app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False,
                      'CORS_ALLOWED_ORIGINS': 'https://board-game-ai-lab-ui.onrender.com',
                      'PUBLIC_REQUEST_RATE_PER_SECOND': 1,
                      'PUBLIC_REQUEST_BURST': 3,
                      'PUBLIC_START_RATE_PER_SECOND': 1,
                      'PUBLIC_START_BURST': 1})
    client = app.test_client()
    origin = {'Origin': 'https://board-game-ai-lab-ui.onrender.com'}
    first = client.post(BASE + '/start_game', json={}, headers=origin)
    assert first.status_code == 201
    denied = client.post(BASE + '/start_game', json={}, headers={
        **origin, 'X-Forwarded-For': '192.0.2.55'})
    assert denied.status_code == 429
    assert denied.json['code'] == 'rate_limited'
    assert int(denied.headers['Retry-After']) >= 1
    assert denied.headers['Cache-Control'] == 'no-store'
    assert denied.headers['Access-Control-Allow-Origin'] == origin['Origin']
    # Changing the spoofable header cannot evade the process-wide limit.
    denied2 = client.post(BASE + '/start_game', json={}, headers={
        **origin, 'X-Forwarded-For': '203.0.113.17'})
    assert denied2.status_code == 429
    assert len(app.extensions['connect4_games']._games) == 1
    assert client.get(BASE + '/health').status_code == 200
    preflight = client.options(BASE + '/start_game', headers={
        **origin, 'Access-Control-Request-Method': 'POST',
        'Access-Control-Request-Headers': 'Content-Type'})
    assert preflight.status_code == 200
    assert preflight.headers['Access-Control-Allow-Origin'] == origin['Origin']


def test_search_capacity_preserves_state_and_releases_after_busy(monkeypatch):
    app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False,
                      'PUBLIC_SEARCH_CONCURRENCY': 1})
    client = app.test_client()
    players = {'player1': {'type': 'negamax', 'depth': 2},
               'player2': {'type': 'human'}}
    first = client.post(BASE + '/start_game', json=players).json
    second = client.post(BASE + '/start_game', json=players).json
    entered, release = Event(), Event()

    def blocked(self, board):
        entered.set()
        assert release.wait(5)
        return 3

    monkeypatch.setattr(NegamaxAgent, 'choose_move', blocked)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(lambda: app.test_client().post(
            BASE + '/make_move', json={'game_id': first['game_id'], 'revision': 0}))
        try:
            assert entered.wait(5)
            rejected = client.post(BASE + '/make_move', json={
                'game_id': second['game_id'], 'revision': 0})
            assert rejected.status_code == 503
            assert rejected.json['code'] == 'agent_busy'
            assert client.get(BASE + '/games/' + second['game_id']).json == second
            # Reads and unrelated game creation remain responsive.
            assert client.post(BASE + '/start_game', json={}).status_code == 201
        finally:
            release.set()
        assert future.result().status_code == 200
    assert client.post(BASE + '/make_move', json={
        'game_id': second['game_id'], 'revision': 0}).status_code == 200


def test_search_exception_releases_capacity(monkeypatch):
    app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False,
                      'PUBLIC_SEARCH_CONCURRENCY': 1})
    client = app.test_client()
    game = client.post(BASE + '/start_game', json={
        'player1': {'type': 'negamax', 'depth': 2}, 'player2': {'type': 'human'}}).json
    move = {'game_id': game['game_id'], 'revision': 0}

    def fails(self, board):
        raise RuntimeError('private search failure')

    with monkeypatch.context() as patch:
        patch.setattr(NegamaxAgent, 'choose_move', fails)
        response = client.post(BASE + '/make_move', json=move)
        assert response.status_code == 503
        assert response.json['code'] == 'agent_failed'
    assert client.post(BASE + '/make_move', json=move).status_code == 200
