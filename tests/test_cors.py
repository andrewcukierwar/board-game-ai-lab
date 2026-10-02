import pytest

from api.app import create_app

FRONTEND = 'https://board-game-ai-lab-ui.onrender.com'
BASE = '/v1/connect4'


@pytest.fixture
def client():
    return create_app({'TESTING': True, 'CORS_ALLOWED_ORIGINS': FRONTEND}).test_client()


def test_json_preflight_and_gameplay(client):
    headers = {'Origin': FRONTEND, 'Access-Control-Request-Method': 'POST',
               'Access-Control-Request-Headers': 'content-type'}
    response = client.options(BASE + '/start_game', headers=headers)
    assert response.status_code == 200
    assert response.headers['Access-Control-Allow-Origin'] == FRONTEND
    assert response.headers['Access-Control-Allow-Methods'] == 'GET, POST, OPTIONS'
    assert response.headers['Access-Control-Allow-Headers'] == 'Content-Type'
    assert 'Access-Control-Allow-Credentials' not in response.headers
    assert 'Origin' in response.headers['Vary']
    state = client.post(BASE + '/start_game', json={}, headers={'Origin': FRONTEND})
    assert state.status_code == 201
    assert state.headers['Access-Control-Allow-Origin'] == FRONTEND
    response = client.get(BASE + '/games/' + state.json['game_id'], headers={'Origin': FRONTEND})
    assert response.json == state.json
    assert response.headers['Access-Control-Allow-Origin'] == FRONTEND
    assert response.headers['Cache-Control'] == 'no-store'


@pytest.mark.parametrize('origin', ['https://other.onrender.com', 'null',
    FRONTEND + '.evil.example', 'http://board-game-ai-lab-ui.onrender.com',
    'https://board-game-ai-lab-ui.onrender.com:8443'])
def test_unlisted_origins_cannot_read_or_preflight(client, origin):
    for response in [client.get(BASE + '/health', headers={'Origin': origin}),
                     client.options(BASE + '/make_move', headers={'Origin': origin,
                         'Access-Control-Request-Method': 'POST'})]:
        assert 'Access-Control-Allow-Origin' not in response.headers
        assert 'Origin' in response.headers['Vary']


def test_error_responses_and_health_include_cors(client):
    headers = {'Origin': FRONTEND}
    for response, status in [
        (client.get(BASE + '/health', headers=headers), 200),
        (client.get(BASE + '/games/missing', headers=headers), 404),
        (client.post(BASE + '/make_move', json={}, headers=headers), 400),
        (client.post(BASE + '/start_game', json={'x': 'x' * 5000}, headers=headers), 413),
    ]:
        assert response.status_code == status
        assert response.headers['Access-Control-Allow-Origin'] == FRONTEND
    assert client.get(BASE + '/health').data == b'OK'


def test_default_policy_keeps_same_origin_proxy_functional(monkeypatch):
    monkeypatch.delenv('CORS_ALLOWED_ORIGINS', raising=False)
    client = create_app({'TESTING': True}).test_client()
    response = client.post(BASE + '/start_game', json={}, headers={'Origin': 'http://localhost:3000'})
    assert response.status_code == 201
    assert 'Access-Control-Allow-Origin' not in response.headers


def test_environment_allowlist_and_multiple_explicit_origins(monkeypatch):
    monkeypatch.setenv('CORS_ALLOWED_ORIGINS', f' {FRONTEND}, https://games.example.com ')
    client = create_app({'TESTING': True}).test_client()
    for origin in [FRONTEND, 'https://games.example.com']:
        assert client.get(BASE + '/health', headers={'Origin': origin}).headers['Access-Control-Allow-Origin'] == origin
    assert 'Access-Control-Allow-Origin' not in client.get('/missing', headers={'Origin': FRONTEND}).headers


@pytest.mark.parametrize('origin', ['*', 'https://*.onrender.com', 'null',
    FRONTEND + '/', FRONTEND + '/connect4', FRONTEND + '?query=1',
    FRONTEND + '#fragment', 'https://user:password@example.com', 'https://example.com:bad',
    'https://bad host.example'])
def test_invalid_cors_configuration_fails_at_startup(origin):
    with pytest.raises(ValueError):
        create_app({'CORS_ALLOWED_ORIGINS': origin})
