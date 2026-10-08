import pytest

from api.app import create_app
from api.connect4.provenance import public_provenance

EXPECTED = dict(api_provenance_version=1, api_contract_version=2,
                connect4_engine_version=1, agents=dict(random=2, negamax=2, mcts=2),
                stochastic_seed_version=1, source_commit=None)


def test_safe_read_only_manifest_and_no_cache(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'secret-must-never-appear')
    app = create_app({'TESTING': True, 'SOURCE_COMMIT': None,
                      'CORS_ALLOWED_ORIGINS': 'http://localhost:4176'})
    client = app.test_client()
    games = app.extensions['connect4_games']
    for _ in range(2):
        response = client.get('/v1/connect4/provenance', headers={'Origin': 'http://localhost:4176'})
        assert response.status_code == 200
        assert response.json == EXPECTED
        assert response.headers['Cache-Control'] == 'no-store'
        assert response.headers['Access-Control-Allow-Origin'] == 'http://localhost:4176'
        assert 'Origin' in response.headers['Vary']
        assert 'secret' not in response.get_data(as_text=True)
        assert games._games == {}
    assert client.post('/v1/connect4/provenance').status_code == 405


@pytest.mark.parametrize('commit', [None, '', 'latest', 'abc123', '/private/secret', 'A' * 40, 123, 'a' * 40, 'b' * 64])
def test_optional_commit_is_sanitized(commit):
    client = create_app({'TESTING': True, 'SOURCE_COMMIT': commit}).test_client()
    assert client.get('/v1/connect4/provenance').json == {
        **EXPECTED, 'source_commit': commit if commit in ('a' * 40, 'b' * 64) else None}


def test_configured_commit_and_history_share_manifest(monkeypatch):
    commit = 'a' * 40
    monkeypatch.setenv('EVALUATION_SOURCE_COMMIT', commit)
    client = create_app({'TESTING': True}).test_client()
    start = client.post('/v1/connect4/start_game', json={'player1': {'type': 'random'}, 'rng_seed': 42}).json
    before = client.get(f"/v1/connect4/games/{start['game_id']}").json
    history = client.get(f"/v1/connect4/games/{start['game_id']}/history").json
    assert history['provenance'] == public_provenance(commit)
    assert history['state'] == before == start
    assert history['moves'] == []
    assert client.get('/v1/connect4/provenance').json == history['provenance']
