"""Phase 4D.4K-A: AlphaZero feature flag off (default) or unusable; existing API unchanged. Torch-free."""
import pytest

from api.app import create_app
from api.connect4.alphazero import environment_config

BASE = '/v1/connect4'
PROVENANCE = {'agents': {'mcts': 2, 'mcts_tactical': 1, 'negamax': 2, 'random': 2}, 'api_contract_version': 3,
              'api_provenance_version': 1, 'connect4_engine_version': 1, 'source_commit': None,
              'stochastic_seed_version': 1}


@pytest.fixture
def client():
    return create_app({'TESTING': True}).test_client()


def test_flag_defaults_off_and_parses_strictly(monkeypatch):
    monkeypatch.delenv('ALPHAZERO_ENABLED', raising=False)
    assert environment_config()['ALPHAZERO_ENABLED'] is False
    for raw, value in (('true', True), ('1', True), ('FALSE', False), ('0', False)):
        monkeypatch.setenv('ALPHAZERO_ENABLED', raw)
        assert environment_config()['ALPHAZERO_ENABLED'] is value
    for raw in ('yes', 'on', '', 'enabled'):
        monkeypatch.setenv('ALPHAZERO_ENABLED', raw)
        with pytest.raises(ValueError):
            environment_config()
    with pytest.raises(ValueError):
        create_app({'TESTING': True, 'ALPHAZERO_ENABLED': 'true'})  # never coerced from strings


@pytest.mark.parametrize('config', [{'type': 'alphazero'}, {'type': 'alphazero', 'tactical_guard': True},
                                    {'type': 'alphazero', 'tactical_guard': 'yes'}])
def test_disabled_alphazero_requests_get_explicit_error_and_no_game(client, config):
    old = client.post(BASE + '/start_game', json={}).json
    response = client.post(BASE + '/start_game', json={'player2': config, 'replace_game_id': old['game_id']})
    assert response.status_code == 400
    assert response.json == {'code': 'invalid_agent', 'error': 'The AlphaZero agent is not enabled on this server.'}
    assert client.get(BASE + '/games/' + old['game_id']).status_code == 200  # nothing replaced


def test_disabled_status_health_and_provenance(client):
    assert client.get(BASE + '/agents/alphazero').json == {
        'type': 'alphazero', 'enabled': False, 'available': False, 'code': 'alphazero_disabled'}
    assert client.get(BASE + '/health').data == b'OK'
    assert client.get(BASE + '/provenance').json == PROVENANCE


@pytest.mark.parametrize('opponent', [{'type': 'random'}, {'type': 'negamax', 'depth': 2},
                                      {'type': 'mcts', 'simulation_limit': 20},
                                      {'type': 'mcts_tactical', 'simulation_limit': 20}])
def test_existing_agents_unchanged_with_alphazero_disabled_or_enabled(tmp_path, opponent):
    for config in ({'TESTING': True},
                   {'TESTING': True, 'ALPHAZERO_ENABLED': True, 'ALPHAZERO_MANIFEST': str(tmp_path / 'none.json')}):
        client = create_app(config).test_client()
        game = client.post(BASE + '/start_game', json={'player1': opponent, 'player2': {'type': 'human'},
                                                       'rng_seed': 7}).json
        assert game['players'][0] == {'type': opponent['type'], **{k: v for k, v in opponent.items() if k != 'type'}}
        moved = client.post(BASE + '/make_move', json={'game_id': game['game_id'], 'revision': 0})
        assert moved.status_code == 200 and moved.json['revision'] == 1
        history = client.get(BASE + f"/games/{game['game_id']}/history").json
        assert 'alphazero' not in history['moves'][0] and 'tactical_guard' not in history['moves'][0]['agent']
        assert history['provenance'] == PROVENANCE


def test_enabled_but_unusable_reports_unavailable_without_details(tmp_path):
    app = create_app({'TESTING': True, 'ALPHAZERO_ENABLED': True, 'ALPHAZERO_MANIFEST': str(tmp_path / 'none.json')})
    client = app.test_client()
    status = client.get(BASE + '/agents/alphazero').json
    assert status == {'type': 'alphazero', 'enabled': True, 'available': False, 'code': 'alphazero_unavailable'}
    response = client.post(BASE + '/start_game', json={'player2': {'type': 'alphazero'}})
    assert response.status_code == 503 and response.json['code'] == 'agent_unavailable'
    assert str(tmp_path) not in response.get_data(as_text=True)
    assert app.extensions['connect4_alphazero'].failure in ('runtime_unavailable', 'artifact_missing')
    assert client.get(BASE + '/health').data == b'OK'
    assert client.get(BASE + '/provenance').json == PROVENANCE
