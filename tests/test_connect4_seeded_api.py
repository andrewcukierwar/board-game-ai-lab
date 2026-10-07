"""Reproducible seeded execution, including explicit restart from revision zero."""
import random
import pytest
from api.app import create_app
from api.connect4 import ply_seed

BASE = '/v1/connect4'

@pytest.fixture
def app():
    return create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False})

@pytest.mark.parametrize('seed', [0, 1, 4294967295])
def test_seed_provenance(app, seed):
    response = app.test_client().post(BASE + '/start_game', json={'rng_seed': seed})
    assert response.status_code == 201
    gid = response.json['game_id']
    with app.extensions['connect4_games'].access(gid) as session:
        assert session.rng_seed == seed
    assert app.test_client().get(BASE + '/games/' + gid + '/history').json['rng_seed'] == seed

@pytest.mark.parametrize('seed', [True, False, None, 1.0, 0.5, -1, 4294967296, '123', [], {}])
def test_invalid_seed(app, seed):
    response = app.test_client().post(BASE + '/start_game', json={'rng_seed': seed})
    assert response.status_code == 400 and response.json['code'] == 'invalid_rng_seed'
    assert not app.extensions['connect4_games']._games


def test_unseeded_contract_unchanged(app):
    state = app.test_client().post(BASE + '/start_game', json={}).json
    history = app.test_client().get(BASE + '/games/' + state['game_id'] + '/history').json
    assert 'rng_seed' not in history
    with app.extensions['connect4_games'].access(state['game_id']) as session:
        assert session.rng_seed is None


def play(app, players, seed, replace=None, stop=None):
    client = app.test_client()
    state = client.post(BASE + '/start_game', json={'player1': players[0], 'player2': players[1],
        'rng_seed': seed, **({'replace_game_id': replace} if replace else {})}).json
    while not state['gameOver'] and (stop is None or state['revision'] < stop):
        response = client.post(BASE + '/make_move', json={'game_id': state['game_id'], 'revision': state['revision']})
        assert response.status_code == 200
        state = response.json
    history = client.get(BASE + '/games/' + state['game_id'] + '/history').json
    return state, history

@pytest.mark.parametrize('players', [
    [{'type': 'random'}, {'type': 'random'}],
    [{'type': 'mcts', 'simulation_limit': 100}, {'type': 'random'}],
    [{'type': 'mcts', 'simulation_limit': 100}, {'type': 'mcts', 'simulation_limit': 100}],
])
def test_complete_seeded_games_and_restart_are_reproducible_without_global_rng(app, players):
    before = random.getstate()
    first, h1 = play(app, players, 123456789)
    second, h2 = play(app, players, 123456789, first['game_id'])
    assert first['gameOver'] and second['gameOver']
    assert first['board'] == second['board'] and first['winner'] == second['winner']
    assert h1['moves'] == h2['moves']
    interrupted, partial = play(app, players, 123456789, second['game_id'], stop=5)
    restarted, replay = play(app, players, 123456789, interrupted['game_id'])
    assert partial['moves'] == replay['moves'][:5]
    assert replay['moves'] == h1['moves']
    assert random.getstate() == before
    other, different = play(app, players, 987654321, restarted['game_id'])
    assert other['gameOver'] and different['rng_seed'] == 987654321
    assert random.getstate() == before
    assert len(app.extensions['connect4_games']._games) == 1


def test_ply_seed_stable_and_independent():
    assert ply_seed(1234, 0, 0) == ply_seed(1234, 0, 0)
    assert len({ply_seed(1234, i, i % 2) for i in range(42)}) == 42
    assert all(0 <= ply_seed(4294967295, i, i % 2) <= 4294967295 for i in range(42))

@pytest.mark.parametrize('ai', [{'type': 'random'}, {'type': 'mcts', 'simulation_limit': 100}])
@pytest.mark.parametrize('human_index', [0, 1])
def test_seeded_ai_reproduces_for_recorded_human_columns(app, ai, human_index):
    """The seed determines AI choices given explicit Human decisions, in either color."""
    client = app.test_client()
    players = [dict(ai), dict(ai)]
    players[human_index] = {'type': 'human'}
    before = random.getstate()
    human_columns = []

    def run(replay=False, replace=None):
        response = client.post(BASE + '/start_game', json={
            'player1': players[0], 'player2': players[1], 'rng_seed': 1234,
            **({'replace_game_id': replace} if replace else {})})
        assert response.status_code == 201
        state = response.json
        assert state['revision'] == 0
        human_ply = 0
        while not state['gameOver']:
            payload = {'game_id': state['game_id'], 'revision': state['revision']}
            if state['currentPlayer'] == human_index:
                column = (human_columns[human_ply] if replay else
                          next(c for c in [3, 2, 4, 1, 5, 0, 6] if c in state['legalMoves']))
                assert column in state['legalMoves']
                payload['column'] = column
                if not replay:
                    human_columns.append(column)
                human_ply += 1
            moved = client.post(BASE + '/make_move', json=payload)
            assert moved.status_code == 200
            assert moved.json['revision'] == state['revision'] + 1
            state = moved.json
        history = client.get(BASE + '/games/' + state['game_id'] + '/history').json
        return state, history

    first, h1 = run()
    second, h2 = run(replay=True, replace=first['game_id'])
    assert human_columns
    assert first['board'] == second['board'] and first['winner'] == second['winner']
    assert h1['moves'] == h2['moves']
    assert h2['rng_seed'] == 1234
    assert random.getstate() == before
    assert client.get(BASE + '/games/' + first['game_id'] + '/history').status_code == 404
