"""Public MCTS contracts; injected searches isolate dispatch and contention."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from random import Random
from threading import Event

import pytest

from api.app import create_app
from api.connect4.evidence import get_explanation_context
from games.connect4.agents.mcts_agent import MCTSAgent
from games.connect4.grounding import build_explanation_context

BASE = '/v1/connect4'


@pytest.fixture
def app():
    return create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False})


def start(app, player=0, config=None):
    players = [{'type': 'human'}, {'type': 'human'}]
    players[player] = config or {'type': 'mcts', 'simulation_limit': 50}
    result = app.test_client().post(BASE + '/start_game', json=dict(zip(('player1', 'player2'), players)))
    assert result.status_code == 201
    return result.json


def move(app, state, column=None):
    body = {'game_id': state['game_id'], 'revision': state['revision']}
    if column is not None:
        body['column'] = column
    return app.test_client().post(BASE + '/make_move', json=body)


def read(app, state):
    return app.test_client().get(BASE + '/games/' + state['game_id']).json


def history(app, state):
    with app.extensions['connect4_games'].access(state['game_id']) as session:
        return session.history


@pytest.mark.parametrize('player', [0, 1])
@pytest.mark.parametrize('limit', [None, 50, 100, 250])
def test_presets_normalization_dispatch_and_detached_input(app, monkeypatch, player, limit):
    config = {'type': 'mcts', **({} if limit is None else {'simulation_limit': limit})}
    state = start(app, player, config)
    assert state['players'][player] == {'type': 'mcts', 'simulation_limit': limit or 100}
    if player == 1:
        state = move(app, state, 3).json
    def choose(self, game):
        assert self.simulation_limit == (limit or 100)
        game.make_move(0)  # Must not corrupt the board that will be committed.
        return 2
    monkeypatch.setattr(MCTSAgent, 'choose_move', choose)
    result = move(app, state)
    assert result.status_code == 200
    assert result.json['revision'] == state['revision'] + 1
    assert result.json['board'][5][0] == ' '
    assert result.json['board'][5][2] == ('X' if player == 0 else 'O')
    assert history(app, state)[-1].agent_type == 'mcts'


@pytest.mark.parametrize('config', [
    *[{'type': 'mcts', 'simulation_limit': v} for v in
      [True, False, 50.0, '100', None, 0, -50, 1, 51, 1000, [], {}]],
    {'type': 'mcts', 'depth': 2}, {'type': 'mcts', 'extra': 1},
    *[{'type': kind, 'simulation_limit': 100} for kind in ['human', 'random', 'negamax']],
])
def test_invalid_configuration_preserves_existing_session(app, config):
    state = start(app)
    result = app.test_client().post(BASE + '/start_game', json={
        'player2': config, 'replace_game_id': state['game_id']})
    assert result.status_code == 400 and result.json['code'] == 'invalid_agent'
    assert read(app, state) == state
    assert history(app, state) == ()


@pytest.mark.parametrize('failure', ['exception', 'constructor', -1, 7, True, 1.5, None])
def test_failures_are_atomic_and_release_reservation(app, monkeypatch, failure):
    state = start(app)
    def broken(self, game):
        game.make_move(0)
        if failure == 'exception':
            raise RuntimeError('private error')
        return failure
    with monkeypatch.context() as patch:
        if failure == 'constructor':
            def fail_init(self, *args):
                raise RuntimeError('private error')
            patch.setattr(MCTSAgent, '__init__', fail_init)
        else:
            patch.setattr(MCTSAgent, 'choose_move', broken)
        result = move(app, state)
    assert result.status_code == 503 and result.json['code'] == 'agent_failed'
    assert 'private error' not in result.json['error']
    assert read(app, state) == state and history(app, state) == ()
    monkeypatch.setattr(MCTSAgent, 'choose_move', lambda self, game: 2)
    assert move(app, state).status_code == 200
    assert len(history(app, state)) == 1


def test_process_wide_nonblocking_guard_and_other_agents_unaffected(app, monkeypatch):
    state, other = start(app), start(app)
    second_app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False})
    cross_app = start(second_app)
    entered, release = Event(), Event()
    def slow(self, game):
        entered.set()
        assert release.wait(5)
        return 3
    monkeypatch.setattr(MCTSAgent, 'choose_move', slow)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(move, app, state)
        try:
            assert entered.wait(5)
            assert move(app, state).json['code'] == 'game_busy'
            assert app.test_client().post(BASE + '/start_game', json={
                'replace_game_id': state['game_id']}).json['code'] == 'game_busy'
            for target, snapshot in [(app, other), (second_app, cross_app)]:
                rejected = move(target, snapshot)
                assert rejected.status_code == 503 and rejected.json['code'] == 'agent_busy'
                assert read(target, snapshot) == snapshot and history(target, snapshot) == ()
            for kind in ['human', 'random', 'negamax']:
                independent = start(app, config={'type': kind})
                assert move(app, independent, 0 if kind == 'human' else None).status_code == 200
        finally:
            release.set()
        assert future.result().status_code == 200
    assert move(app, state).json['code'] == 'stale_revision'
    assert len(history(app, state)) == 1
    assert move(second_app, cross_app).status_code == 200
    assert move(app, other).status_code == 200


def test_full_column_output_rejected_then_real_search_avoids_it(app, monkeypatch):
    state = start(app)
    with monkeypatch.context() as patch:
        patch.setattr(MCTSAgent, 'choose_move', lambda self, game: 0)
        for _ in range(3):
            state = move(app, state).json
            state = move(app, state, 0).json
        records = history(app, state)
        assert move(app, state).json['code'] == 'agent_failed'
    assert read(app, state) == state and history(app, state) == records
    result = move(app, state)
    assert result.status_code == 200
    assert history(app, state)[-1].column != 0


@pytest.mark.parametrize('player', [0, 1])
def test_complete_real_game_stale_terminal_and_deterministic_replay(app, monkeypatch, player):
    # Keep the real search, but seed its random choices for repeatable regression.
    original = MCTSAgent.__init__
    monkeypatch.setattr(MCTSAgent, '__init__', lambda self, limit: original(self, limit, rng=Random(42)))
    state = start(app, player)
    untouched = start(app)
    old = state
    for _ in range(42):
        result = move(app, state, None if state['currentPlayer'] == player else state['legalMoves'][0])
        assert result.status_code == 200
        state = result.json
        if state['gameOver']:
            break
    assert state['gameOver']
    assert read(app, untouched) == untouched and history(app, untouched) == ()
    records = history(app, state)
    assert move(app, old).json['code'] == 'stale_revision'
    assert move(app, state).json['code'] == 'game_over'
    assert read(app, state) == state and history(app, state) == records
    context = get_explanation_context(app.extensions['connect4_games'], state['game_id'], state['revision'])
    assert context == get_explanation_context(app.extensions['connect4_games'], state['game_id'], state['revision'])
    assert context['provenance']['history_verified_by_replay']
    assert context['provenance']['agent_reasoning_available'] is False
    assert all(r.agent_type == 'mcts' for r in records if r.player == player)
    corrupted = tuple(replace(r, agent_type='random') if r.player == player else r for r in records)
    with pytest.raises(ValueError, match='replay'):
        build_explanation_context(board=state['board'], player_to_move=state['currentPlayer'],
                                  revision=state['revision'], move_history=corrupted, players=state['players'])


def test_draw_with_mcts_history_replays_and_rejects_further_search(app, monkeypatch):
    sequence = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
                6, 6, 6, 6, 6]
    ai_moves = iter(sequence[::2])
    monkeypatch.setattr(MCTSAgent, 'choose_move', lambda self, game: next(ai_moves))
    state = start(app)
    for index, column in enumerate(sequence):
        result = move(app, state, None if index % 2 == 0 else column)
        assert result.status_code == 200
        state = result.json
    assert state['gameOver'] and state['winner'] == 'Draw'
    assert move(app, state).json['code'] == 'game_over'
    context = get_explanation_context(app.extensions['connect4_games'], state['game_id'], 42)
    assert len(context['move_history']) == 42
    assert context['position']['outcome']['status'] == 'draw'
