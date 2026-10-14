"""Phase 4D.4K-A: AlphaZero through the public Connect 4 API with the flag enabled (torch)."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

torch = pytest.importorskip('torch')

from api.app import create_app  # noqa: E402
import api.connect4 as connect4_api  # noqa: E402
from games.connect4.alphazero_v2.production import ProductionModel  # noqa: E402

BASE = '/v1/connect4'
PROVENANCE_KEYS = {'agents', 'api_contract_version', 'api_provenance_version', 'connect4_engine_version',
                   'source_commit', 'stochastic_seed_version'}


@pytest.fixture
def app():
    return create_app({'TESTING': True, 'ALPHAZERO_ENABLED': True, 'PUBLIC_RATE_LIMIT_ENABLED': False})


@pytest.fixture
def client(app):
    return app.test_client()


def start(client, player1, player2, **extra):
    response = client.post(BASE + '/start_game', json={'player1': player1, 'player2': player2, **extra})
    assert response.status_code == 201, response.json
    return response.json


def move(client, state, column=None):
    data = {'game_id': state['game_id'], 'revision': state['revision']}
    if column is not None:
        data['column'] = column
    return client.post(BASE + '/make_move', json=data)


def history(client, state):
    return client.get(BASE + f"/games/{state['game_id']}/history").json


def play_out(client, state):
    while not state['gameOver']:
        response = move(client, state)
        assert response.status_code == 200, response.json
        state = response.json
    return state


def winning_columns(board, piece):
    """Independent matrix scan (no bitboards): columns whose drop completes four for ``piece``."""
    wins = []
    for col in range(7):
        rows = [r for r in range(6) if board[r][col] == ' ']
        if not rows:
            continue
        r = rows[-1]
        grid = [row[:] for row in board]
        grid[r][col] = piece
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            count = 1
            for sign in (1, -1):
                rr, cc = r + sign * dr, col + sign * dc
                while 0 <= rr < 6 and 0 <= cc < 7 and grid[rr][cc] == piece:
                    count, rr, cc = count + 1, rr + sign * dr, cc + sign * dc
            if count >= 4:
                wins.append(col)
                break
    return wins


def drop(board, col, piece):
    grid = [row[:] for row in board]
    grid[max(r for r in range(6) if grid[r][col] == ' ')][col] = piece
    return grid


def test_status_reports_loaded_model_and_fixed_settings(client):
    status = client.get(BASE + '/agents/alphazero').json
    assert status['available'] is True and status['enabled'] is True and status['agent_version'] == 1
    assert status['model']['id'] == 'az2-hc-g6-cnn-s42-v1'
    assert status['model']['source_checkpoint_sha256'] == 'ba2bde2ec71b3ac7486046d1cc26ab591bb59d6b764193ece08c6d6a6f24f58b'
    assert status['search'] == {'algorithm': 'puct', 'simulations': 256, 'exploration': 1.41, 'root_noise': False,
                                'final_selection': 'maximum-visit', 'deterministic': True}
    assert status['options']['tactical_guard']['default'] is False
    text = client.get(BASE + '/agents/alphazero').get_data(as_text=True)
    assert '/Users' not in text and 'models/' not in text and 'runtime' not in text


def test_provenance_manifest_is_unchanged_when_enabled(client):
    provenance = client.get(BASE + '/provenance').json
    assert set(provenance) == PROVENANCE_KEYS and 'alphazero' not in provenance['agents']


@pytest.mark.parametrize('config,stored', [
    ({'type': 'alphazero'}, {'type': 'alphazero', 'tactical_guard': False}),
    ({'type': 'alphazero', 'tactical_guard': False}, {'type': 'alphazero', 'tactical_guard': False}),
    ({'type': 'alphazero', 'tactical_guard': True}, {'type': 'alphazero', 'tactical_guard': True}),
])
def test_valid_configurations_are_canonical(client, config, stored):
    assert start(client, {'type': 'human'}, config)['players'][1] == stored


@pytest.mark.parametrize('config', [
    {'type': 'alphazero', 'tactical_guard': 'true'}, {'type': 'alphazero', 'tactical_guard': 1},
    {'type': 'alphazero', 'tactical_guard': 0}, {'type': 'alphazero', 'tactical_guard': None},
    {'type': 'alphazero', 'tactical_guard': [True]}, {'type': 'alphazero', 'simulations': 256},
    {'type': 'alphazero', 'simulation_limit': 512}, {'type': 'alphazero', 'depth': 4},
    {'type': 'alphazero', 'temperature': 0}, {'type': 'alphazero', 'tactical_guard': True, 'model': 'x'},
    {'type': 'AlphaZero'}, {'type': 'alphazero_guarded'},
])
def test_invalid_configurations_are_rejected_without_replacing_games(client, config):
    old = start(client, {'type': 'human'}, {'type': 'human'})
    response = client.post(BASE + '/start_game', json={'player2': config, 'replace_game_id': old['game_id']})
    assert response.status_code == 400 and response.json['code'] == 'invalid_agent'
    assert client.get(BASE + '/games/' + old['game_id']).status_code == 200


@pytest.mark.parametrize('guard', [False, True])
def test_alphazero_first_and_second_player(client, guard):
    first = start(client, {'type': 'alphazero', 'tactical_guard': guard}, {'type': 'human'})
    after = move(client, first)
    assert after.status_code == 200 and after.json['revision'] == 1 and after.json['currentPlayer'] == 1
    second = start(client, {'type': 'human'}, {'type': 'alphazero', 'tactical_guard': guard})
    human = move(client, second, 0).json
    reply = move(client, human)
    assert reply.status_code == 200 and reply.json['currentPlayer'] == 0
    record = history(client, reply.json)['moves'][1]
    assert record['player'] == 1 and record['agent'] == {'type': 'alphazero', 'tactical_guard': guard}
    evidence = record['alphazero']
    assert evidence['simulations'] == 256 and sum(evidence['root_visits']) == 256
    assert evidence['tactical_guard'] is guard and (evidence['guard_reason'] is None) is (not guard)
    assert evidence['original_column'] == record['column'] or guard


def test_human_column_rejected_on_alphazero_turn_and_full_columns(client):
    state = start(client, {'type': 'human'}, {'type': 'alphazero', 'tactical_guard': True})
    columns_played = []
    while not state['gameOver'] and len(columns_played) < 12:
        if state['currentPlayer'] == 1:
            assert move(client, state, 3).json['code'] == 'invalid_move'
            response = move(client, state)
        else:
            column = 3 if 3 in state['legalMoves'] else state['legalMoves'][0]
            response = move(client, state, column)
        assert response.status_code == 200
        state = response.json
        columns_played.append(state['board'])
    for record in history(client, state)['moves']:
        assert record['board_before'][0][record['column']] == ' '  # never into a full column


def test_guarded_moves_obey_guard_rules_against_random(client):
    """L/M/N through the API: every guarded move re-checked by an independent rules scan."""
    reasons = set()
    for seed in range(6):
        state = start(client, {'type': 'alphazero', 'tactical_guard': True}, {'type': 'random'}, rng_seed=seed)
        state = play_out(client, state)
        for record in history(client, state)['moves']:
            if record['agent']['type'] != 'alphazero':
                continue
            board, mine, theirs = record['board_before'], 'X', 'O'
            reasons.add(record['alphazero']['guard_reason'])
            wins = winning_columns(board, mine)
            legal = [c for c in range(7) if board[0][c] == ' ']
            safe = [c for c in legal if c in wins or not winning_columns(drop(board, c, mine), theirs)]
            if wins:
                assert record['column'] in wins
            elif safe:
                assert record['column'] in safe
            else:
                assert record['column'] == record['alphazero']['original_column']
    assert 'IMMEDIATE_WIN' in reasons


@pytest.mark.parametrize('guards', [(False, False), (True, False), (True, True)])
def test_both_sides_alphazero_complete_game_and_history(client, guards):
    state = start(client, *({'type': 'alphazero', 'tactical_guard': g} for g in guards))
    final = play_out(client, state)
    record = history(client, final)
    moves = record['moves']
    assert len(moves) == final['revision'] and record['state'] == final
    board = [[' '] * 7 for _ in range(6)]
    for index, entry in enumerate(moves):
        assert entry['player'] == index % 2 and entry['board_before'] == board
        assert entry['agent'] == {'type': 'alphazero', 'tactical_guard': guards[index % 2]}
        board = drop(board, entry['column'], 'XO'[index % 2])
        assert entry['board_after'] == board
    assert final['board'] == board and moves[-1]['outcome']['status'] in ('win', 'draw')
    assert final['players'] == [{'type': 'alphazero', 'tactical_guard': g} for g in guards]


def test_deterministic_replay_and_lab_style_seeded_starts(client):
    """Match/Tournament/Season labs start seeded games and replace sessions; AlphaZero is deterministic."""
    players = ({'type': 'alphazero', 'tactical_guard': True}, {'type': 'alphazero'})
    first = play_out(client, start(client, *players, rng_seed=11))
    second = play_out(client, start(client, *players, rng_seed=12, replace_game_id=first['game_id']))
    assert client.get(BASE + '/games/' + first['game_id']).status_code == 404
    assert history(client, second)['rng_seed'] == 12
    third = play_out(client, start(client, *players))
    assert second['board'] == third['board']


def blocking_decide(monkeypatch):
    entered, release, calls = Event(), Event(), []
    original = ProductionModel.decide

    def decide(self, game, *, tactical_guard, rng=None):
        calls.append(game.current_player)
        entered.set()
        assert release.wait(10)
        return original(self, game, tactical_guard=tactical_guard, rng=rng)
    monkeypatch.setattr(ProductionModel, 'decide', decide)
    return entered, release, calls


def test_concurrent_alphazero_searches_are_bounded_and_never_partially_commit(app, monkeypatch):
    client = app.test_client()
    games = [start(client, {'type': 'alphazero'}, {'type': 'human'}) for _ in range(3)]
    entered, release, _ = blocking_decide(monkeypatch)
    with ThreadPoolExecutor(max_workers=2) as pool:
        running = pool.submit(lambda: app.test_client().post(BASE + '/make_move', json={
            'game_id': games[0]['game_id'], 'revision': 0}))
        assert entered.wait(10)
        busy = move(client, games[1])
        same_game = move(client, games[0])
        release.set()
        done = running.result(10)
    assert busy.status_code == 503 and busy.json['code'] == 'agent_busy' and 'AlphaZero' in busy.json['error']
    assert same_game.status_code == 409 and same_game.json['code'] == 'game_busy'
    assert done.status_code == 200 and done.json['revision'] == 1
    assert client.get(BASE + '/games/' + games[1]['game_id']).json['revision'] == 0
    assert history(client, games[1])['moves'] == []
    assert move(client, games[1]).status_code == 200  # reservation released
    assert move(client, games[2]).status_code == 200


def test_shared_search_capacity_exhaustion(app):
    client = app.test_client()
    game = start(client, {'type': 'alphazero'}, {'type': 'human'})
    gate = app.extensions['connect4_search_gate']
    held = 0
    while gate.acquire(blocking=False):
        held += 1
    try:
        response = move(client, game)
        assert response.status_code == 503 and response.json['code'] == 'agent_busy'
        assert 'Too many AI searches' in response.json['error']
        assert client.get(BASE + '/games/' + game['game_id']).json['revision'] == 0
    finally:
        for _ in range(held):
            gate.release()
    assert connect4_api._alphazero_reservation.acquire(blocking=False)
    connect4_api._alphazero_reservation.release()
    assert move(client, game).status_code == 200


def test_stale_revision_never_searches(client, monkeypatch):
    game = start(client, {'type': 'alphazero'}, {'type': 'human'})
    calls = []
    monkeypatch.setattr(ProductionModel, 'decide', lambda *a, **k: calls.append(1))
    response = client.post(BASE + '/make_move', json={'game_id': game['game_id'], 'revision': 5})
    assert response.status_code == 409 and response.json['code'] == 'stale_revision' and calls == []


@pytest.mark.parametrize('failure', [RuntimeError('search exploded'), ValueError('bad'), MemoryError()])
def test_search_exception_releases_everything_and_commits_nothing(app, monkeypatch, failure):
    client = app.test_client()
    game = start(client, {'type': 'alphazero', 'tactical_guard': True}, {'type': 'human'})

    def explode(*args, **kwargs):
        raise failure
    monkeypatch.setattr(ProductionModel, 'decide', explode)
    response = move(client, game)
    assert response.status_code == 503 and response.json['code'] == 'agent_failed'
    assert 'exploded' not in response.get_data(as_text=True) and 'Traceback' not in response.get_data(as_text=True)
    assert client.get(BASE + '/games/' + game['game_id']).json['revision'] == 0
    assert history(client, game)['moves'] == []
    monkeypatch.undo()
    gate = app.extensions['connect4_search_gate']
    assert all(gate.acquire(blocking=False) for _ in range(app.config['PUBLIC_SEARCH_CONCURRENCY']))
    for _ in range(app.config['PUBLIC_SEARCH_CONCURRENCY']):
        gate.release()
    assert move(client, game).status_code == 200


def test_illegal_agent_output_is_rejected_without_commit(app, monkeypatch):
    client = app.test_client()
    game = start(client, {'type': 'alphazero'}, {'type': 'human'})
    from games.connect4.agents import alphazero_agent
    monkeypatch.setattr(alphazero_agent.AlphaZeroAgent, 'choose_move', lambda self, game: 9)
    response = move(client, game)
    assert response.status_code == 503 and response.json['code'] == 'agent_failed'
    assert client.get(BASE + '/games/' + game['game_id']).json['revision'] == 0


def test_repeated_games_reuse_one_model(app):
    service = app.extensions['connect4_alphazero']
    model = service.model
    client = app.test_client()
    for _ in range(3):
        play_out(client, start(client, {'type': 'alphazero', 'tactical_guard': True}, {'type': 'random'}))
    assert create_app({'TESTING': True, 'ALPHAZERO_ENABLED': True}).extensions['connect4_alphazero'].model is model


@pytest.mark.parametrize('preset,expected', [(None, '1'), ('2', '2')])
def test_enabled_service_bounds_openmp_before_torch_import(preset, expected):
    import os
    import subprocess
    import sys
    code = ('import os, sys; from api.app import create_app; '
            'assert "torch" not in sys.modules; '
            'app = create_app({"TESTING": True, "ALPHAZERO_ENABLED": True}); '
            'assert app.extensions["connect4_alphazero"].available; '
            'print(os.environ["OMP_NUM_THREADS"], os.environ["MKL_NUM_THREADS"])')
    env = {k: v for k, v in os.environ.items() if k not in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS')}
    env.update(PYTHONDONTWRITEBYTECODE='1', **({'OMP_NUM_THREADS': preset} if preset else {}))
    root = __import__('pathlib').Path(__file__).resolve().parents[1]
    done = subprocess.run([sys.executable, '-c', code], cwd=root, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == [expected, '1']


def test_versioned_model_provenance_status_and_history(client):
    status = client.get(BASE + '/agents/alphazero').json
    p = status['provenance']
    assert p['version'] == 1 and p['type'] == 'alphazero' and p['agent_version'] == 1
    assert p['model'] == {k: status['model'][k] for k in p['model']}
    assert p['search'] == status['search']
    assert p['tie_break'] == 'connect4-alphazero-production-tie-break-v1'
    assert p['rng_seed'] == 'ignored'
    assert p['tactical_guard']['implementation'] == 'connect4-tactical-root-guard-v1'
    assert p['tactical_guard']['source_sha256'] == 'd2ee54ba1fe9ddbcc4641538e2f5dd89a2435234ff194d6909bed72096632d04'
    for guard in (False, True):
        state = start(client, dict(type='alphazero', tactical_guard=guard), dict(type='human'))
        assert history(client, state)['alphazero_provenance'] == p
        state = move(client, state).json
        evidence = history(client, state)
        assert evidence['alphazero_provenance'] == p
        assert evidence['moves'][0]['agent']['tactical_guard'] is guard
    legacy = start(client, dict(type='random'), dict(type='human'))
    assert 'alphazero_provenance' not in history(client, legacy)
    assert set(client.get(BASE + '/provenance').json) == PROVENANCE_KEYS


def test_pinned_history_survives_unavailable_model_and_blocks_identity_change(app, monkeypatch):
    client = app.test_client()
    service = app.extensions['connect4_alphazero']
    state = start(client, dict(type='alphazero', tactical_guard=True), dict(type='human'))
    pinned = history(client, state)['alphazero_provenance']
    model = service.model
    monkeypatch.setattr(service, 'model', None)
    assert move(client, state).json['code'] == 'agent_unavailable'
    assert history(client, state)['alphazero_provenance'] == pinned
    assert history(client, state)['revision'] == 0
    monkeypatch.setattr(service, 'model', model)
    original_status = service.public_status
    changed = original_status()
    changed['provenance']['model']['id'] = 'other-model'
    monkeypatch.setattr(service, 'public_status', lambda: changed)
    response = move(client, state)
    assert response.status_code == 503 and response.json['code'] == 'agent_unavailable'
    assert history(client, state)['alphazero_provenance'] == pinned
    assert history(client, state)['moves'] == []
    monkeypatch.setattr(service, 'public_status', original_status)
    assert move(client, state).status_code == 200
