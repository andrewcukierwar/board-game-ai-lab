"""Opt-in Victor research agent: disabled by default, bounded, always legal."""
from random import Random
from time import perf_counter

import pytest

import games.connect4.agents.victor_research_agent as research_module
from api.app import create_app, victor_research_enabled
from api.connect4 import _victor_reservation
from games.connect4.agents.victor_research_agent import (
    LABEL, PUBLIC_BUDGET, VictorResearchAgent,
)
from games.connect4.connect4 import Connect4
from games.connect4.victor import SolverBudget

BASE = '/v1/connect4'
RESEARCH_FIELDS = ('move_kind', 'exact_value', 'bound', 'certificate', 'kind', 'research')


def game(history=()):
    g = Connect4()
    for c in history:
        assert g.make_move(c)
    return g


def walk(value):
    if isinstance(value, dict):
        for k, v in value.items():
            yield k
            yield from walk(v)
    elif isinstance(value, list):
        for v in value:
            yield from walk(v)


@pytest.fixture
def enabled():
    return create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False,
                       'VICTOR_RESEARCH_ENABLED': True})


def start(app, research_player=1):
    players = [{'type': 'human'}, {'type': 'human'}]
    players[research_player] = {'type': 'victor_research'}
    return app.test_client().post(BASE + '/start_game',
                                  json=dict(zip(('player1', 'player2'), players)))


def move(app, state, column=None):
    body = {'game_id': state['game_id'], 'revision': state['revision']}
    if column is not None:
        body['column'] = column
    return app.test_client().post(BASE + '/make_move', json=body)


def test_label_and_budget_are_explicit():
    assert 'not perfect' in LABEL and str(VictorResearchAgent()) == LABEL
    assert PUBLIC_BUDGET.deadline is not None and PUBLIC_BUDGET.exact.seconds is not None
    with pytest.raises(ValueError):
        VictorResearchAgent(SolverBudget())  # No deadline: not acceptable for public play.


def test_factory_keeps_legacy_victor_distinct():
    pytest.importorskip('torch')  # The legacy factory imports neural agents.
    from games.connect4.agents.agent_factory import create_agent
    from games.connect4.agents.victor_agent import VictorAgent
    assert type(create_agent({'type': 'victor_research'})) is VictorResearchAgent
    assert type(create_agent({'type': 'victor'})) is VictorAgent


@pytest.mark.parametrize('history', [(), (3,), (3, 3, 2, 4), (0, 1, 0, 1, 0),
                                     (3, 2, 3, 3, 3, 4, 4, 2, 2, 3)])
def test_moves_are_legal_bounded_and_labelled(history):
    g = game(history)
    agent = VictorResearchAgent()
    started = perf_counter()
    column = agent.choose_move(g)
    assert column in g.get_valid_moves()
    assert perf_counter() - started < PUBLIC_BUDGET.deadline + 2.0  # Generous CI margin.
    assert agent.last_decision['move'] == column and agent.last_decision['kind']
    if history == (0, 1, 0, 1, 0):
        assert column == 0 and agent.last_decision['kind'] in ('exact', 'forced_defense',
                                                               'opening_book')


def test_complete_game_against_random_is_legal():
    rng = Random(4)
    agent = VictorResearchAgent()
    g = game()
    while not g.is_game_over():
        column = (agent.choose_move(g) if g.current_player == 1
                  else rng.choice(g.get_valid_moves()))
        assert g.make_move(column)


def test_solver_failures_fall_back_to_legal_heuristic_moves(monkeypatch):
    g = game((3, 3, 3, 3, 3, 3))  # Column 3 is full.
    def broken(*args, **kwargs):
        raise RuntimeError('simulated solver defect')
    monkeypatch.setattr(research_module, 'analyze_position', broken)
    agent = VictorResearchAgent()
    assert agent.choose_move(g) in g.get_valid_moves()
    assert agent.last_decision['kind'] == 'fallback'
    assert 'RuntimeError' in agent.last_decision['failure']

    class Illegal:
        move = 3
    monkeypatch.setattr(research_module, 'analyze_position', lambda *a, **k: Illegal())
    monkeypatch.setattr(research_module.NegamaxAgent, 'choose_move', broken)
    assert agent.choose_move(g) == g.get_valid_moves()[0]
    assert agent.last_decision['failure'] == 'solver returned no legal move'


def test_terminal_positions_raise():
    with pytest.raises(ValueError):
        VictorResearchAgent().choose_move(game((0, 1, 0, 1, 0, 1, 0)))


@pytest.mark.parametrize('value,expected', [(None, False), ('false', False), ('0', False),
                                            ('true', True), ('TRUE', True), ('1', True)])
def test_environment_flag_is_strict_and_off_by_default(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv('VICTOR_RESEARCH_ENABLED', raising=False)
    else:
        monkeypatch.setenv('VICTOR_RESEARCH_ENABLED', value)
    assert victor_research_enabled() is expected


def test_invalid_environment_flag_is_rejected(monkeypatch):
    monkeypatch.setenv('VICTOR_RESEARCH_ENABLED', 'yes')
    with pytest.raises(ValueError):
        victor_research_enabled()


def test_disabled_by_default_with_unchanged_public_error(monkeypatch):
    monkeypatch.delenv('VICTOR_RESEARCH_ENABLED', raising=False)
    app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False})
    assert app.config['VICTOR_RESEARCH_ENABLED'] is False
    result = start(app)
    assert result.status_code == 400 and result.json == {
        'code': 'invalid_agent',
        'error': 'Supported player types are human, random, negamax, and mcts.'}
    # Legacy identifiers are not exposed by the public API either.
    result = app.test_client().post(BASE + '/start_game',
                                    json={'player1': {'type': 'human'}, 'player2': {'type': 'victor'}})
    assert result.status_code == 400


def test_enabled_game_uses_public_fields_only(enabled):
    state = start(enabled).json
    assert state['players'][1] == {'type': 'victor_research'}
    state = move(enabled, state, 3).json
    result = move(enabled, state)
    assert result.status_code == 200 and result.json['revision'] == 2
    assert sum(cell == 'O' for row in result.json['board'] for cell in row) == 1
    records = enabled.test_client().get(BASE + '/games/' + state['game_id'] + '/history').json
    assert records['moves'][-1]['agent'] == {'type': 'victor_research'}
    assert records['moves'][-1]['outcome'] == {'status': 'ongoing', 'winner': None}
    keys = set(walk(result.json)) | set(walk(records))
    assert not keys & set(RESEARCH_FIELDS)


def test_enabled_research_agent_accepts_no_settings_and_can_open(enabled):
    result = enabled.test_client().post(BASE + '/start_game', json={
        'player1': {'type': 'victor_research', 'depth': 3}, 'player2': {'type': 'human'}})
    assert result.status_code == 400 and result.json['code'] == 'invalid_agent'
    state = start(enabled, research_player=0).json
    result = move(enabled, state)
    assert result.status_code == 200 and result.json['currentPlayer'] == 1


def test_busy_reservation_rejects_without_waiting_or_mutating(enabled):
    state = move(enabled, start(enabled).json, 3).json
    assert _victor_reservation.acquire(blocking=False)
    try:
        result = move(enabled, state)
    finally:
        _victor_reservation.release()
    assert result.status_code == 503 and result.json['code'] == 'agent_busy'
    assert 'Victor research' in result.json['error']
    after = enabled.test_client().get(BASE + '/games/' + state['game_id']).json
    assert after['revision'] == state['revision']
    assert move(enabled, state).status_code == 200


def test_disabling_after_start_blocks_research_moves(enabled):
    state = move(enabled, start(enabled).json, 3).json
    enabled.config['VICTOR_RESEARCH_ENABLED'] = False
    result = move(enabled, state)
    assert result.status_code == 409 and result.json['code'] == 'invalid_agent'


def test_opening_moves_come_from_the_exact_book_quickly():
    agent = VictorResearchAgent()
    started = perf_counter()
    assert agent.choose_move(game()) == 3  # the unique winning first move
    assert agent.last_decision['kind'] == 'opening_book'
    assert agent.last_decision['exact_value'] == 1
    assert perf_counter() - started < 0.5


def test_api_game_history_replays_exactly(enabled):
    """A complete research-agent game: every record replays from the empty board."""
    rng = Random(9)
    state = start(enabled, research_player=0).json
    client = enabled.test_client()
    while not state['gameOver']:
        if state['players'][state['currentPlayer']]['type'] == 'human':
            result = move(enabled, state, rng.choice(state['legalMoves']))
        else:
            result = move(enabled, state)
        assert result.status_code == 200
        assert result.json['revision'] == state['revision'] + 1
        state = result.json
    records = client.get(BASE + '/games/' + state['game_id'] + '/history').json
    replay = Connect4()
    for number, record in enumerate(records['moves'], 1):
        assert record['revision'] == number
        assert [list(r) for r in replay.board] == record['board_before']
        assert replay.make_move(record['column'])
        assert [list(r) for r in replay.board] == record['board_after']
        assert record['agent'] in ({'type': 'human'}, {'type': 'victor_research'})
    assert [list(r) for r in replay.board] == state['board']
    assert not set(walk(records)) & set(RESEARCH_FIELDS)


def test_flag_off_app_never_imports_or_runs_the_research_agent(monkeypatch):
    import sys
    monkeypatch.delenv('VICTOR_RESEARCH_ENABLED', raising=False)
    monkeypatch.delitem(sys.modules, 'games.connect4.agents.victor_research_agent', raising=False)
    app = create_app({'TESTING': True, 'EXPLANATIONS_ENABLED': False})
    client = app.test_client()
    state = client.post(BASE + '/start_game', json={'player1': {'type': 'human'},
                                                    'player2': {'type': 'mcts', 'simulation_limit': 50}}).json
    assert move(app, move(app, state, 3).json).status_code == 200
    assert 'games.connect4.agents.victor_research_agent' not in sys.modules
