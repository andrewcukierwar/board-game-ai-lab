"""Independent labels and inference-only evaluation correctness, without candidate inference."""
import json
import random
from collections import Counter
from copy import deepcopy

import pytest

torch = pytest.importorskip('torch')

from games.connect4.dqn import validation as v
from games.connect4.dqn import validation_holdout as h
from games.connect4.dqn.checkpoint import CONTRACT, load_checkpoint
from games.connect4.dqn.network import DQN
from games.connect4.agents.random_agent import RandomAgent
from games.connect4.agents.negamax_agent import NegamaxAgent

ROWS = json.loads(h.PATH.read_text())['positions']
HOLDOUT_SHA = '1c79db45a09e00a8e8806448362eb319b5e9c81e431de03c7b5ece8b2d04bb2f'


def engine_wins(game):
    actions = []
    for col in game.get_valid_moves():
        child = deepcopy(game)
        child.make_move(col)
        if child.check_winner() == game.current_player:
            actions.append(col)
    return actions


@pytest.mark.parametrize('row', ROWS, ids=lambda r: r['name'])
def test_independent_labels(row):
    game = h.replay(row['moves'])
    action = row['expected_action']
    assert game.current_player == row['player']
    threat = game
    if row['category'] == 'win':
        assert engine_wins(game) == [action]
    else:
        assert not engine_wins(game)
        safe = []
        for col in game.get_valid_moves():
            child = deepcopy(game)
            child.make_move(col)
            if child.is_game_over() or not engine_wins(child):
                safe.append(col)
        assert safe == [action]
        threat = deepcopy(game)
        threat.current_player = 1-game.current_player
        threat.piece = 'XO'[threat.current_player]
        assert engine_wins(threat) == [action]
    # Independent direction check: count contiguous pieces through the landing cell.
    landing = max(r for r in range(6) if threat.board[r][action] == ' ')
    piece = threat.piece
    threat.make_move(action)
    directions = set()
    for dr, dc, direction in ((0, 1, 'horizontal'), (1, 0, 'vertical'), (1, 1, 'diagonal'), (1, -1, 'diagonal')):
        count = 1
        for sign in (-1, 1):
            r, c = landing+sign*dr, action+sign*dc
            while 0 <= r < 6 and 0 <= c < 7 and threat.board[r][c] == piece:
                count += 1
                r, c = r+sign*dr, c+sign*dc
        if count >= 4:
            directions.add(direction)
    assert directions == {row['direction']}


def test_frozen_coverage_and_no_overlap():
    assert h.digest(h.PATH) == HOLDOUT_SHA
    assert h.validate(ROWS) == 48
    assert Counter((r['category'], r['direction'], r['player']) for r in ROWS) == {
        (k, d, p): 8 for k in ('win', 'block') for d in ('horizontal', 'vertical', 'diagonal') for p in (0, 1)}
    assert [sum(r['expected_action'] == c for r in ROWS) for c in range(7)] == [14,14,14,12,14,14,14]
    exact = lambda g: (g.current_player, tuple(tuple(row) for row in g.board))
    assert len({exact(h.replay(r['moves'])) for r in ROWS}) == 96
    old = {h.key(h.replay(r['moves'])) for r in json.loads(h.OLD_PATH.read_text())['positions']}
    assert not old & {h.key(h.replay(r['moves'])) for r in ROWS}
    assert h.SEED != json.loads(h.OLD_PATH.read_text())['construction_seed']


def test_duplicate_and_mirror_rejection_and_bad_label():
    game = h.replay(ROWS[0]['moves'])
    seen = set()
    h.reject_duplicate(game, seen)
    for moves in (ROWS[0]['moves'], [6-c for c in ROWS[0]['moves']]):
        with pytest.raises(ValueError, match='Duplicate'):
            h.reject_duplicate(h.replay(moves), seen)
    bad = deepcopy(ROWS)
    bad[0]['expected_action'] = (bad[0]['expected_action']+1) % 7
    with pytest.raises(ValueError, match='label'):
        h.validate(bad)


def test_reproducible_balanced_legal_opening_protocol():
    before = random.getstate()
    a, b = v.configuration(), v.configuration()
    assert a == b and random.getstate() == before
    assert len(a['openings']) == 20
    assert len({tuple(p) for p in a['openings']}) == 20
    assert Counter(s for p in a['openings'] for s in a['sides']) == {0:20, 1:20}
    assert Counter(h.replay(p).current_player for p in a['openings']) == {0:12, 1:8}
    for i in range(0, 20, 2):
        assert a['openings'][i+1] == [6-c for c in a['openings'][i]]
    for p in a['openings']:
        assert not h.replay(p).is_game_over()
    assert a['aggregate_seconds'] == 600 and not a['learning'] and a['epsilon'] == 0


@pytest.mark.parametrize('moves', [[-1], [7], [True], [0]*7, [0,1,0,1,0,1,0], [0,1,0,1,0,1,0,2]])
def test_illegal_or_terminal_prefix_rejected(moves):
    with pytest.raises(ValueError):
        h.replay(moves)


class FixedQ(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.values = torch.nn.Parameter(torch.arange(7, dtype=torch.float32))

    def forward(self, states):
        assert not torch.is_grad_enabled()
        assert not self.training
        return self.values.expand(len(states), -1)


@pytest.mark.parametrize('opponent', [RandomAgent, lambda: NegamaxAgent(1), lambda: NegamaxAgent(2)])
def test_complete_reproducible_games_no_learning(opponent, monkeypatch):
    net = FixedQ()
    weights = net.values.detach().clone()
    def forbidden(*args, **kwargs):
        pytest.fail('Learning during evaluation')
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    random_before, torch_before = random.getstate(), torch.random.get_rng_state().clone()
    games = []
    for side in (0, 1):
        first = v.play(net, opponent(), [3, 2], side, 11, 100, clock=lambda: 0)
        second = v.play(net, opponent(), [3, 2], side, 11, 100, clock=lambda: 0)
        assert first == second and first['complete']
        end = h.replay(first['moves'], terminal=True)
        assert end.is_game_over() and len(first['moves']) <= 42
        assert first['winner'] == end.check_winner()
        assert first['result'] == ('draw' if end.check_winner() == -1 else 'win' if end.check_winner() == side else 'loss')
        assert first['moves'][2:] == [d['action'] for d in first['decisions']]
        games.append(first)
    assert v.summarize_games(games)['starting_sides'] == [1,1]
    assert torch.equal(net.values, weights) and net.values.grad is None
    assert random.getstate() == random_before and torch.equal(torch.random.get_rng_state(), torch_before)


def test_safe_loaded_weights_unchanged_without_checkpoint_creation(monkeypatch):
    # Exercise the existing versioned safe loader on a synthetic in-memory payload.
    net = DQN()
    state = {k: t.clone() for k, t in net.state_dict().items()}
    calls = []
    def safe_load(path, **kwargs):
        calls.append(kwargs)
        return dict(CONTRACT, model_state_dict=state, training_metadata={})
    monkeypatch.setattr(torch, 'load', safe_load)
    loaded, _ = load_checkpoint('synthetic-in-memory-only')
    v.predictions(loaded, ROWS[:2])
    v.play(loaded, RandomAgent(), [0,1], 0, 3, 100, clock=lambda: 0)
    assert calls == [dict(weights_only=True, map_location='cpu')]
    assert all(torch.equal(t, state[k]) for k, t in loaded.state_dict().items())
    assert all(p.grad is None for p in loaded.parameters())


def test_deadline_partial_and_wld_accounting():
    partial = v.play(FixedQ(), RandomAgent(), [], 0, 1, 0, clock=lambda: 0)
    assert not partial['complete'] and partial['result'] is None
    games = [dict(complete=True, side=s, result=v.outcome(w,s), elapsed_seconds=1)
             for s in (0, 1) for w in (-1, 0, 1)] + [partial]
    summary = v.summarize_games(games)
    assert (summary['wins'], summary['losses'], summary['draws']) == (2,2,2)
    assert summary['completed'] == 6 and summary['partial'] == 1
    assert summary['starting_sides'] == [3,3]
    assert summary['by_side'] == {'0': {'draw':1,'win':1,'loss':1}, '1': {'draw':1,'win':1,'loss':1}}
    ticks = iter([0, 0, 2, 2])
    late = v.play(FixedQ(), RandomAgent(), [], 0, 1, 1, clock=lambda: next(ticks))
    assert not late['complete'] and late['moves'] == []


def test_comparison_lists_and_margin_metrics():
    rows = v.predictions(FixedQ(), ROWS[:2])
    for row in rows['rows']:
        q, action = row['q'], row['expected_action']
        assert row['tactical_margin'] == q[action]-max(q[c] for c in range(7) if row['legal'][c] and c != action)
    before = {'rows': [dict(r, tactical_correct=True) for r in rows['rows']]}
    after = {'rows': [dict(r, tactical_correct=False) for r in rows['rows']]}
    comparison = v.compare(before, after)
    assert len(comparison['previously_correct']) == len(comparison['regressed']) == 2
    assert not comparison['improved']
