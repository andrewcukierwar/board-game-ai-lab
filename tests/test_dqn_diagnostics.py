"""Held-out fixture verification and inference-only comparative protocol."""
from copy import deepcopy
from collections import Counter
import random

import pytest

torch = pytest.importorskip('torch')

from games.connect4.dqn import diagnostics as d
from games.connect4.dqn.training import DQNConfig, DQNTrainer, ReplayMemory, collect_transition
from games.connect4.dqn.experiment import Bounds
from games.connect4.dqn.checkpoint import save_checkpoint


def engine_wins(game):
    result = []
    for action in range(7):
        if not game.is_valid_move(action):
            continue
        child = deepcopy(game)
        assert child.make_move(action)
        if child.check_winner() == game.current_player:
            result.append(action)
    return result


@pytest.mark.parametrize('fixture', d.FIXTURES, ids=lambda f: f['name'])
def test_frozen_fixture_independent_engine_verification(fixture):
    game = d.position(fixture['moves'])
    assert game.current_player == fixture['player']
    expected = fixture['expected_action']
    if fixture['category'] == 'win':
        assert engine_wins(game) == [expected]
    elif fixture['category'] == 'block':
        assert engine_wins(game) == []
        safe = []
        for action in range(7):
            if game.is_valid_move(action):
                child = deepcopy(game)
                child.make_move(action)
                if child.is_game_over() or not engine_wins(child):
                    safe.append(action)
        assert safe == [expected]
    if fixture['mirror_of']:
        original = next(f for f in d.FIXTURES if f['name'] == fixture['mirror_of'])
        board = d.position(original['moves']).board
        assert [list(row) for row in game.board] == [list(reversed(row)) for row in board]
        assert expected == (None if original['expected_action'] is None else 6-original['expected_action'])


def test_suite_coverage_and_frozen_identity():
    assert d.validate_fixtures() == d.suite_identity()
    assert len(d.FIXTURES) == 60
    tactical = [f for f in d.FIXTURES if f['expected_action'] is not None]
    assert len(tactical) == 48
    assert Counter((f['category'], f['direction'], f['player']) for f in tactical) == {
        (kind, direction, player): 4 for kind in ('win', 'block')
        for direction in ('horizontal', 'vertical', 'diagonal') for player in (0, 1)}
    assert {f['expected_action'] for f in tactical} == set(range(7))
    assert sum(f['mirror_of'] is not None for f in d.FIXTURES) == 30
    # Fail loudly if future edits silently change the evaluation suite.
    assert d.suite_identity()['sha256'] == 'ec99e0ee080370ab8cfc678c00a49ff552466de73ffe0955778ac87ee340543a'


def test_bad_label_rejected_before_experiment(monkeypatch, tmp_path):
    from games.connect4.dqn import experiment
    bad = dict(d.FIXTURES[0], expected_action=7)
    monkeypatch.setattr(d, 'FIXTURES', (bad,))
    monkeypatch.setattr(experiment, 'run_training', lambda *a, **k: pytest.fail('training started'))
    with pytest.raises(ValueError):
        experiment.run_experiment(tmp_path/'never-created', DQNConfig(), Bounds())
    assert not (tmp_path/'never-created').exists()


class FixedQ(torch.nn.Module):
    def forward(self, states):
        assert not torch.is_grad_enabled()
        return torch.tensor([100., 1., 2., 3., 4., 5., 6.]).expand(len(states), -1)


def test_legal_q_margin_mirror_and_concentration_metrics():
    rows = d.predictions(FixedQ())
    for row in rows:
        legal = [c for c in range(7) if row['legal'][c]]
        assert row['action'] in legal
        assert row['legal_q_min'] == min(row['q'][c] for c in legal)
        assert row['legal_q_max'] == max(row['q'][c] for c in legal)
        if row['expected_action'] is not None:
            expected = row['expected_action']
            assert row['tactical_margin'] == row['q'][expected] - max(row['q'][c] for c in legal if c != expected)
    summary = d.summarize_predictions(rows)
    assert summary['tactical']['total'] == 48
    # Tactical fixtures can also have full columns: use engine availability.
    expected_actions = {f['name']: next(c for c in (0, 6, 5, 4, 3, 2, 1)
                                      if d.position(f['moves']).is_valid_move(c))
                        for f in d.FIXTURES}
    counts = Counter(expected_actions.values())
    assert summary['greedy_actions']['counts'] == [counts[c] for c in range(7)]
    assert all(row['action'] == expected_actions[row['name']] for row in rows)
    assert summary['mirror_pairs'] == 30
    assert summary['mirror_action_matches'] == sum(
        expected_actions[f['name']] == 6-expected_actions[f['mirror_of']]
        for f in d.FIXTURES if f['mirror_of'])
    assert d.action_distribution([0, 0, 6])['counts'] == [2, 0, 0, 0, 0, 0, 1]
    assert d.action_distribution([0, 0, 6])['dominant_fraction'] == 2/3
    assert summary['mirrors'][0]['legal_q_max_difference'] == 94
    assert all(r['action'] == 6 for r in rows if r['name'].startswith('full') and not r['mirror_of'])


def test_hundred_game_protocol_balanced_immutable_and_reproducible():
    trainer = DQNTrainer(DQNConfig(seed=42))
    net = FixedQ()
    before = (random.getstate(), trainer.rng.getstate(), trainer.memory._rng.getstate(), torch.random.get_rng_state().clone())
    first = d.evaluate_random(net, games=100, seed=42, seconds=60, clock=lambda: 0)
    second = d.evaluate_random(net, games=100, seed=42, seconds=60, clock=lambda: 0)
    assert first == second
    assert first['protocol'] == [dict(index=i, seed=42+i, side=i%2) for i in range(100)]
    assert first['completed_games'] == 100 and first['starting_sides'] == [50, 50]
    for side in ('0', '1'):
        assert sum(first['by_side'][side][k] for k in ('wins', 'losses', 'draws')) == 50
    for key in ('wins', 'losses', 'draws'):
        assert first[key] == sum(first['by_side'][s][key] for s in ('0', '1'))
    expected = Counter(c for g in first['games'] for i,c in enumerate(g['moves']) if i%2 == g['side'])
    assert first['greedy_actions']['counts'] == [expected[c] for c in range(7)]
    assert (random.getstate(), trainer.rng.getstate(), trainer.memory._rng.getstate()) == before[:3]
    assert torch.equal(torch.random.get_rng_state(), before[3])
    assert trainer.updates == len(trainer.memory) == 0


@pytest.mark.parametrize('games', [True, -2, 1, 102, 100.0])
def test_invalid_evaluation_budget(games):
    with pytest.raises(ValueError):
        d.evaluate_random(FixedQ(), games=games, seed=42, seconds=60)


def test_replay_composition_eviction_and_rng_preservation():
    memory = ReplayMemory(2, seed=42)
    ongoing = collect_transition(d.position(()), 3)
    win = collect_transition(d.position([0, 1, 0, 1, 0, 2]), 0)
    rng = memory._rng.getstate()
    memory.append(ongoing)
    memory.append(win)
    assert memory.composition() == dict(size=2, terminal=1, nonterminal=1,
                                       rewards={'0':1,'1':1}, actors={'0':2,'1':0})
    memory.append(win)
    assert memory.composition()['terminal'] == 2
    assert memory.composition()['rewards'] == {'0':0,'1':2}
    assert memory._rng.getstate() == rng


def test_three_model_orchestration_without_training(tmp_path, monkeypatch):
    from games.connect4.dqn import experiment
    archive = tmp_path/'archive.pt'
    save_checkpoint(archive, DQNTrainer(DQNConfig(seed=7)).online)
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda value: None)
    monkeypatch.setattr(experiment, 'source_identity', lambda *a: {'fixture': True})
    calls = []
    def no_training(trainer, bounds, **kwargs):
        calls.append(1)
        assert trainer.updates == len(trainer.memory) == 0
        return {'status': 'bounded_stop', 'plies': 0, 'updates': 0}
    monkeypatch.setattr(experiment, 'run_training', no_training)
    report = experiment.run_experiment(tmp_path/'output', DQNConfig(seed=42), Bounds(),
                                       evaluation_games=4, archived_checkpoint=archive)
    assert 'artifact_error' not in report
    assert calls == [1]
    assert set(report['diagnostics']) == {'initial','archived','final'}
    assert report['diagnostics']['initial'] == report['diagnostics']['final']
    assert report['checkpoint']['reload_exact']
    assert all(r['protocol'] == report['evaluation_config']['protocol'] for r in report['evaluation'].values())
