"""Synthetic scaled schedule and inference-only protocol verification."""
from copy import deepcopy
import random
import json

import pytest

torch = pytest.importorskip('torch')

from games.connect4 import neural_self_play as driver
from games.connect4 import neural_evaluation as evaluation
from games.connect4.neural_mcts import NeuralInference
from test_neural_self_play import setup, threads


def test_pilot_preserved_and_scaled_bounds_explicit():
    assert driver.Bounds().max_games == 20
    assert driver.Bounds().max_updates == 200
    assert driver.Bounds.scaled() == driver.Bounds(200, 8400, 2000, 900, 'scaled')
    assert evaluation.SNAPSHOT_GAMES == (50, 100, 150)
    for kwargs in (dict(max_games=201), dict(max_plies=8401), dict(max_updates=2001),
                   dict(max_seconds=901), dict(max_updates=True)):
        with pytest.raises(ValueError):
            driver.Bounds.scaled(**kwargs)
    with pytest.raises(ValueError):
        driver.Bounds(profile='unknown')


def fake_updates(trainer):
    # No optimization or real neural searches in the 200-game schedule test.
    def step(examples):
        assert len(examples) == 32 and all(e.outcome is not None for e in examples)
        trainer.steps += 1
        trainer.last_metrics = dict(policy_loss=1., value_loss=.5, combined_loss=1.5)
        return 1.5
    trainer.step = step


def test_200_games_permit_1970_updates_and_fixed_snapshots():
    trainer, agent = setup()
    fake_updates(trainer)
    snapshots = []
    events = []
    report = driver.run_training(trainer, agent, driver.Bounds.scaled(), emit=events.append,
        on_snapshot=lambda r: snapshots.append((r['completed_games'], r['updates'])))
    assert report['status'] == 'bounded_stop' and report['stop_reason'] == 'max_games'
    assert report['completed_games'] == 200 and report['actual_plies'] == 8400
    assert report['updates'] == 1970 and report['partial_game'] is None
    assert report['smoke_gate'] == 'passed' and report['smoke_updates'] == 0
    assert snapshots == [(50,480), (100,980), (150,1480)]
    assert all(g['update_end'] == 0 for g in report['games'][:2])
    assert report['games'][103]['update_end'] == 1020
    assert report['games'][-1]['update_start'] == report['games'][-1]['update_end'] == 1970
    assert [e['after_game'] for e in events if e['event']=='update'] == [g for g in range(3,200) for _ in range(10)]


def test_scaled_update_cap_and_partial_ply_stop():
    trainer, agent = setup()
    fake_updates(trainer)
    report = driver.run_training(trainer, agent, driver.Bounds.scaled(max_updates=11))
    assert report['completed_games'] == 4 and report['updates'] == 11
    assert report['stop_reason'] == 'max_updates'
    trainer, agent = setup()
    fake_updates(trainer)
    report = driver.run_training(trainer, agent, driver.Bounds.scaled(max_plies=130))
    assert report['completed_games'] == 3 and report['updates'] == 10
    assert report['stop_reason'] == 'max_plies' and report['partial_game']['pending_plies'] == 4
    assert report['collected_plies'] == 126


def test_snapshot_read_only_known_win_scores_and_frozen_mirrors():
    trainer, agent = setup()
    before = deepcopy(trainer.model.state_dict())
    rng = driver.rng_states(random.Random(42), random.Random(42))
    result = evaluation.snapshot(agent.inference, dict(completed_games=0, updates=0, losses=[]))
    json.dumps(result, allow_nan=False)
    assert type(result['fixed_summary']['value']['saturated_099']) is int
    assert result['loss_last10']['value_loss'] is None
    assert len(result['fixed_rows']) == 14
    assert result['fixed_summary']['value']['known_win_count'] == 4
    assert sum(len(s['rows']) for s in result['frozen_suites'].values()) == 252
    assert result['training_feedback'] is False
    for suite in result['frozen_suites'].values():
        assert suite['previously_inspected']
        assert suite['summary']['mcts']['mirror_pairs']
        assert all(r['mcts']['tactical_guard'] is False for r in suite['rows'])
    assert all(torch.equal(v, before[k]) for k,v in trainer.model.state_dict().items())
    assert driver.rng_states(random.Random(42), random.Random(42)) == rng
    assert trainer.steps == 0 and not trainer.optimizer.state


def test_evaluation_schedule_matching_seeds_balanced_sides_and_legal_prefixes():
    schedule = list(evaluation.evaluation_schedule())
    assert len(schedule) == 144
    for i in range(0,144,2):
        first, final = schedule[i:i+2]
        assert first['model'] == 'initial' and final['model'] == 'final'
        assert {k:v for k,v in first.items() if k != 'model'} == {k:v for k,v in final.items() if k != 'model'}
        assert not driver.position(first['opening']).is_game_over()
    for model in ('initial','final'):
        for mode in evaluation.MODES:
            for opponent in evaluation.OPPONENTS:
                assert [sum(s['model']==model and s['mode']==mode and s['opponent']==opponent and s['side']==side
                            for s in schedule) for side in (0,1)] == [6,6]
    assert evaluation.EVALUATION_CONFIG['mcts_temperature'] == 0
    assert not evaluation.EVALUATION_CONFIG['tactical_guard']


class Lowest:
    def choose_move(self, game):
        return min(game.get_valid_moves())


def test_complete_synthetic_evaluation_records_replay_counts_and_no_learning():
    trainer, agent = setup()
    before = deepcopy(trainer.model.state_dict())
    report = evaluation.evaluate(dict(initial=agent.inference, final=agent.inference),
        chooser_factory=lambda inference,spec: Lowest().choose_move, opponent_factory=lambda spec: Lowest())
    json.dumps(report, allow_nan=False)
    assert report['status'] == 'complete' and report['completed_games'] == 144
    assert sum(r['completed'] for r in report['counts']) == 144
    assert all(r['win']+r['loss']+r['draw'] == r['completed'] == 6 for r in report['counts'])
    assert all(driver.position(g['moves']).check_winner() == g['winner'] for g in report['games'])
    assert trainer.steps == 0 and not trainer.optimizer.state
    assert all(torch.equal(v,before[k]) for k,v in trainer.model.state_dict().items())


def test_deadline_quarantines_unapplied_action_no_extension():
    trainer, agent = setup()
    now = [0.]
    def slow(game):
        now[0] += 2
        return min(game.get_valid_moves())
    report = evaluation.evaluate(dict(initial=agent.inference, final=agent.inference), max_seconds=1,
        clock=lambda: now[0], chooser_factory=lambda inference,spec: slow,
        opponent_factory=lambda spec: Lowest())
    assert report['status'] == 'deadline_limited' and report['completed_games'] == 0
    assert report['partial_game']['moves'] == list(evaluation.OPENINGS[0])
    assert report['elapsed_seconds'] == 2
    with pytest.raises(ValueError):
        evaluation.evaluate(dict(initial=agent.inference, final=agent.inference), max_seconds=181)


def test_seeded_existing_random_reproducible_and_process_rng_unchanged():
    state = random.getstate()
    a,b = evaluation.SeededRandom(42),evaluation.SeededRandom(42)
    game = driver.position((3,2))
    assert [a.choose_move(game) for _ in range(20)] == [b.choose_move(game) for _ in range(20)]
    assert random.getstate() == state


def test_illegal_evaluation_action_aborts():
    trainer, agent = setup()
    with pytest.raises(ValueError, match='Illegal evaluation move'):
        evaluation.evaluate(dict(initial=agent.inference, final=agent.inference),
            chooser_factory=lambda inference,spec: lambda game: 9)
