"""Synthetic driver verification: fixed games/batches, never strength-dependent."""

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

torch = pytest.importorskip('torch')

from games.connect4.dqn.checkpoint import load_checkpoint
from games.connect4.dqn.diagnostics import POSITIONS, evaluate_random, position, predictions
from games.connect4.dqn.experiment import (
    Bounds, parser, run_training, save_candidate, sha256, source_identity,
)
from games.connect4.dqn.training import DQNConfig, DQNTrainer

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]
WIN = [0, 6, 1, 6, 2, 5, 3]


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def fixed_trainer(moves=WIN, *, batch_size=2):
    trainer = DQNTrainer(DQNConfig(seed=42, batch_size=batch_size, target_sync_interval=3,
                                   epsilon_min=0.1, epsilon_decay=0.9997))
    actions = iter(moves)
    trainer.choose_move = lambda game: next(actions)
    return trainer


@pytest.mark.parametrize('name,value', [('max_games', 0), ('max_plies', -1),
                                       ('max_updates', True), ('max_seconds', float('nan')),
                                       ('max_seconds', float('inf')), ('max_seconds', 0)])
def test_invalid_bounds(name, value):
    with pytest.raises(ValueError):
        replace(Bounds(), **{name: value})


def test_complete_games_persist_optimizer_replay_and_sync():
    trainer = fixed_trainer(WIN * 2)
    optimizer = trainer.optimizer
    events = []
    report = run_training(trainer, Bounds(max_games=2), emit=events.append)
    assert report['status'] == 'bounded_stop'
    assert report['stop_reason'] == 'max_games'
    assert report['completed_games'] == report['started_games'] == 2
    assert report['plies'] == report['replay_size'] == 14
    assert report['updates'] == 13
    assert len(report['losses']) == 13
    assert report['wins_x'] == 2 and report['wins_o'] == report['draws'] == 0
    assert report['partial_episode'] is None
    assert report['target_sync_updates'] == [3, 6, 9, 12]
    assert report['epsilon_final'] == pytest.approx(0.9997 ** 13)
    assert trainer.optimizer is optimizer
    assert report['episodes'][1]['update_start'] == 6
    assert [e['actor'] for e in events if e['event'] == 'ply'] == [0, 1, 0, 1, 0, 1, 0] * 2
    assert sum(e.get('target_sync', False) for e in events) == 4
    assert all(e['gradient_abs_max'] >= 0 for e in events if e['event'] == 'ply')


@pytest.mark.parametrize('bounds,plies,updates,reason', [
    (Bounds(max_plies=3), 3, 1, 'max_plies'),
    (Bounds(max_updates=2), 3, 2, 'max_updates'),
    (Bounds(max_plies=1), 1, 0, 'max_plies'),
])
def test_first_limit_stops_partial_game(bounds, plies, updates, reason):
    report = run_training(fixed_trainer(), bounds)
    assert report['stop_reason'] == reason
    assert report['plies'] == plies and report['updates'] == updates
    assert report['completed_games'] == 0
    assert len(report['partial_episode']['moves']) == plies


def test_deadline_after_collection_skips_update(monkeypatch):
    from games.connect4.dqn import experiment
    now = [0.0]
    original = experiment.collect_transition
    def collect(game, action):
        transition = original(game, action)
        now[0] = 10.0
        return transition
    monkeypatch.setattr(experiment, 'collect_transition', collect)
    report = run_training(fixed_trainer(batch_size=1), Bounds(max_seconds=10), clock=lambda: now[0])
    assert report['stop_reason'] == 'max_seconds'
    assert report['plies'] == 1 and report['updates'] == 0
    assert report['elapsed_seconds'] == 10


def test_deadline_during_selection_does_not_collect():
    now = [0.0]
    trainer = fixed_trainer()
    def choose(game):
        now[0] = 10.0
        return 0
    trainer.choose_move = choose
    report = run_training(trainer, Bounds(max_seconds=10), clock=lambda: now[0])
    assert report['stop_reason'] == 'max_seconds'
    assert report['plies'] == report['updates'] == 0


def test_clean_stop_retains_partial_progress():
    events = []
    report = run_training(fixed_trainer(), Bounds(), emit=events.append,
                          should_stop=lambda: len(events) == 3)
    assert report['status'] == 'interrupted'
    assert report['plies'] == 3 and report['updates'] == 2
    assert report['completed_games'] == 0
    assert report['partial_episode']['moves'] == WIN[:3]


@pytest.mark.parametrize('moves,winner,key', [(WIN, 0, 'wins_x'),
                                           ([6, 0, 6, 1, 5, 2, 5, 3], 1, 'wins_o'),
                                           (DRAW, -1, 'draws')])
def test_exact_terminal_outcomes(moves, winner, key):
    # No learning: controlled full legal transcripts isolate outcome handling.
    report = run_training(fixed_trainer(moves, batch_size=64), Bounds(max_games=1))
    assert report[key] == 1
    assert report['episodes'][0]['winner'] == winner
    assert report['episodes'][0]['moves'] == moves
    assert report['updates'] == 0


@pytest.mark.parametrize('fault', ['weights', 'q', 'gradients', 'loss'])
def test_numerical_failure_stops_without_further_plies(fault):
    trainer = fixed_trainer(batch_size=1)
    if fault == 'weights':
        with torch.no_grad():
            next(trainer.online.parameters()).flatten()[0] = float('nan')
    elif fault == 'q':
        trainer.online.forward = lambda states: torch.full((len(states), 7), float('inf'))
    elif fault == 'gradients':
        next(trainer.online.parameters()).register_hook(lambda gradient: gradient * float('nan'))
    else:
        trainer.optimize = lambda: float('nan')
    report = run_training(trainer, Bounds())
    assert report['status'] == 'invariant_failure'
    assert report['plies'] <= 1
    assert report['updates'] == 0
    assert report['error']


def test_post_optimizer_nonfinite_weights_are_detected():
    trainer = fixed_trainer(batch_size=1)
    original = trainer.optimizer.step
    def poison(*args, **kwargs):
        original(*args, **kwargs)
        with torch.no_grad():
            next(trainer.online.parameters()).fill_(float('inf'))
    trainer.optimizer.step = poison
    report = run_training(trainer, Bounds())
    assert report['status'] == 'invariant_failure'
    assert report['plies'] == report['updates'] == 1
    assert 'Nonfinite' in report['error']


def test_controlled_updates_are_reproducible():
    first, second = fixed_trainer(), fixed_trainer()
    a = run_training(first, Bounds(max_games=1), clock=lambda: 0.0)
    b = run_training(second, Bounds(max_games=1), clock=lambda: 0.0)
    assert a == b
    assert all(torch.equal(value, second.online.state_dict()[key])
               for key, value in first.online.state_dict().items())


def test_reused_training_state_is_rejected():
    trainer = fixed_trainer()
    run_training(trainer, Bounds(max_games=1))
    with pytest.raises(ValueError, match='fresh'):
        run_training(trainer, Bounds())


def winning_moves(game):
    from copy import deepcopy
    wins = []
    for move in game.get_valid_moves():
        copy = deepcopy(game)
        copy.make_move(move)
        if copy.check_winner() == game.current_player:
            wins.append(move)
    return wins


def test_probes_are_legal_and_tactical_labels_are_true():
    from copy import deepcopy
    assert {position(moves).current_player for _, moves, _ in POSITIONS} == {0, 1}
    for name, moves, expected in POSITIONS:
        game = position(moves)
        if name.startswith('win'):
            assert winning_moves(game) == [expected]
        if name.startswith('block'):
            assert winning_moves(game) == []
            safe = []
            for move in game.get_valid_moves():
                copy = deepcopy(game)
                copy.make_move(move)
                if not winning_moves(copy):
                    safe.append(move)
            assert safe == [expected]
        if name.startswith('full'):
            assert not game.is_valid_move(0)
    rows = predictions(DQNTrainer().online)
    assert len(rows) == 10
    assert all(row['legal'][row['action']] for row in rows)


def test_evaluation_is_seeded_balanced_and_never_learns():
    trainer = DQNTrainer()
    rng = trainer.rng.getstate()
    # Controlled constant greedy outputs; a policy fixture, not an experiment.
    with torch.no_grad():
        for parameter in trainer.online.parameters():
            parameter.zero_()
    state = {key: value.clone() for key, value in trainer.online.state_dict().items()}
    first = evaluate_random(trainer.online, games=2, seed=7, seconds=60, clock=lambda: 0.0)
    second = evaluate_random(trainer.online, games=2, seed=7, seconds=60, clock=lambda: 0.0)
    assert first == second
    assert first['completed_games'] == 2 and first['starting_sides'] == [1, 1]
    assert first['wins'] + first['losses'] + first['draws'] == 2
    assert trainer.updates == len(trainer.memory) == 0
    assert trainer.rng.getstate() == rng
    assert all(torch.equal(value, state[key]) for key, value in trainer.online.state_dict().items())


def test_evaluation_deadline_reports_partial():
    times = iter([0.0, 0.0, 60.0, 60.0])
    report = evaluate_random(DQNTrainer().online, games=2, seed=7, seconds=60,
                             clock=lambda: next(times))
    assert report['completed_games'] == 0
    assert report['partial']['moves'] == []


def test_candidate_round_trip_and_no_overwrite(tmp_path):
    trainer = fixed_trainer()
    training = run_training(trainer, Bounds(max_games=1))
    report = dict(training=training, diagnostics=dict(final=predictions(trainer.online)))
    candidate = save_candidate(tmp_path, trainer, report)
    model, metadata = load_checkpoint(candidate['path'])
    assert metadata == report
    assert predictions(model) == report['diagnostics']['final']
    assert candidate['sha256'] == sha256(Path(candidate['path']))
    assert candidate['reload_exact'] and not candidate['resumable']
    with pytest.raises(FileExistsError):
        save_candidate(tmp_path, trainer, report)


def test_source_identity_includes_untracked_bytes_and_dirty_diff(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    def git(*args):
        return subprocess.run(['git', '-C', str(root), *args], check=True, capture_output=True)
    git('init')
    (root / 'tracked').write_text('before\n')
    git('add', 'tracked')
    git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.test', 'commit', '-m', 'fixture')
    (root / 'tracked').write_text('after\n')
    (root / 'new.py').write_text('one\n')
    output = tmp_path / 'output'
    output.mkdir()
    first = source_identity(root, output)
    assert first['dirty'] and 'new.py' in first['untracked_sha256']
    assert '+after' in (output / 'source.patch').read_text()
    assert (output / 'untracked-source/new.py').read_text() == 'one\n'
    (root / 'new.py').write_text('two\n')
    assert source_identity(root, output)['identity_sha256'] != first['identity_sha256']


def test_cli_authorized_defaults():
    args = parser().parse_args(['--output', '/tmp/synthetic-output'])
    assert (args.seed, args.max_games, args.max_plies, args.max_updates, args.max_seconds) == (42, 200, 8400, 8400, 600)
    assert (args.epsilon_start, args.epsilon_min, args.epsilon_decay) == (1.0, 0.1, 0.9997)
    assert (args.threads, args.interop_threads, args.evaluation_games, args.evaluation_seconds) == (1, 1, 24, 120)


def test_failed_driver_never_saves_or_evaluates_final(tmp_path, monkeypatch):
    from games.connect4.dqn import experiment
    monkeypatch.setattr(torch, 'set_num_interop_threads', lambda value: None)
    monkeypatch.setattr(experiment, 'source_identity', lambda *args: {'fixture': True})
    monkeypatch.setattr(experiment, 'run_training', lambda *args, **kwargs: {'status': 'invariant_failure'})
    calls = []
    monkeypatch.setattr(experiment, 'evaluate_random', lambda *args, **kwargs: calls.append(1) or {})
    def forbidden(*args):
        raise AssertionError('Must not save after invariant failure')
    monkeypatch.setattr(experiment, 'save_candidate', forbidden)
    output = tmp_path / 'failed'
    report = experiment.run_experiment(output, DQNConfig(), Bounds())
    assert calls == [1]
    assert report['checkpoint'] is None and not (output / 'candidate.pt').exists()
    assert json.loads((output / 'report.json').read_text())['training']['status'] == 'invariant_failure'


def test_import_is_inert(tmp_path):
    result = subprocess.run([sys.executable, '-c',
                             'import games.connect4.dqn.experiment'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0 and result.stdout == result.stderr == ''
