"""Synthetic driver/checkpoint verification; no real self-play training launch."""
from copy import deepcopy
from dataclasses import replace
import json
from random import Random

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from games.connect4 import neural_self_play as experiment
from games.connect4.agents.mcts_nn_agent import MCTSNNAgent, Node, SearchResult
from games.connect4.connect4 import Connect4
from games.connect4.neural_mcts import (
    CHECKPOINT_CONTRACT, ENCODING, Connect4Net, NeuralInference,
    checkpoint_payload, encode_current_player, load_checkpoint, save_checkpoint,
)
from games.connect4.train_mcts_nn import NeuralTrainer, capture_example, training_loss, batch_tensors

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]
X_WIN = [0, 1, 0, 1, 0, 2, 0]
O_WIN = [0, 1, 0, 1, 2, 1, 2, 1]


@pytest.fixture(autouse=True, scope='module')
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class ScriptedSearch:
    """Controlled synthetic root targets/history, never a neural search experiment."""
    simulation_limit, exploration, temperature, tactical_guard = 32, 1.41, 1., False

    def __init__(self, model, moves=DRAW):
        self.inference = NeuralInference(model)
        self.moves = moves
        self.calls = 0

    def search(self, game):
        move = self.moves[self.calls % len(self.moves)]
        self.calls += 1
        root = Node(deepcopy(game))
        root.visits = 32
        for action in game.get_valid_moves():
            successor = deepcopy(game)
            assert successor.make_move(action)
            root.children[action] = Node(successor, root, action, 1 / len(game.get_valid_moves()))
        root.children[move].visits = 32
        policy = tuple(float(a == move) for a in range(7))
        visits = tuple(32 if a == move else 0 for a in range(7))
        return SearchResult(move, policy, visits, 1., False, 'disabled', root)


def setup(moves=DRAW):
    with torch.random.fork_rng():
        torch.manual_seed(42)
        trainer = NeuralTrainer(Connect4Net())
    return trainer, ScriptedSearch(trainer.model, moves)


@pytest.mark.parametrize('moves,winner,labels', [
    (X_WIN, 0, {'-1': 6, '0': 0, '1': 8}),
    (O_WIN, 1, {'-1': 8, '0': 0, '1': 8}),
    (DRAW, -1, {'-1': 0, '0': 84, '1': 0}),
])
def test_two_game_collection_only_gate(moves, winner, labels):
    trainer, agent = setup(moves)
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    events = []
    report = experiment.run_training(trainer, agent, experiment.Bounds(max_games=2), emit=events.append)
    assert report['status'] == 'bounded_stop' and report['stop_reason'] == 'max_games'
    assert report['completed_games'] == 2 and report['smoke_gate'] == 'passed'
    assert report['updates'] == 0 and not trainer.optimizer.state
    assert report['labels'] == labels
    assert report['actual_plies'] == report['collected_plies'] == 2 * len(moves)
    assert all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items())
    for event in [e for e in events if e['event'] == 'completed_game']:
        assert event['winner'] == winner
        assert all(e['outcome'] == (0 if winner == -1 else 1 if e['acting_player'] == winner else -1)
                   for e in event['examples'])
    assert [e['event'] for e in events].count('smoke_gate') == 1


def test_updates_only_after_third_completed_game_with_one_optimizer_and_sampling():
    trainer, agent = setup()
    optimizer = trainer.optimizer
    events = []
    report = experiment.run_training(trainer, agent, experiment.Bounds(max_updates=2), emit=events.append)
    assert report['completed_games'] == 3 and report['stop_reason'] == 'max_updates'
    assert report['updates'] == 2 and report['partial_game'] is None
    assert trainer.optimizer is optimizer
    assert all(s['step'].item() == 2 for s in optimizer.state.values())
    updates = [e for e in events if e['event'] == 'update']
    expected_rng = Random(42)
    for update in updates:
        assert update['after_game'] == 3
        assert update['sampled_indices'] == expected_rng.sample(range(126), 32)
        assert update['finite_gradients'] and update['finite_parameters']
        assert update['combined_loss'] == pytest.approx(update['policy_loss'] + update['value_loss'])
        assert update['policy_target_entropy'] == 0
    ply_events = [e for e in events if e['event'] == 'ply']
    assert all(e['updates'] == 0 for e in ply_events)
    assert not trainer.model.training and report['labels'] == {'-1': 0, '0': 126, '1': 0}


def test_predeclared_ten_update_schedule_stops_at_game_limit():
    trainer, agent = setup()
    report = experiment.run_training(trainer, agent, experiment.Bounds(max_games=4))
    assert report['completed_games'] == 4 and report['updates'] == 10
    assert [(e['update_start'], e['update_end']) for e in report['games']] == [(0, 0), (0, 0), (0, 10), (10, 10)]


@pytest.mark.parametrize('defect', ['actor', 'observation', 'visits', 'guard', 'policy', 'winner_label', 'mutation'])
def test_smoke_failure_aborts_without_optimizer_or_retry(monkeypatch, defect):
    trainer, agent = setup()
    if defect in ('actor', 'observation', 'mutation'):
        original = experiment.capture_example
        def capture(result):
            example = original(result)
            if defect == 'actor':
                return replace(example, acting_player=1 - example.acting_player)
            if defect == 'observation':
                return replace(example, observation=[[0] * 7 for _ in range(6)])
            return example
        monkeypatch.setattr(experiment, 'capture_example', capture)
        if defect == 'mutation':
            # Mutate a previously captured frozen observation by deliberately bypassing freeze.
            captured = []
            def corrupt(result):
                if captured:
                    object.__setattr__(captured[0], 'observation', ((1,) * 7,) * 6)
                example = original(result)
                captured.append(example)
                return example
            monkeypatch.setattr(experiment, 'capture_example', corrupt)
    elif defect == 'winner_label':
        original = experiment.finalize_examples
        monkeypatch.setattr(experiment, 'finalize_examples', lambda examples, game:
                            tuple(replace(e, outcome=1) for e in original(examples, game)))
    else:
        original = agent.search
        def search(game):
            result = original(game)
            return replace(result, **{
                'visits': {'visits': (0,) * 7},
                'guard': {'tactical_guard': True},
                'policy': {'policy': (float('nan'),) * 7},
            }[defect])
        agent.search = search
    report = experiment.run_training(trainer, agent)
    assert report['status'] == 'invariant_failure' and report['smoke_gate'] == 'failed'
    assert report['updates'] == 0 and not trainer.optimizer.state
    assert report['started_games'] == 1


def test_partial_game_at_ply_limit_is_never_labeled_or_trained():
    trainer, agent = setup()
    events = []
    report = experiment.run_training(trainer, agent, experiment.Bounds(max_plies=90), emit=events.append)
    assert report['stop_reason'] == 'max_plies' and report['actual_plies'] == 90
    assert report['completed_games'] == 2 and report['collected_plies'] == 84
    assert report['updates'] == 0 and report['partial_game']['pending_plies'] == 6
    assert report['labels'] == {'-1': 0, '0': 84, '1': 0}
    assert len([e for e in events if e['event'] == 'completed_game']) == 2


def test_deadline_after_search_prevents_move_and_update():
    trainer, agent = setup()
    now = [0.]
    original = agent.search
    def slow_search(game):
        result = original(game)
        now[0] += 2
        return result
    agent.search = slow_search
    events = []
    report = experiment.run_training(trainer, agent, experiment.Bounds(max_seconds=1),
                                     clock=lambda: now[0], emit=events.append)
    assert report['stop_reason'] == 'max_seconds' and report['updates'] == 0
    assert report['actual_plies'] == report['collected_plies'] == 0
    assert events[0]['event'] == 'unapplied_search'


def test_interruption_preserves_completed_collection_and_quarantines_partial():
    trainer, agent = setup()
    report = experiment.run_training(trainer, agent, should_stop=lambda: agent.calls >= 45)
    assert report['status'] == 'interrupted' and report['completed_games'] == 1
    assert report['collected_plies'] == 42 and report['actual_plies'] == 44
    assert report['partial_game']['pending_plies'] == 2 and report['updates'] == 0


def test_numerical_update_failure_aborts_no_more_search(monkeypatch):
    trainer, agent = setup()
    handle = trainer.model.policy_fc.bias.register_hook(lambda g: g * float('nan'))
    report = experiment.run_training(trainer, agent)
    handle.remove()
    assert report['status'] == 'invariant_failure' and report['smoke_gate'] == 'passed'
    assert report['updates'] == 0 and agent.calls == 126 and not trainer.optimizer.state
    assert 'gradients' in report['error']


@pytest.mark.parametrize('bounds', [dict(max_games=21), dict(max_plies=841), dict(max_updates=201),
                                   dict(max_seconds=901), dict(max_seconds=float('nan')), dict(max_games=True)])
def test_cannot_extend_authorized_limits(bounds):
    with pytest.raises(ValueError):
        experiment.Bounds(**bounds)


def test_fixed_diagnostics_legal_coverage_guard_off_and_no_model_updates():
    trainer, _ = setup()
    model = trainer.model
    before = {k: v.clone() for k, v in model.state_dict().items()}
    rows = experiment.diagnostics(NeuralInference(model))
    assert len(rows) == 14 and {r['actor'] for r in rows} == {0, 1}
    assert all(r['mcts']['guard_applied'] == 'disabled' and sum(r['mcts']['visits']) == 32 for r in rows)
    assert all(r['nn_only']['action'] in r['legal_moves'] and r['mcts']['action'] in r['legal_moves'] for r in rows)
    assert sum(bool(r['immediate_wins']) for r in rows) == 4
    assert all(r['tactical_required'] for r in rows if r['name'].startswith('block'))
    assert all(r['nn_only']['legal_policy'][0 if not r['name'].endswith('mirror') else 6] == 0
               for r in rows if r['name'].startswith('full_column'))
    for name, moves in experiment.POSITIONS[9:]:
        original = next(seq for n, seq in experiment.POSITIONS if n == name.removesuffix('_mirror'))
        assert moves == tuple(6 - m for m in original)
    assert trainer.steps == 0 and not trainer.optimizer.state
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_checkpoint_minimal_schema_independent_copy_safe_reload_and_no_overwrite(tmp_path):
    trainer, _ = setup()
    model = trainer.model
    rng = torch.get_rng_state().clone()
    payload = checkpoint_payload(model)
    assert set(payload) == {'contract', 'model_state_dict'} and payload['contract'] == CHECKPOINT_CONTRACT
    assert torch.equal(rng, torch.get_rng_state())
    for key, tensor in model.state_dict().items():
        assert payload['model_state_dict'][key].data_ptr() != tensor.data_ptr()
        assert torch.equal(payload['model_state_dict'][key], tensor)
    with torch.no_grad():
        model.conv1.bias.add_(1)
    assert not torch.equal(payload['model_state_dict']['conv1.bias'], model.conv1.bias)
    path = tmp_path / 'synthetic.pt'
    save_checkpoint(path, model)
    actual = torch.load(path, weights_only=True, map_location='cpu')
    assert set(actual) == {'contract', 'model_state_dict'}
    check = experiment.verify_checkpoint(path, model)
    assert check['exact_weight_equality'] and check['exact_prediction_equality'] and check['finite_predictions']
    assert check['fixed_positions'] == 14
    assert load_checkpoint(path).model.representation_version == ENCODING
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        save_checkpoint(path, model)
    assert path.read_bytes() == before


@pytest.mark.parametrize('defect', ['nan', 'dtype', 'representation', 'shape', 'architecture'])
def test_writer_rejects_invalid_model_before_creating_file(tmp_path, defect):
    trainer, _ = setup()
    model = trainer.model
    if defect == 'nan':
        with torch.no_grad():
            model.conv1.bias[0] = float('nan')
    elif defect == 'dtype':
        model.double()
    elif defect == 'representation':
        model.representation_version = 'historical'
    elif defect == 'shape':
        model.conv1.bias = torch.nn.Parameter(torch.zeros(2))
    else:
        model = torch.nn.Linear(2, 2).eval()
    path = tmp_path / 'never.pt'
    with pytest.raises(ValueError):
        save_checkpoint(path, model)
    assert not path.exists()


def test_trainer_metric_components_remain_authoritative():
    trainer, agent = setup(X_WIN)
    events = []
    experiment.run_training(trainer, agent, experiment.Bounds(max_games=2), emit=events.append)
    from games.connect4.train_mcts_nn import TrainingExample
    examples = [TrainingExample(**e) for event in events if event['event'] == 'completed_game' for e in event['examples']]
    states, policies, outcomes = batch_tensors(examples)
    with torch.no_grad():
        total, policy, value = training_loss(*trainer.model(states), policies, outcomes, return_components=True)
    loss = trainer.step(examples)
    assert loss == pytest.approx(total.item())
    assert trainer.last_metrics['policy_loss'] == pytest.approx(policy.item())
    assert trainer.last_metrics['value_loss'] == pytest.approx(value.item())
    assert trainer.last_metrics['gradient_norm'] > 0


def test_rng_provenance_json_serializable_and_independent_sampling():
    search, sampling = Random(42), Random(42)
    initial = experiment.rng_states(search, sampling)
    sampling.sample(range(50), 32)
    final = experiment.rng_states(search, sampling)
    assert initial['search'] == final['search'] and initial['training_sampling'] != final['training_sampling']
    assert all(name in initial for name in ('python', 'numpy', 'torch', 'search', 'training_sampling'))
    json.dumps(initial, allow_nan=False)
