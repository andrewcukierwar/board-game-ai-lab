"""Focused ResNet architecture, identity, serialization and factory-integration cases.

Small synthetic fixtures only: no benchmark, tournament or training experiment.
tests/test_alphazero_resnet.py runs this file in its own one-thread interpreter.
"""
from itertools import permutations
from pathlib import Path
import random
import subprocess

import pytest
import torch

from games.connect4.connect4 import Connect4
from games.connect4.alphazero_v2 import artifacts, experimental as E, experimental_models as M
from games.connect4.alphazero_v2 import network, resnet as R
from games.connect4.alphazero_v2.bitboard import BitboardState
from games.connect4.alphazero_v2.config import ARCHITECTURE, INFERENCE_CONTRACT, V2Config
from games.connect4.alphazero_v2.generation import initial_model
from games.connect4.alphazero_v2.provenance import configure_deterministic_runtime
from games.connect4.alphazero_v2.search import EvaluationAgent, PUCTSearch
from games.connect4.alphazero_v2.training import V2Trainer, batch_tensors, training_loss

BASE_COMMIT = '9325f0a19e6b0106ea1dcd265316b57323f8eb4f'
RESNETS = (R.RESNET3X64, R.RESNET5X64)
PARAMETERS = {R.RESNET3X64: 322_440, R.RESNET5X64: 470_408}
BLOCKS = {R.RESNET3X64: 3, R.RESNET5X64: 5}
FULL_COLUMN = (0,) * 6  # alternating colors fill column 0 without a win
POSITIONS = ((), (3,), (3, 3, 4), FULL_COLUMN, (0, 1, 2, 3, 4, 5, 6, 0), (3, 2, 3, 2, 3, 2))
each = pytest.mark.parametrize('architecture', RESNETS)


@pytest.fixture(autouse=True, scope='module')
def runtime():
    configure_deterministic_runtime(1)


def build(architecture, seed=7):
    return M.model_factory(architecture).construct(seed)


def boards(n, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-1, 2, (n, 1, 6, 7), generator=generator).to(torch.float32)


def position(moves):
    game = Connect4()
    for move in moves:
        assert game.make_move(move)
    return game


def examples():
    """Fourteen labeled positions from two synthetic X wins (vertical and horizontal)."""
    result = []
    for index, moves in enumerate(((0, 1, 0, 1, 0, 1, 0), (3, 3, 4, 4, 5, 5, 6))):
        visits = [[2 if a == m else 1 if a in ((m + 1) % 7, (m + 2) % 7) else 0 for a in range(7)] for m in moves]
        record = dict(index=index, moves=list(moves), winner=0, visits=visits,
                      action_temperatures=[E.ORIGINAL.temperature(ply) for ply in range(len(moves))])
        result.extend(E.rebuild_scheduled_game(1, record, E.ORIGINAL, 4).examples)
    return result


def trained(architecture, steps=1):
    """A model whose weights and batch-norm statistics have left initialization."""
    factory = M.model_factory(architecture)
    model = factory.construct(7)
    trainer = factory.trainer(model, V2Config())
    for _ in range(steps):
        trainer.step(examples())
    return factory, model, trainer


def count(module):
    return sum(p.numel() for p in module.parameters())


@each
def test_shapes_and_input_contract(architecture):
    model = build(architecture)
    for n in (1, 8, 128):
        policy, value = model(boards(n))
        assert policy.shape == (n, 7) and value.shape == (n, 1)
        assert policy.dtype == value.dtype == torch.float32
        assert torch.isfinite(policy).all() and (value.abs() <= 1).all()
    for bad in (boards(2).double(), torch.zeros(2, 2, 6, 7), torch.zeros(1, 6, 7), torch.zeros(1, 1, 7, 6),
                boards(1) * 2, torch.full((1, 1, 6, 7), float('nan')), boards(1).numpy()):
        with pytest.raises(ValueError, match='Input'):
            model(bad)


@each
def test_parameter_counts_and_specification(architecture):
    model, spec = build(architecture), R.RESNET_SPECS[architecture]
    assert spec == R.ResNetSpec(blocks=BLOCKS[architecture], channels=64, policy_channels=32,
                                value_channels=32, value_hidden=64)
    assert count(model) == PARAMETERS[architecture]
    assert count(model.stem_conv) + count(model.stem_bn) == 704
    assert len(model.tower) == spec.blocks and all(count(block) == 73_984 for block in model.tower)
    assert count(model.policy_conv) + count(model.policy_bn) + count(model.policy_fc) == 11_527
    assert sum(count(m) for m in (model.value_conv, model.value_bn, model.value_fc1, model.value_fc2)) == 88_257
    convolutions = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert len(convolutions) == len(norms) == 2 * spec.blocks + 3
    assert all(c.bias is None for c in convolutions)
    assert all(n.momentum == 0.1 and n.eps == 1e-5 and n.track_running_stats for n in norms)
    # The three-block trunk matches the CNN's budget; the heads are the CNN's.
    assert count(initial_model(7)) == 325_896


def test_residual_block_adds_its_branch_to_an_identity_skip():
    block = R.ResidualBlock(64).eval()
    x = torch.randn(2, 64, 6, 7, generator=torch.Generator().manual_seed(0))
    branch = block.bn2(block.conv2(torch.relu(block.bn1(block.conv1(x)))))
    assert branch.abs().max() > 0 and torch.equal(block(x), torch.relu(x + branch))
    with torch.no_grad():
        block.conv2.weight.zero_()  # a silent branch leaves exactly the rectified skip path
    assert torch.equal(block(x), torch.relu(x))


@each
def test_initialization(architecture):
    before = torch.get_rng_state()
    model = build(architecture, 42)
    assert torch.equal(torch.get_rng_state(), before)
    factory = M.model_factory(architecture)
    assert factory.hash(model) == factory.hash(build(architecture, 42)) != factory.hash(build(architecture, 43))
    assert not model.training and all(p.requires_grad for p in model.parameters())
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert (module.weight == 1).all() and (module.bias == 0).all(), name
            assert (module.running_mean == 0).all() and (module.running_var == 1).all(), name
            assert int(module.num_batches_tracked) == 0, name
        elif isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            bound = module.weight[0].numel() ** -0.5  # PyTorch default: U(-1/sqrt(fan_in), 1/sqrt(fan_in))
            assert module.weight.abs().max() <= bound and module.weight.std() > 0.3 * bound, name
    policy, value = model(boards(64))
    assert torch.softmax(policy, 1).max() < 0.5 and value.abs().max() < 0.9


def test_initialization_is_separated_by_architecture():
    hashes = {a: network.weights_sha256(M.model_factory(a).construct(42)) for a in M.SUPPORTED_ARCHITECTURES}
    assert len(set(hashes.values())) == 3
    three, five = build(R.RESNET3X64, 42).state_dict(), build(R.RESNET5X64, 42).state_dict()
    assert not torch.equal(three['stem_conv.weight'], five['stem_conv.weight'])


@each
def test_forward_determinism_and_batch_independence(architecture):
    factory, model, _ = trained(architecture)
    states, before = boards(8), factory.hash(model)
    policy, value = model(states)
    again = model(states)
    assert torch.equal(policy, again[0]) and torch.equal(value, again[1])
    restored = factory.restore(factory.state(model))(states)
    assert torch.equal(policy, restored[0]) and torch.equal(value, restored[1])
    for row in range(8):  # eval-mode batch norm never mixes rows
        single = model(states[row:row + 1])
        assert torch.allclose(single[0][0], policy[row], rtol=0, atol=1e-5)
        assert torch.allclose(single[1][0], value[row], rtol=0, atol=1e-5)
    assert factory.hash(model) == before  # inference leaves running statistics untouched


@each
def test_value_stays_bounded(architecture):
    model = build(architecture)
    with torch.no_grad():
        model.value_fc2.weight.mul_(1e6)
        model.value_fc2.bias.fill_(1e6)
    for mode in (model.eval, model.train):
        mode()
        value = model(boards(32))[1]
        assert torch.isfinite(value).all() and (value.abs() <= 1).all() and (value.abs() == 1).any()


@each
def test_backpropagation_reaches_every_parameter(architecture):
    model = build(architecture).train()
    states, policies, outcomes = batch_tensors(examples())
    combined, policy_loss, value_mse = training_loss(*model(states), policies, outcomes)
    assert torch.isfinite(combined) and combined == policy_loss + value_mse
    combined.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@each
def test_legal_policy_masking(architecture):
    factory, model, _ = trained(architecture)
    inference = factory.inference(model)
    prediction = inference.predict(position(FULL_COLUMN))
    assert len(prediction.logits) == 7 and prediction.policy[0] == 0.0
    assert all(p > 0 for p in prediction.policy[1:]) and abs(sum(prediction.policy) - 1) < 1e-12
    assert -1 <= prediction.value <= 1
    with torch.no_grad():  # an illegal logit, however large, cannot move legal probabilities
        model.policy_fc.bias[0] = 50.0
    shifted = inference.predict(position(FULL_COLUMN))
    assert shifted.logits[0] > 40 and shifted.policy[0] == 0.0
    assert shifted.policy == pytest.approx(prediction.policy, abs=1e-12)
    with pytest.raises(ValueError, match='terminal'):
        inference.predict(position((0, 1, 0, 1, 0, 1, 0)))
    model.train()
    with pytest.raises(ValueError, match='eval'):
        inference.predict(position(()))


@each
def test_batched_inference_matches_single_predictions(architecture):
    factory, model, _ = trained(architecture)
    inference = factory.inference(model)
    states = [BitboardState.from_game(position(moves)) for moves in POSITIONS]
    batch = inference.predict_batch(states)
    assert inference.predict_batch([]) == () and len(batch) == len(states)
    for state, predicted in zip(states, batch):
        single = inference.predict(state)
        legal = set(state.get_valid_moves())
        assert all((predicted.policy[a] > 0) == (a in legal) for a in range(7))
        assert abs(sum(predicted.policy) - 1) < 1e-12 and -1 <= predicted.value <= 1
        assert predicted.logits == pytest.approx(single.logits, abs=1e-5)
        assert predicted.policy == pytest.approx(single.policy, abs=1e-5)
        assert predicted.value == pytest.approx(single.value, abs=1e-5)
    assert inference.predict_batch(states) == batch
    with pytest.raises(ValueError):
        inference.predict_batch([BitboardState.from_game(position((0, 1, 0, 1, 0, 1, 0)))])


@each
@pytest.mark.parametrize('backend', ['engine', 'bitboard'])
def test_neural_puct_accepts_resnet_inference(architecture, backend):
    inference = M.model_factory(architecture).inference(build(architecture))
    search = PUCTSearch(inference, 24, backend=backend)
    assert search.inference is inference and isinstance(inference, network.V2Inference)
    result = search.run(position(FULL_COLUMN), rng=random.Random(1))
    assert sum(result.visits) == 24 and result.visits[0] == 0
    again = search.run(position(FULL_COLUMN), rng=random.Random(1))
    assert again.visits == result.visits
    assert EvaluationAgent(inference, 8, backend=backend).choose_move(position(FULL_COLUMN)) in range(1, 7)


@each
def test_serialization_round_trip(architecture, tmp_path):
    factory, model, _ = trained(architecture, steps=2)
    state = factory.state(model)
    assert all(not t.requires_grad for t in state.values())
    torch.save(state, tmp_path / 'state.pt')
    restored = factory.restore(torch.load(tmp_path / 'state.pt', map_location='cpu', weights_only=True))
    assert not restored.training and restored.alphazero_v2_contract == architecture
    assert factory.hash(restored) == factory.hash(model)
    states = boards(8)
    assert all(torch.equal(a, b) for a, b in zip(model(states), restored(states)))
    state['stem_conv.weight'].zero_()  # the saved state is a detached copy
    assert factory.hash(restored) == factory.hash(model)

    path = tmp_path / 'inference.pt'
    assert R.save_inference_checkpoint(path, model) == artifacts.file_sha256(path)
    loaded = R.load_inference_checkpoint(path, architecture)
    assert isinstance(loaded, R.ResNetInference) and factory.hash(loaded.model) == factory.hash(model)
    stored = torch.load(path, map_location='cpu', weights_only=True)
    assert stored['contract'] == factory.output_contract == R.inference_contract(architecture)
    with pytest.raises(FileExistsError):
        R.save_inference_checkpoint(path, model)


def test_architecture_identities_are_explicit_and_distinct():
    assert M.SUPPORTED_ARCHITECTURES == (ARCHITECTURE, R.RESNET3X64, R.RESNET5X64)
    assert len(set(M.SUPPORTED_ARCHITECTURES)) == 3
    for architecture in RESNETS:
        factory = M.model_factory(architecture)
        contract = factory.output_contract
        assert factory.architecture == contract['architecture'] == architecture
        assert build(architecture).alphazero_v2_contract == architecture
        assert contract['architecture_spec'] == dict(blocks=BLOCKS[architecture], channels=64, policy_channels=32,
                                                     value_channels=32, value_hidden=64)
        # Everything except identity/specification is the CNN's inference contract.
        assert {k: v for k, v in contract.items() if k in INFERENCE_CONTRACT and k != 'architecture'} == \
               {k: v for k, v in INFERENCE_CONTRACT.items() if k != 'architecture'}
        assert set(contract) - set(INFERENCE_CONTRACT) == {'architecture_spec', 'normalization'}


def test_cross_architecture_models_and_states_are_rejected():
    models = {a: M.model_factory(a).construct(3) for a in M.SUPPORTED_ARCHITECTURES}
    for target, source in permutations(M.SUPPORTED_ARCHITECTURES, 2):
        factory, other = M.model_factory(target), M.model_factory(source)
        with pytest.raises(ValueError):
            factory.restore(other.state(models[source]))
        for operation in (factory.state, factory.freeze, factory.inference,
                          lambda model: factory.trainer(model, V2Config())):
            with pytest.raises(ValueError):
                operation(models[source])
    for architecture in RESNETS:  # direct CNN-only entry points
        with pytest.raises(ValueError):
            network.V2Inference(models[architecture])
        with pytest.raises(ValueError):
            V2Trainer(models[architecture])
        with pytest.raises(ValueError):
            R.ResNetInference(models[ARCHITECTURE])
        with pytest.raises(ValueError):
            R.ResNetTrainer(models[ARCHITECTURE])


def test_cross_architecture_checkpoints_are_rejected(tmp_path):
    paths = {}
    for architecture in RESNETS:
        paths[architecture] = tmp_path / f'{BLOCKS[architecture]}.pt'
        R.save_inference_checkpoint(paths[architecture], build(architecture))
    paths[ARCHITECTURE] = tmp_path / 'cnn.pt'
    network.save_inference_checkpoint(paths[ARCHITECTURE], initial_model(3))
    for architecture in RESNETS:
        for source in M.SUPPORTED_ARCHITECTURES:
            if source != architecture:
                with pytest.raises(ValueError, match='inference checkpoint'):
                    R.load_inference_checkpoint(paths[source], architecture)
        with pytest.raises(ValueError, match='v2 inference checkpoint'):
            network.load_inference_checkpoint(paths[architecture])
        with pytest.raises(ValueError, match='Unregistered'):
            R.load_inference_checkpoint(paths[architecture], ARCHITECTURE)
        # A correctly labeled contract cannot carry another architecture's tensors.
        other = next(a for a in RESNETS if a != architecture)
        forged = dict(contract=R.inference_contract(architecture), model_state_dict=R.model_state(build(other)))
        torch.save(forged, tmp_path / f'forged-{BLOCKS[architecture]}.pt')
        with pytest.raises(ValueError, match='State keys|Invalid'):
            R.load_inference_checkpoint(tmp_path / f'forged-{BLOCKS[architecture]}.pt', architecture)
    assert isinstance(network.load_inference_checkpoint(paths[ARCHITECTURE]), network.V2Inference)


@each
def test_relabelled_models_are_rejected(architecture):
    other = next(a for a in RESNETS if a != architecture)
    model = build(architecture)
    for label in (other, ARCHITECTURE, 'connect4-resnet4x64', None):
        model.alphazero_v2_contract = label
        with pytest.raises(ValueError):
            R.require_resnet_model(model)
        with pytest.raises(ValueError):
            R.ResNetInference(model)
    model.alphazero_v2_contract = architecture
    R.require_resnet_model(model, architecture)

    class Subclass(R.AlphaZeroResNet):
        pass
    with pytest.raises(ValueError):
        R.require_resnet_model(Subclass(architecture).eval())


@each
@pytest.mark.parametrize('mutation', ['missing', 'extra', 'shape', 'dtype', 'nan', 'infinite', 'negative_variance',
                                      'negative_counter', 'float_counter', 'not_tensor', 'not_dict'])
def test_invalid_states_are_rejected(architecture, mutation):
    factory = M.model_factory(architecture)
    state = factory.state(build(architecture))
    if mutation == 'missing': del state['tower.0.bn1.running_mean']
    if mutation == 'extra': state['tower.9.conv1.weight'] = torch.zeros(64, 64, 3, 3)
    if mutation == 'shape': state['stem_conv.weight'] = torch.zeros(64, 2, 3, 3)
    if mutation == 'dtype': state['policy_fc.bias'] = state['policy_fc.bias'].double()
    if mutation == 'nan': state['value_fc2.weight'][0, 0] = float('nan')
    if mutation == 'infinite': state['stem_bn.running_mean'][0] = float('inf')
    if mutation == 'negative_variance': state['value_bn.running_var'][0] = -1e-3
    if mutation == 'negative_counter': state['policy_bn.num_batches_tracked'] = torch.tensor(-1)
    if mutation == 'float_counter': state['policy_bn.num_batches_tracked'] = torch.tensor(0.0)
    if mutation == 'not_tensor': state['value_fc1.bias'] = [0.0] * 64
    if mutation == 'not_dict': state = list(state.items())
    with pytest.raises(ValueError):
        factory.restore(state)


@each
def test_optimizer_groups_and_settings(architecture):
    model = build(architecture)
    config = V2Config()
    decay, no_decay = R.parameter_groups(model)
    parameters = dict(model.named_parameters())
    assert sorted(decay + no_decay) == sorted(parameters) and not set(decay) & set(no_decay)
    assert all(parameters[name].ndim > 1 for name in decay)
    assert all(parameters[name].ndim == 1 for name in no_decay)
    assert len(decay) == 2 * BLOCKS[architecture] + 6 and len(no_decay) == 2 * (2 * BLOCKS[architecture] + 3) + 3
    optimizer = M.model_factory(architecture).trainer(model, config).optimizer
    reference = V2Trainer(initial_model(7), config).optimizer
    assert type(optimizer) is type(reference) is torch.optim.AdamW
    for group, expected in zip(optimizer.param_groups, reference.param_groups, strict=True):
        assert {k: v for k, v in group.items() if k != 'params'} == {k: v for k, v in expected.items() if k != 'params'}
    assert [g['weight_decay'] for g in optimizer.param_groups] == [config.weight_decay, 0.0]


@each
def test_trainer_step_updates_weights_and_statistics(architecture):
    factory = M.model_factory(architecture)
    model = factory.construct(7)
    trainer = factory.trainer(model, V2Config())
    assert type(trainer) is R.ResNetTrainer
    before, mean = factory.hash(model), model.stem_bn.running_mean.clone()
    metrics = [trainer.step(examples()) for _ in range(8)]
    assert set(metrics[0]) == set(V2Trainer(initial_model(7), V2Config()).step(examples()))
    assert all(torch.isfinite(torch.tensor(list(m[k] for k in m if k != 'clipped'))).all() for m in metrics)
    assert metrics[-1]['combined_loss'] < metrics[0]['combined_loss']
    assert not model.training and not trainer.failed and trainer.steps == 8
    assert factory.hash(model) != before and not torch.equal(model.stem_bn.running_mean, mean)
    assert all(int(b) == 8 for n, b in model.named_buffers() if n.endswith('num_batches_tracked'))
    assert all(p.grad is None for p in model.parameters())
    factory.inference(model).predict(position(()))  # still a valid eval-mode inference model
    factory.restore_optimizer(trainer, trainer.optimizer.state_dict(), 8)
    with torch.no_grad():
        model.policy_bn.num_batches_tracked += 1
    with pytest.raises(ValueError, match='Batch-norm update counters'):
        factory.restore_optimizer(trainer, trainer.optimizer.state_dict(), 8)


@each
def test_any_failed_step_invalidates_the_trainer(architecture, monkeypatch):
    factory = M.model_factory(architecture)
    model = factory.construct(7)
    trainer = factory.trainer(model, V2Config())
    before = factory.hash(model)

    def fail():
        raise KeyboardInterrupt('optimizer')
    monkeypatch.setattr(trainer.optimizer, 'step', fail)
    with pytest.raises(KeyboardInterrupt):
        trainer.step(examples())
    # The forward already moved running statistics although no optimizer step happened.
    assert trainer.failed and trainer.steps == 0 and not model.training
    assert factory.hash(model) != before and int(model.stem_bn.num_batches_tracked) == 1
    with pytest.raises(RuntimeError, match='failed'):
        trainer.step(examples())
    empty = factory.trainer(factory.construct(7), V2Config())
    with pytest.raises(ValueError):
        empty.step([])
    assert empty.failed


def test_model_factory_integration():
    assert isinstance(M.model_factory(ARCHITECTURE), M.CNNFactory)
    training = V2Config(seed=123, games_per_generation=8, replay_generations=2, replay_max_games=16)
    assert E.ExperimentalConfig(training, 1).architecture == ARCHITECTURE
    for architecture in RESNETS:
        factory = M.model_factory(architecture)
        assert type(factory) is R.ResNetFactory and factory.spec == R.RESNET_SPECS[architecture]
        for name in ('construct', 'restore', 'state', 'hash', 'freeze', 'inference', 'trainer',
                     'restore_optimizer'):
            assert callable(getattr(factory, name)) and callable(getattr(M.CNNFactory(), name))
        model = factory.construct(5)
        snapshot = factory.freeze(model)
        assert snapshot is not model and not snapshot.training
        assert not any(p.requires_grad for p in snapshot.parameters()) and factory.hash(snapshot) == factory.hash(model)
        config = E.ExperimentalConfig(training, 3, architecture=architecture)
        assert E.ExperimentalConfig.from_dict(config.to_dict()) == config
    for unknown in ('resnet', 'connect4-resnet4x64-bn-policy32-value32x64-scalar-tanh-v1',
                    R.RESNET3X64.upper(), '', None, 3, (R.RESNET3X64,)):
        with pytest.raises(ValueError, match='Unsupported'):
            M.model_factory(unknown)
        with pytest.raises(ValueError):
            E.ExperimentalConfig(training, 1, architecture=unknown)


def lifecycle_config(architecture):
    training = V2Config(seed=123, games_per_generation=8, self_play_simulations=2, batch_size=4,
                        replay_generations=2, replay_max_games=16, max_generations=3)
    return E.ExperimentalConfig(training, optimizer_updates=2, architecture=architecture)


@each
def test_experimental_lifecycle_and_exact_resume(architecture, tmp_path):
    config = lifecycle_config(architecture)
    runner = E.ExperimentalRunner(config, lineage='test', arm='resnet')
    assert type(runner.model) is R.AlphaZeroResNet and type(runner.trainer) is R.ResNetTrainer
    path = tmp_path / 'generation-1.pt'
    first = runner.run_generation(path)
    collection = runner.history[-1]['collection']
    assert collection['max_active'] == 8 and collection['batch_sizes'][8] > 0  # pool-eight batched inference
    assert first['collection_weights_sha256'] == runner.factory.hash(runner.factory.construct(123))
    assert first['learner_weights_sha256'] == runner.factory.hash(runner.model) != first['collection_weights_sha256']
    assert all(int(b) == 2 for n, b in runner.model.named_buffers() if n.endswith('num_batches_tracked'))

    sha = artifacts.file_sha256(path)
    loaded = E.load_experimental(path, expected_sha256=sha, expected_config=config)
    assert loaded.model.alphazero_v2_contract == architecture and loaded.trainer.steps == 2
    assert loaded.factory.hash(loaded.model) == runner.factory.hash(runner.model)
    assert E.digest(loaded.trainer.optimizer.state_dict()) == E.digest(runner.trainer.optimizer.state_dict())
    assert loaded.replay.digest() == runner.replay.digest()
    assert E.evaluate_boundary(path, lambda model, counters: (type(model), model.training, counters),
                               expected_sha256=sha) == (R.AlphaZeroResNet, False, runner.counters())
    a, b = runner.run_generation(tmp_path / 'a.pt'), loaded.run_generation(tmp_path / 'b.pt')
    for result in (a, b):
        result.pop('resources'), result.pop('boundary_sha256')
    assert a == b and runner.factory.hash(runner.model) == loaded.factory.hash(loaded.model)


@each
@pytest.mark.parametrize('mutation', ['other_resnet', 'cnn', 'unknown', 'weight', 'statistic', 'counter'])
def test_tampered_boundaries_are_rejected(architecture, mutation, tmp_path):
    runner = E.ExperimentalRunner(lifecycle_config(architecture), lineage='test', arm='resnet')
    runner.save(tmp_path / 'boundary.pt')
    payload = torch.load(tmp_path / 'boundary.pt', map_location='cpu', weights_only=True)
    if mutation == 'other_resnet': payload['config']['architecture'] = next(a for a in RESNETS if a != architecture)
    if mutation == 'cnn': payload['config']['architecture'] = ARCHITECTURE
    if mutation == 'unknown': payload['config']['architecture'] = 'connect4-resnet'
    if mutation == 'weight': payload['model']['tower.0.conv1.weight'][0, 0, 0, 0] += 1
    if mutation == 'statistic': payload['model']['stem_bn.running_var'][0] += 1
    if mutation == 'counter': payload['model']['value_bn.num_batches_tracked'] += 1
    payload['integrity'] = E.digest({k: v for k, v in payload.items() if k != 'integrity'})
    torch.save(payload, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError):
        E.load_experimental(tmp_path / 'tampered.pt')
    E.load_experimental(tmp_path / 'boundary.pt')


def test_cnn_remains_the_unchanged_default():
    root = Path(__file__).resolve().parents[1]
    package = root / 'games/connect4/alphazero_v2'
    changed = {'experimental_models.py', 'resnet.py'}
    for path in sorted(package.rglob('*.py')):
        if path.name in changed:
            continue
        relative = path.relative_to(root).as_posix()
        original = subprocess.run(['git', 'show', f'{BASE_COMMIT}:{relative}'], check=True,
                                  capture_output=True, cwd=root).stdout
        assert path.read_bytes() == original, relative
    factory = M.model_factory(ARCHITECTURE)
    assert (factory.architecture, factory.output_contract) == (ARCHITECTURE, INFERENCE_CONTRACT)
    assert M.CNNFactory.construct is initial_model and M.CNNFactory.restore is network.model_from_state
    assert M.CNNFactory.inference is network.V2Inference and M.CNNFactory.trainer is V2Trainer
    model = factory.construct(123)
    assert type(model) is network.AlphaZeroV2Net and count(model) == 325_896
    assert factory.hash(model) == network.weights_sha256(initial_model(123))
    assert factory.hash(factory.restore(factory.state(model))) == factory.hash(model)
    assert type(factory.inference(factory.freeze(model))) is network.V2Inference
    assert type(factory.trainer(model, V2Config())) is V2Trainer
