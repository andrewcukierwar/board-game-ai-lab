"""AlphaZero v2 (Phase 4D.3A) contracts with fake networks and tiny synthetic budgets.

Optimizer updates here are discarded correctness checks on synthetic data; no
research training, learned checkpoint or strength measurement occurs.
"""
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import json
import math
import os
from pathlib import Path
import random
from random import Random
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from games.connect4.connect4 import Connect4
from games.connect4.agents.mcts_nn_agent import Node
from games.connect4 import neural_mcts as v1
from games.connect4.train_mcts_nn import NeuralTrainer as V1Trainer
from games.connect4.alphazero_v2 import generation as generation_module
from games.connect4.alphazero_v2.artifacts import atomic_torch_save
from games.connect4.alphazero_v2.config import (
    ARCHITECTURE, INFERENCE_CONTRACT, PUCT_EXPLORATION, RESUME_CONTRACT, V2Config, action_temperature,
)
from games.connect4.alphazero_v2.data import (
    GenerationReplay, ReflectionAugmenter, V2Example, capture_example, encode_board,
    finalize_game,
    outcome_for_actor, pre_move_ply, reflect_example, visit_target,
)
from games.connect4.alphazero_v2.generation import GenerationRunner, load_resume_boundary
from games.connect4.alphazero_v2.network import (
    AlphaZeroV2Net, V2Inference, legal_policy_from_logits, load_inference_checkpoint,
    save_inference_checkpoint, weights_sha256,
)
from games.connect4.alphazero_v2.search import (
    EvaluationAgent, PUCTSearch, SelfPlayer, V2RootNoise, select_action,
)
from games.connect4.alphazero_v2.training import (
    V2Trainer, batch_tensors, parameter_groups, training_loss,
)

ROOT = Path(__file__).resolve().parents[1]
X_WIN = [0, 1, 0, 1, 0, 1, 0]
O_WIN = [0, 1, 0, 1, 2, 1, 2, 1]
DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]
WIN_IN_ONE = [([0, 1, 0, 1, 0, 2], 0, 0), ([0, 1, 0, 1, 2, 1, 2], 1, 1)]


@pytest.fixture(autouse=True, scope='module')
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def position(moves):
    game = Connect4()
    for move in moves:
        assert not game.is_game_over() and game.make_move(move)
    return game


def tiny_config(**changes):
    values = dict(self_play_simulations=3, games_per_generation=2, replay_generations=2,
                  replay_max_games=4, batch_size=8, max_generations=6)
    values.update(changes)
    return V2Config(**values)


class FakeV2(torch.nn.Module):
    """Fixed raw logits; value from an optional function of the canonical input."""
    alphazero_v2_contract = ARCHITECTURE

    def __init__(self, logits=None, value_fn=None):
        super().__init__()
        self.register_buffer('logits', torch.tensor([0.] * 7 if logits is None else logits))
        self.value_fn = value_fn
        self.calls = 0
        self.eval()

    def forward(self, states):
        self.calls += 1
        n = states.shape[0]
        value = (torch.zeros(n, 1) if self.value_fn is None
                 else torch.tensor([[float(self.value_fn(s[0]))] for s in states]))
        return self.logits.expand(n, 7).clone(), value


def pending_examples(moves):
    """Synthetic legal visit records for a history (one visit on the played move)."""
    game, examples = Connect4(), []
    for move in moves:
        visits = [0] * 7
        visits[move] = 1
        examples.append(V2Example(encode_board(game), game.current_player, pre_move_ply(game),
                                  tuple(visits), move, 1.0))
        assert game.make_move(move)
    return examples


def completed(generation, index, moves=X_WIN):
    return finalize_game(generation, index, moves, pending_examples(moves))


# Contracts and configuration ---------------------------------------------------

def test_default_configuration_matches_review():
    c = V2Config()
    assert (c.self_play_simulations, c.evaluation_simulations, c.interactive_simulations) == (256, 512, 512)
    assert (c.root_noise_epsilon, c.root_dirichlet_alpha, c.exploratory_plies) == (.25, 1.0, 8)
    assert (c.games_per_generation, c.replay_generations, c.replay_max_games) == (256, 8, 2048)
    assert (c.batch_size, c.samples_per_new_position) == (128, 4)
    assert (c.learning_rate, c.adam_betas, c.adam_eps, c.weight_decay, c.max_grad_norm) == (
        3e-4, (.9, .999), 1e-8, 1e-4, 5.0)
    assert c.reflection_probability == .5 and PUCT_EXPLORATION == 1.41
    assert V2Config.from_dict(json.loads(json.dumps(c.to_dict()))) == c


@pytest.mark.parametrize('new_positions,updates', [(1, 1), (32, 1), (33, 2), (5376, 168), (10752, 336)])
def test_update_budget_is_ceil_four_per_new_position(new_positions, updates):
    assert V2Config().updates_for(new_positions) == updates == math.ceil(4 * new_positions / 128)


@pytest.mark.parametrize('changes', [dict(self_play_simulations=0), dict(batch_size=True),
                                     dict(replay_max_games=100), dict(root_dirichlet_alpha=0),
                                     dict(root_noise_epsilon=1.5), dict(learning_rate=float('nan')),
                                     dict(adam_betas=(.9, 1.0)), dict(exploratory_plies=-1),
                                     dict(reflection_probability=2)])
def test_invalid_configuration_rejected(changes):
    with pytest.raises(ValueError):
        V2Config(**changes)


def test_temperature_schedule_boundary_ply_7_vs_8():
    assert [action_temperature(p, 8) for p in (0, 7, 8, 41)] == [1.0, 1.0, 0.0, 0.0]
    with pytest.raises(ValueError):
        action_temperature(42, 8)


# Network and legal-logit inference ---------------------------------------------

def test_v2_architecture_scalar_tanh_shape_and_range():
    model = AlphaZeroV2Net().eval()
    assert sum(p.numel() for p in model.parameters()) == 325896
    assert len(model.state_dict()) == 16 and model.value_fc2.out_features == 1
    states = torch.randint(-1, 2, (5, 1, 6, 7)).float()
    with torch.no_grad():
        policy, value = model(states)
        assert policy.shape == (5, 7) and value.shape == (5, 1)
        assert ((value >= -1) & (value <= 1)).all()
        model.value_fc2.bias.fill_(1e4)
        _, saturated = model(states)
    assert torch.equal(saturated, torch.ones(5, 1))


def test_value_is_for_the_current_player_to_move():
    # +0.5 when the player to move owns bottom-center, -0.5 when the opponent does.
    inference = V2Inference(FakeV2(value_fn=lambda s: 0.5 * float(s[5, 3])))
    assert inference.predict(position([3])).value == -0.5      # O to move; X owns center
    assert inference.predict(position([3, 0])).value == 0.5    # X to move; X owns center
    assert v1.encode_current_player(position([3]))[0, 0, 5, 3] == -1


def test_pathological_illegal_logit_cannot_destroy_legal_probabilities():
    logits = [1000., 0, 1, 2, 3, 4, 5]
    legal = [1, 2, 3, 4, 5, 6]
    policy = legal_policy_from_logits(logits, legal)
    expected = np.exp(np.arange(6.)) / np.exp(np.arange(6.)).sum()
    assert policy[0] == 0
    assert policy[1:] == pytest.approx(expected, rel=1e-12)
    assert policy[1:] == pytest.approx([.00427, .01161, .03155, .08576, .23312, .63369], abs=1e-5)
    # The historical v1 path (softmax all seven, then mask) collapses to uniform.
    old = v1.legal_policy(torch.softmax(torch.tensor(logits), 0).numpy(), legal)
    assert old[1:] == pytest.approx([1 / 6] * 6)


def test_legal_mask_is_exact_and_stable():
    policy = legal_policy_from_logits([3., -1e30, 1e30, 0, 0, 7, 7], [0, 3, 4, 5, 6])
    assert policy[1] == policy[2] == 0 and policy.sum() == pytest.approx(1, abs=1e-15)
    assert policy[5] == policy[6] and policy[3] == policy[4]
    assert legal_policy_from_logits([0.] * 7, [4]).tolist() == [0, 0, 0, 0, 1, 0, 0]


@pytest.mark.parametrize('logits,legal', [([0.] * 6, [0]), ([float('nan')] + [0.] * 6, [0]),
                                          ([float('inf')] + [0.] * 6, [1]), ([0.] * 7, []),
                                          ([0.] * 7, [0, 0]), ([0.] * 7, [7]), ([0.] * 7, [True])])
def test_legal_mask_rejects_invalid_inputs(logits, legal):
    with pytest.raises(ValueError):
        legal_policy_from_logits(logits, legal)


@pytest.mark.parametrize('moves', [X_WIN, O_WIN, DRAW])
def test_inference_rejects_terminal_states(moves):
    with pytest.raises(ValueError, match='terminal'):
        V2Inference(FakeV2()).predict(position(moves))


def test_inference_rejects_no_legal_actions(monkeypatch):
    game = Connect4()
    monkeypatch.setattr(game, 'get_valid_moves', lambda: [])
    with pytest.raises(ValueError):
        V2Inference(FakeV2()).predict(game)


def test_inference_requires_eval_and_rejects_bad_outputs():
    model = FakeV2()
    model.train()
    with pytest.raises(ValueError, match='eval'):
        V2Inference(model)
    for bad in (lambda s: 2.0, lambda s: float('nan')):
        with pytest.raises(ValueError):
            V2Inference(FakeV2(value_fn=bad)).predict(Connect4())


# Search semantics ---------------------------------------------------------------

def test_parent_selects_by_negated_child_q():
    root = Node(Connect4())
    root.visits = 10
    for move, value_sum in ((2, 2.5), (4, -2.5)):
        after = Connect4()
        after.make_move(move)
        child = Node(after, root, move, 0.5)
        child.visits, child.value_sum = 5, value_sum
        root.children[move] = child
    assert PUCTSearch._select_child(root, Random(0)).move == 4
    assert root.children[4].puct(PUCT_EXPLORATION) == pytest.approx(
        0.5 + 1.41 * 0.5 * math.sqrt(10) / 6)


@pytest.mark.parametrize('prefix', [[], [3]])
def test_leaf_value_sign_alternates_for_both_players(prefix):
    search = PUCTSearch(FakeV2(value_fn=lambda s: 0.6), 1)
    result = search.run(position(prefix), rng=Random(0))
    child = result.root.children[result.visits.index(1)]
    assert child.value_sum == pytest.approx(0.6)    # child's own mover
    assert result.root.value_sum == pytest.approx(-0.6)
    assert -child.q_value == pytest.approx(-0.6)    # what the parent maximizes


@pytest.mark.parametrize('moves,player,win', WIN_IN_ONE)
def test_terminal_win_is_minus_one_for_loser_and_plus_one_for_parent(moves, player, win):
    game = position(moves)
    assert game.current_player == player
    result = PUCTSearch(FakeV2(), 40).run(game, rng=Random(1))
    winning = result.root.children[win]
    assert winning.game_state.check_winner() == player
    assert winning.q_value == -1.0 and -winning.q_value == 1.0
    assert result.visits[win] == max(result.visits)


def test_terminal_draw_contributes_zero():
    result = PUCTSearch(FakeV2(value_fn=lambda s: 0.0), 5).run(position(DRAW[:41]), rng=Random(0))
    assert result.visits == (0, 0, 0, 0, 0, 0, 5) and result.root.children[6].q_value == 0


@pytest.mark.parametrize('budget', [1, 2, 7, 33])
@pytest.mark.parametrize('moves', [[], [3] * 6, DRAW[:38]])
def test_exact_configurable_simulation_accounting(budget, moves):
    game = position(moves)
    before = deepcopy(game.__dict__)
    model = FakeV2()
    result = PUCTSearch(model, budget).run(game, rng=Random(2))
    assert sum(result.visits) == result.root.visits == result.simulations == budget
    assert all(result.visits[a] == 0 for a in range(7) if a not in game.get_valid_moves())
    assert game.__dict__ == before
    assert model.calls <= budget + 1  # root expansion is uncounted initialization


@pytest.mark.parametrize('budget', [0, -1, True, 2.0])
def test_invalid_budgets_rejected(budget):
    with pytest.raises(ValueError):
        PUCTSearch(FakeV2(), budget)


def test_search_rejects_terminal_root():
    with pytest.raises(ValueError, match='terminal'):
        PUCTSearch(FakeV2(), 1).run(position(X_WIN), rng=Random(0))


def test_search_is_deterministic_for_identical_rngs():
    runs = [PUCTSearch(FakeV2(), 25).run(position([3, 3]), rng=Random(9)).visits for _ in range(2)]
    assert runs[0] == runs[1]


# Root noise ----------------------------------------------------------------------

def self_player(config=None, noise_seed=5, model=None):
    config = config or tiny_config(self_play_simulations=6)
    return SelfPlayer(V2Inference(model or FakeV2()), config, search_rng=Random(1), action_rng=Random(2),
                      root_noise=V2RootNoise(config.root_noise_epsilon, config.root_dirichlet_alpha,
                                             seed=noise_seed))


def test_self_play_noise_enabled_evaluation_noise_disabled():
    game = position([0] * 6)
    player = self_player()
    decision = player.decide(game)
    assert decision.search.root_noise is not None and decision.search.root_noise_draw == 1
    assert decision.search.root_noise[0] == 0 and sum(decision.search.root_noise) == pytest.approx(1)
    assert decision.search.search_prior != decision.search.root_prior
    assert decision.search.search_prior == pytest.approx(
        [.75 * p + .25 * n for p, n in zip(decision.search.root_prior, decision.search.root_noise)])
    draws = player.root_noise.draws
    result, selection = EvaluationAgent(V2Inference(FakeV2()), 6, rng=Random(0)).search(game)
    assert result.root_noise is None and result.search_prior == result.root_prior
    assert selection.temperature == 0.0 and player.root_noise.draws == draws


def test_dirichlet_noise_is_root_only():
    logits = [0., 1, 2, 3, 2, 1, 0]
    decision = self_player(model=FakeV2(logits)).decide(Connect4())
    clean = legal_policy_from_logits(logits, range(7))
    root = decision.search.root
    assert [root.children[a].prior_p for a in range(7)] == list(decision.search.search_prior)
    deeper = [n for c in root.children.values() for n in c.children.values()]
    assert deeper and all(n.prior_p == clean[n.move] for n in deeper)


def test_v2_noise_defaults_stream_and_state_roundtrip():
    noise = V2RootNoise(seed=3)
    assert (noise.epsilon, noise.alpha) == (.25, 1.0)
    state = noise.get_state()
    global_state = random.getstate()
    first = noise.mix(np.full(7, 1 / 7), range(7))[1]
    assert random.getstate() == global_state
    noise.set_state(state)
    assert noise.mix(np.full(7, 1 / 7), range(7))[1].tolist() == first.tolist()
    assert V2RootNoise(seed=3).mix(np.full(7, 1 / 7), range(7))[1].tolist() == first.tolist()
    with pytest.raises(ValueError):
        noise.mix([.5, .5, 0, 0, 0, 0, 0], [1, 2])  # prior mass on an illegal action


# Target versus action temperature --------------------------------------------------

def test_policy_target_is_independent_of_action_temperature():
    visits = (0, 3, 5, 5, 2, 1, 0)
    game = position([0] * 6)
    examples = [V2Example(encode_board(game), 0, 6, visits, 2, tau) for tau in (0.0, 1.0, 0.25)]
    assert all(e.policy_target == (0, 3 / 16, 5 / 16, 5 / 16, 2 / 16, 1 / 16, 0) for e in examples)
    hot, cold = select_action(visits, 1.0, Random(0)), select_action(visits, 0.0, Random(0))
    assert hot.distribution == pytest.approx(visit_target(visits))
    assert cold.distribution == (0, 0, .5, .5, 0, 0, 0) and cold.move in (2, 3)


def test_zero_temperature_ties_are_seeded_and_uniform():
    picks = [select_action((0, 0, 4, 4, 0, 0, 0), 0.0, Random(s)).move for s in range(200)]
    assert set(picks) == {2, 3} and 70 < picks.count(2) < 130
    assert picks == [select_action((0, 0, 4, 4, 0, 0, 0), 0.0, Random(s)).move for s in range(200)]


@pytest.mark.parametrize('moves,expected', [([0, 0, 1, 1, 6, 6, 5], 1.0), ([0, 0, 1, 1, 6, 6, 5, 5], 0.0)])
def test_self_play_records_schedule_but_targets_raw_visits(moves, expected):
    game = position(moves)
    decision = self_player(tiny_config(self_play_simulations=5)).decide(game)
    assert decision.ply == len(moves) and decision.action.temperature == expected
    example = capture_example(game, decision)
    assert example.action_temperature == expected and example.action == decision.action.move
    assert example.policy_target == tuple(v / 5 for v in decision.search.visits) == decision.search.visit_target


# Examples, outcomes and reflection ----------------------------------------------

@pytest.mark.parametrize('moves,winner', [(X_WIN, 0), (O_WIN, 1), (DRAW, -1)])
def test_completed_outcomes_for_both_actors(moves, winner):
    game = completed(1, 0, moves)
    assert game.winner == winner
    for example in game.examples:
        expected = 0 if winner == -1 else (1 if example.actor == winner else -1)
        assert example.outcome == expected == outcome_for_actor(winner, example.actor)
    actors = {e.actor: e.outcome for e in game.examples}
    assert actors == ({0: 0, 1: 0} if winner == -1 else {winner: 1, 1 - winner: -1})


def test_unfinished_or_inconsistent_histories_rejected():
    with pytest.raises(ValueError, match='completed'):
        finalize_game(1, 0, X_WIN[:-1], pending_examples(X_WIN[:-1]))
    examples = pending_examples(X_WIN)
    with pytest.raises(ValueError):
        finalize_game(1, 0, X_WIN, examples[:-1])
    with pytest.raises(ValueError):
        finalize_game(1, 0, X_WIN, [examples[1], examples[0]] + examples[2:])
    labeled = completed(1, 0).examples
    with pytest.raises(ValueError):
        finalize_game(1, 0, X_WIN, labeled)


def test_captured_examples_are_immutable():
    game = position([3])
    rows = [list(r) for r in encode_board(game)]
    visits = [0, 0, 1, 2, 0, 0, 0]
    example = V2Example(rows, 1, 1, visits, 3, 1.0)
    rows[5][3], visits[3] = 0, 9
    game.make_move(3)
    assert example.observation[5][3] == -1 and example.visits[3] == 2
    with pytest.raises(FrozenInstanceError):
        example.outcome = 1
    with pytest.raises(TypeError):
        example.observation[0][0] = 1


@pytest.mark.parametrize('changes', [dict(actor=0), dict(ply=2), dict(visits=(0,) * 7),
                                     dict(visits=(1, 0, 0, 0, 0, 0, -1)), dict(action=0),
                                     dict(action_temperature=-1), dict(outcome=2),
                                     dict(visits=(1.0, 0, 0, 1, 0, 0, 0))])
def test_invalid_examples_rejected(changes):
    base = dict(observation=encode_board(position([3])), actor=1, ply=1,
                visits=(0, 0, 1, 2, 0, 0, 0), action=3, action_temperature=1.0)
    with pytest.raises(ValueError):
        V2Example(**dict(base, **changes))


def test_example_rejects_target_mass_on_full_column():
    game = position([0] * 6)
    with pytest.raises(ValueError, match='full column'):
        V2Example(encode_board(game), 0, 6, (1, 1, 0, 0, 0, 0, 0), 1, 1.0)


def test_reflection_of_state_policy_and_outcome():
    example = completed(1, 0, O_WIN).examples[5]
    mirror = reflect_example(example)
    assert mirror.observation == tuple(row[::-1] for row in example.observation)
    assert mirror.policy_target == example.policy_target[::-1]
    assert mirror.action == 6 - example.action
    assert (mirror.outcome, mirror.actor, mirror.ply) == (example.outcome, example.actor, example.ply)
    assert reflect_example(mirror) == example
    with pytest.raises(ValueError):
        reflect_example(pending_examples([3])[0])


def test_reflection_augmenter_probability_and_isolated_stream():
    examples = list(completed(1, 0, DRAW).examples)
    global_state = random.getstate()
    augmenter = ReflectionAugmenter(0.5, seed=11)
    batch, flags = augmenter.batch(examples)
    assert random.getstate() == global_state
    assert 0 < sum(flags) < len(flags)
    assert all(b == (reflect_example(e) if f else e) for b, e, f in zip(batch, examples, flags))
    assert ReflectionAugmenter(0.5, seed=11).batch(examples)[1] == flags
    assert not any(ReflectionAugmenter(0.0, seed=1).batch(examples)[1])
    assert all(ReflectionAugmenter(1.0, seed=1).batch(examples)[1])


# Replay --------------------------------------------------------------------------

def test_replay_retains_and_evicts_whole_generations():
    replay = GenerationReplay(window_generations=3, max_games=6)
    evicted = []
    for generation in range(1, 6):
        moves = X_WIN if generation % 2 else O_WIN
        evicted.append(replay.add_generation(generation, [completed(generation, i, moves) for i in range(2)]))
    assert evicted == [(), (), (), (1,), (2,)]
    assert replay.generations == (3, 4, 5) and replay.games == 6
    assert replay.counts() == {3: dict(games=2, positions=14), 4: dict(games=2, positions=16),
                               5: dict(games=2, positions=14)}
    assert len(replay) == 44
    positions, examples = replay.sample(len(replay), Random(0))
    assert sorted(positions) == list(range(44))
    assert {replay.address(p)[0] for p in positions} == {3, 4, 5}
    assert sorted(map(id, examples)) == sorted(id(e) for g in replay.iter_games() for e in g.examples)


def test_replay_sampling_is_uniform_without_replacement_within_batch():
    replay = GenerationReplay(2, 4)
    replay.add_generation(1, [completed(1, 0), completed(1, 1)])
    positions, _ = replay.sample(10, Random(3))
    assert len(set(positions)) == 10
    with pytest.raises(ValueError):
        replay.sample(15, Random(0))


def test_replay_rejects_incomplete_or_misordered_generations():
    replay = GenerationReplay(2, 2)
    pending = completed(1, 0)
    unlabeled = replace(pending, examples=tuple(replace(e, outcome=None) for e in pending.examples))
    with pytest.raises(ValueError):
        replay.add_generation(1, [unlabeled])
    with pytest.raises(ValueError):
        replay.add_generation(1, [completed(2, 0)])
    with pytest.raises(ValueError):
        replay.add_generation(1, [completed(1, 1)])
    replay.add_generation(2, [completed(2, 0)])
    with pytest.raises(ValueError):
        replay.add_generation(2, [completed(2, 0)])
    with pytest.raises(ValueError):
        replay.add_generation(3, [completed(3, i) for i in range(3)])  # exceeds max_games
    assert replay.generations == (2,) and len(replay) == 7  # failed insertion changed nothing


# Loss and optimizer -----------------------------------------------------------------

def test_loss_is_policy_cross_entropy_plus_value_mse():
    logits = torch.tensor([[2., 0, 0, 0, 0, 0, 50.], [0.] * 7], requires_grad=True)
    values = torch.tensor([[0.5], [-0.25]], requires_grad=True)
    policies = torch.tensor([[.5, .5, 0, 0, 0, 0, 0], [0, 0, 0, 1., 0, 0, 0]])
    outcomes = torch.tensor([[1.], [0.]])
    combined, policy, value = training_loss(logits, values, policies, outcomes)
    log_p = torch.log_softmax(logits.detach(), 1)
    assert policy.item() == pytest.approx((-(.5 * log_p[0, 0] + .5 * log_p[0, 1]) - log_p[1, 3]).item() / 2)
    assert value.item() == pytest.approx(((.5 - 1) ** 2 + .25 ** 2) / 2)
    assert combined.item() == pytest.approx(policy.item() + value.item())
    combined.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad[0, 6] > 0  # illegal raw mass penalized


def test_loss_rejects_masked_infinite_logits_and_bad_targets():
    good = (torch.zeros(1, 7), torch.zeros(1, 1), torch.full((1, 7), 1 / 7), torch.zeros(1, 1))
    with pytest.raises(ValueError):
        training_loss(torch.tensor([[float('-inf')] + [0.] * 6]), *good[1:])
    with pytest.raises(ValueError):
        training_loss(good[0], good[1], torch.zeros(1, 7), good[3])
    with pytest.raises(ValueError):
        training_loss(*good[:3], torch.tensor([[0.5]]))


def test_batch_tensors_carry_visit_targets_and_scalar_z_for_both_actors():
    examples = completed(1, 0, O_WIN).examples[:2]
    states, policies, outcomes = batch_tensors(examples)
    assert states.shape == (2, 1, 6, 7) and policies.shape == (2, 7)
    assert outcomes.tolist() == [[-1.], [1.]]  # X actor lost, O actor won
    with pytest.raises(ValueError):
        batch_tensors(pending_examples([3]))


def test_weight_decay_grouping():
    model = AlphaZeroV2Net()
    decay, no_decay = parameter_groups(model)
    assert all(n.endswith('.weight') for n in decay) and all(n.endswith('.bias') for n in no_decay)
    assert len(decay) == len(no_decay) == 8
    optimizer = V2Trainer(model).optimizer
    assert isinstance(optimizer, torch.optim.AdamW)
    groups = optimizer.param_groups
    assert [g['weight_decay'] for g in groups] == [1e-4, 0.0]
    assert all((g['lr'], g['betas'], g['eps']) == (3e-4, (.9, .999), 1e-8) for g in groups)
    names = {id(p): n for n, p in model.named_parameters()}
    assert [names[id(p)] for p in groups[0]['params']] == decay
    assert [names[id(p)] for p in groups[1]['params']] == no_decay


def seeded_net(seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return AlphaZeroV2Net().eval()


def test_gradient_clipping_and_metrics():
    batch = list(completed(1, 0, DRAW).examples[:16])
    clipped = V2Trainer(seeded_net(), tiny_config(max_grad_norm=1e-3)).step(batch)
    assert clipped['clipped'] and clipped['gradient_norm'] > 1e-3
    assert clipped['clipped_gradient_norm'] == pytest.approx(1e-3, rel=1e-4)
    free = V2Trainer(seeded_net(), tiny_config(max_grad_norm=1e6)).step(batch)
    assert not free['clipped'] and free['clipped_gradient_norm'] == pytest.approx(free['gradient_norm'])
    assert free['gradient_norm'] == pytest.approx(clipped['gradient_norm'])
    assert free['combined_loss'] == pytest.approx(free['policy_loss'] + free['value_mse'])


def test_trainer_restores_eval_and_rejects_nonfinite_before_update(monkeypatch):
    model = seeded_net()
    trainer = V2Trainer(model)
    before = weights_sha256(model)
    original = model.forward

    def poisoned(states):
        policy, value = original(states)
        return policy * float('nan'), value
    monkeypatch.setattr(model, 'forward', poisoned)
    with pytest.raises(ValueError):
        trainer.step(list(completed(1, 0).examples))
    assert not model.training and weights_sha256(model) == before and trainer.steps == 0


def test_v1_and_v2_models_are_mutually_rejected():
    with pytest.raises(ValueError):
        V2Inference(v1.Connect4Net().eval())
    with pytest.raises(ValueError):
        V2Trainer(v1.Connect4Net())
    with pytest.raises(ValueError):
        v1.NeuralInference(AlphaZeroV2Net().eval())
    with pytest.raises(ValueError):
        V1Trainer(AlphaZeroV2Net())


# Inference artifact --------------------------------------------------------------

def test_inference_checkpoint_roundtrip_is_exact_and_minimal(tmp_path):
    model = seeded_net(3)
    path = tmp_path / 'v2.pt'
    digest = save_inference_checkpoint(path, model)
    assert len(digest) == 64
    raw = torch.load(path, map_location='cpu', weights_only=True)
    assert set(raw) == {'contract', 'model_state_dict'} and raw['contract'] == INFERENCE_CONTRACT
    loaded = load_inference_checkpoint(path)
    assert all(torch.equal(t, loaded.model.state_dict()[n]) for n, t in model.state_dict().items())
    for moves in ([], [3], [3, 3, 4]):
        assert V2Inference(model).predict(position(moves)) == loaded.predict(position(moves))
    with pytest.raises(FileExistsError):
        save_inference_checkpoint(path, model)


def test_v1_and_v2_artifacts_reject_each_other(tmp_path):
    v1_path, v2_path = tmp_path / 'v1.pt', tmp_path / 'v2.pt'
    with torch.random.fork_rng(devices=[]):
        v1.save_checkpoint(v1_path, v1.Connect4Net().eval())
    save_inference_checkpoint(v2_path, seeded_net())
    with pytest.raises(ValueError):
        load_inference_checkpoint(v1_path)
    with pytest.raises(ValueError):
        v1.load_checkpoint(v2_path)
    with pytest.raises(ValueError):
        v1.load_historical_network_for_research(v2_path)
    with pytest.raises(ValueError):
        load_resume_boundary(v2_path)
    with pytest.raises(ValueError):
        load_resume_boundary(v1_path)


def test_atomic_save_publishes_nothing_when_validation_fails(tmp_path):
    def reject(path):
        raise ValueError('invalid')
    with pytest.raises(ValueError):
        atomic_torch_save(tmp_path / 'artifact.pt', {'x': torch.ones(1)}, reject)
    assert list(tmp_path.iterdir()) == []


# Generation lifecycle --------------------------------------------------------------

def test_generation_collects_frozen_then_trains(monkeypatch):
    runner = GenerationRunner(tiny_config())
    calls, real_collect = [], generation_module.collect_generation
    real_step = runner.trainer.step

    def spy_collect(player, generation, games):
        learner = weights_sha256(runner.model)
        steps, optimizer_state = runner.trainer.steps, deepcopy(runner.trainer.optimizer.state_dict())
        assert runner.phase == 'collecting'
        assert player.search.inference.model is not runner.model
        assert weights_sha256(player.search.inference.model) == learner
        result = real_collect(player, generation, games)
        assert weights_sha256(runner.model) == learner and runner.trainer.steps == steps
        assert str(runner.trainer.optimizer.state_dict()) == str(optimizer_state)
        calls.append(('collect', generation))
        return result

    def spy_step(batch):
        assert runner.phase == 'training'
        calls.append(('step', runner.completed_generations + 1))
        return real_step(batch)
    monkeypatch.setattr(generation_module, 'collect_generation', spy_collect)
    monkeypatch.setattr(runner.trainer, 'step', spy_step)
    first_collection_hash = weights_sha256(runner.model)
    summary = runner.run_generation()
    second = runner.run_generation()
    assert summary['collection_weights_sha256'] == first_collection_hash
    assert second['collection_weights_sha256'] == summary['learner_weights_sha256'] != first_collection_hash
    assert calls[0] == ('collect', 1) and calls.index(('collect', 2)) == 1 + summary['updates']
    assert all(kind == 'step' for kind, _ in calls[1:1 + summary['updates']])


def test_generation_update_budget_replay_window_and_persistent_optimizer(monkeypatch):
    config = tiny_config(replay_generations=1, replay_max_games=2)
    runner = GenerationRunner(config)
    optimizer = runner.trainer.optimizer
    sampled = []
    real_sample = runner.replay.sample

    def spy_sample(batch_size, rng):
        positions, examples = real_sample(batch_size, rng)
        sampled.append((tuple(runner.replay.generations), {runner.replay.address(p)[0] for p in positions}))
        return positions, examples
    monkeypatch.setattr(runner.replay, 'sample', spy_sample)
    total = 0
    for generation in (1, 2, 3):
        summary = runner.run_generation()
        total += summary['updates']
        assert summary['updates'] == math.ceil(4 * summary['new_positions'] / 8)
        assert summary['games'] == 2 and summary['replay_generations'] == [generation]
        assert summary['evicted_generations'] == ([] if generation == 1 else [generation - 1])
        assert summary['sampled_positions'] == 8 * summary['updates']
        assert runner.trainer.optimizer is optimizer and runner.trainer.steps == total
        assert all(int(s['step']) == total for s in optimizer.state.values())
    assert sampled and all(len(retained) == 1 and eligible == set(retained) for retained, eligible in sampled)


def test_self_play_games_are_complete_with_schedule_and_raw_visit_targets():
    runner = GenerationRunner(tiny_config(self_play_simulations=4))
    runner.run_generation()
    games = list(runner.replay.iter_games())
    assert len(games) == 2
    for game in games:
        assert position(game.moves).is_game_over()
        for example in game.examples:
            assert example.action_temperature == (1.0 if example.ply < 8 else 0.0)
            assert sum(example.visits) == 4
            assert example.policy_target == tuple(v / 4 for v in example.visits)
    assert runner.root_noise.draws == sum(len(g.examples) for g in games)


def test_interrupted_generation_is_discarded_and_runner_must_resume(monkeypatch):
    runner = GenerationRunner(tiny_config())
    runner.run_generation()

    def broken(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(generation_module, 'collect_generation', broken)
    with pytest.raises(KeyboardInterrupt):
        runner.run_generation()
    assert runner.failed and runner.replay.generations == (1,) and runner.completed_generations == 1
    with pytest.raises(RuntimeError):
        runner.run_generation()
    with pytest.raises(RuntimeError):
        runner.boundary_payload()


def test_max_generations_is_enforced():
    runner = GenerationRunner(tiny_config(max_generations=1))
    runner.run_generation()
    with pytest.raises(RuntimeError, match='max_generations'):
        runner.run_generation()


def test_fresh_runners_are_deterministic_and_seeded():
    a, b = GenerationRunner(tiny_config()), GenerationRunner(tiny_config())
    assert weights_sha256(a.model) == weights_sha256(b.model)
    assert weights_sha256(GenerationRunner(tiny_config(seed=7)).model) != weights_sha256(a.model)
    assert a.run_generation() == b.run_generation()


# Resume -----------------------------------------------------------------------------

def test_boundary_files_schema_and_no_overwrite(tmp_path):
    runner = GenerationRunner(tiny_config())
    artifacts = runner.run_generation(tmp_path)['artifacts']
    assert sorted(p.name for p in tmp_path.iterdir()) == ['generation-0001.inference.pt',
                                                         'generation-0001.resume.pt']
    payload = torch.load(artifacts['resume'], map_location='cpu', weights_only=True)
    assert payload['contract'] == RESUME_CONTRACT
    assert set(payload['rng']) == {'python', 'numpy', 'torch', 'search', 'action', 'sampling',
                                   'augmentation', 'root_noise'}
    assert payload['counters']['completed_generations'] == 1
    assert payload['runtime']['intra_op_threads'] == 1 and 'commit' in payload['source']
    assert load_inference_checkpoint(artifacts['inference']).model.state_dict().keys() == \
        runner.model.state_dict().keys()
    with pytest.raises(ValueError):
        load_inference_checkpoint(artifacts['resume'])
    with pytest.raises(FileExistsError):
        runner.save_boundary(tmp_path)


def test_resume_continues_exactly_in_process(tmp_path):
    config = tiny_config()
    uninterrupted = GenerationRunner(config)
    interrupted = GenerationRunner(config)
    for runner in (uninterrupted, interrupted):
        runner.run_generation()
        runner.run_generation()
    path = tmp_path / 'boundary.pt'
    interrupted.save_resume_boundary(path)
    del interrupted
    resumed = load_resume_boundary(path)
    assert resumed.state_sha256() == uninterrupted.state_sha256()
    assert resumed.trainer.optimizer.state_dict()['state'][0]['step'] == \
        uninterrupted.trainer.optimizer.state_dict()['state'][0]['step']
    expected, actual = uninterrupted.run_generation(), resumed.run_generation()
    assert actual == expected
    assert list(resumed.replay.iter_games()) == list(uninterrupted.replay.iter_games())
    assert resumed.state_sha256() == uninterrupted.state_sha256()
    assert weights_sha256(resumed.model) == weights_sha256(uninterrupted.model)


def test_resume_continues_exactly_in_fresh_process(tmp_path):
    config = tiny_config()
    runner = GenerationRunner(config)
    runner.run_generation()
    runner.save_resume_boundary(tmp_path / 'boundary.pt')
    expected = runner.run_generation()
    script = f'''
import json, torch
torch.set_num_threads(1)
from games.connect4.alphazero_v2.generation import load_resume_boundary
runner = load_resume_boundary({str(tmp_path / 'boundary.pt')!r})
summary = runner.run_generation()
print(json.dumps(dict(summary=summary, state=runner.state_sha256())))
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            timeout=120, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
    assert result.returncode == 0, result.stderr
    output = json.loads(result.stdout.strip().splitlines()[-1])
    assert output['summary'] == json.loads(json.dumps(expected))
    assert output['state'] == runner.state_sha256()


def test_resume_restores_global_rng_and_rejects_runtime_or_content_changes(tmp_path, monkeypatch):
    runner = GenerationRunner(tiny_config())
    runner.run_generation()
    python_state, torch_state = random.getstate(), torch.get_rng_state()
    path = tmp_path / 'boundary.pt'
    runner.save_resume_boundary(path)
    random.random(), torch.rand(3), np.random.rand()
    load_resume_boundary(path)
    assert random.getstate() == python_state and torch.equal(torch.get_rng_state(), torch_state)

    tampered = torch.load(path, map_location='cpu', weights_only=True)
    first = tampered['replay']['generations'][0]['games'][0]
    action, original = first['moves'][0], first['visits'][0]
    changed = [0] * 7
    changed[action] = 3  # same budget and legal support, different visit target
    if changed == original:
        changed[action], changed[(action + 1) % 7] = 2, 1
    first['visits'][0] = changed
    torch.save(tampered, tmp_path / 'tampered.pt')
    with pytest.raises(ValueError, match='digest'):
        load_resume_boundary(tmp_path / 'tampered.pt')

    real = generation_module.runtime_identity
    monkeypatch.setattr(generation_module, 'runtime_identity', lambda: dict(real(), torch='0.0'))
    with pytest.raises(ValueError, match='Runtime'):
        load_resume_boundary(path)
    assert load_resume_boundary(path, strict_runtime=False).state_sha256() == runner.state_sha256()


# Isolation ---------------------------------------------------------------------------

def test_v2_path_never_imports_tactical_or_v1_training_machinery():
    script = '''
import sys
import games.connect4.alphazero_v2.generation
banned = {"games.connect4.tactical_value", "games.connect4.train_mcts_nn",
          "games.connect4.neural_self_play", "games.connect4.neural_evaluation",
          "games.connect4.neural_value_target_audit"}
loaded = banned & set(sys.modules)
assert not loaded, loaded
print("ok")
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            timeout=60, cwd=ROOT)
    assert result.returncode == 0, result.stdout + result.stderr


def test_config_and_data_import_without_torch():
    script = '''
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "torch":
            raise AssertionError("torch import attempted: " + name)
sys.meta_path.insert(0, Block())
from games.connect4.alphazero_v2.config import V2Config
from games.connect4.alphazero_v2.data import GenerationReplay, V2Example
print(V2Config().self_play_simulations)
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True,
                            timeout=60, cwd=ROOT)
    assert result.returncode == 0 and result.stdout.strip() == '256', result.stdout + result.stderr
