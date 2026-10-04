"""Deterministic neural-MCTS contracts; no self-play training or saved weights."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import math
from random import Random

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from games.connect4.connect4 import Connect4
from games.connect4.neural_mcts import (
    ARCHITECTURE, CHECKPOINT_CONTRACT, ENCODING, HISTORICAL_ENCODING,
    Connect4Net, NeuralInference, encode_current_player, encode_historical_absolute,
    legal_policy, load_checkpoint, load_historical_network_for_research,
    prediction_from_logits, probability_vector,
)
from games.connect4.agents.mcts_nn_agent import (
    MCTSNNAgent, Node, backup, load_pretrained_mcts_nn_agent, tactical_root_moves,
    terminal_value, visit_policy,
)
from games.connect4.train_mcts_nn import (
    POLICY_TARGET, NeuralTrainer, TrainingExample, batch_tensors, capture_example,
    finalize_examples, outcome_for_player, training_loss,
)

DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]
WINS = [([0, 1, 0, 1, 0, 2], 0, 0), ([0, 1, 0, 1, 2, 1, 2], 1, 1)]
BLOCKS = [([0, 1, 0, 1, 2, 1], 0, 1), ([0, 1, 0, 1, 0], 1, 0)]
FINISHED = [(WINS[0][0] + [0], 0), (WINS[1][0] + [1], 1), (DRAW, -1)]


@pytest.fixture(autouse=True, scope='module')
def bounded_torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def position(moves):
    game = Connect4()
    for move in moves:
        assert not game.is_game_over()
        assert move in game.get_valid_moves()
        assert game.make_move(move)
    return game


class FixedLogits(torch.nn.Module):
    representation_version = ENCODING

    def __init__(self, policy=None, value=None):
        super().__init__()
        self.register_buffer('policy', torch.tensor([0.] * 7 if policy is None else policy, dtype=torch.float32))
        self.register_buffer('value', torch.tensor([0.] * 3 if value is None else value, dtype=torch.float32))
        self.eval()

    def forward(self, states):
        assert not self.training
        assert not torch.is_grad_enabled()
        assert states.shape == (1, 1, 6, 7) and states.dtype == torch.float32
        return self.policy[None, :], self.value[None, :]


def agent(budget=1, **kwargs):
    return MCTSNNAgent(FixedLogits(), budget, rng=Random(4), **kwargs)


def child(parent, move, prior=0.5):
    game = deepcopy(parent.game_state)
    assert game.make_move(move)
    result = Node(game, parent, move, prior)
    parent.children[move] = result
    return result


@pytest.mark.parametrize('moves', [[0, 1], [0, 1, 2]])
def test_both_player_encoding_and_detachment(moves):
    game = position(moves)
    encoded = encode_current_player(game)
    own = game.piece
    assert encoded.shape == (1, 1, 6, 7) and encoded.dtype == torch.float32
    for row in range(6):
        for col in range(7):
            expected = 0 if game.board[row][col] == ' ' else (1 if game.board[row][col] == own else -1)
            assert encoded[0, 0, row, col] == expected
    absolute = encode_historical_absolute(game)
    assert absolute[0, 0, 5, 0] == 1 and absolute[0, 0, 5, 1] == -1
    assert torch.equal(encoded, absolute * (1 if game.current_player == 0 else -1))
    before = encoded.clone()
    game.make_move(6)
    assert torch.equal(encoded, before)


@pytest.mark.parametrize('defect', ['symbol', 'shape', 'player', 'piece'])
def test_encoding_rejects_invalid_states(defect):
    game = Connect4()
    if defect == 'symbol':
        game.board[0][0] = '?'
    elif defect == 'shape':
        game.board.pop()
    elif defect == 'player':
        game.current_player = 2
    else:
        game.piece = 'O'
    with pytest.raises(ValueError):
        encode_current_player(game)


def test_architecture_is_retained_and_forward_exposes_logits():
    model = Connect4Net().eval()
    assert sum(p.numel() for p in model.parameters()) == 326026
    assert len(model.state_dict()) == 16
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.policy_fc.bias.copy_(torch.arange(7.))
        model.value_fc2.bias.copy_(torch.tensor([2., -1., 4.]))
        policies, values = model(torch.zeros(2, 1, 6, 7))
    assert policies.shape == (2, 7) and values.shape == (2, 3)
    assert torch.equal(policies[0], torch.arange(7.))
    assert values[0].tolist() == [2., -1., 4.]


@pytest.mark.parametrize('states', [torch.zeros(1, 6, 7), torch.zeros(0, 1, 6, 7),
                                    torch.zeros(1, 1, 7, 6), torch.zeros(1, 1, 6, 7).double(),
                                    torch.full((1, 1, 6, 7), float('nan')),
                                    torch.full((1, 1, 6, 7), 0.5)])
def test_network_rejects_bad_input(states):
    with pytest.raises(ValueError):
        Connect4Net()(states)


def test_known_logit_softmax_and_scalar_value_conversion():
    prediction = prediction_from_logits(torch.log(torch.tensor([[1., 2, 3, 4, 5, 6, 7]])),
                                        torch.log(torch.tensor([[6., 3., 1.]])))
    assert prediction.policy == pytest.approx(np.arange(1, 8) / 28)
    assert prediction.value_probabilities == pytest.approx([0.6, 0.3, 0.1])
    assert prediction.value == pytest.approx(0.5)
    saturated = prediction_from_logits(torch.tensor([[1e30, -1e30, 0, 0, 0, 0, 0]]),
                                      torch.tensor([[-1e30, 0, 1e30]]))
    assert saturated.policy == (1, 0, 0, 0, 0, 0, 0) and saturated.value == -1


@pytest.mark.parametrize('head', [0, 1])
@pytest.mark.parametrize('bad', ['nan', 'inf', 'shape', 'dtype'])
def test_prediction_rejects_invalid_logits(head, bad):
    tensors = [torch.zeros(1, 7), torch.zeros(1, 3)]
    if bad == 'shape':
        tensors[head] = tensors[head][0]
    elif bad == 'dtype':
        tensors[head] = tensors[head].double()
    else:
        tensors[head][0, 0] = float(bad)
    with pytest.raises(ValueError, match='logits'):
        prediction_from_logits(*tensors)


def test_legal_mask_zero_legal_mass_and_subnormal_fallback():
    p = legal_policy(np.arange(1, 8) / 28, [0, 2, 6])
    assert p == pytest.approx([1/11, 0, 3/11, 0, 0, 0, 7/11])
    assert legal_policy([1, 0, 0, 0, 0, 0, 0], [1, 6]) == pytest.approx([0, .5, 0, 0, 0, 0, .5])
    assert legal_policy([1, 1e-320, 0, 0, 0, 0, 0], [1, 2])[1] == 1


@pytest.mark.parametrize('policy', [[0] * 7, [1] * 7, [1, -1, 1, 0, 0, 0, 0],
                                   [float('nan')] * 7, [float('inf')] * 7, [1] * 6])
def test_bad_probabilities_are_rejected(policy):
    with pytest.raises(ValueError):
        legal_policy(policy, [0, 1])


@pytest.mark.parametrize('moves', [[], [-1], [7], [True], [1.0], [1, 1]])
def test_invalid_legal_actions_rejected(moves):
    with pytest.raises(ValueError):
        legal_policy([1/7] * 7, moves)


@pytest.mark.parametrize('budget', [True, False, None, 1., 1.5, '2', [], {}, float('inf')])
def test_invalid_budget_types(budget):
    with pytest.raises(TypeError, match='positive integer'):
        agent(budget)


@pytest.mark.parametrize('budget', [0, -1, -50])
def test_invalid_budget_values(budget):
    with pytest.raises(ValueError, match='positive integer'):
        agent(budget)


@pytest.mark.parametrize('name', ['temperature', 'exploration'])
@pytest.mark.parametrize('value', [-1., float('nan'), float('inf'), True, None, '1'])
def test_invalid_search_parameters(name, value):
    with pytest.raises((ValueError, TypeError)):
        agent(**{name: value})


@pytest.mark.parametrize('moves', [[], [3]])
def test_puct_negates_child_value_and_respects_unvisited_priors(moves):
    root = Node(position(moves))
    low, high = child(root, 0, .1), child(root, 1, .9)
    search = agent(exploration=2)
    assert math.isfinite(low.puct(2)) and high.puct(2) == pytest.approx(1.8)
    assert search._select_child(root, Random(0)) is high
    root.visits = 10
    low.visits = high.visits = 5
    low.value_sum, high.value_sum = -4., 4.
    assert low.puct(2) == pytest.approx(.8 + 2 * .1 * math.sqrt(10) / 6)
    assert search._select_child(root, Random(0)) is low
    # At the next ply, the same convention applies to the other player.
    good, bad = child(low, 2), child(low, 3)
    good.visits = bad.visits = 1
    good.value_sum, bad.value_sum = -1, 1
    assert search._select_child(low, Random(0)) is good


@pytest.mark.parametrize('moves', [[], [3]])
@pytest.mark.parametrize('value', [-1., -.4, 0., .8, 1.])
def test_full_path_backup_alternates_for_both_players(moves, value):
    root = Node(position(moves))
    first = child(root, 0)
    second = child(first, 1)
    leaf = child(second, 2)
    backup(leaf, value)
    assert [n.visits for n in (root, first, second, leaf)] == [1] * 4
    assert [n.q_value for n in (root, first, second, leaf)] == pytest.approx([-value, value, -value, value])
    backup(leaf, -value)
    assert all(n.visits == 2 and n.q_value == 0 for n in (root, first, second, leaf))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.01, 1.01])
def test_bad_leaf_values_never_reach_statistics(value):
    root = Node(Connect4())
    with pytest.raises(ValueError):
        backup(root, value)
    assert root.visits == 0


@pytest.mark.parametrize('moves,player,win', WINS)
def test_raw_search_known_prior_terminal_win_both_players(moves, player, win):
    logits = [-80.] * 7
    logits[win] = 0
    # A deliberately wrong neural value must not override an engine terminal.
    model = FixedLogits(logits, [80., 0, -80.])
    result = MCTSNNAgent(model, 3, temperature=0, rng=Random(2)).search(position(moves))
    assert result.move == win and result.guard_applied == 'disabled'
    assert result.root.game_state.current_player == player
    leaf = result.root.children[win]
    assert leaf.game_state.current_player == 1-player
    assert leaf.visits == 3 and leaf.q_value == -1 and not leaf.children
    assert result.root.q_value == 1


@pytest.mark.parametrize('moves,winner', FINISHED)
def test_terminal_roots_reject_before_inference_and_terminal_values(moves, winner):
    class NeverCalled(FixedLogits):
        def forward(self, states):
            pytest.fail('Terminal position reached network')
    game = position(moves)
    assert game.check_winner() == winner
    assert terminal_value(game) == (0 if winner == -1 else -1)
    for guard in (False, True):
        with pytest.raises(ValueError, match='terminal'):
            MCTSNNAgent(NeverCalled(), 1, tactical_guard=guard).search(game)
    with pytest.raises(ValueError):
        NeuralInference(NeverCalled()).predict_legal(game)


def test_no_legal_state_rejected(monkeypatch):
    game = Connect4()
    monkeypatch.setattr(game, 'get_valid_moves', lambda: [])
    with pytest.raises(ValueError, match='without legal'):
        agent().search(game)


@pytest.mark.parametrize('moves', [[], [3], [3] * 6, DRAW[:38], DRAW[:41]])
@pytest.mark.parametrize('budget', [1, 2, 12])
@pytest.mark.parametrize('temperature', [0., 1.])
def test_exact_budget_finite_targets_legal_moves_and_full_immutability(moves, budget, temperature):
    game = position(moves)
    before = deepcopy(game.__dict__)
    search = agent(budget, temperature=temperature)
    result = search.search(game)
    assert result.root.visits == sum(result.visits) == budget
    assert result.move in game.get_valid_moves() and result.visits[result.move] > 0
    assert probability_vector(result.policy).sum() == pytest.approx(1)
    assert result.policy == pytest.approx(visit_policy(result.visits, temperature))
    assert all(result.policy[a] == 0 for a in range(7) if a not in game.get_valid_moves())
    assert game.__dict__ == before
    assert not hasattr(search, 'training_examples') and not hasattr(search, 'root')
    if len(moves) == 41:
        assert result.root.q_value == 0
        assert result.root.children[result.move].q_value == 0


def test_one_simulation_uses_root_prior_and_leaf_value_not_root_value():
    model = FixedLogits([0, 0, 0, 0, 0, 0, 10], [math.log(6), math.log(3), 0])
    for moves in ([], [3]):
        result = MCTSNNAgent(model, 1, rng=Random(0)).search(position(moves))
        assert result.move == 6 and result.visits == (0, 0, 0, 0, 0, 0, 1)
        assert result.root.q_value == pytest.approx(-.5)
        assert result.root.children[6].q_value == pytest.approx(.5)


@pytest.mark.parametrize('temperature', [0., .5, 1., 2., 1e-320, 1e300])
def test_stable_temperature(temperature):
    counts = [0, 1, 4, 4, 0, 2, 0]
    policy = visit_policy(counts, temperature)
    assert np.isfinite(policy).all() and policy.sum() == pytest.approx(1)
    assert policy[0] == policy[4] == policy[6] == 0
    if temperature in (0, 1e-320):
        assert policy == pytest.approx([0, 0, .5, .5, 0, 0, 0])
    if temperature == .5:
        assert policy == pytest.approx(np.array(counts) ** 2 / 37)
    if temperature == 1:
        assert policy == pytest.approx(np.array(counts) / 11)


@pytest.mark.parametrize('visits', [[0]*7, [-1, 2, 0, 0, 0, 0, 0], [1.5]*7,
                                   [float('nan')]*7, [float('inf')]*7, [1]*6])
def test_invalid_visit_counts_rejected(visits):
    with pytest.raises(ValueError):
        visit_policy(visits, 1)


def test_root_action_uses_visits_not_q_or_prior(monkeypatch):
    search = agent(5, temperature=0)
    def selection(node, rng):
        return node.children[0 if node.parent is None else next(iter(node.children))]
    monkeypatch.setattr(search, '_select_child', selection)
    result = search.search(Connect4())
    assert result.visits == (5, 0, 0, 0, 0, 0, 0) and result.move == 0


def test_tree_single_parent_ownership_and_detached_states():
    root = agent(96).search(Connect4()).root
    seen, states = set(), set()
    stack = [root]
    while stack:
        node = stack.pop()
        assert id(node) not in seen and id(node.game_state) not in states
        seen.add(id(node))
        states.add(id(node.game_state))
        assert -1 <= node.q_value <= 1 and math.isfinite(node.q_value)
        if node.children:
            assert sum(c.prior_p for c in node.children.values()) == pytest.approx(1)
            # Nonroot expansion backs up one leaf evaluation before descendants.
            assert sum(c.visits for c in node.children.values()) == node.visits - (node.parent is not None)
        for move, descendant in node.children.items():
            assert descendant.parent is node and descendant.move == move
            expected = deepcopy(node.game_state)
            expected.make_move(move)
            assert descendant.game_state.__dict__ == expected.__dict__
            assert all(a is not b for a, b in zip(node.game_state.board, descendant.game_state.board))
            stack.append(descendant)
    assert len(seen) > 96


def test_independent_concurrent_searches_share_read_only_model():
    model = Connect4Net().eval()
    weights = {k: v.clone() for k, v in model.state_dict().items()}
    inference = NeuralInference(model)
    search = MCTSNNAgent(inference, 3, temperature=1)
    game = Connect4()
    before = deepcopy(game.__dict__)
    def run(seed):
        return search.search(game, rng=Random(seed))
    sequential = [run(5), run(9)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(run, [5, 9]))
    assert [(r.move, r.visits) for r in sequential] == [(r.move, r.visits) for r in concurrent]
    assert len({id(r.root) for r in sequential + concurrent}) == 4
    assert all(torch.equal(weights[k], v) for k, v in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters()) and not model.training
    assert game.__dict__ == before


@pytest.mark.parametrize('moves,player,move', WINS + BLOCKS)
def test_opt_in_guard_for_both_players_with_adversarial_prior(moves, player, move):
    game = position(moves)
    before = deepcopy(game.__dict__)
    wrong = next(a for a in game.get_valid_moves() if a != move)
    logits = [-1000.] * 7
    logits[wrong] = 1000.
    model = FixedLogits(logits)
    raw = MCTSNNAgent(model, 1, rng=Random(0)).search(game)
    guarded = MCTSNNAgent(model, 1, rng=Random(0), tactical_guard=True).search(game)
    assert game.current_player == player and game.__dict__ == before
    assert raw.move == wrong and raw.guard_applied == 'disabled'
    assert guarded.move == move and guarded.policy[move] == 1
    assert guarded.guard_applied == ('immediate_win' if (moves, player, move) in WINS else 'safe_responses')
    assert capture_example(guarded).tactical_guard is True


@pytest.mark.parametrize('moves,expected', [([0, 1, 0, 1, 0, 1], 0),
                                            ([0, 1, 0, 1, 2, 1, 0], 1)])
def test_guard_prioritizes_win_over_block(moves, expected):
    assert agent(tactical_guard=True).choose_move(position(moves)) == expected


def test_guard_all_unsafe_and_terminal_draw_successor():
    game = position([3, 3, 2, 3, 4])
    moves, reason = tactical_root_moves(game)
    assert moves == game.get_valid_moves() and reason == 'none'
    game = position(DRAW[:41])
    assert agent(tactical_guard=True).choose_move(game) in game.get_valid_moves()


@pytest.mark.parametrize('moves', [[], [3]])
def test_immutable_pre_move_examples(moves):
    game = position(moves)
    result = agent().search(game)
    expected = encode_current_player(game)[0, 0].tolist()
    acting_player = game.current_player
    game.make_move(result.move)
    example = capture_example(result)
    assert example.acting_player == acting_player
    assert example.observation == tuple(tuple(row) for row in expected)
    assert example.encoding == ENCODING and example.policy_target == POLICY_TARGET
    assert example.policy == result.policy and example.temperature == result.temperature
    result.root.game_state.make_move(result.move)
    assert example.observation == tuple(tuple(row) for row in expected)
    with pytest.raises(FrozenInstanceError):
        example.outcome = 1
    with pytest.raises(TypeError):
        example.observation[0][0] = 1


@pytest.mark.parametrize('moves,winner', FINISHED)
def test_completed_game_targets_for_both_recorded_players(moves, winner):
    examples = [capture_example(agent().search(position(prefix))) for prefix in ([], [0])]
    finished = finalize_examples(examples, position(moves))
    assert [e.outcome for e in finished] == ([0, 0] if winner == -1 else ([1, -1] if winner == 0 else [-1, 1]))
    assert all(e.outcome is None for e in examples)
    assert [outcome_for_player(winner, p) for p in (0, 1)] == [e.outcome for e in finished]
    _, _, labels = batch_tensors(finished)
    assert labels.tolist() == [{1: 0, 0: 1, -1: 2}[e.outcome] for e in finished]
    with pytest.raises(ValueError, match='already labeled'):
        finalize_examples(finished, position(moves))


def test_pending_examples_and_incomplete_games_rejected():
    example = capture_example(agent().search(Connect4()))
    with pytest.raises(ValueError, match='completed game'):
        finalize_examples([example], Connect4())
    for examples in ([], [example]):
        with pytest.raises(ValueError):
            batch_tensors(examples)
    for winner in (True, 2, None, 1.):
        with pytest.raises(ValueError):
            outcome_for_player(winner, 0)


@pytest.mark.parametrize('changes', [dict(policy=[0]*7), dict(policy=[float('nan')]*7),
                                    dict(observation=[[0]*7]*5), dict(acting_player=True),
                                    dict(acting_player=2), dict(outcome=.5), dict(outcome=True),
                                    dict(encoding=HISTORICAL_ENCODING), dict(policy_target='raw'),
                                    dict(temperature=float('nan')), dict(tactical_guard=1)])
def test_invalid_training_examples(changes):
    example = capture_example(agent().search(Connect4()))
    with pytest.raises((ValueError, TypeError)):
        replace(example, **changes)


def test_training_example_owns_nested_inputs_and_rejects_illegal_target():
    board, policy = [[0]*7 for _ in range(6)], [1/7]*7
    example = TrainingExample(board, 0, policy, 1)
    board[0][0] = 1
    policy[0] = 0
    assert example.observation[0][0] == 0 and example.policy[0] == 1/7
    with pytest.raises(ValueError, match='illegal'):
        TrainingExample(board, 0, [1/7]*7, 1)


def tiny_batch():
    examples = []
    for moves, action, outcome in (([], 3, 1), ([0], 2, -1), ([0, 1], 4, 0)):
        game = position(moves)
        policy = [0.]*7
        policy[action] = 1.
        examples.append(TrainingExample(encode_current_player(game)[0, 0].tolist(),
                                        game.current_player, policy, 0., outcome=outcome))
    return examples


def test_loss_matches_hand_computed_cross_entropy_and_gradients():
    p = torch.zeros(2, 7, requires_grad=True)
    v = torch.zeros(2, 3, requires_grad=True)
    targets = torch.tensor([[.25, .75, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1.]])
    loss = training_loss(p, v, targets, torch.tensor([0, 2]))
    assert loss.item() == pytest.approx(math.log(7) + math.log(3))
    loss.backward()
    assert torch.allclose(p.grad, (torch.full((2, 7), 1/7) - targets) / 2)
    expected = (torch.full((2, 3), 1/3) - torch.tensor([[1, 0, 0], [0, 0, 1]])) / 2
    assert torch.allclose(v.grad, expected)
    assert torch.isfinite(p.grad).all() and torch.isfinite(v.grad).all()


def test_fixed_tiny_batch_loss_decreases_with_persistent_optimizer():
    with torch.random.fork_rng():
        torch.manual_seed(4101)
        model = Connect4Net()
    trainer = NeuralTrainer(model)
    optimizer = trainer.optimizer
    states, policies, outcomes = batch_tensors(tiny_batch())
    with torch.no_grad():
        initial = training_loss(*model(states), policies, outcomes).item()
    modes = []
    hook = model.register_forward_pre_hook(lambda module, inputs: modes.append(module.training))
    losses = [trainer.step(tiny_batch()) for _ in range(8)]
    hook.remove()
    with torch.no_grad():
        final = training_loss(*model(states), policies, outcomes).item()
    assert final < initial * .9 and all(math.isfinite(loss) for loss in losses)
    assert trainer.optimizer is optimizer and trainer.steps == 8
    assert modes == [True]*8 and not model.training
    assert all(state['step'].item() == 8 for state in optimizer.state.values())
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    assert sum(p.grad.abs().sum().item() for p in model.parameters()) > 0
    NeuralInference(model).predict(Connect4())


@pytest.mark.parametrize('defect', ['policy_nan', 'policy_negative', 'policy_sum', 'class', 'class_dtype', 'logit_nan'])
def test_loss_rejects_invalid_targets_and_logits(defect):
    p, v = torch.zeros(1, 7), torch.zeros(1, 3)
    targets, classes = torch.full((1, 7), 1/7), torch.tensor([0])
    if defect == 'policy_nan':
        targets[0, 0] = float('nan')
    elif defect == 'policy_negative':
        targets[0, 0] = -1
    elif defect == 'policy_sum':
        targets.zero_()
    elif defect == 'class':
        classes[0] = 3
    elif defect == 'class_dtype':
        classes = classes.float()
    else:
        p[0, 0] = float('nan')
    with pytest.raises(ValueError):
        training_loss(p, v, targets, classes)


def test_nonfinite_gradient_aborts_before_update_and_restores_eval():
    trainer = NeuralTrainer(Connect4Net())
    before = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    handle = trainer.model.policy_fc.bias.register_hook(lambda gradient: gradient * float('nan'))
    with pytest.raises(ValueError, match='gradients'):
        trainer.step(tiny_batch())
    handle.remove()
    assert trainer.steps == 0 and not trainer.optimizer.state and not trainer.model.training
    assert all(torch.equal(v, before[k]) for k, v in trainer.model.state_dict().items())


def test_explicit_eval_boundary_and_model_health():
    model = Connect4Net()
    with pytest.raises(ValueError, match='eval'):
        NeuralInference(model)
    model.eval()
    inference = NeuralInference(model)
    model.train()
    with pytest.raises(ValueError, match='eval'):
        inference.predict(Connect4())
    model.eval()
    with torch.no_grad():
        model.policy_fc.bias[0] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        NeuralInference(model)


def test_checkpoint_contract_and_shared_interface_without_writing_weights(monkeypatch):
    model = Connect4Net().eval()
    payload = {'contract': deepcopy(CHECKPOINT_CONTRACT), 'model_state_dict': model.state_dict()}
    calls = []
    def fake_load(path, **kwargs):
        calls.append((path, kwargs))
        return payload
    monkeypatch.setattr(torch, 'load', fake_load)
    inference = load_checkpoint('in-memory-only')
    assert calls == [('in-memory-only', {'map_location': 'cpu', 'weights_only': True})]
    assert inference.predict(Connect4()) == NeuralInference(model).predict(Connect4())
    result = MCTSNNAgent(inference, 1).search(Connect4())
    assert result.move in range(7) and ARCHITECTURE == payload['contract']['architecture']
    with pytest.raises(ValueError, match='explicit'):
        load_pretrained_mcts_nn_agent()


@pytest.mark.parametrize('defect', ['legacy', 'encoding', 'value_order', 'perspective', 'outputs', 'keys', 'shape', 'dtype', 'nan'])
def test_checkpoint_rejects_incompatible_metadata_and_parameters(monkeypatch, defect):
    payload = {'contract': deepcopy(CHECKPOINT_CONTRACT), 'model_state_dict': Connect4Net().state_dict()}
    if defect == 'legacy':
        payload = {'iteration': 9, 'model_state_dict': payload['model_state_dict']}
    elif defect in ('encoding', 'value_order', 'perspective', 'outputs'):
        field = 'value_perspective' if defect == 'perspective' else defect
        payload['contract'][field] = 'wrong'
    elif defect == 'keys':
        del payload['model_state_dict']['conv1.bias']
    else:
        tensor = payload['model_state_dict']['conv1.bias']
        if defect == 'shape':
            payload['model_state_dict']['conv1.bias'] = tensor[:2]
        elif defect == 'dtype':
            payload['model_state_dict']['conv1.bias'] = tensor.double()
        else:
            tensor[0] = float('nan')
    monkeypatch.setattr(torch, 'load', lambda *args, **kwargs: payload)
    with pytest.raises(ValueError):
        load_checkpoint('in-memory-only')


def test_separate_historical_research_loader_never_silently_canonicalizes(monkeypatch):
    # Synthetic state only: no historical trained parameters are used in tests.
    payload = {'model_state_dict': Connect4Net().state_dict(), 'iteration': 9}
    def fake_load(path, **kwargs):
        assert kwargs == {'map_location': 'cpu', 'weights_only': True}
        return payload
    monkeypatch.setattr(torch, 'load', fake_load)
    historical = load_historical_network_for_research('synthetic-legacy')
    assert historical.representation_version == HISTORICAL_ENCODING
    for construct in (NeuralInference, NeuralTrainer):
        with pytest.raises(ValueError):
            construct(historical)
    with torch.inference_mode():
        p, v = historical(encode_historical_absolute(position([0])))
    assert p.shape == (1, 7) and v.shape == (1, 3)


@pytest.mark.parametrize('side', [0, 1])
def test_bounded_full_game_against_deterministic_legal_opponent(side):
    game = Connect4()
    search = agent(2, temperature=0)
    plies = 0
    while not game.is_game_over():
        before = deepcopy(game.__dict__)
        move = search.choose_move(game) if game.current_player == side else min(game.get_valid_moves())
        assert game.__dict__ == before and move in game.get_valid_moves()
        assert game.make_move(move)
        plies += 1
        assert plies <= 42
    assert game.check_winner() in (-1, 0, 1)
    assert not hasattr(search, 'training_examples')
