"""Deterministic MCTS contracts, accounting, and root-guard regressions.

Tactical guarantees below exercise the explicit root guards, not claims about
random-rollout playing strength. Tree tests independently check UCT/rewards.
"""

from copy import deepcopy
from random import Random

import pytest

from games.connect4.agents.mcts_agent import MCTSAgent, Node
from games.connect4.connect4 import Connect4


DRAW = [2, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
        3, 3, 3, 3, 3, 3, 6, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5,
        6, 6, 6, 6, 6]
WINS = [([0, 1, 0, 1, 0, 2], 0, 0),
        ([0, 1, 0, 1, 2, 1, 2], 1, 1)]
BLOCKS = [([0, 1, 0, 1, 2, 1], 0, 1),
          ([0, 1, 0, 1, 0], 1, 0)]


def position(moves):
    game = Connect4()
    for move in moves:
        assert not game.is_game_over(), 'Fixture continued past a terminal state'
        assert game.is_valid_move(move)
        assert game.make_move(move)
    return game


def child(parent, move):
    game = deepcopy(parent.game_state)
    assert not game.is_game_over()
    assert game.make_move(move)
    result = Node(game, parent=parent, move=move)
    parent.children[move] = result
    parent.untried_moves.remove(move)
    return result


@pytest.mark.parametrize('budget', [True, False, None, 1.0, 1.5, '10', [], {}, float('inf')])
def test_invalid_budget_types(budget):
    with pytest.raises(TypeError, match='positive integer'):
        MCTSAgent(budget)


@pytest.mark.parametrize('budget', [0, -1, -100])
def test_invalid_budget_values(budget):
    with pytest.raises(ValueError, match='positive integer'):
        MCTSAgent(budget)


def test_compatible_construction():
    assert MCTSAgent().simulation_limit == 1000
    assert MCTSAgent(7).simulation_limit == 7
    assert str(MCTSAgent(7)) == repr(MCTSAgent(7)) == 'MCTS Agent (7 sims)'


@pytest.mark.parametrize('moves', [[], [3]])
@pytest.mark.parametrize('budget', [1, 2, 6, 7, 8, 20])
def test_legal_moves_and_exact_search_budget(moves, budget, monkeypatch):
    game = position(moves)
    before = deepcopy(game.__dict__)
    agent = MCTSAgent(budget, rng=Random(42))
    roots = []
    backpropagate = agent._backpropagate

    def record(node, winner):
        backpropagate(node, winner)
        while node.parent is not None:
            node = node.parent
        roots.append(node)

    monkeypatch.setattr(agent, '_backpropagate', record)
    move = agent.choose_move(game)
    assert type(move) is int and move in game.get_valid_moves()
    assert len(roots) == budget
    root = roots[-1]
    assert root.visits == budget == sum(c.visits for c in root.children.values())
    assert all(c.visits > 0 for c in root.children.values())
    assert game.__dict__ == before


@pytest.mark.parametrize('moves', [[], [3]])
@pytest.mark.parametrize('winner', [0, 1, -1])
def test_backpropagation_accounts_for_each_incoming_mover(moves, winner):
    root = Node(position(moves))
    first = child(root, 0)
    second = child(first, 1)
    third = child(second, 2)
    mover = root.game_state.current_player
    assert [n.player_just_moved for n in (root, first, second, third)] == [1-mover, mover, 1-mover, mover]
    MCTSAgent(1)._backpropagate(third, winner)
    for node in (root, first, second, third):
        assert node.visits == 1
        expected = 0.5 if winner == -1 else float(winner == node.player_just_moved)
        assert node.wins == expected


def test_mixed_results_accumulate_draws_as_half_points():
    root = Node(position([]))
    leaf = child(root, 3)
    agent = MCTSAgent(1)
    for winner in (0, 0, 1, -1):
        agent._backpropagate(leaf, winner)
    assert (leaf.visits, leaf.wins) == (4, 2.5)
    assert (root.visits, root.wins) == (4, 1.5)


@pytest.mark.parametrize('moves', [[], [3]])
@pytest.mark.parametrize('depth', [0, 1])
def test_ucb_prefers_current_movers_wins_at_both_tree_levels(moves, depth):
    root = Node(position(moves))
    parent = child(root, 2) if depth else root
    good, bad = child(parent, 0), child(parent, 1)
    agent = MCTSAgent(1, rng=Random(0))
    for _ in range(4):
        agent._backpropagate(good, parent.game_state.current_player)
        agent._backpropagate(bad, 1 - parent.game_state.current_player)
    assert good.visits == bad.visits == 4
    assert good.wins == 4 and bad.wins == 0
    assert agent._select_child(parent) is good
    assert good.ucb1(c=0) == 1 and bad.ucb1(c=0) == 0


def test_ucb_explores_unvisited_and_underexplored_children():
    root = Node(position([]))
    visited, new = child(root, 0), child(root, 1)
    root.visits = visited.visits = 100
    visited.wins = 100
    agent = MCTSAgent(1, rng=Random(0))
    assert new.ucb1() == float('inf')
    assert agent._select_child(root) is new
    new.visits = 1
    root.visits += 1
    assert agent._select_child(root) is new


def test_root_chooses_visits_not_reward_or_ucb(monkeypatch):
    agent = MCTSAgent(12, rng=Random(0))
    # After each root move is expanded, force visits to column 1. Give that
    # branch losses and another branch wins; the robust-child rule uses visits.
    monkeypatch.setattr(agent, '_select_child', lambda node: node.children[1])
    monkeypatch.setattr(agent, '_simulate', lambda game: 1 if game.board[5][1] == 'X' else 0)
    assert agent.choose_move(position([])) == 1


@pytest.mark.parametrize('moves,player,winning_column', WINS)
@pytest.mark.parametrize('seed', [0, 7, 42])
def test_root_guard_takes_immediate_win_even_with_one_simulation(moves, player, winning_column, seed, monkeypatch):
    game = position(moves)
    before = deepcopy(game.__dict__)
    assert game.current_player == player
    agent = MCTSAgent(1, rng=Random(seed))
    monkeypatch.setattr(agent, '_simulate', lambda game: pytest.fail('Immediate-win guard must bypass rollouts'))
    assert agent.choose_move(game) == winning_column
    assert game.__dict__ == before
    game.make_move(winning_column)
    assert game.check_winner() == player


@pytest.mark.parametrize('moves,player,block', BLOCKS)
@pytest.mark.parametrize('budget', [1, 8])
def test_root_guard_keeps_only_forced_defense(moves, player, block, budget):
    game = position(moves)
    before = deepcopy(game.__dict__)
    agent = MCTSAgent(budget, rng=Random(0))
    assert game.current_player == player
    assert agent._winning_moves(game) == []
    assert agent._safe_moves(game) == [block]
    assert agent.choose_move(game) == block
    assert game.__dict__ == before


@pytest.mark.parametrize('moves,player,win', [([0, 1, 0, 1, 0, 1], 0, 0),
                                            ([0, 1, 0, 1, 2, 1, 0], 1, 1)])
def test_win_has_priority_over_blocking(moves, player, win):
    game = position(moves)
    assert game.current_player == player
    assert MCTSAgent(1, rng=Random(0)).choose_move(game) == win


def test_two_opponent_threats_fall_back_to_legal_search():
    game = position([3, 3, 2, 3, 4])
    agent = MCTSAgent(8, rng=Random(0))
    assert agent._winning_moves(game) == []
    assert agent._safe_moves(game) == []
    assert agent.choose_move(game) in game.get_valid_moves()


@pytest.mark.parametrize('moves', [DRAW[:38], DRAW[:40], DRAW[:41], [3] * 6])
@pytest.mark.parametrize('budget', [1, 10])
def test_nearly_full_boards_and_full_columns(moves, budget):
    game = position(moves)
    before = deepcopy(game.__dict__)
    move = MCTSAgent(budget, rng=Random(7)).choose_move(game)
    assert move in game.get_valid_moves()
    assert game.board[0][move] == ' '
    assert game.__dict__ == before
    if len(moves) == 41:
        game.make_move(move)
        assert game.is_game_over() and game.check_winner() == -1


@pytest.mark.parametrize('moves,winner', [([0, 1, 0, 1, 0, 1, 0], 0),
                                        ([0, 1, 0, 1, 2, 1, 2, 1], 1), (DRAW, -1)])
def test_terminal_positions_never_expand_or_roll_out(moves, winner):
    class NoRandom:
        def choice(self, values):
            pytest.fail('Terminal states must not request a random move')

    game = position(moves)
    before = deepcopy(game.__dict__)
    assert game.is_game_over() and game.check_winner() == winner
    node = Node(game)
    assert node.is_terminal() and node.is_fully_expanded()
    assert not node.untried_moves
    agent = MCTSAgent(1, rng=NoRandom())
    assert agent._simulate(game) == winner
    with pytest.raises(ValueError, match='terminal position'):
        agent.choose_move(game)
    assert game.__dict__ == before


def test_legal_move_exhaustion_does_not_call_choice_on_empty_list(monkeypatch):
    # Defensive contract for an engine returning no moves before reporting full.
    game = position([])
    monkeypatch.setattr(game, 'get_valid_moves', lambda: [])
    agent = MCTSAgent(1, rng=Random(0))
    assert agent._simulate(game) == -1
    with pytest.raises(ValueError, match='without legal moves'):
        agent.choose_move(game)


@pytest.mark.parametrize('moves,winner', [(DRAW[:40], -1), (DRAW[:41], -1),
                                        ([0, 1, 0, 1, 0, 2], 0),
                                        ([0, 1, 0, 1, 2, 1, 2], 1)])
def test_rollout_terminal_outcomes_and_input_isolation(moves, winner):
    class FirstLegal:
        def choice(self, values):
            # On the win fixtures, choose the mover's immediate winning column.
            return winner if winner != -1 else values[0]

    game = position(moves)
    before = deepcopy(game.__dict__)
    assert MCTSAgent(1, rng=FirstLegal())._simulate(game) == winner
    assert game.__dict__ == before


def test_fixed_seed_reproduces_search_statistics_and_moves(monkeypatch):
    def run():
        agent = MCTSAgent(30, rng=Random(812))
        snapshots = []
        backpropagate = agent._backpropagate

        def record(node, winner):
            backpropagate(node, winner)
            while node.parent is not None:
                node = node.parent
            snapshots.append(tuple((m, c.visits, c.wins) for m, c in node.children.items()))

        monkeypatch.setattr(agent, '_backpropagate', record)
        moves = [agent.choose_move(position(seq)) for seq in ([], [3], [3, 2])]
        return moves, snapshots

    assert run() == run()


def test_all_random_choices_use_injected_generator(monkeypatch):
    def forbidden(*args):
        pytest.fail('Search must not use module-global randomness')

    monkeypatch.setattr('random.choice', forbidden)
    monkeypatch.setattr('random.random', forbidden)
    game = position([])
    assert MCTSAgent(10, rng=Random(1)).choose_move(game) in game.get_valid_moves()


def test_root_guard_avoids_enabling_a_floating_opponent_win():
    # Legal fixture also used by grounding tests: columns 1 and 5 supply the
    # missing support for the opponent's winning reply one square above.
    game = position([3, 3, 3, 4, 3, 4, 3, 3, 2, 2, 2, 0,
                     2, 0, 2, 2, 4, 0, 0, 0, 4, 0, 4, 4])
    agent = MCTSAgent(1, rng=Random(0))
    assert set(game.get_valid_moves()) == {1, 5, 6}
    assert agent._winning_moves(game) == []
    for move in (1, 5):
        after = deepcopy(game)
        after.make_move(move)
        assert move in agent._winning_moves(after)
    assert agent._safe_moves(game) == [6]
    assert agent.choose_move(game) == 6


def test_search_revisits_terminal_draw_without_expanding_it(monkeypatch):
    agent = MCTSAgent(10, rng=Random(0))
    leaves = []
    backpropagate = agent._backpropagate

    def record(node, winner):
        assert winner == -1
        backpropagate(node, winner)
        leaves.append(node)

    monkeypatch.setattr(agent, '_backpropagate', record)
    assert agent.choose_move(position(DRAW[:41])) == 6
    leaf = leaves[0]
    assert all(node is leaf for node in leaves)
    assert leaf.is_terminal() and not leaf.children and not leaf.untried_moves
    assert (leaf.visits, leaf.wins) == (10, 5)
    assert (leaf.parent.visits, leaf.parent.wins) == (10, 5)
