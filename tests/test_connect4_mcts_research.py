"""Research-only MCTS variants: parity, tactics, proofs, and isolation.

Oracles here are independent of the bitboard threat code: the array engine
for one-move tactics and rollouts, and a memoised exhaustive solver for proofs.
"""
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from random import Random

import pytest

from games.connect4.agents.mcts_agent import MCTSAgent
from games.connect4.agents.mcts_bitboard import BOARD_MASK, BitboardState, has_four
from games.connect4.agents.mcts_research_agent import (
    ResearchConfig, ResearchMCTSAgent, ResearchNode, safe_rollout,
    tactical_rollout, winning_cells,
)
from games.connect4.connect4 import Connect4
from scripts.benchmark_public_agents import DRAW, POSITIONS, position

CONFIGS = {
    'default': ResearchConfig(),
    'decisive': ResearchConfig(rollout='decisive'),
    'safe': ResearchConfig(rollout='safe'),
    'solver': ResearchConfig(solver=True),
    'center': ResearchConfig(expansion='center'),
    'c0.7': ResearchConfig(exploration=0.7),
    'combined': ResearchConfig(exploration=1.0, rollout='safe', solver=True, expansion='center'),
}
FLOATING = [3, 3, 3, 4, 3, 4, 3, 3, 2, 2, 2, 0, 2, 0, 2, 2, 4, 0, 0, 0, 4, 0, 4, 4]
WINS = [([0, 1, 0, 1, 0, 2], 0), ([0, 1, 0, 1, 2, 1, 2], 1)]
BLOCKS = [([0, 1, 0, 1, 2, 1], 1), ([0, 1, 0, 1, 0], 0)]


def random_position(rng, plies):
    """Uniform legal history of exactly ``plies`` moves, nonterminal."""
    while True:
        game, history = Connect4(), []
        while len(history) < plies and not game.is_game_over():
            move = rng.choice(game.get_valid_moves())
            game.make_move(move)
            history.append(move)
        if len(history) == plies and not game.is_game_over():
            return game, history


def winning_columns(game, player):
    """Columns where ``player`` would win at once, by the array engine."""
    columns = []
    for col in game.get_valid_moves():
        trial = Connect4([row[:] for row in game.board], player)
        trial.make_move(col)
        if trial.check_winner() == player:
            columns.append(col)
    return columns


def capture_root(agent, monkeypatch):
    roots = []
    backpropagate = agent._backpropagate

    def record(node, winner):
        backpropagate(node, winner)
        while node.parent is not None:
            node = node.parent
        roots[:] = [node]

    monkeypatch.setattr(agent, '_backpropagate', record)
    return roots


def fingerprint(node):
    return (node.move, node.player_just_moved, node.visits, node.wins,
            tuple(node.untried_moves), node.game_state.pieces,
            tuple(fingerprint(child) for child in node.children.values()))


@lru_cache(maxsize=None)
def exact_value(own, opp, occupied):
    """Game value for the player to move: 1 win, 0.5 draw, 0 loss."""
    if occupied == BOARD_MASK:
        return 0.5
    best = 0.0
    for col in range(7):
        move = (occupied + (1 << (7 * col))) & (63 << (7 * col))
        if not move:
            continue
        if has_four(own | move):
            return 1.0
        best = max(best, 1.0 - exact_value(opp, own | move, occupied | move))
        if best == 1.0:
            break
    return best


def state_value(state):
    mover = state.current_player
    return exact_value(state.pieces[mover], state.pieces[1 - mover], state.occupied)


def test_research_agent_is_not_publicly_registered():
    # The factory imports PyTorch, so inspect sources instead of importing it.
    sources = [Path('games/connect4/agents/agent_factory.py'), *Path('api').rglob('*.py')]
    assert len(sources) > 1
    for source in sources:
        assert 'mcts_research' not in source.read_text()
    assert not issubclass(MCTSAgent, ResearchMCTSAgent)


@pytest.mark.parametrize('kwargs', [dict(rollout='minimax'), dict(expansion='best'),
                                    dict(exploration=-1), dict(exploration=float('nan')),
                                    dict(exploration=True), dict(solver=1)])
def test_invalid_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ResearchConfig(**kwargs)
    with pytest.raises(TypeError):
        ResearchMCTSAgent(10, config=dict(kwargs))


@pytest.mark.parametrize('seed', range(6))
def test_winning_cells_match_engine_four_detection_on_every_empty_cell(seed):
    rng = Random(4100 + seed)
    for plies in (4, 9, 15, 22, 30, 36):
        game, _ = random_position(rng, plies)
        state = BitboardState.from_game(game)
        for bits in state.pieces:
            wins = winning_cells(bits) & BOARD_MASK & ~state.occupied
            for cell in range(49):
                if cell % 7 == 6 or state.occupied >> cell & 1:
                    continue
                assert bool(wins >> cell & 1) == has_four(bits | 1 << cell)


@pytest.mark.parametrize('name', sorted(POSITIONS))
@pytest.mark.parametrize('budget', [1, 7, 60, 400])
def test_default_configuration_reproduces_production_tree_and_rng(name, budget, monkeypatch):
    game = position(POSITIONS[name])
    for seed in (3, 812):
        baseline = MCTSAgent(budget, rng=Random(seed))
        research = ResearchMCTSAgent(budget, rng=Random(seed))
        expected, actual = capture_root(baseline, monkeypatch), capture_root(research, monkeypatch)
        assert research.choose_move(game) == baseline.choose_move(game)
        assert research.rng.getstate() == baseline.rng.getstate()
        assert fingerprint(actual[0]) == fingerprint(expected[0])
        assert research.last_stats == dict(simulations=budget, root_proven=None)


@pytest.mark.parametrize('label', sorted(CONFIGS))
def test_legal_moves_caller_isolation_and_seeded_reproducibility(label):
    rng = Random(77)
    histories = [POSITIONS['midgame-wide'], POSITIONS['late'], DRAW[:38], DRAW[:41], [3] * 6]
    histories += [random_position(rng, plies)[1] for plies in (1, 6, 13, 19, 27, 34)]
    for history in histories:
        game = position(history)
        before = deepcopy(game.__dict__)
        moves = []
        for _ in range(2):
            agent = ResearchMCTSAgent(120, rng=Random(5), config=CONFIGS[label])
            moves.append((agent.choose_move(game), agent.last_stats, agent.rng.getstate()))
            assert game.__dict__ == before
        assert moves[0] == moves[1]
        assert type(moves[0][0]) is int and moves[0][0] in game.get_valid_moves()
        assert 0 <= moves[0][1]['simulations'] <= 120


@pytest.mark.parametrize('label', sorted(CONFIGS))
def test_agent_keeps_no_tree_between_decisions(label):
    agent = ResearchMCTSAgent(80, rng=Random(1), config=CONFIGS[label])
    agent.choose_move(position([3, 2]))
    assert not any(isinstance(value, ResearchNode) for value in vars(agent).values())
    fresh = ResearchMCTSAgent(80, rng=Random(9), config=CONFIGS[label])
    agent.rng = Random(9)
    game = position(POSITIONS['midgame'])
    assert agent.choose_move(game) == fresh.choose_move(game)
    assert agent.rng.getstate() == fresh.rng.getstate()


@pytest.mark.parametrize('label', sorted(CONFIGS))
def test_root_guards_win_block_and_avoid_floating_threat(label):
    config = CONFIGS[label]
    for history, column in WINS + BLOCKS:
        for budget in (1, 30):
            agent = ResearchMCTSAgent(budget, rng=Random(2), config=config)
            assert agent.choose_move(position(history)) == column
    agent = ResearchMCTSAgent(1, rng=Random(0), config=config)
    assert agent.choose_move(position(FLOATING)) == 6
    two_threats = position([3, 3, 2, 3, 4])
    assert ResearchMCTSAgent(8, rng=Random(0), config=config).choose_move(
        two_threats) in two_threats.get_valid_moves()


@pytest.mark.parametrize('label', sorted(CONFIGS))
@pytest.mark.parametrize('history,winner', [([0, 1, 0, 1, 0, 1, 0], 0), (DRAW, -1)])
def test_terminal_positions_are_rejected(label, history, winner):
    game = position(history[:-1])
    game.make_move(history[-1])
    assert game.check_winner() == winner
    with pytest.raises(ValueError, match='terminal position'):
        ResearchMCTSAgent(5, rng=Random(0), config=CONFIGS[label]).choose_move(game)


def reference_rollout(game, rng, avoid_gifts):
    """Array-engine statement of the declared tactical rollout policy."""
    game = deepcopy(game)
    while not game.is_game_over():
        mover = game.current_player
        if winning_columns(game, mover):
            return mover
        threats = winning_columns(game, 1 - mover)
        if len(threats) > 1:
            return 1 - mover
        if threats:
            move = threats[0]
        else:
            legal = game.get_valid_moves()
            safe = legal
            if avoid_gifts:
                safe = []
                for col in legal:
                    after = deepcopy(game)
                    after.make_move(col)
                    if after.is_valid_move(col) and col in winning_columns(after, 1 - mover):
                        continue
                    safe.append(col)
            move = rng.choice(safe or legal)
        game.make_move(move)
    return game.check_winner()


@pytest.mark.parametrize('avoid_gifts', [False, True])
@pytest.mark.parametrize('seed', range(12))
def test_tactical_rollouts_match_engine_policy_outcome_and_rng(avoid_gifts, seed):
    rng = Random(900 + seed)
    rollout = safe_rollout if avoid_gifts else tactical_rollout
    for plies in (0, 3, 8, 14, 21, 29, 37):
        game, _ = random_position(rng, plies)
        before = deepcopy(game.__dict__)
        state = BitboardState.from_game(game)
        for sample in range(6):
            engine_rng, bits_rng = Random(seed * 100 + sample), Random(seed * 100 + sample)
            assert rollout(state, bits_rng) == reference_rollout(game, engine_rng, avoid_gifts)
            assert bits_rng.getstate() == engine_rng.getstate()
        assert game.__dict__ == before


@pytest.mark.parametrize('rollout', [tactical_rollout, safe_rollout])
def test_tactical_rollout_terminals_wins_blocks_and_double_threats(rollout):
    class NoRandom:
        def choice(self, values):
            pytest.fail('Forced tactical outcomes must not consume randomness')

    for history, winner in [([0, 1, 0, 1, 0, 1, 0], 0), (DRAW, -1)]:
        game = position(history[:-1])
        game.make_move(history[-1])
        assert rollout(BitboardState.from_game(game), NoRandom()) == winner
    # Mover to play has an immediate win.
    assert rollout(BitboardState.from_game(position([0, 1, 0, 1, 0, 2])), NoRandom()) == 0
    # Open three on the bottom row: the mover cannot block both ends.
    assert rollout(BitboardState.from_game(position([2, 2, 3, 3, 4])), NoRandom()) == 0
    # Two quiet cells left in one column: singleton choices, then a draw.
    assert rollout(BitboardState.from_game(position(DRAW[:40])), Random(0)) == -1


def test_tactical_node_resolves_one_move_tactics():
    def node(history):
        return ResearchNode(BitboardState.from_game(position(history)), tactics=True)

    assert node([0, 1, 0, 1, 0, 2]).proven == 0.0          # mover wins at once
    assert node([0, 1, 0, 1, 0, 2]).untried_moves == []
    assert node([2, 2, 3, 3, 4]).proven == 1.0             # two threats against mover
    block = node([0, 1, 0, 1, 0])
    assert block.proven is None and block.untried_moves == [0]
    quiet = node([3, 2])
    assert quiet.proven is None and quiet.untried_moves == position([3, 2]).get_valid_moves()
    won = BitboardState.from_game(position([0, 1, 0, 1, 0, 1])).drop(0)
    assert ResearchNode(won, tactics=True).proven == 1.0
    drawn = BitboardState.from_game(position(DRAW[:41])).drop(6)
    assert ResearchNode(drawn, tactics=True).proven == 0.5
    plain = ResearchNode(BitboardState.from_game(position([0, 1, 0, 1, 0])))
    assert plain.proven is None and len(plain.untried_moves) == 7


@pytest.mark.parametrize('rollout', ['uniform', 'safe'])
@pytest.mark.parametrize('seed', range(10))
def test_solver_proofs_agree_with_exhaustive_search(rollout, seed, monkeypatch):
    rng = Random(6000 + seed)
    config = ResearchConfig(solver=True, rollout=rollout)
    proven_roots = searched = 0
    while searched < 3:
        game, _ = random_position(rng, rng.choice((28, 30, 32, 35)))
        agent = ResearchMCTSAgent(3000, rng=Random(seed), config=config)
        roots = capture_root(agent, monkeypatch)
        move = agent.choose_move(game)
        assert move in game.get_valid_moves()
        if not roots:                       # immediate root win, no search
            continue
        searched += 1
        root, pending, total = roots[0], [roots[0]], 0
        assert root.visits == agent.last_stats['simulations'] <= 3000
        while pending:
            node = pending.pop()
            total += 1
            assert node.visits > 0 and 0 <= node.wins <= node.visits
            assert node.player_just_moved == 1 - node.game_state.current_player
            if node.proven is not None:
                # Proven for the previous mover; exact value is for the mover.
                expected = (float(node.game_state.winner == node.player_just_moved)
                            if node.game_state.winner != -1 else
                            0.5 if node.game_state.occupied == BOARD_MASK else
                            1.0 - state_value(node.game_state))
                assert node.proven == expected
            pending.extend(node.children.values())
        assert total <= agent.last_stats['simulations'] + 1
        assert agent.last_stats['root_proven'] == root.proven
        if root.proven is not None:
            proven_roots += 1
            # A proven root means the chosen move achieves the exact value.
            after = BitboardState.from_game(game).drop(move)
            achieved = (1.0 if after.winner != -1 else
                        0.5 if after.occupied == BOARD_MASK else 1.0 - state_value(after))
            assert achieved == state_value(root.game_state) == 1.0 - root.proven
        lost = [m for m, child in root.children.items() if child.proven == 0.0]
        if len(lost) < len(root.children):
            assert move not in lost
    assert proven_roots                     # late positions are routinely solved


def test_solver_stops_early_and_plays_a_proven_win(monkeypatch):
    # X to move wins by force with 3 (open three on the bottom row next turn).
    game = position([3, 3, 2, 2])
    agent = ResearchMCTSAgent(5000, rng=Random(4), config=ResearchConfig(solver=True))
    roots = capture_root(agent, monkeypatch)
    move = agent.choose_move(game)
    assert move in (1, 4)
    assert roots[0].proven == 0.0 and roots[0].children[move].proven == 1.0
    assert agent.last_stats['simulations'] < 5000
    assert agent.last_stats['root_proven'] == 0.0


def test_solver_backs_up_exact_rewards_from_proven_leaves(monkeypatch):
    agent = ResearchMCTSAgent(40, rng=Random(0), config=ResearchConfig(solver=True))
    leaves = []
    backpropagate = agent._backpropagate

    def record(node, winner):
        leaves.append((node, winner))
        backpropagate(node, winner)

    monkeypatch.setattr(agent, '_backpropagate', record)
    assert agent.choose_move(position(DRAW[:41])) == 6
    node, winner = leaves[0]
    assert winner == -1 and node.proven == 0.5
    assert len(leaves) == 1 and agent.last_stats == dict(simulations=1, root_proven=0.5)
    assert (node.visits, node.wins, node.parent.wins) == (1, 0.5, 0.5)


def test_center_expansion_follows_center_order_without_expansion_randomness(monkeypatch):
    agent = ResearchMCTSAgent(7, rng=Random(0), config=ResearchConfig(expansion='center'))
    roots = capture_root(agent, monkeypatch)
    agent.choose_move(position([]))
    assert list(roots[0].children) == [3, 2, 4, 1, 5, 0, 6]


def test_exploration_constant_changes_selection_as_declared():
    root = ResearchNode(BitboardState.from_game(Connect4()))
    for col, (visits, wins) in zip(list(root.untried_moves), [(90, 54.0), (10, 5.0)]):
        child = ResearchNode(root.game_state.drop(col), root, col)
        child.visits, child.wins = visits, wins
        root.children[col] = child
    root.untried_moves, root.visits = [], 100
    greedy = ResearchMCTSAgent(1, rng=Random(0), config=ResearchConfig(exploration=0.0))
    wide = ResearchMCTSAgent(1, rng=Random(0), config=ResearchConfig(exploration=1.41))
    assert greedy._select_child(root).move == 3
    assert wide._select_child(root).move == 2
