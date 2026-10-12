"""Exact differential validation shared by the research runner and regressions."""
from dataclasses import fields
import torch
from games.connect4.alphazero_v2.bitboard import BOARD_MASK, BitboardState
from games.connect4.alphazero_v2.network import encode_bitboard_current_player
from games.connect4.neural_mcts import encode_current_player

def assert_state(state, game, *, encode=True):
    assert state.board == tuple(tuple(row) for row in game.board)
    assert state.current_player == game.current_player
    assert state.piece == game.piece
    assert state.get_valid_moves() == game.get_valid_moves()
    assert state.check_winner() == game.check_winner()
    assert state.is_game_over() == game.is_game_over()
    assert state.occupied == state.pieces[0] | state.pieces[1]
    assert not state.pieces[0] & state.pieces[1]
    assert not state.occupied & ~BOARD_MASK
    for col in range(7):
        assert state.is_valid_move(col) == game.is_valid_move(col)
    if encode:
        bits, old = encode_bitboard_current_player(state), encode_current_player(game)
        assert bits.shape == (1, 1, 6, 7) and bits.dtype == torch.float32 and bits.device.type == 'cpu'
        assert torch.equal(bits, old)
        assert bits.numpy().tobytes() == old.numpy().tobytes()  # includes signed zeros


def assert_tree(a, b):
    """Every allocated node, including unvisited children; no float tolerances."""
    pending = [(a, b)]
    nodes = 0
    while pending:
        old, new = pending.pop()
        assert_state(BitboardState.from_game(new.game_state), old.game_state, encode=False)
        assert old.move == new.move
        assert old.visits == new.visits
        for name in ('value_sum', 'prior_p', 'q_value'):
            assert float(getattr(old, name)).hex() == float(getattr(new, name)).hex()
        if old.parent:
            assert new.parent is not None
            assert old.puct(1.41).hex() == new.puct(1.41).hex()
        else:
            assert new.parent is None
        assert tuple(old.children) == tuple(new.children)
        for move in old.children:
            assert old.children[move].parent is old
            assert new.children[move].parent is new
            pending.append((old.children[move], new.children[move]))
        nodes += 1
    return nodes


def assert_result(a, b):
    for field in fields(a):
        if field.name != 'root':
            assert getattr(a, field.name) == getattr(b, field.name), field.name
    assert a.visit_target == b.visit_target
    return assert_tree(a.root, b.root)
