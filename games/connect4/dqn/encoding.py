"""Versioned, shared training/inference representation; no torch dependency.

Engine player 0 is X, player 1 is O, empty is a space. Encode the player
currently to move as +1, their opponent as -1, empty as 0. Flatten top row
first, left to right, into 42 features. Action i is physical column i (0..6),
independent of the engine's center-first move enumeration.
"""

import numpy as np

ENCODING_VERSION = 'connect4-current-player-row-major-v1'
ACTION_ORDER = list(range(7))
VALUE_CONVENTION = 'actor-reward-next-player-q-negamax-one-ply-v1'


def validate_state(state):
    """Return an independent immutable canonical vector; reject coercion errors."""
    arr = np.asarray(state)
    if arr.shape != (42,) or arr.dtype.kind not in 'ifu':
        raise ValueError('State must be a numeric vector of shape (42,)')
    if not np.isin(arr, (-1, 0, 1)).all():
        raise ValueError('State values must be -1, 0 or 1')
    return tuple(float(value) for value in arr)


def encode_game(game):
    if type(game.current_player) is not int or game.current_player not in (0, 1):
        raise ValueError('current_player must be 0 (X) or 1 (O)')
    own = 'X' if game.current_player == 0 else 'O'
    if game.piece != own:
        raise ValueError('piece and current_player disagree')
    board = np.asarray(game.board)
    if board.shape != (6, 7) or not np.isin(board, ('X', 'O', ' ')).all():
        raise ValueError('Board must have shape (6, 7) and only X/O/space cells')
    return validate_state(np.where(board == own, 1, np.where(board == ' ', 0, -1)).ravel())


def legal_mask(state):
    """Column availability, not terminal status: terminal wins may have space."""
    return tuple(value == 0 for value in validate_state(state)[:7])


def playable_state(game):
    state = encode_game(game)
    if game.is_game_over():
        raise ValueError('Cannot choose a move from a terminal position')
    mask = legal_mask(state)
    if not any(mask):
        raise ValueError('Nonterminal position has no legal actions')
    return state, mask
