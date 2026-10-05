"""Complete self-play games from empty boards with one frozen v2 inference snapshot."""
import time

from ..connect4 import Connect4
from .data import capture_example, finalize_game


def play_game(player, generation, index, *, check=None, observer=None):
    """Play one complete game with ``player`` (a SelfPlayer) moving for both sides.

    ``check()`` runs before every search and may raise to abandon the game
    (cooperative budget/signal stop); an abandoned game yields nothing.
    ``observer(index, decision, seconds)`` sees each decision; it must not
    consume any RNG stream owned by the player.
    """
    game = Connect4()
    moves, pending = [], []
    while not game.is_game_over():
        if check is not None:
            check()
        started = time.perf_counter()
        decision = player.decide(game)
        seconds = time.perf_counter() - started
        if decision.search.simulations != player.config.self_play_simulations:
            raise RuntimeError("Self-play search used an undeclared budget")
        example = capture_example(game, decision)  # Explicitly BEFORE applying the move.
        if not game.make_move(decision.action.move):
            raise RuntimeError("Engine rejected a searched legal move")
        if observer is not None:
            observer(index, decision, seconds)
        moves.append(decision.action.move)
        pending.append(example)
    return finalize_game(generation, index, moves, pending)


def collect_generation(player, generation, games, *, check=None, observer=None):
    """Exactly ``games`` complete games; an exception discards the whole generation."""
    return tuple(play_game(player, generation, index, check=check, observer=observer) for index in range(games))
