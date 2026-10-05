"""Complete self-play games from empty boards with one frozen v2 inference snapshot."""
from ..connect4 import Connect4
from .data import capture_example, finalize_game


def play_game(player, generation, index):
    """Play one complete game with ``player`` (a SelfPlayer) moving for both sides."""
    game = Connect4()
    moves, pending = [], []
    while not game.is_game_over():
        decision = player.decide(game)
        if decision.search.simulations != player.config.self_play_simulations:
            raise RuntimeError("Self-play search used an undeclared budget")
        example = capture_example(game, decision)  # Explicitly BEFORE applying the move.
        if not game.make_move(decision.action.move):
            raise RuntimeError("Engine rejected a searched legal move")
        moves.append(decision.action.move)
        pending.append(example)
    return finalize_game(generation, index, moves, pending)


def collect_generation(player, generation, games):
    """Exactly ``games`` complete games; an exception discards the whole generation."""
    return tuple(play_game(player, generation, index) for index in range(games))
