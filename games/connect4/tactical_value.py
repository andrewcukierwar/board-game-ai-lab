"""Conservative engine-only exact tactical proofs shared by audit and training.

No neural inference, action selection, RNG or deeper solving.
"""
from copy import deepcopy
from .connect4 import Connect4


def require(condition, message):
    if not condition:
        raise ValueError(message)


def immediate_wins(game):
    require(not game.is_game_over(), "Proof requires a nonterminal position")
    actor = game.current_player
    wins = []
    for action in game.get_valid_moves():
        after = deepcopy(game)
        require(after.make_move(action), "Engine rejected legal successor")
        if after.check_winner() == actor:
            wins.append(action)
    return sorted(wins)


def tactical_proof(game):
    """Return +1, conservative -1, or unknown, with successor witnesses.

    A draw successor prevents a loss proof. A loss needs an opponent winning
    reply for EVERY legal move and no immediate win for the current actor.
    A uniquely safe ordinary block establishes no exact outcome here.
    """
    wins = immediate_wins(game)
    if wins:
        return {"value": 1, "winning_actions": wins, "reply_witnesses": {}}
    witnesses = {}
    for action in game.get_valid_moves():
        after = deepcopy(game)
        require(after.make_move(action), "Engine rejected legal successor")
        if after.is_game_over():
            return {"value": None, "winning_actions": [], "reply_witnesses": {}}
        replies = immediate_wins(after)
        if not replies:
            return {"value": None, "winning_actions": [], "reply_witnesses": {}}
        witnesses[str(action)] = replies
    require(bool(witnesses), "No legal actions for nonterminal position")
    return {"value": -1, "winning_actions": [], "reply_witnesses": witnesses}



def reconstruct_canonical(observation, acting_player):
    """Invert actor-relative encoding and validate a nonterminal pre-move board.

    Completed replay supplies legal-history provenance; this boundary additionally
    checks shape, cells, gravity, physical piece counts and turn consistency.
    """
    require(type(acting_player) is int and acting_player in (0, 1), "Invalid acting player")
    require(len(observation) == 6 and all(len(row) == 7 for row in observation), "Invalid board shape")
    require(all(type(c) is int and c in (-1, 0, 1) for row in observation for c in row),
            "Invalid canonical cells")
    own, other = ("X", "O") if acting_player == 0 else ("O", "X")
    board = [[" " if c == 0 else own if c == 1 else other for c in row] for row in observation]
    for col in range(7):
        occupied = False
        for row in range(6):
            require(not occupied or board[row][col] != " ", "Floating piece")
            occupied = occupied or board[row][col] != " "
    x = sum(c == "X" for row in board for c in row)
    o = sum(c == "O" for row in board for c in row)
    require(x == o + acting_player, "Piece counts disagree with acting player")
    game = Connect4(board=board, current_player=acting_player)
    require(not game.is_game_over() and bool(game.get_valid_moves()), "Expected nonterminal pre-move state")
    return game
