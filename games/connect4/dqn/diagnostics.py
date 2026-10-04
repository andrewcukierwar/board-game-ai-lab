"""Fixed, legally replayed probes and bounded greedy-vs-uniform-Random games."""

import random
import time

import torch

from ..connect4 import Connect4
from ..agents.dqn_agent import DQNAgent
from .encoding import encode_game, legal_mask
from .network import checked_q

# All columns are zero-based. These fixtures are fixed before the experiment.
POSITIONS = (
    ('opening_x', (), None),
    ('opening_o', (3,), None),
    ('middle_x', (3, 2, 4, 3, 2, 4, 1, 5), None),
    ('middle_o', (3, 2, 4, 3, 2, 4, 1, 5, 2), None),
    ('win_x', (0, 6, 1, 6, 2, 5), 3),
    ('win_o', (6, 0, 6, 1, 5, 2, 5), 3),
    ('block_x', (6, 0, 6, 1, 5, 2), 3),
    ('block_o', (0, 6, 1, 6, 2), 3),
    ('full_column_x', (0, 0, 0, 0, 0, 0), None),
    ('full_column_o', (0, 0, 0, 0, 0, 0, 3), None),
)


def position(moves):
    game = Connect4()
    for move in moves:
        if game.is_game_over() or not game.is_valid_move(move):
            raise ValueError('Illegal diagnostic sequence')
        game.make_move(move)
    if game.is_game_over():
        raise ValueError('Diagnostic position must be nonterminal')
    return game


def predictions(network):
    network.eval()
    rows = []
    with torch.inference_mode():
        for name, moves, expected in POSITIONS:
            game = position(moves)
            state = encode_game(game)
            mask = legal_mask(state)
            q = checked_q(network, torch.tensor([state], dtype=torch.float32))[0]
            action = int(q.masked_fill(~torch.tensor(mask), -torch.inf).argmax().item())
            rows.append(dict(name=name, moves=list(moves), player=game.current_player,
                             legal=list(mask), q=q.tolist(), action=action,
                             expected_action=expected,
                             tactical_correct=None if expected is None else action == expected))
    return rows


def evaluate_random(network, *, games, seed, seconds, clock=time.monotonic,
                    should_stop=lambda: False):
    """No replay/learning. Alternate DQN sides, restart the same seeded suite per model.

    Random uses the existing RandomAgent's uniform legal policy with a private
    RNG, avoiding any mutation of training/global RNG state. Partial games are
    explicitly excluded from wins/losses/draws and side counts.
    """
    if type(games) is not int or games < 0 or games > 12 or games % 2:
        raise ValueError('Each evaluation must request an even count in 0..12')
    if not 0 < seconds <= 60:
        raise ValueError('Each evaluation is limited to 60 seconds')
    start = clock()
    agent = DQNAgent(network)
    results = []
    partial = None
    for index in range(games):
        rng = random.Random(seed + index)
        side = index % 2
        game = Connect4()
        moves = []
        while not game.is_game_over():
            if should_stop() or clock() - start >= seconds:
                partial = dict(index=index, side=side, moves=moves)
                break
            move = (agent.choose_move(game) if game.current_player == side
                    else rng.choice(game.get_valid_moves()))
            if should_stop() or clock() - start >= seconds:
                partial = dict(index=index, side=side, moves=moves)
                break
            game.make_move(move)
            moves.append(move)
        if partial is not None:
            break
        winner = game.check_winner()
        results.append(dict(index=index, seed=seed + index, side=side, moves=moves,
                            result='draw' if winner == -1 else 'win' if winner == side else 'loss'))
    return dict(requested_games=games, completed_games=len(results), games=results,
                partial=partial, elapsed_seconds=clock() - start,
                wins=sum(r['result'] == 'win' for r in results),
                draws=sum(r['result'] == 'draw' for r in results),
                losses=sum(r['result'] == 'loss' for r in results),
                starting_sides=[sum(r['side'] == side for r in results) for side in (0, 1)])
