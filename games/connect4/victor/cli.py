"""Research gameplay, disconnected from the public factory and web application."""
import argparse
import json
from collections import Counter
from random import Random
from time import monotonic

from games.connect4.connect4 import Connect4
from games.connect4.agents.negamax_agent import NegamaxAgent
from games.connect4.agents.mcts_agent import MCTSAgent
from .exact import SearchBudget
from .solver import SolverBudget, VictorSolver


class RandomPlayer:
    def __init__(self, seed):
        self.rng = Random(seed)

    def choose_move(self, game):
        return self.rng.choice(game.get_valid_moves())


class HumanPlayer:
    def choose_move(self, game):
        print(game.board)
        while True:
            try:
                column = int(input('Column 1–7: ')) - 1
                if column in game.get_valid_moves():
                    return column
            except ValueError:
                pass
            print('Enter a legal column from 1 to 7.')


def player(name, seed, budget):
    if name == 'victor':
        return VictorSolver(budget)
    if name == 'random':
        return RandomPlayer(seed)
    if name == 'human':
        return HumanPlayer()
    if name.startswith('negamax:'):
        return NegamaxAgent(int(name.split(':')[1]))
    if name.startswith('mcts:'):
        return MCTSAgent(int(name.split(':')[1]), rng=Random(seed))
    raise ValueError('player: victor, random, human, negamax:DEPTH, mcts:SIMULATIONS')


def play_game(white, black, *, moves=(), budget=SolverBudget(), seed=0):
    """Complete legal replay and per-Victor decision labels; stop at first terminal."""
    game, history = Connect4(), list(moves)
    for c in history:
        if not game.make_move(c):
            raise ValueError('invalid opening replay')
    agents = (player(white, seed, budget), player(black, seed + 1, budget))
    decisions, start = [], monotonic()
    while not game.is_game_over():
        mover = game.current_player
        column = agents[mover].choose_move(game)
        if column not in game.get_valid_moves():
            raise RuntimeError('agent returned illegal move')
        if isinstance(agents[mover], VictorSolver):
            r = agents[mover].last_result
            decisions.append(dict(ply=len(history), player=mover, move=column,
                                  kind=r.move_kind, exact_value=r.exact_value,
                                  bound=r.bound, reason=r.reason,
                                  exact_nodes=r.exact.nodes, exact_seconds=r.exact.elapsed))
        assert game.make_move(column)
        history.append(column)
    return dict(white=white, black=black, seed=seed, moves=history,
                winner=game.check_winner(), plies=len(history),
                seconds=monotonic() - start, decisions=decisions,
                decision_counts=dict(Counter(d['kind'] for d in decisions)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--white', default='victor')
    parser.add_argument('--black', default='random')
    parser.add_argument('--seed', type=int, default=20261008)
    parser.add_argument('--moves', default='', help='comma separated ZERO-based opening columns')
    parser.add_argument('--nodes', type=int, default=200_000)
    parser.add_argument('--seconds', type=float, default=1.0)
    parser.add_argument('--remaining', type=int, default=24)
    parser.add_argument('--cover-nodes', type=int, default=10_000)
    parser.add_argument('--fallback-depth', type=int, default=4)
    args = parser.parse_args()
    budget = SolverBudget(exact=SearchBudget(nodes=args.nodes, seconds=args.seconds,
                                           max_remaining=args.remaining),
                          cover_nodes=args.cover_nodes, fallback_depth=args.fallback_depth)
    moves = tuple(int(c) for c in args.moves.split(',')) if args.moves else ()
    print(json.dumps(play_game(args.white, args.black, moves=moves, budget=budget,
                               seed=args.seed), indent=2))


if __name__ == '__main__':
    main()
