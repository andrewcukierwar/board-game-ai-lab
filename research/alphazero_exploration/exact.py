"""POST HOC exact-value context for the one-reply tactical events.

Added after the independent review; NOT a predeclared endpoint and never an
input to the decision rule. The one-reply scans count an "unsafe" move even
when the mover was already lost under perfect play, and a "missed win" even
when the executed move still wins. The project's exact solver separates those
value-neutral events from moves that change the game-theoretic result.
Neutral under perfect play is not neutral in practice: a 256-simulation
opponent may not convert a deep theoretical win.
"""
from collections import Counter

from games.connect4.alphazero_v2.oracle import BitboardSolver, SolverBudgetExceeded
from . import analysis as a


class ExactValues:
    """Cached W/D/L for the player to move; None when the node budget is exceeded."""

    def __init__(self, max_nodes=6_000_000):
        self.solver, self.cache, self.unsolved = BitboardSolver(max_nodes), {}, 0

    def __call__(self, moves):
        key = tuple(moves)
        if key not in self.cache:
            self.solver.nodes = 0
            try:
                self.cache[key] = self.solver.value(list(key))
            except SolverBudgetExceeded:
                self.cache[key] = None
                self.unsolved += 1
        return self.cache[key]


def game_exact(game, value):
    """Counts for one game. An unsolved position is counted as value-changing."""
    c, moves, last = Counter(games=1), game['moves'], len(game['plies']) - 1
    for p in game['plies']:
        ply, action, wins, safe, legal = p['ply'], p['action'], p['wins'], p['safe'], p['legal']
        stage = a.stage_of(ply)
        if wins and action not in wins:
            after = 0 if ply == 41 else value(moves[:ply + 1])   # for the opponent, after the executed move
            c['unsolved'] += after is None
            still = after == -1
            c['missed_wins'] += 1
            c['missed_wins:still_winning'] += still
            c['missed_wins:value_changing'] += not still
            c['value_changing_errors'] += not still
        elif safe and len(safe) < len(legal) and action not in safe:
            before = value(moves[:ply])
            c['unsolved'] += before is None
            kind = 'already_lost' if before == -1 else 'value_changing'
            c['unsafe'] += 1
            c[f'unsafe:{kind}'] += 1
            c[f'unsafe:{kind}@{stage}'] += 1
            c[f'unsafe:{kind}:' + a._attribution(p['top'], legal - safe)] += 1
            c['value_changing_errors'] += before != -1
        if safe and action not in safe and ply == last - 1 and game['winner'] == 1 - ply % 2:
            before = value(moves[:ply])
            kind = 'already_lost' if before == -1 else 'value_changing'
            c['game_ending_errors'] += 1
            c[f'game_ending_errors:{kind}'] += 1
            c[f'game_ending_errors:{kind}:' + a._attribution(p['top'], legal - safe)] += 1
            c['value_changing_game_endings'] += before != -1
    return c


def attach(games, value):
    for game in games:
        game['exact'] = game_exact(game, value)


RATES = dict(value_changing_errors_per_game=('value_changing_errors', 'games'),
             value_changing_game_ending_share=('value_changing_game_endings', 'games'),
             already_lost_share_of_unsafe=('unsafe:already_lost', 'unsafe'),
             still_winning_share_of_missed_wins=('missed_wins:still_winning', 'missed_wins'))


def rate(games, numerator, denominator):
    counts = a.total(games, 'exact')
    return counts[numerator] / counts[denominator] if counts[denominator] else float('nan')


def profile(games):
    counts = a.total(games, 'exact')
    return dict(counts={k: counts[k] for k in sorted(counts)},
                rates={name: (counts[n] / counts[d] if counts[d] else None) for name, (n, d) in RATES.items()})
