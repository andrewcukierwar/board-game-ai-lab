"""Opt-in public adapter for the functional Victor research solver.

Experimental research agent, NOT perfect play and NOT the legacy ``VictorAgent``.
Each move is a fresh, stateless analysis under a fixed resource profile with a
whole-move deadline; a legal move is always returned. Research labels (exact,
certified, exploratory, heuristic) stay on ``last_decision`` for logs and tests
and are never part of public game outcomes.
"""
from games.connect4.victor import SearchBudget, SolverBudget, analyze_position

from .negamax_agent import NegamaxAgent

LABEL = 'Victor research solver (experimental; not perfect play)'

# Measured locally (docs/victor-performance-and-integration.md): the cooperative
# deadline is checked between bounded steps, so overshoot is one cover search.
PUBLIC_BUDGET = SolverBudget(
    exact=SearchBudget(nodes=150_000, seconds=0.4, max_remaining=24, table_entries=150_000),
    cover_nodes=10_000,
    white_contexts=4,
    strategic_children=7,
    fallback_depth=4,
    policy_audit=SearchBudget(nodes=10_000, seconds=0.1, max_remaining=10, table_entries=10_000),
    deadline=1.0,
)


class VictorResearchAgent:
    label = LABEL

    def __init__(self, budget: SolverBudget = PUBLIC_BUDGET):
        if type(budget) is not SolverBudget or budget.deadline is None:
            raise ValueError('the research agent requires a SolverBudget with a deadline')
        self.budget = budget
        self.last_decision = None

    def __str__(self):
        return LABEL

    __repr__ = __str__

    def choose_move(self, game):
        legal = game.get_valid_moves()
        if not legal or game.is_game_over():
            raise ValueError('Cannot choose a move from a terminal position')
        try:
            result = analyze_position(game.board, game.current_player, self.budget)
            if result.move in legal:
                self.last_decision = dict(move=result.move, kind=result.move_kind,
                                          exact_value=result.exact_value, bound=result.bound,
                                          deadline_reached=result.deadline_reached)
                return result.move
            failure = 'solver returned no legal move'
        except Exception as exc:  # Any solver defect degrades to bounded heuristic play.
            failure = f'solver error: {type(exc).__name__}'
        try:
            move = NegamaxAgent(self.budget.fallback_depth).choose_move(game)
        except Exception:
            move = None
        if move not in legal:
            move = legal[0]
        self.last_decision = dict(move=move, kind='fallback', exact_value=None, bound=None,
                                  deadline_reached=False, failure=failure)
        return move
