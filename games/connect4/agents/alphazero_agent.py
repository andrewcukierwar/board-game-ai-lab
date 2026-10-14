"""Opt-in AlphaZero v2 agent: the trained CNN with fixed research PUCT settings.

Each instance is per-request and stateless apart from ``last_decision``. The
shared ``ProductionModel`` is immutable after loading. ``tactical_guard`` only
adds the verified one-ply root safeguard after the unchanged search; it is the
same checkpoint and the same search, not a separately trained agent.
"""
from games.connect4.alphazero_v2.production import ProductionModel

LABEL = 'AlphaZero v2 CNN (Phase 4D.4H-C generation 6, seed 42; 256 PUCT simulations)'


class AlphaZeroAgent:
    label = LABEL

    def __init__(self, model, *, tactical_guard=False):
        if not isinstance(model, ProductionModel):
            raise TypeError('AlphaZeroAgent requires a loaded ProductionModel')
        if type(tactical_guard) is not bool:
            raise TypeError('tactical_guard must be a boolean')
        self.model, self.tactical_guard = model, tactical_guard
        self.last_decision = None

    def __str__(self):
        return LABEL + (' with tactical root guard' if self.tactical_guard else '')

    def choose_move(self, game):
        decision = self.model.decide(game, tactical_guard=self.tactical_guard)
        self.last_decision = decision
        return decision.final_action
