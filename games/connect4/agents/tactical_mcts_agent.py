"""Production Tactical MCTS (F1): research-validated solver-enhanced UCT.

The immutable combination was established in MCTS v3. Kept distinct from Classic
MCTS so historical agent identities and experiments remain meaningful.
"""
from .mcts_research_agent import ResearchConfig, ResearchMCTSAgent

F1_CONFIGURATION = ResearchConfig(rollout='safe', solver=True, exploration=0.5)


class TacticalMCTSAgent(ResearchMCTSAgent):
    def __init__(self, simulation_limit=1000, *, rng=None):
        super().__init__(simulation_limit=simulation_limit, rng=rng, config=F1_CONFIGURATION)

    def __str__(self):
        return f'Tactical MCTS Agent ({self.simulation_limit} max sims)'

    __repr__ = __str__
