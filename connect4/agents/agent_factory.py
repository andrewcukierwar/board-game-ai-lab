# agents/agent_factory.py
from agents.human import Human
from agents.negamax_agent import NegamaxAgent
from agents.random_agent import RandomAgent
from agents.mcts_agent import MCTSAgent
from agents.victor_agent import VictorAgent

AGENT_TYPES = {
    'human': Human,
    'negamax': NegamaxAgent,
    'random': RandomAgent,
    'mcts': MCTSAgent,
    'victor': VictorAgent,
    # 'mcts_nn': MCTSNNAgent,  # Uncomment if you want to use MCTSNNAgent
    # Add new agents here
}

def create_agent(agent_config):
    agent_type = agent_config.get('type', 'human').lower()
    AgentClass = AGENT_TYPES.get(agent_type)
    if not AgentClass:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    if agent_type == 'negamax':
        depth = agent_config.get('depth', 3)
        if not isinstance(depth, int) or depth < 1 or depth > 10:
            raise ValueError('Invalid depth for NegamaxAgent. Must be between 1 and 10.')
        return AgentClass(depth=depth)
    else:
        return AgentClass()
