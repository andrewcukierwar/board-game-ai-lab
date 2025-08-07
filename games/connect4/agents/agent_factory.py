# agents/agent_factory.py
from .human import Human
from .negamax_agent import NegamaxAgent
from .random_agent import RandomAgent
from .mcts_agent import MCTSAgent
from .mcts_nn_agent import MCTSNNAgent, load_pretrained_mcts_nn_agent
from .victor_agent import VictorAgent

AGENT_TYPES = {
    'human': Human,
    'negamax': NegamaxAgent,
    'random': RandomAgent,
    'mcts': MCTSAgent,
    'mcts_nn': MCTSNNAgent,
    'victor': VictorAgent,
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
    elif agent_type == 'mcts_nn':
        # Load trained MCTS-NN model
        # simulation_limit = agent_config.get('simulation_limit', 1000)
        mcts_nn_agent = load_pretrained_mcts_nn_agent()
        # temperature = 1.0 # agent_config.get('temperature', 0.1)
        # if not isinstance(simulation_limit, int) or simulation_limit < 1:
        #     raise ValueError('Invalid simulation limit for MCTSNNAgent. Must be a positive integer.')
        # if not (0 <= temperature <= 1):
        #     raise ValueError('Invalid temperature for MCTSNNAgent. Must be between 0 and 1.')
        # return AgentClass(simulation_limit=simulation_limit) #, temperature=temperature)
        return mcts_nn_agent
    else:
        return AgentClass()
