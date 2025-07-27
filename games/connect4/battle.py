from .connect4 import Connect4
from .agents.human import Human
from .agents.random_agent import RandomAgent
from .agents.negamax_agent import NegamaxAgent
from .agents.mcts_agent import MCTSAgent
from .agents.mcts_nn_agent import load_pretrained_mcts_nn_agent
from .agents.victor_agent import VictorAgent

# Initialize agents
human = Human()
random_agent = RandomAgent()
negamax_agent = NegamaxAgent(9)
mcts_agent = MCTSAgent(10000)
mcts_nn_agent = load_pretrained_mcts_nn_agent() # Load trained MCTS-NN model
victor_agent = VictorAgent()

# Start game
connect4 = Connect4()
connect4.battle(human, mcts_nn_agent)  # Can switch between different agents