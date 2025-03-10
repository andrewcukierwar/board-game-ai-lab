from connect4 import Connect4
from agents.human import Human
from agents.random_agent import RandomAgent
from agents.negamax_agent import NegamaxAgent
from agents.mcts_agent import MCTSAgent
from agents.mcts_nn_agent import MCTSNNAgent, Connect4Net
from agents.victor_agent import VictorAgent
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Initialize agents
human = Human()
random_agent = RandomAgent()
negamax_agent = NegamaxAgent(9)
mcts_agent = MCTSAgent(10000)
victor_agent = VictorAgent()

# Load trained MCTS-NN model
model = Connect4Net()
checkpoint = torch.load('connect4_model_iter_100.pt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
mcts_nn_agent = MCTSNNAgent(model, simulation_limit=1000, temperature=0.1)

# Start game
connect4 = Connect4()
connect4.battle(human, victor_agent)  # Can switch between different agents