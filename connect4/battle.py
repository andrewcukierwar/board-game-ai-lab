from connect4 import Connect4
from agents.human import Human
from agents.random_agent import RandomAgent
from agents.negamax_agent import NegamaxAgent
from agents.mcts_agent import MCTSAgent

human = Human()
random_agent = RandomAgent()
negamax_agent = NegamaxAgent(9)
mcts_agent = MCTSAgent(10000)

connect4 = Connect4()
connect4.battle(human, mcts_agent) # mcts_agent # negamax_agent