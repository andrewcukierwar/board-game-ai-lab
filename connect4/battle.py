from connect4 import Connect4
from agents.human import Human
from agents.random_agent import RandomAgent
from agents.negamax_agent import NegamaxAgent

human = Human()
random_agent = RandomAgent()
negamax_agent = NegamaxAgent(8)

connect4 = Connect4()
connect4.battle(human, negamax_agent)